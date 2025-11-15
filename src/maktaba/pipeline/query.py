"""Query pipeline that ties together embedder, store, reranker, and citations."""

import asyncio
from typing import Any, Dict, List, Optional, Tuple, Union

from ..citation.formatter import format_with_citations
from ..embedding.base import BaseEmbedder
from ..keyword.base import BaseKeywordStore
from ..logging import get_logger
from ..models import SearchResult
from ..reranking.base import BaseReranker
from ..retrieval.query_condenser import AutoQueryCondenser, QueryCondenser
from ..storage.base import BaseVectorStore


class QueryPipeline:
    """
    Feature-complete query pipeline with optional parallel keyword search.

    Supports semantic vector search with optional full-text keyword search
    executed in parallel. Results are deduplicated and optionally reranked.

    Usage:
        # Basic semantic search only
        pipeline = QueryPipeline(embedder, store, reranker)
        out = await pipeline.search("What is Tawhid?", rerank=True)

        # With parallel keyword search
        from maktaba.keyword.qdrant import QdrantKeywordStore
        keyword_store = QdrantKeywordStore(client=store.client, collection_name="docs")
        pipeline = QueryPipeline(embedder, store, reranker, keyword_store=keyword_store)
        out = await pipeline.search(
            "What is Tawhid?",
            keyword_queries=["tawhid", "monotheism", "oneness"],
            rerank=True
        )
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        store: BaseVectorStore,
        reranker: Optional[BaseReranker] = None,
        keyword_store: Optional[BaseKeywordStore] = None,
        namespace: Optional[str] = None,
        default_top_k: int = 10,
    ) -> None:
        """
        Initialize query pipeline.

        Args:
            embedder: Embedding model for vector search
            store: Vector store for semantic search
            reranker: Optional reranker for result refinement
            keyword_store: Optional keyword store for full-text search
            namespace: Default namespace for searches
            default_top_k: Default number of results to return
        """
        self.embedder = embedder
        self.store = store
        self.reranker = reranker
        self.keyword_store = keyword_store
        self.namespace = namespace
        self.default_top_k = default_top_k
        self._logger = get_logger("maktaba.pipeline.query")

    async def search(
        self,
        query: str,
        rerank: bool = True,
        top_k: Optional[int] = None,
        rerank_limit: Optional[int] = None,
        min_score: Optional[float] = None,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
        includeMetadata: bool = True,
        keyword_queries: Optional[List[str]] = None,
        keyword_limit: int = 15,
    ) -> Dict[str, object]:
        """
        Perform semantic search with optional parallel keyword search.

        Args:
            query: Main semantic search query
            rerank: Whether to apply reranking
            top_k: Number of semantic search results
            rerank_limit: Number of results after reranking
            min_score: Minimum similarity score threshold
            namespace: Search namespace
            filter: Metadata filters
            includeMetadata: Include metadata in results
            keyword_queries: Optional list of keyword queries for parallel full-text search
            keyword_limit: Number of results per keyword query

        Returns:
            Dict with:
                - formatted_context: Formatted text with citations
                - citations: List of citation entries
                - results: Merged results (semantic + keyword, deduplicated)
                - semantic_results: Separate semantic search results (before merge)
                - keyword_results: Separate keyword search results (before merge)
                - keyword_result_count: Total count of keyword results
        """
        k = top_k or self.default_top_k
        ns = namespace or self.namespace

        # 1) Prepare semantic search task
        self._logger.info("query.start: text='%s' top_k=%s ns=%s keyword_queries=%d",
                         query, k, ns, len(keyword_queries) if keyword_queries else 0)

        async def semantic_search() -> List[SearchResult]:
            """Execute semantic vector search."""
            qvec = await self.embedder.embed_text(query, input_type="query")
            results = await self.store.query(
                vector=qvec,
                topK=k,
                filter=filter,
                includeMetadata=includeMetadata,
                namespace=ns,
            )
            # Apply min_score filter
            if min_score is not None:
                results = [r for r in results if r.score is not None and r.score >= min_score]
                self._logger.info("query.min_score_filter: threshold=%.3f remaining=%d", min_score, len(results))
            return results

        # 2) Prepare keyword search tasks (if provided)
        keyword_tasks: List[asyncio.Task[List[SearchResult]]] = []
        if keyword_queries and self.keyword_store is not None:
            # Capture keyword_store in local variable for type narrowing
            keyword_store = self.keyword_store

            async def create_keyword_search(kw_q: str) -> List[SearchResult]:
                """Execute keyword search for a single query."""
                try:
                    return await keyword_store.search(
                        query=kw_q,
                        limit=keyword_limit,
                        filter=filter,
                        namespace=ns,
                    )
                except Exception as e:
                    self._logger.error(f"Keyword search failed for '{kw_q[:50]}...': {e}", exc_info=True)
                    return []

            for kw_query in keyword_queries:
                keyword_tasks.append(asyncio.create_task(create_keyword_search(kw_query)))

        # 3) Execute searches in parallel
        semantic_task = asyncio.create_task(semantic_search())
        all_tasks = [semantic_task] + keyword_tasks

        results_list = await asyncio.gather(*all_tasks, return_exceptions=True)

        # 4) Process results and deduplicate
        all_chunks: Dict[str, SearchResult] = {}  # Deduplicate by chunk ID

        # Process semantic results
        semantic_results = results_list[0]
        if isinstance(semantic_results, Exception):
            self._logger.error(f"Semantic search failed: {semantic_results}", exc_info=True)
            semantic_results = []
        elif not isinstance(semantic_results, list):
            semantic_results = []

        for chunk in semantic_results:
            if chunk.id not in all_chunks:
                all_chunks[chunk.id] = chunk

        # Process keyword results
        keyword_results: List[SearchResult] = []  # Collect all keyword results separately
        keyword_result_count = 0
        for idx, kw_result in enumerate(results_list[1:], 1):
            if isinstance(kw_result, Exception):
                self._logger.error(f"Keyword search {idx} failed: {kw_result}", exc_info=True)
                continue
            if not isinstance(kw_result, list):
                continue

            keyword_result_count += len(kw_result)
            keyword_results.extend(kw_result)  # Collect keyword results separately
            for chunk in kw_result:
                # Keep first occurrence (semantic results take priority)
                if chunk.id not in all_chunks:
                    all_chunks[chunk.id] = chunk

        initial = list(all_chunks.values())
        self._logger.info(
            "query.parallel_search: semantic=%d keyword_total=%d unique=%d",
            len(semantic_results),
            keyword_result_count,
            len(initial)
        )

        # 5) Rerank combined results (optional)
        if rerank and self.reranker is not None:
            rerank_k = rerank_limit if rerank_limit is not None else k
            ranked = await self.reranker.rerank(query, initial, top_k=rerank_k)
        else:
            # Sort by score (highest first) if available
            # Keyword results (score=None) are sorted after semantic results
            # Use a small negative value to keep them in results but after scored items
            ranked = sorted(
                initial,
                key=lambda r: r.score if r.score is not None else -1.0,
                reverse=True
            )[:k]

        self._logger.info("query.retrieved: initial=%d ranked=%d", len(initial), len(ranked))

        # 6) Format citations
        formatted = format_with_citations(ranked, top_k=k)
        formatted["results"] = ranked
        formatted["semantic_results"] = semantic_results  # Separate semantic results
        formatted["keyword_results"] = keyword_results  # Separate keyword results (before merge)
        formatted["keyword_result_count"] = keyword_result_count  # Count for convenience
        citations = formatted.get("citations", [])
        self._logger.info(
            "query.done: formatted_blocks=%d citations=%d semantic=%d keyword=%d",
            len(ranked),
            len(citations) if hasattr(citations, '__len__') else 0,
            len(semantic_results),
            keyword_result_count,
        )
        return formatted

    async def search_with_history(
        self,
        messages: List[Union[Dict[str, str], Tuple[str, str]]],
        *,
        rerank: bool = True,
        top_k: Optional[int] = None,
        rerank_limit: Optional[int] = None,
        min_score: Optional[float] = None,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
        includeMetadata: bool = True,
        condenser: Optional[QueryCondenser] = None,
        max_history: int = 10,
        keyword_queries: Optional[List[str]] = None,
        keyword_limit: int = 15,
    ) -> Dict[str, object]:
        """
        Perform search with conversation history using query condensation.

        Condenses the conversation history and latest user message into a single
        search query, then performs semantic search with optional parallel keyword search.

        Args:
            messages: Conversation history as list of dicts or (role, content) tuples
            rerank: Whether to apply reranking
            top_k: Number of semantic search results
            rerank_limit: Number of results after reranking
            min_score: Minimum similarity score threshold
            namespace: Search namespace
            filter: Metadata filters
            includeMetadata: Include metadata in results
            condenser: Optional query condenser (defaults to AutoQueryCondenser)
            max_history: Maximum number of history messages to use for condensation
            keyword_queries: Optional list of keyword queries for parallel full-text search
            keyword_limit: Number of results per keyword query

        Returns:
            Dict with:
                - formatted_context: Formatted text with citations
                - citations: List of citation entries
                - results: Merged results (semantic + keyword, deduplicated)
                - semantic_results: Separate semantic search results (before merge)
                - keyword_results: Separate keyword search results (before merge)
                - keyword_result_count: Total count of keyword results
        """
        if not messages:
            raise ValueError("messages must contain at least one item")

        # Normalize to (role, content)
        norm: List[Tuple[str, str]] = []
        for m in messages:
            if isinstance(m, tuple):
                role, content = m
            else:
                role = m.get("role", "user")
                content = m.get("content", "")
            if not isinstance(content, str):
                content = str(content)
            norm.append((role, content))

        # Find latest user message
        last_role, last_content = norm[-1]
        if last_role != "user":
            for role, content in reversed(norm):
                if role == "user":
                    last_role, last_content = role, content
                    break
            else:
                raise ValueError("no user message found in messages")

        history = norm[:-1]
        if max_history > 0:
            history = history[-max_history:]

        cond = condenser or AutoQueryCondenser(max_history=max_history)
        condensed = await cond.condense(history, last_content)
        return await self.search(
            condensed,
            rerank=rerank,
            top_k=top_k,
            rerank_limit=rerank_limit,
            min_score=min_score,
            namespace=namespace,
            filter=filter,
            includeMetadata=includeMetadata,
            keyword_queries=keyword_queries,
            keyword_limit=keyword_limit,
        )
