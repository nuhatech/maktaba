"""Agentic query pipeline with iterative retrieval and LLM-based evaluation."""

import asyncio
from typing import Any, Dict, List, Optional, Tuple, Union

from ..citation.formatter import format_with_citations
from ..embedding.base import BaseEmbedder
from ..keyword.base import BaseKeywordStore
from ..llm.base import BaseLLM
from ..llm.openai import OpenAILLM
from ..logging import get_logger
from ..models import SearchResult, LLMUsage
from ..reranking.base import BaseReranker
from ..storage.base import BaseVectorStore


class AgenticQueryPipeline:
    """
    Agentic RAG pipeline with LLM-based query generation and evaluation.

    Iteratively generates queries, retrieves documents, and evaluates until
    sufficient information is found or budget exhausted.

    Usage:
        pipeline = AgenticQueryPipeline(embedder, store, reranker, llm)
        result = await pipeline.agentic_search(
            messages=[("user", "What is tawhid?")],
            max_iterations=3,
            top_k=50,
            rerank_limit=15,
        )
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        store: BaseVectorStore,
        reranker: Optional[BaseReranker] = None,
        keyword_store: Optional[BaseKeywordStore] = None,
        llm: Optional[BaseLLM] = None,
        llm_api_key: Optional[str] = None,
        llm_model: str = "gpt-4o-mini",
        namespace: Optional[str] = None,
    ) -> None:
        """
        Initialize agentic pipeline.

        Args:
            embedder: Embedding model for vector search
            store: Vector store for retrieval
            reranker: Optional reranker for result refinement
            keyword_store: Optional keyword search store for full-text search
            llm: LLM for query generation/evaluation (defaults to OpenAI)
            llm_api_key: API key for LLM (if not using default)
            llm_model: LLM model name
            namespace: Default namespace for searches
        """
        self.embedder = embedder
        self.store = store
        self.reranker = reranker
        self.keyword_store = keyword_store
        self.namespace = namespace
        self._logger = get_logger("maktaba.pipeline.agentic")

        # Initialize LLM (default to OpenAI if not provided)
        if llm is not None:
            self.llm = llm
        else:
            self.llm = OpenAILLM(api_key=llm_api_key, model=llm_model)

    async def _execute_single_query(
        self,
        query_text: str,
        query_type: str,
        top_k: int,
        rerank_limit: int,
        keyword_limit: int,
        min_score: Optional[float],
        namespace: Optional[str],
        filter: Optional[Dict[str, Any]],
        includeMetadata: bool,
    ) -> List[SearchResult]:
        """
        Execute a single query (semantic or keyword search).

        Args:
            query_text: Query string
            query_type: "semantic" or "keyword"
            top_k: Number of results for semantic search
            rerank_limit: Number of results after reranking semantic results
            keyword_limit: Number of results for keyword search
            min_score: Minimum score threshold (for semantic search)
            namespace: Search namespace
            filter: Metadata filters
            includeMetadata: Include metadata in results

        Returns:
            List of SearchResult objects.
            Handles errors gracefully by returning empty list.
        """
        try:
            # Route based on query type
            if query_type == "keyword":
                # Keyword search (full-text)
                if self.keyword_store is None:
                    self._logger.warning(
                        f"Keyword query requested but no keyword_store available: '{query_text[:50]}...'"
                    )
                    return []

                results: List[SearchResult] = await self.keyword_store.search(
                    query=query_text,
                    limit=keyword_limit,
                    filter=filter,
                    namespace=namespace,
                )

                return results

            else:
                # Semantic search (vector)
                qvec = await self.embedder.embed_text(query_text, input_type="query")

                results: List[SearchResult] = await self.store.query(
                    vector=qvec,
                    topK=top_k,
                    filter=filter,
                    includeMetadata=includeMetadata,
                    namespace=namespace,
                )

                # Apply min_score filter
                if min_score is not None:
                    results = [r for r in results if r.score is not None and r.score >= min_score]

                # Rerank if available
                if self.reranker is not None:
                    results = await self.reranker.rerank(query_text, results, top_k=rerank_limit)
                else:
                    results = results[:rerank_limit]

                return results

        except Exception as e:
            self._logger.error(f"Query execution failed for '{query_text[:50]}...': {e}", exc_info=True)
            return []

    async def agentic_search(
        self,
        messages: List[Union[Dict[str, str], Tuple[str, str]]],
        *,
        max_iterations: int = 3,
        max_queries_per_iter: int = 10,
        token_budget: int = 4096,
        top_k: int = 50,
        rerank_limit: int = 15,
        keyword_limit: int = 15,
        min_score: Optional[float] = None,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
        includeMetadata: bool = True,
        include_query_results: bool = False,
    ) -> Dict[str, Any]:
        """
        Perform agentic search with iterative query generation.

        Args:
            messages: Chat history as list of dicts or tuples
            max_iterations: Maximum number of query generation iterations
            max_queries_per_iter: Maximum queries to generate per iteration (default: 10)
            token_budget: Maximum token budget for iterations (approximate)
            top_k: Number of results to retrieve per query
            rerank_limit: Number of results to keep after reranking
            keyword_limit: Number of results for keyword queries (when implemented)
            min_score: Minimum similarity score threshold
            namespace: Namespace for vector search
            filter: Metadata filter for vector search
            includeMetadata: Include metadata in results
            include_query_results: Include query->results mapping for debugging

        Returns:
            Dict with keys:
                - formatted_context: Citation-formatted text
                - citations: List of citation dicts
                - results: List of SearchResult objects
                - queries_used: List of generated query strings
                - iterations: Number of iterations performed
                - total_chunks: Total unique chunks retrieved
                - query_to_result: (optional) Dict mapping query->results for debugging
        """
        ns = namespace or self.namespace

        # Normalize messages to (role, content) tuples
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

        if not norm:
            raise ValueError("messages must contain at least one item")

        self._logger.info(
            f"agentic_search.start: max_iter={max_iterations} top_k={top_k} rerank_limit={rerank_limit}"
        )

        # Track state across iterations
        all_chunks: Dict[str, SearchResult] = {}  # Deduplicate by chunk ID
        queries_used: List[str] = []
        query_to_result: Dict[str, List[SearchResult]] = {}  # Track query -> results mapping
        iterations_done = 0
        total_usage = LLMUsage()  # Precise token tracking

        # Add first user message as initial query
        last_user_msg = next((c for r, c in reversed(norm) if r == "user"), norm[-1][1])
        queries_used.append(last_user_msg)

        # Iterative search loop
        for iteration in range(max_iterations):
            iterations_done += 1

            # Generate new queries using LLM
            if iteration == 0:
                # First iteration: user's question + LLM-generated queries
                generated_queries = [
                    {"type": "semantic", "query": last_user_msg}
                ]
                # Also generate additional queries via LLM
                additional_queries, query_gen_usage = await self.llm.generate_queries(
                    messages=norm,
                    existing_queries=queries_used,
                    max_queries=max_queries_per_iter,
                )
                total_usage += query_gen_usage
                generated_queries.extend(additional_queries)
            else:
                # Subsequent iterations: only generate new queries
                generated_queries, query_gen_usage = await self.llm.generate_queries(
                    messages=norm,
                    existing_queries=queries_used,
                    max_queries=max_queries_per_iter,
                )
                total_usage += query_gen_usage

            if not generated_queries:
                self._logger.info(f"agentic_search.iter_{iteration}: no new queries generated, stopping")
                break

            # Filter out duplicate queries and prepare for parallel execution
            new_queries = [
                q for q in generated_queries
                if q.get("query") and q.get("query") not in queries_used
            ]

            if not new_queries:
                self._logger.info(f"agentic_search.iter_{iteration}: all queries are duplicates, stopping")
                break

            # Log queries being executed
            for q in new_queries:
                query_text = q.get("query", "")
                query_type = q.get("type", "semantic")
                queries_used.append(query_text)
                self._logger.info(
                    f"agentic_search.iter_{iteration}: queueing {query_type} query: '{query_text[:50]}...'"
                )

            # Execute all queries in parallel
            self._logger.info(f"agentic_search.iter_{iteration}: executing {len(new_queries)} queries in parallel")

            tasks = [
                self._execute_single_query(
                    query_text=q.get("query", ""),
                    query_type=q.get("type", "semantic"),
                    top_k=top_k,
                    rerank_limit=rerank_limit,
                    keyword_limit=keyword_limit,
                    min_score=min_score,
                    namespace=ns,
                    filter=filter,
                    includeMetadata=includeMetadata,
                )
                for q in new_queries
            ]

            # Gather results from parallel execution
            results_list = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results and deduplicate chunks
            for idx, result in enumerate(results_list):
                if isinstance(result, Exception):
                    self._logger.error(f"Query {idx} failed: {result}")
                    continue

                # Track query -> result mapping for debugging
                query_text = new_queries[idx].get("query", "")
                query_to_result[query_text] = result

                # Add retrieved chunks to pool (deduplicate by ID)
                for chunk in result:
                    if chunk.id not in all_chunks:
                        all_chunks[chunk.id] = chunk

            self._logger.info(
                f"agentic_search.iter_{iteration}: parallel execution complete, "
                f"retrieved {sum(len(r) if not isinstance(r, Exception) else 0 for r in results_list)} chunks, "
                f"total unique: {len(all_chunks)}"
            )

            # Evaluate if we have enough information
            if iteration < max_iterations - 1:  # Don't evaluate on last iteration
                sources_text = [chunk.text for chunk in all_chunks.values()]
                can_answer, eval_usage = await self.llm.evaluate_sources(
                    messages=norm,
                    sources=sources_text[:50],  # Limit to avoid huge context
                )
                total_usage += eval_usage

                self._logger.info(
                    f"agentic_search.iter_{iteration}: canAnswer={can_answer} "
                    f"(total tokens: {total_usage.total_tokens}/{token_budget})"
                )

                if can_answer:
                    self._logger.info("agentic_search: sufficient information found, stopping early")
                    break

            # Check token budget (now precise)
            if total_usage.total_tokens >= token_budget:
                self._logger.info(
                    f"agentic_search: token budget reached ({total_usage.total_tokens}/{token_budget}), stopping"
                )
                break

        # Format final results
        final_results = list(all_chunks.values())

        # Sort by score (highest first) if available
        final_results.sort(key=lambda r: r.score if r.score is not None else 0.0, reverse=True)

        formatted = format_with_citations(final_results, top_k=len(final_results))
        formatted["results"] = final_results
        formatted["queries_used"] = queries_used
        formatted["iterations"] = iterations_done
        formatted["total_chunks"] = len(final_results)
        formatted["usage"] = total_usage

        # Optionally include query -> result mapping for debugging
        if include_query_results:
            formatted["query_to_result"] = query_to_result

        self._logger.info(
            f"agentic_search.done: {iterations_done} iterations, "
            f"{len(queries_used)} queries, {len(final_results)} unique chunks, "
            f"tokens: {total_usage.total_tokens} (input: {total_usage.input_tokens}, output: {total_usage.output_tokens})"
        )

        return formatted
