"""Agentic query pipeline with iterative retrieval and LLM-based evaluation."""

from typing import Any, Dict, List, Optional, Tuple, Union

from ..citation.formatter import format_with_citations
from ..embedding.base import BaseEmbedder
from ..llm.base import BaseLLM
from ..llm.openai import OpenAILLM
from ..logging import get_logger
from ..models import SearchResult
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
            llm: LLM for query generation/evaluation (defaults to OpenAI)
            llm_api_key: API key for LLM (if not using default)
            llm_model: LLM model name
            namespace: Default namespace for searches
        """
        self.embedder = embedder
        self.store = store
        self.reranker = reranker
        self.namespace = namespace
        self._logger = get_logger("maktaba.pipeline.agentic")

        # Initialize LLM (default to OpenAI if not provided)
        if llm is not None:
            self.llm = llm
        else:
            self.llm = OpenAILLM(api_key=llm_api_key, model=llm_model)

    async def agentic_search(
        self,
        messages: List[Union[Dict[str, str], Tuple[str, str]]],
        *,
        max_iterations: int = 3,
        token_budget: int = 4096,
        top_k: int = 50,
        rerank_limit: int = 15,
        min_score: Optional[float] = None,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
        includeMetadata: bool = True,
    ) -> Dict[str, Any]:
        """
        Perform agentic search with iterative query generation.

        Args:
            messages: Chat history as list of dicts or tuples
            max_iterations: Maximum number of query generation iterations
            token_budget: Maximum token budget for iterations (approximate)
            top_k: Number of results to retrieve per query
            rerank_limit: Number of results to keep after reranking
            min_score: Minimum similarity score threshold
            namespace: Namespace for vector search
            filter: Metadata filter for vector search
            includeMetadata: Include metadata in results

        Returns:
            Dict with keys:
                - formatted_context: Citation-formatted text
                - citations: List of citation dicts
                - results: List of SearchResult objects
                - queries_used: List of generated query strings
                - iterations: Number of iterations performed
                - total_chunks: Total unique chunks retrieved
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
        iterations_done = 0
        estimated_tokens = 0  # Rough estimate

        # Add first user message as initial query
        last_user_msg = next((c for r, c in reversed(norm) if r == "user"), norm[-1][1])
        queries_used.append(last_user_msg)

        # Iterative search loop
        for iteration in range(max_iterations):
            iterations_done += 1

            # Generate new queries using LLM
            if iteration == 0:
                # First iteration: use user's question directly
                generated_queries = [
                    {"type": "semantic", "query": last_user_msg}
                ]
            else:
                # Subsequent iterations: generate new queries
                generated_queries = await self.llm.generate_queries(
                    messages=norm,
                    existing_queries=queries_used,
                    max_queries=5,  # Limit to avoid explosion
                )

            if not generated_queries:
                self._logger.info(f"agentic_search.iter_{iteration}: no new queries generated, stopping")
                break

            # Execute queries in parallel
            for query_dict in generated_queries:
                query_text = query_dict.get("query", "")
                query_type = query_dict.get("type", "semantic")

                if not query_text or query_text in queries_used:
                    continue  # Skip duplicates

                queries_used.append(query_text)

                try:
                    # Embed and retrieve
                    self._logger.info(
                        f"agentic_search.iter_{iteration}: executing {query_type} query: '{query_text[:50]}...'"
                    )
                    qvec = await self.embedder.embed_text(query_text, input_type="query")

                    results: List[SearchResult] = await self.store.query(
                        vector=qvec,
                        topK=top_k,
                        filter=filter,
                        includeMetadata=includeMetadata,
                        namespace=ns,
                    )

                    # Apply min_score filter
                    if min_score is not None:
                        results = [r for r in results if r.score is not None and r.score >= min_score]

                    # Rerank if available
                    if self.reranker is not None:
                        results = await self.reranker.rerank(query_text, results, top_k=rerank_limit)
                    else:
                        results = results[:rerank_limit]

                    # Add to chunk pool (deduplicate by ID)
                    for result in results:
                        if result.id not in all_chunks:
                            all_chunks[result.id] = result

                    self._logger.info(
                        f"agentic_search.iter_{iteration}: retrieved {len(results)} chunks, "
                        f"total unique: {len(all_chunks)}"
                    )

                except Exception as e:
                    self._logger.error(f"agentic_search.query_failed: {e}", exc_info=True)
                    continue

            # Estimate tokens used (very rough)
            estimated_tokens += len(last_user_msg.split()) * 2  # Query gen
            estimated_tokens += sum(len(c.text.split()) for c in all_chunks.values())  # Context

            # Evaluate if we have enough information
            if iteration < max_iterations - 1:  # Don't evaluate on last iteration
                sources_text = [chunk.text for chunk in all_chunks.values()]
                can_answer = await self.llm.evaluate_sources(
                    messages=norm,
                    sources=sources_text[:50],  # Limit to avoid huge context
                )

                self._logger.info(f"agentic_search.iter_{iteration}: canAnswer={can_answer}")

                if can_answer:
                    self._logger.info("agentic_search: sufficient information found, stopping early")
                    break

            # Check token budget
            if estimated_tokens >= token_budget:
                self._logger.info(f"agentic_search: token budget ({token_budget}) reached, stopping")
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

        self._logger.info(
            f"agentic_search.done: {iterations_done} iterations, "
            f"{len(queries_used)} queries, {len(final_results)} unique chunks"
        )

        return formatted
