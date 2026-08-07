"""Agentic Search v2: bounded retrieval, evidence assessment, and abstention."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, Union

from ..citation.formatter import format_with_citations
from ..embedding.base import BaseEmbedder
from ..keyword.base import BaseKeywordStore
from ..llm.base import BaseLLM
from ..llm.openai import OpenAILLM
from ..llm.prompts import AgenticPrompts
from ..logging import get_logger
from ..models import LLMUsage, SearchResult
from ..reranking.base import BaseReranker
from ..retrieval.fusion import RankedResultList, reciprocal_rank_fusion, select_diverse_results
from ..retrieval.relationships import RelationshipContextExpander
from ..storage.base import BaseVectorStore
from .agentic_models import (
    AgenticSearchConfig,
    EvidenceAssessment,
    EvidenceItem,
    EvidenceProvenance,
    IterationTrace,
    RetrievalAction,
    SearchQuery,
    StopReason,
)


class EvidenceAssessor(Protocol):
    """Optional objective-specific assessment hook for the retrieval loop."""

    async def assess(
        self,
        *,
        llm: BaseLLM,
        messages: List[Tuple[str, str]],
        evidence: List[EvidenceItem],
    ) -> Tuple[EvidenceAssessment, LLMUsage]: ...


class AgenticQueryPipeline:
    """Generic agentic retrieval with explicit evidence and stopping controls.

    Agentic Search v2 keeps the legacy ``agentic_search`` call compatible while
    adding global fusion/reranking, evidence-gap actions, relationship expansion,
    provenance, bounded context, and a machine-readable abstention signal.
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
        prompts: Optional[AgenticPrompts] = None,
        namespace: Optional[str] = None,
        use_max_completion_tokens: bool = False,
        omit_temperature: bool = False,
        llm_timeout_s: float = 30.0,
        llm_reasoning_effort: Optional[str] = None,
        llm_default_max_tokens: Optional[int] = None,
        config: Optional[AgenticSearchConfig] = None,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.reranker = reranker
        self.keyword_store = keyword_store
        self.namespace = namespace
        self.config = config
        self._logger = get_logger("maktaba.pipeline.agentic")
        self._expander = RelationshipContextExpander(store)

        if llm is not None:
            self.llm = llm
        else:
            self.llm = OpenAILLM(
                api_key=llm_api_key,
                model=llm_model,
                prompts=prompts,
                use_max_completion_tokens=use_max_completion_tokens,
                omit_temperature=omit_temperature,
                timeout_s=llm_timeout_s,
                reasoning_effort=llm_reasoning_effort,
                default_max_tokens=llm_default_max_tokens,
            )

    async def _execute_single_query(
        self,
        query: SearchQuery,
        *,
        config: AgenticSearchConfig,
        min_score: Optional[float],
        namespace: Optional[str],
        filter: Optional[Dict[str, Any]],
        includeMetadata: bool,
        includeRelationships: bool,
    ) -> List[SearchResult]:
        """Execute one retrieval tool call and fail locally on provider errors."""
        try:
            if query.type == "keyword":
                if self.keyword_store is None:
                    self._logger.warning(
                        "Keyword query requested but no keyword_store is available: '%s'",
                        query.query[:80],
                    )
                    return []
                return await self.keyword_store.search(
                    query=query.query,
                    limit=config.keyword_limit,
                    filter=filter,
                    namespace=namespace,
                )

            vector = await self.embedder.embed_text(query.query, input_type="query")
            results = await self.store.query(
                vector=vector,
                topK=config.top_k,
                filter=filter,
                includeMetadata=includeMetadata,
                includeRelationships=includeRelationships,
                namespace=namespace,
            )
            if min_score is not None:
                results = [result for result in results if result.score is not None and result.score >= min_score]
            # Per-query reranking would make scores incomparable across query
            # lists. V2 keeps each raw rank and reranks the fused pool once.
            return results[: config.rerank_limit]
        except Exception as exc:
            self._logger.error(
                "Query execution failed for '%s': %s",
                query.query[:80],
                exc,
                exc_info=True,
            )
            return []

    @staticmethod
    def _normalise_messages(
        messages: List[Union[Dict[str, str], Tuple[str, str]]],
    ) -> List[Tuple[str, str]]:
        normalised: List[Tuple[str, str]] = []
        for message in messages:
            if isinstance(message, tuple):
                role, content = message
            else:
                role = message.get("role", "user")
                content = message.get("content", "")
            normalised.append((role, content if isinstance(content, str) else str(content)))
        if not normalised:
            raise ValueError("messages must contain at least one item")
        return normalised

    @staticmethod
    def _parse_queries(values: Sequence[object]) -> List[SearchQuery]:
        queries: List[SearchQuery] = []
        for value in values:
            if not isinstance(value, dict):
                continue
            parsed = SearchQuery.from_mapping(value)
            if parsed is not None:
                queries.append(parsed)
        return queries

    async def _rank_candidates(
        self,
        question: str,
        candidates: Sequence[SearchResult],
        config: AgenticSearchConfig,
    ) -> List[SearchResult]:
        window = list(candidates[: config.candidate_limit])
        if self.reranker is not None and window:
            try:
                window = await self.reranker.rerank(
                    question,
                    window,
                    top_k=max(config.assessment_limit, config.evidence_limit),
                )
            except Exception as exc:
                self._logger.warning("Global reranking failed; using fused rank: %s", exc)
        return select_diverse_results(
            window,
            limit=max(config.assessment_limit, config.evidence_limit),
            max_per_document=config.max_per_document,
            diversity_metadata_keys=config.diversity_metadata_keys,
            max_per_diversity_group=config.max_per_diversity_group,
        )

    @staticmethod
    def _make_evidence(results: Sequence[SearchResult], limit: int) -> List[EvidenceItem]:
        return [
            EvidenceItem(
                id=result.id,
                text=result.text or "",
                rank=rank,
                score=result.score,
                metadata=result.metadata,
            )
            for rank, result in enumerate(results[:limit], start=1)
            if result.text
        ]

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
        includeRelationships: bool = False,
        include_query_results: bool = False,
        evidence_assessor: Optional[EvidenceAssessor] = None,
        config: Optional[AgenticSearchConfig] = None,
    ) -> Dict[str, Any]:
        """Run a bounded evidence-seeking loop.

        Existing keyword arguments remain supported. Passing ``config`` uses
        the complete v2 configuration object. The return value keeps all legacy
        keys and adds ``answerable``, ``stop_reason``, ``assessment``,
        ``provenance``, ``iteration_trace``, and evidence/candidate counts.
        """
        runtime = config or self.config
        if runtime is None:
            runtime = AgenticSearchConfig(
                max_iterations=max_iterations,
                max_queries_per_iter=max_queries_per_iter,
                max_total_queries=max(max_iterations * max_queries_per_iter, max_queries_per_iter + 1),
                token_budget=token_budget,
                top_k=top_k,
                rerank_limit=rerank_limit,
                keyword_limit=keyword_limit,
            )
        else:
            # Copy so this invocation cannot mutate a shared configuration.
            runtime = replace(runtime)

        normalised = self._normalise_messages(messages)
        last_user_message = next(
            (content for role, content in reversed(normalised) if role == "user"),
            normalised[-1][1],
        )
        try:
            standalone_question = await self.llm.condense_query(normalised[:-1], last_user_message)
        except Exception as exc:
            self._logger.warning("Query condensation failed; using latest user message: %s", exc)
            standalone_question = last_user_message

        effective_namespace = namespace or self.namespace
        ranked_lists: List[RankedResultList] = []
        all_chunks: Dict[str, SearchResult] = {}
        query_to_result: Dict[str, List[SearchResult]] = {}
        queries_used: List[str] = []
        seen_queries: set[str] = set()
        expanded_chunk_ids: List[str] = []
        expansion_parents: Dict[str, List[str]] = {}
        traces: List[IterationTrace] = []
        total_usage = LLMUsage()
        pending_actions: List[RetrievalAction] = []
        last_assessment: Optional[EvidenceAssessment] = None
        iterations_done = 0
        stagnant_iterations = 0
        fused_candidates: List[SearchResult] = []
        ranked_evidence: List[SearchResult] = []
        provenance: Dict[str, EvidenceProvenance] = {}
        stop_reason: Optional[StopReason] = None

        self._logger.info(
            "agentic_search.start max_iter=%d query_budget=%d token_budget=%d",
            runtime.max_iterations,
            runtime.max_total_queries,
            runtime.token_budget,
        )

        for iteration in range(runtime.max_iterations):
            iterations_done += 1
            trace = IterationTrace(iteration=iteration)
            iteration_actions = list(pending_actions)
            pending_actions = []

            search_queries = [
                SearchQuery(query=action.query or "", type=action.query_type, rationale=action.rationale)
                for action in iteration_actions
                if action.type == "search" and action.query
            ]
            expand_actions = [action for action in iteration_actions if action.type == "expand"]

            # The first iteration always searches the user's own question and
            # asks for diverse variants. Legacy LLMs continue through this path.
            if iteration == 0:
                search_queries.insert(0, SearchQuery(query=last_user_message, type="semantic"))

            if iteration == 0 or not iteration_actions:
                remaining_queries = runtime.max_total_queries - len(queries_used)
                if remaining_queries > 0:
                    generated, usage = await self.llm.generate_queries(
                        messages=normalised,
                        existing_queries=queries_used,
                        max_queries=min(runtime.max_queries_per_iter, remaining_queries),
                    )
                    total_usage += usage
                    search_queries.extend(self._parse_queries(generated))

            new_queries: List[SearchQuery] = []
            for query in search_queries:
                query_key = " ".join(query.query.casefold().split())
                if not query_key or query_key in seen_queries:
                    continue
                if len(queries_used) >= runtime.max_total_queries:
                    break
                seen_queries.add(query_key)
                queries_used.append(query.query)
                new_queries.append(query)
                if len(new_queries) >= runtime.max_queries_per_iter + (1 if iteration == 0 else 0):
                    break

            trace.actions.extend(
                {"type": "search", "query": query.query, "query_type": query.type, "rationale": query.rationale}
                for query in new_queries
            )
            trace.actions.extend(action.to_dict() for action in expand_actions)

            before_ids = set(all_chunks)
            if new_queries:
                tasks = [
                    self._execute_single_query(
                        query,
                        config=runtime,
                        min_score=min_score,
                        namespace=effective_namespace,
                        filter=filter,
                        includeMetadata=includeMetadata,
                        # Relationships are needed internally only when the
                        # caller explicitly enables graph expansion.
                        includeRelationships=includeRelationships,
                    )
                    for query in new_queries
                ]
                batches = await asyncio.gather(*tasks, return_exceptions=True)
                for query, batch in zip(new_queries, batches):
                    if isinstance(batch, BaseException):
                        self._logger.error("Query task failed: %s", batch)
                        continue
                    query_to_result[query.query] = batch
                    ranked_lists.append((query, iteration, batch))
                    trace.retrieved_chunk_ids.extend(result.id for result in batch)
                    for result in batch:
                        all_chunks.setdefault(result.id, result)

            if includeRelationships and expand_actions:
                remaining_expansion = runtime.max_expanded_chunks - len(expanded_chunk_ids)
                for action in expand_actions:
                    if remaining_expansion <= 0:
                        break
                    seeds = [all_chunks[source_id] for source_id in action.source_ids if source_id in all_chunks]
                    expanded, fetched_parents = await self._expander.expand(
                        seeds,
                        relationship_types=action.relationship_types,
                        max_hops=runtime.max_expansion_hops,
                        max_chunks=remaining_expansion,
                        namespace=effective_namespace,
                        filter=filter,
                        includeMetadata=includeMetadata,
                    )
                    if expanded:
                        synthetic_query = SearchQuery(
                            query=f"relationship expansion from {', '.join(action.source_ids)}",
                            type="semantic",
                            rationale=action.rationale,
                        )
                        ranked_lists.append((synthetic_query, iteration, expanded))
                    for result in expanded:
                        all_chunks.setdefault(result.id, result)
                        if result.id not in expanded_chunk_ids:
                            expanded_chunk_ids.append(result.id)
                        expansion_parents.setdefault(result.id, []).extend(fetched_parents.get(result.id, []))
                    trace.expanded_chunk_ids.extend(result.id for result in expanded)
                    remaining_expansion = runtime.max_expanded_chunks - len(expanded_chunk_ids)

            new_ids = list(set(all_chunks) - before_ids)
            trace.new_chunk_ids = new_ids
            if new_ids:
                stagnant_iterations = 0
            else:
                stagnant_iterations += 1

            fused_candidates, provenance = reciprocal_rank_fusion(
                ranked_lists,
                rrf_k=runtime.rrf_k,
                limit=runtime.candidate_limit,
            )
            for result_id, parent_ids in expansion_parents.items():
                if result_id in provenance:
                    provenance[result_id].expanded_from = list(dict.fromkeys(parent_ids))
            ranked_evidence = await self._rank_candidates(standalone_question, fused_candidates, runtime)
            evidence = self._make_evidence(ranked_evidence, runtime.assessment_limit)

            trace.candidate_count = len(fused_candidates)
            trace.evidence_ids = [item.id for item in evidence]

            if total_usage.total_tokens >= runtime.token_budget:
                stop_reason = StopReason.TOKEN_BUDGET_EXHAUSTED
                trace.usage_total_tokens = total_usage.total_tokens
                traces.append(trace)
                break

            if evidence_assessor is None:
                assessment, usage = await self.llm.assess_evidence(
                    messages=normalised,
                    evidence=evidence,
                )
            else:
                assessment, usage = await evidence_assessor.assess(
                    llm=self.llm,
                    messages=normalised,
                    evidence=evidence,
                )
            total_usage += usage
            last_assessment = assessment
            trace.assessment = assessment.to_dict()
            trace.usage_total_tokens = total_usage.total_tokens
            traces.append(trace)

            if not assessment.valid:
                stop_reason = StopReason.INVALID_ASSESSMENT
                break
            if assessment.answerable:
                stop_reason = StopReason.SUFFICIENT_EVIDENCE
                break
            if total_usage.total_tokens >= runtime.token_budget:
                stop_reason = StopReason.TOKEN_BUDGET_EXHAUSTED
                break
            if stagnant_iterations > runtime.no_progress_patience:
                stop_reason = StopReason.NO_RESULTS if not all_chunks else StopReason.NO_PROGRESS
                break

            pending_actions = assessment.next_actions
            if len(queries_used) >= runtime.max_total_queries and not any(
                action.type == "expand" for action in pending_actions
            ):
                stop_reason = StopReason.QUERY_LIMIT_REACHED
                break

        if stop_reason is None:
            if not all_chunks:
                stop_reason = StopReason.NO_RESULTS
            elif iterations_done >= runtime.max_iterations:
                stop_reason = StopReason.ITERATION_LIMIT_REACHED
            else:
                stop_reason = StopReason.INSUFFICIENT_EVIDENCE

        final_results = select_diverse_results(
            ranked_evidence or fused_candidates,
            limit=runtime.evidence_limit,
            max_per_document=runtime.max_per_document,
            diversity_metadata_keys=runtime.diversity_metadata_keys,
            max_per_diversity_group=runtime.max_per_diversity_group,
        )
        formatted = format_with_citations(final_results, top_k=len(final_results))
        formatted.update(
            {
                "results": final_results,
                "queries_used": queries_used,
                "iterations": iterations_done,
                "total_chunks": len(all_chunks),
                "usage": total_usage,
                "answerable": bool(last_assessment and last_assessment.answerable),
                "stop_reason": stop_reason.value,
                "candidate_count": len(fused_candidates),
                "evidence_count": len(final_results),
                "assessment": last_assessment.to_dict() if last_assessment else None,
                "provenance": {
                    result_id: item.to_dict()
                    for result_id, item in provenance.items()
                },
                "iteration_trace": [trace.to_dict() for trace in traces],
                "expanded_chunk_ids": expanded_chunk_ids,
            }
        )
        if include_query_results:
            formatted["query_to_result"] = query_to_result

        self._logger.info(
            "agentic_search.done iterations=%d queries=%d candidates=%d evidence=%d answerable=%s stop=%s tokens=%d",
            iterations_done,
            len(queries_used),
            len(fused_candidates),
            len(final_results),
            formatted["answerable"],
            stop_reason.value,
            total_usage.total_tokens,
        )
        return formatted
