"""Generic agentic collection built on the Agentic Search v2 loop."""

from __future__ import annotations

import unicodedata
from collections import Counter
from typing import Any, Dict, List, Sequence, Tuple, Union

from ..citation.spans import SpanNormalizer, verify_evidence_span
from ..collection_models import (
    CollectionCandidate,
    CollectionGoal,
    CollectionResult,
    EvidenceSpan,
    RejectedCollectionCandidate,
)
from ..llm.base import BaseLLM
from ..models import LLMUsage, SearchResult
from .agentic import AgenticQueryPipeline
from .agentic_models import CoverageItem, EvidenceAssessment, EvidenceItem

CollectionMessage = Union[Dict[str, str], Tuple[str, str]]


def _canonical_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


class CollectionEvidenceAssessor:
    """Extract, verify, accumulate, and plan toward a collection goal."""

    def __init__(
        self,
        goal: CollectionGoal,
        *,
        max_rejected_diagnostics: int = 100,
        span_normalizer: SpanNormalizer | None = None,
    ) -> None:
        self.goal = goal
        self.accepted_items: List[EvidenceSpan] = []
        self.rejected_candidates: List[RejectedCollectionCandidate] = []
        self._seen_text: set[str] = set()
        self._document_counts: Counter[str] = Counter()
        self._group_counts: Counter[Tuple[str, ...]] = Counter()
        self._max_rejected_diagnostics = max_rejected_diagnostics
        self._span_normalizer = span_normalizer

    def _reject(self, candidate: CollectionCandidate | None, reason: str) -> None:
        if len(self.rejected_candidates) >= self._max_rejected_diagnostics:
            return
        self.rejected_candidates.append(
            RejectedCollectionCandidate(
                source_id=candidate.source_id if candidate is not None else None,
                reason=reason,
            )
        )

    def _diversity_group(self, span: EvidenceSpan) -> Tuple[str, ...] | None:
        if not self.goal.diversity_metadata_keys:
            return None
        return tuple(str(span.source_metadata.get(key, "")) for key in self.goal.diversity_metadata_keys)

    def _accept_candidate(self, candidate: CollectionCandidate, evidence: Sequence[EvidenceItem]) -> None:
        source_results = [
            SearchResult(id=item.id, score=item.score, metadata=dict(item.metadata))
            for item in evidence
        ]
        verification = verify_evidence_span(
            candidate,
            source_results,
            min_characters=self.goal.min_characters,
            require_exact_span=self.goal.require_exact_span,
            normalizer=self._span_normalizer,
        )
        if not verification.valid or verification.span is None:
            self._reject(candidate, verification.reason or "invalid_source_span")
            return

        span = verification.span
        canonical = _canonical_text(span.text)
        if canonical in self._seen_text:
            self._reject(candidate, "duplicate_text")
            return
        if self._document_counts[span.document_id] >= self.goal.max_per_document:
            self._reject(candidate, "document_diversity_limit")
            return

        group = self._diversity_group(span)
        if group is not None and self._group_counts[group] >= self.goal.max_per_diversity_group:
            self._reject(candidate, "metadata_diversity_limit")
            return

        self.accepted_items.append(span)
        self._seen_text.add(canonical)
        self._document_counts[span.document_id] += 1
        if group is not None:
            self._group_counts[group] += 1

    async def assess(
        self,
        *,
        llm: BaseLLM,
        messages: List[Tuple[str, str]],
        evidence: List[EvidenceItem],
    ) -> Tuple[EvidenceAssessment, LLMUsage]:
        remaining = self.goal.target_count - len(self.accepted_items)
        extraction_usage = LLMUsage()
        if remaining > 0 and evidence:
            candidates, extraction_usage = await llm.extract_collection_items(
                goal=self.goal,
                messages=messages,
                evidence=evidence,
                accepted_items=list(self.accepted_items),
            )
            for candidate in candidates[:remaining]:
                self._accept_candidate(candidate, evidence)
                if len(self.accepted_items) >= self.goal.target_count:
                    break

        complete = len(self.accepted_items) >= self.goal.target_count
        supporting_ids = list(dict.fromkeys(item.source_id for item in self.accepted_items))
        coverage = [
            CoverageItem(
                requirement=f"verified collection items: {len(self.accepted_items)}/{self.goal.target_count}",
                covered=complete,
                supporting_source_ids=supporting_ids,
            )
        ]
        if complete:
            return (
                EvidenceAssessment(
                    answerable=True,
                    confidence=1.0,
                    coverage=coverage,
                    supporting_source_ids=supporting_ids,
                    valid=True,
                ),
                extraction_usage,
            )

        plan, planning_usage = await llm.plan_collection_actions(
            goal=self.goal,
            messages=messages,
            evidence=evidence,
            accepted_items=list(self.accepted_items),
            rejected_candidates=list(self.rejected_candidates),
        )
        plan.answerable = False
        plan.coverage = coverage
        plan.supporting_source_ids = supporting_ids
        if not plan.missing_information:
            plan.missing_information = [
                f"{self.goal.target_count - len(self.accepted_items)} additional verified item(s)"
            ]
        return plan, extraction_usage + planning_usage


class AgenticCollectionPipeline:
    """Collect verified exact spans using a configured Agentic Query pipeline."""

    def __init__(
        self,
        search_pipeline: AgenticQueryPipeline,
        *,
        span_normalizer: SpanNormalizer | None = None,
    ) -> None:
        self.search_pipeline = search_pipeline
        self.span_normalizer = span_normalizer

    @staticmethod
    def _with_goal(messages: List[CollectionMessage], goal: CollectionGoal) -> List[CollectionMessage]:
        augmented: List[CollectionMessage] = list(messages)
        for index in range(len(augmented) - 1, -1, -1):
            message = augmented[index]
            if isinstance(message, tuple):
                role, content = message
                if role == "user":
                    augmented[index] = (role, f"{content}\n\n{goal.to_prompt()}")
                    return augmented
            elif message.get("role", "user") == "user":
                updated = dict(message)
                updated["content"] = f"{message.get('content', '')}\n\n{goal.to_prompt()}"
                augmented[index] = updated
                return augmented
        augmented.append(("user", goal.to_prompt()))
        return augmented

    async def collect(
        self,
        messages: List[CollectionMessage],
        *,
        goal: CollectionGoal,
        **search_kwargs: Any,
    ) -> CollectionResult:
        """Run collection and return verified items plus bounded diagnostics."""
        assessor = CollectionEvidenceAssessor(
            goal,
            span_normalizer=self.span_normalizer,
        )
        result = await self.search_pipeline.agentic_search(
            messages=self._with_goal(messages, goal),
            evidence_assessor=assessor,
            **search_kwargs,
        )
        complete = len(assessor.accepted_items) >= goal.target_count
        usage = result.get("usage")
        if not isinstance(usage, LLMUsage):
            usage = LLMUsage()
        raw_results = result.get("results", [])
        search_results = [item for item in raw_results if isinstance(item, SearchResult)]
        return CollectionResult(
            items=list(assessor.accepted_items[: goal.target_count]),
            target_count=goal.target_count,
            complete=complete,
            stop_reason="target_reached" if complete else str(result.get("stop_reason", "insufficient_evidence")),
            queries_used=[str(item) for item in result.get("queries_used", [])],
            iterations=int(result.get("iterations", 0)),
            usage=usage,
            rejected_candidates=list(assessor.rejected_candidates),
            search_results=search_results,
            diagnostics={
                "candidate_count": int(result.get("candidate_count", 0)),
                "evidence_count": int(result.get("evidence_count", 0)),
                "rejection_counts": dict(
                    Counter(item.reason for item in assessor.rejected_candidates)
                ),
                "expanded_chunk_ids": list(result.get("expanded_chunk_ids", [])),
                "assessment": result.get("assessment"),
                "iteration_trace": list(result.get("iteration_trace", [])),
            },
        )
