"""Focused tests for Agentic Search v2 controls."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import pytest

from maktaba.citation.verifier import verify_answer
from maktaba.embedding.base import BaseEmbedder
from maktaba.llm.base import BaseLLM
from maktaba.models import LLMUsage, NodeRelationship, SearchResult
from maktaba.pipeline.agentic import AgenticQueryPipeline
from maktaba.pipeline.agentic_models import (
    AgenticSearchConfig,
    EvidenceAssessment,
    EvidenceItem,
    RetrievalAction,
)
from maktaba.retrieval.fusion import reciprocal_rank_fusion, select_diverse_results
from maktaba.storage.base import BaseVectorStore


class QueryEmbedder(BaseEmbedder):
    @property
    def dimension(self) -> int:
        return 1

    @property
    def model(self) -> str:
        return "query-aware"

    async def embed_batch(self, texts: List[str], input_type: str = "document") -> List[List[float]]:
        return [[1.0 if "missing detail" in text else 0.0] for text in texts]


class ActionStore(BaseVectorStore):
    def __init__(self) -> None:
        self.fetch_scopes: List[Tuple[str | None, Dict[str, Any] | None]] = []

    async def upsert(self, chunks, namespace=None) -> None:
        return None

    async def query(
        self,
        vector,
        topK: int = 10,
        filter=None,
        includeMetadata: bool = True,
        includeRelationships: bool = False,
        namespace=None,
    ) -> List[SearchResult]:
        if vector[0] == 1.0:
            return [SearchResult(id="book#detail", score=0.95, metadata={"text": "The missing detail is here."})]
        return [
            SearchResult(
                id="book#seed",
                score=0.9,
                metadata={"text": "An incomplete introduction.", "tenant": "a"},
                relationships={"NEXT": NodeRelationship(node_id="book#next")}
                if includeRelationships
                else None,
            )
        ]

    async def fetch_by_ids(
        self,
        ids: List[str],
        *,
        namespace=None,
        filter=None,
        includeMetadata: bool = True,
        includeRelationships: bool = False,
    ) -> List[SearchResult]:
        self.fetch_scopes.append((namespace, filter))
        if ids == ["book#next"] and namespace == "tenant-a" and filter == {"tenant": "a"}:
            return [SearchResult(id="book#next", metadata={"text": "Continuation evidence.", "tenant": "a"})]
        return []

    async def delete(self, ids, namespace=None) -> None:
        return None

    async def list(self, prefix=None, limit: int = 100, namespace=None) -> List[str]:
        return []

    async def get_dimensions(self) -> int:
        return 1


class StructuredLLM(BaseLLM):
    def __init__(self, assessments: Sequence[EvidenceAssessment]) -> None:
        self.assessments = list(assessments)
        self.assessment_calls = 0
        self.generation_calls = 0

    async def complete_text(self, **kwargs):
        return "", LLMUsage()

    async def complete_json(self, **kwargs):
        return {}, LLMUsage()

    async def generate_queries(
        self,
        messages: List[Tuple[str, str]],
        existing_queries: List[str],
        max_queries: int = 10,
    ):
        self.generation_calls += 1
        return [], LLMUsage(input_tokens=1)

    async def evaluate_sources(self, messages, sources):
        return False, LLMUsage()

    async def assess_evidence(
        self,
        messages: List[Tuple[str, str]],
        evidence: List[EvidenceItem],
    ):
        index = min(self.assessment_calls, len(self.assessments) - 1)
        self.assessment_calls += 1
        return self.assessments[index], LLMUsage(input_tokens=2)


def test_evidence_assessment_rejects_string_boolean_and_unknown_ids():
    invalid = EvidenceAssessment.from_mapping({"answerable": "false"})
    assert invalid.answerable is False
    assert invalid.valid is False

    valid = EvidenceAssessment.from_mapping(
        {
            "answerable": False,
            "supporting_source_ids": ["known", "invented"],
            "next_actions": [
                {"type": "expand", "source_ids": ["invented"], "relationship_types": ["NEXT"]}
            ],
        },
        valid_source_ids=["known"],
    )
    assert valid.supporting_source_ids == ["known"]
    assert valid.next_actions == []


def test_reciprocal_rank_fusion_records_cross_query_provenance():
    first = [SearchResult(id="shared", score=0.8), SearchResult(id="only-a", score=0.9)]
    second = [SearchResult(id="only-b", score=0.95), SearchResult(id="shared", score=0.7)]
    from maktaba.pipeline.agentic_models import SearchQuery

    fused, provenance = reciprocal_rank_fusion(
        [
            (SearchQuery("question a"), 0, first),
            (SearchQuery("question b"), 0, second),
        ]
    )
    assert fused[0].id == "shared"
    assert provenance["shared"].queries == ["question a", "question b"]
    assert provenance["shared"].ranks == [1, 2]


def test_evidence_selection_can_diversify_by_metadata():
    results = [
        SearchResult(
            id="a-1",
            score=0.99,
            metadata={"text": "first passage", "author": "author-a"},
        ),
        SearchResult(
            id="a-2",
            score=0.98,
            metadata={"text": "second passage", "author": "author-a"},
        ),
        SearchResult(
            id="b-1",
            score=0.97,
            metadata={"text": "third passage", "author": "author-b"},
        ),
    ]

    selected = select_diverse_results(
        results,
        limit=3,
        diversity_metadata_keys=("author",),
        max_per_diversity_group=1,
    )

    assert [item.id for item in selected] == ["a-1", "b-1"]


@pytest.mark.asyncio
async def test_agentic_search_executes_gap_search_action_and_abstains_until_supported():
    llm = StructuredLLM(
        [
            EvidenceAssessment(
                answerable=False,
                missing_information=["missing detail"],
                next_actions=[RetrievalAction(type="search", query="missing detail", query_type="semantic")],
            ),
            EvidenceAssessment(
                answerable=True,
                confidence=0.9,
                supporting_source_ids=["book#detail"],
            ),
        ]
    )
    pipeline = AgenticQueryPipeline(QueryEmbedder(), ActionStore(), llm=llm)
    result = await pipeline.agentic_search(
        [("user", "Initial question")],
        config=AgenticSearchConfig(max_iterations=3, evidence_limit=5),
    )

    assert result["answerable"] is True
    assert result["stop_reason"] == "sufficient_evidence"
    assert result["queries_used"] == ["Initial question", "missing detail"]
    assert result["assessment"]["supporting_source_ids"] == ["book#detail"]
    assert len(result["iteration_trace"]) == 2


@pytest.mark.asyncio
async def test_agentic_relationship_expansion_propagates_scope():
    store = ActionStore()
    llm = StructuredLLM(
        [
            EvidenceAssessment(
                answerable=False,
                next_actions=[
                    RetrievalAction(type="expand", source_ids=["book#seed"], relationship_types=["NEXT"])
                ],
            ),
            EvidenceAssessment(answerable=True, supporting_source_ids=["book#next"]),
        ]
    )
    pipeline = AgenticQueryPipeline(QueryEmbedder(), store, llm=llm)
    result = await pipeline.agentic_search(
        [("user", "Initial question")],
        namespace="tenant-a",
        filter={"tenant": "a"},
        includeRelationships=True,
        config=AgenticSearchConfig(max_iterations=3, evidence_limit=5),
    )

    assert result["answerable"] is True
    assert result["expanded_chunk_ids"] == ["book#next"]
    assert result["provenance"]["book#next"]["expanded_from"] == ["book#seed"]
    assert store.fetch_scopes == [("tenant-a", {"tenant": "a"})]


@pytest.mark.asyncio
async def test_answer_verifier_reports_invalid_citations_quotes_and_uncited_claims():
    evidence = [SearchResult(id="doc#1", metadata={"text": "The supported exact quotation appears here."})]
    report = await verify_answer(
        'A supported claim with "The supported exact quotation". [1] '
        'A second factual statement has no citation. An invalid reference is used here. [3]',
        evidence,
    )
    assert report.valid is False
    assert report.cited_source_ids == ["doc#1"]
    assert report.invalid_citations == ["3"]
    assert "A second factual statement has no citation." in report.uncited_claims
