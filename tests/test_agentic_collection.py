"""Tests for generic verified evidence collection."""

from __future__ import annotations

from typing import List, Tuple

import pytest

from maktaba.citation import verify_evidence_span
from maktaba.embedding.base import BaseEmbedder
from maktaba.llm.base import BaseLLM
from maktaba.llm.openai import OpenAILLM
from maktaba.models import LLMUsage, SearchResult
from maktaba.pipeline import AgenticCollectionPipeline, AgenticSearchConfig, CollectionGoal
from maktaba.pipeline.agentic import AgenticQueryPipeline
from maktaba.pipeline.agentic_models import EvidenceAssessment, EvidenceItem, RetrievalAction
from maktaba.pipeline.collection_models import (
    CollectionCandidate,
    EvidenceSpan,
    RejectedCollectionCandidate,
)
from maktaba.storage.base import BaseVectorStore


class CollectionEmbedder(BaseEmbedder):
    @property
    def dimension(self) -> int:
        return 1

    @property
    def model(self) -> str:
        return "collection-test"

    async def embed_batch(self, texts: List[str], input_type: str = "document") -> List[List[float]]:
        return [[1.0 if "second distinct" in text else 0.0] for text in texts]


class CollectionStore(BaseVectorStore):
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
            return [
                SearchResult(
                    id="book-2#page-2",
                    score=0.95,
                    metadata={"text": "The second distinct exact quotation is here.", "author_id": 2},
                )
            ]
        return [
            SearchResult(
                id="book-1#page-1",
                score=0.9,
                metadata={"text": "The first exact quotation is here.\nWith context.", "author_id": 1},
            )
        ]

    async def delete(self, ids, namespace=None) -> None:
        return None

    async def list(self, prefix=None, limit: int = 100, namespace=None) -> List[str]:
        return []

    async def get_dimensions(self) -> int:
        return 1


class CollectionLLM(BaseLLM):
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
        return [], LLMUsage(input_tokens=1)

    async def evaluate_sources(self, messages, sources):
        return False, LLMUsage()

    async def extract_collection_items(
        self,
        *,
        goal: CollectionGoal,
        messages: List[Tuple[str, str]],
        evidence: List[EvidenceItem],
        accepted_items: List[EvidenceSpan],
    ):
        if not accepted_items:
            return [
                CollectionCandidate(
                    source_id="book-1#page-1",
                    text="The first exact quotation is here. With context.",
                )
            ], LLMUsage(input_tokens=2, output_tokens=1)
        return [
            CollectionCandidate(
                source_id="book-2#page-2",
                text="The second distinct exact quotation is here.",
            )
        ], LLMUsage(input_tokens=2, output_tokens=1)

    async def plan_collection_actions(
        self,
        *,
        goal: CollectionGoal,
        messages: List[Tuple[str, str]],
        evidence: List[EvidenceItem],
        accepted_items: List[EvidenceSpan],
        rejected_candidates: List[RejectedCollectionCandidate],
    ):
        return EvidenceAssessment(
            answerable=False,
            next_actions=[
                RetrievalAction(type="search", query="second distinct quote", query_type="semantic")
            ],
        ), LLMUsage(input_tokens=1)


def test_span_verification_maps_whitespace_to_original_text():
    source = SearchResult(id="book#1", metadata={"text": "An exact quote\nwith a line break."})
    result = verify_evidence_span(
        CollectionCandidate(source_id="book#1", text="An exact quote with a line break."),
        [source],
    )

    assert result.valid is True
    assert result.span is not None
    assert result.span.text == "An exact quote\nwith a line break."
    assert source.text[result.span.start_offset : result.span.end_offset] == result.span.text


def test_span_verification_rejects_unknown_or_invented_text():
    source = SearchResult(id="book#1", metadata={"text": "Only grounded text exists here."})
    unknown = verify_evidence_span(
        CollectionCandidate(source_id="book#2", text="Only grounded text exists here."),
        [source],
    )
    invented = verify_evidence_span(
        CollectionCandidate(source_id="book#1", text="This sentence was invented by a model."),
        [source],
    )

    assert unknown.valid is False
    assert unknown.reason == "unknown_source_id"
    assert invented.valid is False
    assert invented.reason == "span_not_in_source"


@pytest.mark.asyncio
async def test_collection_accumulates_verified_items_until_target():
    search = AgenticQueryPipeline(CollectionEmbedder(), CollectionStore(), llm=CollectionLLM())
    pipeline = AgenticCollectionPipeline(search)
    result = await pipeline.collect(
        [("user", "Find two quotations")],
        goal=CollectionGoal(target_count=2, item_description="distinct exact quotations"),
        config=AgenticSearchConfig(max_iterations=3, evidence_limit=5, max_per_document=4),
    )

    assert result.complete is True
    assert result.stop_reason == "target_reached"
    assert [item.source_id for item in result.items] == ["book-1#page-1", "book-2#page-2"]
    assert result.items[0].text == "The first exact quotation is here.\nWith context."
    assert "second distinct quote" in result.queries_used
    assert result.iterations == 2


class InvalidCollectionLLM(CollectionLLM):
    async def extract_collection_items(self, **kwargs):
        return [
            CollectionCandidate(source_id="book-1#page-1", text="An invented quotation that is absent.")
        ], LLMUsage(input_tokens=1)

    async def plan_collection_actions(self, **kwargs):
        return EvidenceAssessment(answerable=False, next_actions=[]), LLMUsage(input_tokens=1)


@pytest.mark.asyncio
async def test_collection_returns_partial_diagnostics_without_leaking_invalid_item():
    search = AgenticQueryPipeline(CollectionEmbedder(), CollectionStore(), llm=InvalidCollectionLLM())
    pipeline = AgenticCollectionPipeline(search)
    result = await pipeline.collect(
        [("user", "Find one quotation")],
        goal=CollectionGoal(target_count=1, item_description="an exact quotation"),
        config=AgenticSearchConfig(max_iterations=1, evidence_limit=5),
    )

    assert result.complete is False
    assert result.items == []
    assert result.rejected_candidates[0].reason == "span_not_in_source"
    assert result.stop_reason == "iteration_limit_reached"


def test_collection_goal_rejects_invalid_limits():
    with pytest.raises(ValueError):
        CollectionGoal(target_count=0, item_description="quotes")
    with pytest.raises(ValueError):
        CollectionGoal(target_count=1, item_description=" ")


class StubOpenAICollectionLLM(OpenAILLM):
    def __init__(self, responses: List[dict[str, object]]) -> None:
        super().__init__(api_key="test")
        self.responses = list(responses)

    async def complete_json(self, **kwargs):
        return self.responses.pop(0), LLMUsage(input_tokens=2, output_tokens=1)


@pytest.mark.asyncio
async def test_openai_collection_hooks_parse_candidates_and_filter_unknown_expansion_ids():
    llm = StubOpenAICollectionLLM([
        {
            "items": [
                {"source_id": "known", "text": "A known exact source passage.", "attributes": {"kind": "quote"}},
                {"source_id": "ignored", "text": "Capped by remaining count."},
            ]
        },
        {
            "answerable": True,
            "next_actions": [
                {"type": "expand", "source_ids": ["known", "invented"], "relationship_types": ["NEXT"]}
            ],
        },
    ])
    goal = CollectionGoal(target_count=1, item_description="one exact passage")
    evidence = [EvidenceItem(id="known", text="A known exact source passage.", rank=1)]

    candidates, _ = await llm.extract_collection_items(
        goal=goal,
        messages=[("user", "Collect it")],
        evidence=evidence,
        accepted_items=[],
    )
    plan, _ = await llm.plan_collection_actions(
        goal=goal,
        messages=[("user", "Collect it")],
        evidence=evidence,
        accepted_items=[],
        rejected_candidates=[],
    )

    assert len(candidates) == 1
    assert candidates[0].source_id == "known"
    assert candidates[0].attributes == {"kind": "quote"}
    assert plan.answerable is False
    assert plan.next_actions[0].source_ids == ["known"]
