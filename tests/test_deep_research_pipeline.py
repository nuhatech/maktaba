from typing import Any, AsyncIterator, Dict, List, Sequence, Tuple

import pytest

from maktaba.embedding.base import BaseEmbedder
from maktaba.llm.base import BaseLLM
from maktaba.models import LLMUsage, SearchResult
from maktaba.pipeline.deep_research import (
    DeepResearchConfig,
    DeepResearchModelConfig,
    DeepResearchPipeline,
    DeepResearchPrompts,
    DeepResearchQueryOptions,
    SearchResultsCollection,
    SearchResultView,
    create_deep_research_pipeline,
)
from maktaba.storage.base import BaseVectorStore


class FakeLLM(BaseLLM):
    def __init__(self) -> None:
        self.text_responses: Dict[Tuple[str, str], Tuple[str, LLMUsage]] = {}
        self.json_responses: Dict[Tuple[str, str], Tuple[Dict[str, Any], LLMUsage]] = {}
        self.stream_responses: Dict[Tuple[str, str], Tuple[str, LLMUsage]] = {}

    def register_text(self, system: str, prompt: str, response: str) -> None:
        self.text_responses[(system, prompt)] = (response, LLMUsage())

    def register_json(self, system: str, prompt: str, response: Dict[str, Any]) -> None:
        self.json_responses[(system, prompt)] = (response, LLMUsage())

    def register_stream(self, system: str, prompt: str, response: str) -> None:
        self.stream_responses[(system, prompt)] = (response, LLMUsage())

    async def complete_text(
        self,
        *,
        system: str,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> Tuple[str, LLMUsage]:
        return self.text_responses.get((system, prompt), ("", LLMUsage()))

    async def complete_json(
        self,
        *,
        system: str,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> Tuple[Dict[str, Any], LLMUsage]:
        return self.json_responses.get((system, prompt), ({}, LLMUsage()))

    async def stream_text(
        self,
        *,
        system: str,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> Tuple[AsyncIterator[str], LLMUsage]:
        text, usage = self.stream_responses.get((system, prompt), ("", LLMUsage()))

        async def _generator() -> AsyncIterator[str]:
            yield text

        return _generator(), usage

    async def generate_queries(self, *args, **kwargs):  # pragma: no cover - unused in tests
        raise NotImplementedError

    async def evaluate_sources(self, *args, **kwargs):  # pragma: no cover - unused in tests
        raise NotImplementedError


class DummyEmbedder(BaseEmbedder):
    def __init__(self, *, dimension: int = 1536, model: str = "dummy") -> None:
        self._dimension = dimension
        self._model = model

    async def embed_batch(self, texts: List[str], input_type: str = "document") -> List[List[float]]:  # type: ignore[override]
        return [[0.0] * self._dimension for _ in texts]

    @property
    def dimension(self) -> int:  # type: ignore[override]
        return self._dimension

    @property
    def model(self) -> str:  # type: ignore[override]
        return self._model


class DummyVectorStore(BaseVectorStore):
    def __init__(self, *, dimension: int = 1536) -> None:
        self._dimension = dimension

    async def upsert(self, chunks, namespace=None) -> None:  # type: ignore[override]
        return None

    async def query(  # type: ignore[override]
        self,
        vector,
        topK: int = 10,
        filter=None,
        includeMetadata: bool = True,
        includeRelationships: bool = False,
        namespace=None,
    ) -> List[SearchResult]:
        return []

    async def delete(self, ids, namespace=None) -> None:  # type: ignore[override]
        return None

    async def list(self, prefix=None, limit: int = 100, namespace=None) -> List[str]:  # type: ignore[override]
        return []

    async def get_dimensions(self) -> int:  # type: ignore[override]
        return self._dimension


class FakeQueryPipeline:
    def __init__(self, batches: Dict[str, List[SearchResult]]) -> None:
        self.batches = batches
        self.calls: List[str] = []

    async def search(
        self,
        query: str,
        *,
        rerank: bool = True,
        top_k: int | None = None,
        rerank_limit: int | None = None,
        min_score: float | None = None,
        namespace: str | None = None,
        filter: Dict[str, Any] | None = None,
        includeMetadata: bool = True,
    ) -> Dict[str, Any]:
        self.calls.append(query)
        return {"results": list(self.batches.get(query, []))}


@pytest.mark.asyncio
async def test_deep_research_pipeline_basic_flow():
    prompts = DeepResearchPrompts(
        planning_prompt="plan",
        plan_parsing_prompt="plan-parse",
        raw_content_summariser_prompt="summarise",
        evaluation_prompt="evaluate",
        evaluation_parsing_prompt="evaluate-parse",
        filter_prompt="filter",
        source_parsing_prompt="filter-parse",
        answer_prompt="answer",
    )
    planning_llm = FakeLLM()
    json_llm = FakeLLM()
    summary_llm = FakeLLM()
    answer_llm = FakeLLM()

    json_llm.register_json(
        prompts.planning_prompt,
        "Research Topic: Climate impacts",
        {"queries": ["climate effects 2024", "economic impact climate change"]},
    )

    summary_prompt_1 = "<Raw Content>Record warmth observed</Raw Content>\n\n<Research Topic>climate effects 2024</Research Topic>"
    summary_prompt_2 = "<Raw Content>GDP loss projections</Raw Content>\n\n<Research Topic>economic impact climate change</Research Topic>"
    summary_llm.register_text(prompts.raw_content_summariser_prompt, summary_prompt_1, "Summary doc1")
    summary_llm.register_text(prompts.raw_content_summariser_prompt, summary_prompt_2, "Summary doc2")

    evaluation_prompt = (
        "<Research Topic>Climate impacts</Research Topic>\n\n"
        "<Search Queries Used>['climate effects 2024', 'economic impact climate change']</Search Queries Used>\n\n"
        "<Current Search Results>[1] ID: q1\nMetadata: {'text': 'Record warmth observed'}\nContent: Summary doc1\n\n"
        "[2] ID: q2\nMetadata: {'text': 'GDP loss projections'}\nContent: Summary doc2</Current Search Results>"
    )
    planning_llm.register_text(prompts.evaluation_prompt, evaluation_prompt, "No further queries needed.")
    json_llm.register_json(
        prompts.evaluation_parsing_prompt,
        "Evaluation to be parsed: No further queries needed.",
        {"queries": []},
    )

    filter_prompt = (
        "<Research Topic>Climate impacts</Research Topic>\n\n"
        "<Current Search Results>[1] ID: q1\nMetadata: {'text': 'Record warmth observed'}\nContent: Summary doc1\n\n"
        "[2] ID: q2\nMetadata: {'text': 'GDP loss projections'}\nContent: Summary doc2</Current Search Results>"
    )
    planning_llm.register_text(prompts.filter_prompt, filter_prompt, "Keep source 1 only")
    json_llm.register_json(
        prompts.source_parsing_prompt,
        "Filter response to be parsed: Keep source 1 only",
        {"sources": [1]},
    )

    answer_prompt = (
        "Research Topic: Climate impacts\n\n"
        "Search Results:\n[1] ID: q1\nMetadata: {'text': 'Record warmth observed'}\nContent: Summary doc1"
    )
    answer_llm.register_stream(prompts.answer_prompt, answer_prompt, "Final answer content")

    query_results = {
        "climate effects 2024": [
            SearchResult(id="q1", metadata={"text": "Record warmth observed"})
        ],
        "economic impact climate change": [
            SearchResult(id="q2", metadata={"text": "GDP loss projections"})
        ],
    }
    fake_pipeline = FakeQueryPipeline(query_results)

    model_config = DeepResearchModelConfig(
        planning=planning_llm,
        json=json_llm,
        summary=summary_llm,
        answer=answer_llm,
    )

    pipeline = DeepResearchPipeline(
        fake_pipeline,  # type: ignore[arg-type]
        model_config,
        config=DeepResearchConfig(budget=1, max_queries=2, max_sources=2, max_tokens=1024),
        prompts=prompts,
        query_options=DeepResearchQueryOptions(),
    )

    result = await pipeline.run_research("Climate impacts")

    chunks: List[str] = []
    async for chunk in result.stream:
        chunks.append(chunk)

    assert "".join(chunks) == "Final answer content"
    assert result.queries_used == ["climate effects 2024", "economic impact climate change"]
    assert result.source_indices == [1]
    assert [view.id for view in result.results.results] == ["q1"]
    assert fake_pipeline.calls == ["climate effects 2024", "economic impact climate change"]


def test_create_deep_research_pipeline_with_query_pipeline():
    fake_pipeline = FakeQueryPipeline({})
    llm = FakeLLM()
    summary_override = FakeLLM()

    helper_result = create_deep_research_pipeline(
        llm=llm,
        query_pipeline=fake_pipeline,
        config=DeepResearchConfig(budget=3),
        summary_llm=summary_override,
    )

    assert isinstance(helper_result, DeepResearchPipeline)
    assert helper_result.model_config.summary is summary_override
    assert helper_result.model_config.answer is llm


def test_create_deep_research_pipeline_requires_embedder_and_store():
    with pytest.raises(ValueError):
        create_deep_research_pipeline(llm=FakeLLM())

    helper_result = create_deep_research_pipeline(
        llm=FakeLLM(),
        embedder=DummyEmbedder(),
        store=DummyVectorStore(),
    )
    assert isinstance(helper_result, DeepResearchPipeline)


def test_search_results_collection_dedup():
    results = SearchResultsCollection.from_sequence(
        [
            SearchResultView(id="a", metadata={}, content="one"),
            SearchResultView(id="b", metadata={}, content="two"),
            SearchResultView(id="a", metadata={}, content="one duplicate"),
        ],
    )
    deduped = results.dedup()

    assert len(deduped.results) == 2
    assert [view.id for view in deduped.results] == ["a", "b"]


@pytest.mark.asyncio
async def test_conduct_iterative_research_respects_budget():
    fake_pipeline = FakeQueryPipeline({})
    llm = FakeLLM()
    pipeline = DeepResearchPipeline(
        fake_pipeline,  # type: ignore[arg-type]
        DeepResearchModelConfig(planning=llm, json=llm, summary=llm, answer=llm),
        config=DeepResearchConfig(budget=1, max_queries=2),
    )

    initial_results = SearchResultsCollection.from_sequence(
        [SearchResultView(id="initial", metadata={}, content="base")],
    )
    all_queries = ["initial-query"]

    extra_queries = [["follow-up"], ["should-not-run"]]
    perform_calls: List[List[str]] = []

    async def fake_eval(*_args, **_kwargs) -> List[str]:
        return extra_queries.pop(0) if extra_queries else []

    async def fake_search(queries: Sequence[str]) -> SearchResultsCollection:
        perform_calls.append(list(queries))
        return SearchResultsCollection.from_sequence(
            [SearchResultView(id=f"extra-{len(perform_calls)}", metadata={}, content="new")],
        )

    pipeline._evaluate_research_completeness = fake_eval  # type: ignore[assignment]
    pipeline._perform_search = fake_search  # type: ignore[assignment]

    output = await pipeline._conduct_iterative_research("Budget Topic", initial_results, all_queries.copy())

    assert perform_calls == [["follow-up"]]
    assert output.queries_used == ["initial-query", "follow-up"]
    assert len(output.final_results.results) == 2


@pytest.mark.asyncio
async def test_filter_results_limits_and_validates_indices():
    prompts = DeepResearchPrompts(
        planning_prompt="plan",
        plan_parsing_prompt="plan-parse",
        raw_content_summariser_prompt="summarise",
        evaluation_prompt="evaluate",
        evaluation_parsing_prompt="evaluate-parse",
        filter_prompt="filter",
        source_parsing_prompt="filter-parse",
        answer_prompt="answer",
    )
    planning_llm = FakeLLM()
    json_llm = FakeLLM()
    pipeline = DeepResearchPipeline(
        FakeQueryPipeline({}),  # type: ignore[arg-type]
        DeepResearchModelConfig(planning=planning_llm, json=json_llm, summary=FakeLLM(), answer=FakeLLM()),
        config=DeepResearchConfig(max_sources=2),
        prompts=prompts,
    )

    results = SearchResultsCollection.from_sequence(
        [
            SearchResultView(id="one", metadata={"text": "1"}, content="Result 1"),
            SearchResultView(id="two", metadata={"text": "2"}, content="Result 2"),
            SearchResultView(id="three", metadata={"text": "3"}, content="Result 3"),
        ],
    )
    formatted = results.to_string()
    topic = "Filtering Topic"

    filter_request = (
        f"<Research Topic>{topic}</Research Topic>\n\n<Current Search Results>{formatted}</Current Search Results>"
    )
    filter_response = "Keep sources 3 then 1 then 99"

    planning_llm.register_text(prompts.filter_prompt, filter_request, filter_response)
    json_llm.register_json(
        prompts.source_parsing_prompt,
        f"Filter response to be parsed: {filter_response}",
        {"sources": [3, 1, 99]},
    )

    filtered = await pipeline._filter_results(topic, results)

    assert filtered.source_indices == [3, 1]
    assert [view.id for view in filtered.filtered_results.results] == ["three", "one"]
