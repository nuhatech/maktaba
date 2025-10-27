"""Deep research pipeline orchestrating planning, retrieval, and synthesis."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Iterable, List, Optional, Sequence, Tuple

from ...embedding.base import BaseEmbedder
from ...llm.base import BaseLLM
from ...logging import get_logger
from ...models import LLMUsage, SearchResult
from ...reranking.base import BaseReranker
from ...storage.base import BaseVectorStore
from ..query import QueryPipeline
from .config import DEFAULT_DEEP_RESEARCH_CONFIG, DeepResearchConfig
from .models import (
    DeepResearchAnswer,
    FilteredResultsData,
    IterativeResearchOutput,
    ResearchPlan,
    SearchResultsCollection,
    SearchResultView,
    SourceList,
)
from .prompts import DeepResearchPrompts, default_prompts

LOGGER = get_logger("maktaba.pipeline.deep_research")


@dataclass(slots=True)
class DeepResearchModelConfig:
    planning: BaseLLM
    json: BaseLLM
    summary: BaseLLM
    answer: BaseLLM


@dataclass(slots=True)
class DeepResearchQueryOptions:
    top_k: Optional[int] = None
    rerank: bool = True
    rerank_limit: Optional[int] = None
    min_score: Optional[float] = None
    namespace: Optional[str] = None
    filter: Optional[Dict[str, Any]] = None
    include_metadata: bool = True


class DeepResearchPipeline:
    """Asynchronous deep research workflow."""

    def __init__(
        self,
        query_pipeline: QueryPipeline,
        model_config: DeepResearchModelConfig,
        *,
        config: Optional[DeepResearchConfig] = None,
        prompts: Optional[DeepResearchPrompts] = None,
        query_options: Optional[DeepResearchQueryOptions] = None,
    ) -> None:
        self.query_pipeline = query_pipeline
        self.model_config = model_config
        self.config = (config or DEFAULT_DEEP_RESEARCH_CONFIG).copy()
        self.prompts = (prompts or default_prompts()).copy()
        self.query_options = query_options or DeepResearchQueryOptions()
        self._usage = LLMUsage()

    async def run_research(self, topic: str) -> DeepResearchAnswer:
        LOGGER.info("deep_research.start topic='%s'", topic)

        initial_queries = await self._generate_initial_queries(topic)
        initial_results = await self._perform_search(initial_queries)
        iterative_output = await self._conduct_iterative_research(
            topic,
            initial_results,
            list(initial_queries),
        )
        processed_results = await self._process_search_results(topic, iterative_output.final_results)
        filtered = await self._filter_results(topic, processed_results)
        stream, usage = await self._generate_research_answer(topic, filtered.filtered_results)
        self._usage = self._usage + usage

        LOGGER.info(
            "deep_research.done topic='%s' queries=%d sources=%d tokens=%d",
            topic,
            len(iterative_output.queries_used),
            len(filtered.source_indices),
            self._usage.total_tokens,
        )

        return DeepResearchAnswer(
            stream=stream,
            usage=self._usage,
            queries_used=iterative_output.queries_used,
            source_indices=filtered.source_indices,
            results=filtered.filtered_results,
        )

    async def _generate_initial_queries(self, topic: str) -> List[str]:
        queries = await self._generate_research_queries(topic)
        if self.config.max_queries and self.config.max_queries > 0:
            queries = queries[: self.config.max_queries]
        if not queries:
            LOGGER.warning("deep_research.plan empty topic='%s'", topic)
        else:
            LOGGER.debug("deep_research.plan queries=%s", queries)
        return queries

    async def _generate_research_queries(self, topic: str) -> List[str]:
        payload, usage = await self.model_config.json.complete_json(
            system=self.prompts.planning_prompt,
            prompt=f"Research Topic: {topic}",
        )
        self._usage = self._usage + usage
        plan = ResearchPlan.from_iterable(payload.get("queries", []))
        return [q for q in plan.queries if q]

    async def _perform_search(self, queries: Sequence[str]) -> SearchResultsCollection:
        tasks = [self._web_search(query) for query in queries]
        results = await asyncio.gather(*tasks)
        combined = SearchResultsCollection.empty()
        for result in results:
            combined = combined.add(result)
        return combined.dedup()

    async def _web_search(self, query: str) -> SearchResultsCollection:
        truncated_query = query[:400]
        if len(query) > 400:
            LOGGER.debug("deep_research.query truncated original_len=%d", len(query))

        LOGGER.info("deep_research.search query='%s'", truncated_query)
        search_kwargs = self._build_query_kwargs()
        data = await self.query_pipeline.search(
            truncated_query,
            rerank=search_kwargs["rerank"],
            top_k=search_kwargs["top_k"],
            rerank_limit=search_kwargs["rerank_limit"],
            min_score=search_kwargs["min_score"],
            namespace=search_kwargs["namespace"],
            filter=search_kwargs["filter"],
            includeMetadata=search_kwargs["include_metadata"],
        )
        raw_results: Iterable[SearchResult] = data.get("results", [])  # type: ignore[assignment]

        views: List[SearchResultView] = []
        summarisation_tasks: List[asyncio.Future[Tuple[Optional[SearchResultView], LLMUsage]]] = []
        loop = asyncio.get_running_loop()
        for result in raw_results:
            text = result.text
            if not text:
                continue
            summarisation_tasks.append(
                loop.create_task(self._summarise_result(result, truncated_query, text))
            )

        for task in summarisation_tasks:
            view, usage = await task
            if view is not None:
                views.append(view)
            self._usage = self._usage + usage

        LOGGER.info("deep_research.search results=%d", len(views))
        return SearchResultsCollection.from_sequence(views)

    async def _summarise_result(
        self,
        result: SearchResult,
        query: str,
        content: str,
    ) -> Tuple[Optional[SearchResultView], LLMUsage]:
        prompt = f"<Raw Content>{content}</Raw Content>\n\n<Research Topic>{query}</Research Topic>"
        summary, usage = await self.model_config.summary.complete_text(
            system=self.prompts.raw_content_summariser_prompt,
            prompt=prompt,
        )
        if not summary:
            return None, usage
        return (
            SearchResultView(
                id=result.id,
                metadata=result.metadata or {},
                content=summary,
            ),
            usage,
        )

    async def _process_search_results(
        self,
        topic: str,
        results: SearchResultsCollection,
    ) -> SearchResultsCollection:
        deduped = results.dedup()
        LOGGER.debug(
            "deep_research.process topic='%s' before=%d after=%d",
            topic,
            len(results.results),
            len(deduped.results),
        )
        return deduped

    async def _evaluate_research_completeness(
        self,
        topic: str,
        results: SearchResultsCollection,
        queries: Sequence[str],
    ) -> List[str]:
        formatted_results = results.to_string()
        evaluation_text, usage_text = await self.model_config.planning.complete_text(
            system=self.prompts.evaluation_prompt,
            prompt=(
                f"<Research Topic>{topic}</Research Topic>\n\n"
                f"<Search Queries Used>{list(queries)}</Search Queries Used>\n\n"
                f"<Current Search Results>{formatted_results}</Current Search Results>"
            ),
        )
        self._usage = self._usage + usage_text

        parsed, usage_json = await self.model_config.json.complete_json(
            system=self.prompts.evaluation_parsing_prompt,
            prompt=f"Evaluation to be parsed: {evaluation_text}",
        )
        self._usage = self._usage + usage_json
        plan = ResearchPlan.from_iterable(parsed.get("queries", []))
        return [q for q in plan.queries if q]

    async def _filter_results(
        self,
        topic: str,
        results: SearchResultsCollection,
    ) -> FilteredResultsData:
        formatted_results = results.to_string()
        filter_text, usage_filter = await self.model_config.planning.complete_text(
            system=self.prompts.filter_prompt,
            prompt=(
                f"<Research Topic>{topic}</Research Topic>\n\n"
                f"<Current Search Results>{formatted_results}</Current Search Results>"
            ),
        )
        self._usage = self._usage + usage_filter

        parsed, usage_parsed = await self.model_config.json.complete_json(
            system=self.prompts.source_parsing_prompt,
            prompt=f"Filter response to be parsed: {filter_text}",
        )
        self._usage = self._usage + usage_parsed

        source_list = SourceList.from_iterable(parsed.get("sources", []))
        sources = source_list.sources
        if self.config.max_sources and self.config.max_sources > 0:
            sources = sources[: self.config.max_sources]

        filtered_views: List[SearchResultView] = []
        for index in sources:
            if 1 <= index <= len(results.results):
                filtered_views.append(results.results[index - 1])

        LOGGER.info("deep_research.filter kept=%d", len(filtered_views))
        return FilteredResultsData(
            filtered_results=SearchResultsCollection.from_sequence(filtered_views),
            source_indices=sources,
        )

    async def _conduct_iterative_research(
        self,
        topic: str,
        results: SearchResultsCollection,
        all_queries: List[str],
    ) -> IterativeResearchOutput:
        current_results = results

        for iteration in range(self.config.budget):
            additional_queries = await self._evaluate_research_completeness(
                topic,
                current_results,
                all_queries,
            )

            if not additional_queries:
                LOGGER.info("deep_research.iteration_%d no additional queries", iteration)
                break

            queries_to_use = additional_queries
            if self.config.max_queries and self.config.max_queries > 0:
                queries_to_use = additional_queries[: self.config.max_queries]

            LOGGER.info("deep_research.iteration_%d queries=%s", iteration, queries_to_use)

            new_results = await self._perform_search(queries_to_use)
            current_results = current_results.add(new_results)
            all_queries.extend(queries_to_use)

        return IterativeResearchOutput(
            final_results=current_results,
            queries_used=all_queries,
        )

    async def _generate_research_answer(
        self,
        topic: str,
        results: SearchResultsCollection,
    ) -> Tuple[AsyncIterator[str], LLMUsage]:
        formatted_results = results.to_string()
        prompt = f"Research Topic: {topic}\n\nSearch Results:\n{formatted_results}"
        stream, usage = await self.model_config.answer.stream_text(
            system=self.prompts.answer_prompt,
            prompt=prompt,
            max_tokens=self.config.max_tokens,
        )
        return stream, usage

    def _build_query_kwargs(self) -> Dict[str, Any]:
        return {
            "top_k": self.query_options.top_k,
            "rerank": self.query_options.rerank,
            "rerank_limit": self.query_options.rerank_limit,
            "min_score": self.query_options.min_score,
            "namespace": self.query_options.namespace,
            "filter": self.query_options.filter,
            "include_metadata": self.query_options.include_metadata,
        }


def create_deep_research_pipeline(
    *,
    llm: BaseLLM,
    embedder: BaseEmbedder | None = None,
    store: BaseVectorStore | None = None,
    reranker: BaseReranker | None = None,
    query_pipeline: QueryPipeline | None = None,
    config: DeepResearchConfig | None = None,
    prompts: DeepResearchPrompts | None = None,
    query_options: DeepResearchQueryOptions | None = None,
    planning_llm: BaseLLM | None = None,
    json_llm: BaseLLM | None = None,
    summary_llm: BaseLLM | None = None,
    answer_llm: BaseLLM | None = None,
) -> DeepResearchPipeline:
    """
    Convenience helper that wires common defaults for the deep research pipeline.

    Parameters
    ----------
    llm:
        Base LLM implementation used for planning/JSON/summarisation/answer stages unless
        stage-specific overrides are provided.
    embedder, store, reranker:
        Components used to build a `QueryPipeline` if one is not supplied.
    query_pipeline:
        Pre-built query pipeline. If provided, `embedder`/`store`/`reranker` must be omitted.
    config, prompts, query_options:
        Optional overrides for research configuration, prompt templates, and vector search behaviour.
    planning_llm, json_llm, summary_llm, answer_llm:
        Optional stage-specific model overrides.
    """

    if query_pipeline is None:
        if embedder is None or store is None:
            raise ValueError("embedder and store must be provided when query_pipeline is not supplied")
        query_pipeline = QueryPipeline(
            embedder=embedder,
            store=store,
            reranker=reranker,
        )
    else:
        if any(component is not None for component in (embedder, store, reranker)):
            raise ValueError("embedder/store/reranker should not be provided when query_pipeline is supplied")

    model_config = DeepResearchModelConfig(
        planning=planning_llm or llm,
        json=json_llm or llm,
        summary=summary_llm or llm,
        answer=answer_llm or llm,
    )

    return DeepResearchPipeline(
        query_pipeline=query_pipeline,
        model_config=model_config,
        config=config,
        prompts=prompts,
        query_options=query_options,
    )
