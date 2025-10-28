import os
import sys
from pathlib import Path
from typing import List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from maktaba.llm.openai import OpenAILLM
from maktaba.models import SearchResult
from maktaba.pipeline.deep_research import (
    DeepResearchConfig,
    DeepResearchModelConfig,
    DeepResearchPipeline,
)

OPENAI_API_KEY_ENV = "OPENAI_API_KEY"


class StaticQueryPipeline:
    def __init__(self, results: List[SearchResult]) -> None:
        self._results = results
        self.calls: List[str] = []

    async def search(self, query: str, **kwargs):  # pragma: no cover - integration helper
        self.calls.append(query)
        return {"results": self._results}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_deep_research_pipeline_full_run_with_openai():
    api_key = os.getenv(OPENAI_API_KEY_ENV)
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")

    llm = OpenAILLM(api_key=api_key, model="gpt-4o-mini", temperature=0.2)

    sample_results = [
        SearchResult(
            id="water-guide#1",
            score=0.92,
            metadata={
                "text": (
                    "Comprehensive guide describing practical community water conservation measures, "
                    "including rainwater harvesting systems, xeriscaping public areas, fixing municipal leaks, "
                    "tiered pricing to reduce wastage, and education campaigns that promote mindful household usage."
                ),
                "title": "Community Water Conservation Playbook",
                "author": "Hydrology Cooperative",
                "url": "https://example.org/water-conservation-playbook",
            },
        )
    ]

    query_pipeline = StaticQueryPipeline(sample_results)

    pipeline = DeepResearchPipeline(
        query_pipeline,  # type: ignore[arg-type]
        DeepResearchModelConfig(planning=llm, json=llm, summary=llm, answer=llm),
        config=DeepResearchConfig(budget=0, max_queries=1, max_sources=1, max_tokens=800),
    )

    topic = "How can communities conserve water during drought conditions?"
    answer = await pipeline.run_research(topic)

    tokens = []
    async for chunk in answer.stream:
        tokens.append(chunk)
    full_answer = "".join(tokens).strip()

    print("Deep research queries:", answer.queries_used)
    print("Deep research source indices:", answer.source_indices)
    print("Deep research answer preview:", full_answer[:400])

    assert answer.queries_used
    assert answer.source_indices
    assert len(answer.results.results) == len(answer.source_indices)
    assert full_answer
    assert "[1]" in full_answer, "Expected the answer to reference the retrieved source"
