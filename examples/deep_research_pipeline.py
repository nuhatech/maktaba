"""
Example: Deep Research Pipeline

Run with:
    uv run python examples/deep_research_pipeline.py
"""

import asyncio

from maktaba.embedding import OpenAIEmbedder
from maktaba.llm import OpenAILLM
from maktaba.pipeline import create_deep_research_pipeline
from maktaba.pipeline.deep_research import DeepResearchConfig
from maktaba.storage import QdrantStore


async def main() -> None:
    pipeline = create_deep_research_pipeline(
        embedder=OpenAIEmbedder(api_key="YOUR_OPENAI_KEY"),
        store=QdrantStore(url="http://localhost:6333", collection_name="docs"),
        llm=OpenAILLM(api_key="YOUR_OPENAI_KEY", model="gpt-4o-mini"),
        config=DeepResearchConfig(max_sources=4),
    )

    answer = await pipeline.run_research("Impacts of lunar dust on spacecraft design")

    report = "".join([chunk async for chunk in answer.stream])
    print("### Report ###")
    print(report)

    print("\n### Queries used ###")
    for q in answer.queries_used:
        print("-", q)

    print("\n### Source indices ###")
    print(answer.source_indices)


if __name__ == "__main__":
    asyncio.run(main())
