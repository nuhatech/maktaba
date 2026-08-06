"""Agentic Search v2 wiring example.

Fill in provider credentials and component construction for your application.
"""

from maktaba.pipeline import AgenticQueryPipeline, AgenticSearchConfig


async def search_with_evidence_controls(
    pipeline: AgenticQueryPipeline,
    question: str,
) -> dict[str, object]:
    result = await pipeline.agentic_search(
        [("user", question)],
        includeRelationships=True,
        config=AgenticSearchConfig(
            max_iterations=4,
            max_total_queries=20,
            token_budget=8_000,
            evidence_limit=15,
        ),
    )
    if not result["answerable"]:
        return {
            "status": "insufficient_evidence",
            "stop_reason": result["stop_reason"],
            "assessment": result["assessment"],
        }
    return {
        "status": "ready",
        "context": result["formatted_context"],
        "citations": result["citations"],
        "provenance": result["provenance"],
    }
