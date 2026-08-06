# Agentic Search v2

Agentic Search v2 turns Maktaba's original generate-search-check loop into a bounded evidence-seeking architecture. Maktaba remains domain-neutral: applications decide when to use agentic mode and can add domain-specific retrieval tools above this layer.

## What changed

Each iteration now follows this flow:

1. Execute validated semantic or keyword search actions.
2. Fuse all ranked lists with reciprocal-rank fusion (RRF).
3. Rerank the fused candidate pool once against the standalone user question.
4. Select a diverse, bounded evidence set.
5. Assess coverage, missing information, contradictions, supporting source IDs, and next actions.
6. Stop with sufficient evidence or an explicit non-answerable reason.

The assessor cannot create namespaces or metadata filters. Every search and relationship expansion reuses the filter and namespace supplied by the caller.

## Basic usage

```python
from maktaba.pipeline import AgenticQueryPipeline, AgenticSearchConfig

pipeline = AgenticQueryPipeline(
    embedder=embedder,
    store=store,
    keyword_store=keyword_store,
    reranker=reranker,
    llm=llm,
)

result = await pipeline.agentic_search(
    [("user", "What conclusions are supported, and where do the sources disagree?")],
    namespace="tenant-a",
    filter={"collection": "approved"},
    includeRelationships=True,
    config=AgenticSearchConfig(
        max_iterations=4,
        max_queries_per_iter=6,
        max_total_queries=20,
        token_budget=8_000,
        candidate_limit=150,
        assessment_limit=25,
        evidence_limit=15,
        max_expansion_hops=1,
        max_expanded_chunks=8,
    ),
)
```

Always gate answer generation on `result["answerable"]`. Retrieved context may still be useful when this is false, but Maktaba is explicitly reporting that the evidence audit did not establish sufficient support.

```python
if not result["answerable"]:
    return {
        "status": "insufficient_evidence",
        "reason": result["stop_reason"],
        "gaps": (result["assessment"] or {}).get("missing_information", []),
    }

answer = await answer_model(result["formatted_context"])
```

## New result fields

The v1 keys (`formatted_context`, `citations`, `results`, `queries_used`, `iterations`, `total_chunks`, and `usage`) remain available. V2 adds:

- `answerable`: whether the evidence assessor authorises evidence-backed synthesis.
- `stop_reason`: `sufficient_evidence`, `insufficient_evidence`, `token_budget_exhausted`, `iteration_limit_reached`, `query_limit_reached`, `no_progress`, `no_results`, or `invalid_assessment`.
- `assessment`: structured coverage, gaps, contradictions, support IDs, confidence, and next actions.
- `candidate_count` and `evidence_count`: pre- and post-selection sizes.
- `provenance`: query, retrieval type, iteration, rank, original score, RRF score, and expansion parent for each chunk.
- `iteration_trace`: bounded action/retrieval/assessment trace for observability and evaluation.
- `expanded_chunk_ids`: chunks added through relationship traversal.

## Structured LLM compatibility

Existing `BaseLLM` subclasses remain valid. The default `assess_evidence()` adapter calls the existing `evaluate_sources()` boolean method. Such implementations gain fusion, budgets, provenance, and abstention, but use legacy query generation for subsequent iterations.

For gap-directed planning, override `assess_evidence()` and return an `EvidenceAssessment` containing search or expand `RetrievalAction` objects. `OpenAILLM` implements this structured contract out of the box and validates booleans and source IDs fail-closed.

```python
from maktaba.pipeline.agentic_models import EvidenceAssessment, RetrievalAction

assessment = EvidenceAssessment(
    answerable=False,
    missing_information=["The date range is not covered"],
    next_actions=[
        RetrievalAction(
            type="search",
            query="evidence for the missing date range",
            query_type="semantic",
        )
    ],
)
```

## Relationship expansion

Set `includeRelationships=True` to enable assessor-proposed traversal. Maktaba follows only requested relationship types, defaults to NEXT/PREVIOUS, detects cycles, and obeys hop/chunk limits. Qdrant, Pinecone, Chroma, and Weaviate implement scoped `fetch_by_ids()` support.

If a provider cannot safely apply a caller filter to direct ID retrieval, expansion returns no chunks. It never silently broadens the search scope. Existing custom stores can omit `fetch_by_ids()`; the default implementation returns an empty list.

## Post-generation verification

Retrieval sufficiency and answer faithfulness are separate checks. After an application generates an answer, use the generic verifier:

```python
from maktaba.citation import verify_answer

report = await verify_answer(
    answer,
    result["results"],
    require_citations=True,
    verify_quotes=True,
    entailment_checker=my_optional_async_nli_check,
)

if not report.valid:
    # Revise, abstain, or surface the report to a review workflow.
    print(report.to_dict())
```

The deterministic layer checks citation indices, uncited claims, and exact quotes. The optional entailment hook lets consumers plug in any LLM or local NLI model without coupling Maktaba to a provider.

## Deep Research evidence preservation

`SearchResultView.content` remains the planning summary. `raw_content` now preserves the original retrieved text, and final Deep Research synthesis uses this raw evidence. This avoids treating lossy summaries as primary citation material.
