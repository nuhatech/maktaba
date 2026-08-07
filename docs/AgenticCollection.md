# Agentic verified evidence collection

`AgenticCollectionPipeline` reuses Agentic Search v2 to collect a requested number of exact source spans. It is domain-neutral: applications describe the items and validate domain-specific attributes above this layer.

## Contract

```python
from maktaba.pipeline import (
    AgenticCollectionPipeline,
    AgenticQueryPipeline,
    AgenticSearchConfig,
    CollectionGoal,
)

search = AgenticQueryPipeline(
    embedder=embedder,
    store=store,
    keyword_store=keyword_store,
    reranker=reranker,
    llm=llm,
)
collector = AgenticCollectionPipeline(search, span_normalizer=my_optional_normalizer)

result = await collector.collect(
    [("user", "Collect five directly relevant passages")],
    goal=CollectionGoal(
        target_count=5,
        item_description="passages that directly support the requested topic",
        require_exact_span=True,
        min_characters=20,
        max_per_document=1,
        diversity_metadata_keys=("author_id",),
        max_per_diversity_group=1,
    ),
    includeRelationships=True,
    config=AgenticSearchConfig(
        max_iterations=4,
        max_total_queries=30,
        token_budget=8_000,
        diversity_metadata_keys=("author_id",),
        max_per_diversity_group=1,
    ),
)
```

`result.items` contains `EvidenceSpan` objects. Each span has the original source text, source/document IDs, original character offsets, source metadata, and untrusted extractor attributes. `result.complete` is true only when the verified target count is reached. Otherwise, valid partial items remain available with a machine-readable stop reason.

## Loop

For each Agentic Search iteration, the collection objective:

1. receives the globally fused and reranked evidence window;
2. asks the provider for exact-span candidates;
3. rejects unknown source IDs and text absent from the claimed source;
4. maps whitespace-only variations, and optional application-approved
   normalisation variants, back to the original source slice;
5. deduplicates and applies document/metadata diversity limits;
6. stops when the verified target count is reached;
7. otherwise asks for bounded search or PREVIOUS/NEXT expansion actions.

Search filters, namespaces, relationship expansion limits, query budgets, and token budgets are enforced by the same Agentic Search v2 code path used for question answering.

## Provider hooks

`OpenAILLM` implements structured `extract_collection_items()` and `plan_collection_actions()`. Other providers can override those optional `BaseLLM` methods. Existing providers remain compatible and continue to support question-answering search; the default collection extractor returns no candidates rather than producing unverified text.

Applications can also pass their own objective-specific `EvidenceAssessor` to `AgenticQueryPipeline.agentic_search()`.

## Security invariants

- Retrieved text is untrusted data, never instructions.
- The extractor cannot invent filters, namespaces, or source IDs.
- Candidate attributes are untrusted until the application validates them.
- The displayed exact span should always use `EvidenceSpan.text`, not generated candidate text.
- The built-in whitespace tolerance never rewrites letters. A custom
  `span_normalizer` should remove only domain-approved orthographic or
  formatting variants; accepted output still comes byte-for-byte from the
  original source slice.
- Partial verified results are preferable to filling a target with unsupported items.
- Private text should not be copied into logs or analytics traces.

## Stop reasons

Successful collections use `target_reached`. Incomplete collections retain the underlying Agentic Search stop reason, including `iteration_limit_reached`, `token_budget_exhausted`, `query_limit_reached`, `no_progress`, `no_results`, or `invalid_assessment`.
