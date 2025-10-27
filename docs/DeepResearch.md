# Deep Research Pipeline

Maktaba ships an opt-in "deep research" workflow that replicates the full multi-stage research process: it plans queries, performs iterative retrieval, summarises source text, filters the best evidence, and streams a long-form answer — all orchestrated with reusable library primitives.

## When to Use
- Questions that require multiple retrieval cycles, synthesis across many documents, or publication-style output.
- Scenarios where you want citations to be preserved alongside structured reporting.
- Teams seeking a turnkey, production-ready deep research workflow that can run entirely within their own infrastructure.

## Quick Start
```python
import asyncio

from maktaba.embedding import OpenAIEmbedder
from maktaba.llm import OpenAILLM
from maktaba.pipeline import create_deep_research_pipeline
from maktaba.storage import QdrantStore

async def main():
    pipeline = create_deep_research_pipeline(
        embedder=OpenAIEmbedder(api_key="..."),
        store=QdrantStore(url="http://localhost:6333", collection_name="docs"),
        llm=OpenAILLM(api_key="...", model="gpt-4o-mini"),
        config_overrides={"max_sources": 4},  # optional override
    )

    result = await pipeline.run_research("Impacts of lunar dust on spacecraft design")

    report = "".join([chunk async for chunk in result.stream])
    print(report)
    print(result.queries_used)   # All search queries run
    print(result.source_indices) # 1-based indices of retained sources

asyncio.run(main())
```

The helper wires the internal `QueryPipeline`, duplicates the default configuration, and assigns the same LLM to each stage unless you pass overrides like `summary_llm` or `answer_llm`.

## Configuration
`DeepResearchConfig` controls iteration, source selection, and output tokens:

| Field         | Default | Description |
|---------------|---------|-------------|
| `budget`      | `2`     | Number of refinement cycles after the initial search |
| `max_queries` | `2`     | Maximum queries per cycle (after planning / evaluation) |
| `max_sources` | `5`     | Sources forwarded to the final answer |
| `max_tokens`  | `8192`  | Streaming token budget for the answer stage |

Use `config_overrides` in the helper or pass a configured `DeepResearchConfig` directly to `DeepResearchPipeline`.

## Prompt Customisation
Prompts ship in `DeepResearchPrompts`. Override them in two ways:

```python
from maktaba.pipeline.deep_research import default_prompts

prompts = default_prompts()
prompts.answer_prompt = prompts.answer_prompt.replace("at least 5 pages", "concise one-page brief")

pipeline = create_deep_research_pipeline(
    ...,
    prompts=prompts,
)
```

Every prompt receives the current date prefix via `current_date_context()` so the model understands recency requirements.

## Streaming Output
`pipeline.run_research(topic)` returns a `DeepResearchAnswer`:

```python
class DeepResearchAnswer(TypedDict):
    stream: AsyncIterator[str]        # stream chunks of Markdown output
    usage: LLMUsage                   # cumulative token accounting
    queries_used: List[str]           # search queries, including follow-ups
    source_indices: List[int]         # 1-based indices that made the final cut
    results: SearchResultsCollection  # structured view of summarised sources
```

`SearchResultsCollection` can be formatted via `.to_string()` / `.short_string()` if you need to audit the summarised evidence.

## Advanced Overrides
- **Stage-specific LLMs:** `create_deep_research_pipeline(llm=..., summary_llm=..., answer_llm=...)`
- **Custom QueryPipeline:** supply `query_pipeline=QueryPipeline(...)` instead of `embedder`/`store`.
- **Vector search options:** pass `query_options=DeepResearchQueryOptions(top_k=20, rerank=False, namespace="team-123")`.

## Testing Utilities
The test suite provides light-weight doubles (`DummyEmbedder`, `DummyVectorStore`, `FakeLLM`) that you can reuse when writing your own unit tests. See `tests/test_deep_research_pipeline.py` for patterns covering deduplication and iteration budgets.

## Next Steps
- Pair the deep research pipeline with the ingestion pipeline to keep your knowledge base fresh.
- Use the returned `queries_used` to log search usage across teams or to feed analytics dashboards.
