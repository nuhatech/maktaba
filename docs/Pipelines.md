# Pipelines

## QueryPipeline

- Embeds the query (input_type="query")
- Retrieves from store (camelCase: topK, includeMetadata, namespace)
- Optional rerank step (e.g., CohereReranker)
- Formats citations into `[n]:` blocks

## IngestionPipeline

- Chunk (text/file/url) via a BaseChunker (UnstructuredChunker)
- Embed chunks in batches (input_type="document")
- Upsert as `{doc_id}#chunk_{i}` ids
- Optional on_progress callback

## AgenticQueryPipeline

- Runs a bounded search/action loop chosen explicitly by the application
- Fuses semantic and keyword ranked lists with reciprocal-rank fusion
- Applies one global rerank against the standalone user question
- Audits evidence coverage, gaps, contradictions, and supporting source IDs
- Can traverse bounded NEXT/PREVIOUS relationships without changing caller filters
- Returns provenance, traces, stop reasons, and an explicit `answerable` signal
- See [AgenticV2.md](./AgenticV2.md) for configuration and migration guidance

## DeepResearchPipeline

- Iterative planning → retrieval → summarisation → filtering → streamed answer
- Ships with ready-to-use prompts and config tuned for comprehensive research
- Uses summaries for planning but preserves raw retrieved text for final synthesis
- Use `create_deep_research_pipeline(...)` for plug-and-play setup
- See [DeepResearch.md](./DeepResearch.md) for full workflow details
