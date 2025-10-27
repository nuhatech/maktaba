% Pipelines

QueryPipeline
- Embeds the query (input_type="query")
- Retrieves from store (camelCase: topK, includeMetadata, namespace)
- Optional rerank step (e.g., CohereReranker)
- Formats citations into `[n]:` blocks

IngestionPipeline
- Chunk (text/file/url) via a BaseChunker (UnstructuredChunker)
- Embed chunks in batches (input_type="document")
- Upsert as `{doc_id}#chunk_{i}` ids
- Optional on_progress callback

DeepResearchPipeline
- Iterative planning → retrieval → summarisation → filtering → streamed answer
- Ships with ready-to-use prompts and config tuned for comprehensive research
- Use `create_deep_research_pipeline(...)` for plug-and-play setup
- See [DeepResearch.md](./DeepResearch.md) for full workflow details
