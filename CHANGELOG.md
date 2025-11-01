# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.10] - 2025-11-01

### Changed
- Fixed argument error in `SupabaseKeywordStore`'s search method.
- Added configurable `language` option for Supabase full-text search.
- Use PostgreSQL web search syntax for full-text queries in Supabase keyword store.

## [0.1.9] - 2025-10-28

### Added
- Exposed context and per-stage append hooks in `default_prompts(...)` so downstream apps can tailor guidance without copying templates.
- Added `maktaba_templates.md` with examples for Kutub-style deployments.

### Changed
- README now links to the prompt customisation guide.

### Tests
- Added coverage verifying custom context/append hooks propagate through prompts.

## [0.1.8] - 2025-10-28

### Changed
- Update OpenAI JSON prompts to satisfy OpenAI's json requirement.
- Added opt-in OpenAI integration tests covering planning and full pipeline flows.

### Tests
- New prompt regression test ensures json is retained in default templates.
- Added streaming test exercising the full deep research pipeline with stubbed search results.

## [0.1.7] - 2025-10-27

### Added
- **Deep Research Pipeline** matching the full multi-stage research workflow (planning, iterative querying, summarisation, filtering, and streamed report generation).
- Dedicated deep research configuration, prompt templates, and result container models for programmatic integration.
- `create_deep_research_pipeline(...)` helper for one-call setup, plus accompanying guide (`docs/DeepResearch.md`) and runnable example script.

### Changed
- `BaseLLM` interface now exposes generic text/JSON completion utilities and a streaming hook; `OpenAILLM` implements the new methods for reuse across pipelines.

### Tests
- Introduced deep research pipeline unit coverage with fake LLM/query implementations (helper wiring, dedupe utilities, iteration budgets) and refreshed agentic pipeline mocks to honour the expanded interface.

## [0.1.6] - 2025-10-26

### Added
- Rich relationship modelling via new `RelationshipType` enum and `NodeRelationship` dataclass, enabling expressive NEXT/PREVIOUS links between chunks.
- Automatic relationship generation in the ingestion pipeline so sequential text chunks are linked without manual wiring.
- Advanced chunking controls (`overlap`, `max_characters`, `new_after_n_chars`) exposed through `UnstructuredChunker` for finer document splitting strategies.

### Changed
- Vector stores now persist and hydrate relationships consistently, including Pinecone, Qdrant, and Weaviate providers.
- `VectorChunk` and search result models carry relationship metadata to downstream consumers for navigation-aware retrieval.
- Test suite expanded to cover relationship handling and the new chunking configuration knobs.

## [0.1.5] - 2025-10-25

### Added
- **Agentic RAG Pipeline**: Iterative query generation and retrieval with LLM-based evaluation
  - `AgenticQueryPipeline` with support for multi-iteration search
  - Automatic query generation using OpenAI (or custom LLM providers)
  - Source evaluation to determine when sufficient information is retrieved
  - Parallel query execution for improved performance
- **Keyword Search Support**: Full-text search alongside semantic vector search
  - `BaseKeywordStore` abstract interface
  - `QdrantKeywordStore` implementation using Qdrant's full-text search
  - `SupabaseKeywordStore` implementation using PostgreSQL full-text search
  - Query routing by type: "keyword" vs "semantic"
- **LLM Abstractions**:
  - `BaseLLM` interface for query generation and source evaluation
  - `OpenAILLM` implementation with JSON mode and precise token tracking
  - `LLMUsage` dataclass for accurate token counting
- **Enhanced Configuration**:
  - `max_queries_per_iter` parameter for controlling query volume
  - `keyword_limit` parameter for keyword search results
  - `include_query_results` flag for debugging query-to-result mappings
  - Optional `supabase` dependency group

### Changed
- Ruff configuration now ignores E501 (line length) for prompt strings and N806 (variable naming) for test mocks

## [0.1.4] - 2025-10-25

### Fixed
- Renamed the reranker provider from "Zerank" (the model name) to "ZeroEntropy" (the company name) throughout the codebase and documentation for consistency.

### Added
- Added support for both list and single-value filters.

## [0.1.3] - 2025-10-11

### Fixed
- **Critical:** QdrantStore now correctly uses UUIDs for point IDs in all modes (in-memory, local, and server)
- Fixed error: "value book_XXX#chunk_X is not a valid point ID" when using Qdrant server mode
- Query results now return original string IDs (e.g., `book_123#chunk_0`) instead of internal UUIDs
- Migrated from deprecated `search()` to modern `query_points()` API
- **CI Build Fix:** Pinned `chromadb<1.1` to avoid dependency resolution failure with non-existent `mdurl==0.1.3`

### Added
- Comprehensive QdrantStore integration tests covering string ID handling, namespaces, and document deletion

## [0.1.2] - 2025-10-10

### Added
- ZeroEntropy Zerank reranker support via `ZerankReranker` class
- New optional dependency group: `zeroentropy`
- Async reranking with graceful fallback to heuristic scoring
- Comprehensive test coverage for Zerank reranker

## [0.1.0] - 2025-10-09

### Added
- Query pipeline with automatic reranking and citation formatting
- Ingestion pipeline for document processing
- Provider-agnostic embedding support (OpenAI, Azure, Cohere, Voyage)
- Vector store integrations (Qdrant, Pinecone, Chroma, Redis)
- Unstructured document chunking via LlamaIndex
- Cohere reranking support
- Async-first API design
- Full type hints and Pydantic validation
- Comprehensive test coverage
- Arabic and multilingual language support

### Documentation
- Overview, quickstart, and provider guides
- Example scripts for common use cases
- API reference documentation

[Unreleased]: https://github.com/nuhatech/maktaba/compare/v0.1.10...HEAD
[0.1.10]: https://github.com/nuhatech/maktaba/compare/v0.1.9...v0.1.10
[0.1.9]: https://github.com/nuhatech/maktaba/compare/v0.1.8...v0.1.9
[0.1.8]: https://github.com/nuhatech/maktaba/compare/v0.1.7...v0.1.8
[0.1.7]: https://github.com/nuhatech/maktaba/compare/v0.1.6...v0.1.7
[0.1.6]: https://github.com/nuhatech/maktaba/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/nuhatech/maktaba/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/nuhatech/maktaba/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/nuhatech/maktaba/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/nuhatech/maktaba/compare/v0.1.0...v0.1.2
[0.1.0]: https://github.com/nuhatech/maktaba/releases/tag/v0.1.0




