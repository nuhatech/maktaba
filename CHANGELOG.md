# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.18] - 2025-11-15

### Added
- **Keyword search support in `search_with_history`**: Added `keyword_queries` and `keyword_limit` parameters to `QueryPipeline.search_with_history()` method:
  - Enables parallel keyword search when using conversation history with query condensation
  - Parameters are passed through to the underlying `search()` method
  - Maintains consistency with `search()` method API

## [0.1.17] - 2025-11-14

### Fixed
- **UnstructuredChunker overlap functionality**: Fixed overlap not working correctly in `UnstructuredChunker`:
  - Added `overlap_all=True` parameter when `overlap` is specified to ensure overlap applies between ALL chunks, not just split chunks
  - Made `overlap_all` conditional: only enabled when overlap is reasonable (< 50% of max chunk size) to prevent duplicate chunks when chunks end up smaller than expected
  - This ensures proper overlap between consecutive chunks while avoiding near-duplicate chunks when overlap is too large relative to chunk size
  - Applied fix to all three methods: `chunk_text()`, `chunk_file()`, and `chunk_url()`

### Improved
- **Overlap detection in tests**: Enhanced overlap verification in chunking tests:
  - Improved overlap detection algorithm to handle whitespace normalization and text boundary differences
  - Added better diagnostic output showing full chunk text for small chunks
  - More accurate overlap detection that accounts for text normalization by Unstructured.io
  - Better error messages explaining why exact overlap might not be detected at boundaries

## [0.1.16] - 2025-11-14

### Added
- **Parallel keyword search in QueryPipeline**: Added optional parallel keyword search support to `QueryPipeline`:
  - New `keyword_store` parameter in `QueryPipeline.__init__()` for full-text search integration
  - New `keyword_queries` parameter in `QueryPipeline.search()` to specify keyword queries for parallel execution
  - New `keyword_limit` parameter to control number of results per keyword query (default: 15)
  - Semantic and keyword searches execute in parallel using `asyncio.gather()` for improved performance
  - Results are automatically deduplicated by chunk ID (semantic results take priority)
  - Combined results are reranked together when reranking is enabled
  - Backward compatible: existing code continues to work without keyword search
  - User-provided keyword queries (no LLM dependency required, unlike AgenticQueryPipeline)

### Tests
- Comprehensive test suite for parallel keyword search in QueryPipeline:
  - Basic parallel execution test
  - Deduplication and priority handling tests
  - Error handling and edge case tests
  - Multiple keyword queries parallel execution test
  - Namespace and filter parameter propagation test
  - Combined reranking test
  - Integration test with real Qdrant stores

## [0.1.15] - 2025-11-02

### Fixed
- **Supabase keyword search PostgREST query type**: Fixed incorrect parameter value in `SupabaseKeywordStore.search()` that caused failures with multi-word queries:
  - Changed `options={"type": "websearch"}` to `options={"type": "web_search"}` (with underscore)
  - PostgREST requires `"web_search"` to generate `wfts` (websearch_to_tsquery) for proper multi-word query handling
  - The incorrect `"websearch"` (without underscore) caused fallback to plain `fts` (to_tsquery), which fails on queries with spaces
  - This fix ensures multi-word queries are properly processed by PostgreSQL's `websearch_to_tsquery` function

## [0.1.14] - 2025-11-02

### Added
- **Lightweight unstructured chunking option**: Added `unstructured-minimal` extra dependency for users who only need text chunking via llama-index without PDF/OCR/document processing dependencies:
  - `maktaba[unstructured-minimal]` installs base `unstructured` package (no `[all-docs]` extra)
  - Suitable for text-only workflows that don't require PDF parsing, OCR, or image processing
  - Existing `maktaba[unstructured]` extra remains available for full document processing support

## [0.1.13] - 2025-11-02

### Changed
- **Provider-independent LLM prompts**: Refactored agentic LLM prompts to be provider-independent and user-customizable:
  - Created `maktaba.llm.prompts` module with `AgenticPrompts` dataclass and `default_prompts()` function
  - Moved hardcoded prompts from `OpenAILLM` class to separate prompts module
  - Added `prompts` parameter to `OpenAILLM` and `AgenticQueryPipeline` for customization
  - Users can now override prompts or add custom context, similar to deep research pipeline
  - Example: `default_prompts(context="Searching medical texts", generate_queries_append="Focus on evidence-based queries")`
  - This prepares the codebase for additional LLM providers (Bedrock, Anthropic, etc.)

## [0.1.12] - 2025-11-02

### Fixed
- **Supabase keyword search method chaining**: Fixed `AttributeError: 'SyncQueryRequestBuilder' object has no attribute 'limit'` by reordering the query builder chain to call `limit()` and filter methods before `text_search()`. The query execution order remains correct—PostgREST applies all filters first, then limits results.
- **PostgreSQL tsquery syntax error with quoted queries**: Fixed `syntax error in tsquery` (error code 42601) that occurred when LLM-generated queries contained surrounding quotes (e.g., `"tawaf ablution"`). The search method now strips surrounding quotes from queries before passing them to PostgreSQL's `websearch_to_tsquery`, preventing malformed tsquery syntax.

### Added
- **Integration tests for Supabase keyword search**: Added comprehensive integration tests for `SupabaseKeywordStore` that verify:
  - Basic keyword search functionality with real Supabase database
  - Quoted query handling (testing the syntax error fix)
  - Namespace and filter parameter support
  - Tests are marked with `@pytest.mark.integration` and can be run with environment variables (`SUPABASE_URL`, `SUPABASE_KEY`, `SUPABASE_TABLE_NAME`)

## [0.1.11] - 2025-11-02

### Fixed
- Supabase attribute error fixed. The method chain now calls `limit()` before `text_search()`, which resolves the AttributeError. The query will still execute correctly—PostgREST applies filters first, then limits results.


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

[Unreleased]: https://github.com/nuhatech/maktaba/compare/v0.1.18...HEAD
[0.1.18]: https://github.com/nuhatech/maktaba/compare/v0.1.17...v0.1.18
[0.1.17]: https://github.com/nuhatech/maktaba/compare/v0.1.16...v0.1.17
[0.1.16]: https://github.com/nuhatech/maktaba/compare/v0.1.15...v0.1.16
[0.1.15]: https://github.com/nuhatech/maktaba/compare/v0.1.14...v0.1.15
[0.1.14]: https://github.com/nuhatech/maktaba/compare/v0.1.13...v0.1.14
[0.1.13]: https://github.com/nuhatech/maktaba/compare/v0.1.12...v0.1.13
[0.1.12]: https://github.com/nuhatech/maktaba/compare/v0.1.11...v0.1.12
[0.1.11]: https://github.com/nuhatech/maktaba/compare/v0.1.10...v0.1.11
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




