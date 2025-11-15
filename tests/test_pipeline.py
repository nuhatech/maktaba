import os
from typing import Any, Dict, List, Optional

import pytest

from maktaba.embedding.base import BaseEmbedder
from maktaba.keyword.base import BaseKeywordStore
from maktaba.models import SearchResult
from maktaba.pipeline.query import QueryPipeline
from maktaba.reranking.base import BaseReranker
from maktaba.reranking.cohere import CohereReranker
from maktaba.storage.base import BaseVectorStore


class DummyEmbedder(BaseEmbedder):
    async def embed_batch(self, texts: List[str], input_type: str = "document") -> List[List[float]]:
        return [[0.0, 0.0, 0.0] for _ in texts]

    @property
    def dimension(self) -> int:
        return 3

    @property
    def model(self) -> str:
        return "dummy"


class DummyStore(BaseVectorStore):
    def __init__(self, results: Optional[List[SearchResult]] = None):
        """Initialize with optional custom results."""
        self.results = results

    async def upsert(self, chunks, namespace=None) -> None:
        return None

    async def query(self, vector, topK: int = 10, filter=None, includeMetadata: bool = True, namespace=None):
        if self.results is not None:
            return self.results[:topK]
        # Return predictable 10 results with text present
        out = []
        for i in range(topK):
            meta = {"text": f"This is chunk {i} about Tawhid."}
            out.append(SearchResult(id=f"docA#{i}", score=1.0 - i * 0.01, metadata=meta))
        return out

    async def delete(self, ids, namespace=None):
        return None

    async def list(self, prefix=None, limit: int = 100, namespace=None):
        return []

    async def get_dimensions(self) -> int:
        return 3


class DummyKeywordStore(BaseKeywordStore):
    """Mock keyword store for testing."""

    def __init__(
        self,
        results_map: Optional[Dict[str, List[SearchResult]]] = None,
        should_raise: bool = False,
        received_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize DummyKeywordStore.

        Args:
            results_map: Dict mapping query strings to result lists
            should_raise: If True, raise exception on search
            received_params: Dict to store received search parameters (will be modified in-place)
        """
        self.results_map = results_map or {}
        self.should_raise = should_raise
        # Use the provided dict directly to maintain reference
        if received_params is not None:
            self.received_params = received_params
        else:
            self.received_params = {}

    async def search(self, query: str, limit: int = 15, filter=None, namespace=None):
        # Store received parameters for testing
        self.received_params[query] = {
            "limit": limit,
            "filter": filter,
            "namespace": namespace,
        }

        if self.should_raise:
            raise RuntimeError(f"Keyword search failed for '{query}'")

        # Return custom results if provided
        if query in self.results_map:
            return self.results_map[query][:limit]

        # Default: return predictable keyword results
        return [
            SearchResult(
                id=f"keyword#{i}",
                score=None,
                metadata={"text": f"Keyword result {i} for '{query}'"},
            )
            for i in range(min(limit, 3))
        ]

    async def upsert(self, chunks, namespace=None):
        # No-op for tests to satisfy abstract interface
        return None


class DummyReranker(BaseReranker):
    """Mock reranker for testing."""

    def __init__(self, received_inputs: Optional[List[Any]] = None):
        """
        Initialize DummyReranker.

        Args:
            received_inputs: List to store received rerank inputs
        """
        self.received_inputs = received_inputs or []

    async def rerank(self, query: str, results: List[SearchResult], top_k: int = 10):
        # Store input for testing
        self.received_inputs.append({"query": query, "results": results, "top_k": top_k})
        # Just return results as-is (no actual reranking)
        return results[:top_k]


@pytest.mark.asyncio
async def test_query_pipeline_success_metric():
    embedder = DummyEmbedder()
    store = DummyStore()
    reranker = CohereReranker(use_api=False)  # Offline heuristic
    pipeline = QueryPipeline(embedder, store, reranker)

    result = await pipeline.search("What is Tawhid?", rerank=True)

    assert isinstance(result["formatted_context"], str)
    assert "[1]:" in result["formatted_context"]
    assert len(result["citations"]) == 10


# =============================================================================
# Keyword Search Tests
# =============================================================================


@pytest.mark.asyncio
async def test_query_pipeline_parallel_keyword_search():
    """Test basic parallel execution of semantic + keyword searches."""
    # Limit semantic results to ensure keyword results are included
    limited_semantic_results = [
        SearchResult(id=f"docA#{i}", score=1.0 - i * 0.01, metadata={"text": f"Chunk {i}"})
        for i in range(5)  # Only 5 semantic results
    ]

    embedder = DummyEmbedder()
    store = DummyStore(results=limited_semantic_results)
    keyword_store = DummyKeywordStore()
    reranker = DummyReranker()
    pipeline = QueryPipeline(embedder, store, reranker, keyword_store=keyword_store)

    # Use top_k that will include both semantic and keyword results
    result = await pipeline.search(
        "What is Tawhid?",
        keyword_queries=["tawhid", "monotheism"],
        top_k=10,  # 5 semantic + 6 keyword = 11 total, so we'll get both types
        rerank=True,
    )

    # Verify structure
    assert isinstance(result["formatted_context"], str)
    assert "results" in result
    assert "citations" in result

    # Should have results from both semantic and keyword searches
    results = result["results"]
    assert len(results) > 0

    # Check that we have both semantic and keyword results
    semantic_ids = {r.id for r in results if r.id.startswith("docA#")}
    keyword_ids = {r.id for r in results if r.id.startswith("keyword#")}
    assert len(semantic_ids) > 0, "Should have semantic results"
    assert len(keyword_ids) > 0, "Should have keyword results"


@pytest.mark.asyncio
async def test_query_pipeline_keyword_deduplication():
    """Test that duplicate chunk IDs are deduplicated (semantic takes priority)."""
    # Create results with overlapping IDs
    semantic_result = SearchResult(
        id="docA#0",
        score=0.95,
        metadata={"text": "Semantic result", "source": "semantic"},
    )
    keyword_result = SearchResult(
        id="docA#0",
        score=0.80,
        metadata={"text": "Keyword result", "source": "keyword"},
    )

    embedder = DummyEmbedder()
    store = DummyStore(results=[semantic_result])
    keyword_store = DummyKeywordStore(results_map={"tawhid": [keyword_result]})
    reranker = DummyReranker()
    pipeline = QueryPipeline(embedder, store, reranker, keyword_store=keyword_store)

    result = await pipeline.search(
        "What is Tawhid?",
        keyword_queries=["tawhid"],
        rerank=False,
    )

    # Should only have one result with docA#0
    results = result["results"]
    docA_results = [r for r in results if r.id == "docA#0"]
    assert len(docA_results) == 1, "Should deduplicate to single result"

    # Semantic result should take priority
    final_result = docA_results[0]
    assert final_result.score == 0.95, "Should preserve semantic score"
    assert final_result.metadata["source"] == "semantic", "Should preserve semantic metadata"


@pytest.mark.asyncio
async def test_query_pipeline_no_keyword_store():
    """Test graceful handling when keyword_store is None."""
    embedder = DummyEmbedder()
    store = DummyStore()
    reranker = DummyReranker()
    pipeline = QueryPipeline(embedder, store, reranker, keyword_store=None)

    # Should not raise error even if keyword_queries provided
    result = await pipeline.search(
        "What is Tawhid?",
        keyword_queries=["tawhid", "monotheism"],
        rerank=True,
    )

    # Should only return semantic results
    assert isinstance(result["formatted_context"], str)
    assert "results" in result
    results = result["results"]
    assert len(results) > 0
    # All results should be semantic (docA#)
    assert all(r.id.startswith("docA#") for r in results), "Should only have semantic results"


@pytest.mark.asyncio
async def test_query_pipeline_empty_keyword_queries():
    """Test that empty keyword_queries list works normally."""
    embedder = DummyEmbedder()
    store = DummyStore()
    keyword_store = DummyKeywordStore()
    reranker = DummyReranker()
    pipeline = QueryPipeline(embedder, store, reranker, keyword_store=keyword_store)

    # Test with empty list
    result1 = await pipeline.search("What is Tawhid?", keyword_queries=[], rerank=True)
    assert len(result1["results"]) > 0

    # Test with None
    result2 = await pipeline.search("What is Tawhid?", keyword_queries=None, rerank=True)
    assert len(result2["results"]) > 0

    # Both should only have semantic results
    assert all(r.id.startswith("docA#") for r in result1["results"])
    assert all(r.id.startswith("docA#") for r in result2["results"])


@pytest.mark.asyncio
async def test_query_pipeline_keyword_search_error_handling():
    """Test that keyword search failures don't break the pipeline."""
    embedder = DummyEmbedder()
    store = DummyStore()
    keyword_store = DummyKeywordStore(should_raise=True)
    reranker = DummyReranker()
    pipeline = QueryPipeline(embedder, store, reranker, keyword_store=keyword_store)

    # Should not raise exception
    result = await pipeline.search(
        "What is Tawhid?",
        keyword_queries=["tawhid"],
        rerank=True,
    )

    # Should still return semantic results
    assert isinstance(result["formatted_context"], str)
    assert "results" in result
    results = result["results"]
    assert len(results) > 0
    # Should only have semantic results (keyword search failed)
    assert all(r.id.startswith("docA#") for r in results)


@pytest.mark.asyncio
async def test_query_pipeline_multiple_keyword_queries():
    """Test that multiple keyword queries execute in parallel."""
    # Limit semantic results to ensure keyword results are included
    limited_semantic_results = [
        SearchResult(id=f"docA#{i}", score=1.0 - i * 0.01, metadata={"text": f"Chunk {i}"})
        for i in range(5)  # Only 5 semantic results
    ]

    # Create distinct results for each keyword query
    keyword_store = DummyKeywordStore(
        results_map={
            "tawhid": [
                SearchResult(id="kw1#0", score=None, metadata={"text": "Tawhid result 1"}),
                SearchResult(id="kw1#1", score=None, metadata={"text": "Tawhid result 2"}),
            ],
            "monotheism": [
                SearchResult(id="kw2#0", score=None, metadata={"text": "Monotheism result 1"}),
            ],
            "oneness": [
                SearchResult(id="kw3#0", score=None, metadata={"text": "Oneness result 1"}),
                SearchResult(id="kw3#1", score=None, metadata={"text": "Oneness result 2"}),
            ],
        }
    )

    embedder = DummyEmbedder()
    store = DummyStore(results=limited_semantic_results)
    reranker = DummyReranker()
    pipeline = QueryPipeline(embedder, store, reranker, keyword_store=keyword_store)

    # Use top_k that will include both semantic and keyword results
    # 5 semantic + 5 keyword = 10 total, so we'll get both types
    result = await pipeline.search(
        "What is Tawhid?",
        keyword_queries=["tawhid", "monotheism", "oneness"],
        keyword_limit=10,
        top_k=10,  # 5 semantic + 5 keyword results
        rerank=False,
    )

    # Should have results from all keyword queries
    results = result["results"]
    result_ids = {r.id for r in results}

    # Check results from each keyword query are present
    assert "kw1#0" in result_ids or "kw1#1" in result_ids, "Should have tawhid results"
    assert "kw2#0" in result_ids, "Should have monotheism results"
    assert "kw3#0" in result_ids or "kw3#1" in result_ids, "Should have oneness results"


@pytest.mark.asyncio
async def test_query_pipeline_keyword_namespace_filter():
    """Test that namespace and filter parameters are passed to keyword store."""
    received_params = {}
    keyword_store = DummyKeywordStore(received_params=received_params)

    embedder = DummyEmbedder()
    store = DummyStore()
    reranker = DummyReranker()
    pipeline = QueryPipeline(embedder, store, reranker, keyword_store=keyword_store)

    test_filter = {"book_id": 123, "category": "theology"}
    test_namespace = "test_namespace"

    await pipeline.search(
        "What is Tawhid?",
        keyword_queries=["tawhid"],
        filter=test_filter,
        namespace=test_namespace,
    )

    # Verify parameters were passed to keyword store
    assert "tawhid" in received_params
    params = received_params["tawhid"]
    assert params["filter"] == test_filter, "Filter should be passed to keyword store"
    assert params["namespace"] == test_namespace, "Namespace should be passed to keyword store"


@pytest.mark.asyncio
async def test_query_pipeline_keyword_reranking():
    """Test that reranker is applied to merged semantic + keyword results."""
    semantic_results = [
        SearchResult(id="sem#0", score=0.9, metadata={"text": "Semantic 0"}),
        SearchResult(id="sem#1", score=0.8, metadata={"text": "Semantic 1"}),
    ]
    keyword_results = [
        SearchResult(id="kw#0", score=None, metadata={"text": "Keyword 0"}),
        SearchResult(id="kw#1", score=None, metadata={"text": "Keyword 1"}),
    ]

    embedder = DummyEmbedder()
    store = DummyStore(results=semantic_results)
    keyword_store = DummyKeywordStore(results_map={"tawhid": keyword_results})
    reranker = DummyReranker()
    pipeline = QueryPipeline(embedder, store, reranker, keyword_store=keyword_store)

    result = await pipeline.search(
        "What is Tawhid?",
        keyword_queries=["tawhid"],
        rerank=True,
    )

    # Verify result structure
    assert isinstance(result["formatted_context"], str)
    assert "results" in result

    # Verify reranker was called with combined results
    assert len(reranker.received_inputs) > 0
    rerank_input = reranker.received_inputs[0]
    rerank_result_ids = {r.id for r in rerank_input["results"]}

    # Should include both semantic and keyword results
    assert "sem#0" in rerank_result_ids or "sem#1" in rerank_result_ids
    assert "kw#0" in rerank_result_ids or "kw#1" in rerank_result_ids


@pytest.mark.asyncio
async def test_query_pipeline_semantic_priority():
    """Test that semantic results take priority over keyword results for same ID."""
    # Same ID with different scores and metadata
    semantic_result = SearchResult(
        id="doc#0",
        score=0.95,
        metadata={"text": "Semantic version", "priority": "semantic"},
    )
    keyword_result = SearchResult(
        id="doc#0",
        score=0.70,
        metadata={"text": "Keyword version", "priority": "keyword"},
    )

    embedder = DummyEmbedder()
    store = DummyStore(results=[semantic_result])
    keyword_store = DummyKeywordStore(results_map={"tawhid": [keyword_result]})
    reranker = DummyReranker()
    pipeline = QueryPipeline(embedder, store, reranker, keyword_store=keyword_store)

    result = await pipeline.search(
        "What is Tawhid?",
        keyword_queries=["tawhid"],
        rerank=False,
    )

    # Find the result with doc#0
    results = result["results"]
    doc0_results = [r for r in results if r.id == "doc#0"]
    assert len(doc0_results) == 1, "Should have exactly one doc#0 result"

    final_result = doc0_results[0]
    # Should preserve semantic result's data
    assert final_result.score == 0.95, "Should preserve semantic score"
    assert final_result.metadata["priority"] == "semantic", "Should preserve semantic metadata"
    assert final_result.metadata["text"] == "Semantic version", "Should preserve semantic text"


@pytest.mark.asyncio
async def test_query_pipeline_keyword_integration_qdrant():
    """End-to-end integration test with real Qdrant stores."""
    from maktaba.keyword.qdrant import QdrantKeywordStore
    from maktaba.storage.qdrant import QdrantStore

    # Create in-memory Qdrant stores
    collection_name = "test_query_pipeline_keyword_integration"
    vector_store = QdrantStore(url=":memory:", collection_name=collection_name)
    vector_store.create_collection(dimension=3)

    keyword_store = QdrantKeywordStore(
        collection_name=collection_name,
        text_field="text",
        client=vector_store.client,
    )

    # Insert test chunks
    from maktaba.models import VectorChunk

    chunks = [
        VectorChunk(
            id="doc_1#chunk_0",
            vector=[1.0, 0.0, 0.0],
            metadata={"text": "Tawhid is the oneness of Allah in Islamic theology."},
        ),
        VectorChunk(
            id="doc_2#chunk_0",
            vector=[0.0, 1.0, 0.0],
            metadata={"text": "Monotheism refers to the belief in one God."},
        ),
        VectorChunk(
            id="doc_3#chunk_0",
            vector=[0.0, 0.0, 1.0],
            metadata={"text": "Salah refers to the five daily prayers in Islam."},
        ),
    ]

    await vector_store.upsert(chunks)

    # Create pipeline
    embedder = DummyEmbedder()
    reranker = DummyReranker()
    pipeline = QueryPipeline(embedder, vector_store, reranker, keyword_store=keyword_store)

    # Perform search with parallel keyword queries
    # Note: Keyword search may not work without text indexing, but we test parallel execution
    result = await pipeline.search(
        "What is Tawhid?",
        keyword_queries=["Tawhid", "monotheism"],
        top_k=5,
        rerank=False,
    )

    # Verify results structure
    assert isinstance(result["formatted_context"], str)
    assert "results" in result
    assert "citations" in result

    results = result["results"]
    assert len(results) > 0

    # Should have at least semantic results (keyword search may fail without text indexing)
    result_ids = {r.id for r in results}
    # At minimum, semantic search should return results
    assert len(result_ids) > 0

    # Verify citations are formatted
    assert len(result["citations"]) > 0


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("SUPABASE_URL"), reason="SUPABASE_URL not set")
@pytest.mark.asyncio
async def test_query_pipeline_return_structure_with_supabase():
    """Integration test to verify QueryPipeline returns all expected fields with real Supabase."""
    from maktaba.keyword.supabase import SupabaseKeywordStore

    # Get Supabase credentials from environment
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    table_name = os.getenv("SUPABASE_TABLE_NAME", "page_content")

    # Create Supabase keyword store
    keyword_store = SupabaseKeywordStore(
        url=url,
        key=key,
        table_name=table_name,
        id_column=os.getenv("SUPABASE_ID_COLUMN", "page_key"),
        search_vector_column=os.getenv("SUPABASE_SEARCH_VECTOR_COLUMN", "fts"),
        text_column=os.getenv("SUPABASE_TEXT_COLUMN", "original_text"),
        language=os.getenv("SUPABASE_LANGUAGE", "arabic"),
        metadata_columns=['book_id', 'page_from_url', 'page', 'tome', 'page_key']
    )

    # Create pipeline with dummy vector store (we're testing keyword search integration)
    embedder = DummyEmbedder()
    vector_store = DummyStore(results=[
        SearchResult(
            id="semantic_1",
            score=0.95,
            metadata={"text": "This is a semantic search result about Tawhid."}
        ),
        SearchResult(
            id="semantic_2",
            score=0.90,
            metadata={"text": "Another semantic result about Islamic theology."}
        ),
    ])
    reranker = None  # No reranking for this test
    pipeline = QueryPipeline(
        embedder,
        vector_store,
        reranker,
        keyword_store=keyword_store,
    )

    # Perform search with keyword queries
    # Use a query that might exist in the database
    result = await pipeline.search(
        query="An-Naml - Verse 27 قَالَ سَنَنظُرُ أَصَدَقْتَ أَمْ كُنتَ مِنَ الْكَاذِبِينَ Tu peux expliquer ce verset ?",  # Arabic for "Islam" - adjust based on your test data
        keyword_queries=[
  'سورة النمل',
  'النمل 27',
  'سننظر أصدقت أم كنت من الكاذبين',
  'أصدقت',
  'كنت من الكاذبين',
  'الصدق والكذب'
],  # Try both Arabic and English
        top_k=10,
        keyword_limit=5,
        rerank=False,
    )

    # Verify all expected return fields are present
    assert "formatted_context" in result, "Missing formatted_context field"
    assert "citations" in result, "Missing citations field"
    assert "results" in result, "Missing results field"
    assert "semantic_results" in result, "Missing semantic_results field"
    assert "keyword_results" in result, "Missing keyword_results field"
    assert "keyword_result_count" in result, "Missing keyword_result_count field"

    # Verify types
    assert isinstance(result["formatted_context"], str), "formatted_context should be a string"
    assert isinstance(result["citations"], list), "citations should be a list"
    assert isinstance(result["results"], list), "results should be a list"
    assert isinstance(result["semantic_results"], list), "semantic_results should be a list"
    assert isinstance(result["keyword_results"], list), "keyword_results should be a list"
    assert isinstance(result["keyword_result_count"], int), "keyword_result_count should be an int"

    # Verify semantic_results contains SearchResult objects
    semantic_results = result["semantic_results"]
    assert len(semantic_results) > 0, "Should have at least some semantic results"
    for res in semantic_results:
        assert isinstance(res, SearchResult), "semantic_results should contain SearchResult objects"
        assert res.score is not None, "Semantic results should have scores"

    # Verify keyword_results (may be empty if no matches in database)
    keyword_results = result["keyword_results"]
    keyword_result_count = result["keyword_result_count"]
    assert len(keyword_results) == keyword_result_count, "keyword_results length should match keyword_result_count"

    for res in keyword_results:
        assert isinstance(res, SearchResult), "keyword_results should contain SearchResult objects"
        # Keyword results may have score=None (Supabase doesn't expose ts_rank in simple queries)

    # Verify merged results
    merged_results = result["results"]
    assert isinstance(merged_results, list), "results should be a list"
    assert len(merged_results) > 0, "Should have at least some merged results"

    # Verify deduplication: results should be unique by ID
    result_ids = {r.id for r in merged_results}
    assert len(result_ids) == len(merged_results), "Results should be deduplicated (no duplicate IDs)"

    # Verify citations structure
    citations = result["citations"]
    assert len(citations) > 0, "Should have at least some citations"
    for citation in citations:
        assert isinstance(citation, dict), "Each citation should be a dict"
        assert "index" in citation, "Citation should have index"
        assert "id" in citation, "Citation should have id"

    # Verify formatted_context contains citation markers
    formatted_context = result["formatted_context"]
    assert "[1]" in formatted_context or len(formatted_context) > 0, "formatted_context should contain citation markers or text"

    # Write results to file for inspection
    # output_file = "test_query_pipeline_keyword_results.txt"
    # with open(output_file, "w", encoding="utf-8") as f:
    #     f.write("=" * 80 + "\n")
    #     f.write("QueryPipeline Return Structure Test - Results\n")
    #     f.write("=" * 80 + "\n\n")
    #     f.write(f"Semantic results: {len(semantic_results)}\n")
    #     f.write(f"Keyword results: {len(keyword_results)}\n")
    #     f.write(f"Keyword result count: {keyword_result_count}\n")
    #     f.write(f"Merged results: {len(merged_results)}\n")
    #     f.write(f"Citations: {len(citations)}\n")
    #     f.write(f"Formatted context length: {len(formatted_context)}\n")
    #     f.write("\n" + "=" * 80 + "\n")
    #     f.write("SEMANTIC RESULTS\n")
    #     f.write("=" * 80 + "\n\n")
    #     for idx, res in enumerate(semantic_results, 1):
    #         f.write(f"{idx}. ID: {res.id}\n")
    #         f.write(f"   Score: {res.score}\n")
    #         f.write(f"   Text: {res.text[:200] if res.text else 'N/A'}...\n")
    #         f.write(f"   Metadata: {res.metadata}\n")
    #         f.write("\n")
    #     f.write("\n" + "=" * 80 + "\n")
    #     f.write("KEYWORD RESULTS\n")
    #     f.write("=" * 80 + "\n\n")
    #     if keyword_results:
    #         for idx, res in enumerate(keyword_results, 1):
    #             f.write(f"{idx}. ID: {res.id}\n")
    #             f.write(f"   Score: {res.score}\n")
    #             f.write(f"   Text: {res.text[:200] if res.text else 'N/A'}...\n")
    #             f.write(f"   Full Text: {res.text if res.text else 'N/A'}\n")
    #             f.write(f"   Metadata: {res.metadata}\n")
    #             f.write(f"   Full data: {res}\n")
    #             f.write("\n")
    #     else:
    #         f.write("No keyword results found.\n")
    #     f.write("\n" + "=" * 80 + "\n")
    #     f.write("MERGED RESULTS\n")
    #     f.write("=" * 80 + "\n\n")
    #     for idx, res in enumerate(merged_results, 1):
    #         f.write(f"{idx}. ID: {res.id}\n")
    #         f.write(f"   Score: {res.score}\n")
    #         f.write(f"   Text preview: {res.text[:200] if res.text else 'N/A'}...\n")
    #         f.write("\n")
    #     f.write("\n" + "=" * 80 + "\n")
    #     f.write("CITATIONS\n")
    #     f.write("=" * 80 + "\n\n")
    #     for citation in citations:
    #         f.write(f"{citation}\n")
    #     f.write("\n" + "=" * 80 + "\n")
    #     f.write("FORMATTED CONTEXT\n")
    #     f.write("=" * 80 + "\n\n")
    #     f.write(formatted_context)
    #     f.write("\n")
    # print(f"\nResults written to: {output_file}")
