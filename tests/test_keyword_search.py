"""Tests for keyword search stores."""

import pytest

from maktaba.keyword.qdrant import QdrantKeywordStore
from maktaba.keyword.supabase import SupabaseKeywordStore
from maktaba.models import VectorChunk

# =============================================================================
# QdrantKeywordStore Tests
# =============================================================================


@pytest.mark.asyncio
async def test_qdrant_keyword_search_basic():
    """Test basic keyword search functionality."""
    # Create a Qdrant vector store to set up the collection and insert data
    from maktaba.storage.qdrant import QdrantStore

    vector_store = QdrantStore(url=":memory:", collection_name="test_keyword_basic")
    vector_store.create_collection(dimension=3)

    # Create keyword store using the same client
    store = QdrantKeywordStore(
        collection_name="test_keyword_basic",
        text_field="text",
        client=vector_store.client,
    )

    # Insert chunks with text field for keyword search
    chunks = [
        VectorChunk(
            id="doc_1#chunk_0",
            vector=[1.0, 0.0, 0.0],
            metadata={"text": "Tawhid is the oneness of Allah in Islamic theology."},
        ),
        VectorChunk(
            id="doc_2#chunk_0",
            vector=[1.0, 0.0, 0.0],
            metadata={"text": "Zakat is the Islamic practice of charitable giving."},
        ),
        VectorChunk(
            id="doc_3#chunk_0",
            vector=[1.0, 0.0, 0.0],
            metadata={"text": "Salah refers to the five daily prayers in Islam."},
        ),
    ]

    await vector_store.upsert(chunks)

    # Search for keyword
    results = await store.search(query="Tawhid", limit=10)

    # Should return results containing "Tawhid"
    assert len(results) > 0
    assert any("Tawhid" in r.metadata.get("text", "") for r in results)


@pytest.mark.asyncio
async def test_qdrant_keyword_search_filter_single_value():
    """Test keyword search with single value filter."""
    from maktaba.storage.qdrant import QdrantStore

    vector_store = QdrantStore(url=":memory:", collection_name="test_keyword_filter_single")
    vector_store.create_collection(dimension=3)

    store = QdrantKeywordStore(
        collection_name="test_keyword_filter_single",
        client=vector_store.client,
    )

    # Insert chunks with different book_ids
    chunks = [
        VectorChunk(
            id="book_123#chunk_0",
            vector=[1.0, 0.0, 0.0],
            metadata={"text": "Tawhid in book 123", "book_id": 123},
        ),
        VectorChunk(
            id="book_456#chunk_0",
            vector=[1.0, 0.0, 0.0],
            metadata={"text": "Tawhid in book 456", "book_id": 456},
        ),
    ]

    await vector_store.upsert(chunks)

    # Search with filter
    results = await store.search(query="Tawhid", limit=10, filter={"book_id": 123})

    # Should only return book_123
    assert len(results) > 0
    assert all(r.metadata.get("book_id") == 123 for r in results)


@pytest.mark.asyncio
async def test_qdrant_keyword_search_filter_list_values():
    """
    Test keyword search with list value filter (MatchAny).

    This tests the fix we implemented for handling list filters.
    """
    from maktaba.storage.qdrant import QdrantStore

    vector_store = QdrantStore(url=":memory:", collection_name="test_keyword_filter_list")
    vector_store.create_collection(dimension=3)

    store = QdrantKeywordStore(
        collection_name="test_keyword_filter_list",
        client=vector_store.client,
    )

    # Insert chunks with different book_ids
    chunks = [
        VectorChunk(
            id="book_123#chunk_0",
            vector=[1.0, 0.0, 0.0],
            metadata={"text": "Islamic theology in book 123", "book_id": 123},
        ),
        VectorChunk(
            id="book_456#chunk_0",
            vector=[1.0, 0.0, 0.0],
            metadata={"text": "Islamic practices in book 456", "book_id": 456},
        ),
        VectorChunk(
            id="book_789#chunk_0",
            vector=[1.0, 0.0, 0.0],
            metadata={"text": "Islamic history in book 789", "book_id": 789},
        ),
    ]

    await vector_store.upsert(chunks)

    # Search with list filter (should use MatchAny)
    results = await store.search(
        query="Islamic", limit=10, filter={"book_id": [123, 456]}
    )

    # Should return only books 123 and 456 (not 789)
    assert len(results) >= 2
    result_book_ids = {r.metadata.get("book_id") for r in results}
    assert 123 in result_book_ids
    assert 456 in result_book_ids
    assert 789 not in result_book_ids


@pytest.mark.asyncio
async def test_qdrant_keyword_search_namespace_isolation():
    """Test keyword search respects namespace isolation."""
    from maktaba.storage.qdrant import QdrantStore

    vector_store = QdrantStore(url=":memory:", collection_name="test_keyword_namespace")
    vector_store.create_collection(dimension=3)

    store = QdrantKeywordStore(
        collection_name="test_keyword_namespace",
        client=vector_store.client,
    )

    # Insert chunks in different namespaces
    chunks_ns1 = [
        VectorChunk(
            id="doc_1#chunk_0",
            vector=[1.0, 0.0, 0.0],
            metadata={"text": "Tawhid in namespace 1"},
        ),
    ]

    chunks_ns2 = [
        VectorChunk(
            id="doc_2#chunk_0",
            vector=[1.0, 0.0, 0.0],
            metadata={"text": "Tawhid in namespace 2"},
        ),
    ]

    await vector_store.upsert(chunks_ns1, namespace="ns1")
    await vector_store.upsert(chunks_ns2, namespace="ns2")

    # Search in namespace 1
    results_ns1 = await store.search(query="Tawhid", limit=10, namespace="ns1")

    # Should only return namespace 1 results
    assert len(results_ns1) == 1
    assert "namespace 1" in results_ns1[0].metadata.get("text", "")

    # Search in namespace 2
    results_ns2 = await store.search(query="Tawhid", limit=10, namespace="ns2")

    # Should only return namespace 2 results
    assert len(results_ns2) == 1
    assert "namespace 2" in results_ns2[0].metadata.get("text", "")


@pytest.mark.asyncio
async def test_qdrant_keyword_search_empty_results():
    """Test keyword search returns empty list when no matches found."""
    from maktaba.storage.qdrant import QdrantStore

    vector_store = QdrantStore(url=":memory:", collection_name="test_keyword_empty")
    vector_store.create_collection(dimension=3)

    store = QdrantKeywordStore(
        collection_name="test_keyword_empty",
        client=vector_store.client,
    )

    # Insert chunks
    chunks = [
        VectorChunk(
            id="doc_1#chunk_0",
            vector=[1.0, 0.0, 0.0],
            metadata={"text": "This is about Tawhid"},
        ),
    ]

    await vector_store.upsert(chunks)

    # Search for non-existent keyword
    results = await store.search(query="NonExistentKeyword", limit=10)

    # Should return empty list
    assert len(results) == 0


# =============================================================================
# SupabaseKeywordStore Tests
# =============================================================================


def test_supabase_keyword_search_import_error():
    """Test SupabaseKeywordStore raises helpful error when supabase not installed."""
    # This test will fail if supabase IS installed, so we skip in that case
    try:
        import supabase  # noqa: F401

        pytest.skip("supabase is installed, skipping import error test")
    except ImportError:
        pass

    # Mock the import to fail
    import sys
    from unittest.mock import patch

    with patch.dict(sys.modules, {"supabase": None}):
        with pytest.raises(ImportError, match="supabase package is required"):
            SupabaseKeywordStore(
                url="https://test.supabase.co",
                key="test-key",
                table_name="test_table",
            )


# NOTE: Integration tests for SupabaseKeywordStore would require actual Supabase connection
# These would be marked with @pytest.mark.integration and skipped if env vars not set
# Example:
#
# @pytest.mark.integration
# @pytest.mark.skipif(not os.getenv("SUPABASE_URL"), reason="SUPABASE_URL not set")
# async def test_supabase_keyword_search_basic():
#     url = os.getenv("SUPABASE_URL")
#     key = os.getenv("SUPABASE_KEY")
#     store = SupabaseKeywordStore(url=url, key=key, table_name="test_table")
#     results = await store.search(query="test", limit=10)
#     assert isinstance(results, list)
