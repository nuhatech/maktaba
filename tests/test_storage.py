"""Tests for vector storage implementations."""

import pytest

from maktaba.models import VectorChunk
from maktaba.storage.qdrant import QdrantStore


@pytest.mark.asyncio
async def test_qdrant_store_with_string_ids():
    """Test QdrantStore correctly handles string IDs like 'book_123#chunk_0'."""
    # Use in-memory mode for testing (no Docker needed)
    store = QdrantStore(url=":memory:", collection_name="test_collection")

    # Create collection
    store.create_collection(dimension=3)

    # Create chunks with string IDs (format used in production)
    chunks = [
        VectorChunk(
            id="book_2908#chunk_0",
            vector=[1.0, 2.0, 3.0],
            metadata={"text": "First chunk", "page": 1},
        ),
        VectorChunk(
            id="book_2908#chunk_1",
            vector=[2.0, 3.0, 4.0],
            metadata={"text": "Second chunk", "page": 2},
        ),
        VectorChunk(
            id="book_2908#chunk_2",
            vector=[3.0, 4.0, 5.0],
            metadata={"text": "Third chunk", "page": 3},
        ),
    ]

    # Test upsert (should convert to UUIDs internally)
    await store.upsert(chunks)

    # Test query - should return original string IDs, not UUIDs
    results = await store.query(
        vector=[1.5, 2.5, 3.5],
        topK=3,
        includeMetadata=True,
    )

    assert len(results) == 3

    # Verify original IDs are returned (not UUIDs)
    result_ids = {r.id for r in results}
    assert "book_2908#chunk_0" in result_ids
    assert "book_2908#chunk_1" in result_ids
    assert "book_2908#chunk_2" in result_ids

    # Verify metadata is preserved
    for result in results:
        assert "text" in result.metadata
        assert "page" in result.metadata
        assert "_original_id" in result.metadata  # Internal field for UUID mode

    # Test list - should return original IDs
    all_ids = await store.list(limit=10)
    assert len(all_ids) == 3
    assert "book_2908#chunk_0" in all_ids

    # Test list with prefix
    book_ids = await store.list(prefix="book_2908", limit=10)
    assert len(book_ids) == 3

    # Test delete - should work with original IDs
    await store.delete(["book_2908#chunk_1"])

    remaining_ids = await store.list(limit=10)
    assert len(remaining_ids) == 2
    assert "book_2908#chunk_1" not in remaining_ids


@pytest.mark.asyncio
async def test_qdrant_store_delete_by_document():
    """Test deleting all chunks for a document by document ID."""
    store = QdrantStore(url=":memory:", collection_name="test_delete")
    store.create_collection(dimension=3)

    # Create chunks for two documents
    chunks = [
        VectorChunk(
            id="book_123#chunk_0",
            vector=[1.0, 0.0, 0.0],
            metadata={"text": "Book 123 chunk 0"},
        ),
        VectorChunk(
            id="book_123#chunk_1",
            vector=[1.0, 0.0, 0.0],
            metadata={"text": "Book 123 chunk 1"},
        ),
        VectorChunk(
            id="book_456#chunk_0",
            vector=[1.0, 0.0, 0.0],
            metadata={"text": "Book 456 chunk 0"},
        ),
    ]

    await store.upsert(chunks)

    # Delete all chunks for book_123
    await store.delete_by_document("book_123")

    # Verify only book_456 chunks remain
    remaining_ids = await store.list(limit=10)
    assert len(remaining_ids) == 1
    assert "book_456#chunk_0" in remaining_ids
    assert "book_123#chunk_0" not in remaining_ids
    assert "book_123#chunk_1" not in remaining_ids


@pytest.mark.asyncio
async def test_qdrant_store_namespace_isolation():
    """Test namespace isolation in queries."""
    store = QdrantStore(url=":memory:", collection_name="test_namespace")
    store.create_collection(dimension=3)

    # Create chunks in different namespaces
    chunks_ns1 = [
        VectorChunk(
            id="doc_1#chunk_0",
            vector=[1.0, 0.0, 0.0],
            metadata={"text": "Namespace 1"},
        ),
    ]

    chunks_ns2 = [
        VectorChunk(
            id="doc_2#chunk_0",
            vector=[1.0, 0.0, 0.0],
            metadata={"text": "Namespace 2"},
        ),
    ]

    await store.upsert(chunks_ns1, namespace="ns1")
    await store.upsert(chunks_ns2, namespace="ns2")

    # Query namespace 1
    results_ns1 = await store.query(
        vector=[1.0, 0.0, 0.0],
        topK=10,
        namespace="ns1",
    )

    assert len(results_ns1) == 1
    assert results_ns1[0].id == "doc_1#chunk_0"

    # Query namespace 2
    results_ns2 = await store.query(
        vector=[1.0, 0.0, 0.0],
        topK=10,
        namespace="ns2",
    )

    assert len(results_ns2) == 1
    assert results_ns2[0].id == "doc_2#chunk_0"

    # Query without namespace should return all
    results_all = await store.query(
        vector=[1.0, 0.0, 0.0],
        topK=10,
    )

    assert len(results_all) == 2
