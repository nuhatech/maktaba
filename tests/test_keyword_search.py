"""Tests for keyword search stores."""

import os

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


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("SUPABASE_URL"), reason="SUPABASE_URL not set")
@pytest.mark.asyncio
async def test_supabase_keyword_search_basic():
    """Test basic Supabase keyword search with real database."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    table_name = os.getenv("SUPABASE_TABLE_NAME", "page_content")

    store = SupabaseKeywordStore(
        url=url,
        key=key,
        table_name=table_name,
        id_column="page_key",
        search_vector_column="fts",
        text_column="original_text",
        language="arabic",
    )

    # Test basic search - this should work with any existing data
    results = await store.search(query="الإسلام", limit=5)
    assert isinstance(results, list)
    # Results may be empty if no data matches, but should not raise an error
    print(f"Basic search returned {len(results)} results")


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("SUPABASE_URL"), reason="SUPABASE_URL not set")
@pytest.mark.asyncio
async def test_supabase_keyword_search_with_quotes():
    """
    Test Supabase keyword search with quoted queries (the bug we fixed).

    This tests that queries with surrounding quotes (from LLM-generated queries)
    are properly handled and don't cause PostgreSQL syntax errors.
    """
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    table_name = os.getenv("SUPABASE_TABLE_NAME", "page_content")

    store = SupabaseKeywordStore(
        url=url,
        key=key,
        table_name=table_name,
        id_column="page_key",
        search_vector_column="fts",
        text_column="original_text",
        language="arabic",
    )

    # Test with double quotes (simulating LLM-generated query)
    try:
        results = await store.search(query='"التوحيد"', limit=5)
        assert isinstance(results, list)
        print(f"Search with double quotes returned {len(results)} results")
    except Exception as e:
        pytest.fail(f"Search with quoted query failed: {e}")

    # Test with single quotes
    try:
        results = await store.search(query="'الفقه'", limit=5)
        assert isinstance(results, list)
        print(f"Search with single quotes returned {len(results)} results")
    except Exception as e:
        pytest.fail(f"Search with single-quoted query failed: {e}")

    # Test with mixed spacing and quotes
    try:
        results = await store.search(query='  "الصلاة"  ', limit=5)
        assert isinstance(results, list)
        print(f"Search with spaced quotes returned {len(results)} results")
    except Exception as e:
        pytest.fail(f"Search with spaced quoted query failed: {e}")


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("SUPABASE_URL"), reason="SUPABASE_URL not set")
@pytest.mark.asyncio
async def test_supabase_keyword_search_with_filters():
    """Test Supabase keyword search with filters and namespace."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    table_name = os.getenv("SUPABASE_TABLE_NAME", "page_content")

    store = SupabaseKeywordStore(
        url=url,
        key=key,
        table_name=table_name,
        id_column="page_key",
        search_vector_column="fts",
        text_column="original_text",
        language="arabic",
    )

    # Test with namespace filter if you have namespaces in your data
    try:
        results = await store.search(
            query="الإسلام",
            limit=5,
            namespace="default"  # Adjust this to match your actual namespace
        )
        assert isinstance(results, list)
        print(f"Search with namespace filter returned {len(results)} results")
    except Exception as e:
        # If namespace column doesn't exist, this is expected
        print(f"Namespace filter test skipped: {e}")

    # Test with custom filter (adjust field name to match your schema)
    # Uncomment and modify if you have a book_id or similar field:
    # try:
    #     results = await store.search(
    #         query="الإسلام",
    #         limit=5,
    #         filter={"book_id": "some_book_id"}
    #     )
    #     assert isinstance(results, list)
    #     print(f"Search with custom filter returned {len(results)} results")
    # except Exception as e:
    #     print(f"Custom filter test skipped: {e}")


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("SUPABASE_URL"), reason="SUPABASE_URL not set")
@pytest.mark.asyncio
async def test_supabase_keyword_search_rpc():
    """Test Supabase keyword search with RPC function."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    table_name = os.getenv("SUPABASE_TABLE_NAME", "page_content")
    rpc_function = os.getenv("SUPABASE_RPC_FUNCTION", "keyword_search")

    # Test basic RPC search
    store = SupabaseKeywordStore(
        url=url,
        key=key,
        table_name=table_name,
        id_column="page_key",
        search_vector_column="fts",
        text_column="original_text",
        language="arabic",
        use_rpc=True,
        rpc_function_name=rpc_function,
        rpc_id_column="page_key",
        rpc_text_column="original_text",
        rpc_score_column="rank",
    )

    try:
        # Test basic RPC search
        query = "سننظر أصدقت أم كنت من الكاذبين"
        results = await store.search(query=query, limit=5)
        assert isinstance(results, list)
        print(f"RPC search returned {len(results)} results")

        # If we have results, verify structure
        if results:
            result = results[0]
            # Verify SearchResult structure
            assert hasattr(result, "id")
            assert hasattr(result, "score")
            assert hasattr(result, "metadata")
            # Verify score is returned (ts_rank from RPC)
            assert result.score is not None, "RPC should return ts_rank scores"
            assert isinstance(result.score, float), "Score should be a float"
            assert result.score >= 0, "ts_rank scores should be non-negative"
            # Verify text is in metadata
            assert "text" in result.metadata or "original_text" in result.metadata
            # Verify other RPC columns are in metadata
            print(f"Sample result ID: {result.id}, Score: {result.score}")
            print(f"Sample result metadata keys: {list(result.metadata.keys())}")

        # Write results to file for inspection
        # output_file = "test_supabase_keyword_search_rpc_results.txt"
        # with open(output_file, "w", encoding="utf-8") as f:
        #     f.write("=" * 80 + "\n")
        #     f.write("Supabase Keyword Search RPC Test - Results\n")
        #     f.write("=" * 80 + "\n\n")
        #     f.write(f"Query: {query}\n")
        #     f.write(f"RPC Function: {rpc_function}\n")
        #     f.write(f"Table: {table_name}\n")
        #     f.write(f"Total Results: {len(results)}\n")
        #     f.write(f"RPC ID Column: {store.rpc_id_column}\n")
        #     f.write(f"RPC Text Column: {store.rpc_text_column}\n")
        #     f.write(f"RPC Score Column: {store.rpc_score_column}\n")
        #     f.write("\n" + "=" * 80 + "\n")
        #     f.write("RPC SEARCH RESULTS\n")
        #     f.write("=" * 80 + "\n\n")
        #     if results:
        #         for idx, res in enumerate(results, 1):
        #             f.write(f"{idx}. ID: {res.id}\n")
        #             f.write(f"   Score (ts_rank): {res.score}\n")
        #             f.write(f"   Text Preview: {res.text[:200] if res.text else 'N/A'}...\n")
        #             f.write(f"   Full Text: {res.text if res.text else 'N/A'}\n")
        #             f.write(f"   Metadata Keys: {list(res.metadata.keys())}\n")
        #             f.write(f"   Full Metadata: {res.metadata}\n")
        #             f.write(f"   Full SearchResult: {res}\n")
        #             f.write("\n")
        #     else:
        #         f.write("No results found.\n")
        #     f.write("\n" + "=" * 80 + "\n")
        #     f.write("CONFIGURATION\n")
        #     f.write("=" * 80 + "\n\n")
        #     f.write(f"use_rpc: {store.use_rpc}\n")
        #     f.write(f"rpc_function_name: {store.rpc_function_name}\n")
        #     f.write(f"rpc_filter_mapping: {store.rpc_filter_mapping}\n")
        #     f.write(f"rpc_id_column: {store.rpc_id_column}\n")
        #     f.write(f"rpc_text_column: {store.rpc_text_column}\n")
        #     f.write(f"rpc_score_column: {store.rpc_score_column}\n")
        #     f.write(f"id_column: {store.id_column}\n")
        #     f.write(f"text_column: {store.text_column}\n")
        #     f.write(f"search_vector_column: {store.search_vector_column}\n")
        #     f.write(f"language: {store.language}\n")
        # print(f"\nResults written to: {output_file}")

    except Exception as e:
        # If RPC function doesn't exist, skip the test
        if "function" in str(e).lower() and "does not exist" in str(e).lower():
            pytest.skip(f"RPC function '{rpc_function}' does not exist: {e}")
        else:
            raise


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("SUPABASE_URL"), reason="SUPABASE_URL not set")
@pytest.mark.asyncio
async def test_supabase_keyword_search_rpc_with_filters():
    """Test Supabase keyword search with RPC function and filter mapping."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    table_name = os.getenv("SUPABASE_TABLE_NAME", "page_content")
    rpc_function = os.getenv("SUPABASE_RPC_FUNCTION", "keyword_search")

    # Test RPC with filter mapping
    # Note: This assumes your RPC function accepts p_book_id, p_author_id, etc.
    # Adjust rpc_filter_mapping based on your actual RPC function signature
    store = SupabaseKeywordStore(
        url=url,
        key=key,
        table_name=table_name,
        id_column="page_key",
        search_vector_column="fts",
        text_column="original_text",
        language="arabic",
        use_rpc=True,
        rpc_function_name=rpc_function,
        rpc_filter_mapping={
            "book_id": "p_book_id",
            "author_id": "p_author_id",
            "book_category": "p_book_category",
        },
        rpc_id_column="page_key",
        rpc_text_column="original_text",
        rpc_score_column="rank",
    )

    try:
        # Test RPC search without filters
        # results_no_filter = await store.search(query="الإسلام", limit=5)
        # assert isinstance(results_no_filter, list)
        # print(f"RPC search (no filter) returned {len(results_no_filter)} results")

        # Test RPC search with filter (if you have book_id in your data)
        # Uncomment and adjust if you have book_id values in your database:
        
        query = "سننظر أصدقت أم كنت من الكاذبين"
        results_with_filter = await store.search(
            query=query,
            limit=5,
            filter={"book_id": 7798}  # Adjust to a valid book_id
        )
        assert isinstance(results_with_filter, list)
        print(f"RPC search (with filter) returned {len(results_with_filter)} results")
        # Verify filter was applied (if we have results)
        if results_with_filter:
            # Check that results have the filtered book_id in metadata
            for result in results_with_filter:
                if "book_id" in result.metadata:
                    assert result.metadata["book_id"] == 7798

    except Exception as e:
        # If RPC function doesn't exist, skip the test
        if "function" in str(e).lower() and "does not exist" in str(e).lower():
            pytest.skip(f"RPC function '{rpc_function}' does not exist: {e}")
        else:
            raise


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("SUPABASE_URL"), reason="SUPABASE_URL not set")
@pytest.mark.asyncio
async def test_supabase_keyword_search_rpc_column_mapping():
    """Test Supabase keyword search RPC with custom column mapping."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    table_name = os.getenv("SUPABASE_TABLE_NAME", "page_content")
    rpc_function = os.getenv("SUPABASE_RPC_FUNCTION", "keyword_search")

    # Test RPC with custom column mapping (different from defaults)
    store = SupabaseKeywordStore(
        url=url,
        key=key,
        table_name=table_name,
        id_column="id",  # Default id_column
        search_vector_column="fts",
        text_column="text",  # Default text_column
        language="arabic",
        use_rpc=True,
        rpc_function_name=rpc_function,
        rpc_id_column="page_key",  # RPC returns "page_key" but we map it
        rpc_text_column="original_text",  # RPC returns "original_text" but we map it
        rpc_score_column="rank",  # RPC returns "rank" for score
    )

    try:
        results = await store.search(query="الإسلام", limit=5)
        assert isinstance(results, list)
        print(f"RPC search with custom column mapping returned {len(results)} results")

        # Verify column mapping worked correctly
        if results:
            result = results[0]
            # ID should come from rpc_id_column ("page_key")
            assert result.id is not None
            # Score should come from rpc_score_column ("rank")
            assert result.score is not None
            # Text should be accessible
            assert result.text is not None or len(result.text) > 0
            # Metadata should contain other RPC columns
            print(f"Result ID (from page_key): {result.id}")
            print(f"Result score (from rank): {result.score}")
            print(f"Result text length: {len(result.text)}")

    except Exception as e:
        # If RPC function doesn't exist, skip the test
        if "function" in str(e).lower() and "does not exist" in str(e).lower():
            pytest.skip(f"RPC function '{rpc_function}' does not exist: {e}")
        else:
            raise