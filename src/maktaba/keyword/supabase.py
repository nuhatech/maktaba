"""Supabase keyword search implementation using PostgreSQL FTS."""

import os
from typing import Any, Dict, List, Optional

try:
    from supabase import Client as SyncClient
    from supabase import create_client as create_sync_client

    Client = SyncClient
    create_client = create_sync_client
except ImportError:
    Client = None  # type: ignore[assignment]
    create_client = None  # type: ignore[assignment]

from ..exceptions import StorageError
from ..models import SearchResult
from .base import BaseKeywordStore


class SupabaseKeywordStore(BaseKeywordStore):
    """
    Supabase (PostgreSQL) full-text keyword search implementation.

    Uses PostgreSQL Full-Text Search (FTS) with ts_rank() scoring.
    Requires a tsvector column and GIN index for efficient search.

    Database Setup:
        -- Add tsvector column for FTS
        ALTER TABLE your_table
        ADD COLUMN search_vector tsvector;

        -- Create GIN index for performance
        CREATE INDEX idx_search_vector
        ON your_table USING gin(search_vector);

        -- Auto-update search vector on insert/update
        CREATE TRIGGER update_search_vector
        BEFORE INSERT OR UPDATE ON your_table
        FOR EACH ROW EXECUTE FUNCTION
        tsvector_update_trigger(search_vector, 'pg_catalog.english', text_column);

    RPC Function (Optional):
        For better performance and ts_rank scores, you can use a custom RPC function:

        CREATE OR REPLACE FUNCTION public.keyword_search(
          q text,
          limit_rows int DEFAULT 50,
          offset_rows int DEFAULT 0,
          p_book_id bigint DEFAULT NULL,
          p_author_id bigint DEFAULT NULL,
          p_book_category bigint DEFAULT NULL
        )
        RETURNS TABLE (
          page_key text,
          book_id bigint,
          tome int,
          page int,
          page_from_url text,
          original_text text,
          rank real
        )
        LANGUAGE sql
        STABLE
        AS $$
          SELECT
            t.page_key,
            t.book_id,
            t.tome,
            t.page,
            t.page_from_url,
            t.original_text,
            ts_rank_cd(t.fts, websearch_to_tsquery('arabic', q)) AS rank
          FROM public.page_content AS t
          WHERE
            t.fts @@ websearch_to_tsquery('arabic', q)
            AND (p_book_id IS NULL OR t.book_id = p_book_id)
            AND (p_author_id IS NULL OR b.author_id = p_author_id)
            AND (p_book_category IS NULL OR b.book_category = p_book_category)
          ORDER BY rank DESC
          LIMIT limit_rows
          OFFSET offset_rows;
        $$;

        Note: The RPC function handles book table joins internally for author_id
        and book_category filters. Namespace filtering is not supported in RPC
        and must be done in Python if needed.

    Examples:
        # Create keyword store with text_search (default)
        keyword_store = SupabaseKeywordStore(
            url="https://xxx.supabase.co",
            key="your-anon-key",
            table_name="page_content",
            text_column="text",
            search_vector_column="search_vector"
        )

        # Create keyword store with RPC function
        keyword_store = SupabaseKeywordStore(
            url="https://xxx.supabase.co",
            key="your-anon-key",
            table_name="page_content",
            text_column="original_text",
            search_vector_column="fts",
            id_column="page_key",
            use_rpc=True,
            rpc_function_name="keyword_search",
            rpc_filter_mapping={
                "book_id": "p_book_id",
                "author_id": "p_author_id",
                "book_category": "p_book_category"
            },
            rpc_id_column="page_key",      # RPC returns "page_key" column
            rpc_text_column="original_text", # RPC returns "original_text" column
            rpc_score_column="rank"         # RPC returns "rank" column
        )

        # Search
        results = await keyword_store.search(
            query="Islamic jurisprudence",
            limit=15,
            filter={"book_id": 123}  # Maps to p_book_id via rpc_filter_mapping
        )
    """

    def __init__(
        self,
        url: Optional[str] = None,
        key: Optional[str] = None,
        table_name: str = "page_content",
        text_column: str = "text",
        search_vector_column: str = "search_vector",
        id_column: str = "id",
        metadata_columns: Optional[List[str]] = None,
        language: str = "english",
        use_rpc: bool = False,
        rpc_function_name: str = "keyword_search",
        rpc_filter_mapping: Optional[Dict[str, str]] = None,
        rpc_id_column: Optional[str] = None,
        rpc_text_column: Optional[str] = None,
        rpc_score_column: str = "rank",
    ):
        """
        Initialize Supabase keyword store.

        Args:
            url: Supabase project URL (fallback: SUPABASE_URL env var)
            key: Supabase API key (fallback: SUPABASE_KEY env var)
            table_name: Name of the table to search
            text_column: Name of the text column (default: "text")
            search_vector_column: Name of the tsvector column (default: "search_vector")
            id_column: Name of the ID column (default: "id")
            metadata_columns: List of additional columns to include in results
            language: FTS language configuration (default: "english")
            use_rpc: If True, use RPC function instead of text_search() (default: False)
            rpc_function_name: Name of the RPC function to call (default: "keyword_search")
            rpc_filter_mapping: Optional dict mapping filter keys to RPC parameter names.
                Example: {"book_id": "p_book_id", "author_id": "p_author_id"}
                If None, filters are ignored in RPC mode.
            rpc_id_column: RPC return column name for ID (default: uses id_column value)
            rpc_text_column: RPC return column name for text (default: uses text_column value)
            rpc_score_column: RPC return column name for score/rank (default: "rank")

        Raises:
            ImportError: If supabase package is not installed
        """
        if Client is None or create_client is None:
            raise ImportError(
                "supabase package is required for SupabaseKeywordStore. "
                "Install with: pip install supabase"
            )

        # Use explicit parameters or fallback to environment variables
        _url = url or os.getenv("SUPABASE_URL")
        _key = key or os.getenv("SUPABASE_KEY")

        if not _url:
            raise ValueError(
                "Supabase URL must be provided either as url parameter or SUPABASE_URL environment variable"
            )
        if not _key:
            raise ValueError(
                "Supabase API key must be provided either as key parameter or SUPABASE_KEY environment variable"
            )

        self.client = create_client(_url, _key)  # type: ignore[assignment]
        self.table_name = table_name
        self.text_column = text_column
        self.search_vector_column = search_vector_column
        self.id_column = id_column
        self.metadata_columns = metadata_columns or []
        self.language = language
        self.use_rpc = use_rpc
        self.rpc_function_name = rpc_function_name
        self.rpc_filter_mapping = rpc_filter_mapping
        self.rpc_id_column = rpc_id_column or id_column
        self.rpc_text_column = rpc_text_column or text_column
        self.rpc_score_column = rpc_score_column

    async def search(
        self,
        query: str,
        limit: int = 15,
        filter: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Search for documents using PostgreSQL Full-Text Search.

        Args:
            query: Keyword search query (plain text)
            limit: Maximum number of results to return
            filter: Optional metadata filters (column-value pairs)
                - For RPC mode: Filters are mapped to RPC parameters via rpc_filter_mapping
                  (if rpc_filter_mapping is None, filters are ignored)
                - For text_search mode: Supports any column filters
            namespace: Optional namespace filter (if namespace column exists)
                - Note: Namespace filtering not supported in RPC mode, will be filtered in Python

        Returns:
            List of SearchResult objects sorted by ts_rank score (descending)
            - RPC mode: Returns actual ts_rank scores
            - text_search mode: Returns score=None (ts_rank not exposed)

        Note:
            If use_rpc=True, calls the configured RPC function which returns ts_rank scores.
            Otherwise, uses text_search() which doesn't expose scores.
        """
        try:
            # Use RPC path if enabled
            if self.use_rpc:
                return await self._search_rpc(query, limit, filter, namespace)
            else:
                return await self._search_text_search(query, limit, filter, namespace)
        except Exception as e:
            raise StorageError(
                f"Supabase keyword search failed: {str(e)}"
            ) from e

    async def _search_rpc(
        self,
        query: str,
        limit: int,
        filter: Optional[Dict[str, Any]],
        namespace: Optional[str],
    ) -> List[SearchResult]:
        """Search using RPC function."""
        if self.client is None:
            raise StorageError("Supabase client not initialized")

        # Strip surrounding quotes if present
        clean_query = query.strip().strip('"').strip("'")

        # Build RPC parameters
        rpc_params: Dict[str, Any] = {
            "q": clean_query,
            "limit_rows": limit,
            "offset_rows": 0,
        }

        # Map filter parameters to RPC function parameters using configurable mapping
        if filter and self.rpc_filter_mapping:
            for filter_key, rpc_param_name in self.rpc_filter_mapping.items():
                if filter_key in filter:
                    rpc_params[rpc_param_name] = filter[filter_key]

        # Call RPC function
        response = self.client.rpc(self.rpc_function_name, rpc_params).execute()

        if not response.data:
            return []

        # Convert RPC results to SearchResult
        search_results = []
        for row in response.data:
            # Map RPC return columns to SearchResult using configurable column names
            chunk_id = str(row.get(self.rpc_id_column, ""))
            text_content = row.get(self.rpc_text_column, "")
            rank_score = row.get(self.rpc_score_column)

            # Build metadata from all returned columns except id/text/score columns
            metadata: Dict[str, Any] = {}
            excluded_columns = {self.rpc_id_column, self.rpc_text_column, self.rpc_score_column}

            # Include all columns from RPC response as metadata (except id/text/score)
            for col_name, col_value in row.items():
                if col_name not in excluded_columns:
                    metadata[col_name] = col_value

            # Add any additional metadata columns that might be in the result
            for col in self.metadata_columns:
                if col in row and col not in metadata:
                    metadata[col] = row[col]

            # Convert rank (real) to float for score
            score = float(rank_score) if rank_score is not None else None

            # Add text to metadata (SearchResult.text property reads from metadata["text"])
            # Use text_column name for consistency, but also add "text" for compatibility
            metadata["text"] = text_content
            if self.text_column != "text" and self.text_column not in metadata:
                metadata[self.text_column] = text_content

            search_results.append(
                SearchResult(
                    id=chunk_id,
                    score=score,
                    metadata=metadata,
                )
            )

        # Apply namespace filtering in Python if needed (RPC doesn't support it)
        if namespace:
            # Filter by namespace if it exists in metadata
            search_results = [
                r for r in search_results
                if r.metadata.get("namespace") == namespace
            ]

        return search_results

    async def _search_text_search(
        self,
        query: str,
        limit: int,
        filter: Optional[Dict[str, Any]],
        namespace: Optional[str],
    ) -> List[SearchResult]:
        """Search using text_search() method (default)."""
        # Build select columns
        select_columns = [
            self.id_column,
            self.text_column,
            *self.metadata_columns,
        ]

        # Start query builder
        if self.client is None:
            raise StorageError("Supabase client not initialized")
        query_builder = self.client.table(self.table_name).select(
            ", ".join(select_columns)
        )

        # Add limit early (before text_search)
        query_builder = query_builder.limit(limit)

        # Add namespace filter if provided
        if namespace:
            query_builder = query_builder.eq("namespace", namespace)

        # Add custom filters
        if filter:
            for key, value in filter.items():
                query_builder = query_builder.eq(key, value)

        # Add full-text search filter using textSearch
        # Note: text_search() must be called LAST before execute()
        # because it returns SyncQueryRequestBuilder which only has execute()
        # Strip surrounding quotes if present (from LLM-generated queries)
        clean_query = query.strip().strip('"').strip("'")
        query_builder = query_builder.text_search(
            column=self.search_vector_column,
            query=clean_query,
            options={"config": self.language, "type": "web_search"},
        )

        # Execute query
        response = query_builder.execute()

        if not response.data:
            return []

        # Convert to SearchResult
        search_results = []
        for row in response.data:
            # Extract ID
            chunk_id = str(row.get(self.id_column, ""))

            # Build metadata from selected columns
            metadata = {
                col: row.get(col)
                for col in self.metadata_columns
                if col in row
            }

            # Add text to metadata
            metadata[self.text_column] = row.get(self.text_column, "")

            search_results.append(
                SearchResult(
                    id=chunk_id,
                    score=None,  # Supabase doesn't expose ts_rank in simple queries
                    metadata=metadata,
                )
            )

        return search_results

    async def upsert(
        self,
        chunks: List[Dict[str, Any]],
        namespace: Optional[str] = None,
    ) -> None:
        """
        Upsert chunks to Supabase FTS table for keyword search.

        Args:
            chunks: List of chunk dicts with keys:
                - id: Chunk ID
                - text: Text content to index
                - documentId: Parent document ID
                - metadata: Additional metadata (optional)
            namespace: Optional namespace for multi-tenancy

        Note:
            This method inserts chunks into the Supabase table and the
            tsvector column is automatically updated by the database trigger
            (if configured as shown in class docstring).
        """
        try:
            if not chunks:
                return

            rows = []
            for chunk in chunks:
                # Build row for Supabase upsert
                row = {
                    self.id_column: chunk.get("id", ""),
                    self.text_column: chunk.get("text", ""),
                }

                # Add namespace if provided
                if namespace and "namespace" not in self.metadata_columns:
                    row["namespace"] = namespace

                # Add metadata columns if they exist in the chunk
                if "metadata" in chunk and isinstance(chunk["metadata"], dict):
                    for col in self.metadata_columns:
                        if col in chunk["metadata"]:
                            row[col] = chunk["metadata"][col]

                # Add documentId if it's in metadata_columns
                if "documentId" in chunk and "document_id" in self.metadata_columns:
                    row["document_id"] = chunk["documentId"]

                rows.append(row)

            # Bulk upsert to Supabase
            # on_conflict specifies which column to use for conflict resolution
            if self.client is None:
                raise StorageError("Supabase client not initialized")
            self.client.table(self.table_name).upsert(
                rows,
                on_conflict=self.id_column
            ).execute()

        except Exception as e:
            raise StorageError(
                f"Supabase keyword upsert failed: {str(e)}"
            ) from e
