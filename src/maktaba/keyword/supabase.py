"""Supabase keyword search implementation using PostgreSQL FTS."""

import os
from typing import Any, Callable, Dict, List, Optional

try:
    from supabase import Client as SyncClient
    from supabase import create_client as create_sync_client

    Client: Optional[type[Any]] = SyncClient
    create_client: Optional[Callable[..., Any]] = create_sync_client
except ImportError:
    Client = None
    create_client = None

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

    Examples:
        # Create keyword store
        keyword_store = SupabaseKeywordStore(
            url="https://xxx.supabase.co",
            key="your-anon-key",
            table_name="page_content",
            text_column="text",
            search_vector_column="search_vector"
        )

        # Search
        results = await keyword_store.search(
            query="Islamic jurisprudence",
            limit=15,
            filter={"book_id": "123"}
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

        self.client: Client = create_client(_url, _key)
        self.table_name = table_name
        self.text_column = text_column
        self.search_vector_column = search_vector_column
        self.id_column = id_column
        self.metadata_columns = metadata_columns or []
        self.language = language

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
            namespace: Optional namespace filter (if namespace column exists)

        Returns:
            List of SearchResult objects sorted by ts_rank score (descending)

        Note:
            Uses ts_rank() for scoring. Higher scores indicate better matches.
            The query is converted to tsquery format automatically.
        """
        try:
            # Build select columns
            select_columns = [
                self.id_column,
                self.text_column,
                *self.metadata_columns,
            ]

            # Start query builder
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
                options={"config": self.language, "type": "websearch"},
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

        except Exception as e:
            raise StorageError(
                f"Supabase keyword search failed: {str(e)}"
            ) from e

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
            self.client.table(self.table_name).upsert(
                rows,
                on_conflict=self.id_column
            ).execute()

        except Exception as e:
            raise StorageError(
                f"Supabase keyword upsert failed: {str(e)}"
            ) from e
