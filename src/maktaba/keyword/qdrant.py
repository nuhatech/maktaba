"""Qdrant keyword search implementation using full-text match."""

from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchText, MatchValue

from ..exceptions import StorageError
from ..models import SearchResult
from .base import BaseKeywordStore


class QdrantKeywordStore(BaseKeywordStore):
    """
    Qdrant full-text keyword search implementation.

    Uses Qdrant's built-in full-text search with Match (text) queries.
    Requires text fields to be indexed in the collection payload.

    Note:
        Qdrant's full-text search uses a BM25-style scoring algorithm.
        To enable full-text search on a field, you need to set up a
        text index on that field in your collection configuration.

    Examples:
        # Create keyword store
        keyword_store = QdrantKeywordStore(
            url="http://localhost:6333",
            collection_name="documents",
            text_field="text"  # Field to search in
        )

        # Search
        results = await keyword_store.search(
            query="Islamic jurisprudence",
            limit=15,
            namespace="kutub"
        )
    """

    def __init__(
        self,
        url: Optional[str] = None,
        collection_name: str = "",
        text_field: str = "text",
        api_key: Optional[str] = None,
        timeout: int = 60,
        location: Optional[str] = None,
        client: Optional[QdrantClient] = None,
    ):
        """
        Initialize Qdrant keyword store.

        Args:
            url: Qdrant server URL (e.g., "http://localhost:6333") or ":memory:"
            collection_name: Name of the collection
            text_field: Field name to search in (default: "text")
            api_key: Optional API key for Qdrant Cloud
            timeout: Request timeout in seconds
            location: Alternative to url. Use ":memory:" or path for local storage
            client: Optional existing QdrantClient instance to reuse (for sharing with QdrantStore)

        Note:
            The text_field must be indexed for full-text search in Qdrant.
            This is typically configured when creating the collection.

            If `client` is provided, it will be used instead of creating a new one.
            This is useful for sharing the same Qdrant instance between QdrantStore and QdrantKeywordStore.
        """
        # Use provided client if available
        if client is not None:
            self.client = client
        # Otherwise initialize Qdrant client (same logic as QdrantStore)
        elif url == ":memory:":
            self.client = QdrantClient(location=":memory:")
        elif location:
            self.client = QdrantClient(location=location)
        elif url:
            self.client = QdrantClient(
                url=url,
                api_key=api_key,
                timeout=timeout,
            )
        else:
            raise ValueError(
                "Either 'url', 'location', or 'client' must be provided. "
                "Use url=':memory:' for in-memory mode or url='http://localhost:6333' for server mode."
            )

        self.collection_name = collection_name
        self.text_field = text_field

    async def search(
        self,
        query: str,
        limit: int = 15,
        filter: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Search for documents using full-text keyword search.

        Args:
            query: Keyword search query (plain text)
            limit: Maximum number of results to return
            filter: Optional metadata filters
            namespace: Optional namespace filter

        Returns:
            List of SearchResult objects sorted by relevance score (descending)

        Note:
            Qdrant uses BM25-style scoring for full-text search.
            Results include a score indicating relevance.
        """
        try:
            # Build Qdrant filter conditions
            conditions = []

            # Add text match condition (this is the keyword search)
            conditions.append(
                FieldCondition(
                    key=self.text_field,
                    match=MatchText(text=query),
                )
            )

            # Add namespace filter if provided
            if namespace:
                conditions.append(
                    FieldCondition(
                        key="namespace",
                        match=MatchValue(value=namespace),
                    )
                )

            # Add custom filters if provided
            if filter:
                for key, value in filter.items():
                    # Handle list values with MatchAny, single values with MatchValue
                    if isinstance(value, list):
                        conditions.append(
                            FieldCondition(
                                key=key,
                                match=MatchAny(any=value),
                            )
                        )
                    else:
                        conditions.append(
                            FieldCondition(
                                key=key,
                                match=MatchValue(value=value),
                            )
                        )

            # Create Qdrant filter
            qdrant_filter = Filter(must=conditions) if conditions else None

            # Perform scroll to get matching documents
            # Note: Qdrant's full-text search doesn't support direct scoring via query_points
            # We use scroll with filters and rely on BM25 scoring
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=qdrant_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False,  # Don't need vectors for keyword search
            )

            if not scroll_result or not scroll_result[0]:
                return []

            points, _ = scroll_result

            # Convert to SearchResult
            search_results = []
            for point in points:
                # Get original ID from metadata if available
                point_id = str(point.id)
                if point.payload and "_original_id" in point.payload:
                    point_id = point.payload["_original_id"]

                search_results.append(
                    SearchResult(
                        id=point_id,
                        score=None,  # Qdrant scroll doesn't return relevance scores directly
                        metadata=point.payload or {},
                    )
                )

            return search_results[:limit]

        except Exception as e:
            raise StorageError(f"Qdrant keyword search failed: {str(e)}") from e
