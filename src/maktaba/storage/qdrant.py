"""Qdrant vector store implementation - Pinecone-compatible interface."""

import uuid
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from ..exceptions import StorageError
from ..models import SearchResult, VectorChunk
from .base import BaseVectorStore

# UUID namespace for generating deterministic UUIDs from original IDs
_MAKTABA_UUID_NAMESPACE = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')


def _original_id_to_uuid(original_id: str) -> str:
    """
    Convert original chunk ID format to deterministic UUID v5.

    This allows Qdrant in-memory/local modes (which require UUIDs) to work
    while preserving the original {doc_id}#{chunk_id} format in metadata.

    Args:
        original_id: Original format (e.g., "doc_001#chunk_0")

    Returns:
        UUID string (e.g., "a1b2c3d4-e5f6-...")

    Example:
        >>> _original_id_to_uuid("doc_001#chunk_0")
        'a7f3e8c2-1b4d-5a9f-8c3e-1d2f4a6b8c0e'
    """
    return str(uuid.uuid5(_MAKTABA_UUID_NAMESPACE, original_id))


class QdrantStore(BaseVectorStore):
    """
    Qdrant vector storage implementation.

    Interface matches Pinecone for compatibility:
    - camelCase parameters (topK, includeMetadata)
    - Namespace support via metadata filtering
    - Chunk ID format: {doc_id}#{chunk_id}

    Examples:
        # In-memory mode (no Docker needed, great for testing)
        store = QdrantStore(
            url=":memory:",
            collection_name="test"
        )

        # Server mode (requires Qdrant running)
        store = QdrantStore(
            url="http://localhost:6333",
            collection_name="documents"
        )

        await store.upsert(chunks)
        results = await store.query(vector, topK=10)
    """

    def __init__(
        self,
        url: Optional[str] = None,
        collection_name: str = "",
        api_key: Optional[str] = None,
        timeout: int = 60,
        location: Optional[str] = None,
    ):
        """
        Initialize Qdrant store.

        Args:
            url: Qdrant server URL (e.g., "http://localhost:6333") or ":memory:" for in-memory mode
            collection_name: Name of the collection
            api_key: Optional API key for Qdrant Cloud
            timeout: Request timeout in seconds
            location: Alternative to url. Use ":memory:" for in-memory mode or path for local storage

        Examples:
            # In-memory mode (no Docker needed)
            store = QdrantStore(url=":memory:", collection_name="test")
            store = QdrantStore(location=":memory:", collection_name="test")

            # Server mode
            store = QdrantStore(url="http://localhost:6333", collection_name="prod")

            # Local persistent storage
            store = QdrantStore(location="./qdrant_data", collection_name="local")
        """
        # Auto-detect in-memory mode from url parameter
        if url == ":memory:":
            self.client = QdrantClient(location=":memory:")
            self._use_uuid = True  # In-memory mode requires UUIDs
        elif location:
            self.client = QdrantClient(location=location)
            self._use_uuid = True  # Local mode requires UUIDs
        elif url:
            self.client = QdrantClient(
                url=url,
                api_key=api_key,
                timeout=timeout,
            )
            self._use_uuid = True  # Server mode also requires UUIDs (or unsigned integers)
        else:
            raise ValueError(
                "Either 'url' or 'location' must be provided. "
                "Use url=':memory:' for in-memory mode or url='http://localhost:6333' for server mode."
            )

        self.collection_name = collection_name

    async def upsert(
        self,
        chunks: List[VectorChunk],
        namespace: Optional[str] = None,
    ) -> None:
        """
        Insert or update vector chunks.

        Args:
            chunks: List of VectorChunk objects
            namespace: Optional namespace (stored in metadata)

        Note:
            When using in-memory or local mode, the original chunk ID format
            (e.g., "doc_001#chunk_0") is preserved in metadata["_original_id"],
            while the point ID is converted to UUID for Qdrant compatibility.
        """
        if not chunks:
            return

        try:
            points = []
            for chunk in chunks:
                payload = dict(chunk.metadata)

                # Preserve original chunk ID format in metadata
                if self._use_uuid:
                    payload["_original_id"] = chunk.id
                    point_id = _original_id_to_uuid(chunk.id)
                else:
                    # Server mode: use string ID directly
                    point_id = chunk.id

                # Add namespace to payload if provided
                if namespace:
                    payload["namespace"] = namespace

                points.append(
                    PointStruct(
                        id=point_id,
                        vector=chunk.vector,
                        payload=payload,
                    )
                )

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

        except Exception as e:
            raise StorageError(f"Qdrant upsert failed: {str(e)}") from e

    async def query(
        self,
        vector: List[float],
        topK: int = 10,  # camelCase!
        filter: Optional[Dict[str, Any]] = None,
        includeMetadata: bool = True,
        namespace: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Search for similar vectors.

        Args:
            vector: Query embedding vector
            topK: Number of results (camelCase to match Pinecone)
            filter: Optional metadata filters
            includeMetadata: Whether to include metadata
            namespace: Optional namespace filter

        Returns:
            List of SearchResult objects sorted by score (descending)
        """
        try:
            # Build Qdrant filter
            qdrant_filter = None
            if filter or namespace:
                conditions = []

                # Add namespace filter
                if namespace:
                    conditions.append(
                        FieldCondition(
                            key="namespace",
                            match=MatchValue(value=namespace),
                        )
                    )

                # Add custom filters
                if filter:
                    for key, value in filter.items():
                        conditions.append(
                            FieldCondition(
                                key=key,
                                match=MatchValue(value=value),
                            )
                        )

                if conditions:
                    qdrant_filter = Filter(must=conditions)

            # Query (using modern query_points API)
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=vector,
                limit=topK,
                query_filter=qdrant_filter,
                with_payload=includeMetadata,
            )
            results = response.points

            # Convert to SearchResult
            search_results = []
            for result in results:
                # Return original ID from metadata if available (UUID mode)
                result_id = str(result.id)
                if self._use_uuid and result.payload:
                    result_id = result.payload.get("_original_id", result_id)

                search_results.append(
                    SearchResult(
                        id=result_id,
                        score=result.score,
                        metadata=result.payload if includeMetadata else {},
                    )
                )

            return search_results

        except Exception as e:
            raise StorageError(f"Qdrant query failed: {str(e)}") from e

    async def delete(
        self,
        ids: List[str],
        namespace: Optional[str] = None,
    ) -> None:
        """Delete vectors by ID."""
        if not ids:
            return

        try:
            # Convert original IDs to UUIDs if in UUID mode
            delete_ids = [_original_id_to_uuid(id) for id in ids] if self._use_uuid else ids

            # If namespace is provided, filter by namespace
            if namespace:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=Filter(
                        must=[
                            FieldCondition(
                                key="namespace",
                                match=MatchValue(value=namespace),
                            ),
                            FieldCondition(
                                key="id",
                                match=MatchValue(value=delete_ids),
                            ),
                        ]
                    ),
                )
            else:
                # Direct ID deletion
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=delete_ids,
                )

        except Exception as e:
            raise StorageError(f"Qdrant delete failed: {str(e)}") from e

    async def delete_by_document(
        self,
        document_id: str,
        namespace: Optional[str] = None,
    ) -> None:
        """
        Delete all chunks for a document.

        Uses scroll + delete. For in-memory/local mode, checks _original_id metadata.
        For server mode, uses ID prefix matching.
        """
        try:
            # Build filter
            conditions = []
            if namespace:
                conditions.append(
                    FieldCondition(
                        key="namespace",
                        match=MatchValue(value=namespace),
                    )
                )

            # Scroll through all points and filter by document ID
            chunk_ids = []
            offset = None

            while True:
                scroll_result = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=Filter(must=conditions) if conditions else None,
                    limit=100,
                    offset=offset,
                    with_payload=self._use_uuid,  # Need payload to check _original_id
                )

                if not scroll_result or not scroll_result[0]:
                    break

                points, offset = scroll_result

                # Filter by document ID
                for point in points:
                    point_id = str(point.id)

                    # Check using _original_id metadata (UUID mode)
                    if self._use_uuid and point.payload:
                        original_id = point.payload.get("_original_id", "")
                        if original_id.startswith(f"{document_id}#"):
                            # Append original ID (delete method will convert to UUID)
                            chunk_ids.append(original_id)
                    # Direct ID prefix match (non-UUID mode - should not happen now)
                    elif point_id.startswith(f"{document_id}#"):
                        chunk_ids.append(point_id)

                if offset is None:
                    break

            # Delete collected IDs
            if chunk_ids:
                await self.delete(chunk_ids, namespace=namespace)

        except Exception as e:
            raise StorageError(
                f"Qdrant delete_by_document failed: {str(e)}"
            ) from e

    async def list(
        self,
        prefix: Optional[str] = None,
        limit: int = 100,
        namespace: Optional[str] = None,
    ) -> List[str]:
        """
        List chunk IDs.

        Returns original chunk IDs from metadata when in UUID mode.
        """
        try:
            conditions = []
            if namespace:
                conditions.append(
                    FieldCondition(
                        key="namespace",
                        match=MatchValue(value=namespace),
                    )
                )

            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(must=conditions) if conditions else None,
                limit=limit,
                with_payload=self._use_uuid,  # Need payload for _original_id
            )

            if not scroll_result or not scroll_result[0]:
                return []

            points, _ = scroll_result

            # Filter by prefix if provided
            ids = []
            for point in points:
                # Get original chunk ID if available
                if self._use_uuid and point.payload:
                    point_id = point.payload.get("_original_id", str(point.id))
                else:
                    point_id = str(point.id)

                if prefix is None or point_id.startswith(prefix):
                    ids.append(point_id)

            return ids[:limit]

        except Exception as e:
            raise StorageError(f"Qdrant list failed: {str(e)}") from e

    async def get_dimensions(self) -> int:
        """Get vector dimension from collection info."""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            vector_size = collection_info.config.params.vectors.size
            return vector_size

        except Exception as e:
            raise StorageError(
                f"Failed to get dimensions: {str(e)}"
            ) from e

    def create_collection(
        self,
        dimension: int = 3072,  # default dimension (text-embedding-3-large)
        distance: Distance = Distance.COSINE,
    ) -> None:
        """
        Create collection if it doesn't exist.

        Args:
            dimension: Vector dimension (default: 3072 for text-embedding-3-large)
            distance: Distance metric (default: COSINE)
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)

            if not exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=dimension,
                        distance=distance,
                    ),
                )

        except Exception as e:
            raise StorageError(f"Failed to create collection: {str(e)}") from e
