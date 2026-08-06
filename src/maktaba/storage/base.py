"""Base interface for vector stores - Aligned with Pinecone-style."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..models import SearchResult, VectorChunk


def metadata_matches_filter(metadata: Dict[str, Any], filter: Optional[Dict[str, Any]]) -> bool:
    """Conservatively evaluate common metadata-filter syntax.

    This is used only when a provider cannot atomically fetch IDs with a
    server-side filter. Unknown operators fail closed so relationship expansion
    cannot escape the caller's retrieval scope.
    """
    if not filter:
        return True

    def _compare(actual: Any, condition: Any) -> bool:
        if not isinstance(condition, dict):
            return actual == condition or (isinstance(condition, list) and actual in condition)
        for operator, expected in condition.items():
            if operator == "$eq" and actual != expected:
                return False
            if operator == "$ne" and actual == expected:
                return False
            if operator == "$in" and (not isinstance(expected, list) or actual not in expected):
                return False
            if operator == "$nin" and (not isinstance(expected, list) or actual in expected):
                return False
            if operator == "$gt" and not (isinstance(actual, (int, float)) and actual > expected):
                return False
            if operator == "$gte" and not (isinstance(actual, (int, float)) and actual >= expected):
                return False
            if operator == "$lt" and not (isinstance(actual, (int, float)) and actual < expected):
                return False
            if operator == "$lte" and not (isinstance(actual, (int, float)) and actual <= expected):
                return False
            if operator not in {"$eq", "$ne", "$in", "$nin", "$gt", "$gte", "$lt", "$lte"}:
                return False
        return True

    for key, condition in filter.items():
        if key == "$and":
            if not isinstance(condition, list) or not all(
                isinstance(item, dict) and metadata_matches_filter(metadata, item) for item in condition
            ):
                return False
            continue
        if key == "$or":
            if not isinstance(condition, list) or not any(
                isinstance(item, dict) and metadata_matches_filter(metadata, item) for item in condition
            ):
                return False
            continue
        if key.startswith("$") or not _compare(metadata.get(key), condition):
            return False
    return True


class BaseVectorStore(ABC):
    """
    Abstract base class for vector storage providers.

    Interface design matches Pinecone for compatibility:
    - camelCase parameter names (topK, includeMetadata)
    - Namespace support for multi-tenancy
    - Batch operations by default
    - Metadata filtering
    """

    @abstractmethod
    async def upsert(
        self,
        chunks: List[VectorChunk],
        namespace: Optional[str] = None,
    ) -> None:
        """
        Insert or update vector chunks in the store.

        Args:
            chunks: List of VectorChunk objects to upsert
            namespace: Optional namespace for multi-tenancy (Pinecone-style)

        Raises:
            StorageError: If upsert operation fails
        """
        pass

    @abstractmethod
    async def query(
        self,
        vector: List[float],
        topK: int = 10,  # camelCase to match Pinecone
        filter: Optional[Dict[str, Any]] = None,
        includeMetadata: bool = True,
        includeRelationships: bool = False,
        namespace: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Search for similar vectors.

        Args:
            vector: Query embedding vector
            topK: Number of results to return (camelCase!)
            filter: Optional metadata filters (format depends on provider)
            includeMetadata: Whether to include metadata in results
            includeRelationships: Whether to include relationships (NEXT/PREVIOUS links)
            namespace: Optional namespace to search in

        Returns:
            List of SearchResult objects, sorted by similarity score (descending)

        Raises:
            StorageError: If query operation fails
        """
        pass

    @abstractmethod
    async def delete(
        self,
        ids: List[str],
        namespace: Optional[str] = None,
    ) -> None:
        """
        Delete vectors by ID.

        Args:
            ids: List of chunk IDs to delete (format: "{doc_id}#{chunk_id}")
            namespace: Optional namespace

        Raises:
            StorageError: If delete operation fails
        """
        pass

    async def delete_by_document(
        self,
        document_id: str,
        namespace: Optional[str] = None,
    ) -> None:
        """
        Delete all chunks belonging to a document.

        This is a convenience method that filters by document ID prefix.

        Args:
            document_id: Document ID
            namespace: Optional namespace
        """
        # Default implementation: providers can override for efficiency
        # Most providers support prefix filtering
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement delete_by_document"
        )

    async def fetch_by_ids(
        self,
        ids: List[str],
        *,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
        includeMetadata: bool = True,
        includeRelationships: bool = False,
    ) -> List[SearchResult]:
        """Fetch chunks by ID without weakening namespace or metadata scope.

        The optional default keeps existing custom stores source compatible.
        Returning an empty list signals that direct expansion is unsupported.
        """
        return []

    @abstractmethod
    async def list(
        self,
        prefix: Optional[str] = None,
        limit: int = 100,
        namespace: Optional[str] = None,
    ) -> List[str]:
        """
        List chunk IDs in the store.

        Args:
            prefix: Optional prefix filter (e.g., document ID)
            limit: Maximum number of IDs to return
            namespace: Optional namespace

        Returns:
            List of chunk IDs

        Raises:
            StorageError: If list operation fails
        """
        pass

    @abstractmethod
    async def get_dimensions(self) -> int:
        """
        Get the dimension of vectors in this store.

        Returns:
            Vector dimension (e.g., 3072 for text-embedding-3-large)

        Raises:
            StorageError: If unable to determine dimensions
        """
        pass
