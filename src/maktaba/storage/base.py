"""Base interface for vector stores - Aligned with Pinecone-style."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..models import SearchResult, VectorChunk


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
        namespace: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Search for similar vectors.

        Args:
            vector: Query embedding vector
            topK: Number of results to return (camelCase!)
            filter: Optional metadata filters (format depends on provider)
            includeMetadata: Whether to include metadata in results
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
