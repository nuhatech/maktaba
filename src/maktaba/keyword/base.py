"""Base interface for keyword search stores."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..models import SearchResult


class BaseKeywordStore(ABC):
    """
    Abstract base class for keyword search providers.

    Provides full-text search capabilities separate from vector search.
    Implementations can use various backends:
    - Qdrant full-text match
    - PostgreSQL FTS (ts_rank)
    - Azure Cognitive Search
    - Elasticsearch
    - etc.
    """

    @abstractmethod
    async def search(
        self,
        query: str,
        limit: int = 15,
        filter: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Search for documents matching the keyword query.

        Args:
            query: Keyword search query (plain text, not embedding)
            limit: Maximum number of results to return
            filter: Optional metadata filters (format depends on provider)
            namespace: Optional namespace for multi-tenancy

        Returns:
            List of SearchResult objects, sorted by relevance score (descending)

        Note:
            The scoring mechanism depends on the backend:
            - Qdrant: BM25-style scoring
            - PostgreSQL: ts_rank() scoring
            - Azure: Hybrid search scores
        """
        pass

    @abstractmethod
    async def upsert(
        self,
        chunks: List[Dict[str, Any]],
        namespace: Optional[str] = None,
    ) -> None:
        """
        Insert or update chunks in the keyword search store.

        Args:
            chunks: List of chunk dictionaries with keys:
                - id (str): Unique chunk identifier
                - text (str): Text content to index for keyword search
                - documentId (str): Parent document identifier
                - metadata (Dict[str, Any], optional): Additional metadata
            namespace: Optional namespace for multi-tenancy

        Note:
            Implementations should:
            - Index the 'text' field for full-text search
            - Store chunk ID for retrieval
            - Handle duplicates (upsert semantics)
            - Support batch operations for efficiency

        Examples:
            await keyword_store.upsert(
                chunks=[
                    {
                        "id": "book_123#chunk_0",
                        "text": "Chapter 1: Introduction to Islamic Jurisprudence...",
                        "documentId": "book_123",
                        "metadata": {"page": 1, "chapter": 1},
                    }
                ],
                namespace="kutub"
            )
        """
        pass

