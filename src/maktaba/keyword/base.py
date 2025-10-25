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
