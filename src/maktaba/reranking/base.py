"""Reranking interfaces."""

from abc import ABC, abstractmethod
from typing import List, Optional

from ..models import SearchResult


class BaseReranker(ABC):
    """
    Abstract reranker interface.

    Provides a standard interface for post-retrieval reranking of search results.
    """

    @abstractmethod
    async def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None,
    ) -> List[SearchResult]:
        """
        Reorder results with relevance scores to the query.

        Args:
            query: User query text
            results: Initial results from the vector store
            top_k: Optional cap on results after reranking

        Returns:
            New list of SearchResult in ranked order
        """
        raise NotImplementedError
