"""ZeroEntropy Zerank reranker implementation."""

import os
from typing import List, Optional

from ..models import SearchResult
from .base import BaseReranker


class ZerankReranker(BaseReranker):
    """
    ZeroEntropy Zerank reranker wrapper.

    - If use_api=True and zeroentropy is available, calls Zerank API.
    - Otherwise falls back to simple keyword-overlap heuristic.

    Requires:
        - pip install maktaba[zeroentropy]
        - Environment variable: ZEROENTROPY_API_KEY

    Example:
        >>> from maktaba.reranking import ZerankReranker
        >>> reranker = ZerankReranker(use_api=True)
        >>> ranked = await reranker.rerank(query="...", results=[...])
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "zerank-1",
        use_api: bool = True,
    ) -> None:
        """
        Initialize ZerankReranker.

        Args:
            api_key: ZeroEntropy API key. If None, reads from ZEROENTROPY_API_KEY env var.
            model: Model name (default: "zerank-1")
            use_api: Whether to use the API (vs. offline heuristic fallback)
        """
        self.api_key = api_key or os.getenv("ZEROENTROPY_API_KEY", "")
        self.model = model

        # Try to import zeroentropy SDK
        try:
            from zeroentropy import AsyncZeroEntropy

            self._AsyncZeroEntropy = AsyncZeroEntropy
            self._client = None  # Lazy initialization
        except Exception:  # pragma: no cover - optional dependency
            self._AsyncZeroEntropy = None  # type: ignore

        self.use_api = use_api and bool(self.api_key) and self._AsyncZeroEntropy is not None

    async def rerank(
        self, query: str, results: List[SearchResult], top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Rerank search results using ZeroEntropy Zerank API.

        Args:
            query: User query text
            results: Initial results from the vector store
            top_k: Optional cap on results after reranking

        Returns:
            Reranked list of SearchResult objects
        """
        if not results:
            return results

        k = top_k or len(results)

        if self.use_api:
            try:
                # Lazy client initialization
                if self._client is None:
                    if self._AsyncZeroEntropy is None:
                        # Fallback if SDK not available
                        return self._heuristic_rerank(query, results, k)
                    self._client = self._AsyncZeroEntropy(api_key=self.api_key)

                # Extract documents from SearchResult
                docs = [r.text or "" for r in results]

                # Call Zerank API
                response = await self._client.models.rerank(
                    model=self.model,
                    query=query,
                    documents=docs,
                )

                # Process response - map reranked results back to SearchResult objects
                # The API returns results in ranked order
                reranked = []
                for item in response.results[:k]:
                    # ZeroEntropy returns documents with their original index
                    # Get the index to map back to our SearchResult list
                    idx = getattr(item, "index", None)
                    if idx is not None and 0 <= idx < len(results):
                        reranked.append(results[idx])

                # If we got valid results, return them
                if reranked:
                    return reranked

                # Otherwise fall through to heuristic
            except Exception:  # pragma: no cover - network errors
                # Fall through to heuristic on failure
                pass

        # Offline heuristic fallback: simple keyword overlap count
        return self._heuristic_rerank(query, results, k)

    def _heuristic_rerank(
        self, query: str, results: List[SearchResult], k: int
    ) -> List[SearchResult]:
        """
        Fallback reranking using simple keyword overlap heuristic.

        Args:
            query: User query text
            results: Results to rerank
            k: Number of results to return

        Returns:
            Reranked results based on keyword overlap
        """
        q_tokens = _tokenize(query)

        def score(res: SearchResult) -> int:
            text = (res.text or "").lower()
            return sum(1 for t in q_tokens if t in text)

        ranked = sorted(results, key=score, reverse=True)
        return ranked[:k]


def _tokenize(s: str) -> List[str]:
    """
    Simple tokenizer for heuristic fallback.

    Args:
        s: String to tokenize

    Returns:
        List of lowercase tokens
    """
    return [t for t in s.lower().split() if t]
