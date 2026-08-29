"""Voyage AI reranker with a deterministic offline fallback."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import List, Optional

import httpx

from ..models import SearchResult
from .base import BaseReranker

logger = logging.getLogger(__name__)
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class VoyageReranker(BaseReranker):
    """Rerank retrieval results with Voyage AI's multilingual reranker.

    API use is enabled by default when ``VOYAGE_API_KEY`` is present. Missing
    credentials, provider errors, and invalid responses fall back to a stable
    exact-token overlap ordering so retrieval remains available.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "rerank-2.5",
        base_url: str = "https://api.voyageai.com/v1/rerank",
        use_api: bool = True,
        timeout_s: float = 20.0,
        max_retries: int = 2,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("VOYAGE_API_KEY", "")
        self.model = model
        self.base_url = base_url
        self.use_api = use_api and bool(self.api_key)
        self.timeout_s = timeout_s
        self.max_retries = max(max_retries, 0)
        self._http_client = http_client

    async def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None,
    ) -> List[SearchResult]:
        if not results:
            return results

        k = min(max(top_k or len(results), 1), len(results))
        if not self.use_api:
            return self._heuristic_rerank(query, results, k)

        owns_client = self._http_client is None
        client = self._http_client or httpx.AsyncClient(timeout=self.timeout_s)
        try:
            response: Optional[httpx.Response] = None
            for attempt in range(self.max_retries + 1):
                response = await client.post(
                    self.base_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "query": query,
                        "documents": [result.text or "" for result in results],
                        "top_k": k,
                        "return_documents": False,
                        "truncation": True,
                    },
                )
                if (
                    response.status_code not in _RETRYABLE_STATUS_CODES
                    or attempt >= self.max_retries
                ):
                    break
                retry_after = response.headers.get("Retry-After")
                try:
                    delay = float(retry_after) if retry_after is not None else 2**attempt
                except ValueError:
                    delay = 2**attempt
                await asyncio.sleep(max(delay, 0.0))

            if response is None:
                raise RuntimeError("Voyage returned no response")
            response.raise_for_status()
            payload = response.json()
            items = payload.get("data", payload.get("results"))
            if not isinstance(items, list) or not items:
                raise ValueError("Voyage response is missing ranking data")

            ranked: List[SearchResult] = []
            seen_indices = set()
            for item in items:
                if not isinstance(item, dict):
                    raise ValueError("Voyage returned an invalid ranking item")
                index = item.get("index")
                if (
                    isinstance(index, bool)
                    or not isinstance(index, int)
                    or not 0 <= index < len(results)
                    or index in seen_indices
                ):
                    raise ValueError("Voyage returned an invalid candidate index")
                seen_indices.add(index)
                ranked.append(results[index])
            return ranked[:k]
        except (httpx.HTTPError, RuntimeError, ValueError) as exc:
            logger.warning("Voyage reranking failed; using deterministic fallback: %s", exc)
            return self._heuristic_rerank(query, results, k)
        finally:
            if owns_client:
                await client.aclose()

    def _heuristic_rerank(
        self,
        query: str,
        results: List[SearchResult],
        k: int,
    ) -> List[SearchResult]:
        """Return a stable exact-token overlap ordering when the API is unavailable."""

        query_tokens = _tokenize(query)
        if not query_tokens:
            return results[:k]

        def score(result: SearchResult) -> int:
            document_tokens = set(_tokenize(result.text or ""))
            return sum(token in document_tokens for token in query_tokens)

        return sorted(results, key=score, reverse=True)[:k]


def _tokenize(value: str) -> List[str]:
    return [token for token in value.casefold().split() if token]
