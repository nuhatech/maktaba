"""Cohere-based reranker with offline fallback."""

import os
from typing import List, Optional

try:
    import httpx
except Exception:  # pragma: no cover - optional dependency
    httpx = None  # type: ignore

from ..models import SearchResult
from .base import BaseReranker


class CohereReranker(BaseReranker):
    """
    Cohere reranker wrapper.

    - If use_api=True and httpx is available, calls Cohere's rerank endpoint.
    - Otherwise falls back to a simple keyword-overlap heuristic.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "rerank-3.5",
        base_url: str = "https://api.cohere.ai/v1/rerank",
        use_api: bool = False,
        timeout_s: float = 20.0,
    ) -> None:
        self.api_key = api_key or os.getenv("COHERE_API_KEY", "")
        self.model = model
        self.base_url = base_url
        self.use_api = use_api and bool(self.api_key) and httpx is not None
        self.timeout_s = timeout_s

    async def rerank(
        self, query: str, results: List[SearchResult], top_k: Optional[int] = None
    ) -> List[SearchResult]:
        if not results:
            return results

        k = top_k or len(results)

        if self.use_api:
            try:
                docs = [r.text or "" for r in results]
                async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                    resp = await client.post(
                        self.base_url,
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": self.model,
                            "query": query,
                            "documents": docs,
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    # Cohere returns ranks with indices
                    indices = sorted(
                        [(item["index"], item.get("relevance_score", 0.0)) for item in data.get("results", [])],
                        key=lambda x: x[1],
                        reverse=True,
                    )
                    ordered = [results[i] for i, _ in indices if 0 <= i < len(results)]
                    return ordered[:k] if ordered else results[:k]
            except Exception:
                # Fall through to heuristic on failure
                pass

        # Offline heuristic: simple keyword overlap count
        q_tokens = _tokenize(query)

        def score(res: SearchResult) -> int:
            text = (res.text or "").lower()
            return sum(1 for t in q_tokens if t in text)

        ranked = sorted(results, key=score, reverse=True)
        return ranked[:k]


def _tokenize(s: str) -> List[str]:
    return [t for t in s.lower().split() if t]
