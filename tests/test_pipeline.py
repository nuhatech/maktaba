from typing import List

import pytest

from maktaba.embedding.base import BaseEmbedder
from maktaba.models import SearchResult
from maktaba.pipeline.query import QueryPipeline
from maktaba.reranking.cohere import CohereReranker
from maktaba.storage.base import BaseVectorStore


class DummyEmbedder(BaseEmbedder):
    async def embed_batch(self, texts: List[str], input_type: str = "document") -> List[List[float]]:
        return [[0.0, 0.0, 0.0] for _ in texts]

    @property
    def dimension(self) -> int:
        return 3

    @property
    def model(self) -> str:
        return "dummy"


class DummyStore(BaseVectorStore):
    async def upsert(self, chunks, namespace=None) -> None:
        return None

    async def query(self, vector, topK: int = 10, filter=None, includeMetadata: bool = True, namespace=None):
        # Return predictable 10 results with text present
        out = []
        for i in range(topK):
            meta = {"text": f"This is chunk {i} about Tawhid."}
            out.append(SearchResult(id=f"docA#{i}", score=1.0 - i * 0.01, metadata=meta))
        return out

    async def delete(self, ids, namespace=None):
        return None

    async def list(self, prefix=None, limit: int = 100, namespace=None):
        return []

    async def get_dimensions(self) -> int:
        return 3


@pytest.mark.asyncio
async def test_query_pipeline_success_metric():
    embedder = DummyEmbedder()
    store = DummyStore()
    reranker = CohereReranker(use_api=False)  # Offline heuristic
    pipeline = QueryPipeline(embedder, store, reranker)

    result = await pipeline.search("What is Tawhid?", rerank=True)

    assert isinstance(result["formatted_context"], str)
    assert "[1]:" in result["formatted_context"]
    assert len(result["citations"]) == 10
