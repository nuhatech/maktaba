from typing import Any, Dict, List, Optional

import pytest

from maktaba.chunking.base import BaseChunker
from maktaba.embedding.base import BaseEmbedder
from maktaba.models import SearchResult, VectorChunk
from maktaba.pipeline.ingestion import IngestionPipeline
from maktaba.pipeline.query import QueryPipeline
from maktaba.storage.base import BaseVectorStore


class DummyDoc:
    def __init__(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        self.text = text
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "metadata": self.metadata}


class DummyChunker(BaseChunker):
    async def chunk_text(self, text: str, filename: str = "doc.txt", extra_metadata=None, **kwargs):
        parts = [p.strip() for p in text.split(".") if p.strip()]
        docs = [DummyDoc(p + ".", (extra_metadata or {})) for p in parts]
        from maktaba.chunking.models import ChunkMetadata, ChunkResult

        return ChunkResult(
            documents=docs,
            metadata=ChunkMetadata(filename=filename, filetype="text/plain", size_in_bytes=len(text)),
            total_chunks=len(docs),
            total_characters=len(text),
        )

    async def chunk_file(self, file_path, extra_metadata=None, **kwargs):  # pragma: no cover
        raise NotImplementedError

    async def chunk_url(self, url: str, filename: str, extra_metadata=None, **kwargs):  # pragma: no cover
        raise NotImplementedError


class DummyEmbedder(BaseEmbedder):
    async def embed_batch(self, texts: List[str], input_type: str = "document") -> List[List[float]]:
        # Very simple embedding: length and first char code (stable for test)
        def emb(t: str) -> List[float]:
            return [float(len(t)), float(ord(t[0])) if t else 0.0, 0.0]

        return [emb(t) for t in texts]

    @property
    def dimension(self) -> int:
        return 3

    @property
    def model(self) -> str:
        return "dummy"


class DummyStore(BaseVectorStore):
    def __init__(self):
        self._points: Dict[str, Dict[str, Any]] = {}

    async def upsert(self, chunks: List[VectorChunk], namespace: Optional[str] = None) -> None:
        for c in chunks:
            self._points[c.id] = {"vector": c.vector, "metadata": dict(c.metadata), "ns": namespace}

    async def query(self, vector, topK: int = 10, filter=None, includeMetadata: bool = True, namespace=None):
        # cosine-like similarity on length only for determinism
        def score(pid: str) -> float:
            v = self._points[pid]["vector"]
            return 1.0 / (1.0 + abs(v[0] - vector[0]))

        ids = list(self._points.keys())
        ranked = sorted(ids, key=score, reverse=True)[:topK]
        out: List[SearchResult] = []
        for pid in ranked:
            meta = self._points[pid]["metadata"] if includeMetadata else {}
            out.append(SearchResult(id=pid, score=score(pid), metadata=meta))
        return out

    async def delete(self, ids, namespace=None):
        for i in ids:
            self._points.pop(i, None)

    async def list(self, prefix=None, limit: int = 100, namespace=None):
        ids = list(self._points.keys())
        return [i for i in ids if (prefix is None or i.startswith(prefix))][:limit]

    async def get_dimensions(self) -> int:
        return 3


@pytest.mark.asyncio
async def test_ingestion_pipeline_end_to_end_with_query_pipeline():
    # 1) Build ingestion deps
    chunker = DummyChunker()
    embedder = DummyEmbedder()
    store = DummyStore()
    ingest = IngestionPipeline(chunker, embedder, store)

    # 2) Ingest text document
    text = "Tawhid is the oneness of Allah. Zakat is charity. Salah is prayer."
    result = await ingest.ingest_text(text, document_id="book_123", filename="book.txt")
    assert result.total_chunks == 3
    assert result.stored_chunks == 3
    assert all(cid.startswith("book_123#chunk_") for cid in result.chunk_ids)

    # 3) Query via QueryPipeline (no reranker needed for this test)
    q = QueryPipeline(embedder, store, reranker=None)
    out = await q.search("What is Tawhid?", rerank=False, top_k=3)

    assert len(out["results"]) > 0
    assert "[1]:" in out["formatted_context"]

