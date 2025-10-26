from typing import Any, Dict, List, Optional

import pytest

from maktaba.chunking.base import BaseChunker
from maktaba.embedding.base import BaseEmbedder
from maktaba.models import NodeRelationship, SearchResult, VectorChunk
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
            self._points[c.id] = {
                "vector": c.vector,
                "metadata": dict(c.metadata),
                "ns": namespace,
                "relationships": c.relationships,
            }

    async def query(self, vector, topK: int = 10, filter=None, includeMetadata: bool = True, namespace=None, includeRelationships: bool = False):
        # cosine-like similarity on length only for determinism
        def score(pid: str) -> float:
            v = self._points[pid]["vector"]
            return 1.0 / (1.0 + abs(v[0] - vector[0]))

        ids = list(self._points.keys())
        ranked = sorted(ids, key=score, reverse=True)[:topK]
        out: List[SearchResult] = []
        for pid in ranked:
            meta = self._points[pid]["metadata"] if includeMetadata else {}
            rels = self._points[pid].get("relationships") if includeRelationships else None
            out.append(SearchResult(id=pid, score=score(pid), metadata=meta, relationships=rels))
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



# Test 17: ingest_text_pages with relationships
@pytest.mark.asyncio
async def test_ingest_text_pages_with_relationships():
    """Test that ingest_text_pages creates NodeRelationship objects for NEXT/PREVIOUS."""
    chunker = DummyChunker()
    embedder = DummyEmbedder()
    store = DummyStore()
    ingest = IngestionPipeline(chunker, embedder, store)

    pages = [
        {"page_from_url": 1, "text": "Page 1 content."},
        {"page_from_url": 2, "text": "Page 2 content."},
        {"page_from_url": 3, "text": "Page 3 content."},
    ]

    result = await ingest.ingest_text_pages(
        pages=pages,
        document_id="book_123",
    )

    assert result.total_chunks == 3
    assert result.stored_chunks == 3

    # Check that relationships were stored
    page1_chunks = await store.list(prefix="book_123#page_1")
    assert len(page1_chunks) > 0

    # Query with relationships enabled
    chunk_data = store._points[page1_chunks[0]]
    assert chunk_data["relationships"] is not None


# Test 18: NEXT/PREVIOUS relationships are NodeRelationship objects
@pytest.mark.asyncio
async def test_ingest_text_pages_next_previous_links():
    """Test that NEXT/PREVIOUS relationships have correct structure."""
    chunker = DummyChunker()
    embedder = DummyEmbedder()
    store = DummyStore()
    ingest = IngestionPipeline(chunker, embedder, store)

    pages = [
        {"page_from_url": 1, "text": "First page."},
        {"page_from_url": 2, "text": "Second page."},
    ]

    await ingest.ingest_text_pages(
        pages=pages,
        document_id="doc_456",
    )

    # Check page 1 chunk (should have NEXT, no PREVIOUS)
    page1_chunks = await store.list(prefix="doc_456#page_1")
    page1_data = store._points[page1_chunks[0]]

    assert "NEXT" in page1_data["relationships"]
    assert "PREVIOUS" not in page1_data["relationships"]

    next_rel = page1_data["relationships"]["NEXT"]
    assert isinstance(next_rel, NodeRelationship)
    assert next_rel.node_id == "doc_456#page_2"
    assert next_rel.node_type == "1"  # TEXT type

    # Check page 2 chunk (should have PREVIOUS, no NEXT)
    page2_chunks = await store.list(prefix="doc_456#page_2")
    page2_data = store._points[page2_chunks[0]]

    assert "PREVIOUS" in page2_data["relationships"]
    assert "NEXT" not in page2_data["relationships"]

    prev_rel = page2_data["relationships"]["PREVIOUS"]
    assert isinstance(prev_rel, NodeRelationship)
    assert prev_rel.node_id == "doc_456#page_1"
    assert prev_rel.node_type == "1"


# Test 19: Advanced chunking parameters in ingestion
@pytest.mark.asyncio
async def test_ingest_with_advanced_chunking_params():
    """Test that chunking parameters are passed through ingestion pipeline."""
    chunker = DummyChunker()
    embedder = DummyEmbedder()
    store = DummyStore()
    ingest = IngestionPipeline(chunker, embedder, store)

    # This test verifies the parameters are accepted by ingest_text
    # DummyChunker doesn't use them, but real chunker would
    text = "Test document for advanced parameters."

    result = await ingest.ingest_text(
        text,
        document_id="doc_789",
        filename="test.txt",
        chunking_kwargs={
            "overlap": 50,
            "max_characters": 200,
            "new_after_n_chars": 150,
        },
    )

    assert result.total_chunks > 0
    assert result.stored_chunks > 0


# Test 20: Relationship node_type is TEXT
@pytest.mark.asyncio
async def test_relationship_node_type_is_text():
    """Test that relationships use node_type='1' (TEXT) as per LlamaIndex standard."""
    chunker = DummyChunker()
    embedder = DummyEmbedder()
    store = DummyStore()
    ingest = IngestionPipeline(chunker, embedder, store)

    pages = [
        {"page_from_url": 1, "text": "Page one."},
        {"page_from_url": 2, "text": "Page two."},
    ]

    await ingest.ingest_text_pages(
        pages=pages,
        document_id="doc_text_type",
    )

    # Check that all relationships have node_type="1"
    all_chunks = await store.list()
    for chunk_id in all_chunks:
        chunk_data = store._points[chunk_id]
        if chunk_data["relationships"]:
            for rel_type, rel_obj in chunk_data["relationships"].items():
                if isinstance(rel_obj, NodeRelationship):
                    assert rel_obj.node_type == "1", f"Expected node_type='1' for {rel_type} relationship"
