"""Ingestion pipeline: chunk -> embed -> upsert."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..chunking.base import BaseChunker
from ..embedding.base import BaseEmbedder
from ..logging import get_logger
from ..models import VectorChunk
from ..storage.base import BaseVectorStore

ProgressCallback = Callable[[Dict[str, Any]], None]


@dataclass
class IngestionResult:
    document_id: str
    total_chunks: int
    stored_chunks: int
    chunk_ids: List[str]
    metadata: Dict[str, Any]


class IngestionPipeline:
    """
    End-to-end ingestion: chunk a document, embed chunks, and upsert to the store.

    Performs document chunking, embedding, and vector store upsert in a provider-agnostic manner.
    """

    def __init__(
        self,
        chunker: BaseChunker,
        embedder: BaseEmbedder,
        store: BaseVectorStore,
        *,
        namespace: Optional[str] = None,
        batch_size: int = 64,
        on_progress: Optional[ProgressCallback] = None,
    ) -> None:
        self.chunker = chunker
        self.embedder = embedder
        self.store = store
        self.namespace = namespace
        self.batch_size = max(1, batch_size)
        self.on_progress = on_progress
        self._logger = get_logger("maktaba.pipeline.ingestion")

    async def ingest_text(
        self,
        text: str,
        *,
        document_id: str,
        filename: str = "document.txt",
        extra_metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> IngestionResult:
        self._logger.info(
            "ingest_text.start: document_id=%s filename=%s chars=%d",
            document_id,
            filename,
            len(text or ""),
        )
        chunks = await self.chunker.chunk_text(
            text=text, filename=filename, extra_metadata=extra_metadata, **kwargs
        )
        return await self._embed_and_store(chunks.documents, document_id, extra_metadata or {})

    async def ingest_file(
        self,
        file_path: Path | str,
        *,
        document_id: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> IngestionResult:
        self._logger.info("ingest_file.start: document_id=%s path=%s", document_id, file_path)
        chunks = await self.chunker.chunk_file(
            file_path=file_path, extra_metadata=extra_metadata, **kwargs
        )
        return await self._embed_and_store(chunks.documents, document_id, extra_metadata or {})

    async def ingest_url(
        self,
        url: str,
        *,
        filename: str,
        document_id: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> IngestionResult:
        self._logger.info(
            "ingest_url.start: document_id=%s url=%s filename=%s",
            document_id,
            url,
            filename,
        )
        chunks = await self.chunker.chunk_url(
            url=url, filename=filename, extra_metadata=extra_metadata, **kwargs
        )
        return await self._embed_and_store(chunks.documents, document_id, extra_metadata or {})

    async def _embed_and_store(
        self, documents: Sequence[Any], document_id: str, extra_metadata: Dict[str, Any]
    ) -> IngestionResult:
        texts: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for doc in documents:
            text = getattr(doc, "text", None)
            if not isinstance(text, str):
                continue
            md = getattr(doc, "metadata", {})
            md = md if isinstance(md, dict) else {}
            # Normalize into our metadata format
            merged = {**md, **extra_metadata, "text": text}
            texts.append(text)
            metadatas.append(merged)

        total = len(texts)
        stored = 0
        all_ids: List[str] = []

        # Embed in batches as 'document' inputs
        self._logger.info("ingest.embed_store: total_chunks=%d batch_size=%d", total, self.batch_size)
        for start in range(0, total, self.batch_size):
            end = min(start + self.batch_size, total)
            batch_texts = texts[start:end]
            batch_meta = metadatas[start:end]

            if self.on_progress:
                self.on_progress({
                    "stage": "embedding",
                    "start": start,
                    "end": end,
                    "total": total,
                })
            self._logger.info("embedding.batch: start=%d end=%d", start, end)

            vectors = await self.embedder.embed_batch(batch_texts, input_type="document")

            chunks: List[VectorChunk] = []
            for i, (vec, meta) in enumerate(zip(vectors, batch_meta)):
                global_idx = start + i
                chunk_id = f"{document_id}#chunk_{global_idx}"
                chunks.append(VectorChunk(id=chunk_id, vector=vec, metadata=meta))
                all_ids.append(chunk_id)

            if self.on_progress:
                self.on_progress({
                    "stage": "upsert",
                    "start": start,
                    "end": end,
                    "total": total,
                })
            self._logger.info("upsert.batch: start=%d end=%d", start, end)
            await self.store.upsert(chunks, namespace=self.namespace)
            stored += len(chunks)

        return IngestionResult(
            document_id=document_id,
            total_chunks=total,
            stored_chunks=stored,
            chunk_ids=all_ids,
            metadata={"namespace": self.namespace} if self.namespace else {},
        )
