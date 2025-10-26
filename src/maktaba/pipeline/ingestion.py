"""Ingestion pipeline: chunk -> embed -> upsert."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..chunking.base import BaseChunker
from ..embedding.base import BaseEmbedder
from ..keyword.base import BaseKeywordStore
from ..logging import get_logger
from ..models import NodeRelationship, VectorChunk
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
        keyword_store: Optional[BaseKeywordStore] = None,
        namespace: Optional[str] = None,
        batch_size: int = 64,
        on_progress: Optional[ProgressCallback] = None,
    ) -> None:
        self.chunker = chunker
        self.embedder = embedder
        self.store = store
        self.keyword_store = keyword_store
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

            # Also upsert to keyword store if provided (for agentic search)
            if self.keyword_store:
                keyword_chunks = [
                    {
                        "id": chunk.id,
                        "text": chunk.metadata.get("text", ""),
                        "documentId": document_id,
                        "metadata": chunk.metadata,
                    }
                    for chunk in chunks
                ]
                await self.keyword_store.upsert(keyword_chunks, namespace=self.namespace)

            stored += len(chunks)

        return IngestionResult(
            document_id=document_id,
            total_chunks=total,
            stored_chunks=stored,
            chunk_ids=all_ids,
            metadata={"namespace": self.namespace} if self.namespace else {},
        )

    async def ingest_text_pages(
        self,
        pages: List[Dict[str, Any]],
        *,
        document_id: str,
        namespace: Optional[str] = None,
    ) -> IngestionResult:
        """
        Ingest multiple pages with automatic NEXT/PREVIOUS relationships.

        Designed for texts where ideas span multiple consecutive pages.
        Creates bidirectional links between sequential pages for context expansion.

        Args:
            pages: List of page dicts with 'text' and metadata (book_id, page_from_url, etc.)
            document_id: Base document ID (e.g., "book_123")
            namespace: Optional namespace for multi-tenancy

        Returns:
            IngestionResult with all page chunk IDs

        Example:
            pages = [
                {"text": "...", "page_from_url": 1, "book_id": 123},
                {"text": "...", "page_from_url": 2, "book_id": 123},
                {"text": "...", "page_from_url": 3, "book_id": 123},
            ]
            # Creates: page_1 ← → page_2 ← → page_3
        """
        if not pages:
            return IngestionResult(
                document_id=document_id,
                total_chunks=0,
                stored_chunks=0,
                chunk_ids=[],
                metadata={},
            )

        self._logger.info(
            "ingest_text_pages.start: document_id=%s total_pages=%d",
            document_id,
            len(pages),
        )

        # Step 1: Embed all pages
        texts = [page.get("text", "") for page in pages]
        self._logger.info("embedding.pages: total=%d", len(texts))
        vectors = await self.embedder.embed_batch(texts, input_type="document")

        # Step 2: Build chunks with relationships
        chunks: List[VectorChunk] = []
        all_ids: List[str] = []

        for idx, (page, vector) in enumerate(zip(pages, vectors)):
            # Create unique chunk ID based on page_from_url
            page_from_url = page.get("page_from_url")
            if page_from_url is None:
                self._logger.warning(f"Page {idx} missing page_from_url, using index")
                page_from_url = idx

            chunk_id = f"{document_id}#page_{page_from_url}"

            # Build relationships (NEXT/PREVIOUS) using NodeRelationship objects
            relationships = {}
            if idx > 0:
                prev_page_url = pages[idx - 1].get("page_from_url", idx - 1)
                relationships["PREVIOUS"] = NodeRelationship(
                    node_id=f"{document_id}#page_{prev_page_url}",
                    node_type="1",  # TEXT type in LlamaIndex
                    metadata={},
                )

            if idx < len(pages) - 1:
                next_page_url = pages[idx + 1].get("page_from_url", idx + 1)
                relationships["NEXT"] = NodeRelationship(
                    node_id=f"{document_id}#page_{next_page_url}",
                    node_type="1",  # TEXT type in LlamaIndex
                    metadata={},
                )

            # Merge metadata
            metadata = {**page, "text": texts[idx]}

            # Create chunk with relationships
            chunk = VectorChunk(
                id=chunk_id,
                vector=vector,
                metadata=metadata,
                relationships=relationships if relationships else None,
            )
            chunks.append(chunk)
            all_ids.append(chunk_id)

        # Step 3: Upsert all chunks with relationships
        self._logger.info("upsert.pages: total=%d namespace=%s", len(chunks), namespace or self.namespace)
        await self.store.upsert(chunks, namespace=namespace or self.namespace)

        # Also upsert to keyword store if provided (for agentic search)
        if self.keyword_store:
            keyword_chunks = [
                {
                    "id": chunk.id,
                    "text": chunk.metadata.get("text", ""),
                    "documentId": document_id,
                    "metadata": chunk.metadata,
                }
                for chunk in chunks
            ]
            await self.keyword_store.upsert(keyword_chunks, namespace=namespace or self.namespace)

        self._logger.info(
            "ingest_text_pages.done: stored=%d chunks with relationships",
            len(chunks),
        )

        return IngestionResult(
            document_id=document_id,
            total_chunks=len(chunks),
            stored_chunks=len(chunks),
            chunk_ids=all_ids,
            metadata={"namespace": namespace or self.namespace} if namespace or self.namespace else {},
        )
