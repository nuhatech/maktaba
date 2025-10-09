"""Unstructured document chunker - extracted from partition-api."""

from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import httpx

from ..exceptions import ChunkingError
from .base import BaseChunker
from .models import ChunkMetadata, ChunkResult


class UnstructuredChunker(BaseChunker):
    """
    Document chunker using Unstructured.io library via LlamaIndex.

    This implementation extracts the core logic from partition-api (main.py)
    and integrates it directly into Maktaba for better performance and simplicity.

    Supports:
    - Multiple file formats (PDF, DOCX, TXT, HTML, etc.)
    - Different parsing strategies (auto, fast, hi_res, ocr_only)
    - Different chunking strategies (basic, by_title)
    - Automatic file type detection
    - Page number extraction (for PDFs)

    """

    def __init__(
        self,
        strategy: Literal["auto", "fast", "hi_res", "ocr_only"] = "auto",
        chunking_strategy: Literal["basic", "by_title"] = "basic",
        allowed_metadata_types: Tuple[type, ...] = (str, int, float, list, dict, type(None)),
    ):
        """
        Initialize UnstructuredChunker.

        Args:
            strategy: Parsing strategy (default: "auto")
                - "auto": Automatically choose best strategy
                - "fast": Fast parsing (no OCR)
                - "hi_res": High-resolution parsing (with OCR)
                - "ocr_only": OCR-only parsing
            chunking_strategy: How to chunk documents (default: "basic")
                - "basic": Simple text splitting
                - "by_title": Chunk by document structure (headings)
            allowed_metadata_types: Allowed metadata value types
        """
        self.strategy = strategy
        self.chunking_strategy = chunking_strategy
        self.allowed_metadata_types = allowed_metadata_types

    async def chunk_text(
        self,
        text: str,
        filename: str = "document.txt",
        extra_metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> ChunkResult:
        """
        Chunk raw text input.

        Args:
            text: Raw text to chunk
            filename: Filename for metadata (default: "document.txt")
            extra_metadata: Additional metadata to attach to chunks
            **kwargs: Override strategy/chunking_strategy

        Returns:
            ChunkResult with documents and metadata
        """
        try:
            # Import here to avoid requiring unstructured for other operations
            from llama_index.readers.file import UnstructuredReader
            from unstructured.file_utils.filetype import detect_filetype

            # Convert text to bytes
            text_bytes = text.encode("utf-8")
            size_in_bytes = len(text_bytes)
            file_stream = BytesIO(text_bytes)

            # Detect file type
            content_type = detect_filetype(
                file=file_stream,
                metadata_file_path=filename,
            ).mime_type

            # Reset stream position after detection
            file_stream.seek(0)

            # Build unstructured arguments
            unstructured_args = {
                "strategy": kwargs.get("strategy", self.strategy),
                "chunking_strategy": kwargs.get(
                    "chunking_strategy", self.chunking_strategy
                ),
            }

            # Load and chunk document
            reader = UnstructuredReader(
                allowed_metadata_types=self.allowed_metadata_types,
            )
            documents = reader.load_data(
                unstructured_kwargs={
                    "file": file_stream,
                    "metadata_filename": filename,
                    "content_type": content_type,
                    **unstructured_args,
                },
                split_documents=True,
                extra_info=extra_metadata or {},
            )

            if not documents:
                raise ChunkingError("No chunks created from text")

            # Calculate statistics
            total_characters = sum(len(doc.text or "") for doc in documents)
            total_chunks = len(documents)
            total_pages = self._extract_total_pages(documents)

            return ChunkResult(
                documents=documents,
                metadata=ChunkMetadata(
                    filename=filename,
                    filetype=content_type,
                    size_in_bytes=size_in_bytes,
                ),
                total_chunks=total_chunks,
                total_characters=total_characters,
                total_pages=total_pages,
                extra_metadata=extra_metadata or {},
            )

        except Exception as e:
            raise ChunkingError(f"Failed to chunk text: {str(e)}") from e

    async def chunk_file(
        self,
        file_path: Path | str,
        extra_metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> ChunkResult:
        """
        Chunk a local file.

        Args:
            file_path: Path to file to chunk
            extra_metadata: Additional metadata to attach to chunks
            **kwargs: Override strategy/chunking_strategy

        Returns:
            ChunkResult with documents and metadata
        """
        try:
            # Import here to avoid requiring unstructured for other operations
            from llama_index.readers.file import UnstructuredReader
            from unstructured.file_utils.filetype import detect_filetype

            # Convert to Path object
            path = Path(file_path) if isinstance(file_path, str) else file_path

            if not path.exists():
                raise ChunkingError(f"File not found: {file_path}")

            # Read file
            file_bytes = path.read_bytes()
            size_in_bytes = len(file_bytes)
            file_stream = BytesIO(file_bytes)

            # Detect file type
            content_type = detect_filetype(
                file=file_stream,
                metadata_file_path=str(path),
            ).mime_type

            # Reset stream position after detection
            file_stream.seek(0)

            # Build unstructured arguments
            unstructured_args = {
                "strategy": kwargs.get("strategy", self.strategy),
                "chunking_strategy": kwargs.get(
                    "chunking_strategy", self.chunking_strategy
                ),
            }

            # Load and chunk document
            reader = UnstructuredReader(
                allowed_metadata_types=self.allowed_metadata_types,
            )
            documents = reader.load_data(
                unstructured_kwargs={
                    "file": file_stream,
                    "metadata_filename": path.name,
                    "content_type": content_type,
                    **unstructured_args,
                },
                split_documents=True,
                extra_info=extra_metadata or {},
            )

            if not documents:
                raise ChunkingError(f"No chunks created from file: {file_path}")

            # Calculate statistics
            total_characters = sum(len(doc.text or "") for doc in documents)
            total_chunks = len(documents)
            total_pages = self._extract_total_pages(documents)

            return ChunkResult(
                documents=documents,
                metadata=ChunkMetadata(
                    filename=path.name,
                    filetype=content_type,
                    size_in_bytes=size_in_bytes,
                ),
                total_chunks=total_chunks,
                total_characters=total_characters,
                total_pages=total_pages,
                extra_metadata=extra_metadata or {},
            )

        except ChunkingError:
            raise
        except Exception as e:
            raise ChunkingError(f"Failed to chunk file: {str(e)}") from e

    async def chunk_url(
        self,
        url: str,
        filename: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> ChunkResult:
        """
        Download and chunk a file from URL.

        Args:
            url: URL to download file from
            filename: Filename for metadata and type detection
            extra_metadata: Additional metadata to attach to chunks
            **kwargs: Override strategy/chunking_strategy

        Returns:
            ChunkResult with documents and metadata

        Raises:
            ChunkingError: If download or chunking fails
        """
        try:
            # Import here to avoid requiring unstructured for other operations
            from llama_index.readers.file import UnstructuredReader
            from unstructured.file_utils.filetype import detect_filetype

            # Download file
            async with httpx.AsyncClient() as client:
                response = await client.get(url, follow_redirects=True)
                response.raise_for_status()
                file_bytes = response.content

            size_in_bytes = len(file_bytes)
            file_stream = BytesIO(file_bytes)

            # Detect file type
            content_type = detect_filetype(
                file=file_stream,
                metadata_file_path=filename,
            ).mime_type

            # Reset stream position after detection
            file_stream.seek(0)

            # Build unstructured arguments
            unstructured_args = {
                "strategy": kwargs.get("strategy", self.strategy),
                "chunking_strategy": kwargs.get(
                    "chunking_strategy", self.chunking_strategy
                ),
            }

            # Load and chunk document
            reader = UnstructuredReader(
                allowed_metadata_types=self.allowed_metadata_types,
            )
            documents = reader.load_data(
                unstructured_kwargs={
                    "file": file_stream,
                    "metadata_filename": filename,
                    "content_type": content_type,
                    **unstructured_args,
                },
                split_documents=True,
                extra_info=extra_metadata or {},
            )

            if not documents:
                raise ChunkingError(f"No chunks created from URL: {url}")

            # Calculate statistics
            total_characters = sum(len(doc.text or "") for doc in documents)
            total_chunks = len(documents)
            total_pages = self._extract_total_pages(documents)

            return ChunkResult(
                documents=documents,
                metadata=ChunkMetadata(
                    filename=filename,
                    filetype=content_type,
                    size_in_bytes=size_in_bytes,
                ),
                total_chunks=total_chunks,
                total_characters=total_characters,
                total_pages=total_pages,
                extra_metadata=extra_metadata or {},
            )

        except httpx.HTTPError as e:
            raise ChunkingError(f"Failed to download file from URL: {str(e)}") from e
        except ChunkingError:
            raise
        except Exception as e:
            raise ChunkingError(f"Failed to chunk URL: {str(e)}") from e

    def _extract_total_pages(self, documents: List[Any]) -> Optional[int]:
        """
        Extract total page count from documents.

        Matches partition-api logic: finds max page_number in metadata.
        """
        total_pages = None

        for doc in documents:
            if hasattr(doc, "metadata") and isinstance(doc.metadata, dict):
                page_number = doc.metadata.get("page_number")
                if page_number is not None:
                    total_pages = max(
                        total_pages if total_pages is not None else 0,
                        page_number,
                    )

        return total_pages
