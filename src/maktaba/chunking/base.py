"""Base interface for document chunkers."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

from .models import ChunkResult


class BaseChunker(ABC):
    """
    Abstract base class for document chunking implementations.

    All chunkers return LlamaIndex Document objects for compatibility
    with the broader ecosystem.
    """

    @abstractmethod
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
            filename: Filename to use for metadata
            extra_metadata: Additional metadata to attach to chunks
            **kwargs: Implementation-specific options

        Returns:
            ChunkResult with documents and metadata
        """
        pass

    @abstractmethod
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
            **kwargs: Implementation-specific options

        Returns:
            ChunkResult with documents and metadata
        """
        pass

    @abstractmethod
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
            filename: Filename to use for metadata and type detection
            extra_metadata: Additional metadata to attach to chunks
            **kwargs: Implementation-specific options

        Returns:
            ChunkResult with documents and metadata

        Raises:
            ChunkingError: If download or chunking fails
        """
        pass
