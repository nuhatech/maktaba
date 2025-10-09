"""Data models for document chunking."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ChunkMetadata:
    """
    Metadata about the chunked document.

    Matches partition-api output format.
    """

    filename: str
    """Original filename"""

    filetype: str
    """MIME type (e.g., 'application/pdf')"""

    size_in_bytes: int
    """File size in bytes"""


@dataclass
class ChunkResult:
    """
    Result of document chunking operation.

    Contains chunks as LlamaIndex Document objects plus metadata.
    Matches partition-api response structure.
    """

    documents: List[Any]  # List[llama_index.core.schema.Document]
    """Chunked documents in LlamaIndex format"""

    metadata: ChunkMetadata
    """File metadata"""

    total_chunks: int
    """Total number of chunks created"""

    total_characters: int
    """Total character count across all chunks"""

    total_pages: Optional[int] = None
    """Total pages (if applicable to document type)"""

    extra_metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata provided by user"""


    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format (matches partition-api response)."""
        result = {
            "metadata": {
                "filename": self.metadata.filename,
                "filetype": self.metadata.filetype,
                "sizeInBytes": self.metadata.size_in_bytes,
            },
            "total_chunks": self.total_chunks,
            "total_characters": self.total_characters,
            "documents": [doc.to_dict() for doc in self.documents],
        }

        if self.total_pages is not None:
            result["total_pages"] = self.total_pages

        if self.extra_metadata:
            result["extra_metadata"] = self.extra_metadata

        return result
