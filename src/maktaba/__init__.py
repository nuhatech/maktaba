"""
Maktaba (EC*()) - The library for building libraries.

Production-ready RAG infrastructure for Arabic & multilingual applications.
By NuhaTech.
"""

from .exceptions import (
    ChunkingError,
    ConfigurationError,
    EmbeddingError,
    MaktabaError,
    PartitionAPIError,
    StorageError,
)
from .models import (
    EmbeddingConfig,
    PartitionConfig,
    SearchResult,
    VectorChunk,
    VectorStoreConfig,
)

__version__ = "0.1.0"

__all__ = [
    # Exceptions
    "MaktabaError",
    "EmbeddingError",
    "StorageError",
    "ChunkingError",
    "ConfigurationError",
    "PartitionAPIError",
    # Models
    "VectorChunk",
    "SearchResult",
    "EmbeddingConfig",
    "VectorStoreConfig",
    "PartitionConfig",
    # Version
    "__version__",
]
