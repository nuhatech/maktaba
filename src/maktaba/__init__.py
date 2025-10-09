"""
Maktaba (EC*()) - The library for building libraries.

Production-ready RAG infrastructure for Arabic & multilingual applications.
By NuhaTech.
"""

from .exceptions import (
    ChunkingError,
    ConfigurationError,
    EmbeddingError,
    MaktabaException,
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
    "MaktabaException",
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
