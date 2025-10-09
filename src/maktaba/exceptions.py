"""Custom exceptions for Maktaba library."""


class MaktabaError(Exception):
    """Base exception for all Maktaba errors."""

    pass


class EmbeddingError(MaktabaError):
    """Raised when embedding operation fails."""

    pass


class StorageError(MaktabaError):
    """Raised when vector storage operation fails."""

    pass


class ChunkingError(MaktabaError):
    """Raised when document chunking fails."""

    pass


class ConfigurationError(MaktabaError):
    """Raised when configuration is invalid."""

    pass


class PartitionAPIError(MaktabaError):
    """Raised when partition API call fails."""

    pass
