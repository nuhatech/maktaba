"""Custom exceptions for Maktaba library."""


class MaktabaException(Exception):
    """Base exception for all Maktaba errors."""

    pass


class EmbeddingError(MaktabaException):
    """Raised when embedding operation fails."""

    pass


class StorageError(MaktabaException):
    """Raised when vector storage operation fails."""

    pass


class ChunkingError(MaktabaException):
    """Raised when document chunking fails."""

    pass


class ConfigurationError(MaktabaException):
    """Raised when configuration is invalid."""

    pass


class PartitionAPIError(MaktabaException):
    """Raised when partition API call fails."""

    pass
