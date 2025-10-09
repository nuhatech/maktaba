"""Base interface for embedding providers."""

from abc import ABC, abstractmethod
from typing import List, Literal

from ..models import EmbeddingVector


class BaseEmbedder(ABC):
    """
    Abstract base class for text embedding providers.

    Design principles:
    1. Batch-first: embed_batch is the primary method
    2. Input types: Support 'document' vs 'query' for Voyage AI
    3. Async-first: All methods are async
    4. Dimension aware: Each embedder knows its output dimension
    """

    @abstractmethod
    async def embed_batch(
        self,
        texts: List[str],
        input_type: Literal["document", "query"] = "document",
    ) -> List[EmbeddingVector]:
        """
        Embed multiple texts in a single batch (primary method).

        Args:
            texts: List of texts to embed
            input_type: Type of input - 'document' for indexing, 'query' for search
                       (required by Voyage AI, ignored by OpenAI/others)

        Returns:
            List of embedding vectors, one per input text

        Raises:
            EmbeddingError: If embedding fails
        """
        pass

    async def embed_text(
        self,
        text: str,
        input_type: Literal["document", "query"] = "document",
    ) -> EmbeddingVector:
        """
        Embed a single text (convenience wrapper).

        This is implemented as a wrapper around embed_batch for consistency.
        """
        results = await self.embed_batch([text], input_type=input_type)
        return results[0]

    @property
    @abstractmethod
    def dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this embedder.

        Examples:
            - text-embedding-3-large: 3072
            - text-embedding-3-small: 1536
            - voyage-3-large: 1024
        """
        pass

    @property
    @abstractmethod
    def model(self) -> str:
        """Get the model name/identifier."""
        pass
