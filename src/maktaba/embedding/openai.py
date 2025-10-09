"""OpenAI embedder implementation."""

from typing import List, Literal, Optional

from openai import AsyncOpenAI

from ..exceptions import EmbeddingError
from ..models import EmbeddingVector
from .base import BaseEmbedder

# Supported models and their dimensions
SUPPORTED_MODELS = {
    "text-embedding-3-large": 3072,  # Default
    "text-embedding-3-small": 1536,
    "text-embedding-ada-002": 1536,  # Legacy
}


class OpenAIEmbedder(BaseEmbedder):
    """
    OpenAI embedding provider.

    Default model: text-embedding-3-large (3072 dimensions)

    Example:
        embedder = OpenAIEmbedder(api_key="sk-...")
        vectors = await embedder.embed_batch(["text 1", "text 2"])
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-large",  # default model
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
    ):
        """
        Initialize OpenAI embedder.

        Args:
            api_key: OpenAI API key
            model: Model name (default: text-embedding-3-large)
            base_url: Optional custom API base URL
            organization: Optional OpenAI organization ID
        """
        if model not in SUPPORTED_MODELS:
            raise EmbeddingError(
                f"Unsupported model: {model}. "
                f"Supported models: {list(SUPPORTED_MODELS.keys())}"
            )

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
        )
        self._model = model
        self._dimension = SUPPORTED_MODELS[model]

    async def embed_batch(
        self,
        texts: List[str],
        input_type: Literal["document", "query"] = "document",
    ) -> List[EmbeddingVector]:
        """
        Embed multiple texts in a single batch.

        Args:
            texts: List of texts to embed
            input_type: Ignored for OpenAI (kept for interface compatibility)

        Returns:
            List of embedding vectors

        Raises:
            EmbeddingError: If API call fails
        """
        if not texts:
            return []

        try:
            response = await self.client.embeddings.create(
                model=self._model,
                input=texts,
                encoding_format="float",
            )

            # Extract embeddings in order
            embeddings = [item.embedding for item in response.data]

            # Validate dimensions
            for emb in embeddings:
                if len(emb) != self._dimension:
                    raise EmbeddingError(
                        f"Expected {self._dimension} dimensions, got {len(emb)}"
                    )

            return embeddings

        except Exception as e:
            raise EmbeddingError(f"OpenAI embedding failed: {str(e)}") from e

    @property
    def dimension(self) -> int:
        """Get the dimension of embeddings produced by this embedder."""
        return self._dimension

    @property
    def model(self) -> str:
        """Get the model name."""
        return self._model
