"""Core data models for Maktaba."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


@dataclass
class VectorChunk:
    """
    Represents a text chunk with its embedding vector.

    ID format: {document_id}#{chunk_id}
    Metadata follows LlamaIndex node format for compatibility.
    """

    id: str  # Format: "{doc_id}#{chunk_id}"
    vector: List[float]  # Embedding vector (3072 dims for text-embedding-3-large)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate chunk format."""
        if "#" not in self.id:
            raise ValueError(
                f"Chunk ID must follow format '{{doc_id}}#{{chunk_id}}', got: {self.id}"
            )


@dataclass
class SearchResult:
    """
    Search result from vector store.

    Represents a single result from semantic search.
    """

    id: str  # Chunk ID
    score: float  # Similarity score (0.0 - 1.0)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> Optional[str]:
        """Extract text from metadata (LlamaIndex format)."""
        # Check for text in common locations
        if "text" in self.metadata:
            return self.metadata["text"]
        if "_node_content" in self.metadata:
            node_content = self.metadata["_node_content"]
            if isinstance(node_content, dict) and "text" in node_content:
                return node_content["text"]
        return None

    @property
    def document_id(self) -> str:
        """
        Extract document ID from chunk ID.

        For Qdrant in-memory/local mode, the original chunk ID
        is preserved in metadata["_original_id"], so we check there first.
        """
        # Try to get original chunk ID from metadata first
        original_id = self.metadata.get("_original_id", self.id)
        return original_id.split("#")[0] if "#" in original_id else original_id

    @property
    def chunk_id(self) -> str:
        """
        Extract chunk ID from full ID.

        For Qdrant in-memory/local mode, the original chunk ID
        is preserved in metadata["_original_id"], so we check there first.
        """
        # Try to get original chunk ID from metadata first
        original_id = self.metadata.get("_original_id", self.id)
        return original_id.split("#", 1)[1] if "#" in original_id else original_id


@dataclass
class EmbeddingConfig:
    """Configuration for embedding provider."""

    provider: Literal["openai", "azure", "cohere", "voyage", "google"]
    model: str = "text-embedding-3-large"  # default
    api_key: str = ""

    # Azure-specific fields
    resource_name: Optional[str] = None
    deployment: Optional[str] = None
    api_version: str = "2024-02-01"

    # OpenAI-specific
    base_url: Optional[str] = None

    def get_dimension(self) -> int:
        """Get expected dimension for model."""
        # Model name to embedding dimension mapping
        dimensions = {
            # OpenAI
            "text-embedding-3-large": 3072,
            "text-embedding-3-small": 1536,
            "text-embedding-ada-002": 1536,
            # Google
            "text-embedding-004": 768,
            # Voyage
            "voyage-3-large": 1024,
            "voyage-3": 1024,
            "voyage-3-lite": 512,
            "voyage-code-3": 1024,
            "voyage-finance-2": 1024,
            "voyage-law-2": 1024,
        }
        return dimensions.get(self.model, 1536)  # Default fallback


@dataclass
class VectorStoreConfig:
    """Configuration for vector store provider."""

    provider: Literal["qdrant", "pinecone", "chroma", "weaviate"]
    url: str
    collection_name: str
    api_key: Optional[str] = None
    namespace: Optional[str] = None  # For multi-tenancy (Pinecone-style)


@dataclass
class PartitionConfig:
    """Configuration for partition API (document chunking)."""

    api_url: str = "http://localhost:8000"  # Self-hosted by default
    api_key: str = ""
    redis_url: Optional[str] = None  # For fetching batched chunks
    redis_password: Optional[str] = None

    # Chunking parameters (match Unstructured.io)
    default_strategy: Literal["auto", "fast", "hi_res", "ocr_only"] = "auto"
    default_chunking_strategy: Literal["basic", "by_title"] = "basic"
    default_chunk_size: int = 1000
    default_overlap: int = 200
    batch_size: int = 30  # Default


# Type aliases for clarity
EmbeddingVector = List[float]
EmbeddingBatch = List[EmbeddingVector]
ChunkID = str  # Format: "{doc_id}#{chunk_id}"
DocumentID = str
