"""Core data models for Maktaba."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional


class RelationshipType(str, Enum):
    """
    LlamaIndex relationship types.

    Maps to LlamaIndex numeric codes:
    - SOURCE: "1" - Original source document
    - PREVIOUS: "2" - Previous chunk in sequence
    - NEXT: "3" - Next chunk in sequence
    - PARENT: "4" - Parent node in hierarchy
    - CHILD: "5" - Child node in hierarchy
    """
    SOURCE = "SOURCE"
    PREVIOUS = "PREVIOUS"
    NEXT = "NEXT"
    PARENT = "PARENT"
    CHILD = "CHILD"


@dataclass
class NodeRelationship:
    """
    Relationship to another node.

    Stores full relationship metadata including node type and hash
    for complete compatibility with LlamaIndex relationship structure.
    """
    node_id: str
    node_type: Optional[str] = None  # ObjectType from LlamaIndex ("1" for TEXT, etc.)
    metadata: Dict[str, Any] = field(default_factory=dict)
    hash: Optional[str] = None  # Content hash for change detection

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "metadata": self.metadata,
            "hash": self.hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NodeRelationship':
        """Create from dictionary format."""
        return cls(
            node_id=data["node_id"],
            node_type=data.get("node_type"),
            metadata=data.get("metadata", {}),
            hash=data.get("hash"),
        )


@dataclass
class VectorChunk:
    """
    Represents a text chunk with its embedding vector.

    ID format: {document_id}#{chunk_id}
    Metadata follows LlamaIndex node format for compatibility.
    Relationships enable linking chunks together (e.g., NEXT/PREVIOUS for sequential pages).
    """

    id: str  # Format: "{doc_id}#{chunk_id}"
    vector: List[float]  # Embedding vector (3072 dims for text-embedding-3-large)
    metadata: Dict[str, Any] = field(default_factory=dict)
    relationships: Optional[Dict[str, NodeRelationship]] = None  # Full LlamaIndex relationships

    # Deprecated: Use relationships with NodeRelationship objects instead
    # Kept for backward compatibility only
    simple_relationships: Optional[Dict[str, str]] = None  # Legacy: {"NEXT": "doc#page_2"}

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
    Relationships allow navigation to related chunks (e.g., neighboring pages).
    """

    id: str  # Chunk ID
    score: Optional[float] = None  # Similarity score (0.0 - 1.0), None for keyword search
    metadata: Dict[str, Any] = field(default_factory=dict)
    relationships: Optional[Dict[str, NodeRelationship]] = None  # Full LlamaIndex relationships

    # Deprecated: Use relationships with NodeRelationship objects instead
    # Kept for backward compatibility only
    simple_relationships: Optional[Dict[str, str]] = None  # Legacy: {"NEXT": "doc#page_2"}

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
class LLMUsage:
    """
    LLM token usage tracking.

    Tracks input and output tokens for precise cost/budget monitoring.
    Provider-agnostic - works with OpenAI, Anthropic, Bedrock, etc.
    """

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Total tokens used (input + output)."""
        return self.input_tokens + self.output_tokens

    def __add__(self, other: 'LLMUsage') -> 'LLMUsage':
        """Add two usage objects together."""
        return LLMUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens
        )

    def __repr__(self) -> str:
        """String representation."""
        return f"LLMUsage(input={self.input_tokens}, output={self.output_tokens}, total={self.total_tokens})"


# Type aliases for clarity
EmbeddingVector = List[float]
EmbeddingBatch = List[EmbeddingVector]
ChunkID = str  # Format: "{doc_id}#{chunk_id}"
DocumentID = str
