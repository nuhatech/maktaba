"""Pinecone vector store implementation (Pinecone v3+)."""

from typing import Any, Dict, List, Optional

from ..exceptions import StorageError
from ..models import SearchResult, VectorChunk
from .base import BaseVectorStore


class PineconeStore(BaseVectorStore):
    """
    Minimal Pinecone wrapper matching BaseVectorStore interface.

    Notes:
        - Requires `pinecone-client>=3` (optional dependency group: `pinecone`).
        - Listing all IDs and prefix delete are not natively supported by Pinecone;
          `list()` returns an empty list to signal non-support.
    """

    def __init__(
        self,
        api_key: str,
        index_name: str,
        dimension: Optional[int] = None,
        namespace: Optional[str] = None,
    ) -> None:
        try:
            from pinecone import Pinecone  # type: ignore
        except Exception as e:  # pragma: no cover
            raise StorageError(
                "pinecone-client is not installed. Install with 'maktaba[pinecone]'"
            ) from e

        try:
            self._pc = Pinecone(api_key=api_key)
            self._index = self._pc.Index(index_name)
            self._dimension = dimension  # optional hint if stats unavailable
            self._namespace = namespace
        except Exception as e:
            raise StorageError(f"Failed to initialize Pinecone index: {str(e)}") from e

    async def upsert(
        self, chunks: List[VectorChunk], namespace: Optional[str] = None
    ) -> None:
        if not chunks:
            return
        try:
            vectors = [
                {
                    "id": c.id,
                    "values": c.vector,
                    "metadata": dict(c.metadata),
                }
                for c in chunks
            ]
            self._index.upsert(vectors=vectors, namespace=namespace or self._namespace)
        except Exception as e:
            raise StorageError(f"Pinecone upsert failed: {str(e)}") from e

    async def query(
        self,
        vector: List[float],
        topK: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        includeMetadata: bool = True,
        namespace: Optional[str] = None,
    ) -> List[SearchResult]:
        try:
            res = self._index.query(
                vector=vector,
                top_k=topK,
                filter=filter,
                include_values=False,
                include_metadata=includeMetadata,
                namespace=namespace or self._namespace,
            )
            matches = getattr(res, "matches", []) or res.get("matches", [])
            out: List[SearchResult] = []
            for m in matches:
                mid = getattr(m, "id", None) or (m.get("id") if isinstance(m, dict) else None)
                mscore = getattr(m, "score", None) or (m.get("score") if isinstance(m, dict) else None)
                mmeta = getattr(m, "metadata", None) or (m.get("metadata") if isinstance(m, dict) else None)
                out.append(
                    SearchResult(
                        id=str(mid),
                        score=float(mscore or 0.0),
                        metadata=mmeta or {},
                    )
                )
            return out
        except Exception as e:
            raise StorageError(f"Pinecone query failed: {str(e)}") from e

    async def delete(
        self, ids: List[str], namespace: Optional[str] = None
    ) -> None:
        if not ids:
            return
        try:
            self._index.delete(ids=ids, namespace=namespace or self._namespace)
        except Exception as e:
            raise StorageError(f"Pinecone delete failed: {str(e)}") from e

    async def list(
        self,
        prefix: Optional[str] = None,
        limit: int = 100,
        namespace: Optional[str] = None,
    ) -> List[str]:
        # Pinecone API does not provide listing IDs natively.
        return []

    async def get_dimensions(self) -> int:
        # Try describe_index if available, else fallback to hint or 1536
        try:
            desc = self._pc.describe_index(self._index.name)
            dim = getattr(desc, "dimension", None) or (desc.get("dimension") if isinstance(desc, dict) else None)
            if isinstance(dim, int):
                return dim
        except Exception:
            pass
        if isinstance(self._dimension, int):
            return self._dimension
        return 1536
