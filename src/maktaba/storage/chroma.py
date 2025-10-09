"""ChromaDB vector store implementation (local or client)."""

from typing import Any, Dict, List, Optional

from ..exceptions import StorageError
from ..models import SearchResult, VectorChunk
from .base import BaseVectorStore


class ChromaStore(BaseVectorStore):
    """
    Minimal Chroma wrapper matching BaseVectorStore interface.

    Defaults to a local, ephemeral client unless a `persist_directory` is provided.
    """

    def __init__(
        self,
        collection_name: str,
        persist_directory: Optional[str] = None,
    ) -> None:
        try:
            import chromadb
            from chromadb.config import Settings
        except Exception as e:  # pragma: no cover
            raise StorageError(
                "chromadb is not installed. Install with 'maktaba[chroma]'"
            ) from e

        try:
            if persist_directory:
                self._client = chromadb.Client(Settings(persist_directory=persist_directory))
            else:
                self._client = chromadb.Client()

            self._collection = self._client.get_or_create_collection(
                name=collection_name
            )
            self.collection_name = collection_name
        except Exception as e:
            raise StorageError(f"Failed to initialize Chroma collection: {str(e)}") from e

    async def upsert(
        self, chunks: List[VectorChunk], namespace: Optional[str] = None
    ) -> None:
        if not chunks:
            return
        try:
            ids = [c.id for c in chunks]
            embeddings = [c.vector for c in chunks]
            metadatas = [{**c.metadata, **({"namespace": namespace} if namespace else {})} for c in chunks]
            self._collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)
        except Exception as e:
            raise StorageError(f"Chroma upsert failed: {str(e)}") from e

    async def query(
        self,
        vector: List[float],
        topK: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        includeMetadata: bool = True,
        namespace: Optional[str] = None,
    ) -> List[SearchResult]:
        try:
            where = dict(filter or {})
            if namespace:
                where["namespace"] = namespace
            resp = self._collection.query(
                query_embeddings=[vector], n_results=topK, where=where or None
            )
            out: List[SearchResult] = []
            ids = (resp.get("ids") or [[]])[0]
            dists = (resp.get("distances") or [[]])[0]
            metas = (resp.get("metadatas") or [[]])[0]
            for i, sid in enumerate(ids):
                meta = metas[i] if includeMetadata and i < len(metas) else {}
                # Convert Chroma distance to a similarity-like score (simple inverse)
                dist = float(dists[i]) if i < len(dists) else 0.0
                score = 1.0 / (1.0 + dist) if dist >= 0 else 0.0
                out.append(SearchResult(id=str(sid), score=score, metadata=meta or {}))
            return out
        except Exception as e:
            raise StorageError(f"Chroma query failed: {str(e)}") from e

    async def delete(
        self, ids: List[str], namespace: Optional[str] = None
    ) -> None:
        if not ids:
            return
        try:
            self._collection.delete(ids=ids)
        except Exception as e:
            raise StorageError(f"Chroma delete failed: {str(e)}") from e

    async def list(
        self,
        prefix: Optional[str] = None,
        limit: int = 100,
        namespace: Optional[str] = None,
    ) -> List[str]:
        try:
            where = {"namespace": namespace} if namespace else None
            got = self._collection.get(include=["ids"], limit=limit, where=where)
            ids = got.get("ids", [])
            if prefix is not None:
                ids = [i for i in ids if str(i).startswith(prefix)]
            return ids[:limit]
        except Exception as e:
            raise StorageError(f"Chroma list failed: {str(e)}") from e

    async def get_dimensions(self) -> int:
        try:
            peek = self._collection.peek(1)
            emb = (peek.get("embeddings") or [[]])[0]
            if emb:
                return len(emb)
        except Exception:
            pass
        return 1536
