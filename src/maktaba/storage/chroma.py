"""ChromaDB vector store implementation (local or client)."""

import json
from typing import Any, Dict, List, Optional

from ..exceptions import StorageError
from ..models import NodeRelationship, SearchResult, VectorChunk
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
            embeddings: list[list[float]] = [c.vector for c in chunks]
            metadatas: list[dict[str, Any]] = []
            for chunk in chunks:
                metadata = {**chunk.metadata, **({"namespace": namespace} if namespace else {})}
                relationships = chunk.relationships or chunk.simple_relationships
                if relationships:
                    serialized = {
                        rel_type: rel.to_dict() if hasattr(rel, "to_dict") else rel
                        for rel_type, rel in relationships.items()
                    }
                    # Chroma metadata values must be scalar, so relationships
                    # are stored as JSON and hydrated on read.
                    metadata["_relationships_json"] = json.dumps(serialized)
                metadatas.append(metadata)
            self._collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)  # type: ignore[arg-type]
        except Exception as e:
            raise StorageError(f"Chroma upsert failed: {str(e)}") from e

    async def query(
        self,
        vector: List[float],
        topK: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        includeMetadata: bool = True,
        includeRelationships: bool = False,
        namespace: Optional[str] = None,
    ) -> List[SearchResult]:
        try:
            where = dict(filter or {})
            if namespace:
                where["namespace"] = namespace
            resp = self._collection.query(
                query_embeddings=[vector], n_results=topK, where=where or None  # type: ignore[arg-type]
            )
            out: List[SearchResult] = []
            ids = (resp.get("ids") or [[]])[0]
            dists = (resp.get("distances") or [[]])[0]
            metas = (resp.get("metadatas") or [[]])[0]
            for i, sid in enumerate(ids):
                raw_meta = metas[i] if i < len(metas) else {}
                # Convert Chroma distance to a similarity-like score (simple inverse)
                dist = float(dists[i]) if i < len(dists) else 0.0
                score = 1.0 / (1.0 + dist) if dist >= 0 else 0.0
                hydrated_metadata = dict(raw_meta) if raw_meta else {}
                relationships = self._decode_relationships(hydrated_metadata) if includeRelationships else None
                out.append(
                    SearchResult(
                        id=str(sid),
                        score=score,
                        metadata=hydrated_metadata if includeMetadata else {},
                        relationships=relationships,
                    )
                )
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

    async def fetch_by_ids(
        self,
        ids: List[str],
        *,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
        includeMetadata: bool = True,
        includeRelationships: bool = False,
    ) -> List[SearchResult]:
        if not ids:
            return []
        try:
            where = dict(filter or {})
            if namespace:
                where["namespace"] = namespace
            response = self._collection.get(ids=ids, where=where or None, include=["metadatas"])
            result_ids = response.get("ids", [])
            metadatas = response.get("metadatas", []) or []
            results: List[SearchResult] = []
            for index, result_id in enumerate(result_ids):
                metadata = dict(metadatas[index] or {}) if index < len(metadatas) else {}
                results.append(
                    SearchResult(
                        id=str(result_id),
                        metadata=metadata if includeMetadata else {},
                        relationships=self._decode_relationships(metadata) if includeRelationships else None,
                    )
                )
            return results
        except Exception as e:
            raise StorageError(f"Chroma fetch_by_ids failed: {str(e)}") from e

    @staticmethod
    def _decode_relationships(metadata: Dict[str, Any]) -> Optional[Dict[str, NodeRelationship]]:
        raw = metadata.get("_relationships_json")
        if not isinstance(raw, str):
            return None
        try:
            values = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(values, dict):
            return None
        return {
            rel_type: NodeRelationship.from_dict(rel_value)
            if isinstance(rel_value, dict) and "node_id" in rel_value
            else rel_value
            for rel_type, rel_value in values.items()
        }

    async def list(
        self,
        prefix: Optional[str] = None,
        limit: int = 100,
        namespace: Optional[str] = None,
    ) -> List[str]:
        try:
            where = {"namespace": namespace} if namespace else None
            got = self._collection.get(limit=limit, where=where)  # type: ignore[arg-type]
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
