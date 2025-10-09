"""Weaviate vector store implementation (skeleton)."""

from typing import Any, Dict, List, Optional

from ..exceptions import StorageError
from ..models import SearchResult, VectorChunk
from .base import BaseVectorStore


class WeaviateStore(BaseVectorStore):
    """
    Minimal Weaviate wrapper matching BaseVectorStore interface.

    Notes:
        - Requires `weaviate-client` (not included by default). If unavailable,
          initialization will raise a helpful StorageError.
        - This is a minimal skeleton; production setups should define a schema/class
          and consider hybrid/text modules.
    """

    def __init__(
        self,
        url: str,
        api_key: Optional[str] = None,
        class_name: str = "MaktabaChunk",
        namespace: Optional[str] = None,
    ) -> None:
        try:
            import weaviate  # type: ignore
        except Exception as e:  # pragma: no cover
            raise StorageError(
                "weaviate-client is not installed. Install the client to use WeaviateStore."
            ) from e

        try:
            # v3 client API
            auth = weaviate.AuthApiKey(api_key) if api_key else None
            self._client = weaviate.Client(url=url, auth_client_secret=auth)
            self._class_name = class_name
            self._namespace = namespace
            # Ensure class exists (very basic schema)
            if not self._client.schema.contains({"classes": [{"class": class_name}]}):
                self._client.schema.create_class(
                    {
                        "class": class_name,
                        "properties": [
                            {"name": "text", "dataType": ["text"]},
                            {"name": "namespace", "dataType": ["text"]},
                        ],
                        "vectorizer": "none",
                    }
                )
        except Exception as e:
            raise StorageError(f"Failed to initialize Weaviate: {str(e)}") from e

    async def upsert(
        self, chunks: List[VectorChunk], namespace: Optional[str] = None
    ) -> None:
        try:
            with self._client.batch as batch:
                for c in chunks:
                    props = {
                        "text": c.metadata.get("text", ""),
                        "namespace": namespace or self._namespace or "",
                    }
                    batch.add_data_object(
                        data_object=props,
                        class_name=self._class_name,
                        uuid=c.id,
                        vector=c.vector,
                    )
        except Exception as e:
            raise StorageError(f"Weaviate upsert failed: {str(e)}") from e

    async def query(
        self,
        vector: List[float],
        topK: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        includeMetadata: bool = True,
        namespace: Optional[str] = None,
    ) -> List[SearchResult]:
        try:
            near_vec = {"vector": vector}
            where = None
            if namespace or (filter and "namespace" in filter):
                ns = namespace or filter.get("namespace")  # type: ignore
                where = {
                    "path": ["namespace"],
                    "operator": "Equal",
                    "valueText": ns,
                }

            q = self._client.query.get(self._class_name, ["_additional { id distance }", "text", "namespace"])
            if where:
                q = q.with_where(where)
            q = q.with_near_vector(near_vec).with_limit(topK)
            resp = q.do()

            data = (((resp or {}).get("data") or {}).get("Get") or {}).get(self._class_name, [])
            out: List[SearchResult] = []
            for item in data:
                add = item.get("_additional", {})
                wid = add.get("id")
                dist = add.get("distance", 0.0)
                score = 1.0 / (1.0 + float(dist)) if isinstance(dist, (int, float)) else 0.0
                meta = {"text": item.get("text"), "namespace": item.get("namespace")}
                out.append(SearchResult(id=str(wid), score=score, metadata=meta if includeMetadata else {}))
            return out
        except Exception as e:
            raise StorageError(f"Weaviate query failed: {str(e)}") from e

    async def delete(
        self, ids: List[str], namespace: Optional[str] = None
    ) -> None:
        try:
            for _id in ids:
                self._client.data_object.delete(_id, class_name=self._class_name)
        except Exception as e:
            raise StorageError(f"Weaviate delete failed: {str(e)}") from e

    async def list(
        self,
        prefix: Optional[str] = None,
        limit: int = 100,
        namespace: Optional[str] = None,
    ) -> List[str]:
        # Listing all IDs in Weaviate is non-trivial; return empty list for now.
        return []

    async def get_dimensions(self) -> int:
        # We rely on vectors supplied by the embedder; Weaviate doesn't enforce a fixed dimension per class.
        return 1536
