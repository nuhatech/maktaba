"""Weaviate vector store implementation (skeleton)."""

import json
from typing import Any, Dict, List, Optional

from ..exceptions import StorageError
from ..models import NodeRelationship, SearchResult, VectorChunk
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
            schema = self._client.schema.get()
            existing_classes = schema.get("classes", [])
            target_class = next((c for c in existing_classes if c.get("class") == class_name), None)
            if not target_class:
                self._client.schema.create_class(
                    {
                        "class": class_name,
                        "properties": [
                            {"name": "text", "dataType": ["text"]},
                            {"name": "namespace", "dataType": ["text"]},
                            {"name": "metadata", "dataType": ["text"]},
                        ],
                        "vectorizer": "none",
                    }
                )
            else:
                existing_props = {prop.get("name") for prop in target_class.get("properties", [])}
                for prop in [
                    {"name": "metadata", "dataType": ["text"]},
                ]:
                    if prop["name"] not in existing_props:
                        self._client.schema.property.create(class_name, prop)
        except Exception as e:
            raise StorageError(f"Failed to initialize Weaviate: {str(e)}") from e

    async def upsert(
        self, chunks: List[VectorChunk], namespace: Optional[str] = None
    ) -> None:
        try:
            with self._client.batch as batch:
                for c in chunks:
                    metadata = self._build_metadata(c, namespace)
                    props = {
                        "text": metadata.get("text", c.metadata.get("text", "")),
                        "namespace": metadata.get("namespace", namespace or self._namespace or ""),
                        "metadata": json.dumps(metadata),
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
        includeRelationships: bool = False,
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

            q = self._client.query.get(self._class_name, ["_additional { id distance }", "text", "namespace", "metadata"])
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
                raw_meta = item.get("metadata")
                decoded_meta: Dict[str, Any] = {}
                if isinstance(raw_meta, str) and raw_meta:
                    try:
                        decoded_meta = json.loads(raw_meta)
                    except (json.JSONDecodeError, TypeError):
                        decoded_meta = {"_raw_metadata": raw_meta}
                elif isinstance(raw_meta, dict):
                    decoded_meta = raw_meta

                if "text" not in decoded_meta and item.get("text") is not None:
                    decoded_meta["text"] = item.get("text")
                if "namespace" not in decoded_meta and item.get("namespace") is not None:
                    decoded_meta["namespace"] = item.get("namespace")

                relationships = None
                if includeRelationships and decoded_meta:
                    rels = decoded_meta.get("_relationships")
                    if isinstance(rels, dict):
                        relationships = {}
                        for rel_type, rel_value in rels.items():
                            if isinstance(rel_value, dict) and "node_id" in rel_value:
                                relationships[rel_type] = NodeRelationship.from_dict(rel_value)
                            else:
                                relationships[rel_type] = rel_value

                out.append(
                    SearchResult(
                        id=str(wid),
                        score=score,
                        metadata=decoded_meta if includeMetadata else {},
                        relationships=relationships,
                    )
                )
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

    def _build_metadata(
        self,
        chunk: VectorChunk,
        namespace: Optional[str],
    ) -> Dict[str, Any]:
        """Prepare metadata payload with serialized relationships."""
        metadata: Dict[str, Any] = dict(chunk.metadata)

        if chunk.relationships:
            serialized: Dict[str, Any] = {}
            for rel_type, rel_obj in chunk.relationships.items():
                if hasattr(rel_obj, "to_dict"):
                    serialized[rel_type] = rel_obj.to_dict()
                else:
                    serialized[rel_type] = rel_obj
            metadata["_relationships"] = serialized
        elif getattr(chunk, "simple_relationships", None):
            metadata["_relationships"] = chunk.simple_relationships

        if namespace:
            metadata["namespace"] = namespace
        elif self._namespace:
            metadata.setdefault("namespace", self._namespace)

        return metadata
