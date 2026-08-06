"""Bounded traversal of chunk relationships for context expansion."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..logging import get_logger
from ..models import NodeRelationship, SearchResult
from ..storage.base import BaseVectorStore


class RelationshipContextExpander:
    """Expand NEXT/PREVIOUS-like links with cycle and scope protection."""

    def __init__(self, store: BaseVectorStore) -> None:
        self.store = store
        self._logger = get_logger("maktaba.retrieval.relationships")

    @staticmethod
    def _related_ids(result: SearchResult, relationship_types: Iterable[str]) -> List[str]:
        allowed = {item.upper() for item in relationship_types}
        relationships = result.relationships or result.simple_relationships or {}
        related: List[str] = []
        for rel_type, rel_value in relationships.items():
            if str(rel_type).upper() not in allowed:
                continue
            if isinstance(rel_value, NodeRelationship):
                related.append(rel_value.node_id)
            elif isinstance(rel_value, str):
                related.append(rel_value)
            elif isinstance(rel_value, dict) and rel_value.get("node_id"):
                related.append(str(rel_value["node_id"]))
        return related

    async def expand(
        self,
        seeds: Sequence[SearchResult],
        *,
        relationship_types: Iterable[str] = ("PREVIOUS", "NEXT"),
        max_hops: int = 1,
        max_chunks: int = 8,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
        includeMetadata: bool = True,
    ) -> Tuple[List[SearchResult], Dict[str, List[str]]]:
        """Return related chunks and a child-to-parent provenance map."""
        if max_hops <= 0 or max_chunks <= 0 or not seeds:
            return [], {}

        known: Dict[str, SearchResult] = {seed.id: seed for seed in seeds}
        frontier = list(seeds)
        visited = set(known)
        expanded: List[SearchResult] = []
        parents: Dict[str, List[str]] = {}

        for _hop in range(max_hops):
            requested_by: Dict[str, List[str]] = {}
            for parent in frontier:
                for related_id in self._related_ids(parent, relationship_types):
                    if related_id in visited:
                        continue
                    requested_by.setdefault(related_id, []).append(parent.id)

            remaining = max_chunks - len(expanded)
            requested_ids = list(requested_by)[:remaining]
            if not requested_ids:
                break

            try:
                fetched = await self.store.fetch_by_ids(
                    requested_ids,
                    namespace=namespace,
                    filter=filter,
                    includeMetadata=includeMetadata,
                    includeRelationships=True,
                )
            except Exception as exc:
                self._logger.warning("Relationship expansion failed safely: %s", exc)
                break

            frontier = []
            for result in fetched:
                if result.id in visited:
                    continue
                visited.add(result.id)
                known[result.id] = result
                expanded.append(result)
                frontier.append(result)
                parents[result.id] = requested_by.get(result.id, [])
                if len(expanded) >= max_chunks:
                    break
            if not frontier or len(expanded) >= max_chunks:
                break

        return expanded, parents
