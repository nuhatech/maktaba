"""Provider-agnostic ranking utilities for multi-query retrieval."""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence, Tuple

from ..models import SearchResult
from ..pipeline.agentic_models import EvidenceProvenance, SearchQuery

RankedResultList = Tuple[SearchQuery, int, Sequence[SearchResult]]


def reciprocal_rank_fusion(
    ranked_lists: Iterable[RankedResultList],
    *,
    rrf_k: int = 60,
    limit: int = 200,
) -> Tuple[List[SearchResult], Dict[str, EvidenceProvenance]]:
    """Fuse heterogeneous ranked lists without comparing provider scores."""
    by_id: Dict[str, SearchResult] = {}
    provenance: Dict[str, EvidenceProvenance] = {}

    for query, iteration, results in ranked_lists:
        seen_in_list: set[str] = set()
        for rank, result in enumerate(results, start=1):
            if result.id in seen_in_list:
                continue
            seen_in_list.add(result.id)
            by_id.setdefault(result.id, result)
            item = provenance.setdefault(result.id, EvidenceProvenance())
            item.add_retrieval(
                query=query.query,
                query_type=query.type,
                iteration=iteration,
                rank=rank,
                score=result.score,
            )
            item.fusion_score += 1.0 / (rrf_k + rank)

    ordered_ids = sorted(
        by_id,
        key=lambda result_id: (provenance[result_id].fusion_score, result_id),
        reverse=True,
    )
    return [by_id[result_id] for result_id in ordered_ids[:limit]], provenance


_TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


def select_diverse_results(
    results: Sequence[SearchResult],
    *,
    limit: int,
    max_per_document: int = 4,
    diversity_metadata_keys: Sequence[str] = (),
    max_per_diversity_group: int = 1,
    duplicate_threshold: float = 0.94,
) -> List[SearchResult]:
    """Remove near duplicates and bound document or metadata domination."""
    if max_per_diversity_group <= 0:
        raise ValueError("max_per_diversity_group must be greater than zero")
    selected: List[SearchResult] = []
    selected_tokens: List[set[str]] = []
    document_counts: Dict[str, int] = {}
    diversity_counts: Dict[Tuple[str, ...], int] = {}

    for result in results:
        if len(selected) >= limit:
            break
        if document_counts.get(result.document_id, 0) >= max_per_document:
            continue
        diversity_group = tuple(
            str(result.metadata.get(key, "")) for key in diversity_metadata_keys
        )
        if (
            diversity_group
            and diversity_counts.get(diversity_group, 0) >= max_per_diversity_group
        ):
            continue

        tokens = set(_TOKEN_PATTERN.findall((result.text or "").lower()))
        is_duplicate = False
        if tokens:
            for other in selected_tokens:
                union = tokens | other
                if union and len(tokens & other) / len(union) >= duplicate_threshold:
                    is_duplicate = True
                    break
        if is_duplicate:
            continue

        selected.append(result)
        selected_tokens.append(tokens)
        document_counts[result.document_id] = document_counts.get(result.document_id, 0) + 1
        if diversity_group:
            diversity_counts[diversity_group] = diversity_counts.get(diversity_group, 0) + 1

    return selected
