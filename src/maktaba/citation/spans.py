"""Deterministic mapping of proposed evidence spans to source text."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Sequence

from ..collection_models import CollectionCandidate, EvidenceSpan
from ..models import SearchResult


@dataclass(slots=True, frozen=True)
class SpanVerification:
    """Result of checking a candidate against retrieved sources."""

    valid: bool
    span: EvidenceSpan | None = None
    reason: str | None = None


def _whitespace_tolerant_match(candidate: str, source: str) -> tuple[int, int] | None:
    tokens = re.split(r"\s+", candidate.strip())
    if not tokens or any(not token for token in tokens):
        return None
    pattern = r"\s+".join(re.escape(token) for token in tokens)
    match = re.search(pattern, source)
    return (match.start(), match.end()) if match else None


def verify_evidence_span(
    candidate: CollectionCandidate,
    sources: Sequence[SearchResult],
    *,
    min_characters: int = 8,
    require_exact_span: bool = True,
) -> SpanVerification:
    """Verify an extractor candidate and preserve the original source bytes.

    Exact matching is attempted first. A whitespace-only tolerant fallback is
    allowed because provider JSON can collapse line breaks; the returned span
    is always sliced from the original source, never from generated text.
    """

    source_by_id: Dict[str, SearchResult] = {source.id: source for source in sources}
    source = source_by_id.get(candidate.source_id)
    if source is None:
        return SpanVerification(valid=False, reason="unknown_source_id")

    source_text = source.text or ""
    proposed_text = candidate.text.strip()
    if len(proposed_text) < min_characters:
        return SpanVerification(valid=False, reason="span_too_short")

    start = source_text.find(proposed_text)
    end = start + len(proposed_text) if start >= 0 else -1
    if start < 0:
        matched = _whitespace_tolerant_match(proposed_text, source_text)
        if matched is None:
            return SpanVerification(valid=False, reason="span_not_in_source")
        start, end = matched
        if require_exact_span and not source_text[start:end].strip():
            return SpanVerification(valid=False, reason="empty_source_span")

    original_text = source_text[start:end]
    return SpanVerification(
        valid=True,
        span=EvidenceSpan(
            text=original_text,
            source_id=source.id,
            document_id=source.document_id,
            start_offset=start,
            end_offset=end,
            source_metadata=dict(source.metadata),
            attributes=dict(candidate.attributes),
        ),
    )
