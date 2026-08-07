"""Deterministic mapping of proposed evidence spans to source text."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, Sequence

from ..collection_models import CollectionCandidate, EvidenceSpan
from ..models import SearchResult


@dataclass(slots=True, frozen=True)
class SpanVerification:
    """Result of checking a candidate against retrieved sources."""

    valid: bool
    span: EvidenceSpan | None = None
    reason: str | None = None


SpanNormalizer = Callable[[str], str]


def _whitespace_tolerant_match(candidate: str, source: str) -> tuple[int, int] | None:
    tokens = re.split(r"\s+", candidate.strip())
    if not tokens or any(not token for token in tokens):
        return None
    pattern = r"\s+".join(re.escape(token) for token in tokens)
    match = re.search(pattern, source)
    return (match.start(), match.end()) if match else None


def _normalised_text_with_offsets(
    text: str,
    normalizer: SpanNormalizer,
) -> tuple[str, list[int], list[int]]:
    """Normalise characters while retaining their original source offsets."""
    output: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    for index, original_character in enumerate(text):
        for character in normalizer(original_character):
            if character.isspace():
                if output and output[-1] == " ":
                    ends[-1] = index + 1
                    continue
                character = " "
            output.append(character)
            starts.append(index)
            ends.append(index + 1)
    return "".join(output), starts, ends


def _normalizer_tolerant_match(
    candidate: str,
    source: str,
    normalizer: SpanNormalizer,
) -> tuple[int, int] | None:
    normalised_candidate, _, _ = _normalised_text_with_offsets(candidate.strip(), normalizer)
    normalised_source, starts, ends = _normalised_text_with_offsets(source, normalizer)
    normalised_candidate = normalised_candidate.strip()
    if not normalised_candidate:
        return None
    match_start = normalised_source.find(normalised_candidate)
    if match_start < 0:
        return None
    match_end = match_start + len(normalised_candidate)
    return starts[match_start], ends[match_end - 1]


def verify_evidence_span(
    candidate: CollectionCandidate,
    sources: Sequence[SearchResult],
    *,
    min_characters: int = 8,
    require_exact_span: bool = True,
    normalizer: SpanNormalizer | None = None,
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
        if matched is None and normalizer is not None:
            matched = _normalizer_tolerant_match(
                proposed_text,
                source_text,
                normalizer,
            )
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
