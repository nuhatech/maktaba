"""Deterministic citation and evidence checks for generated answers."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Awaitable, Callable, Dict, List, Optional, Sequence

from ..models import SearchResult

EntailmentChecker = Callable[[str, Sequence[SearchResult]], Awaitable[bool]]


@dataclass(slots=True)
class AnswerVerificationReport:
    """Machine-readable post-generation verification result."""

    valid: bool
    cited_source_ids: List[str] = field(default_factory=list)
    invalid_citations: List[str] = field(default_factory=list)
    unsupported_quotes: List[str] = field(default_factory=list)
    uncited_claims: List[str] = field(default_factory=list)
    unsupported_claims: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


_NUMERIC_CITATION = re.compile(r"\[(\d+)\]")
_QUOTE = re.compile(r'["“](.{8,}?)["”]')
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")


async def verify_answer(
    answer: str,
    evidence: Sequence[SearchResult],
    *,
    require_citations: bool = True,
    verify_quotes: bool = True,
    entailment_checker: Optional[EntailmentChecker] = None,
) -> AnswerVerificationReport:
    """Verify references, exact quotes, and optionally claim entailment.

    Numeric citations are interpreted using the evidence order (``[1]`` is the
    first result). The optional checker receives each claim and only the chunks
    cited by that claim. It can be backed by any LLM or local NLI model.
    """
    cited_ids: List[str] = []
    invalid: List[str] = []
    unsupported_quotes: List[str] = []
    uncited_claims: List[str] = []
    unsupported_claims: List[str] = []

    for raw_index in _NUMERIC_CITATION.findall(answer):
        index = int(raw_index)
        if 1 <= index <= len(evidence):
            source_id = evidence[index - 1].id
            if source_id not in cited_ids:
                cited_ids.append(source_id)
        elif raw_index not in invalid:
            invalid.append(raw_index)

    # Writers commonly place citations immediately after sentence punctuation
    # (``claim. [1]``). Move them before the punctuation for claim association.
    claim_text = re.sub(
        r"([.!?])\s+((?:\[\d+\]\s*)+)",
        lambda match: f" {match.group(2).strip()}{match.group(1)} ",
        answer,
    )
    sentences = [sentence.strip() for sentence in _SENTENCE.split(claim_text) if sentence.strip()]
    for sentence in sentences:
        citation_indices = [int(value) for value in _NUMERIC_CITATION.findall(sentence)]
        cited_results = [evidence[index - 1] for index in citation_indices if 1 <= index <= len(evidence)]
        claim = _NUMERIC_CITATION.sub("", sentence).strip()
        # Ignore headings, tiny fragments, and pure list labels.
        is_substantive = len(re.findall(r"\w+", claim, flags=re.UNICODE)) >= 5
        if require_citations and is_substantive and not citation_indices:
            uncited_claims.append(claim)

        if verify_quotes:
            searchable = "\n".join(result.text or "" for result in (cited_results or evidence)).casefold()
            for quote in _QUOTE.findall(claim):
                if quote.casefold() not in searchable:
                    unsupported_quotes.append(quote)

        if entailment_checker is not None and is_substantive:
            if not cited_results or not await entailment_checker(claim, cited_results):
                unsupported_claims.append(claim)

    valid = not (invalid or unsupported_quotes or uncited_claims or unsupported_claims)
    return AnswerVerificationReport(
        valid=valid,
        cited_source_ids=cited_ids,
        invalid_citations=invalid,
        unsupported_quotes=unsupported_quotes,
        uncited_claims=uncited_claims,
        unsupported_claims=unsupported_claims,
    )
