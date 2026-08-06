"""Lightweight data models used throughout the deep research pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterable, List, Sequence

from ...models import LLMUsage, SearchResult


@dataclass(slots=True)
class ResearchPlan:
    """Structured representation of an LLM-generated plan."""

    queries: List[str] = field(default_factory=list)

    @classmethod
    def from_iterable(cls, items: Iterable[str]) -> "ResearchPlan":
        return cls(list(items))


@dataclass(slots=True)
class SourceList:
    """Structured representation of kept source indices (1-based)."""

    sources: List[int] = field(default_factory=list)

    @classmethod
    def from_iterable(cls, items: Iterable[int]) -> "SourceList":
        return cls(list(items))


@dataclass(slots=True)
class SearchResultView:
    """
    Normalised representation of a search result used in deep research.

    ``content`` is the compact planning summary. ``raw_content`` preserves the
    retrieved evidence so final synthesis never has to cite a lossy summary.
    """

    id: str
    content: str
    metadata: dict[str, Any] | None = None
    raw_content: str | None = None

    @classmethod
    def from_search_result(cls, result: SearchResult, *, content: str | None = None) -> "SearchResultView":
        return cls(
            id=result.id,
            metadata=(result.metadata or {}),
            content=content if content is not None else (result.text or ""),
            raw_content=result.text or "",
        )

    def to_string(self) -> str:
        return (
            f"ID: {self.id}\n"
            f"Metadata: {self.metadata}\n"
            f"Content: {self.content[:1000]}"
        )

    def short_string(self) -> str:
        return self.to_string()

    def to_evidence_string(self) -> str:
        return (
            f"ID: {self.id}\n"
            f"Metadata: {self.metadata}\n"
            f"Content: {self.raw_content if self.raw_content is not None else self.content}"
        )


@dataclass(slots=True)
class SearchResultsCollection:
    """Container for working with collections of ``SearchResultView`` items."""

    results: List[SearchResultView] = field(default_factory=list)

    def add(self, other: "SearchResultsCollection") -> "SearchResultsCollection":
        return SearchResultsCollection(self.results + other.results)

    def dedup(self) -> "SearchResultsCollection":
        seen: set[str] = set()
        unique: list[SearchResultView] = []
        for result in self.results:
            if result.id in seen:
                continue
            seen.add(result.id)
            unique.append(result)
        return SearchResultsCollection(unique)

    def to_string(self) -> str:
        return "\n\n".join(f"[{idx + 1}] {result.to_string()}" for idx, result in enumerate(self.results))

    def short_string(self) -> str:
        return "\n\n".join(f"[{idx + 1}] {result.short_string()}" for idx, result in enumerate(self.results))

    def to_evidence_string(self) -> str:
        return "\n\n".join(
            f"[{idx + 1}] {result.to_evidence_string()}" for idx, result in enumerate(self.results)
        )

    @classmethod
    def from_sequence(cls, items: Sequence[SearchResultView]) -> "SearchResultsCollection":
        return cls(list(items))

    @classmethod
    def empty(cls) -> "SearchResultsCollection":
        return cls([])


@dataclass(slots=True)
class IterativeResearchOutput:
    final_results: SearchResultsCollection
    queries_used: List[str]


@dataclass(slots=True)
class FilteredResultsData:
    filtered_results: SearchResultsCollection
    source_indices: List[int]


@dataclass(slots=True)
class DeepResearchAnswer:
    stream: AsyncIterator[str]
    usage: LLMUsage
    queries_used: List[str]
    source_indices: List[int]
    results: SearchResultsCollection
