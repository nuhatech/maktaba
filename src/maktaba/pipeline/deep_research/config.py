"""Configuration objects for the deep research pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(slots=True)
class DeepResearchConfig:
    """Runtime knobs controlling deep research iteration, sources, and output."""

    budget: int = 2
    max_queries: int = 2
    max_sources: int = 5
    max_tokens: int = 8192

    def copy(self) -> "DeepResearchConfig":
        """Return a shallow copy so callers can mutate without side effects."""
        return DeepResearchConfig(
            budget=self.budget,
            max_queries=self.max_queries,
            max_sources=self.max_sources,
            max_tokens=self.max_tokens,
        )


DEFAULT_DEEP_RESEARCH_CONFIG = DeepResearchConfig()


def current_date_context(now: datetime | None = None) -> str:
    """Return the temporal context string used across deep-research prompts."""
    dt = now or datetime.now(timezone.utc)
    month_name = dt.strftime("%B")

    return (
        f"Current date is {dt.year}-{dt.month:02}-{dt.day:02} ({month_name} {dt.day}, {dt.year}).\n"
        f"When searching for recent information, prioritize results from the current year ({dt.year}) and month ({month_name} {dt.year}).\n"
        f"For queries about recent developments, include the current year ({dt.year}) in your search terms.\n"
        "When ranking search results, consider recency as a factor - newer information is generally more relevant for current topics."
    )
