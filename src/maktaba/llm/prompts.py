"""Prompt templates for agentic LLM operations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(slots=True)
class AgenticPrompts:
    """Prompt templates for agentic query generation and source evaluation."""

    generate_queries_prompt: str
    evaluate_sources_prompt: str

    def copy(self) -> "AgenticPrompts":
        """Return a copy of the prompts."""
        return AgenticPrompts(
            generate_queries_prompt=self.generate_queries_prompt,
            evaluate_sources_prompt=self.evaluate_sources_prompt,
        )


def _with_context(body: str, header: str | None) -> str:
    """Prepend optional context header to prompt body."""
    prefix = (header.strip() + "\n\n") if header else ""
    return f"{prefix}{body.strip()}"


def _current_date_context(now: datetime | None = None) -> str:
    """Return the temporal context string for prompts."""
    dt = now or datetime.now(timezone.utc)
    month_name = dt.strftime("%B")

    return (
        f"Current date is {dt.year}-{dt.month:02}-{dt.day:02} ({month_name} {dt.day}, {dt.year}).\n"
        f"When searching for recent information, prioritize results from the current year ({dt.year}) and month ({month_name} {dt.year}).\n"
        f"For queries about recent developments, include the current year ({dt.year}) in your search terms.\n"
        "When ranking search results, consider recency as a factor - newer information is generally more relevant for current topics."
    )


def default_prompts(
    now: datetime | None = None,
    *,
    context: str | None = None,
    generate_queries_append: str | None = None,
    evaluate_sources_append: str | None = None,
) -> AgenticPrompts:
    """
    Return default prompt templates for agentic operations.

    Args:
        now: Current datetime for temporal context (defaults to current UTC time)
        context: Custom context header (overrides default date context if provided)
        generate_queries_append: Additional instructions to append to query generation prompt
        evaluate_sources_append: Additional instructions to append to source evaluation prompt

    Returns:
        AgenticPrompts instance with formatted prompts

    Example:
        # Use default prompts
        prompts = default_prompts()

        # Add custom context
        prompts = default_prompts(
            context="You are searching a medical knowledge base.",
            generate_queries_append="Focus on evidence-based medical queries."
        )

        # Use with LLM
        llm = OpenAILLM(api_key="...", prompts=prompts)
    """
    header = context if context is not None else _current_date_context(now)

    def _append(base: str, extra: str | None) -> str:
        """Append extra instructions to base prompt if provided."""
        return f"{base.rstrip()}\n\n{extra.strip()}" if extra else base

    # Query Generation Prompt
    generate_queries_base = """Given a user question (or a chat history), list the appropriate search queries to find answers.

There are two types of search: keyword search and semantic search. You should return a maximum of {max_queries} queries.

A good keyword search query contains one (or max two) words that are key to finding the result.
A good semantic search query is a complete question or phrase that captures the user's intent.

The results should be returned in json format:
{{"queries": [{{"type": "keyword", "query": "..."}}, {{"type": "semantic", "query": "..."}}]}}"""

    generate_queries_prompt = _with_context(
        _append(generate_queries_base, generate_queries_append),
        header,
    )

    # Source Evaluation Prompt
    evaluate_sources_base = """You are a research assistant. You will be provided with a chat history and a list of sources.
Evaluate if the sources contain sufficient information to answer the user's question.

Return your evaluation in json format:
{{"canAnswer": true}} or {{"canAnswer": false}}

Only return true if the sources directly contain the information needed to provide a comprehensive answer."""

    evaluate_sources_prompt = _with_context(
        _append(evaluate_sources_base, evaluate_sources_append),
        header,
    )

    return AgenticPrompts(
        generate_queries_prompt=generate_queries_prompt,
        evaluate_sources_prompt=evaluate_sources_prompt,
    )
