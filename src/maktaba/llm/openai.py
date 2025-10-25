"""OpenAI LLM implementation for agentic query generation and evaluation."""

import json
from typing import Dict, List, Optional, Tuple

from ..logging import get_logger
from ..models import LLMUsage
from .base import BaseLLM


class OpenAILLM(BaseLLM):
    """
    OpenAI implementation for agentic RAG operations.

    Uses OpenAI's Chat Completions API for query generation and evaluation.
    Falls back gracefully if OpenAI is unavailable.
    """

    GENERATE_QUERIES_PROMPT = """Given a user question (or a chat history), list the appropriate search queries to find answers.

    There are two types of search: keyword search and semantic search. You should return a maximum of {max_queries} queries.

    A good keyword search query contains one (or max two) words that are key to finding the result.
    A good semantic search query is a complete question or phrase that captures the user's intent.

    The results should be returned in JSON format:
    {{"queries": [{{"type": "keyword", "query": "..."}}, {{"type": "semantic", "query": "..."}}]}}"""

    EVALUATE_SOURCES_PROMPT = """You are a research assistant. You will be provided with a chat history and a list of sources.
    Evaluate if the sources contain sufficient information to answer the user's question.

    Return your evaluation in JSON format:
    {{"canAnswer": true}} or {{"canAnswer": false}}

    Only return true if the sources directly contain the information needed to provide a comprehensive answer."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        timeout_s: float = 30.0,
    ) -> None:
        """
        Initialize OpenAI LLM.

        Args:
            api_key: OpenAI API key (or None to use environment variable)
            model: Model name (default: gpt-4o-mini for cost efficiency)
            temperature: Sampling temperature (default: 0 for deterministic)
            timeout_s: Request timeout in seconds
        """
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.timeout_s = timeout_s
        self._logger = get_logger("maktaba.llm.openai")

        # Lazy client initialization
        self._client = None
        try:
            from openai import AsyncOpenAI

            self._OpenAI = AsyncOpenAI
        except ImportError:  # pragma: no cover
            self._OpenAI = None
            self._logger.warning("openai package not installed; agentic mode unavailable")

    def _get_client(self):
        """Lazy initialize OpenAI client."""
        if self._client is None and self._OpenAI is not None:
            self._client = self._OpenAI(api_key=self.api_key, timeout=self.timeout_s)
        return self._client

    def _format_chat_history(self, messages: List[Tuple[str, str]]) -> str:
        """Format chat history as text."""
        lines = []
        for role, content in messages:
            label = "Human" if role == "user" else "Assistant"
            lines.append(f"{label}: {content}")
        return "\n\n".join(lines)

    async def generate_queries(
        self,
        messages: List[Tuple[str, str]],
        existing_queries: List[str],
        max_queries: int = 10,
    ) -> Tuple[List[Dict[str, str]], LLMUsage]:
        """
        Generate search queries using OpenAI.

        Args:
            messages: Chat history as (role, content) tuples
            existing_queries: Previously generated queries to avoid
            max_queries: Maximum number of queries to generate

        Returns:
            Tuple of (queries, usage):
                - queries: List of {"type": "semantic"|"keyword", "query": "..."} dicts
                - usage: LLMUsage with token counts
        """
        client = self._get_client()
        if client is None:
            self._logger.warning("OpenAI client unavailable, returning empty queries")
            return [], LLMUsage()

        try:
            # Build prompt
            chat_history = self._format_chat_history(messages)
            existing_queries_text = ""
            if existing_queries:
                existing_queries_text = (
                    "\n\nThe queries you return should be different from these ones that were tried so far:\n"
                    + "\n".join(f"- {q}" for q in existing_queries)
                )

            user_prompt = f"{existing_queries_text}\n\nChat history:\n{chat_history}".strip()

            # Call OpenAI
            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.GENERATE_QUERIES_PROMPT.format(max_queries=max_queries)},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )

            # Extract usage from response
            usage = LLMUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens
            )

            content = response.choices[0].message.content
            if not content:
                return [], usage

            # Parse JSON response
            result = json.loads(content)
            queries = result.get("queries", [])

            self._logger.info(
                f"Generated {len(queries)} queries: {[q['query'] for q in queries]} "
                f"(tokens: {usage.total_tokens})"
            )
            return queries[:max_queries], usage

        except Exception as e:
            self._logger.error(f"Query generation failed: {e}", exc_info=True)
            return [], LLMUsage()

    async def evaluate_sources(
        self,
        messages: List[Tuple[str, str]],
        sources: List[str],
    ) -> Tuple[bool, LLMUsage]:
        """
        Evaluate if sources can answer the question.

        Args:
            messages: Chat history as (role, content) tuples
            sources: List of retrieved text chunks

        Returns:
            Tuple of (can_answer, usage):
                - can_answer: True if sources contain sufficient information
                - usage: LLMUsage with token counts
        """
        client = self._get_client()
        if client is None:
            self._logger.warning("OpenAI client unavailable, assuming sources are sufficient")
            return True, LLMUsage()  # Optimistic fallback

        try:
            # Format inputs
            chat_history = self._format_chat_history(messages)
            sources_text = "\n\n".join(
                f"<source_{i+1}>\n{source}\n</source_{i+1}>"
                for i, source in enumerate(sources)
            )

            user_prompt = f"Chat history:\n{chat_history}\n\nRetrieved sources:\n{sources_text}"

            # Call OpenAI
            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.EVALUATE_SOURCES_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )

            # Extract usage from response
            usage = LLMUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens
            )

            content = response.choices[0].message.content
            if not content:
                return True, usage  # Optimistic fallback

            # Parse JSON response
            result = json.loads(content)
            can_answer = result.get("canAnswer", True)

            self._logger.info(f"Source evaluation: canAnswer={can_answer} (tokens: {usage.total_tokens})")
            return can_answer, usage

        except Exception as e:
            self._logger.error(f"Source evaluation failed: {e}", exc_info=True)
            return True, LLMUsage()  # Optimistic fallback
