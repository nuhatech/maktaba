"""Base LLM interface for agentic query generation and evaluation."""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Dict, List, Tuple

from ..models import LLMUsage


class BaseLLM(ABC):
    """
    Abstract LLM interface for agentic RAG operations.

    Provides methods for:
    - Generating search queries from conversation history
    - Evaluating if retrieved sources can answer a question
    - Producing generic text/json completions for deep research pipeline

    Implementations should accept an optional `prompts` parameter in their constructor
    to allow users to customize the prompts used for query generation and source evaluation.
    See `maktaba.llm.prompts.AgenticPrompts` and `maktaba.llm.prompts.default_prompts()`.

    Example:
        from maktaba.llm import OpenAILLM, default_prompts

        # Use default prompts
        llm = OpenAILLM(api_key="sk-...")

        # Customize prompts
        custom_prompts = default_prompts(
            context="You are searching a medical knowledge base.",
            generate_queries_append="Focus on evidence-based queries."
        )
        llm = OpenAILLM(api_key="sk-...", prompts=custom_prompts)
    """

    @abstractmethod
    async def complete_text(
        self,
        *,
        system: str,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> Tuple[str, LLMUsage]:
        """
        Generate a text completion with explicit system + user prompts.

        Returns the generated text and usage information.
        """
        raise NotImplementedError

    @abstractmethod
    async def complete_json(
        self,
        *,
        system: str,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> Tuple[Dict[str, object], LLMUsage]:
        """
        Generate a JSON-formatted completion.

        Implementations should request structured output from the provider and
        return the parsed JSON object alongside usage metadata.
        """
        raise NotImplementedError

    async def stream_text(
        self,
        *,
        system: str,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> Tuple[AsyncIterator[str], LLMUsage]:
        """
        Stream text tokens for long-form generation.

        Default implementation yields a single chunk from :meth:`complete_text`.
        Providers with real streaming support should override this method.
        """

        text, usage = await self.complete_text(
            system=system,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        async def _generator() -> AsyncIterator[str]:
            yield text

        return _generator(), usage

    @abstractmethod
    async def generate_queries(
        self,
        messages: List[Tuple[str, str]],
        existing_queries: List[str],
        max_queries: int = 10,
    ) -> Tuple[List[Dict[str, str]], LLMUsage]:
        """
        Generate search queries from conversation history.

        Args:
            messages: List of (role, content) tuples representing chat history
            existing_queries: Previously generated queries to avoid duplication
            max_queries: Maximum number of queries to generate

        Returns:
            Tuple of (queries, usage):
                - queries: List of query dicts with keys:
                    - type: "semantic" or "keyword"
                    - query: The search query string
                - usage: LLMUsage object with token counts

        Example:
            queries = [
                {"type": "semantic", "query": "What is tawhid in Islam?"},
                {"type": "keyword", "query": "tawhid"},
            ]
            usage = LLMUsage(input_tokens=150, output_tokens=25)
            return queries, usage
        """
        raise NotImplementedError

    @abstractmethod
    async def evaluate_sources(
        self,
        messages: List[Tuple[str, str]],
        sources: List[str],
    ) -> Tuple[bool, LLMUsage]:
        """
        Evaluate if retrieved sources contain sufficient information to answer.

        Args:
            messages: List of (role, content) tuples representing chat history
            sources: List of retrieved text chunks

        Returns:
            Tuple of (can_answer, usage):
                - can_answer: True if sources can answer the question
                - usage: LLMUsage object with token counts
        """
        raise NotImplementedError

    async def condense_query(
        self,
        history: List[Tuple[str, str]],
        current_query: str,
    ) -> str:
        """
        Condense conversation history into a standalone query.

        Optional method - implementations can override.
        Default behavior returns the current query unchanged.

        Args:
            history: Previous conversation turns (role, content)
            current_query: Latest user question

        Returns:
            Standalone query incorporating context from history
        """
        return current_query
