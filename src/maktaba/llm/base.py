"""Base LLM interface for agentic query generation and evaluation."""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from ..models import LLMUsage


class BaseLLM(ABC):
    """
    Abstract LLM interface for agentic RAG operations.

    Provides methods for:
    - Generating search queries from conversation history
    - Evaluating if retrieved sources can answer a question
    """

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
