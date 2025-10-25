"""Base LLM interface for agentic query generation and evaluation."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple


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
    ) -> List[Dict[str, str]]:
        """
        Generate search queries from conversation history.

        Args:
            messages: List of (role, content) tuples representing chat history
            existing_queries: Previously generated queries to avoid duplication
            max_queries: Maximum number of queries to generate

        Returns:
            List of query dicts with keys:
                - type: "semantic" or "keyword"
                - query: The search query string

        Example:
            [
                {"type": "semantic", "query": "What is tawhid in Islam?"},
                {"type": "keyword", "query": "tawhid"},
            ]
        """
        raise NotImplementedError

    @abstractmethod
    async def evaluate_sources(
        self,
        messages: List[Tuple[str, str]],
        sources: List[str],
    ) -> bool:
        """
        Evaluate if retrieved sources contain sufficient information to answer.

        Args:
            messages: List of (role, content) tuples representing chat history
            sources: List of retrieved text chunks

        Returns:
            True if sources can answer the question, False otherwise
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
