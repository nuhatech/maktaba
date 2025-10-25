"""LLM abstraction for query generation and evaluation in agentic RAG."""

from .base import BaseLLM
from .openai import OpenAILLM

__all__ = ["BaseLLM", "OpenAILLM"]
