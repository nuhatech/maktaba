"""Embedding providers for Maktaba."""

from .base import BaseEmbedder
from .openai import OpenAIEmbedder

__all__ = [
    "BaseEmbedder",
    "OpenAIEmbedder",
]
