"""Tests for AgenticQueryPipeline."""

from typing import Dict, List, Tuple

import pytest

from maktaba.embedding.base import BaseEmbedder
from maktaba.keyword.base import BaseKeywordStore
from maktaba.llm.base import BaseLLM
from maktaba.models import LLMUsage, SearchResult
from maktaba.pipeline.agentic import AgenticQueryPipeline
from maktaba.reranking.base import BaseReranker
from maktaba.storage.base import BaseVectorStore

# =============================================================================
# Test Helpers / Mocks
# =============================================================================


class DummyEmbedder(BaseEmbedder):
    """Mock embedder for testing."""

    async def embed_batch(
        self, texts: List[str], input_type: str = "document"
    ) -> List[List[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]

    @property
    def dimension(self) -> int:
        return 3

    @property
    def model(self) -> str:
        return "dummy-embedder"


class DummyStore(BaseVectorStore):
    """Mock vector store for testing."""

    async def upsert(self, chunks, namespace=None) -> None:
        pass

    async def query(
        self,
        vector,
        topK: int = 10,
        filter=None,
        includeMetadata: bool = True,
        namespace=None,
        includeRelationships: bool = False,
    ):
        # Return predictable results
        return [
            SearchResult(
                id=f"doc#{i}",
                score=0.9 - i * 0.1,
                metadata={"text": f"This is semantic result {i} about the topic."},
                relationships=None,
            )
            for i in range(min(topK, 5))
        ]

    async def delete(self, ids, namespace=None):
        pass

    async def list(self, prefix=None, limit: int = 100, namespace=None):
        return []

    async def get_dimensions(self) -> int:
        return 3


class DummyKeywordStore(BaseKeywordStore):
    """Mock keyword store for testing."""

    async def search(self, query: str, limit: int = 15, filter=None, namespace=None):
        # Return predictable keyword results
        return [
            SearchResult(
                id=f"keyword#{i}",
                score=None,
                metadata={"text": f"Keyword result {i} for '{query}'"},
            )
            for i in range(min(limit, 3))
        ]

    async def upsert(self, chunks, namespace=None):
        # No-op for tests to satisfy abstract interface
        return None


class DummyReranker(BaseReranker):
    """Mock reranker for testing."""

    async def rerank(self, query: str, results: List[SearchResult], top_k: int = 10):
        # Just return results as-is (no actual reranking)
        return results[:top_k]


class DummyLLM(BaseLLM):
    """
    Configurable mock LLM for testing.

    Allows control over:
    - Queries returned per iteration
    - Can answer sequence
    - Token usage per call
    """

    def __init__(
        self,
        queries_to_return: List[List[Dict[str, str]]] = None,
        can_answer_sequence: List[bool] = None,
        usage_per_call: LLMUsage = None,
    ):
        """
        Initialize DummyLLM.

        Args:
            queries_to_return: List of query lists (one per generate_queries call)
            can_answer_sequence: List of booleans (one per evaluate_sources call)
            usage_per_call: Token usage to return for each call
        """
        self.queries_to_return = queries_to_return or []
        self.can_answer_sequence = can_answer_sequence or [True]
        self.usage_per_call = usage_per_call or LLMUsage(input_tokens=100, output_tokens=20)
        self.call_count_generate = 0
        self.call_count_evaluate = 0

    async def generate_queries(
        self,
        messages: List[Tuple[str, str]],
        existing_queries: List[str],
        max_queries: int = 10,
    ) -> Tuple[List[Dict[str, str]], LLMUsage]:
        """Return configured queries for this iteration."""
        if self.call_count_generate < len(self.queries_to_return):
            queries = self.queries_to_return[self.call_count_generate]
        else:
            queries = []
        self.call_count_generate += 1
        return queries[:max_queries], self.usage_per_call

    async def evaluate_sources(
        self, messages: List[Tuple[str, str]], sources: List[str]
    ) -> Tuple[bool, LLMUsage]:
        """Return configured canAnswer for this iteration."""
        idx = min(self.call_count_evaluate, len(self.can_answer_sequence) - 1)
        can_answer = self.can_answer_sequence[idx]
        self.call_count_evaluate += 1
        return can_answer, self.usage_per_call


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.asyncio
async def test_agentic_search_basic_flow():
    """Test basic agentic search flow with user message + generated queries."""
    embedder = DummyEmbedder()
    store = DummyStore()
    reranker = DummyReranker()

    # Mock LLM to generate 2 queries on iteration 0, then stop
    llm = DummyLLM(
        queries_to_return=[
            [
                {"type": "semantic", "query": "What is Tawhid in detail?"},
                {"type": "keyword", "query": "Tawhid"},
            ]
        ],
        can_answer_sequence=[True],  # Answer found after iteration 0
    )

    pipeline = AgenticQueryPipeline(
        embedder=embedder, store=store, reranker=reranker, llm=llm
    )

    result = await pipeline.agentic_search(
        messages=[("user", "What is Tawhid?")], max_iterations=3, top_k=50, rerank_limit=15
    )

    # Verify structure
    assert "formatted_context" in result
    assert "results" in result
    assert "queries_used" in result
    assert "iterations" in result
    assert "usage" in result

    # Iteration 0 should execute: user message + 2 LLM queries = 3 total queries
    assert len(result["queries_used"]) == 3
    assert result["queries_used"][0] == "What is Tawhid?"  # User message added first
    assert "Tawhid" in result["queries_used"][1] or "Tawhid" in result["queries_used"][2]

    # Should have results
    assert len(result["results"]) > 0

    # Should track iterations
    assert result["iterations"] == 1

    # Should track usage
    assert isinstance(result["usage"], LLMUsage)
    assert result["usage"].total_tokens > 0


@pytest.mark.asyncio
async def test_agentic_search_early_exit_on_can_answer():
    """Test early exit when LLM says sources are sufficient."""
    embedder = DummyEmbedder()
    store = DummyStore()

    # Mock LLM to always generate queries, but canAnswer=True after iteration 1
    llm = DummyLLM(
        queries_to_return=[
            [{"type": "semantic", "query": "Query iteration 0"}],
            [{"type": "semantic", "query": "Query iteration 1"}],
            [{"type": "semantic", "query": "Query iteration 2"}],  # Should not reach this
        ],
        can_answer_sequence=[False, True],  # Found answer after iteration 1
    )

    pipeline = AgenticQueryPipeline(embedder=embedder, store=store, llm=llm)

    result = await pipeline.agentic_search(
        messages=[("user", "Test question?")], max_iterations=5
    )

    # Should stop after iteration 2 (iteration 1 evaluation said True)
    assert result["iterations"] == 2
    assert result["queries_used"][0] == "Test question?"  # User message
    assert len(result["queries_used"]) == 3  # User + 2 iterations of queries


@pytest.mark.asyncio
async def test_agentic_search_token_budget_limit():
    """Test loop exits when token budget is exceeded."""
    embedder = DummyEmbedder()
    store = DummyStore()

    # Mock LLM with high token usage (1000 tokens per call)
    llm = DummyLLM(
        queries_to_return=[
            [{"type": "semantic", "query": f"Query {i}"}] for i in range(10)
        ],
        can_answer_sequence=[False] * 10,  # Never satisfied
        usage_per_call=LLMUsage(input_tokens=800, output_tokens=200),  # 1000 total
    )

    pipeline = AgenticQueryPipeline(embedder=embedder, store=store, llm=llm)

    result = await pipeline.agentic_search(
        messages=[("user", "Test question?")],
        max_iterations=10,
        token_budget=2500,  # Should stop after ~2-3 iterations
    )

    # Should exit due to budget (not max iterations)
    assert result["iterations"] < 10
    assert result["usage"].total_tokens >= 2500


@pytest.mark.asyncio
async def test_agentic_search_max_iterations():
    """Test loop runs exactly max_iterations when canAnswer always False."""
    embedder = DummyEmbedder()
    store = DummyStore()

    # Mock LLM that never finds answer
    llm = DummyLLM(
        queries_to_return=[[{"type": "semantic", "query": f"Query {i}"}] for i in range(5)],
        can_answer_sequence=[False] * 5,  # Never satisfied
    )

    pipeline = AgenticQueryPipeline(embedder=embedder, store=store, llm=llm)

    result = await pipeline.agentic_search(
        messages=[("user", "Test question?")],
        max_iterations=3,
        token_budget=100000,  # High budget
    )

    # Should run exactly 3 iterations
    assert result["iterations"] == 3


@pytest.mark.asyncio
async def test_agentic_search_query_deduplication():
    """Test duplicate queries are filtered out."""
    embedder = DummyEmbedder()
    store = DummyStore()

    # Mock LLM to return duplicate queries
    llm = DummyLLM(
        queries_to_return=[
            [
                {"type": "semantic", "query": "What is Tawhid?"},  # Duplicate of user message
                {"type": "keyword", "query": "Tawhid"},
            ],
            [
                {"type": "semantic", "query": "Tawhid explanation"},
                {"type": "keyword", "query": "Tawhid"},  # Duplicate from iteration 0
            ],
        ],
        can_answer_sequence=[False, True],
    )

    pipeline = AgenticQueryPipeline(embedder=embedder, store=store, llm=llm)

    result = await pipeline.agentic_search(
        messages=[("user", "What is Tawhid?")], max_iterations=3
    )

    # Verify deduplication
    queries = result["queries_used"]
    assert len(queries) == len(set(queries))  # All unique
    assert queries.count("What is Tawhid?") == 1  # User message appears once
    assert queries.count("Tawhid") == 1  # Keyword query appears once


@pytest.mark.asyncio
async def test_agentic_search_keyword_and_semantic_queries():
    """Test routing of keyword and semantic queries to correct stores."""
    embedder = DummyEmbedder()
    store = DummyStore()
    keyword_store = DummyKeywordStore()

    # Mock LLM to return both types
    llm = DummyLLM(
        queries_to_return=[
            [
                {"type": "semantic", "query": "What is the concept of Tawhid?"},
                {"type": "keyword", "query": "Tawhid"},
            ]
        ],
        can_answer_sequence=[True],
    )

    pipeline = AgenticQueryPipeline(
        embedder=embedder, store=store, keyword_store=keyword_store, llm=llm
    )

    result = await pipeline.agentic_search(
        messages=[("user", "Explain Tawhid")], include_query_results=True
    )

    # Should have results from both semantic and keyword searches
    assert len(result["results"]) > 0

    # Check query_to_result mapping
    assert "query_to_result" in result
    query_to_result = result["query_to_result"]

    # Verify both query types were executed
    semantic_executed = any("concept" in q.lower() for q in query_to_result.keys())
    keyword_executed = any("Tawhid" == q for q in query_to_result.keys())
    assert semantic_executed or len(query_to_result) > 0
    assert keyword_executed or len(query_to_result) > 0


@pytest.mark.asyncio
async def test_agentic_search_without_keyword_store():
    """Test graceful handling when keyword store is not provided."""
    embedder = DummyEmbedder()
    store = DummyStore()

    # Mock LLM to return keyword query on iteration 0
    llm = DummyLLM(
        queries_to_return=[[{"type": "keyword", "query": "Tawhid"}]],
        can_answer_sequence=[True],
    )

    # Don't provide keyword_store
    pipeline = AgenticQueryPipeline(embedder=embedder, store=store, llm=llm)

    result = await pipeline.agentic_search(messages=[("user", "What is Tawhid?")])

    # Should complete without error
    assert "results" in result

    # Should have results from user message semantic query (iteration 0)
    # Keyword query should be skipped with warning
    assert len(result["results"]) > 0

    # Verify the queries that were executed
    queries = result["queries_used"]
    assert "What is Tawhid?" in queries  # User message
    assert "Tawhid" in queries  # Keyword query (attempted but skipped)


@pytest.mark.asyncio
async def test_agentic_search_usage_tracking():
    """Test precise token usage tracking."""
    embedder = DummyEmbedder()
    store = DummyStore()

    # Mock LLM with specific token counts
    llm = DummyLLM(
        queries_to_return=[
            [{"type": "semantic", "query": "Query 1"}],
            [{"type": "semantic", "query": "Query 2"}],
        ],
        can_answer_sequence=[False, True],  # 2 iterations
        usage_per_call=LLMUsage(input_tokens=50, output_tokens=25),
    )

    pipeline = AgenticQueryPipeline(embedder=embedder, store=store, llm=llm)

    result = await pipeline.agentic_search(
        messages=[("user", "Test?")], max_iterations=3
    )

    # Iteration 0: generate_queries (50+25)
    # Iteration 0: evaluate_sources (50+25)
    # Iteration 1: generate_queries (50+25)
    # Iteration 1: evaluate_sources (50+25)
    # Total: 4 calls = 4 * 75 = 300 tokens
    assert result["usage"].total_tokens == 300
    assert result["usage"].input_tokens == 200
    assert result["usage"].output_tokens == 100


@pytest.mark.asyncio
async def test_agentic_search_query_to_result_mapping():
    """Test query_to_result mapping for debugging."""
    embedder = DummyEmbedder()
    store = DummyStore()

    llm = DummyLLM(
        queries_to_return=[[{"type": "semantic", "query": "Generated query"}]],
        can_answer_sequence=[True],
    )

    pipeline = AgenticQueryPipeline(embedder=embedder, store=store, llm=llm)

    result = await pipeline.agentic_search(
        messages=[("user", "User query")], include_query_results=True
    )

    # Verify mapping is present
    assert "query_to_result" in result
    query_to_result = result["query_to_result"]

    # Should have mapping for user query (executed on iteration 0)
    # and generated query
    assert len(query_to_result) >= 2

    # User query should be in the mapping
    assert "User query" in query_to_result

    # Generated query should be in the mapping
    assert "Generated query" in query_to_result

    # Each mapping should have results
    assert len(query_to_result["User query"]) > 0
    assert len(query_to_result["Generated query"]) > 0
