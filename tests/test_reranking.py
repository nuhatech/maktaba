import pytest

from maktaba.models import SearchResult
from maktaba.reranking.cohere import CohereReranker
from maktaba.reranking.zeroentropy import ZeroEntropyReranker


@pytest.mark.asyncio
async def test_cohere_reranker_offline_heuristic_orders_results():
    rr = CohereReranker(use_api=False)
    query = "What is Tawhid?"
    results = [
        SearchResult(id="doc#1", score=0.5, metadata={"text": "Completely unrelated."}),
        SearchResult(id="doc#2", score=0.5, metadata={"text": "Tawhid is the oneness of Allah."}),
    ]

    ranked = await rr.rerank(query, results, top_k=2)
    # Expect the item with keyword to rank first
    assert ranked[0].id == "doc#2"


@pytest.mark.asyncio
async def test_zeroentropy_reranker_offline_heuristic_orders_results():
    """Test ZeroEntropyReranker with offline heuristic fallback."""
    rr = ZeroEntropyReranker(use_api=False)
    query = "What is Tawhid?"
    results = [
        SearchResult(id="doc#1", score=0.5, metadata={"text": "Completely unrelated."}),
        SearchResult(id="doc#2", score=0.5, metadata={"text": "Tawhid is the oneness of Allah."}),
    ]

    ranked = await rr.rerank(query, results, top_k=2)
    # Expect the item with keyword to rank first
    assert ranked[0].id == "doc#2"
    assert len(ranked) == 2


@pytest.mark.asyncio
async def test_zeroentropy_reranker_empty_results():
    """Test ZeroEntropyReranker with empty results list."""
    rr = ZeroEntropyReranker(use_api=False)
    query = "What is RAG?"
    results = []

    ranked = await rr.rerank(query, results)
    assert ranked == []


@pytest.mark.asyncio
async def test_zeroentropy_reranker_respects_top_k():
    """Test ZeroEntropyReranker respects top_k parameter."""
    rr = ZeroEntropyReranker(use_api=False)
    query = "Retrieval Augmented Generation"
    results = [
        SearchResult(id="doc#1", score=0.5, metadata={"text": "Completely unrelated."}),
        SearchResult(id="doc#2", score=0.5, metadata={"text": "RAG combines retrieval with generation."}),
        SearchResult(id="doc#3", score=0.5, metadata={"text": "Retrieval systems are important."}),
        SearchResult(id="doc#4", score=0.5, metadata={"text": "Generation models use transformers."}),
    ]

    ranked = await rr.rerank(query, results, top_k=2)
    # Should return only 2 results
    assert len(ranked) == 2
    # Top results should contain keywords from query
    assert any(word in ranked[0].text.lower() for word in ["retrieval", "generation", "rag"])
