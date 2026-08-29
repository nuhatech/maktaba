import json

import httpx
import pytest

from maktaba.models import SearchResult
from maktaba.reranking.cohere import CohereReranker
from maktaba.reranking.voyage import VoyageReranker
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


@pytest.mark.asyncio
async def test_voyage_reranker_uses_api_contract_and_maps_indices():
    results = [
        SearchResult(id="doc#1", metadata={"text": "باب زكاة العروض"}),
        SearchResult(id="doc#2", metadata={"text": "ضبط عرض المسعى"}),
        SearchResult(id="doc#3", metadata={"text": "عرض المسألة"}),
    ]

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.url == "https://api.voyageai.com/v1/rerank"
        assert request.headers["Authorization"] == "Bearer voyage-test"
        assert body == {
            "model": "rerank-2.5",
            "query": "ضبط عرض المسعى",
            "documents": ["باب زكاة العروض", "ضبط عرض المسعى", "عرض المسألة"],
            "top_k": 2,
            "return_documents": False,
            "truncation": True,
        }
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "relevance_score": 0.99},
                    {"index": 2, "relevance_score": 0.5},
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        reranker = VoyageReranker(api_key="voyage-test", http_client=client)
        ranked = await reranker.rerank("ضبط عرض المسعى", results, top_k=2)

    assert [result.id for result in ranked] == ["doc#2", "doc#3"]


@pytest.mark.asyncio
async def test_voyage_reranker_falls_back_without_credentials(monkeypatch):
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    results = [
        SearchResult(id="doc#1", metadata={"text": "باب زكاة العروض"}),
        SearchResult(id="doc#2", metadata={"text": "ضبط عرض المسعى"}),
    ]

    reranker = VoyageReranker(use_api=True)
    ranked = await reranker.rerank("عرض المسعى", results, top_k=2)

    assert reranker.use_api is False
    assert [result.id for result in ranked] == ["doc#2", "doc#1"]


@pytest.mark.asyncio
async def test_voyage_reranker_falls_back_on_invalid_api_response():
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 99}]})

    results = [
        SearchResult(id="doc#1", metadata={"text": "unrelated"}),
        SearchResult(id="doc#2", metadata={"text": "tawhid evidence"}),
    ]
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        reranker = VoyageReranker(api_key="voyage-test", http_client=client)
        ranked = await reranker.rerank("tawhid", results, top_k=2)

    assert [result.id for result in ranked] == ["doc#2", "doc#1"]
