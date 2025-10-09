import pytest

from maktaba.models import SearchResult
from maktaba.reranking.cohere import CohereReranker


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
