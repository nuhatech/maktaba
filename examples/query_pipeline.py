import asyncio
import os

from maktaba.embedding import OpenAIEmbedder  # requires OPENAI_API_KEY
from maktaba.models import VectorChunk
from maktaba.pipeline.query import QueryPipeline
from maktaba.reranking.cohere import CohereReranker  # optional; requires COHERE_API_KEY
from maktaba.storage.qdrant import QdrantStore


async def main():
    # Example: in-memory Qdrant for quick demo; replace with server URL in prod
    store = QdrantStore(url=":memory:", collection_name="demo")
    embedder = OpenAIEmbedder(
        api_key=os.getenv("OPENAI_API_KEY", "your-api-key-here"),
        model="text-embedding-3-large",
    )
    reranker = CohereReranker(use_api=False)  # set use_api=True with valid COHERE_API_KEY to call API

    # Create collection (dimension must match embedder)
    store.create_collection(dimension=embedder.dimension)

    # Index a few sample texts (same workflow as basic_usage, plus reranker/citations)
    texts = [
        "Tawhid is the Islamic concept of monotheism.",
        "Prayer (Salah) is performed five times a day.",
        "Zakat is obligatory charity in Islam.",
    ]
    vectors = await embedder.embed_batch(texts, input_type="document")
    chunks = [
        VectorChunk(id=f"doc_001#chunk_{i}", vector=vectors[i], metadata={"text": t})
        for i, t in enumerate(texts)
    ]
    await store.upsert(chunks)

    pipeline = QueryPipeline(embedder, store, reranker)
    out = await pipeline.search("What is Tawhid?", rerank=True)

    print(out["formatted_context"])  # contains [1]: ... blocks
    print(f"Citations: {len(out['citations'])}")


if __name__ == "__main__":
    asyncio.run(main())
