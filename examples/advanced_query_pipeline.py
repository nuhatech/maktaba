import asyncio
import os

from maktaba.embedding import OpenAIEmbedder
from maktaba.models import VectorChunk
from maktaba.pipeline.query import QueryPipeline
from maktaba.reranking.cohere import CohereReranker
from maktaba.storage.qdrant import QdrantStore


async def main():
    store = QdrantStore(url=":memory:", collection_name="advanced")
    embedder = OpenAIEmbedder(
        api_key=os.getenv("OPENAI_API_KEY", "your-api-key-here"),
        model="text-embedding-3-large",
    )
    store.create_collection(dimension=embedder.dimension)

    # Load sample data into a namespace
    namespace = "theology"
    texts = [
        ("doc_tawhid#1", "Tawhid is the oneness of Allah."),
        ("doc_tawhid#2", "Shirk is associating partners with Allah."),
        ("doc_prayer#1", "Salah is performed five times daily."),
    ]
    vecs = await embedder.embed_batch([t for _, t in texts], input_type="document")
    chunks = [VectorChunk(id=i, vector=v, metadata={"text": t, "topic": "aqeedah"}) for (i, t), v in zip(texts, vecs)]
    await store.upsert(chunks, namespace=namespace)

    # Reranker: try API if COHERE_API_KEY provided; otherwise offline heuristic
    use_api = bool(os.getenv("COHERE_API_KEY"))
    reranker = CohereReranker(use_api=use_api)

    qp = QueryPipeline(embedder, store, reranker=reranker, namespace=namespace)
    out = await qp.search("What is Tawhid?", rerank=True, top_k=5, filter={"topic": "aqeedah"})

    print("— Formatted Context —")
    print(out["formatted_context"])
    print("— Summary —")
    print(f"Citations: {len(out['citations'])}")
    print(f"Results: {len(out['results'])}")


if __name__ == "__main__":
    asyncio.run(main())

