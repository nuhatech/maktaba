import asyncio
import os

from maktaba.chunking.unstructured import UnstructuredChunker
from maktaba.embedding import OpenAIEmbedder
from maktaba.pipeline.ingestion import IngestionPipeline
from maktaba.pipeline.query import QueryPipeline
from maktaba.storage.qdrant import QdrantStore


async def main():
    print("🧭 Initializing Maktaba ingestion demo...")

    # Qdrant in-memory for demo
    store = QdrantStore(url=":memory:", collection_name="demo")
    print("✔️  Qdrant store: :memory:")

    # Create embedder
    embedder = OpenAIEmbedder(
        api_key=os.getenv("OPENAI_API_KEY", "your-api-key-here"),
        model="text-embedding-3-large",
    )
    print(f"✔️  Embedder: {embedder.model} ({embedder.dimension} dims)")

    # Create collection with correct dimension
    try:
        store.create_collection(dimension=embedder.dimension)
        print("✔️  Collection created/verified: demo")
    except Exception as e:
        print(f"⚠️  Collection setup warning: {e}")

    # Chunker (uses Unstructured/LlamaIndex under the hood)
    chunker = UnstructuredChunker(strategy="auto", chunking_strategy="basic")

    # Ingestion pipeline
    pipeline = IngestionPipeline(chunker, embedder, store)

    # Ingest ONE document composed of multiple paragraphs (simulating pages/sections)
    print("\n📥 Ingesting a single multi-section document...")
    paragraphs = [
        "# Chapter 1: Tawhid\nTawhid is the Islamic concept of monotheism, affirming the oneness of Allah.",
        "Tawhid rejects all forms of shirk and emphasizes that worship is due to Allah alone.",
        "It has dimensions including Tawhid al-Rububiyyah, al-Uluhiyyah, and al-Asma wa al-Sifat.",
        "# Chapter 2: Salah\nSalah is the five daily prayers that structure a Muslim's day and connect them to Allah.",
        "It includes specific movements and recitations, performed facing the Qiblah.",
        "The congregational (jama'ah) prayer builds community and discipline.",
        "# Chapter 3: Zakat\nZakat is obligatory charity, a purification of wealth and a pillar of social justice.",
        "It is calculated on certain assets and distributed to eligible recipients.",
        "Through Zakat, wealth circulates and supports the needy and the common good.",
        "# Appendix: Belief and Practice\nCore beliefs inform acts of worship, shaping ethics and everyday life.",
        "The Shariah provides guidance that balances individual duty and communal welfare.",
        "Learning and sincerity strengthen one’s practice and understanding.",
    ]
    long_text = "\n\n".join(paragraphs)
    doc_id = "book_123"
    await pipeline.ingest_text(long_text, document_id=doc_id, filename=f"{doc_id}.txt")
    print(f"  ✔️  Ingested: {doc_id}")

    # Query with QueryPipeline (Phase 2)
    print("\n🔎 Searching for 'What is Tawhid?'...")
    qp = QueryPipeline(embedder, store, reranker=None)
    out = await qp.search("What is Tawhid?", rerank=False, top_k=10)

    # Show formatted context and a brief summary like basic_usage
    print("\n— Formatted Context —")
    print(out["formatted_context"])  # shows [n]: ... blocks
    print("\n— Summary —")
    print(f"Citations: {len(out['citations'])}")
    print(f"Results: {len(out['results'])}")


if __name__ == "__main__":
    asyncio.run(main())
