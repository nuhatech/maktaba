# -*- coding: utf-8 -*-
"""
Basic usage example for Maktaba.

This example demonstrates:
1. Using text-embedding-3-large (default)
2. camelCase parameters (topK, includeMetadata)
3. Chunk ID format: {doc_id}#{chunk_id}
4. Batch embedding (primary method)
"""

import asyncio
import os
from typing import List

from maktaba.embedding import OpenAIEmbedder
from maktaba.models import VectorChunk
from maktaba.storage import QdrantStore


async def main():
    # Initialize with defaults
    print("🚀 Initializing Maktaba configuration...")

    # 1. Create embedder with text-embedding-3-large (3072 dimensions)
    embedder = OpenAIEmbedder(
        api_key=os.getenv("OPENAI_API_KEY", "your-api-key-here"),
        model="text-embedding-3-large",  # default
    )

    print(f"✅ Embedder created: {embedder.model} ({embedder.dimension} dimensions)")

    # 2. Create Qdrant store with Pinecone-compatible interface
    # Options:
    # - "http://localhost:6333" = Local Qdrant server (requires Docker)
    # - ":memory:" = In-memory mode (no Docker needed, great for testing!)
    qdrant_url = os.getenv("QDRANT_URL", ":memory:")

    store = QdrantStore(
        url=qdrant_url,
        collection_name="maktaba_demo",
    )

    print(f"✅ Qdrant mode: {qdrant_url}")

    # Create collection if it doesn't exist (3072 dims for text-embedding-3-large)
    try:
        store.create_collection(dimension=embedder.dimension)
        print(f"✅ Collection created/verified: {store.collection_name}")
    except Exception as e:
        print(f"⚠️  Collection setup: {e}")

    # 3. Prepare sample documents (Islamic knowledge base example)
    documents = [
        {
            "id": "doc_001",
            "text": "Tawhid is the Islamic concept of the oneness and uniqueness of God.",
            "metadata": {"category": "theology", "source": "Islamic Studies"},
        },
        {
            "id": "doc_002",
            "text": "The five pillars of Islam are Shahada, Salah, Zakat, Sawm, and Hajj.",
            "metadata": {"category": "practice", "source": "Islamic Studies"},
        },
        {
            "id": "doc_003",
            "text": "Ihsan means to worship God as if you see Him, and if you cannot see Him, know that He sees you.",
            "metadata": {"category": "spirituality", "source": "Hadith"},
        },
    ]

    # 4. Embed documents in batch
    print("\n📝 Embedding documents...")
    texts = [doc["text"] for doc in documents]
    embeddings = await embedder.embed_batch(texts, input_type="document")

    print(f"✅ Embedded {len(embeddings)} documents")

    # 5. Create VectorChunks with ID format: {doc_id}#{chunk_id}
    chunks: List[VectorChunk] = []
    for idx, doc in enumerate(documents):
        chunk = VectorChunk(
            id=f"{doc['id']}#chunk_0",  # Pineconeformat!
            vector=embeddings[idx],
            metadata={
                "text": doc["text"],
                **doc["metadata"],
            },
        )
        chunks.append(chunk)

    # 6. Upsert to Qdrant
    print("\n💾 Storing vectors...")
    await store.upsert(chunks)
    print(f"✅ Stored {len(chunks)} chunks")

    # 7. Query with camelCase parameters (Pinecone style)
    print("\n🔍 Searching for 'What is Tawhid?'...")
    query_text = "What is Tawhid?"

    # Embed query
    query_vector = await embedder.embed_text(query_text, input_type="query")

    # Search with camelCase parameters!
    results = await store.query(
        vector=query_vector,
        topK=3,  # NOT top_k!
        includeMetadata=True,  # NOT include_metadata!
        filter={"category": "theology"},  # Optional metadata filter
    )

    print(f"\n📊 Found {len(results)} results:\n")

    for i, result in enumerate(results, 1):
        print(f"{i}. Score: {result.score:.4f}")
        print(f"   ID: {result.id}")
        print(f"   Text: {result.text}")
        print(f"   Document ID: {result.document_id}")
        print(f"   Chunk ID: {result.chunk_id}")
        print(f"   Category: {result.metadata.get('category')}")
        print()

    # 8. Demonstrate document deletion
    print("🧹 Cleaning up...")
    await store.delete_by_document("doc_001")
    print("✅ Deleted all chunks for doc_001")

    # Verify deletion
    remaining_ids = await store.list(limit=10)
    print(f"✅ Remaining chunks: {len(remaining_ids)}")

    print("\n🎉 Demo completed successfully!")
    print("\n🔑 Key features demonstrated:")
    print("   ✅ text-embedding-3-large (3072 dims)")
    print("   ✅ camelCase params (topK, includeMetadata)")
    print("   ✅ Chunk ID format: {doc_id}#{chunk_id}")
    print("   ✅ Batch embedding (primary method)")
    print("   ✅ Namespace support (Pinecone-style)")


if __name__ == "__main__":
    asyncio.run(main())
