import asyncio
from typing import List

from maktaba.embedding.base import BaseEmbedder
from maktaba.models import VectorChunk
from maktaba.pipeline.query import QueryPipeline
from maktaba.storage.qdrant import QdrantStore


class SimpleEmbedder(BaseEmbedder):
    async def embed_batch(self, texts: List[str], input_type: str = "document") -> List[List[float]]:
        # Toy embedding: [len, vowels, q marks]
        def emb(t: str) -> List[float]:
            v = sum(1 for c in t.lower() if c in "aeiou")
            q = t.count("?")
            return [float(len(t)), float(v), float(q)]
        return [emb(t) for t in texts]

    @property
    def dimension(self) -> int:
        return 3

    @property
    def model(self) -> str:
        return "simple-embedder"


async def main():
    store = QdrantStore(url=":memory:", collection_name="custom")
    embedder = SimpleEmbedder()
    store.create_collection(dimension=embedder.dimension)

    texts = ["hello world", "what is tawhid?", "salah is prayer"]
    vecs = await embedder.embed_batch(texts, input_type="document")
    chunks = [VectorChunk(id=f"doc#{i}", vector=v, metadata={"text": t}) for i, (t, v) in enumerate(zip(texts, vecs))]
    await store.upsert(chunks)

    qp = QueryPipeline(embedder, store)
    out = await qp.search("What is Tawhid?", rerank=False)
    print(out["formatted_context"])  # [1]: ...


if __name__ == "__main__":
    asyncio.run(main())

