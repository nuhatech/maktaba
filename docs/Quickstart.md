% Quickstart

Install with UV:

```bash
cd maktaba
uv sync --extra openai --extra qdrant
```

Basic usage:

```python
from maktaba.embedding import OpenAIEmbedder
from maktaba.storage import QdrantStore
from maktaba.models import VectorChunk

embedder = OpenAIEmbedder(api_key="sk-...", model="text-embedding-3-large")
store = QdrantStore(url=":memory:", collection_name="demo")
store.create_collection(dimension=embedder.dimension)

vectors = await embedder.embed_batch(["hello", "world"], input_type="document")
chunks = [VectorChunk(id=f"doc#chunk_{i}", vector=vectors[i], metadata={"text": t}) for i, t in enumerate(["hello","world"])]
await store.upsert(chunks)
```

Query pipeline:

```python
from maktaba.pipeline.query import QueryPipeline
from maktaba.reranking.cohere import CohereReranker

pipeline = QueryPipeline(embedder, store, reranker=CohereReranker(use_api=False))
result = await pipeline.search("hello", rerank=True)
print(result["formatted_context"])  # [1]: ...
```
