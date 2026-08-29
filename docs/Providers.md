% Providers

Embedders
- OpenAI (implemented): `maktaba.embedding.openai.OpenAIEmbedder`
- Others can be added by subclassing `BaseEmbedder`

Vector Stores
- Qdrant (implemented): `maktaba.storage.qdrant.QdrantStore`
- Pinecone/Weaviate/Chroma stubs exist; interface follows BaseVectorStore

Rerankers
- CohereReranker with offline heuristic by default
- VoyageReranker (`rerank-2.5`) with API-by-key and deterministic fallback
- ZeroEntropyReranker with offline heuristic fallback
