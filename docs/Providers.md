% Providers

Embedders
- OpenAI (implemented): `maktaba.embedding.openai.OpenAIEmbedder`
- Others can be added by subclassing `BaseEmbedder`

Vector Stores
- Qdrant (implemented): `maktaba.storage.qdrant.QdrantStore`
- Pinecone/Weaviate/Chroma stubs exist; interface follows BaseVectorStore

Rerankers
- CohereReranker with offline heuristic by default
