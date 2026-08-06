# Maktaba

[![CI](https://github.com/nuhatech/maktaba/actions/workflows/ci.yml/badge.svg)](https://github.com/nuhatech/maktaba/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/maktaba.svg)](https://badge.fury.io/py/maktaba)
[![Version](https://img.shields.io/badge/version-0.2.0-blue.svg)](https://github.com/nuhatech/maktaba/releases)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**The library for building libraries** - By NuhaTech

> From the Arabic word for library, Maktaba is a modern RAG infrastructure for building intelligent knowledge systems in any language.

## Features

- 🔌 **Provider-agnostic**: Works with OpenAI, Cohere, Azure, and more
- 🚀 **Production-ready**: Built for scale with async-first design
- 🧩 **Modular**: Use only what you need
- 🌍 **Multilingual**: Optimized for Arabic and international languages
- 📊 **Type-safe**: Full type hints and Pydantic validation
- 🧪 **Well-tested**: Comprehensive test coverage
- 🔍 **Deep research**: Built-in iterative planning for long-form reports
- 🧭 **Agentic Search v2**: Evidence-gap planning, global fusion, graph expansion, provenance, and explicit abstention

## Installation

### Using UV (Recommended)

```bash
# Install UV if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add maktaba to your project
uv add maktaba

# With OpenAI + Qdrant
uv add "maktaba[openai,qdrant]"

# With all providers
uv add "maktaba[all]"
```

### Using pip

```bash
# Basic installation
pip install maktaba

# With OpenAI + Qdrant
pip install "maktaba[openai,qdrant]"

# With all providers
pip install "maktaba[all]"
```

## Quick Start

```python
from maktaba.pipeline import QueryPipeline
from maktaba.embedding import OpenAIEmbedder
from maktaba.storage import QdrantStore
from maktaba.reranking import CohereReranker

# Create pipeline
pipeline = QueryPipeline(
    embedder=OpenAIEmbedder(api_key="..."),
    vector_store=QdrantStore(url="http://localhost:6333", collection_name="docs"),
    reranker=CohereReranker(api_key="...")
)

# Search with automatic reranking and citation formatting
result = await pipeline.search(
    query="What is Tawhid?",
    top_k=10,
    rerank=True
)

# Use in your LLM prompt
print(result["formatted_context"])  # [1]: content... [2]: content...
print(result["citations"])          # [{id: 1, source: "...", score: 0.95}, ...]
```

### Agentic Search v2

```python
from maktaba.pipeline import AgenticQueryPipeline, AgenticSearchConfig

pipeline = AgenticQueryPipeline(
    embedder=embedder,
    store=store,
    keyword_store=keyword_store,
    reranker=reranker,
    llm=llm,
)

result = await pipeline.agentic_search(
    [("user", "Compare the available evidence and identify disagreements")],
    includeRelationships=True,
    config=AgenticSearchConfig(
        max_iterations=4,
        max_total_queries=20,
        token_budget=8_000,
        evidence_limit=15,
    ),
)

if not result["answerable"]:
    print("Insufficient evidence:", result["stop_reason"])
else:
    print(result["formatted_context"])
```

Agentic v2 preserves the original method and return keys. It adds structured assessment, evidence provenance, iteration traces, bounded relationship expansion, and a fail-closed `answerable` signal. See [AgenticV2.md](./docs/AgenticV2.md) for the complete contract and migration notes.

### Deep Research Pipeline
Learn how to customise the default prompts via [`maktaba_templates.md`](./docs/Templates.md).


```python
from maktaba.pipeline import create_deep_research_pipeline
from maktaba.embedding import OpenAIEmbedder
from maktaba.storage import QdrantStore
from maktaba.llm import OpenAILLM

pipeline = create_deep_research_pipeline(
    embedder=OpenAIEmbedder(api_key="..."),
    store=QdrantStore(url="http://localhost:6333", collection_name="docs"),
    llm=OpenAILLM(api_key="...", model="gpt-4o-mini"),
)

result = await pipeline.run_research("Impacts of lunar dust on spacecraft design")

chunks = [chunk async for chunk in result.stream]
print("".join(chunks))       # Final long-form report
print(result.queries_used)   # Queries issued during research
print(result.source_indices) # 1-based indices of retained sources
```

For a full walkthrough (configuration knobs, streaming, stage overrides), see `docs/DeepResearch.md` and `examples/deep_research_pipeline.py`.

## Development

### Running Checks Before Push

Before pushing to the remote repository, run all quality checks:

**Linux/Mac/Git Bash:**
```bash
./scripts/check.sh
```

**Windows CMD:**
```cmd
scripts\check.bat
```

This will run:
- Ruff linting
- MyPy type checking
- Pytest tests

All checks must pass before pushing.

## Documentation

- Overview: docs/Overview.md
- Quickstart: docs/Quickstart.md
- Pipelines: docs/Pipelines.md
- Agentic Search v2: docs/AgenticV2.md
- Providers: docs/Providers.md
- Examples: docs/Examples.md
- Troubleshooting: docs/Troubleshooting.md

Website (coming soon): maktaba.nuhatech.com

## License

MIT License - see [LICENSE](LICENSE)

## About NuhaTech

Built by [NuhaTech](https://nuhatech.com) - creators of Kutub and Muqabia.

