# Maktaba

[![CI](https://github.com/nuhatech/maktaba/actions/workflows/ci.yml/badge.svg)](https://github.com/nuhatech/maktaba/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/maktaba.svg)](https://badge.fury.io/py/maktaba)
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
- Providers: docs/Providers.md
- Examples: docs/Examples.md
- Troubleshooting: docs/Troubleshooting.md

Website (coming soon): maktaba.nuhatech.com

## License

MIT License - see [LICENSE](LICENSE)

## About NuhaTech

Built by [NuhaTech](https://nuhatech.com) - creators of Kutub and Muqabia.
