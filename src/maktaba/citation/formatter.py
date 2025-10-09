"""Citation formatter for search results."""

from typing import Dict, List

from ..models import SearchResult


def format_with_citations(
    results: List[SearchResult], top_k: int = 10
) -> Dict[str, object]:
    """
    Produce formatted context and citation entries from search results.

    Returns a dict with:
      - formatted_context: str
      - citations: List[Dict]
    """
    limited = results[:top_k]

    blocks: List[str] = []
    citations: List[Dict[str, object]] = []

    for idx, res in enumerate(limited, start=1):
        text = res.text or ""
        blocks.append(f"[{idx}]: {text}")
        citations.append(
            {
                "index": idx,
                "id": res.id,
                "document_id": res.document_id,
                "chunk_id": res.chunk_id,
                "metadata": res.metadata,
            }
        )

    formatted_context = "\n\n".join(blocks)
    return {"formatted_context": formatted_context, "citations": citations}
