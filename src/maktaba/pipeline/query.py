"""Query pipeline that ties together embedder, store, reranker, and citations."""

from typing import Any, Dict, List, Optional, Tuple, Union

from ..citation.formatter import format_with_citations
from ..embedding.base import BaseEmbedder
from ..logging import get_logger
from ..models import SearchResult
from ..reranking.base import BaseReranker
from ..retrieval.query_condenser import AutoQueryCondenser, QueryCondenser
from ..storage.base import BaseVectorStore


class QueryPipeline:
    """
    Feature-complete query pipeline.

    Usage:
        pipeline = QueryPipeline(embedder, store, reranker)
        out = await pipeline.search("What is Tawhid?", rerank=True)
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        store: BaseVectorStore,
        reranker: Optional[BaseReranker] = None,
        namespace: Optional[str] = None,
        default_top_k: int = 10,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.reranker = reranker
        self.namespace = namespace
        self.default_top_k = default_top_k
        self._logger = get_logger("maktaba.pipeline.query")

    async def search(
        self,
        query: str,
        rerank: bool = True,
        top_k: Optional[int] = None,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
        includeMetadata: bool = True,
    ) -> Dict[str, object]:
        k = top_k or self.default_top_k
        ns = namespace or self.namespace

        # 1) Embed query
        self._logger.info("query.start: text='%s' top_k=%s ns=%s", query, k, ns)
        qvec = await self.embedder.embed_text(query, input_type="query")

        # 2) Retrieve
        initial: List[SearchResult] = await self.store.query(
            vector=qvec,
            topK=k,
            filter=filter,
            includeMetadata=includeMetadata,
            namespace=ns,
        )

        # 3) Rerank (optional)
        if rerank and self.reranker is not None:
            ranked = await self.reranker.rerank(query, initial, top_k=k)
        else:
            ranked = initial[:k]

        self._logger.info("query.retrieved: initial=%d ranked=%d", len(initial), len(ranked))

        # 4) Format citations
        formatted = format_with_citations(ranked, top_k=k)
        formatted["results"] = ranked
        self._logger.info(
            "query.done: formatted_blocks=%d citations=%d",
            len(ranked),
            len(formatted.get("citations", [])),
        )
        return formatted

    async def search_with_history(
        self,
        messages: List[Union[Dict[str, str], Tuple[str, str]]],
        *,
        rerank: bool = True,
        top_k: Optional[int] = None,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
        includeMetadata: bool = True,
        condenser: Optional[QueryCondenser] = None,
        max_history: int = 10,
    ) -> Dict[str, object]:
        if not messages:
            raise ValueError("messages must contain at least one item")

        # Normalize to (role, content)
        norm: List[Tuple[str, str]] = []
        for m in messages:
            if isinstance(m, tuple):
                role, content = m
            else:
                role = m.get("role", "user")
                content = m.get("content", "")
            if not isinstance(content, str):
                content = str(content)
            norm.append((role, content))

        # Find latest user message
        last_role, last_content = norm[-1]
        if last_role != "user":
            for role, content in reversed(norm):
                if role == "user":
                    last_role, last_content = role, content
                    break
            else:
                raise ValueError("no user message found in messages")

        history = norm[:-1]
        if max_history > 0:
            history = history[-max_history:]

        cond = condenser or AutoQueryCondenser(max_history=max_history)
        condensed = await cond.condense(history, last_content)
        return await self.search(
            condensed,
            rerank=rerank,
            top_k=top_k,
            namespace=namespace,
            filter=filter,
            includeMetadata=includeMetadata,
        )
