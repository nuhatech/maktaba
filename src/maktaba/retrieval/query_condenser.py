"""Query condensers (heuristic + LLM with graceful fallback)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..logging import get_logger


class QueryCondenser:
    """
    Heuristic condenser: returns the latest user query as-is.

    Backwards-compatible with earlier minimal implementation.
    """

    async def condense(self, history: List[Tuple[str, str]], current_query: str) -> str:
        return current_query


class HeuristicQueryCondenser(QueryCondenser):
    """Explicit heuristic condenser alias (same behavior)."""

    pass


class OpenAIQueryCondenser(QueryCondenser):
    """
    LLM-based query condenser using OpenAI's Chat Completions API.

    Falls back to heuristic if OpenAI client is unavailable or any error occurs.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        max_history: int = 10,
        temperature: float = 0.0,
        timeout_s: float = 15.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.max_history = max_history
        self.temperature = temperature
        self.timeout_s = timeout_s
        self._logger = get_logger("maktaba.condenser.openai")

        # Lazy client init
        self._client = None
        try:
            from openai import AsyncOpenAI

            self._OpenAI = AsyncOpenAI
        except Exception:  # pragma: no cover - optional dependency
            self._OpenAI = None  # type: ignore

    def _build_messages(self, history: List[Tuple[str, str]], last: str) -> List[Dict[str, Any]]:
        # Use up to last N turns from history
        recent = history[-self.max_history :] if self.max_history > 0 else history
        # Format: Human/Assistant lines embedded into a user prompt
        history_lines = []
        for role, content in recent:
            label = "Human" if role == "user" else "Assistant"
            history_lines.append(f"- {label}: {content}")
        chat_history_block = "\n\n".join(history_lines)

        system_prompt = (
            "Given a conversation history between Human and Assistant and a follow-up question, "
            "rewrite the question into a standalone query that:\n"
            "1) Incorporates relevant context from the prior conversation\n"
            "2) Preserves specific details and technical terms\n"
            "3) Maintains the user's language and tone\n"
            "4) Focuses on searchable keywords for vector retrieval\n"
            "5) Removes conversational elements (e.g., 'as mentioned before')\n"
            "6) Expands pronouns/references to full forms (e.g., 'it' -> 'the database schema')\n"
            "Output only the condensed query."
        )

        user_prompt = f"Chat History:\n{chat_history_block}\n\nFollow Up Message:\n{last}"

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    async def condense(self, history: List[Tuple[str, str]], current_query: str) -> str:
        # If no history, skip LLM call
        if not history:
            return current_query

        # Check availability
        if self._OpenAI is None:
            self._logger.info("openai package not installed; using heuristic condenser")
            return current_query

        try:
            if self._client is None:
                if self._OpenAI is None:
                    return current_query
                self._client = self._OpenAI(api_key=self.api_key)

            messages = self._build_messages(history, current_query)
            # Use Chat Completions for broad compatibility
            resp = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                timeout=self.timeout_s,
            )
            text = (
                resp.choices[0].message.content
                if getattr(resp, "choices", None)
                else None
            )
            if not text or not isinstance(text, str):
                return current_query
            return text.strip()
        except Exception as e:  # pragma: no cover - network
            self._logger.info("condenser fallback: %s", str(e))
            return current_query


class CohereQueryCondenser(QueryCondenser):
    """
    LLM-based query condenser using Cohere's Chat API.

    Falls back to heuristic if cohere client is unavailable or any error occurs.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "command-r",
        max_history: int = 10,
        temperature: float = 0.0,
        timeout_s: float = 15.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.max_history = max_history
        self.temperature = temperature
        self.timeout_s = timeout_s
        self._logger = get_logger("maktaba.condenser.cohere")
        try:
            import cohere

            self._cohere = cohere
        except Exception:  # pragma: no cover - optional dependency
            self._cohere = None  # type: ignore

    def _build_prompts(self, history: List[Tuple[str, str]], last: str) -> str:
        recent = history[-self.max_history :] if self.max_history > 0 else history
        history_lines = []
        for role, content in recent:
            label = "Human" if role == "user" else "Assistant"
            history_lines.append(f"- {label}: {content}")
        chat_history_block = "\n\n".join(history_lines)
        system_prompt = (
            "Given a conversation history and a follow-up question, rewrite it as a standalone, "
            "context-rich query for retrieval. Output only the condensed query."
        )
        user_prompt = f"Chat History:\n{chat_history_block}\n\nFollow Up Message:\n{last}"
        return system_prompt + "\n\n" + user_prompt

    async def condense(self, history: List[Tuple[str, str]], current_query: str) -> str:
        if not history:
            return current_query
        if self._cohere is None:
            self._logger.info("cohere package not installed; using heuristic condenser")
            return current_query
        try:
            # cohere client is sync; run in thread to keep API async
            import asyncio

            prompt = self._build_prompts(history, current_query)
            def _call() -> str:
                client = self._cohere.Client(self.api_key)
                resp = client.chat(model=self.model, message=prompt, temperature=self.temperature)
                # newer cohere clients use .text
                text = getattr(resp, "text", None) or getattr(resp, "output_text", None)
                return text.strip() if isinstance(text, str) else ""

            text = await asyncio.to_thread(_call)
            return text or current_query
        except Exception as e:  # pragma: no cover - network
            self._logger.info("condenser fallback (cohere): %s", str(e))
            return current_query


class AutoQueryCondenser(QueryCondenser):
    """
    Try LLM condenser(s) first when available; fallback to heuristic.
    Provider order can be customized.
    """

    def __init__(
        self,
        *,
        providers: Optional[List[str]] = None,
        max_history: int = 10,
        openai_api_key: Optional[str] = None,
        openai_model: str = "gpt-4o-mini",
        cohere_api_key: Optional[str] = None,
        cohere_model: str = "command-r",
    ) -> None:
        self._heuristic = HeuristicQueryCondenser()
        self._providers = providers or ["openai", "cohere"]
        self._openai = OpenAIQueryCondenser(
            api_key=openai_api_key, model=openai_model, max_history=max_history
        )
        self._cohere = CohereQueryCondenser(
            api_key=cohere_api_key, model=cohere_model, max_history=max_history
        )

    async def condense(self, history: List[Tuple[str, str]], current_query: str) -> str:
        if not history:
            return current_query
        for p in self._providers:
            if p == "openai":
                out = await self._openai.condense(history, current_query)
            elif p == "cohere":
                out = await self._cohere.condense(history, current_query)
            else:
                out = ""
            if out and out != current_query:
                return out
        return current_query
