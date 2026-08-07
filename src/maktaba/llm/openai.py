"""OpenAI LLM implementation for agentic query generation and evaluation."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional, Tuple

from ..logging import get_logger
from ..models import LLMUsage
from .base import BaseLLM
from .prompts import AgenticPrompts, default_prompts

if TYPE_CHECKING:
    from ..collection_models import (
        CollectionCandidate,
        CollectionGoal,
        EvidenceSpan,
        RejectedCollectionCandidate,
    )
    from ..pipeline.agentic_models import EvidenceAssessment, EvidenceItem


class OpenAILLM(BaseLLM):
    """
    OpenAI implementation for agentic RAG operations.

    Uses OpenAI's Chat Completions API for query generation and evaluation.
    Falls back gracefully if OpenAI is unavailable.

    Example:
        # Use default prompts
        llm = OpenAILLM(api_key="sk-...")

        # Customize prompts
        from maktaba.llm.prompts import default_prompts
        custom_prompts = default_prompts(
            context="You are searching a medical knowledge base.",
            generate_queries_append="Focus on evidence-based queries."
        )
        llm = OpenAILLM(api_key="sk-...", prompts=custom_prompts)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        timeout_s: float = 30.0,
        prompts: Optional[AgenticPrompts] = None,
        use_max_completion_tokens: bool = False,
        omit_temperature: bool = False,
        reasoning_effort: Optional[str] = None,
        default_max_tokens: Optional[int] = None,
    ) -> None:
        """
        Initialize OpenAI LLM.

        Args:
            api_key: OpenAI API key (or None to use environment variable)
            model: Model name (default: gpt-4o-mini for cost efficiency)
            temperature: Sampling temperature (default: 0 for deterministic)
            timeout_s: Request timeout in seconds
            prompts: Custom prompts for agentic operations (defaults to default_prompts())
            use_max_completion_tokens: Use ``max_completion_tokens`` instead of
                ``max_tokens`` in API calls. Required for newer OpenAI models
                (o1, o3, gpt-5-nano, etc.) that no longer accept ``max_tokens``.
            omit_temperature: Omit the ``temperature`` parameter from API calls.
                Required for reasoning models (o1, o3, gpt-5-nano, etc.) that
                only support the default temperature value.
            reasoning_effort: Optional OpenAI reasoning effort (for example,
                ``minimal`` or ``low``). Omitted by default for compatibility.
            default_max_tokens: Optional token cap used when a completion call
                does not provide its own ``max_tokens`` value. Explicit values
                passed by callers always take precedence.
        """
        if default_max_tokens is not None and default_max_tokens <= 0:
            raise ValueError("default_max_tokens must be greater than zero")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.prompts = prompts or default_prompts()
        self.use_max_completion_tokens = use_max_completion_tokens
        self.omit_temperature = omit_temperature
        self.reasoning_effort = reasoning_effort
        self.default_max_tokens = default_max_tokens
        self._logger = get_logger("maktaba.llm.openai")

        # Lazy client initialization
        self._client: Optional[Any] = None
        self._OpenAI: Optional[type[Any]] = None
        try:
            from openai import AsyncOpenAI

            self._OpenAI = AsyncOpenAI
        except ImportError:  # pragma: no cover
            self._logger.warning("openai package not installed; agentic mode unavailable")

    def _get_client(self) -> Optional[Any]:
        """Lazy initialize OpenAI client."""
        if self._client is None and self._OpenAI is not None:
            self._client = self._OpenAI(api_key=self.api_key, timeout=self.timeout_s)
        return self._client

    def _temperature_kwargs(self, temperature: float) -> Dict[str, Any]:
        """Build the temperature keyword argument for the OpenAI API.

        Returns an empty dict when :attr:`omit_temperature` is ``True`` so the
        parameter is omitted entirely (reasoning models reject custom values).
        """
        if self.omit_temperature:
            return {}
        return {"temperature": temperature}

    def _token_limit_kwargs(self, max_tokens: int | None) -> Dict[str, Any]:
        """Build the token-limit keyword argument for the OpenAI API.

        Returns an empty dict when *max_tokens* is ``None`` so the parameter
        is omitted entirely (some models reject ``null``).  When a value is
        provided the key name depends on :attr:`use_max_completion_tokens`.
        """
        effective_max_tokens = (
            max_tokens if max_tokens is not None else self.default_max_tokens
        )
        if effective_max_tokens is None:
            return {}
        key = "max_completion_tokens" if self.use_max_completion_tokens else "max_tokens"
        return {key: effective_max_tokens}

    def _reasoning_kwargs(self) -> Dict[str, Any]:
        """Build the optional reasoning-control argument for OpenAI calls."""
        if self.reasoning_effort is None:
            return {}
        return {"reasoning_effort": self.reasoning_effort}

    def _format_chat_history(self, messages: List[Tuple[str, str]]) -> str:
        """Format chat history as text."""
        lines = []
        for role, content in messages:
            label = "Human" if role == "user" else "Assistant"
            lines.append(f"{label}: {content}")
        return "\n\n".join(lines)

    async def complete_text(
        self,
        *,
        system: str,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> Tuple[str, LLMUsage]:
        """
        Generic text completion call used by deep research steps.
        """
        client = self._get_client()
        if client is None:
            self._logger.warning("OpenAI client unavailable, returning empty completion")
            return "", LLMUsage()

        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                **self._temperature_kwargs(temperature),
                **self._token_limit_kwargs(max_tokens),
                **self._reasoning_kwargs(),
            )

            usage = LLMUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )

            content = response.choices[0].message.content or ""
            return content, usage

        except Exception as exc:  # pragma: no cover - network failure path
            self._logger.error(f"Text completion failed: {exc}", exc_info=True)
            return "", LLMUsage()

    async def complete_json(
        self,
        *,
        system: str,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> Tuple[Dict[str, object], LLMUsage]:
        """
        Request structured JSON response from the LLM.
        """
        client = self._get_client()
        if client is None:
            self._logger.warning("OpenAI client unavailable, returning empty JSON completion")
            return {}, LLMUsage()

        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                **self._temperature_kwargs(temperature),
                **self._token_limit_kwargs(max_tokens),
                **self._reasoning_kwargs(),
                response_format={"type": "json_object"},
            )

            usage = LLMUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )

            content = response.choices[0].message.content or "{}"
            return json.loads(content), usage

        except Exception as exc:  # pragma: no cover - network failure path
            self._logger.error(f"JSON completion failed: {exc}", exc_info=True)
            return {}, LLMUsage()

    async def stream_text(
        self,
        *,
        system: str,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> Tuple[AsyncIterator[str], LLMUsage]:
        client = self._get_client()
        if client is None:
            return await super().stream_text(
                system=system,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        try:
            stream = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                **self._temperature_kwargs(temperature),
                **self._token_limit_kwargs(max_tokens),
                **self._reasoning_kwargs(),
                stream=True,
            )

            async def _generator() -> AsyncIterator[str]:
                async for chunk in stream:
                    for choice in chunk.choices:
                        if choice.delta.content:
                            yield choice.delta.content

            usage = LLMUsage()
            if getattr(stream, "usage", None):
                usage = LLMUsage(
                    input_tokens=stream.usage.prompt_tokens,
                    output_tokens=stream.usage.completion_tokens,
                )

            return _generator(), usage

        except Exception as exc:  # pragma: no cover - network failure path
            self._logger.error(f"Streaming completion failed: {exc}", exc_info=True)
            return await super().stream_text(
                system=system,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )

    async def generate_queries(
        self,
        messages: List[Tuple[str, str]],
        existing_queries: List[str],
        max_queries: int = 10,
    ) -> Tuple[List[Dict[str, str]], LLMUsage]:
        """
        Generate search queries using OpenAI.

        Args:
            messages: Chat history as (role, content) tuples
            existing_queries: Previously generated queries to avoid
            max_queries: Maximum number of queries to generate

        Returns:
            Tuple of (queries, usage):
                - queries: List of {"type": "semantic"|"keyword", "query": "..."} dicts
                - usage: LLMUsage with token counts
        """
        client = self._get_client()
        if client is None:
            self._logger.warning("OpenAI client unavailable, returning empty queries")
            return [], LLMUsage()

        try:
            # Build prompt
            chat_history = self._format_chat_history(messages)
            existing_queries_text = ""
            if existing_queries:
                existing_queries_text = (
                    "\n\nThe queries you return should be different from these ones that were tried so far:\n"
                    + "\n".join(f"- {q}" for q in existing_queries)
                )

            user_prompt = f"{existing_queries_text}\n\nChat history:\n{chat_history}".strip()

            # Call OpenAI
            result, usage = await self.complete_json(
                system=self.prompts.generate_queries_prompt.format(max_queries=max_queries),
                prompt=user_prompt,
                temperature=self.temperature,
            )
            queries_raw = result.get("queries", [])
            # Type check: ensure queries is a list
            if not isinstance(queries_raw, list):
                queries: List[Dict[str, Any]] = []
            else:
                queries = queries_raw

            self._logger.info(
                f"Generated {len(queries)} queries: {[q.get('query', '') if isinstance(q, dict) else str(q) for q in queries]} "
                f"(tokens: {usage.total_tokens})"
            )
            return queries[:max_queries], usage

        except Exception as e:
            self._logger.error(f"Query generation failed: {e}", exc_info=True)
            return [], LLMUsage()

    async def evaluate_sources(
        self,
        messages: List[Tuple[str, str]],
        sources: List[str],
    ) -> Tuple[bool, LLMUsage]:
        """
        Evaluate if sources can answer the question.

        Args:
            messages: Chat history as (role, content) tuples
            sources: List of retrieved text chunks

        Returns:
            Tuple of (can_answer, usage):
                - can_answer: True if sources contain sufficient information
                - usage: LLMUsage with token counts
        """
        chat_history = self._format_chat_history(messages)
        sources_text = "\n\n".join(
            f"<source_{i+1}>\n{source}\n</source_{i+1}>"
            for i, source in enumerate(sources)
        )

        user_prompt = f"Chat history:\n{chat_history}\n\nRetrieved sources:\n{sources_text}"

        try:
            result, usage = await self.complete_json(
                system=self.prompts.evaluate_sources_prompt,
                prompt=user_prompt,
                temperature=self.temperature,
            )
            can_answer_raw = result.get("canAnswer")
            # Fail closed: strings such as "false" must never become truthy.
            can_answer = can_answer_raw if isinstance(can_answer_raw, bool) else False

            self._logger.info(f"Source evaluation: canAnswer={can_answer} (tokens: {usage.total_tokens})")
            return can_answer, usage

        except Exception as e:
            self._logger.error(f"Source evaluation failed: {e}", exc_info=True)
            return False, LLMUsage()

    async def assess_evidence(
        self,
        messages: List[Tuple[str, str]],
        evidence: List["EvidenceItem"],
    ) -> Tuple["EvidenceAssessment", LLMUsage]:
        """Assess ranked evidence and return validated Agentic v2 controls."""
        from ..pipeline.agentic_models import EvidenceAssessment

        chat_history = self._format_chat_history(messages)
        evidence_text = "\n\n".join(
            (
                f'<evidence id="{item.id}" rank="{item.rank}" score="{item.score}">\n'
                f"{item.text}\n</evidence>"
            )
            for item in evidence
        )
        prompt = f"Chat history:\n{chat_history}\n\nRanked evidence:\n{evidence_text}"
        result, usage = await self.complete_json(
            system=self.prompts.assess_evidence_prompt or self.prompts.evaluate_sources_prompt,
            prompt=prompt,
            temperature=self.temperature,
        )
        assessment = EvidenceAssessment.from_mapping(
            result,
            valid_source_ids=[item.id for item in evidence],
        )
        self._logger.info(
            "Evidence assessment: answerable=%s valid=%s gaps=%d actions=%d (tokens: %d)",
            assessment.answerable,
            assessment.valid,
            len(assessment.missing_information),
            len(assessment.next_actions),
            usage.total_tokens,
        )
        return assessment, usage

    async def extract_collection_items(
        self,
        *,
        goal: "CollectionGoal",
        messages: List[Tuple[str, str]],
        evidence: List["EvidenceItem"],
        accepted_items: List["EvidenceSpan"],
    ) -> Tuple[List["CollectionCandidate"], LLMUsage]:
        """Extract untrusted exact-span candidates using structured JSON."""
        from ..collection_models import CollectionCandidate

        chat_history = self._format_chat_history(messages)
        evidence_text = "\n\n".join(
            (
                f'<evidence id="{item.id}" rank="{item.rank}" score="{item.score}">\n'
                f"{item.text}\n</evidence>"
            )
            for item in evidence
        )
        accepted_text = "\n".join(
            f'- source_id={item.source_id}; text={item.text[:240]!r}' for item in accepted_items
        ) or "- none"
        remaining = max(goal.target_count - len(accepted_items), 0)
        prompt = (
            f"{goal.to_prompt()}\n\n"
            f"Remaining count: {remaining}\n\n"
            f"Chat history:\n{chat_history}\n\n"
            f"Already accepted items (do not duplicate):\n{accepted_text}\n\n"
            f"Retrieved evidence:\n{evidence_text}"
        )
        result, usage = await self.complete_json(
            system=self.prompts.extract_collection_prompt or self.prompts.assess_evidence_prompt or "",
            prompt=prompt,
            temperature=self.temperature,
        )
        raw_items = result.get("items", [])
        parsed: List[CollectionCandidate] = []
        if isinstance(raw_items, list):
            for raw in raw_items[:remaining]:
                candidate = CollectionCandidate.from_mapping(raw) if isinstance(raw, dict) else None
                if candidate is not None:
                    parsed.append(candidate)
        return parsed, usage

    async def plan_collection_actions(
        self,
        *,
        goal: "CollectionGoal",
        messages: List[Tuple[str, str]],
        evidence: List["EvidenceItem"],
        accepted_items: List["EvidenceSpan"],
        rejected_candidates: List["RejectedCollectionCandidate"],
    ) -> Tuple["EvidenceAssessment", LLMUsage]:
        """Return validated search/expand actions for remaining items."""
        from ..pipeline.agentic_models import EvidenceAssessment

        chat_history = self._format_chat_history(messages)
        evidence_text = "\n\n".join(
            (
                f'<evidence id="{item.id}" rank="{item.rank}" score="{item.score}">\n'
                f"{item.text}\n</evidence>"
            )
            for item in evidence
        )
        accepted_text = "\n".join(
            f'- source_id={item.source_id}; text={item.text[:180]!r}' for item in accepted_items
        ) or "- none"
        rejected_text = "\n".join(
            f"- source_id={item.source_id or 'unknown'}; reason={item.reason}"
            for item in rejected_candidates[-20:]
        ) or "- none"
        prompt = (
            f"{goal.to_prompt()}\n\n"
            f"Verified count: {len(accepted_items)} of {goal.target_count}\n\n"
            f"Chat history:\n{chat_history}\n\n"
            f"Accepted item summaries:\n{accepted_text}\n\n"
            f"Recent rejected candidates:\n{rejected_text}\n\n"
            f"Ranked evidence:\n{evidence_text}"
        )
        result, usage = await self.complete_json(
            system=self.prompts.plan_collection_prompt or self.prompts.assess_evidence_prompt or "",
            prompt=prompt,
            temperature=self.temperature,
        )
        assessment = EvidenceAssessment.from_mapping(
            result,
            valid_source_ids=[item.id for item in evidence],
        )
        assessment.answerable = False
        return assessment, usage
