"""Tests for LLM implementations."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from maktaba.llm.openai import OpenAILLM

# =============================================================================
# OpenAILLM Tests
# =============================================================================


@pytest.mark.asyncio
async def test_openai_llm_generate_queries_success():
    """Test successful query generation with OpenAI API."""
    with patch("openai.AsyncOpenAI") as MockOpenAI:
        # Mock the API response
        mock_client = AsyncMock()
        MockOpenAI.return_value = mock_client

        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 150
        mock_response.usage.completion_tokens = 50
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps({
                        "queries": [
                            {"type": "semantic", "query": "What is Tawhid in Islam?"},
                            {"type": "keyword", "query": "Tawhid"},
                        ]
                    })
                )
            )
        ]

        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        # Create LLM
        llm = OpenAILLM(api_key="test-key", model="gpt-4o-mini")

        # Generate queries
        queries, usage = await llm.generate_queries(
            messages=[("user", "What is Tawhid?")],
            existing_queries=[],
            max_queries=10,
        )

        # Verify queries
        assert len(queries) == 2
        assert queries[0]["type"] == "semantic"
        assert queries[0]["query"] == "What is Tawhid in Islam?"
        assert queries[1]["type"] == "keyword"
        assert queries[1]["query"] == "Tawhid"

        # Verify usage
        assert usage.input_tokens == 150
        assert usage.output_tokens == 50
        assert usage.total_tokens == 200


@pytest.mark.asyncio
async def test_openai_llm_generate_queries_with_existing_queries():
    """Test query generation includes existing queries in prompt."""
    with patch("openai.AsyncOpenAI") as MockOpenAI:
        mock_client = AsyncMock()
        MockOpenAI.return_value = mock_client

        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 200
        mock_response.usage.completion_tokens = 30
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps({
                        "queries": [
                            {"type": "semantic", "query": "New query about Tawhid"}
                        ]
                    })
                )
            )
        ]

        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        llm = OpenAILLM(api_key="test-key")

        # Provide existing queries
        queries, usage = await llm.generate_queries(
            messages=[("user", "What is Tawhid?")],
            existing_queries=["What is Tawhid?", "Tawhid definition"],
            max_queries=10,
        )

        # Verify API was called
        assert mock_client.chat.completions.create.called

        # Check that existing queries were included in the prompt
        call_args = mock_client.chat.completions.create.call_args
        user_message = call_args.kwargs["messages"][1]["content"]
        assert "What is Tawhid?" in user_message
        assert "Tawhid definition" in user_message


@pytest.mark.asyncio
async def test_openai_llm_generate_queries_respects_max_queries():
    """Test max_queries limit is enforced."""
    with patch("openai.AsyncOpenAI") as MockOpenAI:
        mock_client = AsyncMock()
        MockOpenAI.return_value = mock_client

        # Mock response with more queries than max_queries
        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 80
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps({
                        "queries": [
                            {"type": "semantic", "query": f"Query {i}"} for i in range(15)
                        ]
                    })
                )
            )
        ]

        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        llm = OpenAILLM(api_key="test-key")

        # Request max 5 queries
        queries, usage = await llm.generate_queries(
            messages=[("user", "Test?")],
            existing_queries=[],
            max_queries=5,
        )

        # Should return only 5 queries
        assert len(queries) == 5


@pytest.mark.asyncio
async def test_openai_llm_evaluate_sources_true():
    """Test source evaluation returns True when sources are sufficient."""
    with patch("openai.AsyncOpenAI") as MockOpenAI:
        mock_client = AsyncMock()
        MockOpenAI.return_value = mock_client

        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 500
        mock_response.usage.completion_tokens = 10
        mock_response.choices = [
            MagicMock(message=MagicMock(content=json.dumps({"canAnswer": True})))
        ]

        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        llm = OpenAILLM(api_key="test-key")

        # Evaluate sources
        can_answer, usage = await llm.evaluate_sources(
            messages=[("user", "What is Tawhid?")],
            sources=["Tawhid is the oneness of Allah.", "It is the first pillar."],
        )

        # Verify result
        assert can_answer is True
        assert usage.input_tokens == 500
        assert usage.output_tokens == 10


@pytest.mark.asyncio
async def test_openai_llm_evaluate_sources_false():
    """Test source evaluation returns False when sources are insufficient."""
    with patch("openai.AsyncOpenAI") as MockOpenAI:
        mock_client = AsyncMock()
        MockOpenAI.return_value = mock_client

        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 300
        mock_response.usage.completion_tokens = 10
        mock_response.choices = [
            MagicMock(message=MagicMock(content=json.dumps({"canAnswer": False})))
        ]

        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        llm = OpenAILLM(api_key="test-key")

        # Evaluate with insufficient sources
        can_answer, usage = await llm.evaluate_sources(
            messages=[("user", "What is Tawhid?")],
            sources=["Unrelated text."],
        )

        # Verify result
        assert can_answer is False
        assert usage.input_tokens == 300
        assert usage.output_tokens == 10


@pytest.mark.asyncio
async def test_openai_llm_generate_queries_api_failure():
    """Test graceful handling of API failures during query generation."""
    with patch("openai.AsyncOpenAI") as MockOpenAI:
        mock_client = AsyncMock()
        MockOpenAI.return_value = mock_client

        # Simulate API failure
        mock_client.chat.completions.create = AsyncMock(
            side_effect=Exception("API Error")
        )

        llm = OpenAILLM(api_key="test-key")

        # Should return empty list and zero usage
        queries, usage = await llm.generate_queries(
            messages=[("user", "Test?")], existing_queries=[], max_queries=10
        )

        assert queries == []
        assert usage.total_tokens == 0


@pytest.mark.asyncio
async def test_openai_llm_evaluate_sources_api_failure():
    """Test graceful handling of API failures during source evaluation."""
    with patch("openai.AsyncOpenAI") as MockOpenAI:
        mock_client = AsyncMock()
        MockOpenAI.return_value = mock_client

        # Simulate API failure
        mock_client.chat.completions.create = AsyncMock(
            side_effect=Exception("API Error")
        )

        llm = OpenAILLM(api_key="test-key")

        # Should return True (optimistic fallback) and zero usage
        can_answer, usage = await llm.evaluate_sources(
            messages=[("user", "Test?")], sources=["Some text"]
        )

        assert can_answer is True  # Optimistic fallback
        assert usage.total_tokens == 0


def test_openai_llm_client_unavailable():
    """Test graceful handling when openai package is not available."""
    with patch("openai.AsyncOpenAI", None):
        # Should create without error
        llm = OpenAILLM(api_key="test-key")

        # But client should be None
        assert llm._get_client() is None


@pytest.mark.asyncio
async def test_openai_llm_usage_tracking_accuracy():
    """Test token usage tracking matches API response exactly."""
    with patch("openai.AsyncOpenAI") as MockOpenAI:
        mock_client = AsyncMock()
        MockOpenAI.return_value = mock_client

        # Mock response with specific token counts
        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 1234
        mock_response.usage.completion_tokens = 567
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps({
                        "queries": [{"type": "semantic", "query": "Test query"}]
                    })
                )
            )
        ]

        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        llm = OpenAILLM(api_key="test-key")

        queries, usage = await llm.generate_queries(
            messages=[("user", "Test?")], existing_queries=[], max_queries=10
        )

        # Verify exact token counts
        assert usage.input_tokens == 1234
        assert usage.output_tokens == 567
        assert usage.total_tokens == 1801
