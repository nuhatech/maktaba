import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from maktaba.llm.openai import OpenAILLM
from maktaba.pipeline.deep_research.prompts import default_prompts

OPENAI_API_KEY_ENV = "OPENAI_API_KEY"

pytestmark = pytest.mark.integration

@pytest.mark.asyncio
async def test_planning_prompt_yields_json_with_openai():
    api_key = os.getenv(OPENAI_API_KEY_ENV)
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")

    llm = OpenAILLM(api_key=api_key, model="gpt-4o-mini", temperature=0.0)
    prompts = default_prompts()

    payload, usage = await llm.complete_json(
        system=prompts.planning_prompt,
        prompt="Research Topic: water conservation",
        temperature=0.0,
        max_tokens=200,
    )

    print(f"OpenAI planning queries: {payload.get('queries')}")

    assert isinstance(payload, dict)
    assert "queries" in payload
    assert isinstance(payload["queries"], list)
    assert payload["queries"], "Expected at least one query in OpenAI response"
