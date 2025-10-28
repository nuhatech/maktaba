from maktaba.pipeline.deep_research.prompts import default_prompts


def test_default_prompts_contain_json_keywords():
    prompts = default_prompts(now=None)
    candidates = [
        prompts.planning_prompt,
        prompts.plan_parsing_prompt,
        prompts.evaluation_parsing_prompt,
        prompts.source_parsing_prompt,
    ]
    for text in candidates:
        assert "json" in text.lower(), f"Expected 'json' keyword in prompt: {text[:60]}"



def test_default_prompts_custom_context_header():
    prompts = default_prompts(now=None, context="Islamic heritage focus")
    assert prompts.planning_prompt.startswith("Islamic heritage focus")


def test_default_prompts_custom_append():
    prompts = default_prompts(planning_append="Please prioritise classical sources.")
    assert "Please prioritise classical sources." in prompts.planning_prompt


def test_default_prompts_empty_context_removes_header():
    prompts = default_prompts(context="")
    assert prompts.planning_prompt.startswith("You are a strategic research planner")
