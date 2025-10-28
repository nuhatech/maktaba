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
