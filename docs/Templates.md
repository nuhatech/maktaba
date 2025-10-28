# Customising Deep Research Templates

Maktaba ships opinionated deep-research prompts tuned for contemporary, general-purpose corpora. When your catalogue is a specialised collection (for example Kutub's classical Islamic texts), you can keep the upstream template structure while tailoring the guidance to your domain.

## Global Context Override

```python
from maktaba.pipeline.deep_research.prompts import default_prompts

prompts = default_prompts(
    context="Classical Islamic heritage focus. Prioritise primary manuscripts and pre-modern scholars."
)
```

Passing `context` replaces the default current-date header. Provide an empty string (`context=""`) to remove the header entirely.

## Stage-Specific Guidance

Each stage exposes an optional `*_append` argument so you can retain the base instructions while appending domain rules:

```python
prompts = default_prompts(
    context="Kutub Research: focus on Maliki fiqh references only.",
    planning_append="Restrict queries to recognised jurists and avoid modern commentary.",
    raw_content_append="Highlight isnad chains and cite original manuscripts when available.",
    answer_append="Write in formal Arabic and include [n] references to primary sources."
)
```

Available append hooks:

- `planning_append`
- `plan_parsing_append`
- `raw_content_append`
- `evaluation_append`
- `evaluation_parsing_append`
- `filter_append`
- `source_parsing_append`
- `answer_append`

## Example: Wiring Into Kutub

```python
from maktaba.pipeline.deep_research import (
    DeepResearchConfig,
    DeepResearchModelConfig,
    DeepResearchPipeline,
)
from maktaba.pipeline.deep_research.prompts import default_prompts

custom_prompts = default_prompts(
    context="Kutub - Islamic heritage research. Ignore contemporary news and emphasise classical scholars.",
    planning_append="Frame search queries around well-known madhhab terminology.",
    answer_append="Do not mention current dates. Emphasise sanad credibility and reference original works."
)

pipeline = DeepResearchPipeline(
    query_pipeline=my_query_pipeline,
    model_config=DeepResearchModelConfig(planning=my_llm, json=my_llm, summary=my_llm, answer=my_llm),
    config=DeepResearchConfig(max_queries=2),
    prompts=custom_prompts,
)
```

These hooks let you adapt the pipeline to a curated corpus while still benefiting from upstream prompt improvements.
