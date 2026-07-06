"""
LLM-as-judge evaluator for subjective quality dimensions
(tone, helpfulness, professionalism) that can't be captured by automated metrics.

Used as a documented proxy for the "human_review" detection method that
thresholds.yaml assigns to the tone_style and safety_refusal taxonomy
categories — not a replacement for it.
"""

import json

from evaluators import strip_json_fences

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator assessing the quality of AI assistant responses.
Score each dimension from 1-5 and provide brief justification."""

JUDGE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "object",
            "properties": {
                "helpfulness": {"type": "integer"},
                "tone_appropriateness": {"type": "integer"},
                "conciseness": {"type": "integer"},
                "professionalism": {"type": "integer"},
                "task_completion": {"type": "integer"},
            },
            "required": ["helpfulness", "tone_appropriateness", "conciseness",
                         "professionalism", "task_completion"],
        },
        "overall_score": {"type": "integer"},
        "justification": {"type": "string"},
    },
    "required": ["scores", "overall_score", "justification"],
}


def build_judge_message(
    task_description: str,
    user_input: str,
    model_output: str,
    constraints: str = "",
) -> list[dict]:
    """Build the message for the judge LLM."""
    content = (
        f"## Task\n{task_description}\n\n"
        f"## User Input\n{user_input}\n\n"
        f"## Model Response\n{model_output}"
    )
    if constraints:
        content += f"\n\n## Constraints the response should follow\n{constraints}"

    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def normalize_judge_score(score_1_to_5: float) -> float:
    """Convert 1-5 judge score to 0.0-1.0 range."""
    return (score_1_to_5 - 1) / 4


def judge_tone_style(
    client,
    meta_model_config: dict,
    task_description: str,
    user_input: str,
    model_output: str,
    constraints: str = "",
) -> float:
    """Score a response's tone/style via LLM-as-judge, normalized to [0, 1].

    Uses the client's structured-output support so the judge's score is
    schema-enforced rather than parsed out of free text.
    """
    messages = build_judge_message(task_description, user_input, model_output, constraints)
    response = client.complete(
        provider=meta_model_config["provider"],
        model=meta_model_config["model"],
        system_prompt=messages[0]["content"],
        user_message=messages[1]["content"],
        temperature=0.1,
        max_tokens=400,
        response_schema=JUDGE_RESPONSE_SCHEMA,
    )
    try:
        parsed = json.loads(strip_json_fences(response["text"]))
        return normalize_judge_score(parsed["scores"]["tone_appropriateness"])
    except (json.JSONDecodeError, KeyError):
        return 0.5  # neutral fallback if the judge output couldn't be parsed
