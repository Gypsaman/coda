"""
LLM-as-judge evaluator for subjective quality dimensions
(tone, helpfulness, professionalism) that can't be captured by automated metrics.
"""

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator assessing the quality of AI assistant responses.
Score each dimension from 1-5 and provide brief justification.

Respond ONLY with a JSON object:
{
  "scores": {
    "helpfulness": <1-5>,
    "tone_appropriateness": <1-5>,
    "conciseness": <1-5>,
    "professionalism": <1-5>,
    "task_completion": <1-5>
  },
  "overall_score": <1-5>,
  "justification": "<2-3 sentences>"
}"""


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
