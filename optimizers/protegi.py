"""
ProTeGi: Automatic Prompt Optimization with 'Gradient Descent'.
Iteratively generates textual criticisms (gradients) from failure cases
and applies targeted edits to the prompt.

Reference: Pryzant et al. 2023, "Automatic Prompt Optimization with
'Gradient Descent' and Beam Search", EMNLP 2023. arXiv:2305.03495
"""

import json
from typing import Callable

from scripts.llm_client import LLMClient

GRADIENT_SYSTEM_PROMPT = """You are a prompt optimization expert. You will be shown:
1. A system prompt that is being used with a language model
2. Several failure cases where the prompt produced bad outputs
3. A classification of what types of failures occurred

Your job is to analyze WHY the prompt fails on these cases and suggest SPECIFIC edits.

Respond with JSON:
{
  "diagnosis": "<2-3 sentences explaining the root cause of failures>",
  "edits": [
    {
      "location": "<which part of the prompt to edit>",
      "current_text": "<relevant excerpt from current prompt>",
      "suggested_text": "<your improved version>",
      "rationale": "<why this edit should fix the failure>"
    }
  ]
}"""

APPLY_EDIT_SYSTEM_PROMPT = """You are a prompt editor. Given an original prompt and a set of edits to apply,
produce the complete updated prompt incorporating all edits.

Return ONLY the complete updated prompt text. No commentary, no markdown fences."""


def generate_gradient(
    client: LLMClient,
    current_prompt: str,
    failure_cases: list[dict],
    classification_summary: dict,
) -> dict:
    """Generate a textual gradient from failure cases."""
    case_descriptions = []
    for fc in failure_cases[:5]:
        desc = f"Input: {json.dumps(fc['input'])[:200]}\n"
        desc += f"Expected: {json.dumps(fc['expected'])[:200]}\n"
        desc += f"Got: {fc.get('new_output', '')[:300]}\n"
        desc += f"Failed metrics: {', '.join(fc.get('failed_metrics', []))}"
        case_descriptions.append(desc)

    user_msg = (
        f"## Current Prompt\n{current_prompt}\n\n"
        f"## Failure Classification\n{json.dumps(classification_summary, indent=2)}\n\n"
        f"## Failure Cases\n" + "\n---\n".join(case_descriptions)
    )

    response = client.complete(
        provider="openai",
        model="gpt-4o",
        system_prompt=GRADIENT_SYSTEM_PROMPT,
        user_message=user_msg,
        temperature=0.3,
        max_tokens=1500,
    )

    text = response["text"].strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    if text.startswith("json"):
        text = text[4:].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"diagnosis": text, "edits": []}


def apply_edits(client: LLMClient, current_prompt: str, gradient: dict) -> str:
    """Apply suggested edits to produce an updated prompt."""
    if not gradient.get("edits"):
        return current_prompt

    response = client.complete(
        provider="openai",
        model="gpt-4o",
        system_prompt=APPLY_EDIT_SYSTEM_PROMPT,
        user_message=(
            f"## Original Prompt\n{current_prompt}\n\n"
            f"## Edits to Apply\n{json.dumps(gradient['edits'], indent=2)}"
        ),
        temperature=0.1,
        max_tokens=2000,
    )
    return response["text"].strip()


def optimize(
    prompt: str,
    test_cases: list[dict],
    eval_fn: Callable[[str], float],
    client: LLMClient,
    model_config: dict,
    config: dict,
) -> tuple[str, dict]:
    """
    ProTeGi optimization loop.

    config keys:
        max_iterations (int, default 10)
        patience (int, default 3)
        failure_cases (list[dict]) — injected at runtime by run_optimization.py
        classification_summary (dict) — injected at runtime
    """
    max_iterations = config.get("max_iterations", 10)
    patience = config.get("patience", 3)
    failure_cases = config.get("failure_cases", [])
    classification_summary = config.get("classification_summary", {})

    baseline_score = eval_fn(prompt)
    best_prompt = prompt
    best_score = baseline_score
    no_improvement_count = 0
    log = []

    for iteration in range(1, max_iterations + 1):
        print(f"\n  [ProTeGi] Iteration {iteration}/{max_iterations}")
        gradient = generate_gradient(client, best_prompt, failure_cases, classification_summary)
        print(f"    Diagnosis: {gradient.get('diagnosis', 'N/A')[:100]}...")
        print(f"    Edits suggested: {len(gradient.get('edits', []))}")

        if not gradient.get("edits"):
            print("    No edits suggested. Stopping.")
            break

        candidate = apply_edits(client, best_prompt, gradient)
        candidate_score = eval_fn(candidate)
        accepted = candidate_score > best_score

        print(f"    Score: {candidate_score:.4f} (best: {best_score:.4f}) — {'ACCEPTED' if accepted else 'REJECTED'}")

        log.append({
            "iteration": iteration,
            "gradient_diagnosis": gradient.get("diagnosis", ""),
            "num_edits": len(gradient.get("edits", [])),
            "candidate_score": round(candidate_score, 4),
            "best_score": round(best_score, 4),
            "accepted": accepted,
        })

        if accepted:
            best_prompt, best_score, no_improvement_count = candidate, candidate_score, 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= patience:
            print(f"    No improvement for {patience} iterations. Stopping early.")
            break

    return best_prompt, {
        "original_score": round(baseline_score, 4),
        "optimized_score": round(best_score, 4),
        "improvement": round(best_score - baseline_score, 4),
        "iterations_run": len(log),
        "log": log,
    }
