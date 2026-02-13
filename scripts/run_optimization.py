"""
CODA Phase 3: OPTIMIZE

Reads the classification report from Phase 2 and applies targeted
optimization strategies based on failure type.

Implements a simplified ProTeGi-style textual gradient optimization loop:
1. Collect failure cases as a mini-batch
2. Ask an LLM to generate a "textual gradient" (criticism + edit suggestion)
3. Ask an LLM to apply the gradient (edit the prompt)
4. Evaluate the edited prompt on the failure cases
5. Keep the edit if it improves scores; revert if not
6. Repeat until convergence or max iterations

Usage:
    python scripts/run_optimization.py --case a --results-dir results/case_a/<timestamp>
"""

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

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
    # Build failure case descriptions
    case_descriptions = []
    for fc in failure_cases[:5]:  # Limit to 5 cases per gradient
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

    # Parse response
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
    """Apply the suggested edits to produce an updated prompt."""
    if not gradient.get("edits"):
        return current_prompt

    edits_text = json.dumps(gradient["edits"], indent=2)

    response = client.complete(
        provider="openai",
        model="gpt-4o",
        system_prompt=APPLY_EDIT_SYSTEM_PROMPT,
        user_message=f"## Original Prompt\n{current_prompt}\n\n## Edits to Apply\n{edits_text}",
        temperature=0.1,
        max_tokens=2000,
    )

    return response["text"].strip()


def quick_evaluate(client, model_config, prompt_text, test_cases, evaluator_fn) -> float:
    """Run a quick evaluation on a subset of test cases. Returns average score."""
    scores = []
    for test in test_cases:
        if isinstance(test["input"], dict):
            user_msg = test["input"].get("message", json.dumps(test["input"]))
        else:
            user_msg = str(test["input"])

        response = client.complete(
            provider=model_config["provider"],
            model=model_config["model"],
            system_prompt=prompt_text,
            user_message=user_msg,
            temperature=model_config.get("temperature", 0.3),
            max_tokens=model_config.get("max_tokens", 1024),
        )

        result = {
            "test_id": test["id"],
            "input": test["input"],
            "expected": test.get("expected", {}),
            "output_text": response["text"],
            "tool_calls": response.get("tool_calls", []),
        }
        case_scores = evaluator_fn(result)
        if case_scores:
            scores.append(sum(case_scores.values()) / len(case_scores))

    return sum(scores) / len(scores) if scores else 0.0


def run_optimization(case_id: str, results_dir: str):
    config_path = Path(__file__).parent.parent / "config" / "models.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    case_config = config["cases"][case_id]
    defaults = config["defaults"]

    results_path = Path(results_dir)
    classification = json.loads((results_path / "classification_report.json").read_text())
    new_results = json.loads((results_path / "new_model_results.json").read_text())

    # Load original prompt
    prompt_file = Path(__file__).parent.parent / case_config["prompt_file"]
    if str(prompt_file).endswith(".json"):
        with open(prompt_file) as f:
            prompt_data = json.load(f)
        current_prompt = prompt_data["system_prompt"]
    else:
        with open(prompt_file) as f:
            current_prompt = f.read()

    print(f"\n{'='*60}")
    print(f"CODA Phase 3: OPTIMIZE - Case {case_id.upper()}")
    print(f"  Zone: {classification['triage_zone'].upper()}")
    print(f"  Primary failure: {classification['summary']['prioritized_action_items'][0]['category']}")
    print(f"{'='*60}\n")

    # Import the right evaluator
    from scripts.run_diagnosis import EVALUATORS
    evaluator_fn = EVALUATORS[case_id]

    # Collect failure cases for gradient computation
    failure_cases = []
    baseline_results = json.loads((results_path / "baseline_results.json").read_text())
    for new_r, base_r in zip(new_results, baseline_results):
        failed_metrics = []
        for metric, new_score in new_r.get("scores", {}).items():
            baseline_score = base_r.get("scores", {}).get(metric, 1.0)
            if new_score < baseline_score:
                failed_metrics.append(f"{metric}: {baseline_score:.2f} -> {new_score:.2f}")
        if failed_metrics:
            failure_cases.append({
                "input": new_r["input"],
                "expected": new_r["expected"],
                "new_output": new_r["output_text"],
                "failed_metrics": failed_metrics,
            })

    # Load test cases for quick evaluation
    test_path = Path(__file__).parent.parent / case_config["test_suite"]
    test_cases = []
    with open(test_path) as f:
        for line in f:
            if line.strip():
                test_cases.append(json.loads(line))

    client = LLMClient()
    max_iterations = defaults.get("optimization_max_iterations", 10)
    patience = defaults.get("optimization_patience", 3)

    # Baseline score on new model
    print("Computing baseline score on new model...")
    baseline_score = quick_evaluate(
        client, case_config["new_model"], current_prompt, test_cases, evaluator_fn
    )
    print(f"  Baseline score: {baseline_score:.4f}")

    best_prompt = current_prompt
    best_score = baseline_score
    no_improvement_count = 0
    optimization_log = []

    for iteration in range(1, max_iterations + 1):
        print(f"\n--- Iteration {iteration}/{max_iterations} ---")

        # Generate textual gradient
        print("  Generating gradient...")
        gradient = generate_gradient(
            client, best_prompt, failure_cases, classification["summary"]
        )
        print(f"  Diagnosis: {gradient.get('diagnosis', 'N/A')[:100]}...")
        print(f"  Suggested edits: {len(gradient.get('edits', []))}")

        if not gradient.get("edits"):
            print("  No edits suggested. Stopping.")
            break

        # Apply edits
        print("  Applying edits...")
        candidate_prompt = apply_edits(client, best_prompt, gradient)

        # Evaluate candidate
        print("  Evaluating candidate...")
        candidate_score = quick_evaluate(
            client, case_config["new_model"], candidate_prompt, test_cases, evaluator_fn
        )
        print(f"  Candidate score: {candidate_score:.4f} (best: {best_score:.4f})")

        # Log this iteration
        optimization_log.append({
            "iteration": iteration,
            "gradient_diagnosis": gradient.get("diagnosis", ""),
            "num_edits": len(gradient.get("edits", [])),
            "candidate_score": round(candidate_score, 4),
            "best_score": round(best_score, 4),
            "accepted": candidate_score > best_score,
        })

        # Accept or reject
        if candidate_score > best_score:
            print(f"  ACCEPTED (+{candidate_score - best_score:.4f})")
            best_prompt = candidate_prompt
            best_score = candidate_score
            no_improvement_count = 0
        else:
            print(f"  REJECTED")
            no_improvement_count += 1

        if no_improvement_count >= patience:
            print(f"\n  No improvement for {patience} iterations. Stopping early.")
            break

    # Save optimized prompt
    optimized_path = Path(__file__).parent.parent / "prompts" / f"case_{case_id}_optimized.txt"
    with open(optimized_path, "w") as f:
        f.write(best_prompt)

    # Save optimization log
    log_output = {
        "case": case_id,
        "original_score": round(baseline_score, 4),
        "optimized_score": round(best_score, 4),
        "improvement": round(best_score - baseline_score, 4),
        "iterations_run": len(optimization_log),
        "log": optimization_log,
        "optimized_prompt_path": str(optimized_path),
    }
    with open(results_path / "optimization_log.json", "w") as f:
        json.dump(log_output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Original score:  {baseline_score:.4f}")
    print(f"  Optimized score: {best_score:.4f}")
    print(f"  Improvement:     {best_score - baseline_score:+.4f}")
    print(f"  Iterations:      {len(optimization_log)}")
    print(f"  Optimized prompt saved to: {optimized_path}")

    return log_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CODA Phase 3: Optimization")
    parser.add_argument("--case", required=True, choices=["a", "b", "c"])
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()

    run_optimization(args.case, args.results_dir)
