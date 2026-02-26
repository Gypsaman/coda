"""
CODA Phase 3: OPTIMIZE

Reads the classification report from Phase 2, selects the appropriate
optimizer backend via the CODA router, and runs the optimization loop.

Optimizer selection is driven by two factors:
  1. PPI triage zone (green/yellow/orange/red/critical)
  2. Primary failure category from the taxonomy

Routing table:
  green    → no optimization
  yellow   → ProTeGi (light, 3 iterations)
  orange   → ProTeGi (full, up to 10 iterations)
  red      → OPRO (trajectory-guided)
  critical → EvoPrompt (evolutionary) or APE (for reasoning_quality failures)

Available optimizers:
  ProTeGi   — textual gradient descent (Pryzant et al. 2023)
  APE       — parallel candidate generation (Zhou et al. 2023)
  OPRO      — trajectory-guided optimization (Yang et al. 2024)
  EvoPrompt — evolutionary population search (Guo et al. 2024)

Usage:
    python scripts/run_optimization.py --case a --results-dir results/case_a/<timestamp>
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.llm_client import LLMClient
from optimizers.router import route_and_optimize


def quick_evaluate(client, model_config, prompt_text, test_cases, evaluator_fn) -> float:
    """Run evaluation on the full test suite. Returns average score [0, 1]."""
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
    # Load configs
    root = Path(__file__).parent.parent
    config_path = root / "config" / "models.yaml"
    thresholds_path = root / "config" / "thresholds.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)
    with open(thresholds_path) as f:
        thresholds = yaml.safe_load(f)

    case_config = config["cases"][case_id]
    optimizer_configs = thresholds.get("optimizer_configs", {})

    # Load results from previous phases
    results_path = Path(results_dir)
    classification = json.loads((results_path / "classification_report.json").read_text())
    new_results = json.loads((results_path / "new_model_results.json").read_text())
    baseline_results = json.loads((results_path / "baseline_results.json").read_text())

    # Load original prompt
    prompt_file = root / case_config["prompt_file"]
    if str(prompt_file).endswith(".json"):
        with open(prompt_file) as f:
            current_prompt = json.load(f)["system_prompt"]
    else:
        with open(prompt_file) as f:
            current_prompt = f.read()

    print(f"\n{'='*60}")
    print(f"CODA Phase 3: OPTIMIZE - Case {case_id.upper()}")
    print(f"  Zone: {classification['triage_zone'].upper()}")
    try:
        primary = classification["summary"]["prioritized_action_items"][0]["category"]
        print(f"  Primary failure: {primary}")
    except (KeyError, IndexError):
        pass
    print(f"{'='*60}")

    # Load evaluator and test cases
    from scripts.run_diagnosis import EVALUATORS
    evaluator_fn = EVALUATORS[case_id]

    test_path = root / case_config["test_suite"]
    test_cases = []
    with open(test_path) as f:
        for line in f:
            if line.strip():
                test_cases.append(json.loads(line))

    # Collect failure cases (inputs where new model underperformed baseline)
    failure_cases = []
    for new_r, base_r in zip(new_results, baseline_results):
        failed_metrics = []
        for metric, new_score in new_r.get("scores", {}).items():
            base_score = base_r.get("scores", {}).get(metric, 1.0)
            if new_score < base_score:
                failed_metrics.append(f"{metric}: {base_score:.2f} -> {new_score:.2f}")
        if failed_metrics:
            failure_cases.append({
                "input": new_r["input"],
                "expected": new_r["expected"],
                "new_output": new_r["output_text"],
                "failed_metrics": failed_metrics,
            })

    # Inject runtime context into each optimizer's config
    classification_summary = classification.get("summary", {})
    for opt_name in optimizer_configs:
        optimizer_configs[opt_name]["failure_cases"] = failure_cases
        optimizer_configs[opt_name]["classification_summary"] = classification_summary

    # Build eval_fn closure (binds client + model_config + evaluator)
    client = LLMClient()
    eval_fn = lambda prompt_text: quick_evaluate(
        client, case_config["new_model"], prompt_text, test_cases, evaluator_fn
    )

    # Dispatch to router
    best_prompt, log_dict = route_and_optimize(
        classification_report=classification,
        prompt=current_prompt,
        test_cases=test_cases,
        eval_fn=eval_fn,
        client=client,
        model_config=case_config["new_model"],
        optimizer_configs=optimizer_configs,
    )

    # Save optimized prompt
    optimized_path = root / "prompts" / f"case_{case_id}_optimized.txt"
    with open(optimized_path, "w") as f:
        f.write(best_prompt)

    # Save optimization log
    log_output = {
        "case": case_id,
        **log_dict,
        "optimized_prompt_path": str(optimized_path),
    }
    with open(results_path / "optimization_log.json", "w") as f:
        json.dump(log_output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Optimizer:       {log_dict.get('optimizer', 'none')}")
    if log_dict.get("original_score") is not None:
        print(f"  Original score:  {log_dict['original_score']:.4f}")
        print(f"  Optimized score: {log_dict['optimized_score']:.4f}")
        print(f"  Improvement:     {log_dict['improvement']:+.4f}")
    print(f"  Iterations:      {log_dict.get('iterations_run', 0)}")
    print(f"  Optimized prompt saved to: {optimized_path}")

    return log_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CODA Phase 3: Optimization")
    parser.add_argument("--case", required=True, choices=["a", "b", "c", "d", "e"])
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()

    run_optimization(args.case, args.results_dir)
