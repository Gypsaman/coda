"""
CODA Phase 4: VALIDATE

Re-runs the evaluation suite with the optimized prompt on the new model.
Also runs regression test against the old model.
Produces a final validation report with pass/fail determination.

Usage:
    python scripts/run_validation.py --case a --results-dir results/case_a/<timestamp>
"""

import argparse
import json
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.llm_client import LLMClient
from scripts.run_diagnosis import EVALUATORS, load_test_suite
from evaluators.metrics import compute_mhs, compute_ppi, get_triage_zone


def run_validation(case_id: str, results_dir: str):
    config_path = Path(__file__).parent.parent / "config" / "models.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    case_config = config["cases"][case_id]

    results_path = Path(results_dir)
    diagnosis = json.loads((results_path / "diagnosis_report.json").read_text())
    optimization = json.loads((results_path / "optimization_log.json").read_text())

    # Load optimized prompt
    optimized_path = Path(optimization["optimized_prompt_path"])
    if not optimized_path.is_absolute():
        optimized_path = Path(__file__).parent.parent / optimized_path
    with open(optimized_path) as f:
        optimized_prompt = f.read()

    # Load test suite
    tests = load_test_suite(case_config["test_suite"])
    evaluator = EVALUATORS[case_id]
    client = LLMClient()

    print(f"\n{'='*60}")
    print(f"CODA Phase 4: VALIDATE - Case {case_id.upper()}")
    print(f"  Original PPI: {diagnosis['ppi']:.1f}")
    print(f"  Optimization improvement: {optimization['improvement']:+.4f}")
    print(f"{'='*60}\n")

    # Run optimized prompt on NEW model
    print("Evaluating optimized prompt on new model...")
    new_results = []
    for test in tqdm(tests, desc="New model (optimized)"):
        if isinstance(test["input"], dict):
            user_msg = test["input"].get("message", json.dumps(test["input"]))
        else:
            user_msg = str(test["input"])

        response = client.complete(
            provider=case_config["new_model"]["provider"],
            model=case_config["new_model"]["model"],
            system_prompt=optimized_prompt,
            user_message=user_msg,
            temperature=case_config["new_model"].get("temperature", 0.3),
            max_tokens=case_config["new_model"].get("max_tokens", 1024),
        )

        result = {
            "test_id": test["id"],
            "input": test["input"],
            "expected": test.get("expected", {}),
            "output_text": response["text"],
            "tool_calls": response.get("tool_calls", []),
            "usage": response["usage"],
            "latency_ms": response["latency_ms"],
        }
        result["scores"] = evaluator(result)
        new_results.append(result)

    # Run optimized prompt on OLD model (regression test)
    print("\nRegression test: optimized prompt on old model...")
    regression_results = []
    for test in tqdm(tests, desc="Old model (regression)"):
        if isinstance(test["input"], dict):
            user_msg = test["input"].get("message", json.dumps(test["input"]))
        else:
            user_msg = str(test["input"])

        response = client.complete(
            provider=case_config["old_model"]["provider"],
            model=case_config["old_model"]["model"],
            system_prompt=optimized_prompt,
            user_message=user_msg,
            temperature=case_config["old_model"].get("temperature", 0.3),
            max_tokens=case_config["old_model"].get("max_tokens", 1024),
        )

        result = {
            "test_id": test["id"],
            "input": test["input"],
            "expected": test.get("expected", {}),
            "output_text": response["text"],
            "tool_calls": response.get("tool_calls", []),
            "usage": response["usage"],
            "latency_ms": response["latency_ms"],
        }
        result["scores"] = evaluator(result)
        regression_results.append(result)

    # Aggregate scores
    def aggregate(results):
        all_scores = {}
        for r in results:
            for metric, score in r["scores"].items():
                if metric not in all_scores:
                    all_scores[metric] = []
                all_scores[metric].append(score)
        return {m: sum(s) / len(s) for m, s in all_scores.items()}

    new_agg = aggregate(new_results)
    regression_agg = aggregate(regression_results)
    baseline_agg = diagnosis["baseline_metrics"]

    # Compute MHS/PPI
    weights = case_config["evaluator_config"]["metric_weights"]

    def simple_mhs(agg):
        """Compute MHS from aggregated scores using available metrics."""
        available = {k: v for k, v in agg.items() if isinstance(v, (int, float))}
        if not available:
            return 0.0
        return sum(available.values()) / len(available)

    mhs_baseline = simple_mhs(baseline_agg)
    mhs_optimized_new = simple_mhs(new_agg)
    mhs_regression = simple_mhs(regression_agg)

    ppi_optimized = compute_ppi(mhs_optimized_new, mhs_baseline) if mhs_baseline > 0 else 0
    ppi_regression = compute_ppi(mhs_regression, mhs_baseline) if mhs_baseline > 0 else 0

    # Pass/fail criteria
    checks = {
        "ppi_meets_baseline": ppi_optimized >= 95,
        "no_metric_drops_over_10pct": True,
        "regression_acceptable": ppi_regression >= 90,
    }

    # Check individual metric regression > 10%
    for metric, baseline_val in baseline_agg.items():
        if isinstance(baseline_val, (int, float)) and baseline_val > 0:
            new_val = new_agg.get(metric, 0)
            if isinstance(new_val, (int, float)):
                decline = (baseline_val - new_val) / baseline_val
                if decline > 0.10:
                    checks["no_metric_drops_over_10pct"] = False

    passed = all(checks.values())

    # Build report
    report = {
        "case": case_id,
        "status": "PASS" if passed else "FAIL",
        "ppi_before_optimization": round(diagnosis["ppi"], 2),
        "ppi_after_optimization": round(ppi_optimized, 2),
        "ppi_regression_old_model": round(ppi_regression, 2),
        "triage_zone_before": diagnosis["triage_zone"],
        "triage_zone_after": get_triage_zone(ppi_optimized),
        "pass_fail_checks": checks,
        "metrics": {
            "baseline": {k: round(v, 4) if isinstance(v, float) else v
                         for k, v in baseline_agg.items()},
            "optimized_new_model": {k: round(v, 4) for k, v in new_agg.items()},
            "regression_old_model": {k: round(v, 4) for k, v in regression_agg.items()},
        },
    }

    with open(results_path / "validation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Summary file
    summary = {
        "case": case_id,
        "case_name": case_config["name"],
        "status": report["status"],
        "ppi_trajectory": {
            "before": diagnosis["ppi"],
            "after": ppi_optimized,
            "delta": round(ppi_optimized - diagnosis["ppi"], 2),
        },
        "triage_zone_trajectory": {
            "before": diagnosis["triage_zone"],
            "after": get_triage_zone(ppi_optimized),
        },
        "optimization_iterations": optimization["iterations_run"],
    }

    with open(results_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print results
    print(f"\n{'='*60}")
    print(f"VALIDATION {'PASSED' if passed else 'FAILED'}")
    print(f"{'='*60}")
    print(f"  PPI trajectory: {diagnosis['ppi']:.1f} -> {ppi_optimized:.1f} "
          f"({ppi_optimized - diagnosis['ppi']:+.1f})")
    print(f"  Zone: {diagnosis['triage_zone'].upper()} -> {get_triage_zone(ppi_optimized).upper()}")
    print(f"  Regression PPI (old model): {ppi_regression:.1f}")
    print(f"\n  Pass/fail checks:")
    for check, result in checks.items():
        status = "PASS" if result else "FAIL"
        print(f"    [{status}] {check}")

    if not passed:
        print(f"\n  RECOMMENDATION: Return to Phase 2 with updated failure data.")
    else:
        print(f"\n  Optimized prompt is ready for deployment.")

    print(f"\n  Reports saved to: {results_path}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CODA Phase 4: Validation")
    parser.add_argument("--case", required=True, choices=["a", "b", "c"])
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()

    run_validation(args.case, args.results_dir)
