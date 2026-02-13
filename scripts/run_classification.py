"""
CODA Phase 2: CLASSIFY

Reads the diagnostic report and raw failure data from Phase 1,
uses an LLM to classify each failure against the taxonomy,
and produces a prioritized classification report.

Usage:
    python scripts/run_classification.py --case a --results-dir results/case_a/<timestamp>
"""

import argparse
import json
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.llm_client import LLMClient
from evaluators.classifier import (
    build_classification_message,
    parse_classification,
    aggregate_classifications,
)


def load_prompt_text(case_config: dict) -> str:
    path = Path(__file__).parent.parent / case_config["prompt_file"]
    with open(path) as f:
        return f.read()


def run_classification(case_id: str, results_dir: str):
    # Load configs
    config_path = Path(__file__).parent.parent / "config" / "models.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    case_config = config["cases"][case_id]

    results_path = Path(results_dir)
    diagnosis = json.loads((results_path / "diagnosis_report.json").read_text())
    new_results = json.loads((results_path / "new_model_results.json").read_text())
    baseline_results = json.loads((results_path / "baseline_results.json").read_text())

    print(f"\n{'='*60}")
    print(f"CODA Phase 2: CLASSIFY - Case {case_id.upper()}")
    print(f"  PPI: {diagnosis['ppi']:.1f} ({diagnosis['triage_zone'].upper()} zone)")
    print(f"  Flagged degradations: {diagnosis['total_flagged']}")
    print(f"{'='*60}\n")

    if diagnosis["total_flagged"] == 0:
        print("No flagged degradations. Classification not needed.")
        report = {
            "case": case_id,
            "status": "no_failures_to_classify",
            "ppi": diagnosis["ppi"],
        }
        with open(results_path / "classification_report.json", "w") as f:
            json.dump(report, f, indent=2)
        return report

    # Identify failure cases (where any score < 1.0)
    prompt_text = load_prompt_text(case_config)
    client = LLMClient()
    classifications = []

    failure_cases = []
    for new_r, base_r in zip(new_results, baseline_results):
        # Check if this test case had degraded scores
        failed_metrics = []
        for metric, new_score in new_r.get("scores", {}).items():
            baseline_score = base_r.get("scores", {}).get(metric, 1.0)
            if new_score < baseline_score:
                failed_metrics.append(f"{metric}: {baseline_score:.2f} -> {new_score:.2f}")

        if failed_metrics:
            failure_cases.append({
                "test_id": new_r["test_id"],
                "input": new_r["input"],
                "expected": new_r["expected"],
                "baseline_output": base_r["output_text"],
                "new_output": new_r["output_text"],
                "new_tool_calls": new_r.get("tool_calls", []),
                "failed_metrics": failed_metrics,
            })

    print(f"Classifying {len(failure_cases)} failure cases...")

    for fc in tqdm(failure_cases, desc="Classifying"):
        # Build expected description
        expected_desc = json.dumps(fc["expected"], indent=2)

        failure_details = "Degraded metrics:\n" + "\n".join(f"  - {m}" for m in fc["failed_metrics"])
        if fc["new_tool_calls"]:
            failure_details += f"\n\nTool calls made: {json.dumps(fc['new_tool_calls'])}"

        messages = build_classification_message(
            prompt=prompt_text,
            test_input=json.dumps(fc["input"]) if isinstance(fc["input"], dict) else fc["input"],
            expected_output=expected_desc,
            actual_output=fc["new_output"] or "(no text output)",
            failure_details=failure_details,
        )

        # Use a capable model for classification
        response = client.complete(
            provider="openai",
            model="gpt-4o",
            system_prompt=messages[0]["content"],
            user_message=messages[1]["content"],
            temperature=0.1,
            max_tokens=500,
        )

        classification = parse_classification(response["text"])
        classification["test_id"] = fc["test_id"]
        classifications.append(classification)

    # Aggregate results
    summary = aggregate_classifications(classifications)

    # Build full report
    report = {
        "case": case_id,
        "ppi": diagnosis["ppi"],
        "triage_zone": diagnosis["triage_zone"],
        "total_failure_cases": len(failure_cases),
        "summary": summary,
        "individual_classifications": classifications,
    }

    with open(results_path / "classification_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"CLASSIFICATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Failures classified: {summary['total_failures_classified']}")
    print(f"  Category frequency:")
    for cat, count in summary["category_frequency"].items():
        print(f"    - {cat}: {count}")
    print(f"\n  Prioritized actions:")
    for item in summary["prioritized_action_items"][:3]:
        print(f"    1. [{item['category']}] score={item['priority_score']:.1f}, "
              f"freq={item['frequency']}")
        print(f"       Fix: {item['representative_fix'][:80]}...")
    print(f"\n  Report saved to: {results_path / 'classification_report.json'}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CODA Phase 2: Classification")
    parser.add_argument("--case", required=True, choices=["a", "b", "c", "d", "e"])
    parser.add_argument("--results-dir", required=True, help="Path to Phase 1 results directory")
    args = parser.parse_args()

    run_classification(args.case, args.results_dir)
