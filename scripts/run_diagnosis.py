"""
CODA Phase 1: DIAGNOSE

Runs the original prompt against both the old model (baseline) and the new model,
computes all metrics, flags degradation, and produces a diagnostic report.

Usage:
    python scripts/run_diagnosis.py --case a
    python scripts/run_diagnosis.py --case a --dry-run
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.llm_client import LLMClient
from evaluators.metrics import (
    format_compliance_json,
    format_compliance_max_words,
    format_compliance_ends_with,
    format_compliance_single_paragraph,
    format_compliance_no_bullets,
    format_compliance_no_exclamation,
    format_compliance_greeting,
    format_compliance_max_apologies,
    format_compliance_step_headers,
    format_compliance_inline_arithmetic,
    format_compliance_decimal_places,
    format_compliance_raw_json,
    format_compliance_exactly_n_risks,
    instruction_adherence_must_not_contain,
    instruction_adherence_must_contain,
    tool_calling_success,
    reasoning_quality_ratios,
    reasoning_quality_shows_work,
    cost_estimation_accuracy,
    cost_estimation_shows_work,
    cost_estimation_inline_arithmetic,
    cost_estimation_raw_json,
    cost_estimation_decimal_places,
    report_format_header,
    report_format_metadata,
    report_format_sections,
    report_format_closing,
    report_severity_valid,
    report_status_valid,
    report_severity_accuracy,
    report_status_accuracy,
    report_reporter_accuracy,
    report_section_word_limits,
    report_no_speculation,
    consistency_score,
    compute_mhs,
    compute_ppi,
    get_triage_zone,
)


def load_config(case_id: str) -> dict:
    config_path = Path(__file__).parent.parent / "config" / "models.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    if case_id not in config["cases"]:
        raise ValueError(f"Unknown case: {case_id}. Available: {list(config['cases'].keys())}")
    return {**config["defaults"], **config["cases"][case_id]}


def load_prompt(prompt_file: str) -> dict:
    path = Path(__file__).parent.parent / prompt_file
    if prompt_file.endswith(".json"):
        with open(path) as f:
            return json.load(f)
    else:
        with open(path) as f:
            return {"system_prompt": f.read(), "tools": None}


def load_test_suite(test_file: str) -> list[dict]:
    path = Path(__file__).parent.parent / test_file
    tests = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                tests.append(json.loads(line))
    return tests


def run_single_test(client: LLMClient, model_config: dict, prompt: dict, test_case: dict) -> dict:
    """Run a single test case and return the raw response."""
    # Build the user message from the test input
    if isinstance(test_case["input"], dict):
        user_msg = test_case["input"].get("message", json.dumps(test_case["input"]))
    else:
        user_msg = str(test_case["input"])

    response = client.complete(
        provider=model_config["provider"],
        model=model_config["model"],
        system_prompt=prompt["system_prompt"],
        user_message=user_msg,
        temperature=model_config.get("temperature", 0.3),
        max_tokens=model_config.get("max_tokens", 1024),
        tools=prompt.get("tools"),
    )

    return {
        "test_id": test_case["id"],
        "input": test_case["input"],
        "expected": test_case.get("expected", {}),
        "output_text": response["text"],
        "tool_calls": response["tool_calls"],
        "usage": response["usage"],
        "latency_ms": response["latency_ms"],
    }


def evaluate_case_a(result: dict) -> dict:
    """Evaluate a Case A (customer service) result against expected criteria."""
    output = result["output_text"]
    expected = result["expected"]
    scores = {}

    # Format checks -- these are where GPT-4o will fail
    if "max_words" in expected:
        scores["format_word_limit"] = format_compliance_max_words(output, expected["max_words"])
    if "must_end_with_closing" in expected and expected["must_end_with_closing"]:
        scores["format_closing"] = format_compliance_ends_with(
            output, "Is there anything else I can help with?"
        )
    if expected.get("must_be_single_paragraph"):
        scores["format_single_paragraph"] = format_compliance_single_paragraph(output)
    if expected.get("must_not_use_bullets"):
        scores["format_no_bullets"] = format_compliance_no_bullets(output)
    if expected.get("must_not_use_exclamation"):
        scores["format_no_exclamation"] = format_compliance_no_exclamation(output)
    if "greeting_format" in expected:
        scores["format_greeting"] = format_compliance_greeting(output, expected["greeting_format"])
    if "max_apologies" in expected:
        scores["format_max_apologies"] = format_compliance_max_apologies(
            output, expected["max_apologies"]
        )

    # Instruction adherence
    if "must_not_mention_competitors" in expected and expected["must_not_mention_competitors"]:
        scores["adherence_no_competitors"] = instruction_adherence_must_not_contain(
            output, ["CompetitorX", "CompetitorY", "CompetitorZ"]
        )
    if expected.get("should_escalate") and "escalation_phrase" in expected:
        scores["adherence_escalation"] = instruction_adherence_must_contain(
            output, expected["escalation_phrase"]
        )
    if expected.get("must_not_promise_refund"):
        scores["adherence_no_refund_promise"] = instruction_adherence_must_not_contain(
            output, [
                "I can process your refund", "I'll issue a refund", "we'll refund",
                "I can offer you a refund", "refund will be", "process a refund",
                "full refund", "partial refund",
            ]
        )

    return scores


def evaluate_case_b(result: dict) -> dict:
    """Evaluate a Case B (tool-calling) result."""
    scores = {}
    scores["tool_calling"] = tool_calling_success(result["tool_calls"], result["expected"])
    return scores


def evaluate_case_c(result: dict) -> dict:
    """Evaluate a Case C (financial analysis CoT) result against strict formatting requirements."""
    output = result["output_text"]
    expected = result["expected"]
    scores = {}

    # Reasoning accuracy
    if "ratios" in expected:
        scores["reasoning_ratios"] = reasoning_quality_ratios(output, expected["ratios"])
    scores["reasoning_shows_work"] = reasoning_quality_shows_work(output)

    # Format compliance -- these are where Claude Haiku will fail
    if expected.get("must_produce_valid_json"):
        scores["format_json"] = format_compliance_json(output)
    if expected.get("must_have_step_headers"):
        scores["format_step_headers"] = format_compliance_step_headers(output)
    if expected.get("must_show_inline_arithmetic"):
        scores["format_inline_arithmetic"] = format_compliance_inline_arithmetic(output)
    if expected.get("must_have_4_decimal_places"):
        scores["format_decimal_places"] = format_compliance_decimal_places(output, 4)
    if expected.get("json_must_be_raw"):
        scores["format_raw_json"] = format_compliance_raw_json(output)
    if expected.get("must_have_exactly_2_risks"):
        scores["format_exactly_2_risks"] = format_compliance_exactly_n_risks(output, 2)

    return scores


def evaluate_case_d(result: dict) -> dict:
    """Evaluate a Case D (cost estimation) result against expected calculations."""
    output = result["output_text"]
    expected = result["expected"]
    scores = {}

    # Cost accuracy -- these are where GPT-4o-mini will fail
    if "costs" in expected:
        scores["cost_accuracy"] = cost_estimation_accuracy(output, expected["costs"])
    scores["cost_shows_work"] = cost_estimation_shows_work(output)

    # Format compliance
    if expected.get("must_produce_valid_json"):
        scores["format_json"] = format_compliance_json(output)
    if expected.get("must_have_step_headers"):
        scores["format_step_headers"] = format_compliance_step_headers(output)
    if expected.get("must_show_inline_arithmetic"):
        scores["format_inline_arithmetic"] = cost_estimation_inline_arithmetic(output)
    if expected.get("json_must_be_raw"):
        scores["format_raw_json"] = cost_estimation_raw_json(output)
    if expected.get("must_have_2_decimal_places"):
        scores["format_decimal_places"] = cost_estimation_decimal_places(output)

    return scores


def evaluate_case_e(result: dict) -> dict:
    """Evaluate a Case E (incident report) result against format and accuracy requirements."""
    output = result["output_text"]
    expected = result["expected"]
    scores = {}

    # Format compliance -- these are where GPT-4o-mini will degrade
    if expected.get("must_have_report_header"):
        scores["report_header"] = report_format_header(output)
    if expected.get("must_have_metadata"):
        scores["report_metadata"] = report_format_metadata(output)
    if expected.get("must_have_sections"):
        scores["report_sections"] = report_format_sections(output)
    if expected.get("must_end_with_closing"):
        scores["report_closing"] = report_format_closing(output)
    if expected.get("must_not_use_exclamation"):
        scores["report_no_exclamation"] = format_compliance_no_exclamation(output)
    if expected.get("must_not_use_speculation"):
        scores["report_no_speculation"] = report_no_speculation(output)

    # Task accuracy -- severity/status/reporter extraction
    if "severity" in expected:
        scores["report_severity_valid"] = report_severity_valid(output)
        scores["report_severity_accuracy"] = report_severity_accuracy(output, expected["severity"])
    if "status" in expected:
        scores["report_status_valid"] = report_status_valid(output)
        scores["report_status_accuracy"] = report_status_accuracy(output, expected["status"])
    if "reporter" in expected:
        scores["report_reporter_accuracy"] = report_reporter_accuracy(output, expected["reporter"])

    # Instruction adherence -- word limits per section
    if "section_word_limits" in expected:
        scores["report_word_limits"] = report_section_word_limits(output, expected["section_word_limits"])

    return scores


EVALUATORS = {
    "a": evaluate_case_a,
    "b": evaluate_case_b,
    "c": evaluate_case_c,
    "d": evaluate_case_d,
    "e": evaluate_case_e,
}


def run_diagnosis(case_id: str, dry_run: bool = False):
    config = load_config(case_id)
    prompt = load_prompt(config["prompt_file"])
    tests = load_test_suite(config["test_suite"])
    evaluator = EVALUATORS[case_id]

    print(f"\n{'='*60}")
    print(f"CODA Phase 1: DIAGNOSE - Case {case_id.upper()}")
    print(f"  {config['name']}: {config['description']}")
    print(f"  Old model: {config['old_model']['provider']}/{config['old_model']['model']}")
    print(f"  New model: {config['new_model']['provider']}/{config['new_model']['model']}")
    print(f"  Test cases: {len(tests)}")
    print(f"{'='*60}\n")

    if dry_run:
        est_input = len(tests) * 500 * 2  # rough estimate: 500 tokens/test, 2 models
        est_output = len(tests) * 300 * 2
        print(f"[DRY RUN] Estimated tokens: ~{est_input} input, ~{est_output} output")
        print(f"[DRY RUN] Estimated cost: $2-8 depending on models")
        return

    client = LLMClient()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent.parent / "results" / f"case_{case_id}" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run on old model (baseline)
    print("Running baseline (old model)...")
    baseline_results = []
    for test in tqdm(tests, desc="Baseline"):
        result = run_single_test(client, config["old_model"], prompt, test)
        result["scores"] = evaluator(result)
        baseline_results.append(result)

    # Run on new model
    print("\nRunning new model...")
    new_results = []
    for test in tqdm(tests, desc="New model"):
        result = run_single_test(client, config["new_model"], prompt, test)
        result["scores"] = evaluator(result)
        new_results.append(result)

    # Aggregate metrics
    def aggregate_scores(results):
        all_scores = {}
        for r in results:
            for metric, score in r["scores"].items():
                if metric not in all_scores:
                    all_scores[metric] = []
                all_scores[metric].append(score)
        return {m: sum(s) / len(s) for m, s in all_scores.items()}

    baseline_agg = aggregate_scores(baseline_results)
    new_agg = aggregate_scores(new_results)

    # Compute MHS
    weights = config["evaluator_config"]["metric_weights"]
    # Map detailed scores to weight categories
    metric_mapping = {
        "task_accuracy": ["reasoning_ratios", "cost_accuracy",
                          "report_severity_accuracy", "report_status_accuracy",
                          "report_reporter_accuracy"],
        "format_compliance": ["format_word_limit", "format_closing", "format_json",
                              "report_header", "report_metadata", "report_sections",
                              "report_closing", "report_severity_valid",
                              "report_status_valid"],
        "instruction_adherence": [
            "adherence_no_competitors", "adherence_escalation",
            "adherence_no_refund_promise",
            "report_no_exclamation", "report_no_speculation",
            "report_word_limits",
        ],
        "tool_calling_success": ["tool_calling"],
        "reasoning_quality": ["reasoning_ratios", "reasoning_shows_work",
                              "cost_accuracy", "cost_shows_work"],
        "consistency": [],
    }

    def map_to_weighted(agg_scores, mapping, weight_keys):
        mapped = {}
        for wk in weight_keys:
            relevant = [agg_scores[m] for m in mapping.get(wk, []) if m in agg_scores]
            if relevant:
                mapped[wk] = sum(relevant) / len(relevant)
        return mapped

    baseline_mapped = map_to_weighted(baseline_agg, metric_mapping, weights.keys())
    new_mapped = map_to_weighted(new_agg, metric_mapping, weights.keys())

    mhs_baseline = compute_mhs(baseline_mapped, weights)
    mhs_new = compute_mhs(new_mapped, weights)
    ppi = compute_ppi(mhs_new, mhs_baseline)
    zone = get_triage_zone(ppi)

    # Find flagged failures (>5% relative decline)
    threshold = config.get("degradation_threshold", 0.05)
    flagged = []
    for metric in set(list(baseline_agg.keys()) + list(new_agg.keys())):
        old_val = baseline_agg.get(metric, 0)
        new_val = new_agg.get(metric, 0)
        if old_val > 0:
            relative_decline = (old_val - new_val) / old_val
            if relative_decline > threshold:
                flagged.append({
                    "metric": metric,
                    "baseline": round(old_val, 4),
                    "new_model": round(new_val, 4),
                    "relative_decline": round(relative_decline, 4),
                })

    # Build diagnostic report
    report = {
        "case": case_id,
        "timestamp": timestamp,
        "config": {
            "old_model": config["old_model"],
            "new_model": config["new_model"],
            "test_cases": len(tests),
        },
        "baseline_metrics": {k: round(v, 4) for k, v in baseline_agg.items()},
        "new_model_metrics": {k: round(v, 4) for k, v in new_agg.items()},
        "mhs_baseline": round(mhs_baseline, 4),
        "mhs_new_model": round(mhs_new, 4),
        "ppi": round(ppi, 2),
        "triage_zone": zone,
        "flagged_degradations": flagged,
        "total_flagged": len(flagged),
    }

    # Save report and raw results
    with open(output_dir / "diagnosis_report.json", "w") as f:
        json.dump(report, f, indent=2)

    with open(output_dir / "baseline_results.json", "w") as f:
        json.dump(baseline_results, f, indent=2, default=str)

    with open(output_dir / "new_model_results.json", "w") as f:
        json.dump(new_results, f, indent=2, default=str)

    # Print summary
    print(f"\n{'='*60}")
    print(f"DIAGNOSIS COMPLETE")
    print(f"{'='*60}")
    print(f"  MHS (baseline):  {report['mhs_baseline']:.4f}")
    print(f"  MHS (new model): {report['mhs_new_model']:.4f}")
    print(f"  PPI:             {report['ppi']:.1f}")
    print(f"  Triage zone:     {zone.upper()}")
    print(f"  Flagged metrics: {len(flagged)}")
    for f_item in flagged:
        print(f"    - {f_item['metric']}: {f_item['baseline']:.3f} -> {f_item['new_model']:.3f} "
              f"({f_item['relative_decline']:.1%} decline)")
    print(f"\n  Results saved to: {output_dir}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CODA Phase 1: Diagnosis")
    parser.add_argument("--case", required=True, choices=["a", "b", "c", "d", "e", "all"])
    parser.add_argument("--dry-run", action="store_true", help="Estimate cost without API calls")
    args = parser.parse_args()

    cases = ["a", "b", "c", "d", "e"] if args.case == "all" else [args.case]
    for case_id in cases:
        run_diagnosis(case_id, dry_run=args.dry_run)
