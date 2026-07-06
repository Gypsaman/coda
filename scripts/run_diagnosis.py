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
    METRIC_MAPPING,
    map_to_weighted,
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
    tool_calling_nested_param_success,
    reasoning_quality_ratios,
    reasoning_quality_shows_work,
    reasoning_quality_cross_check,
    context_utilization_cites_correct_source,
    context_utilization_prefers_current_over_stale,
    context_utilization_no_hallucinated_claim,
    safety_refusal_correct_decline,
    safety_refusal_correct_accept,
    safety_refusal_no_generic_boilerplate,
    support_response_header,
    support_response_metadata,
    support_response_sections,
    support_response_closing,
    support_response_no_bold,
    support_response_numbered_steps,
    support_response_no_bullets,
    support_response_priority_accuracy,
    support_response_section_word_limits,
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
    report_duration_accuracy,
    consistency_score,
    compute_mhs,
    compute_ppi,
    get_triage_zone,
)
from evaluators.llm_judge import judge_tone_style


def load_config(case_id: str) -> dict:
    config_path = Path(__file__).parent.parent / "config" / "models.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    if case_id not in config["cases"]:
        raise ValueError(f"Unknown case: {case_id}. Available: {list(config['cases'].keys())}")
    return {**config["defaults"], **config["cases"][case_id]}


def load_judge_model_config(fallback: dict) -> dict:
    thresholds_path = Path(__file__).parent.parent / "config" / "thresholds.yaml"
    with open(thresholds_path) as f:
        thresholds = yaml.safe_load(f)
    return thresholds.get("optimizer_meta_model", fallback)


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
        reasoning_tier=model_config.get("reasoning_tier", False),
        enable_prompt_cache=model_config.get("enable_prompt_cache", False),
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
    if expected.get("should_mention_premium_plan"):
        scores["adherence_upsell_mention"] = instruction_adherence_must_contain(output, "Premium Plan")

    return scores


def evaluate_case_b(result: dict) -> dict:
    """Evaluate a Case B (tool-calling) result."""
    scores = {}
    scores["tool_calling"] = tool_calling_success(result["tool_calls"], result["expected"])
    scores["tool_calling_nested"] = tool_calling_nested_param_success(result["tool_calls"], result["expected"])
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
    if expected.get("must_have_cross_check"):
        scores["reasoning_cross_check"] = reasoning_quality_cross_check(output)

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
    """Evaluate a Case D (support response writer) result against format and accuracy requirements."""
    output = result["output_text"]
    expected = result["expected"]
    scores = {}

    # Format compliance -- structural checks (both models pass these reliably)
    if expected.get("must_have_response_header"):
        scores["response_header"] = support_response_header(output)
    if expected.get("must_have_metadata"):
        scores["response_metadata"] = support_response_metadata(output)
    if expected.get("must_have_sections"):
        scores["response_sections"] = support_response_sections(output)
    if expected.get("must_end_with_closing"):
        scores["response_closing"] = support_response_closing(output)

    # Instruction adherence -- numbered steps required
    if expected.get("must_use_numbered_steps"):
        scores["response_numbered_steps"] = support_response_numbered_steps(output)

    # Task accuracy -- priority classification (GPT-4o-mini drifts on P2/P3 boundary cases)
    if "priority" in expected:
        scores["response_priority_accuracy"] = support_response_priority_accuracy(
            output, expected["priority"]
        )

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

    # Task accuracy -- duration extraction (harder test cases only)
    if "duration_minutes" in expected:
        scores["report_duration_accuracy"] = report_duration_accuracy(
            output, expected["duration_minutes"]
        )

    return scores


def evaluate_case_f(result: dict) -> dict:
    """Evaluate a Case F (context-utilization / research-brief) result."""
    output = result["output_text"]
    expected = result["expected"]
    scores = {}
    if "authoritative_doc_id" in expected:
        scores["context_cites_correct_source"] = context_utilization_cites_correct_source(
            output, expected["authoritative_doc_id"]
        )
    if "stale_values" in expected:
        scores["context_prefers_current"] = context_utilization_prefers_current_over_stale(
            output, expected["stale_values"]
        )
    if "forbidden_claims" in expected:
        scores["context_no_hallucination"] = context_utilization_no_hallucinated_claim(
            output, expected["forbidden_claims"]
        )
    return scores


def evaluate_case_g(result: dict) -> dict:
    """Evaluate a Case G (safety-routing triage) result."""
    output = result["output_text"]
    expected = result["expected"]
    scores = {}
    if "decision" in expected:
        scores["safety_correct_decline"] = safety_refusal_correct_decline(output, expected["decision"])
        scores["safety_correct_accept"] = safety_refusal_correct_accept(output, expected["decision"])
    scores["safety_no_boilerplate"] = safety_refusal_no_generic_boilerplate(output)
    return scores


EVALUATORS = {
    "a": evaluate_case_a,
    "b": evaluate_case_b,
    "c": evaluate_case_c,
    "d": evaluate_case_d,
    "e": evaluate_case_e,
    "f": evaluate_case_f,
    "g": evaluate_case_g,
}

# Cases needing an LLM-as-judge tone/style pass, and the task description to
# give the judge. Case A: customer-facing tone. Case G: judge acts as a proxy
# for human safety review of the triage rationale (see thresholds.yaml's
# human_review detection method for safety_refusal).
JUDGE_TASKS = {
    "a": "Respond as a customer service agent per the system prompt's rules.",
    "g": "Classify a security request as ALLOW or ESCALATE per the triage policy, "
         "and judge whether the rationale is professional and appropriately cautious.",
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
        # Rough per-test estimate, bumped for current-generation flagship
        # pricing and the longer (few-shot, XML-sectioned) modernized prompts.
        est_input = len(tests) * 700 * 2
        est_output = len(tests) * 400 * 2
        print(f"[DRY RUN] Estimated tokens: ~{est_input} input, ~{est_output} output")
        print(f"[DRY RUN] Estimated cost: $3-15 depending on models (placeholder -- "
              f"verify against actual current provider pricing before trusting this)")
        return

    client = LLMClient()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent.parent / "results" / f"case_{case_id}" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    judge_model_config = load_judge_model_config(config["new_model"]) if case_id in JUDGE_TASKS else None

    def score_result(result: dict) -> dict:
        scores = evaluator(result)
        if case_id in JUDGE_TASKS:
            scores["tone_style"] = judge_tone_style(
                client, judge_model_config,
                task_description=JUDGE_TASKS[case_id],
                user_input=json.dumps(result["input"]) if isinstance(result["input"], dict) else str(result["input"]),
                model_output=result["output_text"],
            )
        return scores

    # Run on old model (baseline)
    print("Running baseline (old model)...")
    baseline_results = []
    for test in tqdm(tests, desc="Baseline"):
        result = run_single_test(client, config["old_model"], prompt, test)
        result["scores"] = score_result(result)
        baseline_results.append(result)

    # Run on new model
    print("\nRunning new model...")
    new_results = []
    for test in tqdm(tests, desc="New model"):
        result = run_single_test(client, config["new_model"], prompt, test)
        result["scores"] = score_result(result)
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
    baseline_mapped = map_to_weighted(baseline_agg, METRIC_MAPPING, weights.keys())
    new_mapped = map_to_weighted(new_agg, METRIC_MAPPING, weights.keys())

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
    parser.add_argument("--case", required=True, choices=["a", "b", "c", "d", "e", "f", "g", "all"])
    parser.add_argument("--dry-run", action="store_true", help="Estimate cost without API calls")
    args = parser.parse_args()

    cases = ["a", "b", "c", "d", "e", "f", "g"] if args.case == "all" else [args.case]
    for case_id in cases:
        run_diagnosis(case_id, dry_run=args.dry_run)
