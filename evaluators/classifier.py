"""
LLM-as-classifier for mapping prompt failures to the CODA taxonomy.

Uses an LLM to categorize each failure case into one or more taxonomy categories,
producing a structured classification report.
"""

import json
from typing import Any

CLASSIFIER_SYSTEM_PROMPT = """You are an expert prompt failure classifier. Given a prompt, its expected output, 
and its actual output from a language model, classify the failure into one or more categories from this taxonomy:

1. format_compliance - Output doesn't match the required structural format (JSON invalid, wrong delimiters, extra text around structured output, missing required sections)
2. reasoning_quality - Logical reasoning is degraded (calculation errors, skipped steps, wrong conclusions, shallow analysis)
3. instruction_drift - Model doesn't follow explicit constraints (exceeds length limits, ignores negative constraints like "do NOT", violates persona rules, breaks scope restrictions)
4. tool_calling - Tool invocations are wrong (incorrect tool chosen, missing parameters, hallucinated function names, malformed arguments)
5. safety_refusal - Model refuses a valid request it shouldn't, or accepts something it should refuse
6. tone_style - Output tone, verbosity, formality, or communication style differs significantly from what's expected
7. context_utilization - Model mishandles information placement, ignores parts of long context, or fails to integrate retrieved information

Respond with a JSON object:
{
  "categories": ["<primary_category>", "<secondary_category_if_any>"],
  "primary_category": "<the most impactful failure>",
  "severity": "critical" | "high" | "medium" | "low",
  "explanation": "<brief explanation of why this failure occurred>",
  "suggested_fix_direction": "<brief suggestion for what to change in the prompt>"
}

Only return the JSON. No other text."""


def build_classification_message(
    prompt: str,
    test_input: str,
    expected_output: str,
    actual_output: str,
    failure_details: str,
) -> list[dict]:
    """Build the message for the classifier LLM."""
    return [
        {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"## Original Prompt\n{prompt}\n\n"
                f"## Test Input\n{test_input}\n\n"
                f"## Expected Behavior\n{expected_output}\n\n"
                f"## Actual Output\n{actual_output}\n\n"
                f"## Detected Failures\n{failure_details}"
            ),
        },
    ]


def parse_classification(raw_response: str) -> dict:
    """Parse the classifier's JSON response, with fallback handling."""
    # Strip markdown fences if present
    cleaned = raw_response.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    if cleaned.startswith("json"):
        cleaned = cleaned[4:].strip()

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        result = {
            "categories": ["unknown"],
            "primary_category": "unknown",
            "severity": "medium",
            "explanation": f"Failed to parse classifier output: {raw_response[:200]}",
            "suggested_fix_direction": "Manual review required",
        }

    # Validate required fields
    for field in ["categories", "primary_category", "severity"]:
        if field not in result:
            result[field] = "unknown" if field != "categories" else ["unknown"]

    return result


def aggregate_classifications(classifications: list[dict]) -> dict:
    """
    Aggregate individual failure classifications into a summary report.

    Returns:
        Dict with category frequencies, severity distribution, and prioritized action items.
    """
    category_counts: dict[str, int] = {}
    severity_counts: dict[str, int] = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    category_severity: dict[str, list[str]] = {}
    fix_suggestions: dict[str, list[str]] = {}

    for c in classifications:
        primary = c.get("primary_category", "unknown")
        severity = c.get("severity", "medium")

        # Count categories
        for cat in c.get("categories", [primary]):
            category_counts[cat] = category_counts.get(cat, 0) + 1
            if cat not in category_severity:
                category_severity[cat] = []
            category_severity[cat].append(severity)

        # Count severities
        if severity in severity_counts:
            severity_counts[severity] += 1

        # Collect fix suggestions
        fix = c.get("suggested_fix_direction", "")
        if fix and primary not in fix_suggestions:
            fix_suggestions[primary] = []
        if fix:
            fix_suggestions[primary].append(fix)

    # Sort categories by frequency
    sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)

    # Build prioritized action items
    severity_weight = {"critical": 4, "high": 3, "medium": 2, "low": 1}
    priority_scores = {}
    for cat, severities in category_severity.items():
        avg_severity = sum(severity_weight.get(s, 2) for s in severities) / len(severities)
        frequency = category_counts[cat]
        priority_scores[cat] = avg_severity * frequency

    prioritized = sorted(priority_scores.items(), key=lambda x: x[1], reverse=True)

    return {
        "total_failures_classified": len(classifications),
        "category_frequency": dict(sorted_categories),
        "severity_distribution": severity_counts,
        "prioritized_action_items": [
            {
                "category": cat,
                "priority_score": round(score, 2),
                "frequency": category_counts[cat],
                "representative_fix": fix_suggestions.get(cat, ["No suggestion"])[0],
            }
            for cat, score in prioritized
        ],
    }
