"""
Core metric computation for CODA evaluation.

Each metric function takes model outputs and expected values,
returns a score between 0.0 and 1.0.
"""

import json
import re
from typing import Any


def word_count(text: str) -> int:
    return len(text.split())


def task_accuracy_exact(output: str, expected: str) -> float:
    """Exact match accuracy. Returns 1.0 or 0.0."""
    return 1.0 if output.strip() == expected.strip() else 0.0


def task_accuracy_numeric(output_value: float, expected_value: float, tolerance: float = 0.01) -> float:
    """Numeric accuracy within tolerance. Returns 1.0 if within tolerance, else partial credit."""
    if expected_value == 0:
        return 1.0 if abs(output_value) < tolerance else 0.0
    relative_error = abs(output_value - expected_value) / abs(expected_value)
    if relative_error <= tolerance:
        return 1.0
    elif relative_error <= tolerance * 5:
        return 0.5
    else:
        return 0.0


def format_compliance_json(output: str) -> float:
    """Check if output contains valid JSON. Returns 1.0 if valid, 0.0 if not."""
    # Try to find JSON in the output
    json_patterns = [
        r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # nested one level
        r'\{.*\}',  # greedy
    ]
    for pattern in json_patterns:
        matches = re.findall(pattern, output, re.DOTALL)
        for match in matches:
            try:
                json.loads(match)
                return 1.0
            except json.JSONDecodeError:
                continue
    return 0.0


def format_compliance_max_words(output: str, max_words: int) -> float:
    """Check if output is within word limit. Returns 1.0 if compliant, 0.0 if not."""
    return 1.0 if word_count(output) <= max_words else 0.0


def format_compliance_ends_with(output: str, expected_ending: str) -> float:
    """Check if output ends with expected phrase. Returns 1.0 if yes, 0.0 if not."""
    return 1.0 if output.strip().endswith(expected_ending) else 0.0


def instruction_adherence_must_not_contain(output: str, forbidden_terms: list[str]) -> float:
    """Check that output does not contain any forbidden terms. Case-insensitive."""
    output_lower = output.lower()
    for term in forbidden_terms:
        if term.lower() in output_lower:
            return 0.0
    return 1.0


def instruction_adherence_must_contain(output: str, required_phrase: str) -> float:
    """Check that output contains a required phrase. Case-insensitive."""
    return 1.0 if required_phrase.lower() in output.lower() else 0.0


def tool_calling_success(tool_calls: list[dict], expected: dict) -> float:
    """
    Evaluate tool-calling correctness.

    Args:
        tool_calls: List of tool call dicts with 'name' and 'arguments' keys.
        expected: Dict with 'correct_tool', 'required_params', optionally 'expected_source'.

    Returns:
        Score between 0.0 and 1.0.
    """
    if not tool_calls:
        return 0.0

    score = 0.0
    total_checks = 0

    # Check if correct tool was called
    if "correct_tool" in expected:
        total_checks += 1
        called_tools = [tc.get("name") for tc in tool_calls]
        if expected["correct_tool"] in called_tools:
            score += 1.0

    # Check if wrong tool was NOT called
    if "should_not_use" in expected:
        total_checks += 1
        called_tools = [tc.get("name") for tc in tool_calls]
        if expected["should_not_use"] not in called_tools:
            score += 1.0

    # Check required parameters
    if "required_params" in expected and "correct_tool" in expected:
        for tc in tool_calls:
            if tc.get("name") == expected["correct_tool"]:
                args = tc.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                for param in expected["required_params"]:
                    total_checks += 1
                    if param in args and args[param] is not None:
                        score += 1.0
                break

    # Check expected source parameter value
    if "expected_source" in expected:
        total_checks += 1
        for tc in tool_calls:
            if tc.get("name") == "query_database":
                args = tc.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                if args.get("source") == expected["expected_source"]:
                    score += 1.0
                break

    return score / total_checks if total_checks > 0 else 0.0


def reasoning_quality_ratios(output: str, expected_ratios: dict[str, float], tolerance: float = 0.02) -> float:
    """
    Check if financial ratios in the output match expected values.

    Extracts the JSON summary from the output and compares ratio values.
    """
    # Try to extract JSON from output
    json_match = None
    for match in re.finditer(r'\{[^{}]*"ratios"[^{}]*\{[^{}]*\}[^{}]*\}', output, re.DOTALL):
        try:
            json_match = json.loads(match.group())
            break
        except json.JSONDecodeError:
            continue

    if not json_match or "ratios" not in json_match:
        # Try broader extraction
        for match in re.finditer(r'\{.*?\}', output, re.DOTALL):
            try:
                candidate = json.loads(match.group())
                if "ratios" in candidate:
                    json_match = candidate
                    break
            except (json.JSONDecodeError, RecursionError):
                continue

    if not json_match or "ratios" not in json_match:
        return 0.0

    output_ratios = json_match["ratios"]
    correct = 0
    total = 0

    for key, expected_val in expected_ratios.items():
        total += 1
        output_val = output_ratios.get(key)
        if expected_val is None:
            if output_val is None:
                correct += 1
        elif output_val is not None:
            correct += task_accuracy_numeric(float(output_val), float(expected_val), tolerance)

    return correct / total if total > 0 else 0.0


def reasoning_quality_shows_work(output: str) -> float:
    """Check if the output shows intermediate calculation steps."""
    # Look for mathematical expressions or step markers
    indicators = [
        r'step\s*\d',
        r'=\s*[\d,]+\.?\d*\s*/\s*[\d,]+\.?\d*',  # division shown
        r'[\d,]+\.?\d*\s*[/\*\+\-]\s*[\d,]+\.?\d*',  # any arithmetic
        r'profit\s*margin\s*=',
        r'current\s*ratio\s*=',
        r'STEP\s*\d',
    ]
    matches = sum(1 for pattern in indicators if re.search(pattern, output, re.IGNORECASE))
    return min(1.0, matches / 3)  # at least 3 indicators for full credit


def consistency_score(outputs: list[str]) -> float:
    """
    Measure consistency across multiple runs of the same input.
    Uses simple jaccard similarity of word sets across runs.
    Returns average pairwise similarity.
    """
    if len(outputs) < 2:
        return 1.0

    similarities = []
    for i in range(len(outputs)):
        for j in range(i + 1, len(outputs)):
            words_i = set(outputs[i].lower().split())
            words_j = set(outputs[j].lower().split())
            if not words_i and not words_j:
                similarities.append(1.0)
            elif not words_i or not words_j:
                similarities.append(0.0)
            else:
                jaccard = len(words_i & words_j) / len(words_i | words_j)
                similarities.append(jaccard)

    return sum(similarities) / len(similarities)


def compute_mhs(metric_scores: dict[str, float], weights: dict[str, float]) -> float:
    """
    Compute Migration Health Score as weighted average of metrics.

    Args:
        metric_scores: Dict of metric_name -> score (0.0 to 1.0)
        weights: Dict of metric_name -> weight

    Returns:
        Weighted average score (0.0 to 1.0)
    """
    total_weight = 0.0
    weighted_sum = 0.0
    for metric, weight in weights.items():
        if metric in metric_scores:
            weighted_sum += weight * metric_scores[metric]
            total_weight += weight
    return weighted_sum / total_weight if total_weight > 0 else 0.0


def compute_ppi(mhs_new: float, mhs_old: float) -> float:
    """Compute Prompt Portability Index. Returns percentage."""
    if mhs_old == 0:
        return 0.0
    return (mhs_new / mhs_old) * 100


def get_triage_zone(ppi: float) -> str:
    """Map PPI to triage zone."""
    if ppi >= 98:
        return "green"
    elif ppi >= 95:
        return "yellow"
    elif ppi >= 85:
        return "orange"
    elif ppi >= 70:
        return "red"
    else:
        return "critical"
