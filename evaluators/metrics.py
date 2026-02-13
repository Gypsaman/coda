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


def format_compliance_single_paragraph(output: str) -> float:
    """Check output is a single paragraph (no line breaks within the body)."""
    # Strip leading/trailing whitespace, then check for internal line breaks
    stripped = output.strip()
    # Allow one trailing line break but not internal paragraph breaks
    lines = [l for l in stripped.split("\n") if l.strip()]
    return 1.0 if len(lines) == 1 else 0.0


def format_compliance_no_bullets(output: str) -> float:
    """Check output doesn't use bullet points or numbered lists."""
    bullet_patterns = [
        r'^\s*[-*]\s',       # markdown bullets
        r'^\s*\d+\.\s',      # numbered lists
        r'^\s*\d+\)\s',      # numbered with parens
    ]
    for line in output.split("\n"):
        for pattern in bullet_patterns:
            if re.match(pattern, line):
                return 0.0
    return 1.0


def format_compliance_no_exclamation(output: str) -> float:
    """Check output contains no exclamation marks."""
    return 0.0 if "!" in output else 1.0


def format_compliance_greeting(output: str, expected_greeting: str) -> float:
    """Check output starts with the exact expected greeting format."""
    return 1.0 if output.strip().startswith(expected_greeting) else 0.0


def format_compliance_max_apologies(output: str, max_count: int = 1) -> float:
    """Check output doesn't apologize more than max_count times."""
    apology_patterns = [
        r'\bsorry\b', r'\bapologize\b', r'\bapologies\b', r'\bapology\b',
        r'\bregret\b', r'\bregrettable\b',
    ]
    count = 0
    output_lower = output.lower()
    for pattern in apology_patterns:
        count += len(re.findall(pattern, output_lower))
    return 1.0 if count <= max_count else 0.0


def format_compliance_step_headers(output: str) -> float:
    """Check that output contains all 5 required step headers."""
    required = [
        r'##\s*STEP\s*1',
        r'##\s*STEP\s*2',
        r'##\s*STEP\s*3',
        r'##\s*STEP\s*4',
        r'##\s*STEP\s*5',
    ]
    found = sum(1 for pat in required if re.search(pat, output, re.IGNORECASE))
    return found / len(required)


def format_compliance_inline_arithmetic(output: str) -> float:
    """Check that ratio calculations show inline arithmetic (value / value = result)."""
    # Look for patterns like "= $X / $Y = 0.XXXX" or "X / Y = X.XXXX"
    arithmetic_pattern = r'=\s*[\$]?[\d,]+[\.\d]*\s*/\s*[\$]?[\d,]+[\.\d]*\s*=\s*[\-]?[\d]+\.[\d]+'
    matches = re.findall(arithmetic_pattern, output)
    # Expect at least 4 ratios to show work (profit margin, current ratio, D/E, ROE)
    return min(1.0, len(matches) / 4)


def format_compliance_decimal_places(output: str, expected_places: int = 4) -> float:
    """Check that ratio values in the output use exactly the expected decimal places."""
    # Find ratio-like decimal numbers after = signs
    ratio_pattern = r'=\s*([\-]?\d+\.\d+)\s*$'
    matches = re.findall(ratio_pattern, output, re.MULTILINE)
    if not matches:
        # Try finding them inline
        ratio_pattern2 = r'=\s*([\-]?\d+\.\d+)'
        matches = re.findall(ratio_pattern2, output)
    if not matches:
        return 0.0
    correct = sum(1 for m in matches if len(m.split('.')[-1]) == expected_places)
    return correct / len(matches) if matches else 0.0


def format_compliance_raw_json(output: str) -> float:
    """Check that JSON output is raw (no markdown code fences around it)."""
    # Find the JSON in the output
    if '```' in output:
        # Check if there are code fences around a JSON block
        if re.search(r'```(?:json)?\s*\{', output):
            return 0.0  # Markdown-wrapped JSON
    # Try to find raw JSON
    json_match = re.search(r'(?<!\`)\{[^`]*"company"[^`]*\}', output, re.DOTALL)
    return 1.0 if json_match else 0.0


def format_compliance_exactly_n_risks(output: str, n: int = 2) -> float:
    """Check that exactly n risks are listed."""
    risk_pattern = r'RISK\s*\d+'
    matches = re.findall(risk_pattern, output, re.IGNORECASE)
    return 1.0 if len(matches) == n else 0.0


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


def cost_estimation_accuracy(output: str, expected_costs: dict[str, float], tolerance: float = 0.01) -> float:
    """
    Check if cost estimation values in the output JSON match expected values.

    Extracts the JSON summary from the output and compares numeric fields.
    """
    json_match = None
    for match in re.finditer(r'\{[^{}]+\}', output, re.DOTALL):
        try:
            candidate = json.loads(match.group())
            if "total_cost" in candidate:
                json_match = candidate
                break
        except json.JSONDecodeError:
            continue

    if not json_match:
        return 0.0

    correct = 0
    total = 0
    for key, expected_val in expected_costs.items():
        total += 1
        output_val = json_match.get(key)
        if output_val is not None and expected_val is not None:
            correct += task_accuracy_numeric(float(output_val), float(expected_val), tolerance)

    return correct / total if total > 0 else 0.0


def cost_estimation_shows_work(output: str) -> float:
    """Check if cost estimation output shows intermediate calculation steps."""
    indicators = [
        r'STEP\s*\d',
        r'\$[\d,]+\s*[×x\*]\s*\d+',           # $rate × hours
        r'[\d,]+\s*[×x\*]\s*[\d,]+\s*[×x\*=]', # rate × hours × weeks
        r'(?i)base\s*cost\s*=',
        r'(?i)after\s*complexity\s*=',
        r'(?i)after\s*urgency\s*=',
        r'(?i)infrastructure\s*total\s*=',
        r'(?i)discount\s*=',
        r'(?i)total\s*project\s*cost\s*=',
        r'(?i)subtotal\s*=',
    ]
    matches = sum(1 for pattern in indicators if re.search(pattern, output))
    return min(1.0, matches / 5)


def cost_estimation_inline_arithmetic(output: str) -> float:
    """Check that cost calculations show inline multiplication arithmetic."""
    # Look for patterns like "$85 × 20 × 4 = $6,800" or "85 * 20 * 4 = 6800"
    arithmetic_pattern = r'[\$]?[\d,]+[\.\d]*\s*[×x\*]\s*[\d,]+[\.\d]*\s*[×x\*=]'
    matches = re.findall(arithmetic_pattern, output)
    # Expect at least 3 multiplication calculations (team members + adjustments)
    return min(1.0, len(matches) / 3)


def cost_estimation_raw_json(output: str) -> float:
    """Check that JSON output is raw (no markdown code fences) and contains project key."""
    if '```' in output:
        if re.search(r'```(?:json)?\s*\{', output):
            return 0.0
    json_match = re.search(r'(?<!\`)\{[^`]*"project"[^`]*\}', output, re.DOTALL)
    return 1.0 if json_match else 0.0


def cost_estimation_decimal_places(output: str, expected_places: int = 2) -> float:
    """Check that monetary values in JSON use exactly 2 decimal places."""
    # Extract the JSON portion
    json_match = None
    for match in re.finditer(r'\{[^{}]+\}', output, re.DOTALL):
        try:
            candidate = json.loads(match.group())
            if "total_cost" in candidate:
                json_match = match.group()
                break
        except json.JSONDecodeError:
            continue

    if not json_match:
        return 0.0

    # Find numeric values in the JSON that should have 2 decimal places
    # (cost fields, not multipliers or percentages)
    cost_fields = ["base_cost", "infrastructure_cost", "discount_amount", "total_cost"]
    money_pattern = r'"(?:' + '|'.join(cost_fields) + r')"\s*:\s*(\d+\.?\d*)'
    matches = re.findall(money_pattern, json_match)
    if not matches:
        return 0.0

    correct = 0
    for m in matches:
        if '.' in m and len(m.split('.')[-1]) == expected_places:
            correct += 1
        elif '.' not in m:
            # Integer like 1700 is acceptable as 1700 (no .00 required in JSON)
            correct += 1

    return correct / len(matches) if matches else 0.0


def report_format_header(output: str) -> float:
    """Check that output starts with 'INCIDENT REPORT'."""
    return 1.0 if output.strip().startswith("INCIDENT REPORT") else 0.0


def report_format_metadata(output: str) -> float:
    """Check that output contains all 5 required metadata fields."""
    required = [
        r'(?i)incident\s*id\s*:',
        r'(?i)severity\s*:',
        r'(?i)status\s*:',
        r'(?i)reported\s*by\s*:',
        r'(?i)duration\s*:',
    ]
    found = sum(1 for pat in required if re.search(pat, output))
    return found / len(required)


def report_format_sections(output: str) -> float:
    """Check that output contains all 5 required section headers."""
    required = [
        r'##\s*SUMMARY',
        r'##\s*IMPACT',
        r'##\s*ROOT\s*CAUSE',
        r'##\s*RESOLUTION',
        r'##\s*FOLLOW[\s-]*UP',
    ]
    found = sum(1 for pat in required if re.search(pat, output))
    return found / len(required)


def report_format_closing(output: str) -> float:
    """Check that output ends with the required closing line."""
    return 1.0 if "--- End of Report ---" in output else 0.0


def report_severity_valid(output: str) -> float:
    """Check that severity field is exactly P1, P2, P3, or P4."""
    match = re.search(r'(?i)severity\s*:\s*(P[1-4])\b', output)
    return 1.0 if match else 0.0


def report_status_valid(output: str) -> float:
    """Check that status field is one of the allowed values."""
    valid = ["RESOLVED", "MITIGATED", "ONGOING", "INVESTIGATING"]
    match = re.search(r'(?i)status\s*:\s*(\w+)', output)
    if match and match.group(1).upper() in valid:
        return 1.0
    return 0.0


def report_severity_accuracy(output: str, expected_severity: str) -> float:
    """Check that severity classification matches expected value."""
    match = re.search(r'(?i)severity\s*:\s*(P[1-4])\b', output)
    if match and match.group(1).upper() == expected_severity.upper():
        return 1.0
    return 0.0


def report_status_accuracy(output: str, expected_status: str) -> float:
    """Check that status matches expected value."""
    match = re.search(r'(?i)status\s*:\s*(\w+)', output)
    if match and match.group(1).upper() == expected_status.upper():
        return 1.0
    return 0.0


def report_reporter_accuracy(output: str, expected_reporter: str) -> float:
    """Check that reporter name matches expected value."""
    match = re.search(r'(?i)reported\s*by\s*:\s*(.+)', output)
    if match and expected_reporter.lower() in match.group(1).lower():
        return 1.0
    return 0.0


def report_section_word_limits(output: str, limits: dict[str, int]) -> float:
    """Check that each section respects its word limit."""
    section_pattern = r'##\s*(SUMMARY|IMPACT|ROOT\s*CAUSE|RESOLUTION|FOLLOW[\s-]*UP)\s*\n(.*?)(?=##|\-\-\-|$)'
    matches = re.findall(section_pattern, output, re.DOTALL | re.IGNORECASE)
    if not matches:
        return 0.0
    compliant = 0
    checked = 0
    for header, content in matches:
        normalized = re.sub(r'\s+', ' ', header.strip()).upper()
        if "FOLLOW" in normalized:
            normalized = "FOLLOW-UP"
        if "ROOT" in normalized:
            normalized = "ROOT CAUSE"
        limit = limits.get(normalized)
        if limit is not None:
            checked += 1
            wc = len(content.strip().split())
            if wc <= limit:
                compliant += 1
    return compliant / checked if checked > 0 else 0.0


def report_no_speculation(output: str) -> float:
    """Check that output does not use speculative language."""
    speculation = [
        r'\bmight\b', r'\bperhaps\b', r'\bpossibly\b', r'\bprobably\b',
        r'\bmaybe\b', r'\bi think\b', r'\bi believe\b',
    ]
    output_lower = output.lower()
    for pat in speculation:
        if re.search(pat, output_lower):
            return 0.0
    return 1.0


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
