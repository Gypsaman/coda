"""
CODA Optimizer Router.
Reads the classification report (triage zone + primary failure category)
and dispatches to the appropriate optimizer backend.

Routing table (two-pass: zone default, then category override):

Zone defaults:
  green    → None (no optimization)
  yellow   → protegi (light gradient, 3-iter)
  orange   → protegi (full gradient, 10-iter)
  red      → opro    (trajectory-guided, deeper failures)
  critical → evoprompt (population search, ground-up redesign)

Category overrides:
  (critical, reasoning_quality) → ape
  (critical, format_compliance) → evoprompt
  (critical, tool_calling)      → evoprompt
  (red,      format_compliance) → protegi
"""

from typing import Callable

from scripts.llm_client import LLMClient

# Lazy imports — avoids circular dependency at module load time
_REGISTRY: dict | None = None


def _get_registry() -> dict:
    global _REGISTRY
    if _REGISTRY is None:
        from optimizers import protegi, ape, opro, evoprompt
        _REGISTRY = {
            "protegi": protegi.optimize,
            "ape": ape.optimize,
            "opro": opro.optimize,
            "evoprompt": evoprompt.optimize,
        }
    return _REGISTRY


# Zone → default optimizer name (None = no optimization needed)
ZONE_DEFAULTS: dict[str, str | None] = {
    "green": None,
    "yellow": "protegi",
    "orange": "protegi",
    "red": "opro",
    "critical": "evoprompt",
}

# (zone, primary_failure_category) → override optimizer name
CATEGORY_OVERRIDES: dict[tuple[str, str], str] = {
    ("critical", "reasoning_quality"): "ape",
    ("critical", "format_compliance"): "evoprompt",
    ("critical", "tool_calling"): "evoprompt",
    ("red", "format_compliance"): "protegi",
}


def _get_primary_category(classification_report: dict) -> str | None:
    """Extract the top-priority failure category from the classification report."""
    try:
        items = classification_report["summary"]["prioritized_action_items"]
        if items:
            return items[0]["category"]
    except (KeyError, IndexError, TypeError):
        pass
    return None


def route_and_optimize(
    classification_report: dict,
    prompt: str,
    test_cases: list[dict],
    eval_fn: Callable[[str], float],
    client: LLMClient,
    model_config: dict,
    optimizer_configs: dict,
) -> tuple[str, dict]:
    """
    Select and run the appropriate optimizer based on triage zone and failure category.

    Args:
        classification_report: full dict from classification_report.json
        prompt:                 current system prompt string
        test_cases:             list of test case dicts from the JSONL suite
        eval_fn:                callable prompt_text -> float [0,1], pre-bound to
                                quick_evaluate with the correct client/model
        client:                 LLMClient instance
        model_config:           new_model config dict (provider, model, temp, max_tokens)
        optimizer_configs:      per-optimizer hyperparameter dicts from thresholds.yaml,
                                with failure_cases and classification_summary already injected

    Returns:
        (best_prompt, log_dict) — log_dict includes 'optimizer' and 'routing' keys
    """
    zone = classification_report.get("triage_zone", "critical")
    primary_category = _get_primary_category(classification_report)

    # Two-pass routing
    optimizer_name = ZONE_DEFAULTS.get(zone, "protegi")
    if primary_category:
        override = CATEGORY_OVERRIDES.get((zone, primary_category))
        if override:
            optimizer_name = override

    print(f"\n  Router: zone={zone}, primary_failure={primary_category}")
    print(f"  Router: selected optimizer → {optimizer_name or 'none (green zone)'}")

    if optimizer_name is None:
        return prompt, {
            "optimizer": "none",
            "routing": {"zone": zone, "primary_category": primary_category},
            "original_score": None,
            "optimized_score": None,
            "improvement": 0.0,
            "iterations_run": 0,
            "log": [],
            "reason": "Green zone — no optimization needed",
        }

    registry = _get_registry()
    optimizer_fn = registry[optimizer_name]
    config = dict(optimizer_configs.get(optimizer_name, {}))

    best_prompt, log_dict = optimizer_fn(
        prompt, test_cases, eval_fn, client, model_config, config
    )

    log_dict["optimizer"] = optimizer_name
    log_dict["routing"] = {"zone": zone, "primary_category": primary_category}
    return best_prompt, log_dict
