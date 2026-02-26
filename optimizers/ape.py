"""
APE: Automatic Prompt Engineer.
Generates N candidate prompts in a single parallel batch, evaluates all,
and returns the best. Suited for critical-zone cases where the starting
prompt is too broken to serve as a gradient seed.

Reference: Zhou et al. 2023, "Large Language Models Are Human-Level Prompt
Engineers", ICLR 2023. arXiv:2211.01910
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from scripts.llm_client import LLMClient

APE_SYSTEM_PROMPT = (
    "You are a prompt engineer. Given a broken system prompt and a description of how it "
    "fails, write a complete replacement system prompt that fixes the failures.\n\n"
    "Rules:\n"
    "- Keep the same core task and domain as the original\n"
    "- Directly address the failure categories described\n"
    "- Return ONLY the new prompt text, no commentary\n\n"
    "This is candidate #{candidate_num} of {n_candidates} — explore a different angle "
    "than you might have taken otherwise."
)


def generate_candidate(
    client: LLMClient,
    original_prompt: str,
    classification_summary: dict,
    candidate_num: int,
    n_candidates: int,
    model_config: dict,
) -> str:
    """Generate one candidate replacement prompt."""
    response = client.complete(
        provider="openai",
        model="gpt-4o",
        system_prompt=APE_SYSTEM_PROMPT.format(
            candidate_num=candidate_num, n_candidates=n_candidates
        ),
        user_message=(
            f"## Original Prompt (broken)\n{original_prompt}\n\n"
            f"## Failure Analysis\n{json.dumps(classification_summary, indent=2)}"
        ),
        temperature=0.9,
        max_tokens=2000,
    )
    return response["text"].strip()


def optimize(
    prompt: str,
    test_cases: list[dict],
    eval_fn: Callable[[str], float],
    client: LLMClient,
    model_config: dict,
    config: dict,
) -> tuple[str, dict]:
    """
    APE optimization: generate N candidates, evaluate in parallel, pick best.

    config keys:
        n_candidates (int, default 10)
        max_parallel_evals (int, default 5)
        classification_summary (dict) — injected at runtime
    """
    n = config.get("n_candidates", 10)
    max_workers = config.get("max_parallel_evals", 5)
    classification_summary = config.get("classification_summary", {})

    baseline_score = eval_fn(prompt)
    print(f"\n  [APE] Generating {n} candidate prompts...")

    candidates = [
        generate_candidate(client, prompt, classification_summary, i + 1, n, model_config)
        for i in range(n)
    ]

    print(f"  [APE] Evaluating {n} candidates in parallel (workers={max_workers})...")
    scores: dict[int, float] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(eval_fn, c): i for i, c in enumerate(candidates)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            scores[idx] = future.result()
            print(f"    Candidate {idx + 1}: {scores[idx]:.4f}")

    best_idx = max(scores, key=scores.__getitem__)
    best_score = scores[best_idx]
    best_prompt = candidates[best_idx] if best_score > baseline_score else prompt
    final_score = max(best_score, baseline_score)

    print(f"  [APE] Best candidate: #{best_idx + 1} score={best_score:.4f} (baseline={baseline_score:.4f})")

    return best_prompt, {
        "original_score": round(baseline_score, 4),
        "optimized_score": round(final_score, 4),
        "improvement": round(final_score - baseline_score, 4),
        "iterations_run": 1,
        "log": [{"candidate": i + 1, "score": round(scores[i], 4)} for i in range(n)],
    }
