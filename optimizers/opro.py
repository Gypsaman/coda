"""
OPRO: Optimization by PROmpting.
Maintains a rolling trajectory of (prompt, score) pairs sorted by score.
Each iteration the LLM studies the trajectory and generates a better prompt,
guided by what has and hasn't worked.

Reference: Yang et al. 2024, "Large Language Models as Optimizers",
ICLR 2024. arXiv:2309.03409
"""

import json
from typing import Callable

from scripts.llm_client import LLMClient

OPRO_SYSTEM_PROMPT = (
    "You are an optimizer. Below is a trajectory of system prompts and their quality "
    "scores (0=worst, 1=best). Study how each change affected the score, then generate "
    "a new system prompt that achieves a higher score than the best one seen so far.\n\n"
    "Return ONLY the new prompt text. No commentary."
)


def optimize(
    prompt: str,
    test_cases: list[dict],
    eval_fn: Callable[[str], float],
    client: LLMClient,
    model_config: dict,
    config: dict,
) -> tuple[str, dict]:
    """
    OPRO trajectory-guided optimization loop.

    config keys:
        max_iterations (int, default 15)
        patience (int, default 4)
        trajectory_window (int, default 5) — how many past entries the LLM sees
        classification_summary (dict) — injected at runtime
    """
    max_iterations = config.get("max_iterations", 15)
    patience = config.get("patience", 4)
    window = config.get("trajectory_window", 5)
    classification_summary = config.get("classification_summary", {})

    baseline_score = eval_fn(prompt)
    trajectory = [{"prompt": prompt, "score": baseline_score}]
    best_prompt, best_score = prompt, baseline_score
    no_improvement = 0
    log = []

    for iteration in range(1, max_iterations + 1):
        print(f"\n  [OPRO] Iteration {iteration}/{max_iterations}")

        # Pass last `window` entries sorted ascending so the LLM sees the improvement trend
        window_entries = sorted(trajectory[-window:], key=lambda x: x["score"])
        traj_text = "\n\n".join(
            f"[Score: {e['score']:.4f}]\n{e['prompt']}" for e in window_entries
        )

        response = client.complete(
            provider="openai",
            model="gpt-4o",
            system_prompt=OPRO_SYSTEM_PROMPT,
            user_message=(
                f"## Failure Context\n{json.dumps(classification_summary, indent=2)}\n\n"
                f"## Prompt Trajectory (ascending by score)\n{traj_text}"
            ),
            temperature=0.4,
            max_tokens=2000,
        )
        candidate = response["text"].strip()
        candidate_score = eval_fn(candidate)
        trajectory.append({"prompt": candidate, "score": candidate_score})

        accepted = candidate_score > best_score
        print(f"    Score: {candidate_score:.4f} (best: {best_score:.4f}) — {'ACCEPTED' if accepted else 'REJECTED'}")

        log.append({
            "iteration": iteration,
            "candidate_score": round(candidate_score, 4),
            "best_score": round(best_score, 4),
            "trajectory_size": len(trajectory),
            "accepted": accepted,
        })

        if accepted:
            best_prompt, best_score, no_improvement = candidate, candidate_score, 0
        else:
            no_improvement += 1

        if no_improvement >= patience:
            print(f"    No improvement for {patience} iterations. Stopping early.")
            break

    return best_prompt, {
        "original_score": round(baseline_score, 4),
        "optimized_score": round(best_score, 4),
        "improvement": round(best_score - baseline_score, 4),
        "iterations_run": len(log),
        "log": log,
    }
