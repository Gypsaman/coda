"""
EvoPrompt: Evolutionary prompt optimization.
Maintains a population of prompts; each generation applies LLM-based
mutation and crossover operators, then keeps the top-k survivors (elitist).

Reference: Guo et al. 2024, "Connecting Large Language Models with Evolutionary
Algorithms Yields Powerful Prompt Optimizers", ICLR 2024. arXiv:2309.08532
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from scripts.llm_client import LLMClient
from optimizers.ape import generate_candidate  # reused for population seeding

MUTATE_SYSTEM_PROMPT = (
    "You are a prompt mutation operator. Make ONE targeted improvement to the given "
    "system prompt to address the described failure. Keep all other parts intact.\n\n"
    "Return ONLY the complete mutated prompt. No commentary."
)

CROSSOVER_SYSTEM_PROMPT = (
    "You are a prompt crossover operator. Combine two parent prompts into a child that "
    "inherits the best elements of each. "
    "Parent A scored {score_a:.4f}; Parent B scored {score_b:.4f}. "
    "Weight your selection toward the higher-scoring parent's structure, but incorporate "
    "effective elements from the lower-scoring one.\n\n"
    "Return ONLY the complete child prompt. No commentary."
)


def _mutate(client: LLMClient, prompt: str, failure_hint: str) -> str:
    response = client.complete(
        provider="openai",
        model="gpt-4o",
        system_prompt=MUTATE_SYSTEM_PROMPT,
        user_message=f"## Prompt to Mutate\n{prompt}\n\n## Failure to Fix\n{failure_hint}",
        temperature=0.7,
        max_tokens=2000,
    )
    return response["text"].strip()


def _crossover(client: LLMClient, prompt_a: str, score_a: float, prompt_b: str, score_b: float) -> str:
    response = client.complete(
        provider="openai",
        model="gpt-4o",
        system_prompt=CROSSOVER_SYSTEM_PROMPT.format(score_a=score_a, score_b=score_b),
        user_message=f"## Parent A\n{prompt_a}\n\n## Parent B\n{prompt_b}",
        temperature=0.5,
        max_tokens=2000,
    )
    return response["text"].strip()


def _eval_population(prompts: list[str], eval_fn: Callable[[str], float], max_workers: int) -> list[float]:
    scores = [0.0] * len(prompts)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(eval_fn, p): i for i, p in enumerate(prompts)}
        for future in as_completed(futures):
            scores[futures[future]] = future.result()
    return scores


def optimize(
    prompt: str,
    test_cases: list[dict],
    eval_fn: Callable[[str], float],
    client: LLMClient,
    model_config: dict,
    config: dict,
) -> tuple[str, dict]:
    """
    EvoPrompt: population-based evolutionary optimization.

    config keys:
        population_size (int, default 6)
        n_generations (int, default 4)
        max_parallel_evals (int, default 5)
        classification_summary (dict) — injected at runtime
        failure_cases (list[dict]) — injected at runtime, used for mutation hints
    """
    k = config.get("population_size", 6)
    n_gen = config.get("n_generations", 4)
    max_workers = config.get("max_parallel_evals", 5)
    classification_summary = config.get("classification_summary", {})
    failure_cases = config.get("failure_cases", [])

    # Build failure hint: summary + up to 3 example cases
    failure_hint = json.dumps(classification_summary, indent=2)
    if failure_cases:
        examples = "\n---\n".join(
            f"Input: {fc['input']}\nFailed metrics: {', '.join(fc.get('failed_metrics', []))}"
            for fc in failure_cases[:3]
        )
        failure_hint += f"\n\nExample failures:\n{examples}"

    # Seed population: original + (k-1) APE-style diverse candidates
    print(f"\n  [EvoPrompt] Seeding population (size={k})...")
    population = [prompt] + [
        generate_candidate(client, prompt, classification_summary, i, k, model_config)
        for i in range(1, k)
    ]

    print(f"  [EvoPrompt] Evaluating initial population...")
    scores = _eval_population(population, eval_fn, max_workers)
    baseline_score = scores[0]
    print(f"    Initial best: {max(scores):.4f}, mean: {sum(scores)/len(scores):.4f}")

    gen_log = []

    for gen in range(1, n_gen + 1):
        print(f"\n  [EvoPrompt] Generation {gen}/{n_gen}")

        # Mutation: one offspring per individual
        print(f"    Mutating {k} individuals...")
        offspring = [_mutate(client, ind, failure_hint) for ind in population]

        # Crossover: pair adjacent individuals by descending rank
        ranked = sorted(zip(population, scores), key=lambda x: x[1], reverse=True)
        print(f"    Crossing over {len(ranked) // 2} pairs...")
        for i in range(0, len(ranked) - 1, 2):
            (pa, sa), (pb, sb) = ranked[i], ranked[i + 1]
            offspring.append(_crossover(client, pa, sa, pb, sb))

        # Evaluate all offspring in parallel
        print(f"    Evaluating {len(offspring)} offspring...")
        offspring_scores = _eval_population(offspring, eval_fn, max_workers)

        # Elitist selection: combine parent + offspring pool, keep top-k
        combined = sorted(
            zip(population + offspring, scores + offspring_scores),
            key=lambda x: x[1],
            reverse=True,
        )
        population = [x[0] for x in combined[:k]]
        scores = [x[1] for x in combined[:k]]

        gen_best = scores[0]
        gen_mean = sum(scores) / len(scores)
        print(f"    Gen {gen} best: {gen_best:.4f}, mean: {gen_mean:.4f}")

        gen_log.append({
            "generation": gen,
            "best_score": round(gen_best, 4),
            "mean_score": round(gen_mean, 4),
            "population_size": len(population),
        })

    best_prompt = population[0]
    best_score = scores[0]

    return best_prompt, {
        "original_score": round(baseline_score, 4),
        "optimized_score": round(best_score, 4),
        "improvement": round(best_score - baseline_score, 4),
        "iterations_run": n_gen,
        "log": gen_log,
    }
