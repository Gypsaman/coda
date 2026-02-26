"""
CODA Optimizer Package.

Available optimizers:
  protegi   — ProTeGi textual gradient descent (Pryzant et al. 2023)
  ape       — Automatic Prompt Engineer, parallel candidate generation (Zhou et al. 2023)
  opro      — Optimization by PROmpting, trajectory-guided (Yang et al. 2024)
  evoprompt — Evolutionary prompt optimization (Guo et al. 2024)

Entry point:
  from optimizers.router import route_and_optimize
"""

from optimizers.router import route_and_optimize

__all__ = ["route_and_optimize"]
