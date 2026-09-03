"""Algorithms module.

Registers all algorithms in one place. Adding a new algorithm only
requires registering it here — no changes to the core simulation.
"""
from algorithms.base_algorithm import BaseAlgorithm
from algorithms.epsilon_greedy import EpsilonGreedy
from algorithms.gradient import GradientBandit
from algorithms.ucb import UCB

ALGORITHM_REGISTRY = {
    UCB.name: UCB,
    EpsilonGreedy.name: EpsilonGreedy,
    GradientBandit.name: GradientBandit,
}


def create_algorithm(name: str, action_set, params: dict, seed: int = None) -> BaseAlgorithm:
    """Create an algorithm instance by name and parameters (reproducible)."""
    if name not in ALGORITHM_REGISTRY:
        raise KeyError(
            f"Unknown algorithm '{name}'. Available: {sorted(ALGORITHM_REGISTRY)}"
        )
    import numpy as np

    rng = np.random.default_rng(seed)
    return ALGORITHM_REGISTRY[name](action_set, rng=rng, **params)


__all__ = [
    "BaseAlgorithm",
    "UCB",
    "EpsilonGreedy",
    "GradientBandit",
    "ALGORITHM_REGISTRY",
    "create_algorithm",
]
