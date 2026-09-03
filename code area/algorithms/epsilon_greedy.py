"""Epsilon-Greedy algorithm.

Supports: value estimation (Q), exploration (random), exploitation
(greedy), the parameter epsilon, and optimistic initialization.
All parameters are passed in externally, never hard-coded.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np

from algorithms.base_algorithm import BaseAlgorithm


class EpsilonGreedy(BaseAlgorithm):
    name = "epsilon_greedy"

    def __init__(
        self,
        action_set: Iterable,
        epsilon: float = 0.1,
        optimistic_init: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__(action_set, rng=rng)
        self.epsilon = float(epsilon)
        self.counts = np.zeros(self.n_actions, dtype=float)
        self.values = np.full(self.n_actions, float(optimistic_init), dtype=float)

    def select_action(self, state: Dict, available_actions: Optional[Iterable] = None):
        actions = self._resolve_actions(available_actions)
        idx = [self.action_index[a] for a in actions]

        # Exploration: pick a random action with probability epsilon
        if self.rng.random() < self.epsilon:
            j = int(self.rng.integers(len(actions)))
            return actions[j]

        # Exploitation: pick the action with the highest estimated value
        # (ties broken randomly)
        values = np.array([self.values[i] for i in idx])
        return actions[self._argmax_random(values)]

    def update(self, observation: Dict) -> None:
        idx = self.action_index[observation["order_up_to"]]
        reward = float(observation["profit"])
        self.counts[idx] += 1
        # Incremental mean update
        self.values[idx] += (reward - self.values[idx]) / self.counts[idx]
