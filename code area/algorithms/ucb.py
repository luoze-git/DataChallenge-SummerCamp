"""UCB (Upper Confidence Bound) algorithm.

Supports: action value estimation (Q), action counts, the exploration
term, and the parameter c. All parameters are passed in externally
(config / tuning), never hard-coded inside the algorithm.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, Optional

import numpy as np

from algorithms.base_algorithm import BaseAlgorithm


class UCB(BaseAlgorithm):
    name = "ucb"

    def __init__(
        self,
        action_set: Iterable,
        c: float = 1.0,
        optimistic_init: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__(action_set, rng=rng)
        self.c = float(c)
        self.counts = np.zeros(self.n_actions, dtype=float)
        self.values = np.full(self.n_actions, float(optimistic_init), dtype=float)

    def select_action(self, state: Dict, available_actions: Optional[Iterable] = None):
        actions = self._resolve_actions(available_actions)
        idx = [self.action_index[a] for a in actions]

        # Explore never-played actions first (UCB score = +inf; ties broken randomly)
        unplayed = [i for i in idx if self.counts[i] == 0]
        if unplayed:
            return actions[self.rng.integers(len(unplayed))]

        t = self.counts.sum() + 1.0
        ucb = np.array(
            [self.values[i] + self.c * math.sqrt(math.log(t) / self.counts[i]) for i in idx]
        )
        best = self._argmax_random(ucb)
        return actions[best]

    def update(self, observation: Dict) -> None:
        idx = self.action_index[observation["order_up_to"]]
        reward = float(observation["profit"])
        self.counts[idx] += 1
        # Incremental mean update
        self.values[idx] += (reward - self.values[idx]) / self.counts[idx]
