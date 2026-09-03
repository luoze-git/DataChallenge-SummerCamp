"""Unified algorithm interface.

All algorithms must inherit from BaseAlgorithm. The main flow
(main / comparison / tuning) depends only on this interface and knows
nothing about the internal implementation:

    action = algorithm.select_action(state, available_actions)
    algorithm.update(observation)

To add a new algorithm, just add a file in algorithms/ and register it
in ALGORITHM_REGISTRY (algorithms/__init__.py).
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np


class BaseAlgorithm:
    """Unified interface for all online learning algorithms."""

    name = "base"

    def __init__(
        self,
        action_set: Iterable,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.actions: List = list(action_set)
        self.n_actions = len(self.actions)
        self.action_index: Dict = {a: i for i, a in enumerate(self.actions)}
        self.rng = rng if rng is not None else np.random.default_rng()

    # ------------------------------------------------------------------
    # Interface to implement
    # ------------------------------------------------------------------
    def select_action(self, state: Dict, available_actions: Optional[Iterable] = None):
        """Select an order-up-to level based on the pre-decision state.

        Note: the state must not contain today's/future demand
        (data leakage is forbidden).
        """
        raise NotImplementedError

    def update(self, observation: Dict) -> None:
        """Update internal state from the post-decision observation
        returned by the Environment.

        Actual demand is intentionally absent: after a stockout, only sales
        (a lower bound on demand) are observable.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------
    def _resolve_actions(self, available_actions: Optional[Iterable]) -> List:
        actions = list(available_actions) if available_actions is not None else self.actions
        if not actions:
            raise ValueError("available_actions must not be empty")
        return actions

    def _argmax_random(self, values) -> int:
        """Argmax with random tie-breaking (reproducible)."""
        values = np.asarray(values, dtype=float)
        best = np.flatnonzero(values == values.max())
        return int(self.rng.choice(best))
