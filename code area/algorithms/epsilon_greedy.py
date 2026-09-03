"""ε-Greedy 算法。

支持：value estimation (Q)、exploration（随机探索）、
exploitation（贪心）、参数 epsilon、乐观初始化。
参数均从外部传入，不在算法内部 hard-code。
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

        # exploration：以 epsilon 概率随机探索
        if self.rng.random() < self.epsilon:
            j = int(self.rng.integers(len(actions)))
            return actions[j]

        # exploitation：选择当前估计值最高的动作（平局随机打破）
        values = np.array([self.values[i] for i in idx])
        return actions[self._argmax_random(values)]

    def update(self, observation: Dict) -> None:
        idx = self.action_index[observation["order_up_to"]]
        reward = float(observation["profit"])
        self.counts[idx] += 1
        # 增量式均值更新
        self.values[idx] += (reward - self.values[idx]) / self.counts[idx]
