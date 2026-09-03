"""Gradient Algorithm（gradient bandit / policy-gradient 风格）。

支持：preference H、probability π（softmax）、learning rate α、
reward baseline。参数均从外部传入，不在算法内部 hard-code。
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np

from algorithms.base_algorithm import BaseAlgorithm


class GradientBandit(BaseAlgorithm):
    name = "gradient"

    def __init__(
        self,
        action_set: Iterable,
        alpha: float = 0.1,
        use_baseline: bool = True,
        reward_scale: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__(action_set, rng=rng)
        self.alpha = float(alpha)
        self.use_baseline = bool(use_baseline)
        self.reward_scale = float(reward_scale) if reward_scale else 1.0
        self.preferences = np.zeros(self.n_actions, dtype=float)
        self._baseline_sum = 0.0
        self._baseline_count = 0

    # ------------------------------------------------------------------
    def _softmax(self, idx) -> np.ndarray:
        h = np.array([self.preferences[i] for i in idx])
        h = h - h.max()  # 数值稳定
        e = np.exp(h)
        return e / e.sum()

    @property
    def baseline(self) -> float:
        if self._baseline_count == 0:
            return 0.0
        return self._baseline_sum / self._baseline_count

    def select_action(self, state: Dict, available_actions: Optional[Iterable] = None):
        actions = self._resolve_actions(available_actions)
        idx = [self.action_index[a] for a in actions]
        pi = self._softmax(idx)
        j = int(self.rng.choice(len(actions), p=pi))
        return actions[j]

    def update(self, observation: Dict) -> None:
        idx = self.action_index[observation["order_up_to"]]
        reward = float(observation["profit"]) / self.reward_scale

        # reward baseline（所有已观察 reward 的均值）
        baseline = 0.0
        if self.use_baseline:
            self._baseline_sum += reward
            self._baseline_count += 1
            baseline = self.baseline

        # 只更新本日可选动作的偏好（保持动作集合一致性）
        actions = self.actions
        pi = self._softmax(range(self.n_actions))
        indicator = np.zeros(self.n_actions)
        indicator[idx] = 1.0
        # ∂log π(a_t)/∂H : onehot - π
        grad = indicator - pi
        self.preferences += self.alpha * (reward - baseline) * grad
