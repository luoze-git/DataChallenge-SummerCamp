"""算法统一 Interface。

所有算法必须继承 BaseAlgorithm，主流程（main / comparison / tuning）
只依赖该接口，不关心算法内部实现：

    action = algorithm.select_action(state, available_actions)
    algorithm.update(observation)

新增算法时只需在 algorithms/ 中增加一个文件并在
algorithms/__init__.py 的 ALGORITHM_REGISTRY 中注册。
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np


class BaseAlgorithm:
    """所有在线学习算法的统一接口。"""

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
    # 必须实现的接口
    # ------------------------------------------------------------------
    def select_action(self, state: Dict, available_actions: Optional[Iterable] = None):
        """根据决策前可获得的 state 选择一个 order-up-to level。

        注意：state 中不允许包含当天/未来的 demand（data leakage 禁止）。
        """
        raise NotImplementedError

    def update(self, observation: Dict) -> None:
        """根据决策后 Environment 返回的 observation 更新内部状态。"""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 通用工具
    # ------------------------------------------------------------------
    def _resolve_actions(self, available_actions: Optional[Iterable]) -> List:
        actions = list(available_actions) if available_actions is not None else self.actions
        if not actions:
            raise ValueError("available_actions 不能为空")
        return actions

    def _argmax_random(self, values) -> int:
        """argmax，平局时随机选择（保证可复现）。"""
        values = np.asarray(values, dtype=float)
        best = np.flatnonzero(values == values.max())
        return int(self.rng.choice(best))

