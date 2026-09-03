"""算法模块。

统一注册所有算法，新增算法时只需在此注册，无需修改核心 simulation。
"""
from algorithms.base_algorithm import BaseAlgorithm
from algorithms.epsilon_greedy import EpsilonGreedy
from algorithms.ewf import EWF
from algorithms.gradient import GradientBandit
from algorithms.ucb import UCB

ALGORITHM_REGISTRY = {
    UCB.name: UCB,
    EpsilonGreedy.name: EpsilonGreedy,
    GradientBandit.name: GradientBandit,
    EWF.name: EWF,
}


def create_algorithm(name: str, action_set, params: dict, seed: int = None) -> BaseAlgorithm:
    """根据名称和参数创建算法实例（可复现）。"""
    if name not in ALGORITHM_REGISTRY:
        raise KeyError(
            f"未知算法 '{name}'，可选：{sorted(ALGORITHM_REGISTRY)}"
        )
    import numpy as np

    rng = np.random.default_rng(seed)
    return ALGORITHM_REGISTRY[name](action_set, rng=rng, **params)


__all__ = [
    "BaseAlgorithm",
    "UCB",
    "EpsilonGreedy",
    "GradientBandit",
    "EWF",
    "ALGORITHM_REGISTRY",
    "create_algorithm",
]
