"""自动参数优化模块。

流程：Define Search Space → Run Simulation → Calculate Total Profit
      → Compare Parameters → Select Best Parameters → Save Results

支持 Grid Search / Random Search，结构便于以后扩展 Bayesian Optimization：
新增方法只需继承 BaseSearch 并实现 _generate_candidates。

重要：每一次 simulation 都严格遵守 31-day online learning information
constraint（见 experiments/simulation.py），不会因为调参而看到未来 demand。
"""
from __future__ import annotations

import itertools
import json
import random
from abc import ABC, abstractmethod
from typing import Dict, List

import pandas as pd

from config import N_DAYS, TUNING_DIR, ensure_dirs
from experiments.compare_algorithms import run_single
from experiments.simulation import (
    build_env,
    records_to_dataframe,
    report_profit_vs_upper,
)


class BaseSearch(ABC):
    """参数搜索方法基类，便于扩展 Bayesian Optimization 等。"""

    name = "base"

    def __init__(self, param_grid: Dict[str, List], seed: int = None) -> None:
        self.param_grid = param_grid
        self.seed = seed

    @abstractmethod
    def generate_candidates(self, n_trials: int = None) -> List[Dict]:
        """生成待测试的参数组合列表。"""
        raise NotImplementedError


class GridSearch(BaseSearch):
    """网格搜索：遍历所有参数组合。"""

    name = "grid"

    def generate_candidates(self, n_trials: int = None) -> List[Dict]:
        keys = list(self.param_grid)
        combos = []
        for values in itertools.product(*(self.param_grid[k] for k in keys)):
            combos.append(dict(zip(keys, values)))
        return combos


class RandomSearch(BaseSearch):
    """随机搜索：从搜索空间中随机采样 n_trials 组参数。"""

    name = "random"

    def generate_candidates(self, n_trials: int = 20) -> List[Dict]:
        rng = random.Random(self.seed)
        keys = list(self.param_grid)
        # 保证每个候选至少覆盖随机维度
        n_trials = max(1, min(n_trials, 10000))
        return [
            {k: rng.choice(self.param_grid[k]) for k in keys}
            for _ in range(n_trials)
        ]


SEARCH_METHODS = {
    GridSearch.name: GridSearch,
    RandomSearch.name: RandomSearch,
}


def tune_parameters(
    algorithm_name: str,
    param_grid: Dict[str, List],
    demand_df: pd.DataFrame,
    method: str = "grid",
    n_trials: int = 20,
    seed: int = None,
) -> pd.DataFrame:
    """对指定算法进行自动参数优化，结果保存并返回（按 total profit 排序）。"""
    ensure_dirs()
    if method not in SEARCH_METHODS:
        raise KeyError(f"未知搜索方法 '{method}'，可选：{sorted(SEARCH_METHODS)}")

    search = SEARCH_METHODS[method](param_grid, seed=seed)
    candidates = search.generate_candidates(n_trials=n_trials)

    # 完全预知上限（q_t = demand），任何算法/参数都无法超过
    env = build_env(demand_df)
    upper = env.profit_upper_bound()
    print(f"\nProfit upper bound (q=demand, clairvoyant): {upper:,.2f}\n")

    rows = []
    for i, params in enumerate(candidates):
        # 每次运行都是独立、干净的 31-day online simulation
        records = run_single(algorithm_name, params=params, seed=seed,
                             demand_df=demand_df)
        df = records_to_dataframe(records)
        total_profit = float(df["profit"].sum())
        pct = report_profit_vs_upper(
            total_profit, env, label=f"{algorithm_name} {params}"
        )
        rows.append({
            "algorithm": algorithm_name,
            "search_method": method,
            "params": json.dumps(params, ensure_ascii=False, sort_keys=True),
            "total_profit": total_profit,
            "average_daily_profit": total_profit / N_DAYS,
            "pct_of_upper": pct,
        })

    result = pd.DataFrame(rows)
    result["rank"] = (
        result["total_profit"].rank(ascending=False, method="min").astype(int)
    )
    result = result.sort_values("rank").reset_index(drop=True)

    out_path = TUNING_DIR / f"parameter_tuning_{algorithm_name}.csv"
    result.to_csv(out_path, index=False)
    return result
