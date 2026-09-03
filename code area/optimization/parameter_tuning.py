"""自动参数优化模块。

流程：Define Search Space → Run Simulation → Calculate Total Profit
      → Compare Parameters → Select Best Parameters → Save Results

支持 Grid Search / Random Search，结构便于以后扩展 Bayesian Optimization：
新增方法只需继承 BaseSearch 并实现 _generate_candidates。

重要：每一次 simulation 都严格遵守 31-day online learning information
constraint（见 experiments/simulation.py），不会因为调参而看到未来 demand。
"""
from __future__ import annotations

import difflib
import itertools
import json
import random
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List

import pandas as pd

from config import N_DAYS, TUNING_DIR, ensure_dirs
from experiments.compare_algorithms import run_single
from experiments.simulation import (
    build_env,
    records_to_dataframe,
    report_profit_vs_upper,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.py"


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


def apply_best_params(
    algorithm_name: str,
    best_params: Dict,
    config_path = None,
) -> None:
    """把网格/随机搜索到的最佳参数回写到 config.py 的参数库 ALGORITHM_PARAMS。

    - 只修改指定算法的参数块，其余内容不动；
    - 已存在参数的行内注释会被保留；
    - 写入前在控制台打印 unified diff；
    - 回写成功后同步更新当前进程内的 config.ALGORITHM_PARAMS。

    用法（通常由 experiments/run_tuning.py 的 --apply-best 开关调用）：
        apply_best_params("ucb", {"c": 5000.0, "optimistic_init": 0.0})
    """
    import config as config_module

    config_path = Path(config_path) if config_path else CONFIG_PATH
    text = config_path.read_text(encoding="utf-8")

    # 定位该算法在 ALGORITHM_PARAMS 中的参数块，例如：
    #     "ucb": {
    #         "c": 2000.0,           # exploration 系数
    #         "optimistic_init": 0.0
    #     },
    pattern = re.compile(
        r'(?P<head>    "%s": \{\n)(?P<body>.*?)(?P<tail>\n    \},)'
        % re.escape(algorithm_name),
        re.DOTALL,
    )
    match = pattern.search(text)
    if match is None:
        raise KeyError(f"config.py 中找不到算法 '{algorithm_name}' 的参数块")

    # 解析旧参数块，记录每个 key 的行内注释
    comments: Dict[str, str] = {}
    for line in match.group("body").splitlines():
        m = re.match(r'\s*"(\w+)":\s*[^#]+?(#.+)$', line)
        if m:
            comments[m.group(1)] = m.group(2).rstrip()

    # 重新生成参数块（与原文件风格一致：除最后一行外均带逗号）
    new_lines = []
    items = list(best_params.items())
    for i, (key, value) in enumerate(items):
        comma = "," if i < len(items) - 1 else ""
        comment = f"  {comments[key]}" if key in comments else ""
        new_lines.append(f'        "{key}": {value!r}{comma}{comment}')

    old_block = match.group(0)
    new_block = match.group("head") + "\n".join(new_lines) + match.group("tail")

    # 打印 diff 供确认
    diff = difflib.unified_diff(
        old_block.splitlines(),
        new_block.splitlines(),
        fromfile=f"config.py (原)",
        tofile=f"config.py (新)",
        lineterm="",
    )
    print("\n===== 更新 config.py 参数库 =====")
    print("\n".join(diff))

    config_path.write_text(
        text[: match.start()] + new_block + text[match.end():], encoding="utf-8"
    )

    # 同步更新内存中的参数库（无需重新 import）
    config_module.ALGORITHM_PARAMS[algorithm_name] = dict(best_params)
    print(f"✅ 算法 '{algorithm_name}' 的最佳参数已写入 config.py 并同步到内存")
