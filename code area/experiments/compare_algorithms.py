"""算法比较实验。

单独运行某个算法 / 比较多个算法，结果保存到 results/comparison/。
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from algorithms import create_algorithm
from config import (
    ACTION_SET,
    ALGORITHM_PARAMS,
    N_DAYS,
    COMPARISON_DIR,
    ensure_dirs,
)
from experiments.simulation import (
    build_env,
    load_demand,
    records_to_dataframe,
    report_profit_vs_upper,
    run_simulation,
)


def run_single(algorithm_name: str, params: Dict = None, seed: int = None,
               demand_df: pd.DataFrame = None) -> List[Dict]:
    """运行单个算法的一次完整模拟（严格遵守 31-day online constraint）。"""
    if demand_df is None:
        demand_df = load_demand_by_config()
    params = dict(params) if params else dict(ALGORITHM_PARAMS.get(algorithm_name, {}))
    algorithm = create_algorithm(algorithm_name, ACTION_SET, params, seed=seed)
    env = build_env(demand_df)
    return run_simulation(algorithm, env)


def load_demand_by_config() -> pd.DataFrame:
    from config import DATA_PATH

    return load_demand(DATA_PATH)


def compare_algorithms(algorithm_names: List[str], seeds: List[int] = None) -> pd.DataFrame:
    """比较多个算法（支持多次运行取平均，保证可复现）。

    Returns
    -------
    DataFrame: algorithm, total_profit, average_daily_profit,
               以及 cumulative profit 曲线数据另存。
    """
    ensure_dirs()
    seeds = seeds if seeds else [0]

    # 完全预知上限（q_t = demand），任何算法都无法超过
    demand_df = load_demand_by_config()
    env = build_env(demand_df)
    upper = env.profit_upper_bound()
    print(f"Profit upper bound (q=demand, clairvoyant): {upper:,.2f}\n")

    summary_rows = []
    cumulative_frames = {}

    for name in algorithm_names:
        run_totals = []
        run_curves = []
        for k, seed in enumerate(seeds):
            records = run_single(name, seed=None if seed is None else int(seed) + k)
            df = records_to_dataframe(records)
            run_totals.append(df["profit"].sum())
            run_curves.append(df["cumulative_profit"].values)

        total_mean = float(np.mean(run_totals))
        pct = report_profit_vs_upper(total_mean, env, label=name)
        summary_rows.append({
            "algorithm": name,
            "total_profit": total_mean,
            "average_daily_profit": total_mean / N_DAYS,
            "pct_of_upper": pct,
            "n_runs": len(seeds),
        })
        cumulative_frames[name] = np.mean(run_curves, axis=0)

    summary = pd.DataFrame(summary_rows)
    summary["rank"] = summary["total_profit"].rank(ascending=False, method="min").astype(int)
    summary = summary.sort_values("rank")
    summary.to_csv(COMPARISON_DIR / "algorithm_comparison.csv", index=False)

    # 保存 cumulative profit 曲线（供可视化使用）
    curve_df = pd.DataFrame({"day": range(1, N_DAYS + 1)})
    for name, curve in cumulative_frames.items():
        curve_df[name] = curve
    curve_df.to_csv(COMPARISON_DIR / "algorithm_comparison_curves.csv", index=False)

    return summary


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from config import RANDOM_SEED
    from visualization.visualize import plot_algorithm_comparison

    parser = argparse.ArgumentParser(description="Compare algorithms")
    parser.add_argument("--algorithms", nargs="+",
                        default=sorted(ALGORITHM_PARAMS),
                        help="要比较的算法列表")
    parser.add_argument("--n-runs", type=int, default=3, help="每个算法运行次数（取平均）")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    seeds = [args.seed + i for i in range(args.n_runs)]
    summary = compare_algorithms(args.algorithms, seeds=seeds)
    print("\n===== Algorithm comparison =====")
    print(summary.to_string(index=False))
    print("Saved: results/comparison/algorithm_comparison.csv")

    curves = pd.read_csv(COMPARISON_DIR / "algorithm_comparison_curves.csv")
    fig = plot_algorithm_comparison(summary, curves)
    print(f"Figure saved: {fig}")
