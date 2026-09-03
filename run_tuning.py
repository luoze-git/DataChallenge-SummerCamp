"""参数优化实验入口。

用法示例：
    python experiments/run_tuning.py --algorithm ucb --method grid
    python experiments/run_tuning.py --algorithm gradient --method random --n-trials 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 保证以任意 cwd 运行时都能找到项目根目录的模块
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ALGORITHM_PARAMS
from experiments.compare_algorithms import load_demand_by_config
from optimization.parameter_tuning import tune_parameters


def main() -> None:
    parser = argparse.ArgumentParser(description="Run parameter tuning")
    parser.add_argument("--algorithm", required=True,
                        choices=sorted(ALGORITHM_PARAMS),
                        help="要调参的算法")
    parser.add_argument("--method", default="grid", choices=["grid", "random"],
                        help="搜索方法：grid / random")
    parser.add_argument("--n-trials", type=int, default=20,
                        help="random search 的试验次数")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 默认搜索空间：围绕 config 中的默认值构造（可按需扩展）
    default = ALGORITHM_PARAMS[args.algorithm]
    if args.algorithm == "ucb":
        param_grid = {"c": [500.0, 1000.0, 2000.0, 5000.0, 10000.0],
                      "optimistic_init": [0.0]}
    elif args.algorithm == "epsilon_greedy":
        param_grid = {"epsilon": [0.02, 0.05, 0.1, 0.2, 0.3],
                      "optimistic_init": [0.0]}
    elif args.algorithm == "ewf":
        # EWF / FSF：需求在环境中事后可观测 → feedback 用 "full"；
        # eta 决定学习快慢、gamma 决定探索、share_alpha>0 时启用 FSF 跟踪非平稳。
        param_grid = {
            "eta": [1e-4, 3e-4, 1e-3, 3e-3, 5e-3, 1e-2],
            "gamma": [0.0, 0.02],
            "share_alpha": [0.0, 1.0 / 31.0, 0.1],
            "feedback": ["full"],
            "cost": ["newsvendor", "env_profit"],
            "overage": [None],
            "underage": [None],
        }
    else:  # gradient
        param_grid = {"alpha": [0.1, 0.25, 0.5, 1.0, 2.0],
                      "use_baseline": [True],
                      "reward_scale": [default["reward_scale"]]}

    demand_df = load_demand_by_config()
    result = tune_parameters(
        algorithm_name=args.algorithm,
        param_grid=param_grid,
        demand_df=demand_df,
        method=args.method,
        n_trials=args.n_trials,
        seed=args.seed,
    )

    best = result.iloc[0]
    print("\n===== Best parameters =====")
    print(f"params: {best['params']}")
    print(f"total_profit: {best['total_profit']:.2f}")
    print(f"results saved to results/tuning/parameter_tuning_{args.algorithm}.csv")


if __name__ == "__main__":
    main()

