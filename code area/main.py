"""完整流程主入口。

流程：Load Config -> Load Data -> Create Environment -> Create Algorithm
      -> Run 31 Days -> Save Results -> Generate Visualization

用法：
    python main.py --algorithm ucb
    python main.py --algorithm epsilon_greedy
    python main.py --algorithm gradient
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms import ALGORITHM_REGISTRY, create_algorithm
from config import (
    ACTION_SET,
    ALGORITHM_PARAMS,
    DATA_PATH,
    HOLDING_COST,
    INITIAL_INVENTORY,
    LEAD_TIME,
    N_DAYS,
    PRICE,
    RANDOM_SEED,
    UNIT_COST,
    ensure_dirs,
)
from environment.inventory_env import InventoryEnv
from experiments.simulation import (
    load_demand,
    records_to_dataframe,
    report_profit_vs_upper,
    run_simulation,
    save_daily_results,
)
from visualization.visualize import (
    plot_cumulative_profit,
    plot_daily_profit,
    plot_order_up_to,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Data Challenge: 31 天库存在线学习")
    parser.add_argument("--algorithm", default="ucb", choices=sorted(ALGORITHM_REGISTRY),
                        help="使用的算法")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="随机种子（复现）")
    parser.add_argument("--no-figures", action="store_true", help="不生成图表")
    args = parser.parse_args()

    ensure_dirs()

    # 1. 读取数据
    demand_df = load_demand(DATA_PATH)

    # 2. 创建环境（库存/利润逻辑完全封装在 Environment 内）
    env = InventoryEnv(
        demand=demand_df["demand"].tolist(),
        price=PRICE,
        unit_cost=UNIT_COST,
        holding_cost=HOLDING_COST,
        initial_inventory=INITIAL_INVENTORY,
        lead_time=LEAD_TIME,
        n_days=N_DAYS,
    )

    # 3. 创建算法（参数来自 config，不 hard-code）
    params = dict(ALGORITHM_PARAMS.get(args.algorithm, {}))
    algorithm = create_algorithm(args.algorithm, ACTION_SET, params, seed=args.seed)
    print(f"Algorithm: {args.algorithm} | params: {params} | seed: {args.seed}")

    # 4. 运行 31 天
    records = run_simulation(algorithm, env)

    # 5. 保存结果
    df = records_to_dataframe(records)
    out_csv = save_daily_results(df, args.algorithm)
    total = df["profit"].sum()
    print("\n===== 31 天模拟汇总 =====")
    print(df.to_string(index=False))
    print(f"\n总利润   : {total:,.2f}")
    print(f"日均利润 : {total / N_DAYS:,.2f}")
    report_profit_vs_upper(float(total), env, label=args.algorithm)
    print(f"每日结果已保存: {out_csv}")

    # 6. 生成可视化
    if not args.no_figures:
        f1 = plot_order_up_to(df, args.algorithm)
        f2 = plot_daily_profit(df, args.algorithm)
        f3 = plot_cumulative_profit(df, args.algorithm)
        for f in (f1, f2, f3):
            print(f"图表已保存: {f}")


if __name__ == "__main__":
    main()

