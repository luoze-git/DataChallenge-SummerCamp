"""31 天在线学习模拟核心流程。

严格遵循 information constraint：
每天 Environment 只把「决策前可获得的信息」交给 algorithm.select_action，
决策之后才把 observation 交给 algorithm.update。
任何一次 simulation（包括调参时的重复运行）都遵守该约束。
"""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

from algorithms import BaseAlgorithm
from environment.inventory_env import InventoryEnv


def run_simulation(algorithm: BaseAlgorithm, env: InventoryEnv) -> List[Dict]:
    """运行完整 31 天模拟，返回逐日记录列表。

    主流程只通过统一 Interface 与算法交互：
        action = algorithm.select_action(state)
        algorithm.update(observation)
    """
    state = env.reset()
    records: List[Dict] = []
    cumulative_profit = 0.0

    while not env.done:
        # 1. 算法决策（只能看到 state：day / inventory_before）
        action = algorithm.select_action(state)

        # 2. Environment 执行当天（订货、销售、利润、状态更新）
        observation = env.step(action)

        # 3. 只把可观测的 observation 交给算法（不含真实 demand）
        algorithm.update(observation)

        # 4. 完整 demand 仅用于评估和结果记录，不回传给算法
        cumulative_profit += observation["profit"]
        record = env.get_last_record()
        record["cumulative_profit"] = cumulative_profit
        records.append(record)

        state = env.get_state()

    return records


def records_to_dataframe(records: List[Dict]) -> pd.DataFrame:
    """将逐日记录整理为 DataFrame（列顺序符合开发文档要求）。"""
    columns = [
        "day",
        "inventory_before",
        "order_up_to",
        "order_quantity",
        "demand",
        "sales",
        "inventory_after",
        "profit",
        "cumulative_profit",
    ]
    df = pd.DataFrame(records)
    return df[columns]


def build_env(demand_df: pd.DataFrame) -> InventoryEnv:
    """根据 config 构造环境（供 main / comparison / tuning 复用）。"""
    from config import (
        HOLDING_COST,
        INITIAL_INVENTORY,
        LEAD_TIME,
        N_DAYS,
        PRICE,
        UNIT_COST,
    )

    return InventoryEnv(
        demand=demand_df["demand"].tolist(),
        price=PRICE,
        unit_cost=UNIT_COST,
        holding_cost=HOLDING_COST,
        initial_inventory=INITIAL_INVENTORY,
        lead_time=LEAD_TIME,
        n_days=min(N_DAYS, len(demand_df)),
    )


def report_profit_vs_upper(total_profit: float, env: InventoryEnv, label: str = "") -> float:
    """打印利润相对完全预知上限（q_t = demand）的百分比，返回该百分比。"""
    upper = env.profit_upper_bound()
    pct = total_profit / upper * 100 if upper else float("nan")
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}Profit vs upper bound (q=demand): "
          f"{total_profit:,.2f} / {upper:,.2f} = {pct:.2f}%")
    return pct


def load_demand(path) -> pd.DataFrame:
    """读取 demand 数据文件（date, demand）。"""
    df = pd.read_csv(path)
    if not {"date", "demand"}.issubset(df.columns):
        raise ValueError(f"{path} 必须包含列: date, demand")
    return df


def save_daily_results(df: pd.DataFrame, algorithm_name: str, run_label: str = "") -> str:
    """保存每日结果到 results/<algorithm>/。"""
    from config import algorithm_dir

    filename = f"daily_results{('_' + run_label) if run_label else ''}.csv"
    out = algorithm_dir(algorithm_name) / filename
    df.to_csv(out, index=False)
    return str(out)
