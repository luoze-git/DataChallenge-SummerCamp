"""统一可视化模块。

所有图表均在此生成，并保存到 results 各目录。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")  # 无显示环境下也可保存图片
import matplotlib.pyplot as plt
import pandas as pd

from config import COMPARISON_DIR, algorithm_figures_dir


def _save(fig, filename: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / filename
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_order_up_to(df: pd.DataFrame, algorithm_name: str = "") -> Path:
    """(1) Daily Order-up-to Level：X: Day 1-31，Y: q_t"""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.step(df["day"], df["order_up_to"], where="mid", marker="o", color="tab:blue")
    ax.set_xlabel("Day")
    ax.set_ylabel("Order-up-to Level")
    ax.set_title(f"Daily Order-up-to Level {f'({algorithm_name})' if algorithm_name else ''}")
    ax.grid(alpha=0.3)
    return _save(fig, "order_up_to.png", algorithm_figures_dir(algorithm_name or "run"))


def plot_daily_profit(df: pd.DataFrame, algorithm_name: str = "") -> Path:
    """(2) Daily Profit：X: Day，Y: 每日利润"""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df["day"], df["profit"], color="tab:green", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Day")
    ax.set_ylabel("Daily Profit")
    ax.set_title(f"Daily Profit {f'({algorithm_name})' if algorithm_name else ''}")
    ax.grid(alpha=0.3, axis="y")
    return _save(fig, "daily_profit.png", algorithm_figures_dir(algorithm_name or "run"))


def plot_cumulative_profit(df: pd.DataFrame, algorithm_name: str = "") -> Path:
    """(3) Cumulative Profit：X: Day，Y: 累计利润"""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["day"], df["cumulative_profit"], marker="o", color="tab:red")
    ax.set_xlabel("Day")
    ax.set_ylabel("Cumulative Profit")
    ax.set_title(f"Cumulative Profit {f'({algorithm_name})' if algorithm_name else ''}")
    ax.grid(alpha=0.3)
    return _save(fig, "cumulative_profit.png", algorithm_figures_dir(algorithm_name or "run"))


def plot_algorithm_comparison(summary: pd.DataFrame = None,
                              curves: pd.DataFrame = None) -> Path:
    """(4) Algorithm Comparison：多个算法的 cumulative profit 对比图。

    Parameters
    ----------
    summary : results/comparison/algorithm_comparison.csv 的内容
              （total profit 对比）
    curves  : results/comparison/algorithm_comparison_curves.csv 的内容
              （cumulative profit 曲线）
    """
    if curves is None:
        curves = pd.read_csv(COMPARISON_DIR / "algorithm_comparison_curves.csv")

    fig, ax = plt.subplots(figsize=(10, 5))
    y_col = "day" if "day" in curves.columns else curves.columns[0]
    for col in curves.columns:
        if col == y_col:
            continue
        ax.plot(curves[y_col], curves[col], marker="o", label=col)
    ax.set_xlabel("Day")
    ax.set_ylabel("Cumulative Profit")
    ax.set_title("Algorithm Comparison — Cumulative Profit")
    ax.legend()
    ax.grid(alpha=0.3)

    if summary is not None and len(summary):
        text = "\n".join(
            f"{r.algorithm}: {r.total_profit:,.0f}"
            for r in summary.itertuples()
        )
        ax.text(0.02, 0.98, text, transform=ax.transAxes,
                va="top", fontsize=9,
                bbox=dict(boxstyle="round", alpha=0.15))

    return _save(fig, "algorithm_comparison.png", COMPARISON_DIR)
