"""统一可视化模块。

所有图表在此生成并保存到 results/figures/。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")  # 无界面环境下也能保存图片
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
    """① Daily Order-up-to Level: X: Day 1–31, Y: q_t"""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.step(df["day"], df["order_up_to"], where="mid", marker="o", color="tab:blue")
    ax.set_xlabel("Day")
    ax.set_ylabel("Order-up-to Level")
    ax.set_title(f"Daily Order-up-to Level {f'({algorithm_name})' if algorithm_name else ''}")
    ax.grid(alpha=0.3)
    return _save(fig, "order_up_to.png", algorithm_figures_dir(algorithm_name or "run"))


def plot_daily_profit(df: pd.DataFrame, algorithm_name: str = "") -> Path:
    """② Daily Profit: X: Day, Y: Daily Profit"""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df["day"], df["profit"], color="tab:green", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Day")
    ax.set_ylabel("Daily Profit")
    ax.set_title(f"Daily Profit {f'({algorithm_name})' if algorithm_name else ''}")
    ax.grid(alpha=0.3, axis="y")
    return _save(fig, "daily_profit.png", algorithm_figures_dir(algorithm_name or "run"))


def plot_cumulative_profit(df: pd.DataFrame, algorithm_name: str = "") -> Path:
    """③ Cumulative Profit: X: Day, Y: Cumulative Profit"""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["day"], df["cumulative_profit"], marker="o", color="tab:red")
    ax.set_xlabel("Day")
    ax.set_ylabel("Cumulative Profit")
    ax.set_title(f"Cumulative Profit {f'({algorithm_name})' if algorithm_name else ''}")
    ax.grid(alpha=0.3)
    return _save(fig, "cumulative_profit.png", algorithm_figures_dir(algorithm_name or "run"))


def plot_algorithm_comparison(summary: pd.DataFrame = None,
                              curves: pd.DataFrame = None) -> Path:
    """④ Algorithm Comparison：多个算法的 cumulative profit 放在同一张图中。

    Parameters
    ----------
    summary : results/comparison/algorithm_comparison.csv 的内容（总利润对比）
    curves  : results/comparison/algorithm_comparison_curves.csv 的内容（累计利润曲线）
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
