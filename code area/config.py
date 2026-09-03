"""全局配置模块。

所有重要参数集中在此，避免 hard-code 散落在代码各处。
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# ---------------- 经济参数 ----------------
PRICE = 10.0            # 单位售价
UNIT_COST = 4.0         # 单位成本
HOLDING_COST = 1.0      # 单位期末库存持有成本

# ---------------- 环境设置 ----------------
INITIAL_INVENTORY = 0   # 期初库存
LEAD_TIME = 0           # 订货提前期（0 表示当天下单当天到货，可用于销售）

# ---------------- 规划期与动作集 ----------------
N_DAYS = 31
ACTION_SET = list(range(0, 50000 + 1, 1000))  # order-up-to level: 0,1000,...,50000

# ---------------- 复现性 ----------------
RANDOM_SEED = 42

# ---------------- 算法默认参数 ----------------
# 算法参数统一在此配置，也可通过命令行 / tuning 覆盖
ALGORITHM_PARAMS = {
    "ucb": {
        "c": 2000.0,           # exploration 系数
        "optimistic_init": 0.0  # Q 值乐观初始化
    },
    "epsilon_greedy": {
        "epsilon": 0.1,
        "optimistic_init": 0.0
    },
    "gradient": {
        "alpha": 0.5,           # 学习率
        "use_baseline": True,   # 是否使用 reward baseline
        "reward_scale": 10000.0 # reward 缩放（利润数值较大，避免偏好值爆炸）
    },
    # EWF（依据 1234.pdf；本仓库为 censored 环境：observation 不含 demand）
    "ewf": {
        # 默认值经 run_tuning 调优（多种子取均值后确定）。eta/gamma=None
        # 会回退到论文 Theorem 1/2 的理论值；share_alpha>0 切换为 FSF 变体。
        "eta": 0.0001,          # 学习率
        "gamma": 0.0,           # 均匀探索率（论文 Eq.4 的 γ/N）
        "share_alpha": 0.0,     # 0 → 纯 EWF；>0（如 1/N_DAYS）→ FSF 固定份额
        "feedback": "censored", # 本环境只披露被审查的 sales（不披露 demand）
        "cost": "newsvendor",   # 报童损失口径（censored 下唯一可用）
        "overage": None,        # None → unit_cost + holding_cost
        "underage": None,       # None → price − unit_cost
    },
}

# ---------------- 路径 ----------------
DATA_PATH = PROJECT_ROOT / "data" / "daily_demand.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
COMPARISON_DIR = RESULTS_DIR / "comparison"   # 跨算法比较结果
TUNING_DIR = RESULTS_DIR / "tuning"           # 参数优化结果
TABLES_DIR = RESULTS_DIR / "tables"


def ensure_dirs() -> None:
    """确保公共结果目录存在（单算法目录按需在 algorithm_dir 中创建）。"""
    for d in (COMPARISON_DIR, TUNING_DIR, TABLES_DIR):
        d.mkdir(parents=True, exist_ok=True)


def algorithm_dir(algorithm_name: str) -> Path:
    """单个算法的结果目录：results/<algorithm>/（自动创建）。"""
    d = RESULTS_DIR / algorithm_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def algorithm_figures_dir(algorithm_name: str) -> Path:
    """单个算法的图表目录：results/<algorithm>/figures/（自动创建）。"""
    d = algorithm_dir(algorithm_name) / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d
