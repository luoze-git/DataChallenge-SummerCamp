"""Global configuration module.

All important parameters are centralized here to avoid hard-coding
values scattered across the codebase.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# ---------------- Economic parameters ----------------
PRICE = 10.0            # selling price per unit
UNIT_COST = 4.0         # procurement cost per unit
HOLDING_COST = 1.0      # holding cost per unit of ending inventory

# ---------------- Environment settings ----------------
INITIAL_INVENTORY = 0   # starting inventory
LEAD_TIME = 0           # order lead time (0 = ordered units arrive the same day and can be sold)

# ---------------- Planning horizon & action set ----------------
N_DAYS = 31
ACTION_SET = list(range(0, 50000 + 1, 1000))  # order-up-to level: 0,1000,...,50000

# ---------------- Reproducibility ----------------
RANDOM_SEED = 42

# ---------------- Default algorithm parameters ----------------
# Algorithm parameters are configured here and can be overridden via
# command line / parameter tuning.
ALGORITHM_PARAMS = {
    "ucb": {
        "c": 2000.0,           # exploration coefficient
        "optimistic_init": 0.0  # optimistic initialization of Q values
    },
    "epsilon_greedy": {
        "epsilon": 0.1,
        "optimistic_init": 0.0
    },
    "gradient": {
        "alpha": 0.5,           # learning rate
        "use_baseline": True,   # whether to use a reward baseline
        "reward_scale": 10000.0 # reward scaling (profits are large; keeps preferences stable)
    },
}

# ---------------- Paths ----------------
DATA_PATH = PROJECT_ROOT / "data" / "daily_demand.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
COMPARISON_DIR = RESULTS_DIR / "comparison"   # cross-algorithm comparison results
TUNING_DIR = RESULTS_DIR / "tuning"           # parameter tuning results
TABLES_DIR = RESULTS_DIR / "tables"


def ensure_dirs() -> None:
    """Ensure the shared result directories exist.

    Per-algorithm directories are created on demand in algorithm_dir().
    """
    for d in (COMPARISON_DIR, TUNING_DIR, TABLES_DIR):
        d.mkdir(parents=True, exist_ok=True)


def algorithm_dir(algorithm_name: str) -> Path:
    """Per-algorithm result directory: results/<algorithm>/ (auto-created)."""
    d = RESULTS_DIR / algorithm_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def algorithm_figures_dir(algorithm_name: str) -> Path:
    """Per-algorithm figure directory: results/<algorithm>/figures/ (auto-created)."""
    d = algorithm_dir(algorithm_name) / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d
