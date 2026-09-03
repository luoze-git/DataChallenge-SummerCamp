# Data Challenge — 31 天 Inventory Online Learning

每天早晨根据可获得的信息决定 order-up-to level `q_t`，当天结束后获得 sales / profit，
再用于下一天决策。**严格避免 data leakage**（决策时不能使用当天或未来的 demand）。

## 代码结构

```text
code area/
├── main.py                  # 完整流程入口
├── config.py                # 全局参数（价格、成本、动作集、算法参数等）
├── environment/
│   └── inventory_env.py     # 库存、销售、利润、状态更新
├── algorithms/
│   ├── base_algorithm.py    # 统一 Interface
│   ├── ucb.py
│   ├── epsilon_greedy.py
│   └── gradient.py
├── optimization/
│   └── parameter_tuning.py  # Grid Search / Random Search 自动调参
├── visualization/
│   └── visualize.py         # 统一图表生成
├── experiments/
│   ├── simulation.py        # 核心 31-day simulation 流程
│   ├── compare_algorithms.py
│   └── run_tuning.py
├── data/
│   └── daily_demand.csv
└── results/
    ├── <algorithm>/          # 每个算法独立文件夹（ucb / epsilon_greedy / gradient ...）
    │   ├── daily_results.csv
    │   └── figures/          # 该算法的 order_up_to / daily_profit / cumulative_profit 图
    ├── comparison/           # 跨算法比较结果
    │   ├── algorithm_comparison.csv
    │   ├── algorithm_comparison_curves.csv
    │   └── algorithm_comparison.png
    └── tuning/               # 参数优化结果
        └── parameter_tuning_<algorithm>.csv
```

## 运行方式

```bash
# 运行单个算法（31 天完整模拟）
python main.py --algorithm ucb
python main.py --algorithm epsilon_greedy
python main.py --algorithm gradient

# 比较多个算法
python -m experiments.compare_algorithms

# 参数优化（Grid / Random Search）
python experiments/run_tuning.py --algorithm ucb --method grid
python experiments/run_tuning.py --algorithm gradient --method random --n-trials 20
```

## 关键假设（可在 config.py 中修改）

* 订货提前期 `LEAD_TIME = 0`：当天下单当天到货，可用于当天销售。
* 每日利润 = `price × sales − unit_cost × order_quantity − holding_cost × inventory_after`。
* 未满足的 demand 视为 lost sales（只损失潜在收入，无额外缺货罚款）。

## 输出说明

* `results/<algorithm>/daily_results.csv`：该算法的逐日结果
  （day, inventory_before, order_up_to, order_quantity, demand, sales,
  inventory_after, profit, cumulative_profit）。
* `results/<algorithm>/figures/`：该算法的 order-up-to level、daily profit、
  cumulative profit 图。
* `results/comparison/algorithm_comparison.csv`：算法比较（total / average profit + rank）。
* `results/tuning/parameter_tuning_<algorithm>.csv`：所有测试过的参数组合及排名。
* `results/comparison/algorithm_comparison.png`：多算法累计利润对比图。

每次运行（main / comparison / tuning）都会额外打印**利润上限百分比**：
完全预知（每天令 `q_t = demand`）时的理论上限
`upper = Σ (price − unit_cost) × demand`，算法利润占该上限的百分比即
`pct_of_upper`（调参 CSV 中也会保存该列）。

## 扩展新算法

1. 在 `algorithms/` 中新增文件，继承 `BaseAlgorithm`
   （实现 `select_action(state, available_actions)` 与 `update(observation)`）。
2. 在 `algorithms/__init__.py` 的 `ALGORITHM_REGISTRY` 中注册。
3. 在 `config.py` 的 `ALGORITHM_PARAMS` 中加入默认参数。

无需修改核心 simulation 代码。
