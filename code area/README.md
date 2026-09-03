# Data Challenge — 31 天库存在线学习

每天早上仅使用当时可获得的信息决定 order-up-to level `q_t`；当天结束后观察 sales / profit 并用于下一天决策。**严格避免 data leakage**（决策时绝不能使用当天/未来的 demand）。

## 项目结构

```text
code area/
├── main.py                  # 完整流程主入口
├── config.py                # 全局参数（价格、成本、动作集、算法参数等）
├── environment/
│   └── inventory_env.py     # 库存、销售、利润、状态更新
├── algorithms/
│   ├── base_algorithm.py    # 统一接口
│   ├── ucb.py
│   ├── epsilon_greedy.py
│   └── gradient.py
├── optimization/
│   └── parameter_tuning.py  # Grid Search / Random Search 自动调参
├── visualization/
│   └── visualize.py         # 统一图表生成
├── experiments/
│   ├── simulation.py        # 核心 31 天模拟流程
│   ├── compare_algorithms.py
│   └── run_tuning.py
├── data/
│   └── daily_demand.csv
└── results/
    ├── <algorithm>/          # 单算法目录（ucb / epsilon_greedy / gradient ...）
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
# 运行单个算法（完整 31 天模拟）
python main.py --algorithm ucb
python main.py --algorithm epsilon_greedy
python main.py --algorithm gradient

# 比较多个算法
python -m experiments.compare_algorithms

# 参数优化（Grid / Random Search）
python experiments/run_tuning.py --algorithm ucb --method grid
python experiments/run_tuning.py --algorithm gradient --method random --n-trials 20
```

## 关键假设（均可在 config.py 中配置）

* 订货提前期 `LEAD_TIME = 0`：当天订货当天到货，可用于当天销售。
* 每日利润 = `price × sales − unit_cost × order_quantity − holding_cost × inventory_after`。
* 未满足的需求视为 lost sales（仅损失潜在收入，不额外扣缺货罚款）。

## 输出文件

* `results/<algorithm>/daily_results.csv`：该算法的每日结果
  （day, inventory_before, order_up_to, order_quantity, demand, sales,
  inventory_after, profit, cumulative_profit）。
* `results/<algorithm>/figures/`：该算法的 order-up-to level、每日利润、
  累计利润图。
* `results/comparison/algorithm_comparison.csv`：算法比较
  （总利润 / 平均利润 + 排名）。
* `results/tuning/parameter_tuning_<algorithm>.csv`：所有测试过的参数
  组合及排名。
* `results/comparison/algorithm_comparison.png`：跨算法 cumulative
  profit 对比图。

每次运行（main / comparison / tuning）都会同时打印**利润相对上限的百分比**：完全预知（每天令 `q_t = demand`）的理论上限为
`upper = Σ (price − unit_cost) × demand`，算法利润占该上限的百分比即
`pct_of_upper`（也会作为列保存在 tuning 的 CSV 中）。

## 扩展新算法

1. 在 `algorithms/` 中新增文件并继承 `BaseAlgorithm`
   （实现 `select_action(state, available_actions)` 和 `update(observation)`）。
2. 在 `algorithms/__init__.py` 的 `ALGORITHM_REGISTRY` 中注册。
3. 在 `config.py` 的 `ALGORITHM_PARAMS` 中加入默认参数。

无需修改核心 simulation 代码。
