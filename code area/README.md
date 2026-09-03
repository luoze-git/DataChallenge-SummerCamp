# Data Challenge — 31 天库存在线学习

每天早上仅使用当时可获得的信息决定 order-up-to level `q_t`；当天结束后观察 sales / profit 并用于下一天决策。**严格避免 data leakage**（决策时绝不能使用当天/未来的 demand）。

> 英文文档见根目录 [README.md](../README.md)。

## 环境要求

Python 3.9+，以及三个库：`pip install numpy pandas matplotlib`

## 项目结构

```text
code area/
├── main.py                  # 完整流程主入口
├── config.py                # 全局参数（价格、成本、动作集、算法参数等）
├── environment/
│   └── inventory_env.py     # 库存、销售、利润、状态更新
├── algorithms/
│   ├── __init__.py          # ALGORITHM_REGISTRY
│   ├── base_algorithm.py    # 统一接口
│   ├── ucb.py
│   ├── epsilon_greedy.py
│   ├── gradient.py
│   └── ewf.py               # EWF / FSF（见 1234.pdf）
├── optimization/
│   └── parameter_tuning.py  # Grid Search / Random Search 自动调参
├── visualization/
│   └── visualize.py         # 统一图表生成
├── experiments/
│   ├── simulation.py        # 核心 31 天模拟流程
│   ├── compare_algorithms.py
│   └── run_tuning.py
├── data/
│   └── daily_demand.csv     # 输入数据：`date,demand` 两列，31 行
└── results/
    ├── <algorithm>/          # 单算法目录（ucb / epsilon_greedy / gradient / ewf ...）
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

所有命令都在 `code area` 目录内执行（文件夹名含空格，注意加引号：`cd "code area"`）。

```bash
# 运行单个算法（完整 31 天模拟）
python main.py --algorithm ucb
python main.py --algorithm epsilon_greedy
python main.py --algorithm gradient
python main.py --algorithm ewf      # EWF / FSF（见 1234.pdf）

# 比较多个算法（默认自动包含全部已注册算法）
python -m experiments.compare_algorithms

# 参数优化（Grid / Random Search）
python experiments/run_tuning.py --algorithm ucb --method grid
python experiments/run_tuning.py --algorithm gradient --method random --n-trials 20

# 参数优化 + 自动把最佳参数回写到 config.py
python experiments/run_tuning.py --algorithm ucb --method grid --apply-best
```

### 参数调优工作流

调参**绝不会悄悄修改你的配置**：搜索本身只产出一份带排名的报告；最佳参数只有在你明确要求时，才会进入"参数库"（`config.py` 中的 `ALGORITHM_PARAMS`）：

1. **运行搜索。** 每个候选参数组合都由一次独立的 31 天在线模拟评分（无 data leakage），全部结果按总利润排名：
   ```bash
   python experiments/run_tuning.py --algorithm ucb --method grid
   ```

2. **查看结果。** 所有测试过的组合保存在 `results/tuning/parameter_tuning_<algorithm>.csv`，包含 `params / total_profit / pct_of_upper / rank` 列（`rank = 1` 即最优）。最优参数同时会打印到控制台。

3. **应用最佳参数（需要显式开启）。** 加上 `--apply-best`，把 `rank = 1` 的参数回写到 `config.py` 的 `ALGORITHM_PARAMS`：
   ```bash
   python experiments/run_tuning.py --algorithm ucb --method grid --apply-best
   ```
   具体行为：
   * 写入前在控制台打印 unified diff 展示改动；
   * 保留已存在参数的行内注释；
   * 同时更新当前运行进程内的 `ALGORITHM_PARAMS`。

4. **运行正式实验。** `main.py` 和 `compare_algorithms` 始终从 `config.py` 读取 `ALGORITHM_PARAMS`，因此第 3 步之后自动使用新参数，无需改代码：
   ```bash
   python main.py --algorithm ucb
   ```

如果你想完全手动控制：不加 `--apply-best` 即可，`config.py` 不会被改动；你也可以从 tuning 的 CSV 里把最优的 `params` JSON 手动复制到 `ALGORITHM_PARAMS` 中。

## 关键假设（均可在 config.py 中配置）

* 订货提前期 `LEAD_TIME = 0`：当天订货当天到货，可用于当天销售。
* 每日利润 = `price × sales − unit_cost × order_quantity − holding_cost × inventory_after`
  —— 已售部分计入收入，减去整批订货成本，再减去隔夜库存持有费用。
* 订货量 = `max(0, q_t − inventory_before)` —— 只补齐到目标水平的缺口，绝不退货。
* 销量 = `min(demand, inventory_before + order_quantity)` —— 销量既不能超过需求，也不能超过可用量。
* 未满足的需求视为 lost sales（仅损失潜在收入，不额外扣缺货罚款）。

当前取值：`PRICE = 10.0`、`UNIT_COST = 4.0`、`HOLDING_COST = 1.0`、
`INITIAL_INVENTORY = 0`、`N_DAYS = 31`、`ACTION_SET = 0, 1000, ..., 50000`、
`RANDOM_SEED = 42`。

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
`pct_of_upper`（也会作为列保存在 tuning 的 CSV 中）。在当前数据集下
`upper = 4,014,930`。任何在线算法都无法达到 100%，因为它只能从过去学习，
而上限已经预知了未来。

## 新增算法：EWF / FSF

依据 1234.pdf（Lugosi, Markakis & Neu, arXiv:1710.05739）。EWF（Exponentially
Weighted Forecaster）对每个 order-up-to level 维护指数权重，以
`p_i = (1−γ)·W_i/ΣW + γ/N` 采样。**本仓库是论文严格意义上的 censored
设定**：Environment 交给策略的 observation 里不包含真实 demand，缺货时只能
看到 sales（需求下界）。EWF 的优势正在于对这种“被审查的销售数据”利用其
**局部可观测性**（Lemma 1）做低方差更新，遗憾 `O(√(T log N))`：

* 未缺货日（`sales` < 当天可用量）：`demand == sales` 精确已知，对**全部**
  备选水平做精确的报童损失更新；
* 缺货日（`sales` == 当天可用量）：只知 `demand ≥ 可用量`，用论文 §2.3 的
  估计器仅更新 `≤ 当天水平` 的动作（i 越大越不易被选中，重要性分母
  `P(I_t ≥ i)` 自动校正）。

`share_alpha > 0` 时切换为 FSF（论文 §3，Eq. 6）：权重更新时给每个水平固定
份额 `α/N`，用于跟踪需求漂移。主要参数见 `config.py` 的
`ALGORITHM_PARAMS["ewf"]`：`eta`（学习率，None→理论值）、`gamma`（均匀探索，
None→理论值）、`share_alpha`（0→EWF；>0→FSF）、`feedback`（本环境为
`"censored"`；`"full"` 需 observation 披露 demand）、`cost`（报童损失口径；
env_profit 需逐日真实 demand，censored 下不可用）。细节见
`algorithms/ewf.py` 模块 docstring。

## 扩展新算法

1. 在 `algorithms/` 中新增文件并继承 `BaseAlgorithm`
   （设置 `name` 类属性，然后实现 `select_action(state, available_actions=None)`
   和 `update(observation)`）。
2. 在 `algorithms/__init__.py` 的 `ALGORITHM_REGISTRY` 中注册。
3. 在 `config.py` 的 `ALGORITHM_PARAMS` 中加入默认参数 —— 这些键会作为
   关键字参数传给你的 `__init__`，名字必须一致。

无需修改核心 simulation 代码。

## 常见问题排查

| 问题 | 解决办法 |
|---|---|
| `ModuleNotFoundError: No module named 'algorithms'` | 你运行了 `python experiments/compare_algorithms.py`。请在 `code area` 内改用 `python -m experiments.compare_algorithms` |
| `ModuleNotFoundError: No module named 'pandas'` | `pip install numpy pandas matplotlib` |
| `cd: too many arguments` | 文件夹名含空格 —— 请用 `cd "code area"` |
| `ValueError: ... must contain the columns: date, demand` | `daily_demand.csv` 的表头必须严格是 `date,demand` |
| `ValueError: Not enough demand data` | CSV 行数少于 `N_DAYS` |
| `KeyError: Unknown algorithm '...'` | 算法名严格为 `ucb`、`epsilon_greedy`、`gradient`、`ewf` |
| 写结果时 `PermissionError` | 关闭 Excel 中打开的 CSV 后重试 |
| 两次运行结果不一致 | 传入相同的 `--seed`。注意 `compare_algorithms` 默认跑 3 次取平均 |
