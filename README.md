# Data Challenge — 31-Day Inventory Online Learning

Each morning, decide the order-up-to level `q_t` using only the information available at that time; after the day ends, sales / profit are observed and used for the next day's decision. **Data leakage is strictly avoided** (today's or future demand must never be used at decision time).

## Requirements

Python 3.9+ and three libraries: `pip install numpy pandas matplotlib`

## Project Structure

```text
code area/
├── main.py                  # Full pipeline entry point
├── config.py                # Global parameters (price, costs, action set, algorithm params, etc.)
├── environment/
│   └── inventory_env.py     # Inventory, sales, profit, state updates
├── algorithms/
│   ├── __init__.py          # ALGORITHM_REGISTRY
│   ├── base_algorithm.py    # Unified interface
│   ├── ucb.py
│   ├── epsilon_greedy.py
│   └── gradient.py
├── optimization/
│   └── parameter_tuning.py  # Grid Search / Random Search automatic tuning
├── visualization/
│   └── visualize.py         # Unified figure generation
├── experiments/
│   ├── simulation.py        # Core 31-day simulation flow
│   ├── compare_algorithms.py
│   └── run_tuning.py
├── data/
│   └── daily_demand.csv     # Input: columns `date,demand`, 31 rows
└── results/
    ├── <algorithm>/          # Per-algorithm folder (ucb / epsilon_greedy / gradient ...)
    │   ├── daily_results.csv
    │   └── figures/          # order_up_to / daily_profit / cumulative_profit figures for this algorithm
    ├── comparison/           # Cross-algorithm comparison results
    │   ├── algorithm_comparison.csv
    │   ├── algorithm_comparison_curves.csv
    │   └── algorithm_comparison.png
    └── tuning/               # Parameter tuning results
        └── parameter_tuning_<algorithm>.csv
```

## How to Run

Run everything from inside `code area` (the folder name contains a space, so quote it: `cd "code area"`).

```bash
# Run a single algorithm (full 31-day simulation)
python main.py --algorithm ucb
python main.py --algorithm epsilon_greedy
python main.py --algorithm gradient

# Compare multiple algorithms
python -m experiments.compare_algorithms

# Parameter tuning (Grid / Random Search)
python experiments/run_tuning.py --algorithm ucb --method grid
python experiments/run_tuning.py --algorithm gradient --method random --n-trials 20

# Parameter tuning + write the best parameters back into config.py automatically
python experiments/run_tuning.py --algorithm ucb --method grid --apply-best
```

### Parameter Tuning Workflow

Tuning never silently changes your configuration. The search itself only
produces a ranked report; the best parameters reach the "parameter library"
(`ALGORITHM_PARAMS` in `config.py`) only when you explicitly ask for it:

1. **Run the search.** Every candidate combination is scored by an independent
   31-day online simulation (no data leakage), and all results are ranked by
   total profit:
   ```bash
   python experiments/run_tuning.py --algorithm ucb --method grid
   ```

2. **Inspect the results.** All tested combinations are saved to
   `results/tuning/parameter_tuning_<algorithm>.csv` with columns
   `params / total_profit / pct_of_upper / rank` (`rank = 1` is the best).
   The winner is also printed to the console.

3. **Apply the best parameters (opt-in).** Add `--apply-best` to write the
   `rank = 1` parameters back into `ALGORITHM_PARAMS` in `config.py`:
   ```bash
   python experiments/run_tuning.py --algorithm ucb --method grid --apply-best
   ```
   What it does:
   * prints a unified diff of the change before writing;
   * keeps the inline comments of parameters that already existed;
   * also updates `ALGORITHM_PARAMS` in the running process.

4. **Run the final experiment.** `main.py` and `compare_algorithms` always
   read `ALGORITHM_PARAMS` from `config.py`, so after step 3 they use the new
   parameters automatically — no code changes needed:
   ```bash
   python main.py --algorithm ucb
   ```

If you prefer to stay in control, simply omit `--apply-best`: nothing is
written to `config.py`, and you can copy the winning `params` JSON from the
tuning CSV into `ALGORITHM_PARAMS` by hand.

## Key Assumptions (configurable in config.py)

* Order lead time `LEAD_TIME = 0`: units ordered today arrive today and can be sold today.
* Daily profit = `price × sales − unit_cost × order_quantity − holding_cost × inventory_after`
  — revenue on what sold, minus the cost of what was ordered, minus a charge on whatever is left overnight.
* Order quantity = `max(0, q_t − inventory_before)` — you only buy the gap up to your target level, never returning stock.
* Sales = `min(demand, inventory_before + order_quantity)` — you can't sell more than customers want, or more than you have.
* Unmet demand is treated as lost sales (only the potential revenue is lost; no extra stockout penalty).

Current values: `PRICE = 10.0`, `UNIT_COST = 4.0`, `HOLDING_COST = 1.0`, `INITIAL_INVENTORY = 0`, `N_DAYS = 31`, `ACTION_SET = 0, 1000, ..., 50000`, `RANDOM_SEED = 42`.

## Output Files

* `results/<algorithm>/daily_results.csv`: daily results for that algorithm
  (day, inventory_before, order_up_to, order_quantity, demand, sales,
  inventory_after, profit, cumulative_profit).
* `results/<algorithm>/figures/`: order-up-to level, daily profit, and
  cumulative profit figures for that algorithm.
* `results/comparison/algorithm_comparison.csv`: algorithm comparison
  (total / average profit + rank).
* `results/tuning/parameter_tuning_<algorithm>.csv`: all tested parameter
  combinations with rankings.
* `results/comparison/algorithm_comparison.png`: cumulative profit
  comparison across algorithms.

Every run (main / comparison / tuning) also prints the **percentage of the profit upper bound**: the theoretical upper bound under full foresight (setting `q_t = demand` every day) is
`upper = Σ (price − unit_cost) × demand`, and the algorithm's profit as a percentage of that bound is `pct_of_upper` (also saved as a column in the tuning CSV). With the current dataset `upper = 4,014,930`. No online algorithm reaches 100%, since it must learn from the past while the bound already knows the future.

## Extending with a New Algorithm

1. Add a new file in `algorithms/` inheriting from `BaseAlgorithm`
   (set a `name` class attribute, then implement `select_action(state, available_actions=None)` and `update(observation)`).
2. Register it in `ALGORITHM_REGISTRY` in `algorithms/__init__.py`.
3. Add default parameters to `ALGORITHM_PARAMS` in `config.py` — the keys are passed as keyword arguments to your `__init__`, so the names must match.

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'algorithms'` | You ran `python experiments/compare_algorithms.py`. Use `python -m experiments.compare_algorithms` from inside `code area` |
| `ModuleNotFoundError: No module named 'pandas'` | `pip install numpy pandas matplotlib` |
| `cd: too many arguments` | The folder name has a space — use `cd "code area"` |
| `ValueError: ... must contain the columns: date, demand` | The header row in `daily_demand.csv` must be exactly `date,demand` |
| `ValueError: Not enough demand data` | The CSV has fewer rows than `N_DAYS` |
| `KeyError: Unknown algorithm '...'` | Names are exactly `ucb`, `epsilon_greedy`, `gradient` |
| `PermissionError` writing results | Close the CSV in Excel and re-run |
| Results differ between runs | Pass the same `--seed`. Note `compare_algorithms` averages 3 runs by default |
