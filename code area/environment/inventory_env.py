"""库存环境模块。

Environment 统一负责：库存、订货量、销售、期末库存、收入、成本、
持有成本、每日利润与状态更新。

核心限制：demand 只能被 Environment 内部使用，
绝不能在 decision 之前暴露给 Algorithm（严格避免 data leakage）。
"""
from __future__ import annotations

import math
from numbers import Real
from typing import Dict, List, Optional


class InventoryEnv:
    """31 天库存管理环境。

    每天流程：
        算法根据 state（决策前可获得的信息）选择 order-up-to level q_t
        → Environment 计算 order quantity
        → Environment 根据 demand 计算 sales / ending inventory / daily profit
        → 返回 observation（决策之后的信息，可以交给算法用于更新）
    """

    def __init__(
        self,
        demand: List[int],
        price: float,
        unit_cost: float,
        holding_cost: float,
        initial_inventory: int = 0,
        lead_time: int = 0,
        n_days: int = 31,
    ) -> None:
        if n_days <= 0:
            raise ValueError("n_days must be positive")
        if len(demand) < n_days:
            raise ValueError(
                f"demand 数据不足：需要 {n_days} 天，实际只有 {len(demand)} 天"
            )
        if any(
            not isinstance(value, Real)
            or not math.isfinite(float(value))
            or float(value) < 0
            or not float(value).is_integer()
            for value in demand[:n_days]
        ):
            raise ValueError("demand must contain non-negative integer quantities")
        if any(value < 0 for value in (price, unit_cost, holding_cost)):
            raise ValueError("price, unit_cost, and holding_cost must be non-negative")
        if initial_inventory < 0:
            raise ValueError("initial_inventory must be non-negative")
        if lead_time != 0:
            raise ValueError("This assignment assumes lead_time=0")

        # Private simulator input: never include this sequence in policy state.
        self._demand = [int(value) for value in demand[:n_days]]
        self.price = float(price)
        self.unit_cost = float(unit_cost)
        self.holding_cost = float(holding_cost)
        self.initial_inventory = int(initial_inventory)
        self.lead_time = int(lead_time)
        self.n_days = int(n_days)

        self.day = 0                 # 当前即将进行的决策日（1-based）
        self.inventory = int(initial_inventory)
        self._done = False
        self._last_record: Optional[Dict] = None

    # ------------------------------------------------------------------
    # 决策前可获得的信息（绝不含当天/未来 demand）
    # ------------------------------------------------------------------
    def get_state(self) -> Dict:
        """返回算法决策时允许看到的信息。"""
        return {
            "day": self.day,
            "inventory_before": self.inventory,
        }

    # ------------------------------------------------------------------
    # 单日执行
    # ------------------------------------------------------------------
    def step(self, order_up_to: float) -> Dict:
        """执行一天。

        Parameters
        ----------
        order_up_to : 当天算法选择的 order-up-to level q_t

        Returns
        -------
        observation : 决策之后才能获得的信息（可安全交给 algorithm.update）
        """
        if self._done:
            raise RuntimeError("Simulation 已经结束，请先调用 reset()。")
        if self.day < 1 or self.day > self.n_days:
            raise RuntimeError("非法的 day 状态，请先调用 reset()。")
        if (
            isinstance(order_up_to, bool)
            or not isinstance(order_up_to, Real)
            or not math.isfinite(float(order_up_to))
            or float(order_up_to) < 0
            or not float(order_up_to).is_integer()
        ):
            raise ValueError("order_up_to must be a non-negative integer level")

        order_up_to = int(order_up_to)
        inventory_before = self.inventory

        # 1. 根据 order-up-to level 计算订货量
        order_quantity = max(0, order_up_to - inventory_before)

        # 2. 提前期为 0：当天订货可用于当天销售
        available = inventory_before + order_quantity

        # 3. 销售（demand 只在 Environment 内部使用）
        demand_today = self._demand[self.day - 1]
        sales = min(demand_today, available)
        inventory_after = available - sales

        # 4. 利润计算（统一封装在 Environment 中）
        revenue = self.price * sales
        cost = self.unit_cost * order_quantity
        holding = self.holding_cost * inventory_after
        profit = revenue - cost - holding

        # 5. 更新状态
        self.inventory = inventory_after
        # Safe policy-facing observation. True demand is intentionally omitted:
        # after a stockout, sales reveal only a lower bound on demand.
        observation = {
            "day": self.day,
            "inventory_before": inventory_before,
            "order_up_to": order_up_to,
            "order_quantity": order_quantity,
            "sales": sales,
            "inventory_after": inventory_after,
            "profit": profit,
        }
        # Full record is kept separately for evaluation and result reporting.
        self._last_record = {**observation, "demand": demand_today}

        # 推进到下一天
        if self.day >= self.n_days:
            self._done = True
        else:
            self.day += 1

        return observation

    def get_last_record(self) -> Dict:
        """Return the evaluator's full record for the most recent day.

        This record contains true demand and must not be passed to the policy.
        """
        if self._last_record is None:
            raise RuntimeError("No day has been executed yet.")
        return dict(self._last_record)

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def reset(self) -> Dict:
        """重置环境，开始新一轮 31 天模拟。"""
        self.day = 1
        self.inventory = int(self.initial_inventory)
        self._done = False
        self._last_record = None
        return self.get_state()

    @property
    def done(self) -> bool:
        return self._done

    # ------------------------------------------------------------------
    # 理论上限（完全预知）
    # ------------------------------------------------------------------
    def profit_upper_bound(self) -> float:
        """完全预知（每次都令 q_t = demand）时的利润理论上限。

        此时每天销售恰好等于 demand、期末库存为 0：
            profit_upper = Σ (price - unit_cost) * demand_t
        无缺货损失、无持有成本。任何在线算法都无法超过该值。
        """
        return float(sum((self.price - self.unit_cost) * d for d in self._demand))
