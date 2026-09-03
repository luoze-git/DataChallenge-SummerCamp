"""库存环境模块。

Environment 统一负责：库存、订货量、销售、期末库存、收入、成本、
持有成本、每日利润与状态更新。

核心限制：demand 只能被 Environment 内部使用，
绝不能在 decision 之前暴露给 Algorithm（严格避免 data leakage）。
"""
from __future__ import annotations

from typing import Dict, List


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
        if len(demand) < n_days:
            raise ValueError(
                f"demand 数据不足：需要 {n_days} 天，实际只有 {len(demand)} 天"
            )
        self.demand = list(demand[:n_days])
        self.price = float(price)
        self.unit_cost = float(unit_cost)
        self.holding_cost = float(holding_cost)
        self.initial_inventory = int(initial_inventory)
        self.lead_time = int(lead_time)
        self.n_days = int(n_days)

        self.day = 0                 # 当前即将进行的决策日（1-based）
        self.inventory = int(initial_inventory)
        self.pending_orders: Dict[int, int] = {}  # 到货日 -> 订货量
        self._done = False

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

        inventory_before = self.inventory

        # 1. 根据 order-up-to level 计算订货量
        order_quantity = max(0, int(round(order_up_to)) - inventory_before)

        # 2. 订货到货（按 lead_time 安排）
        arrival_day = self.day + self.lead_time
        if self.lead_time == 0:
            available = inventory_before + order_quantity
        else:
            self.pending_orders[arrival_day] = (
                self.pending_orders.get(arrival_day, 0) + order_quantity
            )
            available = inventory_before + self.pending_orders.pop(self.day, 0)

        # 3. 销售（demand 只在 Environment 内部使用）
        demand_today = self.demand[self.day - 1]
        sales = min(demand_today, available)
        inventory_after = available - sales

        # 4. 利润计算（统一封装在 Environment 中）
        revenue = self.price * sales
        cost = self.unit_cost * order_quantity
        holding = self.holding_cost * inventory_after
        profit = revenue - cost - holding

        # 5. 更新状态
        self.inventory = inventory_after
        observation = {
            "day": self.day,
            "inventory_before": inventory_before,
            "order_up_to": order_up_to,
            "order_quantity": order_quantity,
            "demand": demand_today,          # 决策后才可观察
            "sales": sales,
            "inventory_after": inventory_after,
            "profit": profit,
        }

        # 推进到下一天
        if self.day >= self.n_days:
            self._done = True
        else:
            self.day += 1

        return observation

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def reset(self) -> Dict:
        """重置环境，开始新一轮 31 天模拟。"""
        self.day = 1
        self.inventory = int(self.initial_inventory)
        self.pending_orders = {}
        self._done = False
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
        return float(sum((self.price - self.unit_cost) * d for d in self.demand))
