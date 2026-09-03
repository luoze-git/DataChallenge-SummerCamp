"""Inventory environment module.

The Environment is solely responsible for: inventory, order quantity,
sales, ending inventory, revenue, cost, holding cost, daily profit,
and state updates.

Core constraint: demand is used only inside the Environment and must
never be exposed to the Algorithm before its decision (strictly no
data leakage).
"""
from __future__ import annotations

import math
from numbers import Real
from typing import Dict, List, Optional


class InventoryEnv:
    """31-day inventory management environment.

    Daily flow:
        The algorithm selects an order-up-to level q_t based on the state
        (information available before the decision)
        -> the Environment computes the order quantity
        -> the Environment computes sales / ending inventory / daily profit
        from the demand
        -> returns an observation (post-decision information, safe to pass
        to algorithm.update)
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
                f"Not enough demand data: need {n_days} days, got {len(demand)}"
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

        self.day = 0                 # current decision day (1-based)
        self.inventory = int(initial_inventory)
        self._done = False
        self._last_record: Optional[Dict] = None

    # ------------------------------------------------------------------
    # Information available BEFORE the decision (never today's/future demand)
    # ------------------------------------------------------------------
    def get_state(self) -> Dict:
        """Return the information the algorithm is allowed to see."""
        return {
            "day": self.day,
            "inventory_before": self.inventory,
        }

    # ------------------------------------------------------------------
    # One-day execution
    # ------------------------------------------------------------------
    def step(self, order_up_to: float) -> Dict:
        """Execute one day.

        Parameters
        ----------
        order_up_to : the order-up-to level q_t chosen by the algorithm today

        Returns
        -------
        observation : post-decision information (safe to pass to
            algorithm.update)
        """
        if self._done:
            raise RuntimeError("Simulation has finished; call reset() first.")
        if self.day < 1 or self.day > self.n_days:
            raise RuntimeError("Invalid day state; call reset() first.")
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

        # 1. Compute the order quantity from the order-up-to level
        order_quantity = max(0, order_up_to - inventory_before)

        # 2. Zero lead time: today's order is available for today's sales
        available = inventory_before + order_quantity

        # 3. Sales (demand is used only inside the Environment)
        demand_today = self._demand[self.day - 1]
        sales = min(demand_today, available)
        inventory_after = available - sales

        # 4. Profit computation (encapsulated entirely in the Environment)
        revenue = self.price * sales
        cost = self.unit_cost * order_quantity
        holding = self.holding_cost * inventory_after
        profit = revenue - cost - holding

        # 5. Update state
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

        # Advance to the next day
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
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(self) -> Dict:
        """Reset the environment for a fresh 31-day simulation."""
        self.day = 1
        self.inventory = int(self.initial_inventory)
        self._done = False
        self._last_record = None
        return self.get_state()

    @property
    def done(self) -> bool:
        return self._done

    # ------------------------------------------------------------------
    # Theoretical upper bound (clairvoyant)
    # ------------------------------------------------------------------
    def profit_upper_bound(self) -> float:
        """Theoretical profit upper bound with full foresight (q_t = demand).

        Sales equal demand exactly every day and ending inventory is 0:
            profit_upper = sum((price - unit_cost) * demand_t)
        No lost sales, no holding cost. No online algorithm can exceed this.
        """
        return float(sum((self.price - self.unit_cost) * d for d in self._demand))
