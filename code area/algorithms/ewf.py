"""EWF / FSF 算法 —— 对应 1234.pdf（Lugosi, Markakis & Neu, arXiv:1710.05739）。

论文针对的是重复报童（repeated newsvendor）问题：无需求先验、只能看到被审查
（censored / sales）数据，需要一边探索一边利用地决定每个周期的库存水平。
本仓库的 Environment 正是这种**严格被审查设定**：交给策略的 observation 中
故意不包含真实 demand，缺货时只能看到 sales（真实需求的一个下界）。

实现的两个算法本质是同一套指数权重机制：

* EWF  —— Exponentially Weighted Forecaster（论文 §2.3）：
    对每个可选 order-up-to level 维护指数权重 W_i，采样概率
        p_i(t) = (1 − γ) * W_i(t−1)/ΣW + γ / N          （论文 Eq. 4）
    其中 γ/N 保证每个水平至少有固定概率被尝试（探索）。
* FSF  —— Fixed-Share Forecaster（论文 §3，跟踪遗憾 / 非平稳需求变体）：
    更新权重时额外把一小块质量 α/N * ΣW 分给每个专家（论文 Eq. 6），
    让先前“变差”的订货水平也能在需求漂移后快速重新获得权重。

与普通 bandit 的关键差异 —— 成本/收益的“结构”被显式利用：
论文证明销售信号存在局部可观测性（Lemma 1），一次被审查的销量观察其实
已经携带了所有“低于等于当天库存水平”的备选动作的信息。于是更新不再只
针对被选中的那个水平，而是能对备选水平做低方差更新，遗憾可达 O(√(T log N))。

信息自适应（本实现与全信息环境的区别）：
    update(observation) 从不假设 observation 里有 demand：
      * 若 observation 确实披露了 demand（兼容全信息版本），直接用它；
      * 否则（本仓库的 censored 环境）：
          - 未缺货日：sales < 当天可用量 ⇒ 销量就是真实需求（demand=sales），
            因此可对所有备选水平做**精确**的报童损失更新；
          - 缺货日：sales == 当天可用量 ⇒ 真实需求被截断，仅知道 demand ≥
            可用量。此时用论文 §2.3 的局部可观测性估计器更新 i ≤ I_t 的水平
            （Lemma 1），其余水平本次不更新。

经济损失映射（报童损失与利润一致）：任意两个动作的利润差 = − 其报童损失差
（差一个仅与当日 demand 有关的公共常数，softmax 指数权重对其不变），因此
最小化 newsvendor 损失与最大化利润等价。取
    overage  = unit_cost + holding_cost     （多订：付了货款又承担持有）
    underage = price − unit_cost            （少订：损失边际利润）

实现细节：权重以 log-space 保存（G_i = log W_i），用 log-sum-exp 做 softmax，
避免 η·成本 较大时概率下溢成 0 后“永不复活”；FSF 的线性混合份额在概率空间
用正 α 恢复一个下界，天然稳定。

参数均从外部传入（config / tuning），不在算法内部 hard-code。
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, Optional

import numpy as np

from algorithms.base_algorithm import BaseAlgorithm


def _default_rates() -> Dict[str, float]:
    """缺省 overage/underage：把报童损失映射到本项目的经济参数。"""
    try:
        from config import HOLDING_COST, PRICE, UNIT_COST
    except Exception:  # 兜底：若在 config 不可用的环境下直接实例化
        PRICE, UNIT_COST, HOLDING_COST = 10.0, 4.0, 1.0
    return {
        "overage": float(UNIT_COST + HOLDING_COST),
        "underage": float(PRICE - UNIT_COST),
    }


class EWF(BaseAlgorithm):
    """Exponentially Weighted Forecaster（+ Fixed-Share 变体），censored 设定下。

    最小化逐日报童损失：
        select_action 按 p_i = (1−γ)·softmax_i + γ/N 采样一个 order-up-to
        update         自适应地使用销售反馈更新权重（见模块 docstring）
    """

    name = "ewf"

    def __init__(
        self,
        action_set: Iterable,
        overage: Optional[float] = None,
        underage: Optional[float] = None,
        eta: Optional[float] = None,
        gamma: Optional[float] = None,
        share_alpha: float = 0.0,
        feedback: str = "censored",
        cost: str = "newsvendor",
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__(action_set, rng=rng)
        if feedback not in ("full", "censored"):
            raise ValueError(f"feedback 只能是 'full'/'censored'，got {feedback!r}")
        if cost not in ("newsvendor", "env_profit"):
            raise ValueError(f"cost 只能是 'newsvendor'/'env_profit'，got {cost!r}")
        if cost == "env_profit" and feedback == "censored":
            raise ValueError(
                "env_profit 口径需要逐日真实 demand（观察必须披露 demand）；"
                "本仓库是 censored 环境，请用 cost='newsvendor'"
            )

        rates = _default_rates()
        self.overage = float(overage if overage is not None else rates["overage"])
        self.underage = float(underage if underage is not None else rates["underage"])

        # β = D · max{overage, underage}：损失估计的上界常量（论文 Lemma 2）
        self._D = float(max(self.actions))
        self.beta = self._D * max(self.overage, self.underage)

        self.feedback = feedback
        self.cost = cost
        self.share_alpha = float(share_alpha)  # >0 → FSF 固定份额混合

        # 缺省学习率 / 探索率按论文理论值设定（详见 Theorem 1 / 2）
        if eta is None or gamma is None:
            theo_eta, theo_gamma = self._theory_params()
            self.eta = float(eta if eta is not None else theo_eta)
            self.gamma = float(gamma if gamma is not None else theo_gamma)
        else:
            self.eta = float(eta)
            self.gamma = float(gamma)

        # 指数权重（log-space），初始均匀
        self._log_weights = np.zeros(self.n_actions, dtype=float)
        # 最近一次用于采样的分布（censored 更新需要 P(I_t ≥ i)）
        self._sel_probs = np.full(self.n_actions, 1.0 / self.n_actions, dtype=float)

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _theory_params(self):
        """论文 Theorem 1 / 2 的理论参数（已知 T 时的默认值）。"""
        try:
            from config import N_DAYS
        except Exception:
            N_DAYS = 31
        T = max(1, int(N_DAYS))
        N = max(2, self.n_actions)
        b = self.beta
        gamma = 1.0 / (2.0 * b * T)
        eta = math.sqrt(
            math.log(N) / (4.0 * b * b * T * math.log(2.0 * b * T * N ** 3 + N + 2))
        )
        return eta, gamma

    def _softmax(self) -> np.ndarray:
        """由 log-weights 得到 softmax 概率（log-sum-exp，数值稳定）。"""
        g = self._log_weights - self._log_weights.max()
        e = np.exp(g)
        return e / e.sum()

    def _action_values(self) -> np.ndarray:
        """动作集合转为 array（int）。"""
        return np.asarray(self.actions, dtype=float)

    def _same_day_cost(self, d: float, inventory_before: float) -> np.ndarray:
        """已知当日真实需求 d 时，每个动作的精确报童损失（越小越优）。

        loss(q; d) = overage·(q−d)⁺ + underage·(d−q)⁺
        与每日反事实会计利润只差一个仅与 d 有关的公共常数，因此最小化它
        等价于最大化利润。仅当 demand 精确已知（观察披露，或未缺货日
        demand==sales）才调用；缺货日请走论文 §2.3 估计器。
        """
        q = self._action_values()
        diff = q - d
        return self.overage * np.maximum(diff, 0.0) + self.underage * np.maximum(-diff, 0.0)

    def _try_recover_demand(self, observation: Dict) -> Optional[float]:
        """自适应地恢复当天真实需求；无法恢复（缺货日）返回 None。

        * 若 observation 直接披露 demand（兼容全信息环境）→ 用之；
        * 否则按 censored 规则：
            - sales < 当天可用量 ⇒ 未缺货 ⇒ demand == sales；
            - sales == 当天可用量 ⇒ 缺货，demand 只知下界 → 返回 None。
        """
        if self.feedback == "full":
            # 全信息环境承诺每天披露 demand；缺失则报错（而不是悄悄退化）
            if "demand" not in observation:
                raise KeyError(
                    "feedback='full' 要求 observation 披露 'demand'；"
                    "本环境未提供，请改用 feedback='censored'"
                )
            return float(observation["demand"])

        if "demand" in observation:  # censored 环境也可能额外给出（评估用），直接用
            return float(observation["demand"])

        sales = float(observation["sales"])
        available = float(observation["inventory_before"]) + float(observation["order_quantity"])
        if sales < available - 1e-9:  # 未缺货：卖了多少就是真实需求
            return sales
        return None  # 缺货日：demand 被截断，只能用估计器

    def _censored_cost_estimates(self, observation: Dict) -> np.ndarray:
        """论文 §2.3 的局部可观测性估计器：只用被审查的 sales 更新 i ≤ I_t。

        缺货日（sales == 当天可用量 ≥ I_t）只说明 demand ≥ I_t。对任意
        i ≤ I_t，min(i, demand) = i（需求必超过该水平），于是销售信号是
        “完全被审查”的，Lemma 1 的代理量有效：
            surrogate(i) = overage·i − (overage+underage)·sales_i,
            其中 sales_i = min(i, sales) = i（因 i ≤ I_t ≤ sales）。
        再除以 P(I_t ≥ i) 做重要性校正（i > I_t 的动作本次不更新）。
        """
        It = float(observation["order_up_to"])
        sales = float(observation["sales"])
        # 注意：I_t 是 order-up-to 目标水平；缺货日必有 sales==available≥I_t
        q = self._action_values()
        hb = self.overage + self.underage
        # 对 i ≤ I_t 的动作，被审查的销量 sales_i = min(i, demand) = i
        surrogate = self.overage * q - hb * q          # == −underage·i
        est = np.zeros(self.n_actions, dtype=float)
        valid = q <= It
        if valid.any():
            tail = self._sel_probs  # P(I_t = j)（本算法当天的采样分布）
            # P(I_t ≥ i) = 选择概率中所有 j ≥ i 之和
            cum_from_top = np.cumsum(tail[::-1])[::-1]
            p_ge = cum_from_top.copy()
            est[valid] = (surrogate[valid] + self.beta) / p_ge[valid]
        return est

    # ------------------------------------------------------------------
    # BaseAlgorithm 接口
    # ------------------------------------------------------------------
    def select_action(self, state: Dict, available_actions: Optional[Iterable] = None):
        actions = self._resolve_actions(available_actions)
        idx = [self.action_index[a] for a in actions]

        p = self._softmax()
        p_sel = (1.0 - self.gamma) * p + self.gamma / self.n_actions

        # 支持受限动作子集：在子集内重归一化（仅当需要时）
        if len(idx) < self.n_actions:
            restricted = p_sel[idx]
            restricted = restricted / restricted.sum()
            full = np.zeros(self.n_actions)
            full[idx] = restricted
            p_sel = full

        self._sel_probs = p_sel.copy()
        # 论文建议用概率采样做探索（p_sel ≥ γ/N > 0 保证每动作都有机会）
        j = int(self.rng.choice(len(idx), p=p_sel[idx]))
        return actions[j]

    def update(self, observation: Dict) -> None:
        d = self._try_recover_demand(observation)
        if d is not None:
            # 当天真实需求已知（全信息 / 未缺货）：对所有动作精确更新
            cost = self._same_day_cost(d, float(observation.get("inventory_before", 0.0)))
        else:
            # 缺货日：真实需求被截断 → 论文 §2.3 局部可观测性估计器
            cost = self._censored_cost_estimates(observation)

        # log-space：W_i ← W_i · e^{−η·ĉ_i}   （== G_i ← G_i − η·ĉ_i）
        self._log_weights -= self.eta * cost

        # FSF（论文 Eq. 6）：每个专家额外获得 α/N · ΣW 的固定份额。
        # 概率空间实现：m_i = p_i·e^{−ηĉ_i}；p'_i = (m_i + α/N) / (Σm + α)
        if self.share_alpha > 0:
            p = self._softmax()
            m = p * np.exp(-self.eta * cost)
            alpha = self.share_alpha
            p_new = (m + alpha / self.n_actions) / (m.sum() + alpha)
            p_new = np.clip(p_new, 0.0, 1.0)
            p_new /= p_new.sum()
            self._log_weights = np.log(p_new)
