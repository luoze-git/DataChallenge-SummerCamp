"""EWF / FSF 算法 —— 对应 1234.pdf（Lugosi, Markakis & Neu, arXiv:1710.05739）。

论文针对的是重复报童（repeated newsvendor）问题：无需求先验、只能看到被审查
（censored / sales）数据，需要一边探索一边利用地决定每个周期的库存水平。

实现的两个算法本质是同一套指数权重机制：

* EWF  —— Exponentially Weighted Forecaster（论文 §2.3 / §2.4）：
    对每个可选 order-up-to level 维护指数权重 W_i，采样概率
        p_i(t) = (1 − γ) * W_i(t−1)/ΣW + γ / N          （论文 Eq. 4）
    其中 γ/N 保证每个水平至少有固定概率被尝试（探索）。
* FSF  —— Fixed-Share Forecaster（论文 §3，跟踪遗憾 / 非平稳需求变体）：
    更新权重时额外把一小块质量 α/N * ΣW 分给每个专家（论文 Eq. 6），
    让先前“变差”的订货水平也能在需求漂移后快速重新获得权重。

与普通 bandit 的关键差异 —— 成本/收益的“结构”被显式利用：
论文证明销售信号存在局部可观测性（Lemma 1），一次被审查的销量观察其实
已经携带了所有“低于等于当天库存水平”的备选动作的信息。于是更新不再只
针对被选中的那个水平，而是能对（接近）全部备选水平做低方差更新，遗憾
可达 O(√(T log N))。

本实现通过两个正交参数把论文的两处自由度做成可配置：

* feedback: 当日结束后算法能拿到什么信息
    - "full"     使用环境披露的真实 demand（对应论文 §2.4 的无审查情形，
                 每次把全部动作的损失都精确算出来，最省样本）。
    - "censored" 仅使用被审查的 sales + 是否缺货，配合论文 §2.3 的
                 局部可观测性估计器来更新权重（即使 demand 被隐藏也适用）。
* cost:     每个动作按什么口径计算当日损失
    - "newsvendor"  损失 = overage·(q−d)⁺ + underage·(d−q)⁺
                    （论文标准损失；overage/underage 见下映射）。
    - "env_profit"  损失 = −(当日反事实利润)，即用真实库存/销量口径重放
                    “若选了该 order-up-to 今天会计上会记多少利润”，取负并
                    逐日平移为非负。与比较口径（累计利润）最一致。

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
from typing import Dict, Iterable, List, Optional

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
    """Exponentially Weighted Forecaster（+ Fixed-Share 变体）。

    最小化逐日“报童式”损失（或环境口径利润的负值）：
        select_action 按 p_i = (1−γ)·softmax_i + γ/N 采样一个 order-up-to
        update         用当天反馈对（部分或全部）备选水平更新指数权重
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
        feedback: str = "full",
        cost: str = "newsvendor",
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__(action_set, rng=rng)
        if feedback not in ("full", "censored"):
            raise ValueError(f"feedback 只能是 'full'/'censored'，got {feedback!r}")
        if cost not in ("newsvendor", "env_profit"):
            raise ValueError(f"cost 只能是 'newsvendor'/'env_profit'，got {cost!r}")
        if cost == "env_profit" and feedback == "censored":
            raise ValueError("env_profit 口径需要真实 demand，feedback 必须为 'full'")

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

    def _same_day_cost(self, d: int, inventory_before: float) -> np.ndarray:
        """计算每个动作当日的非负“损失”（越小越优）。

        newsvendor：loss = overage·(q−d)⁺ + underage·(d−q)⁺
        env_profit：loss = −反事实当日利润，逐日平移至非负。
        """
        q = self._action_values()
        if self.cost == "newsvendor":
            diff = q - d
            return self.overage * np.maximum(diff, 0.0) + self.underage * np.maximum(-diff, 0.0)

        # env_profit：用真实库存口径重放“若选了 q，今天会计利润是多少”。
        # 口径与环境保持一致：order_qty=(q−inv)⁺；available=max(inv,q)。
        try:
            from config import HOLDING_COST, PRICE, UNIT_COST
        except Exception:
            PRICE, UNIT_COST, HOLDING_COST = 10.0, 4.0, 1.0
        inv = float(inventory_before)
        order_qty = np.maximum(q - inv, 0.0)
        available = inv + order_qty                       # == max(inv, q)
        sales = np.minimum(d, available)
        inventory_after = available - sales
        profit = (float(PRICE) * sales
                  - float(UNIT_COST) * order_qty
                  - float(HOLDING_COST) * inventory_after)
        # 平移为非负（同一天公共常数不改变 softmax 相对关系）
        return profit.max() - profit

    def _censored_cost_estimates(self, observation: Dict) -> np.ndarray:
        """论文 §2.3 的局部可观测性估计器：只用 sales 重建所有 ≤ 当天水平动作的损失。

        记当天所选水平为 I_t（order_up_to），observed sales 为 s：
          - 若未缺货（s < 当天可用量）：真实 demand 已知 = s；
          - 若缺货（s == 当天可用量）：demand ≥ 可用量，则对任意 i ≤ I_t 都有
            min(i, demand) = i（全部被审查），只需假设库存不高于目标水平。
        用 Lemma 1 的代理量 surrogate(i) = overage·i − (overage+underage)·sales_i
        与 β 构造非负的 ĉ(i)，再除以 P(I_t ≥ i) 做重要性校正。
        """
        It = float(observation["order_up_to"])
        sales = float(observation["sales"])
        available = float(observation["inventory_before"]) + float(observation["order_quantity"])
        censored = sales >= available - 1e-9  # 缺货：只看到销量，不知道真实 demand

        q = self._action_values()
        hb = self.overage + self.underage
        # 对 i ≤ It 的动作，sales_i = min(i, demand) 可由当前反馈确定
        sales_i = np.where(q <= It, np.minimum(q, sales), np.nan)
        surrogate = self.overage * q - hb * sales_i        # 仅 i ≤ It 处有效
        est = np.zeros(self.n_actions, dtype=float)
        valid = q <= It
        if valid.any():
            tail = self._sel_probs  # P(I_t = j)
            # P(I_t ≥ i) = 选择概率中所有 j ≥ i 之和
            cum_from_top = np.cumsum(tail[::-1])[::-1]
            # q 单调递增 ⇒ 每列取第一个满足 j≥q_i 的位置（即 i 处）
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
        d = int(round(float(observation["demand"])))
        if self.feedback == "full":
            cost = self._same_day_cost(d, observation["inventory_before"])
        else:
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
