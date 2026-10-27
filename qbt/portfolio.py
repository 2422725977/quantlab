"""多标的组合层。

单标的回测回答「这个信号有没有用」，组合层回答「一堆信号放在一起怎么配」。
核心是三件事：**权重怎么定、多久再平衡、换手要花多少钱**。

与单标的引擎保持同一口径：t 日收盘产生目标权重 → t+1 日开盘成交。
再平衡日之间权重随价格自然漂移（不是每天强行拉回，那会低估成本）。

参考：bt（pmoronczyk）的组合树、qlib 的 portfolio 层、zipline 的 optimize。
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .costs import CostModel
from .metrics import _annual_factor, compute_metrics


# --------------------------------------------------------------------------
# 数据准备
# --------------------------------------------------------------------------
def build_panel(data: dict[str, pd.DataFrame], field: str = "open") -> pd.DataFrame:
    """多标的拼成宽表（列名=标的代码）。

    各标的上市/停牌日期不同，先按各自序列前向填充，再取**日期交集**——
    保证每一行都是「所有标的都有真实价格」的交易日，避免用 NaN 造出假收益。
    """
    cols = {}
    for k, df in data.items():
        if df is None or len(df) == 0 or field not in df.columns:
            continue
        s = df[field].astype(float)
        s = s[~s.index.duplicated(keep="last")]
        cols[k] = s
    if not cols:
        raise ValueError("没有可用标的")
    panel = pd.DataFrame(cols).sort_index()
    panel = panel.ffill().dropna(how="any")
    if len(panel) < 30:
        raise ValueError(f"共同交易日仅 {len(panel)} 天，标的日期重叠不足")
    return panel


def open_to_open_returns(panel_open: pd.DataFrame) -> pd.DataFrame:
    """与单标的引擎一致：t 日开盘 → t+1 日开盘的收益。"""
    return (panel_open.shift(-1) / panel_open - 1.0).fillna(0.0)


# --------------------------------------------------------------------------
# 权重分配
# --------------------------------------------------------------------------
def allocate(
    signals: pd.DataFrame,
    returns: pd.DataFrame,
    method: str = "equal",
    lookback: int = 60,
    max_weight: float = 0.6,
) -> pd.DataFrame:
    """把「是否持仓」的 0/1 信号转成实际权重矩阵。

    equal      : 持仓标的等权
    inv_vol    : 按波动率倒数加权（低波动多配）
    risk_parity: 风险平价（迭代求解，使各标的风险贡献相等）
                 注：标的少（<3）且波动率差异大时会退化成角点解，
                 故对单标的权重施加 max_weight 上限，避免全押一个。
    signal     : 直接用信号数值当权重（信号本身可带强度）
    """
    sig = signals.reindex(returns.index).ffill().fillna(0.0)
    active = (sig.abs() > 1e-9).astype(float)

    if method == "signal":
        w = sig.abs()
        s = w.sum(axis=1).replace(0, np.nan)
        return (w.div(s, axis=0)).fillna(0.0)

    if method in ("inv_vol", "risk_parity"):
        vol = returns.rolling(lookback, min_periods=max(10, lookback // 3)).std(ddof=1)
        vol = vol.clip(lower=1e-6).ffill().bfill()
        if method == "inv_vol":
            raw = active / vol
        else:
            raw = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
            minp = max(10, lookback // 3)
            for i in range(len(returns)):
                mask = active.iloc[i].to_numpy() > 0
                if not mask.any():
                    continue
                c = returns.iloc[max(0, i - lookback) : i]
                if len(c) < minp:
                    # 样本不足：退化为等权
                    raw.iloc[i, mask] = 1.0 / mask.sum()
                    continue
                # 只在**持仓子集**内求风险贡献；必须用协方差而非相关系数
                # （相关系数对角线恒为 1，会丢掉波动率差异，退化成等权）
                # 协方差随窗口滚动，故不做缓存——缓存会复用过期权重
                sub = c.iloc[:, mask]
                cov = np.nan_to_num(sub.cov().to_numpy())
                n = cov.shape[0]
                cap = max(1.0 / n, min(max_weight, 1.0))
                w = np.ones(n) / n
                for _ in range(80):            # 投影不动点迭代
                    rc = w * (cov @ w)         # 各标的风险贡献
                    s = rc.sum()
                    if s <= 0:
                        break
                    target = s / n             # 目标是每标的风险贡献 = s/n
                    # 贡献多的减仓、贡献少的加仓（方向不能反，否则发散到角点）
                    w = w * (target / np.maximum(rc, 1e-12)) ** 0.5
                    w = _cap_normalize(w, cap)  # 带上限的归一化
                raw.iloc[i, mask] = w
    else:
        raw = active

    s = raw.sum(axis=1).replace(0, np.nan)
    w = raw.div(s, axis=0).fillna(0.0)
    # 保留信号方向（允许做空时为负）
    direction = np.sign(sig).replace(0, 1.0)
    return w * direction


def _cap_normalize(w: np.ndarray, cap: float) -> np.ndarray:
    """归一化到和为 1，且每个分量 ≤ cap（水位填充）。

    直接 clip 再归一化会让被截断的权重重新膨胀超过 cap —— 这是风险平价
    迭代中最常见的实现错误，必须用「把超额部分分给未触顶者」的方式投影。
    """
    w = np.clip(np.asarray(w, dtype=float), 0.0, None)
    n = len(w)
    if n == 0:
        return w
    cap = max(cap, 1.0 / n)
    total = w.sum()
    if total <= 0:
        return np.full(n, 1.0 / n)
    w = w / total
    for _ in range(64):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w[over] = cap
        free = ~over
        base = float(w[free].sum())
        if not free.any() or base <= 1e-12:
            w[:] = 1.0 / n
            break
        w[free] += excess * w[free] / base
    return w / max(w.sum(), 1e-12)


def cross_sectional_momentum(
    close_panel: pd.DataFrame,
    lookback: int = 60,
    top_k: int = 3,
    skip_recent: int = 5,
    rebalance: str = "ME",
    allow_short_bottom: bool = False,
    bottom_k: int = 0,
) -> pd.DataFrame:
    """截面动量轮动：每个再平衡期按过去 N 日涨幅排序，买最强的 top_k 个。

    skip_recent：跳过最近几日（避开短期反转效应，学术上常见做法）。
    """
    # 用 t-1 及之前的信息打分（避免未来函数）
    lag = close_panel.shift(skip_recent)
    mom = (lag / lag.shift(lookback) - 1.0)
    mom = mom.iloc[lookback:]

    reb = _rebalance_mask(mom.index, rebalance)
    target = pd.DataFrame(0.0, index=close_panel.index, columns=close_panel.columns)
    cur = pd.Series(0.0, index=close_panel.columns)
    for i, d in enumerate(close_panel.index):
        if d in mom.index and reb.get(d, False):
            row = mom.loc[d].dropna()
            if len(row) == 0:
                continue
            ranked = row.sort_values(ascending=False)
            w = pd.Series(0.0, index=close_panel.columns)
            for sym in ranked.index[:top_k]:
                w[sym] = 1.0 / max(top_k, 1)
            if allow_short_bottom and bottom_k > 0:
                for sym in ranked.index[-bottom_k:]:
                    w[sym] -= 1.0 / bottom_k
            cur = w
        target.loc[d] = cur
    return target


def _rebalance_mask(index: pd.Index, freq: str) -> dict:
    """标记哪些交易日是再平衡日。"""
    idx = pd.DatetimeIndex(index)
    if freq in ("D", "", None) or len(idx) == 0:
        return {d: True for d in idx}
    key = {
        "W": idx.isocalendar().week.astype(str) + idx.year.astype(str),
        "ME": idx.to_period("M"),
        "QE": idx.to_period("Q"),
        "YE": idx.to_period("Y"),
    }.get(freq, idx.to_period("M"))
    s = pd.Series(idx, index=idx)
    first = s.groupby(key).transform("min")
    return {d: (d == f) for d, f in zip(idx, first.to_numpy())}


# --------------------------------------------------------------------------
# 组合回测
# --------------------------------------------------------------------------
def backtest_portfolio(
    returns: pd.DataFrame,
    target_w: pd.DataFrame,
    *,
    initial_cash: float = 1_000_000.0,
    rebalance: str = "ME",
    cost: CostModel | None = None,
    max_weight_per_asset: float = 1.0,
    gross_leverage: float = 1.0,
) -> dict:
    """组合回测：再平衡日之间权重随价格漂移，只在再平衡日付成本。

    returns : 各标的 open-to-open 收益（宽表）
    target_w: 各标的 t 日收盘得出的目标权重（宽表）
    """
    if cost is None:
        cost = CostModel(commission=0.0005, slippage=0.0005, slippage_mode="pct")
    ret = returns.reindex(sorted(returns.index))
    tgt = target_w.reindex(ret.index).ffill().fillna(0.0).reindex(columns=ret.columns).fillna(0.0)
    tgt = tgt.clip(lower=-max_weight_per_asset, upper=max_weight_per_asset)
    gross = tgt.abs().sum(axis=1)
    over = gross / max(gross_leverage, 1e-9)
    tgt = tgt.div(over.where(over > 1.0, 1.0), axis=0)

    reb = _rebalance_mask(ret.index, rebalance)
    n = len(ret)
    cols = list(ret.columns)
    r_np = ret.to_numpy(dtype=float)
    t_np = tgt.to_numpy(dtype=float)

    held = np.zeros(len(cols))
    net = np.zeros(n)
    held_w = np.zeros((n, len(cols)))
    turnover = np.zeros(n)
    trade_days = 0

    # 逐日推进：第 i 日的收益由「第 i 日开盘时的持仓 held」产生。
    # held 来自 t_{i-1} 收盘的决策 → 在 t_i 开盘成交（与单标的引擎同一滞后口径）。
    for i in range(n - 1):
        is_reb = bool(reb.get(ret.index[i], False))
        if is_reb or i == 0:
            # t-1 收盘定下的目标（t_np[i-1]），t 日开盘执行；首日无前值则空仓
            target = t_np[i - 1] if i > 0 else np.zeros(len(cols))
            delta = target - held
            turnover[i] = float(np.abs(delta).sum())
            if turnover[i] > 1e-9:
                trade_days += 1
                # 成本按换手额 × 费率近似（买入部分与卖出部分）
                buy = float(np.clip(delta, 0, None).sum())
                sell = float(np.clip(-delta, 0, None).sum())
                rate_b = cost.commission + cost.transfer_fee + (
                    cost.stamp_duty if cost.stamp_duty_both_sides else 0.0) + cost.slippage
                rate_s = cost.commission + cost.transfer_fee + cost.stamp_duty + cost.slippage
                cost_i = buy * rate_b + sell * rate_s
            else:
                cost_i = 0.0
            held = target.copy()
        else:
            cost_i = 0.0
            # 权重随价格漂移
            grown = held * (1.0 + (r_np[i - 1] if i > 0 else 0.0))
            s = np.abs(grown).sum()
            held = grown / s if s > 1e-12 else grown

        net[i] = float(np.sum(held * r_np[i])) - cost_i
        held_w[i] = held

    net[n - 1] = 0.0
    equity = pd.Series(initial_cash * np.cumprod(1.0 + net), index=ret.index, name="equity")

    # 各标的贡献（持仓权重 × 该标的收益 的逐日累加）
    contrib = {}
    for j, c in enumerate(cols):
        contrib[c] = float(np.sum(held_w[:, j] * r_np[:, j]))

    metrics = compute_metrics(equity)
    metrics["turnover_total"] = float(turnover.sum())
    metrics["turnover_per_year"] = float(turnover.sum() / max(len(equity) / _annual_factor(equity.index), 1e-9))
    metrics["rebalance_days"] = trade_days
    metrics["n_assets"] = len(cols)

    return {
        "equity": equity,
        "weights": pd.DataFrame(held_w, index=ret.index, columns=cols),
        "turnover": pd.Series(turnover, index=ret.index),
        "contribution": contrib,
        "metrics": {k: v for k, v in metrics.items() if k != "monthly_returns"},
        "params": {
            "rebalance": rebalance,
            "initial_cash": initial_cash,
            "cost": cost.to_dict(),
            "gross_leverage": gross_leverage,
        },
    }


def compare_weights(
    returns: pd.DataFrame,
    signals: pd.DataFrame,
    methods: tuple = ("equal", "inv_vol", "risk_parity"),
    max_weight: float = 0.6,
    **bt_kw,
) -> list[dict]:
    """对比不同加权方式的组合绩效（同一信号、同一成本、同一再平衡频率）。"""
    out = []
    for m in methods:
        try:
            w = allocate(signals, returns, method=m, max_weight=max_weight)
            r = backtest_portfolio(returns, w, **bt_kw)
            out.append({"method": m, **{k: v for k, v in r["metrics"].items() if k != "monthly_returns"}})
        except Exception as e:  # noqa: BLE001
            out.append({"method": m, "error": str(e)})
    return out


def correlation_matrix(returns: pd.DataFrame) -> dict:
    """相关性矩阵（用于判断组合是否真的分散了风险）。"""
    c = returns.corr()
    vals = c.to_numpy()
    n = vals.shape[0]
    off = vals[~np.eye(n, dtype=bool)] if n > 1 else np.array([])
    return {
        "labels": [str(x) for x in c.columns],
        "matrix": [[None if not np.isfinite(v) else float(v) for v in row] for row in vals],
        "avg_pairwise_corr": float(np.nanmean(off)) if len(off) else None,
    }
