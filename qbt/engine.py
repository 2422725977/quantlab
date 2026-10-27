"""回测引擎：向量化（快，用于参数扫描） + 事件驱动（精确，用于最终验证）。

两条路线的共同约定
------------------
signal / target：Series，值与 df 同索引，语义是「在第 t 根 K 线收盘时做出的决策」，
    从而在第 t+1 根 K 线的**开盘价**成交 —— 这是避免未来函数的底线。
weight：目标仓位权重，1.0 = 满仓，0 = 空仓，负数 = 做空（需 allow_short）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .costs import CostModel


def _align(df: pd.DataFrame, signal: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    s = pd.Series(signal).reindex(df.index).ffill().fillna(0.0).astype(float)
    return df, s


# --------------------------------------------------------------------------
# 1) 向量化引擎
# --------------------------------------------------------------------------
def backtest_vector(
    df: pd.DataFrame,
    signal: pd.Series,
    *,
    initial_cash: float = 1_000_000.0,
    commission: float = 0.0005,
    slippage: float = 0.0005,
    allow_short: bool = False,
    cost: CostModel | None = None,
) -> dict:
    """向量化回测。

    收益口径：在 t 日开盘建仓、t+1 日开盘平仓/换仓，因此单 bar 收益 = open_{t+1}/open_t - 1。
    成本：买入换手 × 买入费率 + 卖出换手 × 卖出费率（含印花税、规费、最低佣金）。
    速度极快，适合扫几千组参数；但不模拟盘中止损/止盈 —— 那由事件驱动引擎负责。
    """
    df, sig = _align(df, signal)
    if cost is not None:
        cm = cost
    else:
        cm = CostModel(commission=commission, slippage=slippage, slippage_mode="pct")
    if not allow_short:
        sig = sig.clip(lower=0.0)

    op = df["open"].astype(float)
    pos = sig.shift(1).fillna(0.0)          # t 日持有的仓位（决策于 t-1 收盘）
    # t 日开盘建仓 → t+1 日开盘换仓，对应收益
    ret = (op.shift(-1) / op - 1.0).fillna(0.0)
    gross = pos * ret

    dpos = (pos - pos.shift(1).fillna(0.0))
    buy_turn = dpos.clip(lower=0.0)
    sell_turn = (-dpos).clip(lower=0.0)
    slip = cm.slippage
    buy_rate = cm.commission + cm.transfer_fee + (cm.stamp_duty if cm.stamp_duty_both_sides else 0.0) + slip
    sell_rate = cm.commission + cm.transfer_fee + cm.stamp_duty + slip
    cost_series = buy_turn * buy_rate + sell_turn * sell_rate

    # 最低佣金：按「当日有交易」摊销到净值比例
    if cm.min_commission > 0:
        active = ((buy_turn.abs() + sell_turn.abs()) > 1e-9).astype(float)
        eq_approx = initial_cash * (1.0 + gross).cumprod().shift(1).fillna(initial_cash)
        cost_series = cost_series + active * (cm.min_commission / eq_approx.replace(0, np.nan)).fillna(0.0)

    net = gross - cost_series

    equity = initial_cash * (1.0 + net).cumprod()
    equity.iloc[0] = initial_cash

    return {
        "equity": equity,
        "position": pos,
        "trades": _trades_from_position(df, pos, buy_rate + sell_rate),
        "params": {
            "engine": "vector",
            "initial_cash": initial_cash,
            "commission": cm.commission,
            "slippage": cm.slippage,
            "stamp_duty": cm.stamp_duty,
            "cost_model": cm.to_dict(),
        },
    }


def _trades_from_position(df: pd.DataFrame, pos: pd.Series, cost_rate: float) -> list[dict]:
    """由仓位序列反推成交记录（向量化引擎用）。"""
    trades = []
    op = df["open"].astype(float)
    idx = list(df.index)
    cur = None
    for i, d in enumerate(idx):
        p = float(pos.iloc[i])
        if cur is None and abs(p) > 1e-9:
            cur = {
                "entry_date": d, "entry_price": float(op.iloc[i]),
                "weight": p, "bars": 0,
            }
        elif cur is not None:
            cur["bars"] += 1
            if abs(p) <= 1e-9 or i == len(idx) - 1:
                exit_price = float(op.iloc[i]) if i < len(idx) else float(df["close"].iloc[-1])
                raw = (exit_price / cur["entry_price"] - 1.0) * np.sign(cur["weight"] or 1)
                net = raw - 2 * cost_rate
                trades.append(
                    {
                        "entry_date": str(cur["entry_date"])[:10],
                        "exit_date": str(d)[:10],
                        "entry_price": round(cur["entry_price"], 4),
                        "exit_price": round(exit_price, 4),
                        "weight": round(cur["weight"], 3),
                        "bars": cur["bars"],
                        "return": float(net),
                        "pnl": None,
                        "reason": "signal",
                    }
                )
                cur = None
    return trades


# --------------------------------------------------------------------------
# 2) 事件驱动引擎
# --------------------------------------------------------------------------
def backtest_event(
    df: pd.DataFrame,
    signal: pd.Series,
    *,
    initial_cash: float = 1_000_000.0,
    commission: float = 0.0005,
    slippage: float = 0.0005,
    allow_short: bool = False,
    t_plus_1: bool = False,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    trailing_stop: float | None = None,
    max_weight: float = 1.0,
    rebalance_tol: float = 0.02,
    cost: CostModel | None = None,
) -> dict:
    """事件驱动回测：逐 bar 推进，精确模拟开盘成交、盘中止损/止盈、T+1。

    stop_loss / take_profit / trailing_stop 为比例（0.10 表示 -10%）。
    rebalance_tol：目标权重与当前权重差小于该阈值则不换仓，避免每日微幅抖动刷出假成交。
    cost：传入 CostModel 时启用真实费率（印花税 / 最低佣金 / 成交量冲击滑点）。
    """
    df, sig = _align(df, signal)
    if cost is not None:
        cm = cost
        slippage = cm.slippage
    else:
        cm = CostModel(commission=commission, slippage=slippage, slippage_mode="pct")
    op = df["open"].to_numpy(dtype=float)
    hi = df["high"].to_numpy(dtype=float)
    lo = df["low"].to_numpy(dtype=float)
    cl = df["close"].to_numpy(dtype=float)
    vol = df["volume"].to_numpy(dtype=float) if "volume" in df.columns else np.zeros(len(df))
    idx = list(df.index)

    cash = float(initial_cash)
    shares = 0.0
    pending = 0.0                     # 待在下一根开盘执行的目标权重
    entry_price = 0.0
    peak_price = 0.0
    entry_bar = -1
    cur_trade: dict | None = None
    equity_curve = []
    trades: list[dict] = []
    pos_curve = []

    def buy(price: float, qty: float, bar: int, reason: str) -> None:
        nonlocal cash, shares, entry_price, peak_price, entry_bar, cur_trade
        if qty <= 0:
            return
        px = cm.buy_price(price, vol[bar] if bar < len(vol) else 0.0, qty)
        gross = qty * px
        fee = cm.fee(gross, "buy")
        if gross + fee > cash:
            rate = (fee / gross) if gross > 0 else cm.commission
            qty = max(cash / (px * (1 + rate)), 0.0)
            gross = qty * px
            fee = cm.fee(gross, "buy")
        cash -= gross + fee
        shares += qty
        if cur_trade is None:
            cur_trade = {"entry_date": idx[bar], "entry_price": px, "qty": qty, "fee": fee}
            entry_price = px
            peak_price = px
            entry_bar = bar
        else:
            cur_trade["qty"] += qty
            cur_trade["fee"] += fee

    def sell(price: float, qty: float, bar: int, reason: str) -> None:
        nonlocal cash, shares, cur_trade, entry_price, peak_price
        if qty <= 0 or shares <= 0:
            return
        qty = min(qty, shares)
        px = cm.sell_price(price, vol[bar] if bar < len(vol) else 0.0, qty)
        gross = qty * px
        fee = cm.fee(gross, "sell")
        cash += gross - fee
        shares -= qty
        if cur_trade is not None:
            ct = cur_trade
            realized = (px - ct["entry_price"]) * qty - ct["fee"] - fee
            trades.append(
                {
                    "entry_date": str(ct["entry_date"])[:10],
                    "exit_date": str(idx[bar])[:10],
                    "entry_price": round(ct["entry_price"], 4),
                    "exit_price": round(px, 4),
                    "qty": round(qty, 4),
                    "bars": bar - idx.index(ct["entry_date"]) if ct["entry_date"] in idx else 0,
                    "return": float(realized / (ct["entry_price"] * qty)) if ct["entry_price"] > 0 else 0.0,
                    "pnl": float(realized),
                    "reason": reason,
                }
            )
            if shares <= 1e-12:
                cur_trade = None
                entry_price = 0.0
                peak_price = 0.0
            else:
                cur_trade = {"entry_date": ct["entry_date"], "entry_price": ct["entry_price"],
                             "qty": shares, "fee": ct["fee"]}

    for t in range(len(df)):
        o, h, l, c = op[t], hi[t], lo[t], cl[t]
        equity_open = cash + shares * o

        # --- 1. 执行上一根收盘产生的目标仓位（本根开盘成交） ---
        target = pending
        if not allow_short:
            target = max(0.0, target)
        target = min(target, max_weight)
        cur_weight = (shares * o) / equity_open if equity_open > 0 else 0.0
        if abs(target - cur_weight) < rebalance_tol:
            desired_shares = shares
        else:
            desired_shares = (target * equity_open) / (o * (1 + slippage)) if o > 0 else 0.0
        delta = desired_shares - shares
        if abs(delta) > 1e-9 and desired_shares > 1e-12:
            if delta > 0:
                buy(o, delta, t, "signal")
            else:
                if t_plus_1 and entry_bar == t:
                    pass  # A股 T+1：当日买入不可当日卖出，顺延
                else:
                    sell(o, -delta, t, "signal")
        elif desired_shares <= 1e-12 and shares > 0:
            if not (t_plus_1 and entry_bar == t):
                sell(o, shares, t, "signal")

        # --- 2. 盘中风险单 ---
        if shares > 0 and entry_price > 0:
            peak_price = max(peak_price, h)
            exit_px = None
            reason = None
            if stop_loss is not None:
                sl_px = entry_price * (1 - stop_loss)
                if l <= sl_px:
                    exit_px, reason = min(sl_px, o), "stop_loss"
            if exit_px is None and take_profit is not None:
                tp_px = entry_price * (1 + take_profit)
                if h >= tp_px:
                    exit_px, reason = max(tp_px, o), "take_profit"
            if exit_px is None and trailing_stop is not None and peak_price > 0:
                tr_px = peak_price * (1 - trailing_stop)
                if l <= tr_px and peak_price > entry_price:
                    exit_px, reason = min(tr_px, o), "trailing_stop"
            if exit_px is not None:
                if not (t_plus_1 and entry_bar == t):
                    sell(exit_px, shares, t, reason)

        # --- 3. 收盘结算 ---
        equity_curve.append(cash + shares * c)
        pos_curve.append((shares * c) / (cash + shares * c) if (cash + shares * c) > 0 else 0.0)

        # --- 4. 依据今日收盘产生下一根的目标 ---
        pending = float(sig.iloc[t])
        if not allow_short:
            pending = max(0.0, pending)
        pending = min(pending, max_weight)

    # 收尾平仓
    if shares > 0:
        sell(cl[-1], shares, len(df) - 1, "close_end")

    equity = pd.Series(equity_curve, index=df.index, name="equity")
    return {
        "equity": equity,
        "position": pd.Series(pos_curve, index=df.index, name="position"),
        "trades": trades,
        "cash": cash,
        "params": {
            "engine": "event",
            "initial_cash": initial_cash,
            "commission": cm.commission,
            "slippage": cm.slippage,
            "stamp_duty": cm.stamp_duty,
            "min_commission": cm.min_commission,
            "t_plus_1": t_plus_1,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "trailing_stop": trailing_stop,
            "cost_model": cm.to_dict(),
        },
    }


def run(
    df: pd.DataFrame,
    signal: pd.Series,
    engine: str = "vector",
    **kw,
) -> dict:
    fn = backtest_vector if engine == "vector" else backtest_event
    return fn(df, signal, **kw)
