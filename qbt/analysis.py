"""交易级分析与滚动绩效。

净值曲线能骗人（一个幸运的大单可以撑起整条曲线），逐笔分析不能。
本模块把「怎么赚的」拆开看：

- MAE / MFE：每笔交易的最大不利/有利偏移 —— 判断止损止盈设得对不对
- SQN（System Quality Number）：Van Tharp 的策略质量分，对样本量敏感
- 暴露时间 / 持仓时长 / 连续亏损：资金效率与心理承受
- 滚动夏普 / 滚动回撤：绩效是不是只来自某一段时间
- 年度 × 月度热力：季节性

参考：backtesting.py（kroitor）、pyfolio / Quantopian tear sheet、Van Tharp。
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .metrics import _annual_factor, drawdown_series, max_drawdown


def _loc(index: pd.Index, day) -> int:
    """把日期映射到 bar 下标（找不到就取最近的前一个）。"""
    try:
        ts = pd.Timestamp(day)
    except Exception:
        return -1
    idx = pd.DatetimeIndex(index)
    if len(idx) == 0:
        return -1
    pos = idx.searchsorted(ts)
    return int(min(pos, len(idx) - 1))


def analyze_trades(trades: list, df: pd.DataFrame | None = None) -> dict:
    """逐笔交易分析。df 提供时可计算 MAE / MFE（需要 high/low 列）。"""
    if not trades:
        return {"count": 0}

    def pnl_of(t: dict) -> float:
        v = t.get("pnl")
        if v is not None:
            return float(v)
        r = float(t.get("return") or 0.0)
        return r

    pnls = [pnl_of(t) for t in trades]
    arr = np.asarray(pnls, dtype=float)
    wins = arr[arr > 0]
    losses = arr[arr <= 0]

    mean = float(arr.mean())
    sd = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    sqn = float(math.sqrt(len(arr)) * mean / sd) if sd > 1e-12 else 0.0

    # 连续亏损
    max_lose_streak = cur = 0
    max_win_streak = curw = 0
    for p in pnls:
        if p <= 0:
            cur += 1
            curw = 0
        else:
            cur = 0
            curw += 1
        max_lose_streak = max(max_lose_streak, cur)
        max_win_streak = max(max_win_streak, curw)

    # 持仓时长 / 暴露
    bars = [int(t.get("bars") or 0) for t in trades]
    total_bars = sum(bars)
    exposure = None
    if df is not None and len(df) > 0:
        exposure = float(min(total_bars / len(df), 1.0))

    # MAE / MFE
    mae_list, mfe_list, edge_list = [], [], []
    if df is not None and {"high", "low"} <= set(df.columns):
        hi = df["high"].to_numpy(dtype=float)
        lo = df["low"].to_numpy(dtype=float)
        for t in trades:
            i0 = _loc(df.index, t.get("entry_date"))
            i1 = _loc(df.index, t.get("exit_date"))
            if i0 < 0 or i1 < 0 or i1 <= i0:
                continue
            ep = float(t.get("entry_price") or 0.0)
            if ep <= 0:
                continue
            seg_hi = hi[i0 : i1 + 1]
            seg_lo = lo[i0 : i1 + 1]
            mfe = float(seg_hi.max() / ep - 1.0)
            mae = float(seg_lo.min() / ep - 1.0)   # 负数
            mae_list.append(mae)
            mfe_list.append(mfe)
            if abs(mae) > 1e-9:
                edge_list.append(mfe / abs(mae))

    # 按出场原因分组
    by_reason: dict[str, list] = {}
    for t, p in zip(trades, pnls):
        by_reason.setdefault(str(t.get("reason") or "signal"), []).append(p)
    reason_stats = {
        k: {
            "count": len(v),
            "sum": float(np.sum(v)),
            "mean": float(np.mean(v)),
            "win_rate": float(np.mean([1.0 if x > 0 else 0.0 for x in v])),
        }
        for k, v in by_reason.items()
    }

    out = {
        "count": len(trades),
        "expectancy": mean,
        "std": sd,
        "sqn": sqn,
        "sqn_grade": _sqn_grade(sqn),
        "best": float(arr.max()),
        "worst": float(arr.min()),
        "median": float(np.median(arr)),
        "win_rate": float(len(wins) / len(arr)) if len(arr) else 0.0,
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "payoff_ratio": float(abs(wins.mean() / losses.mean())) if len(wins) and losses.mean() != 0 else None,
        "gross_profit": float(wins.sum()) if len(wins) else 0.0,
        "gross_loss": float(losses.sum()) if len(losses) else 0.0,
        "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else None,
        "max_consecutive_losses": max_lose_streak,
        "max_consecutive_wins": max_win_streak,
        "avg_hold_bars": float(np.mean(bars)) if bars else 0.0,
        "max_hold_bars": int(max(bars)) if bars else 0,
        "exposure_time": exposure,
        "by_reason": reason_stats,
        "pnl_list": [float(x) for x in pnls],
    }

    if mae_list:
        out.update(
            {
                "mae_mean": float(np.mean(mae_list)),
                "mae_worst": float(np.min(mae_list)),
                "mfe_mean": float(np.mean(mfe_list)),
                "mfe_best": float(np.max(mfe_list)),
                "edge_ratio": float(np.mean(edge_list)) if edge_list else None,
                "pct_trades_mae_over_10pct": float(np.mean([1.0 if m < -0.10 else 0.0 for m in mae_list])),
            }
        )
    return out


def _sqn_grade(sqn: float) -> str:
    if sqn >= 3.0:
        return "优异（>3.0，可考虑实盘）"
    if sqn >= 2.0:
        return "良好（2.0–3.0）"
    if sqn >= 1.6:
        return "可用（1.6–2.0）"
    if sqn >= 0.0:
        return "偏弱（0–1.6，样本不足或逻辑不强）"
    return "不合格（<0）"


def rolling_metrics(equity: pd.Series, window: int = 63) -> dict:
    """滚动夏普 / 滚动波动 / 滚动最大回撤 / 滚动年化收益。"""
    eq = pd.Series(equity).astype(float).dropna()
    if len(eq) < window + 5:
        return {"error": f"样本不足以计算 {window} 日滚动指标"}
    af = _annual_factor(eq.index)
    ret = eq.pct_change().dropna()

    roll_sharpe = (
        ret.rolling(window).mean() / ret.rolling(window).std(ddof=1) * math.sqrt(af)
    )
    roll_vol = ret.rolling(window).std(ddof=1) * math.sqrt(af)
    roll_ret = (eq / eq.shift(window) - 1.0)
    roll_dd = eq.rolling(window, min_periods=window).apply(
        lambda x: max_drawdown(pd.Series(x)), raw=False
    )

    def pack(s: pd.Series) -> dict:
        s = s.dropna()
        return {
            "dates": [str(d)[:10] for d in s.index],
            "values": [None if (v is None or not np.isfinite(v)) else float(v) for v in s.to_numpy()],
        }

    return {
        "window": window,
        "sharpe": pack(roll_sharpe),
        "volatility": pack(roll_vol),
        "return": pack(roll_ret),
        "max_drawdown": pack(roll_dd),
        "summary": {
            "sharpe_median": float(np.nanmedian(roll_sharpe.dropna())) if roll_sharpe.notna().any() else None,
            "sharpe_pct_positive": float(np.nanmean(roll_sharpe.dropna() > 0)) if roll_sharpe.notna().any() else None,
            "sharpe_min": float(np.nanmin(roll_sharpe.dropna())) if roll_sharpe.notna().any() else None,
            "sharpe_max": float(np.nanmax(roll_sharpe.dropna())) if roll_sharpe.notna().any() else None,
        },
    }


def yearly_returns(equity: pd.Series, benchmark: pd.Series | None = None) -> list[dict]:
    """分年度收益（含与基准对比）。"""
    eq = pd.Series(equity).astype(float).dropna()
    if len(eq) < 2:
        return []
    first = eq.index[0]
    years = []
    for y, grp in eq.groupby(eq.index.year):
        s = grp.iloc[-1]
        # 年初基准：上一年最后一个值（第一年用首值）
        prev = eq[eq.index < grp.index[0]]
        base = prev.iloc[-1] if len(prev) else grp.iloc[0]
        if base <= 0:
            continue
        row = {"year": int(y), "return": float(s / base - 1.0), "bars": int(len(grp))}
        if benchmark is not None and len(benchmark) > 1:
            b = pd.Series(benchmark).astype(float).dropna()
            bg = b[b.index.year == y]
            if len(bg):
                bprev = b[b.index < bg.index[0]]
                bb = bprev.iloc[-1] if len(bprev) else bg.iloc[0]
                row["benchmark"] = float(bg.iloc[-1] / bb - 1.0) if bb > 0 else None
                row["excess"] = (
                    row["return"] - row["benchmark"] if row["benchmark"] is not None else None
                )
        dd = drawdown_series(grp)
        row["max_drawdown"] = float(dd.min()) if len(dd) else None
        years.append(row)
    return years


def monthly_table(equity: pd.Series) -> dict:
    """年 × 月 收益矩阵（用于热力图）。"""
    eq = pd.Series(equity).astype(float).dropna()
    if len(eq) < 2:
        return {"years": [], "months": list(range(1, 13)), "matrix": []}
    m = eq.resample("ME").last().pct_change().dropna()
    if len(m) == 0:
        return {"years": [], "months": list(range(1, 13)), "matrix": []}
    years = sorted({d.year for d in m.index})
    matrix = []
    for y in years:
        row = []
        for mo in range(1, 13):
            v = m[(m.index.year == y) & (m.index.month == mo)]
            row.append(float(v.iloc[0]) if len(v) else None)
        # 年度合计
        ye = eq[eq.index.year == y]
        if len(ye):
            prev = eq[eq.index < ye.index[0]]
            base = prev.iloc[-1] if len(prev) else ye.iloc[0]
            row.append(float(ye.iloc[-1] / base - 1.0) if base > 0 else None)
        else:
            row.append(None)
        matrix.append(row)
    return {"years": [int(y) for y in years], "months": list(range(1, 13)), "matrix": matrix}


def distribution_stats(equity: pd.Series) -> dict:
    """收益分布诊断：尾部、正态性、稳定性。"""
    eq = pd.Series(equity).astype(float).dropna()
    if len(eq) < 10:
        return {}
    ret = eq.pct_change().dropna().to_numpy(dtype=float)
    if len(ret) < 10:
        return {}
    af = _annual_factor(eq.index)

    var95 = float(np.percentile(ret, 5))
    cvar95 = float(ret[ret <= var95].mean()) if (ret <= var95).any() else var95
    tail_ratio = float(abs(np.percentile(ret, 95) / np.percentile(ret, 5))) if np.percentile(ret, 5) != 0 else None

    # Omega(0)
    omega = float(ret[ret > 0].sum() / abs(ret[ret < 0].sum())) if (ret < 0).any() else None

    # 稳定性：累计对数收益对线性趋势的 R²
    logc = np.log(eq / eq.iloc[0]).to_numpy()
    x = np.arange(len(logc))
    if len(logc) > 3 and logc.std() > 0:
        slope, intercept = np.polyfit(x, logc, 1)
        fit = slope * x + intercept
        ss_res = float(((logc - fit) ** 2).sum())
        ss_tot = float(((logc - logc.mean()) ** 2).sum())
        stability = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
    else:
        stability = None

    mdd = max_drawdown(eq)
    total = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    return {
        "var_95_daily": var95,
        "cvar_95_daily": cvar95,
        "var_95_annual": float(var95 * math.sqrt(af)),
        "tail_ratio": tail_ratio,
        "omega": omega,
        "skewness": float(pd.Series(ret).skew()),
        "kurtosis": float(pd.Series(ret).kurt()),
        "stability": stability,
        "recovery_factor": float(total / abs(mdd)) if mdd < -1e-9 else None,
        "ulcer_index": float(np.sqrt((drawdown_series(eq).clip(upper=0) ** 2).mean()) * 100),
        "pct_bars_in_drawdown": float(np.mean(drawdown_series(eq) < -1e-9)),
        "longest_drawdown_bars": int(_longest_dd_bars(eq)),
    }


def _longest_dd_bars(eq: pd.Series) -> int:
    dd = drawdown_series(eq)
    longest = cur = 0
    for v in dd.to_numpy():
        if v < -1e-9:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 0
    return longest
