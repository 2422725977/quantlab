"""绩效指标计算。"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _clean(v):
    """NaN → 0.0，±inf → None（JSON 不支持 inf，前端显示 —）。"""
    if isinstance(v, float):
        if math.isnan(v):
            return 0.0
        if math.isinf(v):
            return None
        return v
    return v


def max_drawdown(equity: pd.Series) -> float:
    """最大回撤，返回负数（如 -0.35 表示 -35%）。"""
    if len(equity) == 0:
        return 0.0
    peak = equity.cummax()
    return float((equity / peak - 1.0).min())


def drawdown_series(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity / peak - 1.0


def _annual_factor(index: pd.Index) -> float:
    """根据数据频率推断每年的bar数量。"""
    if len(index) < 3:
        return TRADING_DAYS
    try:
        idx = pd.DatetimeIndex(index)
        deltas = pd.Index(index).to_series().diff().dropna()
        med = deltas.median()
        if pd.isna(med):
            return TRADING_DAYS
        secs = med.total_seconds()
        if secs <= 0:
            return TRADING_DAYS
        if secs <= 1.6 * 86400:                 # 日线：区分交易日历(252) 与 7x24(365)
            weekend = float((idx.dayofweek >= 5).mean())
            return 365.0 if weekend > 0.10 else TRADING_DAYS
        per_year = 365 * 24 * 3600 / secs
        if secs <= 9 * 86400:                   # 周线
            return 52.0
        if per_year > 200:                      # 分钟/小时级数据
            return float(min(per_year, 252 * 6.5 * 60))
        return 12.0                             # 月线
    except Exception:
        return TRADING_DAYS


def compute_metrics(
    equity: pd.Series,
    trades: list | None = None,
    benchmark: pd.Series | None = None,
    risk_free: float = 0.0,
) -> dict:
    """由净值曲线计算全套绩效指标。

    equity  : 净值曲线（Series，index 为日期）
    trades  : 成交记录列表，每项含 pnl / entry_date / exit_date 等
    benchmark: 基准净值曲线（可选，用于计算超额与 Beta）
    """
    equity = pd.Series(equity).astype(float).dropna()
    if len(equity) < 2:
        return {"error": "数据不足"}

    ret = equity.pct_change().dropna()
    n = len(ret)
    years = n / _annual_factor(equity.index) if _annual_factor(equity.index) else n / TRADING_DAYS
    years = max(years, 1e-9)

    total_ret = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1.0) if equity.iloc[0] > 0 else 0.0
    vol = float(ret.std(ddof=1) * np.sqrt(_annual_factor(equity.index)))

    rf_per_bar = risk_free / _annual_factor(equity.index)
    excess = ret - rf_per_bar
    sharpe = float(excess.mean() / excess.std(ddof=1) * np.sqrt(_annual_factor(equity.index))) if excess.std(ddof=1) > 0 else 0.0

    downside = excess[excess < 0]
    dd_std = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    sortino = float(excess.mean() / dd_std * np.sqrt(_annual_factor(equity.index))) if dd_std > 0 else 0.0

    mdd = max_drawdown(equity)
    calmar = float(cagr / abs(mdd)) if mdd < -1e-9 else 0.0

    dd = drawdown_series(equity)
    ulcer = float(np.sqrt((dd.clip(upper=0) ** 2).mean()) * 100)

    # 月度收益
    try:
        monthly = (equity.resample("ME").last().pct_change().dropna())
        monthly_ret = {
            f"{d.year}-{d.month:02d}": float(v)
            for d, v in monthly.items()
        }
    except Exception:
        monthly_ret = {}

    out = {
        "total_return": total_ret,
        "cagr": cagr,
        "volatility": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "calmar": calmar,
        "ulcer_index": ulcer,
        "bars": n,
        "years": years,
        "best_bar": float(ret.max()),
        "worst_bar": float(ret.min()),
        "monthly_returns": monthly_ret,
    }

    if trades:
        # 事件驱动引擎给绝对盈亏 pnl；向量化引擎只给收益率 return。二选一，口径统一。
        use_pnl = all(t.get("pnl") is not None for t in trades)
        key = "pnl" if use_pnl else "return"
        pnls = [float(t.get(key, 0.0) or 0.0) for t in trades]
        out["pnl_basis"] = "absolute" if use_pnl else "return"
        if pnls:
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p <= 0]
            gp = sum(wins)
            gl = abs(sum(losses))
            out.update(
                {
                    "trades": len(pnls),
                    "win_rate": len(wins) / len(pnls),
                    "avg_pnl": float(np.mean(pnls)),
                    "avg_win": float(np.mean(wins)) if wins else 0.0,
                    "avg_loss": float(np.mean(losses)) if losses else 0.0,
                    "profit_factor": float(gp / gl) if gl > 1e-12 else float("inf"),
                    "expectancy": float(np.mean(pnls)),
                }
            )
        else:
            out.update({"trades": 0, "win_rate": 0.0, "profit_factor": 0.0})
    else:
        out.update({"trades": 0, "win_rate": 0.0, "profit_factor": 0.0})

    if benchmark is not None and len(benchmark) > 1:
        b = pd.Series(benchmark).astype(float)
        b = b.reindex(equity.index).ffill().dropna()
        if len(b) > 1:
            br = b.pct_change().dropna()
            joined = pd.concat([ret, br], axis=1).dropna()
            if len(joined) > 2:
                x = joined.iloc[:, 1]
                y = joined.iloc[:, 0]
                var_x = x.var(ddof=1)
                beta = float(y.cov(x) / var_x) if var_x > 0 else 0.0
                alpha = float((y.mean() - beta * x.mean()) * _annual_factor(equity.index))
                out["beta"] = beta
                out["alpha"] = alpha
                out["benchmark_return"] = float(b.iloc[-1] / b.iloc[0] - 1.0)
                out["excess_return"] = total_ret - out["benchmark_return"]

    out = {k: _clean(v) for k, v in out.items()}
    out["monthly_returns"] = {k: _clean(v) for k, v in monthly_ret.items()}
    return out
