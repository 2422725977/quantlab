"""内置策略库。

每个策略形如 fn(df, **params) -> Series，语义是「第 t 根收盘时确定的目标仓位」，
引擎会把它顺延到 t+1 开盘成交，因此策略内部可以放心使用当日收盘价。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# 指标
# --------------------------------------------------------------------------
def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(int(n)).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=int(n), adjust=False).mean()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0)
    dn = (-d).clip(lower=0)
    ru = up.ewm(alpha=1 / int(n), adjust=False).mean()
    rd = dn.ewm(alpha=1 / int(n), adjust=False).mean()
    return 100 - 100 / (1 + ru / rd.replace(0, np.nan))


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / int(n), adjust=False).mean()


def _state_machine(entries: pd.Series, exits: pd.Series) -> pd.Series:
    """entries 触发→持仓 1；exits 触发→回到 0。向量化展开为循环。"""
    e = entries.fillna(False).to_numpy(dtype=bool)
    x = exits.fillna(False).to_numpy(dtype=bool)
    out = np.zeros(len(e), dtype=float)
    state = 0.0
    for i in range(len(e)):
        if state == 0.0 and e[i]:
            state = 1.0
        elif state == 1.0 and x[i]:
            state = 0.0
        out[i] = state
    return pd.Series(out, index=entries.index)


# --------------------------------------------------------------------------
# 策略实现
# --------------------------------------------------------------------------
def strat_buy_hold(df: pd.DataFrame, **kw) -> pd.Series:
    return pd.Series(1.0, index=df.index)


def strat_ma_cross(df: pd.DataFrame, fast: int = 20, slow: int = 60, **kw) -> pd.Series:
    c = df["close"]
    return (sma(c, fast) > sma(c, slow)).astype(float).fillna(0.0)


def strat_ma_triple(df: pd.DataFrame, fast: int = 5, mid: int = 20, slow: int = 60, **kw) -> pd.Series:
    c = df["close"]
    f, m, s = sma(c, fast), sma(c, mid), sma(c, slow)
    return ((f > m) & (m > s)).astype(float).fillna(0.0)


def strat_boll_revert(df: pd.DataFrame, period: int = 20, nstd: float = 2.0, **kw) -> pd.Series:
    c = df["close"]
    ma = sma(c, period)
    sd = c.rolling(int(period)).std(ddof=1)
    lower = ma - nstd * sd
    return _state_machine(c < lower, c > ma)


def strat_rsi_revert(df: pd.DataFrame, period: int = 14, oversold: float = 30.0,
                     overbought: float = 70.0, **kw) -> pd.Series:
    r = rsi(df["close"], period)
    return _state_machine(r < oversold, r > overbought)


def strat_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, **kw) -> pd.Series:
    c = df["close"]
    line = ema(c, fast) - ema(c, slow)
    sig = ema(line, signal)
    return (line > sig).astype(float).fillna(0.0)


def strat_donchian(df: pd.DataFrame, period: int = 20, **kw) -> pd.Series:
    hi = df["high"].rolling(int(period)).max().shift(1)
    lo = df["low"].rolling(int(period)).min().shift(1)
    c = df["close"]
    return _state_machine(c > hi, c < lo)


def strat_momentum(df: pd.DataFrame, lookback: int = 60, threshold: float = 0.0, **kw) -> pd.Series:
    c = df["close"]
    m = c / c.shift(int(lookback)) - 1.0
    return (m > threshold).astype(float).fillna(0.0)


def strat_vol_target(df: pd.DataFrame, lookback: int = 20, target_vol: float = 0.20,
                     max_w: float = 1.0, **kw) -> pd.Series:
    """波动率目标仓位：按 realized vol 缩放，震荡市自动降仓。"""
    r = df["close"].pct_change()
    vol = r.rolling(int(lookback)).std(ddof=1) * np.sqrt(252)
    w = (target_vol / vol.replace(0, np.nan)).clip(upper=max_w)
    return w.fillna(0.0)


def strat_atr_breakout(df: pd.DataFrame, period: int = 20, mult: float = 2.0, **kw) -> pd.Series:
    """ATR 通道突破：收盘价突破 均线 + k×ATR 进场，跌破均线出场。"""
    c = df["close"]
    ma = sma(c, period)
    a = atr(df, period)
    upper = ma + mult * a
    return _state_machine(c > upper, c < ma)


def strat_grid(df: pd.DataFrame, period: int = 20, bands: float = 0.05, **kw) -> pd.Series:
    """简易网格：价格相对均线每跌一格加一档仓位。"""
    c = df["close"]
    ma = sma(c, period)
    dev = (c / ma - 1.0) / float(bands)
    w = (1.0 - dev.clip(lower=-1.0, upper=3.0)) / 4.0
    return w.clip(lower=0.0, upper=1.0).fillna(0.0)


# --------------------------------------------------------------------------
# 注册表
# --------------------------------------------------------------------------
STRATEGIES: dict[str, dict] = {
    "buy_hold": {
        "name": "买入持有（基准）",
        "desc": "始终满仓，作为对比基准。",
        "fn": strat_buy_hold,
        "params": {},
        "grid": {},
    },
    "ma_cross": {
        "name": "双均线交叉",
        "desc": "短均线上穿长均线满仓，下穿空仓。最经典的趋势策略。",
        "fn": strat_ma_cross,
        "params": {"fast": 20, "slow": 60},
        "grid": {"fast": [5, 10, 20, 30], "slow": [30, 60, 90, 120]},
    },
    "ma_triple": {
        "name": "三线多头排列",
        "desc": "快>中>慢 三线多头才持仓，过滤震荡市的假突破。",
        "fn": strat_ma_triple,
        "params": {"fast": 5, "mid": 20, "slow": 60},
        "grid": {"fast": [3, 5, 10], "mid": [10, 20, 30], "slow": [40, 60, 90]},
    },
    "boll_revert": {
        "name": "布林带均值回归",
        "desc": "跌破下轨买入，回到中轨卖出。适合震荡行情。",
        "fn": strat_boll_revert,
        "params": {"period": 20, "nstd": 2.0},
        "grid": {"period": [10, 20, 30], "nstd": [1.5, 2.0, 2.5]},
    },
    "rsi_revert": {
        "name": "RSI 超卖反弹",
        "desc": "RSI 跌破超卖线买入，升破超买线卖出。",
        "fn": strat_rsi_revert,
        "params": {"period": 14, "oversold": 30.0, "overbought": 70.0},
        "grid": {"period": [7, 14, 21], "oversold": [20, 25, 30], "overbought": [65, 70, 80]},
    },
    "macd": {
        "name": "MACD 金叉死叉",
        "desc": "MACD 线上穿信号线买入，下穿卖出。",
        "fn": strat_macd,
        "params": {"fast": 12, "slow": 26, "signal": 9},
        "grid": {"fast": [8, 12, 16], "slow": [20, 26, 34], "signal": [6, 9, 12]},
    },
    "donchian": {
        "name": "唐奇安通道突破",
        "desc": "突破 N 日最高价买入，跌破 N 日最低价卖出。海龟策略核心。",
        "fn": strat_donchian,
        "params": {"period": 20},
        "grid": {"period": [10, 20, 40, 60, 90]},
    },
    "momentum": {
        "name": "动量择时",
        "desc": "过去 N 日涨幅超过阈值才持仓，否则空仓。",
        "fn": strat_momentum,
        "params": {"lookback": 60, "threshold": 0.0},
        "grid": {"lookback": [20, 40, 60, 120], "threshold": [-0.05, 0.0, 0.05, 0.10]},
    },
    "vol_target": {
        "name": "波动率目标仓位",
        "desc": "按已实现波动率反向调仓，目标年化波动 20%。震荡自动降仓。",
        "fn": strat_vol_target,
        "params": {"lookback": 20, "target_vol": 0.20, "max_w": 1.0},
        "grid": {"lookback": [10, 20, 40], "target_vol": [0.10, 0.15, 0.20, 0.30], "max_w": [1.0]},
    },
    "atr_breakout": {
        "name": "ATR 通道突破",
        "desc": "突破 均线+k×ATR 进场，跌破均线出场。波动自适应。",
        "fn": strat_atr_breakout,
        "params": {"period": 20, "mult": 2.0},
        "grid": {"period": [10, 20, 40], "mult": [1.5, 2.0, 3.0]},
    },
    "grid": {
        "name": "均线网格",
        "desc": "价格相对均线越低仓位越重，分批建仓思路。",
        "fn": strat_grid,
        "params": {"period": 20, "bands": 0.05},
        "grid": {"period": [20, 40, 60], "bands": [0.03, 0.05, 0.08]},
    },
}


def get_strategy(key: str) -> dict:
    if key not in STRATEGIES:
        raise KeyError(f"未知策略：{key}，可选：{list(STRATEGIES)}")
    return STRATEGIES[key]


def signal_for(key: str, df: pd.DataFrame, params: dict | None = None) -> pd.Series:
    meta = get_strategy(key)
    p = {**meta["params"], **(params or {})}
    return meta["fn"](df, **p)


def list_strategies() -> list[dict]:
    return [
        {"key": k, "name": v["name"], "desc": v["desc"], "params": v["params"], "grid": v["grid"]}
        for k, v in STRATEGIES.items()
    ]
