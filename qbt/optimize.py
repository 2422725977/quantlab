"""参数扫描与样本外检验（Walk-Forward）。"""

from __future__ import annotations

import itertools
import math

import pandas as pd

from .engine import backtest_event, backtest_vector
from .metrics import compute_metrics
from .strategies import get_strategy

DEFAULT_METRIC = "sharpe"


def _run(df: pd.DataFrame, key: str, params: dict, engine: str, bt_kw: dict) -> dict:
    meta = get_strategy(key)
    sig = meta["fn"](df, **params)
    fn = backtest_vector if engine == "vector" else backtest_event
    res = fn(df, sig, **bt_kw)
    m = compute_metrics(res["equity"], res.get("trades"))
    return {"params": params, "metrics": m, "result": res}


def _score(m: dict, metric: str) -> float:
    v = m.get(metric)
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return -1e9
    return float(v)


def param_combinations(grid: dict) -> list[dict]:
    keys = [k for k, v in grid.items() if v]
    if not keys:
        return [{}]
    values = [list(grid[k]) for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def grid_search(
    df: pd.DataFrame,
    strategy_key: str,
    grid: dict | None = None,
    *,
    engine: str = "vector",
    metric: str = DEFAULT_METRIC,
    top: int = 20,
    min_trades: int = 0,
    **bt_kw,
) -> dict:
    """在给定参数网格上全组合回测，按 metric 排序。"""
    meta = get_strategy(strategy_key)
    grid = grid or meta["grid"]
    combos = param_combinations(grid)
    rows = []
    for p in combos:
        full = {**meta["params"], **p}
        try:
            out = _run(df, strategy_key, full, engine, bt_kw)
        except Exception:
            continue
        m = out["metrics"]
        if m.get("trades", 0) < min_trades:
            continue
        rows.append({**p, **{k: v for k, v in m.items() if k != "monthly_returns"}})
    if not rows:
        return {"rows": [], "best": None, "metric": metric, "count": 0}
    res = pd.DataFrame(rows)
    res = res.sort_values(metric, ascending=False)
    return {
        "rows": res.head(top).to_dict(orient="records"),
        "all": res.to_dict(orient="records"),
        "best": res.iloc[0].to_dict(),
        "metric": metric,
        "count": len(res),
    }


def walk_forward(
    df: pd.DataFrame,
    strategy_key: str,
    grid: dict | None = None,
    *,
    n_splits: int = 5,
    train_ratio: float = 0.7,
    anchored: bool = False,
    engine: str = "vector",
    metric: str = DEFAULT_METRIC,
    **bt_kw,
) -> dict:
    """滚动前推检验。

    anchored=False → 滚动窗口：每折的训练集是「该折前 train_ratio 的数据」；
    anchored=True  → 锚定：训练集从数据起点开始，样本量逐折增大。
    返回每折的样本内最优参数 + 样本外真实绩效，以及拼接后的样本外净值曲线。
    """
    n = len(df)
    n_splits = max(2, int(n_splits))
    fold_size = max(30, n // n_splits)
    folds = []
    for i in range(n_splits):
        start = i * fold_size
        end = n if i == n_splits - 1 else min((i + 1) * fold_size, n)
        if start >= end:
            continue
        train_end = start + int((end - start) * train_ratio)
        train_df = df.iloc[:train_end] if anchored else df.iloc[start:train_end]
        test_df = df.iloc[train_end:end]
        if len(train_df) > 40 and len(test_df) > 10:
            folds.append((train_df, test_df, end))

    results = []
    oos_equity = []
    scale = 1.0
    for i, (tr, te, _) in enumerate(folds):
        gs = grid_search(tr, strategy_key, grid, engine=engine, metric=metric, **bt_kw)
        if not gs.get("best"):
            continue
        best_p = {k: gs["best"][k] for k in get_strategy(strategy_key)["params"]}
        out = _run(te, strategy_key, best_p, engine, bt_kw)
        eq = out["result"]["equity"]
        seg = eq / eq.iloc[0] * scale
        oos_equity.append(seg)
        scale = float(seg.iloc[-1])
        results.append(
            {
                "fold": i + 1,
                "train_start": str(tr.index[0])[:10],
                "train_end": str(tr.index[-1])[:10],
                "test_start": str(te.index[0])[:10],
                "test_end": str(te.index[-1])[:10],
                "best_params": best_p,
                "in_sample": {k: v for k, v in gs["best"].items() if k not in best_p},
                "oos": {k: v for k, v in out["metrics"].items() if k != "monthly_returns"},
            }
        )

    combined = pd.concat(oos_equity) if oos_equity else pd.Series(dtype=float)
    oos_metrics = compute_metrics(combined) if len(combined) > 2 else {}
    full = _run(df, strategy_key, get_strategy(strategy_key)["params"], engine, bt_kw)

    return {
        "folds": results,
        "oos_equity": combined,
        "oos_metrics": {k: v for k, v in oos_metrics.items() if k != "monthly_returns"},
        "full_metrics": {k: v for k, v in full["metrics"].items() if k != "monthly_returns"},
        "n_splits": len(results),
    }
