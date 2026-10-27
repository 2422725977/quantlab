"""FastAPI 服务：为 Web 工作台提供回测 / 扫描 / 样本外检验 / 报告接口。"""

from __future__ import annotations

import math
import os
import traceback
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import data as data_mod
from . import report as report_mod
from .engine import backtest_event, backtest_vector
from .metrics import compute_metrics, drawdown_series
from .optimize import grid_search, walk_forward
from .strategies import get_strategy, list_strategies

WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")

app = FastAPI(title="QuantLab", version="0.1.0")

_DATA_CACHE: dict[str, pd.DataFrame] = {}
MAX_POINTS = 4000


# --------------------------------------------------------------------------
# 请求模型
# --------------------------------------------------------------------------
class DataReq(BaseModel):
    symbol: str
    market: str = "A"
    start: str = "2018-01-01"
    end: str | None = None
    exchange: str = "binance"
    use_cache: bool = True
    csv_path: str | None = None


class BacktestReq(DataReq):
    strategy: str = "ma_cross"
    params: dict[str, Any] = Field(default_factory=dict)
    engine: str = "vector"
    initial_cash: float = 1_000_000.0
    commission: float = 0.0005
    slippage: float = 0.0005
    allow_short: bool = False
    t_plus_1: bool = False
    stop_loss: float | None = None
    take_profit: float | None = None
    trailing_stop: float | None = None
    benchmark: bool = True


class OptimizeReq(BacktestReq):
    grid: dict[str, list] | None = None
    metric: str = "sharpe"
    top: int = 30
    min_trades: int = 0


class WalkForwardReq(BacktestReq):
    grid: dict[str, list] | None = None
    metric: str = "sharpe"
    n_splits: int = 5
    train_ratio: float = 0.7
    anchored: bool = False


# --------------------------------------------------------------------------
# 工具
# --------------------------------------------------------------------------
def _load(req: DataReq) -> pd.DataFrame:
    if req.csv_path:
        return data_mod.load_csv(req.csv_path)
    key = f"{req.market}|{req.symbol}|{req.start}|{req.end}|{req.exchange}"
    if key in _DATA_CACHE:
        return _DATA_CACHE[key]
    df = data_mod.fetch(
        req.symbol, req.market, req.start, req.end,
        use_cache=req.use_cache, exchange=req.exchange,
    )
    if df.empty:
        raise HTTPException(400, f"未获取到数据：{req.market}:{req.symbol}")
    _DATA_CACHE[key] = df
    return df


def _bt_kwargs(req: BacktestReq) -> dict:
    """按引擎类型裁剪参数：止损/止盈/T+1 只有事件驱动引擎支持。"""
    base = dict(
        initial_cash=req.initial_cash,
        commission=req.commission,
        slippage=req.slippage,
        allow_short=req.allow_short,
    )
    if req.engine == "event":
        base.update(
            t_plus_1=req.t_plus_1,
            stop_loss=req.stop_loss,
            take_profit=req.take_profit,
            trailing_stop=req.trailing_stop,
        )
    return base


def _slice(df: pd.DataFrame) -> tuple[list[str], list[float]]:
    step = max(1, len(df) // MAX_POINTS)
    s = df.iloc[::step]
    return [str(d)[:10] for d in s.index], [round(float(v), 4) for v in s["close"]]


def _json_safe(obj):
    """递归替换 NaN/±inf，避免 JSON 序列化报 'Out of range float'。"""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj):
            return 0.0
        if math.isinf(obj):
            return None
        return obj
    return obj


def _pack(res: dict, metrics: dict) -> dict:
    eq = res["equity"]
    dd = drawdown_series(eq)
    step = max(1, len(eq) // MAX_POINTS)
    return {
        "dates": [str(d)[:10] for d in eq.index[::step]],
        "equity": [round(float(v), 2) for v in eq.iloc[::step]],
        "drawdown": [round(float(v) * 100, 2) for v in dd.iloc[::step]],
        "position": [round(float(v), 3) for v in res["position"].iloc[::step]],
        "trades": res.get("trades", [])[:500],
        "metrics": metrics,
        "params": res.get("params", {}),
    }


# --------------------------------------------------------------------------
# 路由
# --------------------------------------------------------------------------
@app.get("/api/strategies")
def api_strategies():
    return list_strategies()


@app.get("/api/symbols")
def api_symbols(q: str = "", market: str = "A"):
    try:
        return {"items": data_mod.search_symbol(q, market)}
    except Exception as e:  # noqa: BLE001
        return {"items": [], "error": str(e)}


@app.post("/api/data")
def api_data(req: DataReq):
    try:
        df = _load(req)
        dates, close = _slice(df)
        return {
            "symbol": req.symbol,
            "market": req.market,
            "rows": len(df),
            "start": str(df.index[0])[:10],
            "end": str(df.index[-1])[:10],
            "dates": dates,
            "close": close,
        }
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"{e}")


@app.post("/api/backtest")
def api_backtest(req: BacktestReq):
    try:
        df = _load(req)
        meta = get_strategy(req.strategy)
        params = {**meta["params"], **req.params}
        sig = meta["fn"](df, **params)

        fn = backtest_vector if req.engine == "vector" else backtest_event
        res = fn(df, sig, **_bt_kwargs(req))

        bench = None
        if req.benchmark and req.strategy != "buy_hold":
            bh = backtest_vector(
                df, pd.Series(1.0, index=df.index),
                initial_cash=req.initial_cash, commission=req.commission, slippage=req.slippage,
            )
            bench = bh["equity"]

        m = compute_metrics(res["equity"], res.get("trades"), benchmark=bench)
        out = _pack(res, m)
        d, cl = _slice(df)
        out["price"] = {"dates": d, "close": cl}
        out["strategy"] = {"key": req.strategy, "name": meta["name"], "params": params}
        out["symbol"] = f"{req.market}:{req.symbol}"
        return _json_safe(out)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        raise HTTPException(400, f"{e}")


@app.post("/api/optimize")
def api_optimize(req: OptimizeReq):
    try:
        df = _load(req)
        gs = grid_search(
            df, req.strategy, req.grid, engine=req.engine,
            metric=req.metric, top=req.top, min_trades=req.min_trades, **_bt_kwargs(req),
        )
        return _json_safe({
            "metric": gs["metric"],
            "count": gs["count"],
            "rows": gs.get("rows", []),
            "best": gs.get("best"),
            "strategy": req.strategy,
        })
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        raise HTTPException(400, f"{e}")


@app.post("/api/walkforward")
def api_walkforward(req: WalkForwardReq):
    try:
        df = _load(req)
        wf = walk_forward(
            df, req.strategy, req.grid, n_splits=req.n_splits,
            train_ratio=req.train_ratio, anchored=req.anchored,
            engine=req.engine, metric=req.metric, **_bt_kwargs(req),
        )
        eq = wf["oos_equity"]
        step = max(1, len(eq) // MAX_POINTS) if len(eq) else 1
        return _json_safe({
            "folds": wf["folds"],
            "oos_metrics": wf["oos_metrics"],
            "full_metrics": wf["full_metrics"],
            "oos_curve": {
                "dates": [str(d)[:10] for d in eq.index[::step]] if len(eq) else [],
                "equity": [round(float(v), 2) for v in eq.iloc[::step]] if len(eq) else [],
            },
        })
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        raise HTTPException(400, f"{e}")


@app.post("/api/report")
def api_report(req: BacktestReq):
    """生成可下载的独立 HTML 报告。"""
    try:
        df = _load(req)
        meta = get_strategy(req.strategy)
        params = {**meta["params"], **req.params}
        sig = meta["fn"](df, **params)
        fn = backtest_vector if req.engine == "vector" else backtest_event
        res = fn(df, sig, **_bt_kwargs(req))
        bench = None
        if req.strategy != "buy_hold":
            bh = backtest_vector(
                df, pd.Series(1.0, index=df.index),
                initial_cash=req.initial_cash, commission=req.commission, slippage=req.slippage,
            )
            bench = bh["equity"]
        m = compute_metrics(res["equity"], res.get("trades"), benchmark=bench)
        html = report_mod.build_html(
            f"{req.market}:{req.symbol}", meta["name"], params, res, m,
            benchmark=bench,
        )
        return HTMLResponse(content=html)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        raise HTTPException(400, f"{e}")


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")


def serve(host: str = "127.0.0.1", port: int = 8765, reload: bool = False):
    import uvicorn

    uvicorn.run("qbt.server:app", host=host, port=port, reload=reload)
