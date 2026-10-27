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
from .analysis import (
    analyze_trades,
    distribution_stats,
    monthly_table,
    rolling_metrics,
    yearly_returns,
)
from .costs import CostModel, list_presets
from .engine import backtest_event, backtest_vector
from .metrics import compute_metrics, drawdown_series
from .optimize import grid_search, walk_forward
from .portfolio import (
    allocate,
    backtest_portfolio,
    build_panel,
    compare_weights,
    correlation_matrix,
    cross_sectional_momentum,
    open_to_open_returns,
)
from .robustness import (
    bootstrap_confidence,
    deflated_sharpe,
    param_plateau,
    random_strategy_test,
)
from .strategies import get_strategy, list_strategies

WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")

app = FastAPI(title="QuantLab", version="0.2.0")

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
    cost_market: str = ""        # A / HK / US / CRYPTO / IDEAL；空=用上面手填的费率
    slippage_mode: str = "pct"   # pct | volume | spread
    volume_impact: float = 0.10


class RobustReq(BacktestReq):
    metric: str = "sharpe"
    n_boot: int = 800
    block: int = 20
    n_iter: int = 300
    n_trials: int = 200
    random_method: str = "markov"
    do_bootstrap: bool = True
    do_random: bool = True
    do_plateau: bool = True
    do_dsr: bool = True


class PortfolioReq(BaseModel):
    symbols: list[str] = Field(default_factory=lambda: ["01810", "00700"])
    market: str = "HK"
    start: str = "2021-01-01"
    end: str | None = None
    mode: str = "momentum"       # momentum=截面动量轮动 / hold=全标的持有
    weighting: str = "equal"     # equal / inv_vol / risk_parity
    lookback: int = 60
    top_k: int = 2
    skip_recent: int = 5
    rebalance: str = "ME"        # D / W / ME / QE
    cost_market: str = "HK"
    initial_cash: float = 1_000_000.0
    max_weight: float = 0.6
    compare_all: bool = True


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


def _cost(req) -> CostModel | None:
    """按市场套用真实费率（印花税 / 最低佣金 / 成交量冲击滑点）。"""
    market = getattr(req, "cost_market", "") or ""
    if not market:
        return None
    return CostModel.for_market(
        market,
        slippage_mode=getattr(req, "slippage_mode", "pct") or "pct",
        volume_impact=getattr(req, "volume_impact", 0.10),
    )


def _bt_kwargs(req: BacktestReq) -> dict:
    """按引擎类型裁剪参数：止损/止盈/T+1 只有事件驱动引擎支持。"""
    base = dict(
        initial_cash=req.initial_cash,
        commission=req.commission,
        slippage=req.slippage,
        allow_short=req.allow_short,
        cost=_cost(req),
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

        # 诊断层：逐笔分析 / 滚动绩效 / 收益分布 / 分年月度
        eq = res["equity"]
        trades = res.get("trades", [])
        try:
            out["analysis"] = analyze_trades(trades, df)
            out["distribution"] = distribution_stats(eq)
            out["yearly"] = yearly_returns(eq, bench)
            out["monthly_table"] = monthly_table(eq)
            win = max(20, min(63, len(eq) // 12))
            out["rolling"] = rolling_metrics(eq, window=win)
        except Exception as e:  # noqa: BLE001
            out["diagnostics_error"] = str(e)
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


@app.get("/api/costs")
def api_costs():
    """各市场的真实费率预设 + 往返成本示例。"""
    items = []
    for p in list_presets():
        cm = CostModel.for_market(p["market"])
        rt = cm.round_trip_cost(100_000.0)
        items.append({**p, "round_trip_rate": rt["round_trip_rate"], **cm.to_dict()})
    return _json_safe({"items": items})


@app.post("/api/robust")
def api_robust(req: RobustReq):
    """稳健性四件套：自助置信区间 / 随机策略对照 / 通缩夏普 / 参数高原。"""
    try:
        df = _load(req)
        meta = get_strategy(req.strategy)
        params = {**meta["params"], **req.params}
        sig = meta["fn"](df, **params)
        fn = backtest_vector if req.engine == "vector" else backtest_event
        res = fn(df, sig, **_bt_kwargs(req))
        eq = res["equity"]
        metrics = compute_metrics(eq, res.get("trades"))

        out: dict = {
            "strategy": {"key": req.strategy, "name": meta["name"], "params": params},
            "real_metrics": {k: v for k, v in metrics.items() if k != "monthly_returns"},
        }
        if req.do_bootstrap:
            out["bootstrap"] = bootstrap_confidence(
                eq, n_boot=req.n_boot, block=req.block
            )
        if req.do_random:
            # 随机对照要跑几百次，固定用向量化引擎；事件驱动专属参数需剔除
            vec_kw = {
                k: v for k, v in _bt_kwargs(req).items()
                if k not in ("t_plus_1", "stop_loss", "take_profit", "trailing_stop")
            }
            out["random_test"] = random_strategy_test(
                df, sig, n_iter=req.n_iter, method=req.random_method,
                engine="vector", metric=req.metric, **vec_kw,
            )
        if req.do_dsr:
            sharpe = float(metrics.get("sharpe") or 0.0)
            out["deflated_sharpe"] = deflated_sharpe(
                sharpe, n_trials=req.n_trials, n_obs=len(eq)
            )
        if req.do_plateau:
            out["plateau"] = param_plateau(
                df, req.strategy, params, engine="vector", metric=req.metric,
                **_bt_kwargs(req),
            )
        return _json_safe(out)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        raise HTTPException(400, f"{e}")


@app.post("/api/portfolio")
def api_portfolio(req: PortfolioReq):
    """多标的组合回测：截面动量轮动 / 全持有，支持三种加权方式对比。"""
    try:
        if len(req.symbols) < 2:
            raise HTTPException(400, "至少需要 2 个标的")
        data = {}
        failed = []
        for s in req.symbols:
            try:
                d = data_mod.fetch(s, req.market, req.start, req.end, use_cache=True)
                if len(d) > 30:
                    data[s] = d
                else:
                    failed.append(f"{s}(数据过少)")
            except Exception as e:  # noqa: BLE001
                failed.append(f"{s}({e})")
        if len(data) < 2:
            raise HTTPException(400, f"可用标的不足：{'; '.join(failed)}")

        panel = build_panel(data, "open")
        rets = open_to_open_returns(panel)
        close_panel = build_panel(data, "close")
        cost = CostModel.for_market(req.cost_market) if req.cost_market else None

        if req.mode == "momentum":
            target = cross_sectional_momentum(
                close_panel, lookback=req.lookback, top_k=req.top_k,
                skip_recent=req.skip_recent, rebalance=req.rebalance,
            )
        else:
            target = pd.DataFrame(1.0, index=rets.index, columns=rets.columns)

        weights = allocate(target, rets, method=req.weighting, max_weight=req.max_weight)
        res = backtest_portfolio(
            rets, weights, initial_cash=req.initial_cash,
            rebalance=req.rebalance, cost=cost,
        )
        eq = res["equity"]
        step = max(1, len(eq) // MAX_POINTS)

        out = {
            "symbol": f"{req.market}:{'+'.join(data.keys())}",
            "range": {"start": str(eq.index[0])[:10], "end": str(eq.index[-1])[:10], "bars": len(eq)},
            "failed": failed,
            "dates": [str(d)[:10] for d in eq.index[::step]],
            "equity": [round(float(v), 2) for v in eq.iloc[::step]],
            "drawdown": [round(float(v) * 100, 2) for v in drawdown_series(eq).iloc[::step]],
            "metrics": res["metrics"],
            "contribution": res["contribution"],
            "correlation": correlation_matrix(rets),
            "current_weights": {
                k: round(float(v), 4) for k, v in weights.iloc[-1].items()
            },
            "params": {
                "mode": req.mode, "weighting": req.weighting,
                "lookback": req.lookback, "top_k": req.top_k,
                "rebalance": req.rebalance, "cost_market": req.cost_market,
            },
        }
        if req.compare_all:
            out["compare"] = compare_weights(
                rets, target, methods=("equal", "inv_vol", "risk_parity"),
                initial_cash=req.initial_cash, rebalance=req.rebalance, cost=cost,
                max_weight=req.max_weight,
            )
        return _json_safe(out)
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
