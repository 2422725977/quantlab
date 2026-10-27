#!/usr/bin/env python3
"""QuantLab 命令行入口。

  python run.py serve                     启动 Web 工作台（默认 http://127.0.0.1:8765）
  python run.py backtest --symbol 01810 --market HK --strategy ma_cross
  python run.py scan     --symbol 01810 --market HK --strategy ma_cross --metric sharpe
  python run.py wf       --symbol 01810 --market HK --strategy ma_cross --splits 5
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from qbt.data import fetch
from qbt.engine import backtest_event, backtest_vector
from qbt.metrics import compute_metrics
from qbt.optimize import grid_search, walk_forward
from qbt.strategies import get_strategy, list_strategies


def _load(args) -> pd.DataFrame:
    return fetch(args.symbol, args.market, args.start, args.end, exchange=args.exchange)


def _bt_kw(args) -> dict:
    base = dict(
        initial_cash=args.cash,
        commission=args.commission,
        slippage=args.slippage,
    )
    if args.engine == "event":
        base.update(
            t_plus_1=args.t_plus_1,
            stop_loss=args.stop_loss,
            take_profit=args.take_profit,
            trailing_stop=args.trailing_stop,
        )
    return base


def cmd_serve(args):
    from qbt.server import serve

    print(f"QuantLab → http://{args.host}:{args.port}")
    serve(host=args.host, port=args.port, reload=args.reload)


def cmd_backtest(args):
    df = _load(args)
    meta = get_strategy(args.strategy)
    params = {**meta["params"], **json.loads(args.params)} if args.params else dict(meta["params"])
    sig = meta["fn"](df, **params)
    fn = backtest_vector if args.engine == "vector" else backtest_event
    res = fn(df, sig, **_bt_kw(args))
    bench = backtest_vector(df, pd.Series(1.0, index=df.index),
                            initial_cash=args.cash, commission=args.commission, slippage=args.slippage)["equity"]
    m = compute_metrics(res["equity"], res.get("trades"), benchmark=bench)
    print(f"\n{args.market}:{args.symbol} · {meta['name']} · 参数 {params}")
    print(f"区间 {str(df.index[0])[:10]} → {str(df.index[-1])[:10]}（{len(df)} 根）\n")
    for k in ["total_return", "cagr", "sharpe", "sortino", "max_drawdown", "calmar",
              "volatility", "trades", "win_rate", "profit_factor", "excess_return"]:
        v = m.get(k)
        if v is None:
            continue
        print(f"  {k:16s} {v*100:8.2f}%" if k in ("total_return", "cagr", "max_drawdown",
                                                  "volatility", "win_rate", "excess_return")
              else f"  {k:16s} {v:9.3f}")
    if args.out:
        res["equity"].to_csv(args.out)
        print(f"\n净值曲线已写入 {args.out}")


def cmd_scan(args):
    df = _load(args)
    gs = grid_search(df, args.strategy, None, engine=args.engine, metric=args.metric,
                     top=args.top, min_trades=args.min_trades, **_bt_kw(args))
    print(f"\n扫描 {gs['count']} 组 · 排序 {args.metric}\n")
    rows = gs["rows"]
    if rows:
        cols = list(rows[0].keys())
        print("  " + "  ".join(f"{c[:12]:>12s}" for c in cols))
        for r in rows[:args.top]:
            print("  " + "  ".join(f"{(round(v,4) if isinstance(v,float) else v)!s:>12s}" for v in r.values()))


def cmd_wf(args):
    df = _load(args)
    wf = walk_forward(df, args.strategy, None, n_splits=args.splits, train_ratio=args.train_ratio,
                      anchored=args.anchored, engine=args.engine, metric=args.metric, **_bt_kw(args))
    print(f"\nWalk-Forward：{wf['n_splits']} 折 · 锚定={args.anchored}\n")
    for f in wf["folds"]:
        o = f["oos"]
        print(f"  折{f['fold']}  测试 {f['test_start']}→{f['test_end']}  "
              f"参数 {f['best_params']}  收益 {o.get('total_return',0)*100:+.2f}%  "
              f"夏普 {o.get('sharpe',0):.2f}  回撤 {o.get('max_drawdown',0)*100:.2f}%")
    om, fm = wf["oos_metrics"], wf["full_metrics"]
    print(f"\n  样本外合计：收益 {om.get('total_return',0)*100:+.2f}%  夏普 {om.get('sharpe',0):.2f}  "
          f"回撤 {om.get('max_drawdown',0)*100:.2f}%")
    print(f"  全区间（含参数拟合）：收益 {fm.get('total_return',0)*100:+.2f}%  夏普 {fm.get('sharpe',0):.2f}")
    print("\n  若样本外夏普明显低于全区间，说明参数存在过拟合。")


def cmd_list(args):
    for s in list_strategies():
        print(f"  {s['key']:14s} {s['name']}")
        print(f"  {'':14s} {s['desc']}")
        if s["params"]:
            print(f"  {'':14s} 默认参数 {s['params']}")


def main():
    p = argparse.ArgumentParser(description="QuantLab 量化回测")
    p.add_argument("cmd", choices=["serve", "backtest", "scan", "wf", "list"])
    p.add_argument("--symbol", default="01810")
    p.add_argument("--market", default="HK", choices=["A", "HK", "US", "CRYPTO"])
    p.add_argument("--start", default="2021-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--exchange", default="binance")
    p.add_argument("--strategy", default="ma_cross")
    p.add_argument("--params", default=None, help='JSON，如 \'{"fast":10,"slow":60}\'')
    p.add_argument("--engine", default="vector", choices=["vector", "event"])
    p.add_argument("--cash", type=float, default=1_000_000)
    p.add_argument("--commission", type=float, default=0.0005)
    p.add_argument("--slippage", type=float, default=0.0005)
    p.add_argument("--t_plus_1", action="store_true")
    p.add_argument("--stop_loss", type=float, default=None)
    p.add_argument("--take_profit", type=float, default=None)
    p.add_argument("--trailing_stop", type=float, default=None)
    p.add_argument("--metric", default="sharpe")
    p.add_argument("--splits", type=int, default=5)
    p.add_argument("--train-ratio", dest="train_ratio", type=float, default=0.7)
    p.add_argument("--anchored", action="store_true")
    p.add_argument("--min-trades", dest="min_trades", type=int, default=0)
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--out", default=None)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--reload", action="store_true")
    args = p.parse_args()

    if args.cmd == "list":
        cmd_list(args)
        return
    try:
        {"serve": cmd_serve, "backtest": cmd_backtest, "scan": cmd_scan, "wf": cmd_wf}[args.cmd](args)
    except Exception as e:  # noqa: BLE001
        print(f"错误：{e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
