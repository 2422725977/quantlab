#!/usr/bin/env python3
"""QuantLab 命令行入口。

  python run.py serve                     启动 Web 工作台（默认 http://127.0.0.1:8765）
  python run.py backtest --symbol 01810 --market HK --strategy ma_cross
  python run.py scan     --symbol 01810 --market HK --strategy ma_cross --metric sharpe
  python run.py wf       --symbol 01810 --market HK --strategy ma_cross --splits 5
  python run.py robust   --symbol 01810 --market HK --strategy ma_cross
  python run.py cost     --market HK
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from qbt.analysis import analyze_trades, distribution_stats, yearly_returns
from qbt.costs import CostModel, list_presets
from qbt.data import fetch
from qbt.engine import backtest_event, backtest_vector
from qbt.metrics import compute_metrics
from qbt.optimize import grid_search, walk_forward
from qbt.robustness import (
    bootstrap_confidence,
    deflated_sharpe,
    param_plateau,
    random_strategy_test,
)
from qbt.strategies import get_strategy, list_strategies


def _load(args) -> pd.DataFrame:
    return fetch(args.symbol, args.market, args.start, args.end, exchange=args.exchange)


def _cost(args) -> CostModel | None:
    return CostModel.for_market(args.cost_market) if getattr(args, "cost_market", "") else None


def _bt_kw(args) -> dict:
    base = dict(
        initial_cash=args.cash,
        commission=args.commission,
        slippage=args.slippage,
        cost=_cost(args),
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

    a = analyze_trades(res.get("trades", []), df)
    d = distribution_stats(res["equity"])
    if a.get("count"):
        print("\n  逐笔分析")
        print(f"    SQN            {a['sqn']:9.2f}   ({a['sqn_grade']})")
        print(f"    胜率/盈亏比    {a['win_rate']*100:8.1f}%  {a.get('payoff_ratio') or 0:.2f}")
        print(f"    最大连亏       {a['max_consecutive_losses']:9d}")
        print(f"    平均持仓       {a['avg_hold_bars']:9.1f} 日   暴露 {(a.get('exposure_time') or 0)*100:.0f}%")
        if a.get("mae_mean") is not None:
            print(f"    MAE / MFE      {a['mae_mean']*100:8.2f}% / {a['mfe_mean']*100:.2f}%")
        if d:
            print(f"    日 VaR95       {d['var_95_daily']*100:8.2f}%   尾部比 {d.get('tail_ratio') or 0:.2f}")
    yr = yearly_returns(res["equity"], df["close"])
    if yr:
        print("\n  分年收益（括号内为买入持有）")
        for y in yr:
            b = f"（{y['benchmark']*100:+.1f}%）" if y.get("benchmark") is not None else ""
            print(f"    {y['year']}  {y['return']*100:+8.2f}%  {b}")

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


def cmd_robust(args):
    """四项稳健性检验。"""
    df = _load(args)
    meta = get_strategy(args.strategy)
    params = {**meta["params"], **json.loads(args.params)} if args.params else dict(meta["params"])
    sig = meta["fn"](df, **params)
    fn = backtest_vector if args.engine == "vector" else backtest_event
    res = fn(df, sig, **_bt_kw(args))
    m = compute_metrics(res["equity"], res.get("trades"))
    print(f"\n{args.market}:{args.symbol} · {meta['name']} · {params}")
    print(f"真实绩效：收益 {m['total_return']*100:+.2f}%  夏普 {m['sharpe']:.3f}  回撤 {m['max_drawdown']*100:.2f}%\n")

    b = bootstrap_confidence(res["equity"], n_boot=args.n_boot)
    print("① 自助置信区间（块自助，%d 次）" % b["n_boot"])
    for k, lab in [("total_return", "总收益"), ("sharpe", "夏普"), ("max_drawdown", "最大回撤")]:
        v = b["bands"][k]
        print(f"    {lab:8s} 真实 {b['real'][k]:8.3f} | 5% {v['5']:8.3f} | 50% {v['50']:8.3f} | 95% {v['95']:8.3f}")
    print(f"    夏普为正概率 {b['prob_sharpe_positive']*100:.0f}%   盈利概率 {b['prob_profit']*100:.0f}%\n")

    rt = random_strategy_test(df, sig, n_iter=args.n_iter, engine="vector", metric=args.metric,
                              **{k: v for k, v in _bt_kw(args).items()
                                 if k not in ("t_plus_1", "stop_loss", "take_profit", "trailing_stop")})
    print("② 随机策略对照（%d 次随机入场）" % rt["n_iter"])
    print(f"    你的 {rt['real']:.3f} | 随机均值 {rt['random_mean']:.3f} ± {rt['random_std']:.3f} | 随机95分位 {rt['random_p95']:.3f}")
    print(f"    百分位 {rt['percentile']:.1f}%   p 值 {rt['p_value']:.3f}")
    print(f"    → {rt['verdict']}\n")

    dsr = deflated_sharpe(m["sharpe"], n_trials=args.n_trials, n_obs=len(df))
    print("③ 通缩夏普比（试验次数 %d）" % dsr["n_trials"])
    print(f"    观测夏普 {dsr['observed_sharpe']:.3f} | 零假设下期望最大 {dsr['expected_max_sharpe_under_null']:.3f} | DSR {dsr['deflated_sharpe']:.4f}")
    print(f"    → {dsr['verdict']}\n")

    pl = param_plateau(df, args.strategy, params, engine="vector", metric=args.metric, **_bt_kw(args))
    print("④ 参数高原")
    for a in pl["areas"]:
        sc = "  ".join("—" if s is None else f"{s:.3f}" for s in a["scores"])
        print(f"    {a['param']:6s} 档位 {a['values']} → {sc}")
    if pl.get("overall"):
        o = pl["overall"]
        print(f"    中心 {o['center_score']:.3f} · 最大衰减 {o['worst_neighbor_drop']:.3f} · 阈值 {o['threshold']:.3f}")
        print(f"    → {o['verdict']}")


def cmd_cost(args):
    mk = args.market
    for p in list_presets():
        cm = CostModel.for_market(p["market"])
        rt = cm.round_trip_cost(100_000)
        mark = "*" if p["market"] == mk else " "
        print(f" {mark} {p['market']:7s} 往返成本 {rt['round_trip_rate']*100:6.4f}%  "
              f"佣金 {cm.commission*100:.4f}%  印花税 {cm.stamp_duty*100:.3f}%  "
              f"最低佣金 {cm.min_commission:.0f} 元")
        print(f"   {p['label']}")
    print("\n * = 当前 --market 对应的预设。10 万元一次往返的成本即为策略的盈亏平衡门槛。")


def cmd_list(args):
    for s in list_strategies():
        print(f"  {s['key']:14s} {s['name']}")
        print(f"  {'':14s} {s['desc']}")
        if s["params"]:
            print(f"  {'':14s} 默认参数 {s['params']}")


def main():
    p = argparse.ArgumentParser(description="QuantLab 量化回测")
    p.add_argument("cmd", choices=["serve", "backtest", "scan", "wf", "robust", "cost", "list"])
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
    p.add_argument("--cost-market", dest="cost_market", default="",
                   help="按市场套用真实费率：A / HK / US / CRYPTO / IDEAL")
    p.add_argument("--n-boot", dest="n_boot", type=int, default=800)
    p.add_argument("--n-iter", dest="n_iter", type=int, default=300)
    p.add_argument("--n-trials", dest="n_trials", type=int, default=200)
    p.add_argument("--out", default=None)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--reload", action="store_true")
    args = p.parse_args()

    if args.cmd == "list":
        cmd_list(args)
        return
    try:
        {"serve": cmd_serve, "backtest": cmd_backtest, "scan": cmd_scan,
         "wf": cmd_wf, "robust": cmd_robust, "cost": cmd_cost}[args.cmd](args)
    except Exception as e:  # noqa: BLE001
        print(f"错误：{e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
