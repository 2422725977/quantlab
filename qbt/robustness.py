"""稳健性验证：把「运气」和「真 alpha」分开。

回测最大的风险不是策略差，而是**你以为策略好**——参数网格扫了几千组，
最好的那组里有多少是拟合噪声？本模块给出四种相互独立、可交叉验证的检验：

1. bootstrap_confidence —— 平稳块自助法，给出绩效指标的置信区间
2. random_strategy_test —— 随机策略对照，给出真实策略的百分位与 p 值
3. deflated_sharpe       —— 通缩夏普，校正「试验次数」带来的选择偏差
4. param_plateau         —— 参数高原，判断最优点是不是孤岛

参考
----
- Aronson《Evidence-Based Technical Analysis》：随机置换检验
- Bailey & López de Prado《The Deflated Sharpe Ratio》(2012/2014)
- Politis & Romano (1994)：Stationary Block Bootstrap，保留波动率聚集
- White (2000)：Reality Check for Data Snooping
"""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .engine import backtest_event, backtest_vector
from .metrics import _annual_factor, compute_metrics
from .strategies import get_strategy

DEFAULT_LEVELS = (5, 25, 50, 75, 95)


# --------------------------------------------------------------------------
# 1) 平稳块自助法
# --------------------------------------------------------------------------
def _stationary_bootstrap(rng: np.random.Generator, x: np.ndarray, block: int) -> np.ndarray:
    """Politis-Romano 平稳块自助。块长服从几何分布（均值=block），保留序列相关。"""
    n = len(x)
    out = np.empty(n, dtype=float)
    pos = 0
    while pos < n:
        start = int(rng.integers(0, n))
        length = int(rng.geometric(1.0 / max(block, 1)))
        take = min(length, n - pos)
        out[pos : pos + take] = x[(start + np.arange(take)) % n]
        pos += take
    return out


def bootstrap_confidence(
    equity: pd.Series,
    *,
    n_boot: int = 1000,
    block: int = 20,
    seed: int = 42,
    levels: tuple = DEFAULT_LEVELS,
    risk_free: float = 0.0,
) -> dict:
    """对净值曲线做块自助重采样，给出绩效指标的分布与置信区间。

    与普通 bootstrap 的区别：块自助保留了波动率聚集（volatility clustering），
    不会像 iid 抽样那样人为抹平极端行情，因此回撤类指标的区间是可信的。
    """
    eq = pd.Series(equity).astype(float).dropna()
    if len(eq) < 30:
        return {"error": "样本过少（<30），无法做自助推断"}

    ret = eq.pct_change().dropna().to_numpy(dtype=float)
    n = len(ret)
    af = _annual_factor(eq.index)
    rf = risk_free / af
    block = max(2, min(int(block), n // 3))

    rng = np.random.default_rng(seed)
    sharpe_s, cagr_s, mdd_s, ret_s = [], [], [], []

    for _ in range(int(n_boot)):
        r = _stationary_bootstrap(rng, ret, block)
        ex = r - rf
        sd = ex.std(ddof=1)
        sharpe_s.append(float(ex.mean() / sd * math.sqrt(af)) if sd > 0 else 0.0)
        curve = np.cumprod(1.0 + r)
        wealth = float(curve[-1])
        ret_s.append(wealth - 1.0)
        cagr_s.append(wealth ** (af / n) - 1.0 if wealth > 0 else -1.0)
        peak = np.maximum.accumulate(curve)
        mdd_s.append(float((curve / peak - 1.0).min()))

    def band(arr):
        a = np.asarray(arr, dtype=float)
        return {str(p): float(np.percentile(a, p)) for p in levels}

    real = compute_metrics(eq, risk_free=risk_free)
    return {
        "n_boot": int(n_boot),
        "block": block,
        "real": {
            "total_return": real.get("total_return"),
            "sharpe": real.get("sharpe"),
            "cagr": real.get("cagr"),
            "max_drawdown": real.get("max_drawdown"),
        },
        "bands": {
            "total_return": band(ret_s),
            "sharpe": band(sharpe_s),
            "cagr": band(cagr_s),
            "max_drawdown": band(mdd_s),
        },
        "prob_sharpe_positive": float(np.mean(np.asarray(sharpe_s) > 0)),
        "prob_profit": float(np.mean(np.asarray(ret_s) > 0)),
        "hist": {
            "sharpe": _hist(sharpe_s, 24),
            "total_return": _hist(ret_s, 24),
            "max_drawdown": _hist(mdd_s, 24),
        },
    }


def _hist(arr, bins: int) -> list[dict]:
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return []
    counts, edges = np.histogram(a, bins=bins)
    return [
        {"x0": float(edges[i]), "x1": float(edges[i + 1]), "n": int(counts[i])}
        for i in range(bins)
    ]


# --------------------------------------------------------------------------
# 2) 随机策略对照检验
# --------------------------------------------------------------------------
def _holding_segments(pos: np.ndarray) -> list[tuple[int, int]]:
    """从 0/1 仓位序列提取持仓区间 [(start, end_exclusive), ...]。"""
    segs = []
    i = 0
    n = len(pos)
    while i < n:
        if pos[i] > 1e-9:
            j = i
            while j < n and pos[j] > 1e-9:
                j += 1
            segs.append((i, j))
            i = j
        else:
            i += 1
    return segs


def randomize_signal(
    signal: pd.Series,
    rng: np.random.Generator,
    method: str = "markov",
) -> pd.Series:
    """在不改变「交易次数 + 持仓时长分布」的前提下随机化入场时机。

    这是检验的核心：如果随机放置的持仓区间也能拿到差不多的收益，
    说明策略的择时没有信息量，赚的钱来自资产本身的 beta。

    markov   : 保持每段持仓长度与权重，随机放回时间轴（推荐，最严格）
    block    : 切成大块后随机重排块（保留局部结构，宽松些）
    shuffle  : 整条序列随机打乱（会破坏持仓连续性，只作对照）
    """
    s = pd.Series(signal).astype(float)
    vals = s.to_numpy(dtype=float)
    n = len(vals)
    if n == 0:
        return s

    if method == "shuffle":
        return pd.Series(rng.permutation(vals), index=s.index)

    if method == "block":
        block = max(2, n // 20)
        nb = math.ceil(n / block)
        order = rng.permutation(nb)
        out = np.concatenate([vals[i * block : (i + 1) * block] for i in order])[:n]
        return pd.Series(out, index=s.index)

    # markov：保持持仓段长度与权重，随机放置
    pos = (np.abs(vals) > 1e-9).astype(int)
    segs = _holding_segments(pos)
    out = np.zeros(n, dtype=float)
    if not segs:
        return pd.Series(out, index=s.index)
    lengths = [e - b for b, e in segs]
    weights = [float(np.mean(np.abs(vals[b:e]))) for b, e in segs]
    order = rng.permutation(len(segs))
    lens = np.array(lengths)[order]
    wts = np.array(weights)[order]
    sign = np.sign(vals[segs[0][0]]) if vals[segs[0][0]] != 0 else 1.0

    placed = 0
    attempts = 0
    max_attempts = 200
    while placed < len(lens) and attempts < max_attempts:
        attempts += 1
        for L, w in zip(lens[placed:], wts[placed:]):
            if L > n:
                continue
            start = int(rng.integers(0, n - L + 1))
            if not out[start : start + L].any():      # 不重叠
                out[start : start + L] = w * sign
                placed += 1
    return pd.Series(out, index=s.index)


def random_strategy_test(
    df: pd.DataFrame,
    signal: pd.Series,
    *,
    n_iter: int = 500,
    method: str = "markov",
    engine: str = "vector",
    metric: str = "sharpe",
    seed: int = 7,
    **bt_kw,
) -> dict:
    """随机策略对照检验。

    把所有入场时机随机重排 N 次，得到「纯运气」的指标分布，
    再看真实策略落在哪个百分位。p 值 = 随机策略中表现 ≥ 真实策略的比例。
    """
    fn = backtest_vector if engine == "vector" else backtest_event
    real_res = fn(df, signal, **bt_kw)
    real_m = compute_metrics(real_res["equity"], real_res.get("trades"))
    real_v = float(real_m.get(metric) or 0.0)

    rng = np.random.default_rng(seed)
    sims = []
    for _ in range(int(n_iter)):
        rsig = randomize_signal(signal, rng, method=method)
        try:
            r = fn(df, rsig, **bt_kw)
            m = compute_metrics(r["equity"], r.get("trades"))
            v = float(m.get(metric) or 0.0)
        except Exception:
            continue
        if math.isfinite(v):
            sims.append(v)

    if not sims:
        return {"error": "随机检验未能完成"}

    arr = np.asarray(sims, dtype=float)
    better = float(np.mean(arr >= real_v))
    return {
        "metric": metric,
        "method": method,
        "n_iter": len(sims),
        "real": real_v,
        "random_mean": float(arr.mean()),
        "random_median": float(np.median(arr)),
        "random_std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "random_p95": float(np.percentile(arr, 95)),
        "percentile": float(np.mean(arr < real_v) * 100),
        "p_value": better,
        "verdict": _verdict(better),
        "hist": _hist(arr, 26),
        "real_metrics": {k: v for k, v in real_m.items() if k != "monthly_returns"},
    }


def _verdict(p: float) -> str:
    if p < 0.01:
        return "极显著：随机策略几乎无法达到该绩效，择时含真实信息"
    if p < 0.05:
        return "显著：在 5% 水平上优于随机，但需结合样本外检验确认"
    if p < 0.20:
        return "弱：未能与运气明确区分，不建议据此实盘"
    return "不显著：绩效与随机入场无异，大概率是 beta 或过拟合"


# --------------------------------------------------------------------------
# 3) 通缩夏普比（Deflated Sharpe Ratio）
# --------------------------------------------------------------------------
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def deflated_sharpe(
    observed_sharpe: float,
    n_trials: int,
    n_obs: int,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> dict:
    """多重检验校正后的夏普比显著性。

    扫 N 组参数后挑最大的夏普，其期望最大值约为
        SR0 = sqrt(V) * [(1-γ)Z^-1(1 - 1/N) + γ Z^-1(1 - 1/(N·e))]
    DSR = Prob(真实夏普 ≤ 观测夏普 | 已考虑 N 次试验)。

    n_trials 越大、样本越短、收益越偏/越厚尾，观测夏普就越可能是运气。
    """
    if n_obs < 30 or n_trials < 2:
        return {"error": "样本或试验次数不足"}
    try:
        from statistics import NormalDist

        nd = NormalDist()
        zinv = nd.inv_cdf
        cdf = nd.cdf
    except Exception:  # pragma: no cover
        zinv = lambda q: math.sqrt(2) * _inv_erf(2 * q - 1)
        cdf = _norm_cdf

    N = float(max(n_trials, 2))
    gamma = 0.5772156649  # Euler-Mascheroni
    try:
        z1 = zinv(1.0 - 1.0 / N)
        z2 = zinv(1.0 - 1.0 / (N * math.e))
    except Exception:
        return {"error": "分位数计算失败"}
    sr0 = math.sqrt(1.0 / n_obs) * ((1.0 - gamma) * z1 + gamma * z2)

    sr = float(observed_sharpe)
    # 观测夏普的标准误：含偏度与峰度修正（Lo 2002 / Mertens 2002）
    var_sr = (1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr) / max(n_obs - 1, 1)
    se = math.sqrt(max(var_sr, 1e-12))
    dsr = cdf((sr - sr0) / se) if se > 0 else 0.0

    return {
        "observed_sharpe": sr,
        "n_trials": int(n_trials),
        "n_obs": int(n_obs),
        "expected_max_sharpe_under_null": float(sr0),
        "std_error": float(se),
        "deflated_sharpe": float(dsr),
        "significant_at_95": bool(dsr > 0.95),
        "verdict": (
            "通过：即便考虑试验次数，夏普仍显著"
            if dsr > 0.95
            else "未通过：观测夏普可被「扫参数」本身解释，存在过拟合嫌疑"
        ),
    }


def _inv_erf(x: float) -> float:
    # 近似反误差函数（仅在 statistics 不可用时兜底）
    a = 0.147
    ln1mx2 = math.log(max(1 - x * x, 1e-16))
    t1 = 2.0 / (math.pi * a) + ln1mx2 / 2.0
    return math.copysign(math.sqrt(max(math.sqrt(t1 * t1 - ln1mx2 / a) - t1, 0.0)), x)


# --------------------------------------------------------------------------
# 4) 参数高原（邻域稳健性）
# --------------------------------------------------------------------------
def param_plateau(
    df: pd.DataFrame,
    strategy_key: str,
    best_params: dict,
    grid: dict | None = None,
    *,
    engine: str = "vector",
    metric: str = "sharpe",
    **bt_kw,
) -> dict:
    """检验最优点是否位于「高原」上：邻域参数小幅扰动后表现是否依然稳健。

    若最优点四周断崖式下跌 → 典型的过拟合孤岛，实盘不可用；
    若邻域表现接近 → 参数不敏感，可放心用。
    """
    meta = get_strategy(strategy_key)
    grid = grid or meta.get("grid") or {}
    if not grid:
        return {"error": f"策略 {strategy_key} 未定义参数网格"}

    # 找出可数值化的参数（至少 3 个取值）
    axes = {}
    for k, v in grid.items():
        try:
            vals = sorted({float(x) for x in v})
        except Exception:
            continue
        if len(vals) >= 3:
            axes[k] = vals
    if not axes:
        return {"error": "没有可扫描的数值参数"}

    def score(p: dict) -> float | None:
        full = {**meta["params"], **p}
        if not _valid(full, axes):
            return None
        try:
            sig = meta["fn"](df, **full)
            fn = backtest_vector if engine == "vector" else backtest_event
            r = fn(df, sig, **bt_kw)
            m = compute_metrics(r["equity"], r.get("trades"))
            v = float(m.get(metric) or 0.0)
            return v if math.isfinite(v) else None
        except Exception:
            return None

    def _valid(p: dict, axes: dict) -> bool:
        # 快/慢线类参数需保持大小关系
        if "fast" in p and "slow" in p and float(p["fast"]) >= float(p["slow"]):
            return False
        if "fast" in p and "mid" in p and float(p["fast"]) >= float(p["mid"]):
            return False
        if "mid" in p and "slow" in p and float(p["mid"]) >= float(p["slow"]):
            return False
        return True

    center_score = score(best_params)
    areas = []
    for k, vals in axes.items():
        vals_sorted = sorted(vals)
        cur = float(best_params.get(k, vals_sorted[0]))
        pos = min(range(len(vals_sorted)), key=lambda i: abs(vals_sorted[i] - cur))
        # 邻域 = 最优档位 ± 1 档（真正的「小幅扰动」）
        lo = max(0, pos - 1)
        hi = min(len(vals_sorted), pos + 2)
        window = vals_sorted[lo:hi]
        row = {"param": k, "values": window, "center_value": cur, "scores": []}
        for v in window:
            p = {**best_params, k: v}
            row["scores"].append(score(p))
        sc = [s for s in row["scores"] if s is not None]
        if sc:
            row["neighbor_min"] = float(min(sc))
            row["neighbor_mean"] = float(sum(sc) / len(sc))
            row["center"] = center_score
            row["fragility"] = (
                float(center_score - min(sc)) if center_score is not None else None
            )
        areas.append(row)

    frags = [a.get("fragility") for a in areas if a.get("fragility") is not None]
    overall = None
    if frags and center_score is not None:
        base = float(center_score)
        worst = float(max(frags))
        # 阈值：相对 25%（或绝对 0.15，照顾夏普这类绝对值小的指标）
        thr = max(0.25 * abs(base), 0.15)
        stable = worst < thr
        overall = {
            "center_score": base,
            "worst_neighbor_drop": worst,
            "threshold": thr,
            "stable": bool(stable),
            "verdict": (
                "参数高原：邻域表现接近，参数不敏感，可放心使用"
                if stable
                else "参数孤岛：微调即大幅衰减，存在过拟合，建议换更平滑的参数区"
            ),
        }
    return {"areas": areas, "metric": metric, "overall": overall}
