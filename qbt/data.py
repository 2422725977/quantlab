"""行情数据获取与缓存。

统一输出 DataFrame：index=date(datetime64)，列为 open/high/low/close/volume（float）。
所有数据源均为惰性导入 + 失败自动降级，缺库时给出明确提示而不是崩溃。
"""

from __future__ import annotations

import json
import os
import urllib.request
from datetime import datetime

import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache")
os.makedirs(CACHE_DIR, exist_ok=True)

OHLCV = ["open", "high", "low", "close", "volume"]

UA = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}


# --------------------------------------------------------------------------
# 基础工具
# --------------------------------------------------------------------------
def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """整理成标准 OHLCV 结构。"""
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    elif df.index.name == "date":
        df.index = pd.to_datetime(df.index)
    else:
        df.index = pd.to_datetime(df.index)
    for c in OHLCV:
        if c not in df.columns:
            df[c] = 0.0 if c == "volume" else float("nan")
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[OHLCV].astype(float)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["close"])
    return df


def _cache_path(symbol: str, market: str, start: str, end: str) -> str:
    key = f"{market}_{symbol}_{start}_{end}".replace("/", "-")
    return os.path.join(CACHE_DIR, key + ".csv")


def _read_cache(path: str, max_age_days: int = 1) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    age = (datetime.now().timestamp() - os.path.getmtime(path)) / 86400
    if age > max_age_days:
        return None
    try:
        return normalize(pd.read_csv(path, index_col=0))
    except Exception:
        return None


# --------------------------------------------------------------------------
# 各数据源
# --------------------------------------------------------------------------
def _fetch_tencent(code: str, market: str, start: str = "", end: str = "", limit: int = 1000) -> pd.DataFrame:
    """腾讯财经接口：港股 hk{5位} / 美股 us{TICKER}。

    实测单次返回上限 1000 根，所以想要长历史必须**分段回溯**：
    固定 end、逐段往前取，直到覆盖 start（或已触到上市首日）。
    """
    # 美股走独立的 usfqkline 接口，且需要交易所后缀（.OQ=纳斯达克 / .N=纽交所）
    endpoint = "usfqkline" if market == "US" else "hkfqkline"
    if market == "US":
        cands = [f"us{code}.OQ", f"us{code}.N", f"us{code}"]
    elif market == "HK":
        cands = [f"hk{code}"]
    else:
        cands = [f"us{code}"]

    def _get(n: int, s: str = "", e: str = "", only: str = "") -> tuple[list, str]:
        last: Exception | None = None
        for sym in ([only] if only else cands):
            url = (
                f"https://web.ifzq.gtimg.cn/appstock/app/{endpoint}/get"
                f"?param={sym},day,{s},{e},{n},qfq"
            )
            try:
                req = urllib.request.Request(url, headers=UA)
                raw = urllib.request.urlopen(req, timeout=40).read().decode("utf-8", "ignore")
                payload = json.loads(raw)
                block = payload.get("data")
                if not isinstance(block, dict) or sym not in block:
                    last = ValueError(f"无数据：{sym}")
                    continue
                node = block[sym]
                if not isinstance(node, dict):
                    last = ValueError(f"结构异常：{sym}")
                    continue
                rows = node.get("qfqday") or node.get("day")
                if rows:
                    return rows, sym
                last = ValueError(f"无数据：{sym}")
            except Exception as ex:  # noqa: BLE001
                last = ex
        raise ValueError(f"腾讯接口无数据（{code}）：{last}")

    rows, used = _get(limit)
    # 分段回溯：若最早一根仍晚于 start，说明被 1000 根上限截断，继续往前拉
    if start:
        seen: set[str] = set()
        for _ in range(6):                      # 最多补 6 段 ≈ 6000 根，足够覆盖 20 年
            first = rows[0][0]
            if first <= start or first in seen:
                break
            seen.add(first)
            # 把上一段最早日期的前一天作为新的 end
            prev_day = (pd.Timestamp(first) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            try:
                more, _ = _get(limit, start, prev_day, only=used)
            except Exception:  # noqa: BLE001
                break
            if not more or more[-1][0] >= first:
                break
            rows = more + rows
    # 行结构：date, open, close, high, low, volume
    df = pd.DataFrame(
        [
            {
                "date": r[0],
                "open": float(r[1]),
                "close": float(r[2]),
                "high": float(r[3]),
                "low": float(r[4]),
                "volume": float(r[5]),
            }
            for r in rows
        ]
    )
    return normalize(df)


def _fetch_yfinance(code: str, market: str, start: str = "", end: str = "") -> pd.DataFrame:
    import yfinance as yf

    ticker = code
    if market == "HK" and not code.endswith(".HK"):
        ticker = code.lstrip("0") + ".HK"
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise ValueError(f"yfinance 无数据：{code}")
    df = df.reset_index()
    df.columns = [str(c).lower() for c in df.columns]
    if "date" not in df.columns:
        df = df.rename(columns={"datetime": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return normalize(df)


def _fetch_akshare(code: str, start: str, end: str) -> pd.DataFrame:
    import akshare as ak

    df = ak.stock_zh_a_hist(
        symbol=code, period="daily", start_date=start.replace("-", ""),
        end_date=end.replace("-", ""), adjust="qfq",
    )
    if df is None or df.empty:
        raise ValueError(f"akshare 无数据：{code}")
    rename = {
        "日期": "date", "开盘": "open", "收盘": "close",
        "最高": "high", "最低": "low", "成交量": "volume",
    }
    df = df.rename(columns=rename)
    return normalize(df)


def _fetch_ccxt(symbol: str, start: str, end: str, exchange: str = "binance") -> pd.DataFrame:
    import ccxt  # noqa: F401

    ex_cls = getattr(__import__("ccxt"), exchange)
    ex = ex_cls({"enableRateLimit": True})
    since = int(pd.Timestamp(start).timestamp() * 1000)
    end_ms = int(pd.Timestamp(end).timestamp() * 1000)
    out = []
    while since < end_ms:
        batch = ex.fetch_ohlcv(symbol, timeframe="1d", since=since, limit=1000)
        if not batch:
            break
        out.extend(batch)
        since = batch[-1][0] + 86400000
        if len(batch) < 1000:
            break
    if not out:
        raise ValueError(f"ccxt 无数据：{symbol}")
    df = pd.DataFrame(out, columns=["ts", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["ts"], unit="ms")
    return normalize(df.drop(columns=["ts"]))


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    for want in ("date", "datetime", "time", "timestamp"):
        if want in cols:
            df = df.rename(columns={cols[want]: "date"})
            break
    return normalize(df)


# --------------------------------------------------------------------------
# 统一入口
# --------------------------------------------------------------------------
def fetch(
    symbol: str,
    market: str = "A",
    start: str = "2018-01-01",
    end: str | None = None,
    use_cache: bool = True,
    exchange: str = "binance",
) -> pd.DataFrame:
    """获取行情。market ∈ A / HK / US / CRYPTO。"""
    end = end or datetime.now().strftime("%Y-%m-%d")
    market = market.upper()
    path = _cache_path(symbol, market, start, end)
    if use_cache:
        cached = _read_cache(path)
        if cached is not None and len(cached) > 0:
            return cached

    errors = []
    df = None

    if market == "A":
        try:
            df = _fetch_akshare(symbol, start, end)
        except Exception as e:  # noqa: BLE001
            errors.append(f"akshare: {e}")
    elif market in ("HK", "US"):
        for fn, name in ((_fetch_tencent, "tencent"), (_fetch_yfinance, "yfinance")):
            try:
                df = fn(symbol, market, start, end)
                break
            except Exception as e:  # noqa: BLE001
                errors.append(f"{name}: {e}")
    elif market == "CRYPTO":
        try:
            df = _fetch_ccxt(symbol, start, end, exchange)
        except Exception as e:  # noqa: BLE001
            errors.append(f"ccxt: {e}")

    if df is None or df.empty:
        raise RuntimeError(f"获取 {market}:{symbol} 失败 → " + " | ".join(errors))

    df = df.loc[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
    if use_cache:
        try:
            df.to_csv(path)
        except Exception:
            pass
    return df


def search_symbol(q: str, market: str = "A") -> list[dict]:
    """简单的标的代码搜索（A股走 akshare 名称表，其余返回原样）。"""
    q = q.strip()
    if not q:
        return []
    if market.upper() == "A":
        try:
            import akshare as ak

            df = ak.stock_info_a_code_name()
            hit = df[df["code"].str.contains(q) | df["name"].str.contains(q)]
            return [{"code": r["code"], "name": r["name"]} for _, r in hit.head(20).iterrows()]
        except Exception:
            return [{"code": q, "name": q}]
    return [{"code": q.upper(), "name": q.upper()}]
