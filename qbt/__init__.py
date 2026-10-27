"""qbt — 轻量量化回测引擎（向量化 + 事件驱动混合）。"""

__version__ = "0.1.0"

from .metrics import compute_metrics, max_drawdown
from .engine import backtest_vector, backtest_event
from .strategies import STRATEGIES, get_strategy, signal_for
from .optimize import grid_search, walk_forward

__all__ = [
    "compute_metrics",
    "max_drawdown",
    "backtest_vector",
    "backtest_event",
    "STRATEGIES",
    "get_strategy",
    "signal_for",
    "grid_search",
    "walk_forward",
]
