"""qbt — 轻量量化回测引擎（向量化 + 事件驱动混合）。"""

__version__ = "0.2.0"

from .metrics import compute_metrics, max_drawdown
from .engine import backtest_vector, backtest_event
from .strategies import STRATEGIES, get_strategy, signal_for
from .optimize import grid_search, walk_forward
from .costs import CostModel, market_defaults, list_presets
from .robustness import (
    bootstrap_confidence,
    random_strategy_test,
    deflated_sharpe,
    param_plateau,
)
from .analysis import (
    analyze_trades,
    rolling_metrics,
    yearly_returns,
    monthly_table,
    distribution_stats,
)
from .portfolio import (
    build_panel,
    open_to_open_returns,
    allocate,
    cross_sectional_momentum,
    backtest_portfolio,
    compare_weights,
    correlation_matrix,
)

__all__ = [
    # 核心
    "compute_metrics",
    "max_drawdown",
    "backtest_vector",
    "backtest_event",
    "STRATEGIES",
    "get_strategy",
    "signal_for",
    "grid_search",
    "walk_forward",
    # 成本
    "CostModel",
    "market_defaults",
    "list_presets",
    # 稳健性
    "bootstrap_confidence",
    "random_strategy_test",
    "deflated_sharpe",
    "param_plateau",
    # 分析
    "analyze_trades",
    "rolling_metrics",
    "yearly_returns",
    "monthly_table",
    "distribution_stats",
    # 组合
    "build_panel",
    "open_to_open_returns",
    "allocate",
    "cross_sectional_momentum",
    "backtest_portfolio",
    "compare_weights",
    "correlation_matrix",
]
