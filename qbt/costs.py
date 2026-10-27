"""真实交易成本模型。

很多回测「看起来能赚」，实盘一上就被成本吃掉——因为默认参数太理想：
A 股漏了印花税和最低佣金，港股漏了双边印花税，滑点当成固定万分之一。
本模块把这些补齐，并提供 zipline / LEAN 风格的滑点模型。

费率口径（2026 年）：
- A 股：佣金万 2.5（最低 5 元）+ 卖出印花税 0.05% + 过户费 0.001%（双边）
- 港股：印花税 0.10%（双边）+ 交易征费 0.0027% + 交收费 0.00565%
        + 交易费 0.00565% + 券商佣金约 0.05%
- 美股：多数券商零佣，成本主要在买卖价差与冲击成本
- 加密货币：taker 0.05% / maker 0.02%
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

# 市场 → 默认费率
PRESETS: dict[str, dict] = {
    "A": {
        "label": "A股（佣金万2.5 / 卖出印花税0.05% / 过户费0.001%）",
        "commission": 0.00025,
        "min_commission": 5.0,
        "stamp_duty": 0.0005,      # 仅卖出
        "transfer_fee": 0.00001,   # 双边
        "slippage": 0.001,
        "slippage_mode": "volume",
        "volume_impact": 0.10,
        "allow_short": False,
        "t_plus_1": True,
    },
    "HK": {
        "label": "港股（印花税0.10%双边 + 各项征费 + 佣金0.05%）",
        "commission": 0.0005,
        "min_commission": 0.0,
        "stamp_duty": 0.001,       # 双边
        "transfer_fee": 0.0001425, # 交易征费+交收费+交易费合计约 0.01425%
        "slippage": 0.001,
        "slippage_mode": "volume",
        "volume_impact": 0.08,
        "allow_short": False,
        "t_plus_1": False,
    },
    "US": {
        "label": "美股（零佣 + 价差冲击）",
        "commission": 0.0,
        "min_commission": 0.0,
        "stamp_duty": 0.0,
        "transfer_fee": 0.0,
        "slippage": 0.0005,
        "slippage_mode": "volume",
        "volume_impact": 0.05,
        "allow_short": True,
        "t_plus_1": False,
    },
    "CRYPTO": {
        "label": "加密货币（taker 0.05%）",
        "commission": 0.0005,
        "min_commission": 0.0,
        "stamp_duty": 0.0,
        "transfer_fee": 0.0,
        "slippage": 0.0005,
        "slippage_mode": "volume",
        "volume_impact": 0.05,
        "allow_short": False,
        "t_plus_1": False,
    },
    "IDEAL": {
        "label": "零成本（仅作对照）",
        "commission": 0.0,
        "min_commission": 0.0,
        "stamp_duty": 0.0,
        "transfer_fee": 0.0,
        "slippage": 0.0,
        "slippage_mode": "pct",
        "volume_impact": 0.0,
        "allow_short": True,
        "t_plus_1": False,
    },
}


@dataclass
class CostModel:
    """一次交易的完整成本模型。"""

    commission: float = 0.0005          # 佣金率（双边）
    min_commission: float = 0.0         # 单笔最低佣金（元）
    stamp_duty: float = 0.0             # 印花税率（A股仅卖出；港股双边）
    stamp_duty_both_sides: bool = False  # True=双边征收
    transfer_fee: float = 0.0           # 过户费/其他规费（双边）
    slippage: float = 0.0005            # 滑点基准
    slippage_mode: str = "pct"          # pct | volume | spread
    volume_impact: float = 0.10         # volume 模式的冲击系数
    spread: float = 0.0                 # spread 模式的固定买卖价差

    @classmethod
    def for_market(cls, market: str, **overrides) -> "CostModel":
        p = dict(PRESETS.get(market.upper(), PRESETS["US"]))
        p.pop("label", None)
        p.pop("allow_short", None)
        p.pop("t_plus_1", None)
        # A 股印花税单边
        both = p.pop("stamp_duty_both_sides", market.upper() == "HK")
        p.update(overrides)
        return cls(stamp_duty_both_sides=bool(both), **p)

    def buy_price(self, price: float, volume: float = 0.0, qty: float = 0.0) -> float:
        return price * (1.0 + self._slip(volume, qty))

    def sell_price(self, price: float, volume: float = 0.0, qty: float = 0.0) -> float:
        return price * (1.0 - self._slip(volume, qty))

    def _slip(self, volume: float, qty: float) -> float:
        """滑点比例。"""
        if self.slippage_mode == "volume" and volume > 0 and qty > 0:
            share = min(qty / volume, 1.0)
            # 冲击随成交量占比的平方增长（zipline VolumeShareSlippage 思路）
            return min(self.slippage + self.volume_impact * share ** 2, 0.05)
        if self.slippage_mode == "spread":
            return max(self.slippage, self.spread / 2.0)
        return self.slippage

    def fee(self, notional: float, side: str) -> float:
        """该笔交易的全部费用（佣金 + 印花税 + 规费）。"""
        n = abs(notional)
        fee = n * self.commission
        if self.min_commission > 0:
            fee = max(fee, self.min_commission)
        if self.stamp_duty > 0:
            if self.stamp_duty_both_sides or side == "sell":
                fee += n * self.stamp_duty
        fee += n * self.transfer_fee
        return fee

    def round_trip_cost(self, notional: float = 100_000.0) -> dict:
        """单次往返（买+卖）成本率，用于快速判断策略毛利需要多高。"""
        buy_n = notional
        sell_n = notional
        cost = self.fee(buy_n, "buy") + self.fee(sell_n, "sell")
        cost += buy_n * self._slip(0, 0) + sell_n * self._slip(0, 0)
        return {
            "notional": notional,
            "round_trip_cost": float(cost),
            "round_trip_rate": float(cost / notional),
            "breakeven_move": float(cost / notional),
        }

    def to_dict(self) -> dict:
        return asdict(self)


def market_defaults(market: str) -> dict:
    """给 UI 用的市场默认值（含交易规则）。"""
    p = dict(PRESETS.get(market.upper(), PRESETS["US"]))
    return p


def list_presets() -> list[dict]:
    return [{"market": k, "label": v["label"]} for k, v in PRESETS.items()]
