# QuantLab

向量化 + 事件驱动混合的量化回测工作台。Web 界面操作，支持参数扫描与样本外（Walk-Forward）检验。

```
python run.py serve     # → http://127.0.0.1:8765
```

## 为什么自己写

主流框架各有取舍，这里按「研究阶段快、验证阶段准」的工业惯例把两种范式合在一起：

| 引擎 | 用途 | 特点 |
|---|---|---|
| **向量化** | 参数扫描、想法快速验证 | NumPy 全程向量化，上千组组合秒级完成；单 bar 收益 = open<sub>t+1</sub>/open<sub>t</sub> − 1 |
| **事件驱动** | 最终验证 | 逐 bar 推进，真实股数/手续费/滑点，**盘中止损·止盈·移动止损**、A股 T+1 |

两个引擎共用同一套策略信号，在小米 01810 上的实测结果差异 < 1%（见下方验证），互为交叉校验。

## 快速开始

```bash
pip install -r requirements.txt
python run.py serve
```

命令行也能跑：

```bash
python run.py list
python run.py backtest --symbol 01810 --market HK --strategy ma_cross --engine event --stop_loss 0.10
python run.py scan     --symbol 01810 --market HK --strategy ma_cross --metric sharpe
python run.py wf       --symbol 01810 --market HK --strategy ma_cross --splits 5
```

## 无未来函数

策略在第 t 根 K 线**收盘**产生信号 → 引擎顺延到第 t+1 根**开盘价**成交。两条引擎都严格遵循这一约定，指标可放心使用当日收盘价。

## 内置策略

| key | 名称 | 默认参数 |
|---|---|---|
| `buy_hold` | 买入持有（基准） | — |
| `ma_cross` | 双均线交叉 | fast=20, slow=60 |
| `ma_triple` | 三线多头排列 | 5 / 20 / 60 |
| `boll_revert` | 布林带均值回归 | period=20, nstd=2.0 |
| `rsi_revert` | RSI 超卖反弹 | 14 / 30 / 70 |
| `macd` | MACD 金叉死叉 | 12 / 26 / 9 |
| `donchian` | 唐奇安通道突破 | period=20 |
| `momentum` | 动量择时 | lookback=60, threshold=0 |
| `vol_target` | 波动率目标仓位 | 20 / 0.20 |
| `atr_breakout` | ATR 通道突破 | period=20, mult=2.0 |
| `grid` | 均线网格 | period=20, bands=0.05 |

新增策略：在 `qbt/strategies.py` 写一个 `fn(df, **params) -> Series`（返回目标仓位权重），登记到 `STRATEGIES` 并给出默认参数与 `grid`（扫描网格），前端会自动出现。

## 样本外检验怎么用

`Walk-Forward` 把区间切成 N 折，每折前 70% 训练（网格寻优挑最佳参数），后 30% 测试，**只把各折测试段的净值拼接**成最终曲线——这条曲线从未参与任何参数选择。

判据：**样本外夏普若明显低于全区间夏普，就是过拟合**。实测例子（小米 01810，2022-10 ~ 2026-10）：

| 策略 | 全区间（含拟合） | 样本外（诚实） | 结论 |
|---|---|---|---|
| `ma_cross` | +129.9%，夏普 0.78 | +39.9%，夏普 1.35 | 参数不敏感，可用 |
| `donchian` | +195.0%，夏普 0.99 | +14.0%，夏普 0.56 | 有明显衰减 |
| `rsi_revert` | +56.6%，夏普 0.53 | **−11.5%，夏普 −0.42** | **典型过拟合，不能用** |

## 数据源

| 市场 | 主源 | 降级源 | 备注 |
|---|---|---|---|
| A 股 | AkShare（前复权） | — | 需 `pip install akshare` |
| 港股 | 腾讯财经（前复权，约 2000 根） | yfinance | 单只上限约 2000 根日线 |
| 美股 | yfinance | — | 腾讯美股接口在部分环境不可用 |
| 加密货币 | ccxt / Binance | 换 `--exchange okx` | 国内访问 Binance 可能需要代理 |

行情缓存在 `quantlab/.cache/`（CSV，默认 1 天过期），重复回测不重复联网。

## 目录结构

```
quantlab/
├── qbt/
│   ├── data.py        行情获取、标准化、缓存
│   ├── engine.py      向量化引擎 + 事件驱动引擎
│   ├── strategies.py  指标与内置策略库
│   ├── metrics.py     绩效指标（夏普/索提诺/回撤/Calmar/月度…）
│   ├── optimize.py    网格扫描 + Walk-Forward
│   ├── report.py      独立 HTML 报告（ECharts 内联，可离线打开）
│   └── server.py      FastAPI 接口
├── web/               前端（原生 JS + ECharts，无构建步骤）
├── run.py             CLI 入口
└── requirements.txt
```

## 已知边界

- 单标的择时回测，不含多资产组合/因子模型/订单簿。
- 向量化引擎不模拟盘中止损（那属于事件驱动引擎的职责）。
- Walk-Forward 折内若训练样本不足（< 40 根）会跳过该折。
- 加密货币与部分美股数据源依赖外网，国内环境可能需代理。

## 免责声明

本项目仅用于量化研究与方法验证。回测结果不代表未来表现，不构成任何投资建议。
