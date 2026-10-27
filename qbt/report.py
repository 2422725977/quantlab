"""生成独立 HTML 绩效报告（ECharts 内联，可离线打开）。"""

from __future__ import annotations

import json
import os
from datetime import datetime

import pandas as pd

from .metrics import drawdown_series

VENDOR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web", "vendor", "echarts.min.js"
)


def _pct(x, digits=2):
    return "—" if x is None else f"{x * 100:.{digits}f}%"


def build_html(
    symbol: str,
    strategy_name: str,
    params: dict,
    res: dict,
    metrics: dict,
    benchmark: pd.Series | None = None,
) -> str:
    eq = res["equity"]
    dd = drawdown_series(eq)
    step = max(1, len(eq) // 3000)
    dates = [str(d)[:10] for d in eq.index[::step]]
    equity = [round(float(v), 2) for v in eq.iloc[::step]]
    ddpct = [round(float(v) * 100, 2) for v in dd.iloc[::step]]
    bench = []
    if benchmark is not None:
        b = benchmark.reindex(eq.index).ffill()
        bench = [round(float(v), 2) for v in b.iloc[::step]]

    monthly = metrics.get("monthly_returns", {})
    m_keys = sorted(monthly)
    trades = res.get("trades", [])[:100]

    lib = ""
    if os.path.exists(VENDOR):
        lib = open(VENDOR, encoding="utf-8").read()

    card = lambda k, v, cls="": f'<div class="card"><div class="k">{k}</div><div class="v {cls}">{v}</div></div>'  # noqa: E731
    tr_cls = "up" if metrics.get("total_return", 0) >= 0 else "dn"

    rows = "".join(
        f"<tr><td>{t.get('entry_date','')}</td><td>{t.get('exit_date','')}</td>"
        f"<td>{t.get('entry_price','')}</td><td>{t.get('exit_price','')}</td>"
        f"<td class='{'up' if (t.get('return') or 0)>=0 else 'dn'}'>{_pct(t.get('return'))}</td>"
        f"<td>{t.get('reason','')}</td></tr>"
        for t in trades
    )

    data = {
        "dates": dates, "equity": equity, "dd": ddpct, "bench": bench,
        "mk": m_keys, "mv": [round(monthly[k] * 100, 2) for k in m_keys],
    }

    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{symbol} · {strategy_name} 回测报告</title>
{f'<script>{lib}</script>' if lib else '<script src="https://cdnjs.cloudflare.com/ajax/libs/echarts/5.4.3/echarts.min.js"></script>'}
<style>
*{{box-sizing:border-box}}
body{{margin:0;background:#f5f6f8;color:#1f2328;font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif;line-height:1.6}}
.wrap{{max-width:1120px;margin:0 auto;padding:26px 20px 60px}}
h1{{font-size:23px;margin:0 0 6px}} .sub{{color:#6b7280;font-size:13px}}
h2{{font-size:17px;margin:30px 0 12px;padding-left:10px;border-left:4px solid #2563eb}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:14px 0}}
.card{{background:#fff;border:1px solid #e5e7eb;border-radius:9px;padding:12px 14px}}
.card .k{{font-size:12px;color:#6b7280}} .card .v{{font-size:20px;font-weight:600;margin-top:3px}}
.up{{color:#c0392b}} .dn{{color:#1e8e5a}}
.chart{{width:100%;height:340px;background:#fff;border:1px solid #e5e7eb;border-radius:9px}}
table{{width:100%;border-collapse:collapse;font-size:13px;background:#fff}}
th,td{{padding:7px 10px;border-bottom:1px solid #eef0f3;text-align:right}}
th:first-child,td:first-child{{text-align:left}}
th{{background:#f0f2f5;color:#374151}}
.disc{{margin-top:26px;background:#f0f2f5;border-radius:8px;padding:12px 16px;font-size:12px;color:#4b5563}}
</style></head><body><div class="wrap">
<h1>{symbol} · {strategy_name}</h1>
<div class="sub">参数 {json.dumps(params, ensure_ascii=False)} · 生成于 {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
<div class="cards">
{card('总收益', _pct(metrics.get('total_return')), tr_cls)}
{card('年化 (CAGR)', _pct(metrics.get('cagr')))}
{card('夏普比率', f"{metrics.get('sharpe', 0):.2f}")}
{card('最大回撤', _pct(metrics.get('max_drawdown')), 'dn')}
{card('Calmar', f"{metrics.get('calmar', 0):.2f}")}
{card('年化波动', _pct(metrics.get('volatility')))}
{card('交易次数', metrics.get('trades', 0))}
{card('胜率', _pct(metrics.get('win_rate')))}
{card('盈亏比', f"{metrics.get('profit_factor', 0):.2f}")}
{card('索提诺', f"{metrics.get('sortino', 0):.2f}")}
</div>
<h2>净值与回撤</h2><div id="c1" class="chart"></div>
<h2>月度收益</h2><div id="c2" class="chart" style="height:260px"></div>
<h2>成交记录（前 {len(trades)} 笔）</h2>
<table><thead><tr><th>买入</th><th>卖出</th><th>买入价</th><th>卖出价</th><th>收益</th><th>原因</th></tr></thead><tbody>{rows}</tbody></table>
<div class="disc">本报告由 QuantLab 自动生成，仅供研究参考，不构成投资建议。回测结果不代表未来表现。</div>
</div>
<script>
var D = {json.dumps(data, ensure_ascii=False)};
var UP='#c0392b', DN='#1e8e5a';
var ax={{axisLine:{{lineStyle:{{color:'#c8ccd2'}}}},axisLabel:{{color:'#6b7280',fontSize:11}},splitLine:{{lineStyle:{{color:'#eef0f3'}}}}}};
var c1=echarts.init(document.getElementById('c1'));
c1.setOption({{
 tooltip:{{trigger:'axis'}},
 legend:{{data:['策略净值','基准','回撤 %'],top:4}},
 grid:{{left:60,right:60,top:40,bottom:60}},
 xAxis:{{type:'category',data:D.dates,...ax,splitLine:{{show:false}}}},
 yAxis:[{{type:'value',name:'净值',...ax}},{{type:'value',name:'回撤%',...ax,splitLine:{{show:false}},axisLabel:{{formatter:'{{value}}%',color:'#6b7280'}}}}],
 dataZoom:[{{type:'inside'}},{{type:'slider',height:18,bottom:16}}],
 series:[
  {{name:'策略净值',type:'line',data:D.equity,symbol:'none',lineStyle:{{color:UP,width:2}}}},
  {{name:'基准',type:'line',data:D.bench,symbol:'none',lineStyle:{{color:'#9ca3af',width:1.4,type:'dashed'}}}},
  {{name:'回撤 %',type:'line',yAxisIndex:1,data:D.dd,symbol:'none',lineStyle:{{color:DN,width:1}},areaStyle:{{color:'rgba(30,142,90,.12)'}}}}
 ]
}});
var c2=echarts.init(document.getElementById('c2'));
c2.setOption({{
 tooltip:{{trigger:'axis',formatter:function(p){{return p[0].axisValue+'：'+p[0].data+'%';}}}},
 grid:{{left:60,right:20,top:20,bottom:50}},
 xAxis:{{type:'category',data:D.mk,...ax,splitLine:{{show:false}},axisLabel:{{color:'#6b7280',fontSize:10,rotate:45}}}},
 yAxis:{{type:'value',name:'%',...ax}},
 series:[{{type:'bar',data:D.mv,itemStyle:{{color:function(p){{return p.value>=0?UP:DN;}}}}}}]
}});
window.addEventListener('resize',function(){{c1.resize();c2.resize();}});
</script></body></html>"""
