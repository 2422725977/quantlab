/* QuantLab 前端 */

const $ = (id) => document.getElementById(id);
const UP = '#c0392b', DN = '#1e8e5a';
const AX = {
  axisLine: { lineStyle: { color: '#c8ccd2' } },
  axisLabel: { color: '#6b7280', fontSize: 11 },
  splitLine: { lineStyle: { color: '#eef0f3' } },
};

const pct = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v)) ? '—' : (v * 100).toFixed(d) + '%';
const num = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v)) ? '—' : Number(v).toFixed(d);
const cls = (v) => v >= 0 ? 'up' : 'dn';

let STRATS = [];
let charts = {};

/* ---------------- 初始化 ---------------- */
async function init() {
  const r = await fetch('/api/strategies').then(r => r.json());
  STRATS = r;
  const sel = $('strategy');
  sel.innerHTML = r.map(s => `<option value="${s.key}">${s.name}</option>`).join('');
  sel.value = 'ma_cross';
  sel.onchange = renderParams;
  renderParams();

  document.querySelectorAll('.tab').forEach(t => {
    t.onclick = () => {
      document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
      document.querySelectorAll('.tabpane').forEach(x => x.classList.remove('active'));
      t.classList.add('active');
      $('pane-' + t.dataset.tab).classList.add('active');
      setTimeout(() => Object.values(charts).forEach(c => c && c.resize && c.resize()), 30);
    };
  });

  $('btnData').onclick = loadData;
  $('btnRun').onclick = runBacktest;
  $('btnOpt').onclick = runOptimize;
  $('btnWf').onclick = runWF;
  $('btnReport').onclick = downloadReport;
}

function renderParams() {
  const s = STRATS.find(x => x.key === $('strategy').value);
  $('strategyDesc').textContent = s ? s.desc : '';
  const box = $('params');
  box.innerHTML = '';
  const params = s.params || {};
  const grid = s.grid || {};
  Object.keys(params).forEach(k => {
    const wrap = document.createElement('label');
    const hasGrid = grid[k] && grid[k].length > 1;
    if (hasGrid) {
      const opts = grid[k].map(v => `<option value="${v}" ${v == params[k] ? 'selected' : ''}>${v}</option>`).join('');
      wrap.innerHTML = `${k}<select data-p="${k}">${opts}</select>`;
    } else {
      wrap.innerHTML = `${k}<input type="number" data-p="${k}" value="${params[k]}" step="any">`;
    }
    box.appendChild(wrap);
  });
}

function collectParams() {
  const out = {};
  document.querySelectorAll('#params [data-p]').forEach(el => {
    out[el.dataset.p] = isNaN(parseFloat(el.value)) ? el.value : parseFloat(el.value);
  });
  return out;
}

function baseReq(extra = {}) {
  const pct2 = (v) => (v === '' || v === null) ? null : parseFloat(v) / 100;
  const bp = (v) => parseFloat(v || 0) / 10000;
  return {
    symbol: $('symbol').value.trim(),
    market: $('market').value,
    start: $('start').value,
    end: $('end').value,
    exchange: $('exchange').value || 'binance',
    strategy: $('strategy').value,
    params: collectParams(),
    engine: $('engine').value,
    initial_cash: parseFloat($('cash').value || 1000000),
    commission: bp($('commission').value),
    slippage: bp($('slippage').value),
    t_plus_1: $('tplus1').checked,
    stop_loss: pct2($('stopLoss').value),
    take_profit: pct2($('takeProfit').value),
    trailing_stop: pct2($('trailing').value),
    ...extra,
  };
}

function setStatus(msg, isErr = false) {
  const el = $('status');
  el.textContent = msg || '';
  el.className = 'status' + (isErr ? ' err' : '');
}

/* ---------------- 数据 ---------------- */
async function loadData() {
  setStatus('载入行情…');
  try {
    const r = await fetch('/api/data', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(baseReq()),
    }).then(r => r.json());
    if (r.detail) throw new Error(r.detail);
    $('dataInfo').textContent = `${r.rows} 根 K 线 · ${r.start} → ${r.end}`;
    setStatus('数据就绪');
  } catch (e) { setStatus('数据失败：' + e.message, true); }
}

/* ---------------- 单策略回测 ---------------- */
async function runBacktest() {
  setStatus('回测中…');
  try {
    const r = await fetch('/api/backtest', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(baseReq()),
    }).then(r => r.json());
    if (r.detail) throw new Error(r.detail);
    renderBacktest(r);
    setStatus(`完成 · ${r.symbol} · ${r.strategy.name}`);
  } catch (e) { setStatus('回测失败：' + e.message, true); }
}

function renderBacktest(r) {
  const m = r.metrics;
  const cards = [
    ['总收益', pct(m.total_return), cls(m.total_return)],
    ['年化 CAGR', pct(m.cagr), cls(m.cagr)],
    ['夏普', num(m.sharpe), m.sharpe >= 1 ? 'up' : ''],
    ['索提诺', num(m.sortino), ''],
    ['最大回撤', pct(m.max_drawdown), 'dn'],
    ['Calmar', num(m.calmar), ''],
    ['年化波动', pct(m.volatility), ''],
    ['交易次数', m.trades, ''],
    ['胜率', pct(m.win_rate), ''],
    ['盈亏比', num(m.profit_factor), ''],
    ['超额收益', m.excess_return !== undefined ? pct(m.excess_return) : '—', cls(m.excess_return || 0)],
    ['Beta', m.beta !== undefined ? num(m.beta) : '—', ''],
  ];
  $('cards').innerHTML = cards.map(([k, v, c]) =>
    `<div class="card"><div class="k">${k}</div><div class="v ${c}">${v}</div></div>`).join('');

  chart('chartEquity', {
    tooltip: { trigger: 'axis' },
    legend: { data: ['策略净值', '回撤 %'], top: 2, textStyle: { fontSize: 11 } },
    grid: { left: 62, right: 58, top: 34, bottom: 52 },
    xAxis: { type: 'category', data: r.dates, ...AX, splitLine: { show: false } },
    yAxis: [
      { type: 'value', name: '净值', ...AX },
      { type: 'value', name: '回撤%', ...AX, splitLine: { show: false }, axisLabel: { formatter: '{value}%', color: '#6b7280', fontSize: 11 } },
    ],
    dataZoom: [{ type: 'inside' }, { type: 'slider', height: 16, bottom: 14 }],
    series: [
      { name: '策略净值', type: 'line', data: r.equity, symbol: 'none', lineStyle: { color: UP, width: 1.8 } },
      { name: '回撤 %', type: 'line', yAxisIndex: 1, data: r.drawdown, symbol: 'none', lineStyle: { color: DN, width: 1 }, areaStyle: { color: 'rgba(30,142,90,.12)' } },
    ],
  });

  chart('chartPrice', {
    tooltip: { trigger: 'axis' },
    legend: { data: ['收盘价', '仓位'], top: 2, textStyle: { fontSize: 11 } },
    grid: { left: 62, right: 58, top: 34, bottom: 34 },
    xAxis: { type: 'category', data: r.price.dates, ...AX, splitLine: { show: false } },
    yAxis: [
      { type: 'value', name: '价格', scale: true, ...AX },
      { type: 'value', name: '仓位', min: 0, max: 1, ...AX, splitLine: { show: false } },
    ],
    series: [
      { name: '收盘价', type: 'line', data: r.price.close, symbol: 'none', lineStyle: { color: '#374151', width: 1.2 } },
      { name: '仓位', type: 'line', yAxisIndex: 1, data: r.position, symbol: 'none', lineStyle: { color: '#2563eb', width: 1 }, areaStyle: { color: 'rgba(37,99,235,.15)' } },
    ],
  });

  const mk = Object.keys(m.monthly_returns || {});
  chart('chartMonthly', {
    tooltip: { trigger: 'axis', formatter: p => `${p[0].axisValue}：${p[0].data}%` },
    grid: { left: 58, right: 18, top: 16, bottom: 56 },
    xAxis: { type: 'category', data: mk, ...AX, splitLine: { show: false }, axisLabel: { color: '#6b7280', fontSize: 10, rotate: 45 } },
    yAxis: { type: 'value', name: '%', ...AX },
    series: [{ type: 'bar', data: mk.map(k => +(m.monthly_returns[k] * 100).toFixed(2)), itemStyle: { color: p => p.value >= 0 ? UP : DN } }],
  });

  const t = r.trades || [];
  $('tradeTable').innerHTML =
    `<thead><tr><th>买入</th><th>卖出</th><th>买入价</th><th>卖出价</th><th>收益</th><th>原因</th></tr></thead><tbody>` +
    t.map(x => `<tr><td>${x.entry_date || ''}</td><td>${x.exit_date || ''}</td><td>${x.entry_price ?? ''}</td><td>${x.exit_price ?? ''}</td>` +
      `<td class="${cls(x.return || 0)}">${pct(x.return)}</td><td>${x.reason || ''}</td></tr>`).join('') +
    `</tbody>`;
}

/* ---------------- 参数扫描 ---------------- */
async function runOptimize() {
  setStatus('扫描中…');
  try {
    const r = await fetch('/api/optimize', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(baseReq({
        metric: $('metric').value,
        min_trades: parseInt($('minTrades').value || 0),
        top: 30,
      })),
    }).then(r => r.json());
    if (r.detail) throw new Error(r.detail);
    renderOptimize(r);
    setStatus(`扫描完成 · ${r.count} 组组合`);
  } catch (e) { setStatus('扫描失败：' + e.message, true); }
}

function renderOptimize(r) {
  const metric = r.metric;
  const rows = r.rows || [];
  $('optInfo').textContent = `共 ${r.count} 组 · 按 ${metric} 排序`;

  const pkeys = rows.length ? Object.keys(rows[0]).filter(k =>
    !['trades', 'win_rate', 'profit_factor', 'sharpe', 'sortino', 'cagr', 'calmar', 'total_return',
      'max_drawdown', 'volatility', 'ulcer_index', 'bars', 'years', 'best_bar', 'worst_bar',
      'avg_pnl', 'avg_win', 'avg_loss', 'expectancy', 'alpha', 'beta'].includes(k)) : [];

  chart('chartOpt', {
    tooltip: { trigger: 'axis' },
    grid: { left: 62, right: 20, top: 20, bottom: 80 },
    xAxis: {
      type: 'category',
      data: rows.map(x => pkeys.map(k => `${k}=${x[k]}`).join(' ')),
      ...AX, splitLine: { show: false },
      axisLabel: { color: '#6b7280', fontSize: 10, rotate: 40 },
    },
    yAxis: { type: 'value', name: metric, ...AX },
    series: [{ type: 'bar', data: rows.map(x => +Number(x[metric]).toFixed(3)), itemStyle: { color: p => p.value >= 0 ? UP : DN } }],
  });

  const vals = rows.map(x => Number(x[metric])).filter(v => !Number.isNaN(v));
  chart('chartDist', {
    tooltip: { trigger: 'axis', formatter: p => `${p[0].name}：${p[0].data} 组` },
    grid: { left: 62, right: 20, top: 20, bottom: 40 },
    xAxis: { type: 'category', data: ['<0', '0~0.5', '0.5~1', '1~1.5', '1.5~2', '>2'], ...AX, splitLine: { show: false } },
    yAxis: { type: 'value', name: '组合数', ...AX },
    series: [{
      type: 'bar',
      data: [
        vals.filter(v => v < 0).length,
        vals.filter(v => v >= 0 && v < 0.5).length,
        vals.filter(v => v >= 0.5 && v < 1).length,
        vals.filter(v => v >= 1 && v < 1.5).length,
        vals.filter(v => v >= 1.5 && v < 2).length,
        vals.filter(v => v >= 2).length,
      ],
      itemStyle: { color: '#2563eb' },
    }],
  });

  const cols = [...pkeys, 'total_return', 'cagr', 'sharpe', 'max_drawdown', 'calmar', 'trades', 'win_rate', 'profit_factor'];
  $('optTable').innerHTML =
    `<thead><tr>${cols.map(c => `<th>${c}</th>`).join('')}</tr></thead><tbody>` +
    rows.map(x => `<tr>${cols.map(c => {
      if (['total_return', 'cagr', 'max_drawdown', 'win_rate'].includes(c)) return `<td class="${cls(x[c] || 0)}">${pct(x[c])}</td>`;
      if (['sharpe', 'calmar', 'profit_factor'].includes(c)) return `<td>${num(x[c])}</td>`;
      return `<td>${x[c]}</td>`;
    }).join('')}</tr>`).join('') + `</tbody>`;
}

/* ---------------- 样本外检验 ---------------- */
async function runWF() {
  setStatus('Walk-Forward 运行中…');
  try {
    const r = await fetch('/api/walkforward', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(baseReq({
        n_splits: parseInt($('nSplits').value || 5),
        train_ratio: parseFloat($('trainRatio').value || 0.7),
        anchored: $('anchored').checked,
        metric: $('metric').value,
      })),
    }).then(r => r.json());
    if (r.detail) throw new Error(r.detail);
    renderWF(r);
    setStatus(`完成 · ${r.folds.length} 折`);
  } catch (e) { setStatus('失败：' + e.message, true); }
}

function renderWF(r) {
  const m = r.oos_metrics || {}, f = r.full_metrics || {};
  $('wfCards').innerHTML = [
    ['样本外总收益', pct(m.total_return), cls(m.total_return || 0)],
    ['样本外 CAGR', pct(m.cagr), cls(m.cagr || 0)],
    ['样本外夏普', num(m.sharpe), ''],
    ['样本外最大回撤', pct(m.max_drawdown), 'dn'],
    ['全区间夏普（含拟合）', num(f.sharpe), ''],
    ['全区间最大回撤', pct(f.max_drawdown), 'dn'],
  ].map(([k, v, c]) => `<div class="card"><div class="k">${k}</div><div class="v ${c}">${v}</div></div>`).join('');

  chart('chartWF', {
    tooltip: { trigger: 'axis' },
    grid: { left: 62, right: 20, top: 20, bottom: 50 },
    xAxis: { type: 'category', data: r.oos_curve.dates, ...AX, splitLine: { show: false } },
    yAxis: { type: 'value', name: '净值', scale: true, ...AX },
    series: [{ type: 'line', data: r.oos_curve.equity, symbol: 'none', lineStyle: { color: UP, width: 1.8 } }],
  });

  const rows = r.folds || [];
  $('wfTable').innerHTML =
    `<thead><tr><th>折</th><th>训练区间</th><th>测试区间</th><th>选中参数</th><th>样本外收益</th><th>样本外夏普</th><th>样本外回撤</th></tr></thead><tbody>` +
    rows.map(x => `<tr><td>${x.fold}</td><td>${x.train_start} → ${x.train_end}</td><td>${x.test_start} → ${x.test_end}</td>` +
      `<td>${JSON.stringify(x.best_params)}</td><td class="${cls((x.oos.total_return) || 0)}">${pct(x.oos.total_return)}</td>` +
      `<td>${num(x.oos.sharpe)}</td><td class="dn">${pct(x.oos.max_drawdown)}</td></tr>`).join('') + `</tbody>`;
}

/* ---------------- 报告下载 ---------------- */
async function downloadReport() {
  setStatus('生成报告…');
  try {
    const resp = await fetch('/api/report', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(baseReq()),
    });
    if (!resp.ok) throw new Error((await resp.json()).detail || '生成失败');
    const blob = await resp.blob();
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `quantlab-${$('symbol').value}-${$('strategy').value}.html`;
    a.click();
    setStatus('报告已下载');
  } catch (e) { setStatus('报告失败：' + e.message, true); }
}

/* ---------------- 图表工具 ---------------- */
function chart(id, option) {
  const el = $(id);
  if (!el) return;
  if (!charts[id]) charts[id] = echarts.init(el);
  charts[id].setOption(option, true);
}
window.addEventListener('resize', () => Object.values(charts).forEach(c => c && c.resize && c.resize()));

init();
