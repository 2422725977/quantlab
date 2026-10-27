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
  $('btnRobust').onclick = runRobust;
  $('btnPf').onclick = runPortfolio;
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
    cost_market: $('costMarket').value,
    slippage_mode: $('slippageMode').value,
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

  renderDiagnostics(r);
}

/* ---------------- 诊断层：逐笔分析 / 滚动 / 年度 / 分布 ---------------- */
function renderDiagnostics(r) {
  const a = r.analysis || {};
  const d = r.distribution || {};
  if (!a.count) { $('diagCards').innerHTML = ''; return; }

  const mae = a.mae_mean, mfe = a.mfe_mean;
  $('diagCards').innerHTML = [
    ['SQN 策略质量', num(a.sqn), a.sqn >= 2 ? 'up' : (a.sqn < 0 ? 'dn' : '')],
    ['期望收益/笔', num(a.expectancy, 4), cls(a.expectancy || 0)],
    ['盈亏比', num(a.payoff_ratio), ''],
    ['最大连亏', a.max_consecutive_losses, a.max_consecutive_losses >= 5 ? 'dn' : ''],
    ['平均持仓(日)', num(a.avg_hold_bars, 1), ''],
    ['暴露时间', pct(a.exposure_time), ''],
    ['MAE 均值', pct(mae), 'dn'],
    ['MFE 均值', pct(mfe), 'up'],
    ['日 VaR95', pct(d.var_95_daily), 'dn'],
    ['尾部比', num(d.tail_ratio), ''],
    ['收益稳定性', num(d.stability), ''],
    ['最长回撤(日)', d.longest_drawdown_bars, ''],
  ].map(([k, v, c]) => `<div class="card"><div class="k">${k}</div><div class="v ${c}">${v}</div><div class="n">${k === 'SQN 策略质量' ? (a.sqn_grade || '') : ''}</div></div>`).join('');

  // 滚动夏普 + 滚动回撤
  const rl = r.rolling || {};
  if (rl.sharpe && rl.sharpe.dates.length) {
    chart('chartRolling', {
      tooltip: { trigger: 'axis' },
      legend: { data: ['滚动夏普', '滚动回撤 %'], top: 2, textStyle: { fontSize: 11 } },
      grid: { left: 58, right: 58, top: 32, bottom: 50 },
      xAxis: { type: 'category', data: rl.sharpe.dates, ...AX, splitLine: { show: false } },
      yAxis: [
        { type: 'value', name: '夏普', ...AX },
        { type: 'value', name: '回撤%', ...AX, splitLine: { show: false } },
      ],
      dataZoom: [{ type: 'inside' }],
      series: [
        { name: '滚动夏普', type: 'line', data: rl.sharpe.values, symbol: 'none', lineStyle: { color: '#2563eb', width: 1.5 } },
        { name: '滚动回撤 %', type: 'line', yAxisIndex: 1, data: (rl.max_drawdown.values || []).map(v => v === null ? null : +(v * 100).toFixed(2)), symbol: 'none', lineStyle: { color: DN, width: 1 }, areaStyle: { color: 'rgba(30,142,90,.10)' } },
      ],
    });
  }

  // 年度 × 月度热力图
  const mt = r.monthly_table || {};
  if (mt.years && mt.years.length) {
    const data = [];
    mt.matrix.forEach((row, yi) => {
      row.forEach((v, mi) => { if (v !== null) data.push([mi, yi, +(v * 100).toFixed(2)]); });
    });
    chart('chartHeat', {
      tooltip: { formatter: p => `${mt.years[p.value[1]]} 年 ${p.value[0] + 1} 月：${p.value[2]}%` },
      grid: { left: 58, right: 18, top: 16, bottom: 50 },
      xAxis: { type: 'category', data: [...Array(13)].map((_, i) => i === 12 ? '全年' : (i + 1) + '月'), ...AX, splitLine: { show: false } },
      yAxis: { type: 'category', data: mt.years.map(String), ...AX, splitLine: { show: false } },
      visualMap: { min: -20, max: 20, calculable: true, orient: 'horizontal', left: 'center', bottom: 6, inRange: { color: ['#1e8e5a', '#f5f6f8', '#c0392b'] } },
      series: [{
        type: 'heatmap', data,
        label: { show: true, fontSize: 9, formatter: p => p.value[2] + '%' },
        itemStyle: { borderColor: '#fff', borderWidth: 1 },
      }],
    });
  }

  // 分年收益表
  const yr = r.yearly || [];
  $('yearlyTable').innerHTML =
    `<thead><tr><th>年份</th><th>策略收益</th><th>基准(买入持有)</th><th>超额</th><th>当年最大回撤</th><th>交易日</th></tr></thead><tbody>` +
    yr.map(y => `<tr><td>${y.year}</td><td class="${cls(y.return)}">${pct(y.return)}</td>` +
      `<td class="${cls(y.benchmark || 0)}">${pct(y.benchmark)}</td>` +
      `<td class="${cls(y.excess || 0)}">${pct(y.excess)}</td>` +
      `<td class="dn">${pct(y.max_drawdown)}</td><td>${y.bars}</td></tr>`).join('') + `</tbody>`;

  // 分布诊断表
  const distRows = [
    ['日 VaR 95%', pct(d.var_95_daily)],
    ['日 CVaR 95%（尾部均值）', pct(d.cvar_95_daily)],
    ['年化 VaR 95%', pct(d.var_95_annual)],
    ['尾部比（右尾/左尾）', num(d.tail_ratio)],
    ['Omega(0)', num(d.omega)],
    ['偏度', num(d.skewness)],
    ['超额峰度', num(d.kurtosis)],
    ['收益稳定性 R²', num(d.stability)],
    ['恢复因子（收益/回撤）', num(d.recovery_factor)],
    ['Ulcer 指数', num(d.ulcer_index)],
    ['处于回撤中的时间占比', pct(d.pct_bars_in_drawdown)],
    ['最长回撤持续（日）', d.longest_drawdown_bars],
  ];
  $('distTable').innerHTML =
    `<thead><tr><th>指标</th><th>数值</th></tr></thead><tbody>` +
    distRows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join('') + `</tbody>`;

  // 出场原因
  const br = a.by_reason || {};
  if (Object.keys(br).length > 1) {
    $('distTable').innerHTML +=
      `<thead><tr><th>出场原因</th><th>次数 / 均值 / 胜率</th></tr></thead><tbody>` +
      Object.entries(br).map(([k, v]) =>
        `<tr><td>${k}</td><td>${v.count} 笔 · 均值 ${num(v.mean, 4)} · 胜率 ${pct(v.win_rate)}</td></tr>`).join('') + `</tbody>`;
  }
}

/* ---------------- 稳健性验证 ---------------- */
async function runRobust() {
  setStatus('稳健性检验运行中（含数百次随机对照，需十几秒）…');
  try {
    const r = await fetch('/api/robust', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(baseReq({
        metric: $('rbMetric').value,
        n_boot: parseInt($('nBoot').value || 800),
        n_iter: parseInt($('nIter').value || 300),
        n_trials: parseInt($('nTrials').value || 200),
      })),
    }).then(r => r.json());
    if (r.detail) throw new Error(r.detail);
    renderRobust(r);
    setStatus('稳健性检验完成');
  } catch (e) { setStatus('检验失败：' + e.message, true); }
}

function renderRobust(r) {
  const b = r.bootstrap || {}, rt = r.random_test || {}, dsr = r.deflated_sharpe || {}, pl = r.plateau || {};
  const rm = r.real_metrics || {};

  // 综合结论：四项各自独立，这里把它们拼成一句可执行的判断
  const sigs = [];
  if (rt.p_value !== undefined) sigs.push(rt.p_value < 0.05
    ? `随机对照 <b>p=${rt.p_value.toFixed(3)}</b>，择时显著优于随机入场`
    : `随机对照 <b>p=${rt.p_value.toFixed(3)}</b>，<b>与随机入场无显著差异</b>——收益很可能主要来自标的本身的 beta，而非择时`);
  if (dsr.deflated_sharpe !== undefined) sigs.push(dsr.significant_at_95
    ? `通缩夏普 <b>${dsr.deflated_sharpe.toFixed(3)}</b>，扣除试验次数后仍显著`
    : `通缩夏普 <b>${dsr.deflated_sharpe.toFixed(3)}</b>，考虑你扫过的参数量后<b>不再显著</b>`);
  if (pl.overall) sigs.push(pl.overall.stable
    ? `参数位于<b>高原</b>，微调不敏感`
    : `参数位于<b>孤岛</b>，微调一档即明显衰减`);
  if (b.prob_sharpe_positive !== undefined) sigs.push(
    `自助重采样中夏普为正的比例 <b>${(b.prob_sharpe_positive * 100).toFixed(0)}%</b>`);

  const caution = (rt.p_value !== undefined && rt.p_value >= 0.05) || (pl.overall && !pl.overall.stable);
  $('verdict').innerHTML =
    `<div class="vtitle ${caution ? 'warn' : 'ok'}">${caution ? '⚠ 存在过拟合或 beta 伪装风险' : '✓ 四项检验未发现明显过拟合信号'}</div>` +
    `<ul>${sigs.map(s => `<li>${s}</li>`).join('')}</ul>` +
    `<div class="vnote">注意：<b>通缩夏普</b>问的是「夏普是否显著 &gt; 0」，<b>随机对照</b>问的是「是否优于随机择时」。
     两者可以不一致——比如策略显著为正但只是跟着标的涨，这时 DSR 通过、随机对照不通过，说明<b>择时没有增量信息</b>。</div>`;

  // ① 自助置信区间
  const bandRows = [];
  const labels = { total_return: '总收益', sharpe: '夏普比率', cagr: '年化收益', max_drawdown: '最大回撤' };
  Object.keys(labels).forEach(k => {
    const v = b.bands && b.bands[k];
    if (!v) return;
    bandRows.push([labels[k], pct(v['5']), num(v['5'], 3), pct(v['50']), pct(v['95']), pct(rm[k])]);
  });
  $('bootTable').innerHTML =
    `<thead><tr><th>指标</th><th>真实值</th><th>5% 分位</th><th>中位数</th><th>95% 分位</th><th>说明</th></tr></thead><tbody>` +
    bandRows.map(x => `<tr><td>${x[0]}</td><td><b>${x[1]}</b></td><td>${x[2]}</td><td>${x[3]}</td><td>${x[4]}</td><td>真实值落在区间内 = 结果可复现</td></tr>`).join('') +
    `</tbody>`;
  $('rbInfo').textContent = `块自助 ${b.n_boot} 次（块长 ${b.block}）· 随机对照 ${rt.n_iter} 次`;

  if (b.hist && b.hist.sharpe) {
    const h = b.hist.sharpe;
    chart('chartBoot', {
      tooltip: { trigger: 'axis', formatter: p => `${p[0].axisValue}：${p[0].data} 次` },
      grid: { left: 58, right: 20, top: 16, bottom: 40 },
      xAxis: { type: 'category', data: h.map(x => x.x0.toFixed(2)), ...AX, splitLine: { show: false }, axisLabel: { color: '#6b7280', fontSize: 9 } },
      yAxis: { type: 'value', name: '次数', ...AX },
      series: [{
        type: 'bar', data: h.map(x => x.n), itemStyle: { color: '#93c5fd' },
        markLine: {
          symbol: 'none',
          data: [{ xAxis: h.findIndex(x => x.x1 >= rm.sharpe), lineStyle: { color: UP, width: 2 }, label: { formatter: '真实 ' + num(rm.sharpe, 2), fontSize: 10 } }],
        },
      }],
    });
  }

  // ② 随机对照
  $('randCards').innerHTML = [
    ['你的策略', num(rt.real, 3), 'up'],
    ['随机均值', num(rt.random_mean, 3), ''],
    ['随机标准差', num(rt.random_std, 3), ''],
    ['随机 95 分位', num(rt.random_p95, 3), ''],
    ['你的百分位', (rt.percentile || 0).toFixed(0) + '%', rt.percentile > 95 ? 'up' : ''],
    ['p 值', num(rt.p_value, 3), rt.p_value < 0.05 ? 'up' : 'dn'],
  ].map(([k, v, c]) => `<div class="card"><div class="k">${k}</div><div class="v ${c}">${v}</div></div>`).join('');

  if (rt.hist) {
    const h = rt.hist;
    chart('chartRand', {
      tooltip: { trigger: 'axis', formatter: p => `${p[0].axisValue}：${p[0].data} 次` },
      grid: { left: 58, right: 20, top: 16, bottom: 40 },
      xAxis: { type: 'category', data: h.map(x => x.x0.toFixed(2)), ...AX, splitLine: { show: false }, axisLabel: { color: '#6b7280', fontSize: 9 } },
      yAxis: { type: 'value', name: '次数', ...AX },
      series: [{
        type: 'bar', data: h.map(x => x.n), itemStyle: { color: '#cbd5e1' },
        markLine: {
          symbol: 'none',
          data: [
            { xAxis: h.findIndex(x => x.x1 >= rt.real), lineStyle: { color: UP, width: 2 }, label: { formatter: '你的策略', fontSize: 10 } },
            { xAxis: h.findIndex(x => x.x1 >= rt.random_p95), lineStyle: { color: '#f59e0b', width: 1.5, type: 'dashed' }, label: { formatter: '随机95%', fontSize: 10 } },
          ],
        },
      }],
    });
  }

  // ③ 通缩夏普
  $('dsrTable').innerHTML =
    `<thead><tr><th>项目</th><th>数值</th><th>含义</th></tr></thead><tbody>` +
    [
      ['观测夏普', num(dsr.observed_sharpe, 3), '回测直接算出来的值'],
      ['试验次数 N', dsr.n_trials, '你在参数网格上试了多少组（越大越要打折）'],
      ['零假设下期望最大夏普', num(dsr.expected_max_sharpe_under_null, 3), '纯运气时，扫 N 组能扫到的最高夏普'],
      ['夏普标准误', num(dsr.std_error, 3), '含偏度/峰度修正'],
      ['通缩夏普 DSR', num(dsr.deflated_sharpe, 4), '&gt; 0.95 才算通过'],
    ].map(([k, v, h]) => `<tr><td>${k}</td><td><b>${v}</b></td><td>${h}</td></tr>`).join('') +
    `<tr><td>结论</td><td colspan="2">${dsr.verdict || '—'}</td></tr></tbody>`;

  // ④ 参数高原
  const areas = pl.areas || [];
  if (areas.length) {
    $('plateauVerdict').innerHTML = pl.overall
      ? `<b>${pl.overall.verdict}</b> · 中心 ${num(pl.overall.center_score, 3)} / 最大衰减 ${num(pl.overall.worst_neighbor_drop, 3)} / 阈值 ${num(pl.overall.threshold, 3)}`
      : '';
    chart('chartPlateau', {
      tooltip: { trigger: 'axis' },
      legend: { top: 2, textStyle: { fontSize: 11 }, data: areas.map(a => a.param) },
      grid: { left: 58, right: 20, top: 34, bottom: 40 },
      xAxis: { type: 'category', data: areas.map(a => a.param + ' 邻域档位'), ...AX, splitLine: { show: false } },
      yAxis: { type: 'value', name: pl.metric || 'metric', ...AX },
      series: areas.map((a, i) => ({
        name: a.param, type: 'bar',
        data: a.scores.map(s => s === null ? null : +Number(s).toFixed(3)),
        itemStyle: { color: ['#2563eb', '#f59e0b', '#1e8e5a', '#7c3aed'][i % 4] },
        label: { show: true, fontSize: 9, position: 'top' },
      })),
    });
  }
}

/* ---------------- 组合回测 ---------------- */
async function runPortfolio() {
  setStatus('组合回测中…');
  try {
    const symbols = $('pfSymbols').value.split(/[,，\s]+/).map(s => s.trim()).filter(Boolean);
    const r = await fetch('/api/portfolio', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        symbols,
        market: $('pfMarket').value,
        start: $('start').value,
        end: $('end').value,
        mode: $('pfMode').value,
        weighting: $('pfWeight').value,
        lookback: parseInt($('pfLookback').value || 60),
        top_k: parseInt($('pfTopK').value || 2),
        rebalance: $('pfReb').value,
        cost_market: $('costMarket').value,
        initial_cash: parseFloat($('cash').value || 1000000),
      }),
    }).then(r => r.json());
    if (r.detail) throw new Error(r.detail);
    renderPortfolio(r);
    setStatus(`组合完成 · ${r.symbol}`);
  } catch (e) { setStatus('组合回测失败：' + e.message, true); }
}

function renderPortfolio(r) {
  const m = r.metrics || {};
  $('pfCards').innerHTML = [
    ['总收益', pct(m.total_return), cls(m.total_return || 0)],
    ['年化 CAGR', pct(m.cagr), cls(m.cagr || 0)],
    ['夏普', num(m.sharpe), ''],
    ['最大回撤', pct(m.max_drawdown), 'dn'],
    ['年换手率', num(m.turnover_per_year), ''],
    ['再平衡次数', m.rebalance_days, ''],
    ['标的数', m.n_assets, ''],
  ].map(([k, v, c]) => `<div class="card"><div class="k">${k}</div><div class="v ${c}">${v}</div></div>`).join('');

  chart('chartPF', {
    tooltip: { trigger: 'axis' },
    legend: { data: ['组合净值', '回撤 %'], top: 2, textStyle: { fontSize: 11 } },
    grid: { left: 62, right: 58, top: 34, bottom: 50 },
    xAxis: { type: 'category', data: r.dates, ...AX, splitLine: { show: false } },
    yAxis: [
      { type: 'value', name: '净值', scale: true, ...AX },
      { type: 'value', name: '回撤%', ...AX, splitLine: { show: false } },
    ],
    dataZoom: [{ type: 'inside' }],
    series: [
      { name: '组合净值', type: 'line', data: r.equity, symbol: 'none', lineStyle: { color: UP, width: 1.8 } },
      { name: '回撤 %', type: 'line', yAxisIndex: 1, data: r.drawdown, symbol: 'none', lineStyle: { color: DN, width: 1 }, areaStyle: { color: 'rgba(30,142,90,.12)' } },
    ],
  });

  const cmp = r.compare || [];
  $('pfCompare').innerHTML =
    `<thead><tr><th>加权方式</th><th>总收益</th><th>年化</th><th>夏普</th><th>最大回撤</th><th>年换手</th></tr></thead><tbody>` +
    cmp.map(c => `<tr><td>${c.method}${c.error ? ' (失败)' : ''}</td>` +
      `<td class="${cls(c.total_return || 0)}">${pct(c.total_return)}</td><td>${pct(c.cagr)}</td>` +
      `<td>${num(c.sharpe)}</td><td class="dn">${pct(c.max_drawdown)}</td><td>${num(c.turnover_per_year)}</td></tr>`).join('') +
    `</tbody>`;

  const ct = Object.entries(r.contribution || {}).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  chart('chartContrib', {
    tooltip: { trigger: 'axis', formatter: p => `${p[0].axisValue}：${(p[0].data * 100).toFixed(2)}%` },
    grid: { left: 58, right: 20, top: 16, bottom: 46 },
    xAxis: { type: 'category', data: ct.map(x => x[0]), ...AX, splitLine: { show: false }, axisLabel: { color: '#6b7280', fontSize: 10, rotate: 30 } },
    yAxis: { type: 'value', name: '累计贡献', ...AX },
    series: [{ type: 'bar', data: ct.map(x => +(x[1] * 100).toFixed(2)), itemStyle: { color: p => p.value >= 0 ? UP : DN } }],
  });

  const cm = r.correlation || {};
  const labels = cm.labels || [];
  const cdata = [];
  (cm.matrix || []).forEach((row, i) => row.forEach((v, j) => { if (v !== null) cdata.push([j, i, +Number(v).toFixed(2)]); }));
  chart('chartCorr', {
    tooltip: { formatter: p => `${labels[p.value[0]]} × ${labels[p.value[1]]} = ${p.value[2]}` },
    grid: { left: 58, right: 18, top: 16, bottom: 46 },
    xAxis: { type: 'category', data: labels, ...AX, splitLine: { show: false }, axisLabel: { color: '#6b7280', fontSize: 10 } },
    yAxis: { type: 'category', data: labels, ...AX, splitLine: { show: false } },
    visualMap: { min: -1, max: 1, calculable: true, orient: 'horizontal', left: 'center', bottom: 4, inRange: { color: ['#1e8e5a', '#f5f6f8', '#c0392b'] } },
    series: [{
      type: 'heatmap', data: cdata,
      label: { show: true, fontSize: 9, formatter: p => p.value[2] },
      itemStyle: { borderColor: '#fff', borderWidth: 1 },
    }],
  });

  $('pfInfo').innerHTML =
    `区间 ${r.range.start} → ${r.range.end}（${r.range.bars} 个共同交易日）· ` +
    `平均两两相关 <b>${num(cm.avg_pairwise_corr)}</b>` +
    (cm.avg_pairwise_corr > 0.7 ? ' —— <b>相关性过高，分散效果有限</b>' : '') +
    ` · 当前权重 ${JSON.stringify(r.current_weights)}` +
    (r.failed && r.failed.length ? ` · 载入失败：${r.failed.join('，')}` : '');
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
