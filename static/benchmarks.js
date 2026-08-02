// ── Tab switching ────────────────────────────────────────────────────────────

let benchmarksData = null;

document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const tabId = btn.dataset.tab;
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    document.getElementById('tab-' + tabId).classList.add('active');
    if (tabId === 'live') {
      startLivePolling();
    } else {
      stopLivePolling();
      if (tabId === 'benchmarks') fetchBenchmarks();
      if (tabId === 'control') document.dispatchEvent(new CustomEvent('dashboard:control-active'));
    }
  });
});

// ── Sub-tab switching ────────────────────────────────────────────────────────

document.querySelectorAll('.sub-tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const subId = btn.dataset.subtab;
    document.querySelectorAll('.sub-tab-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    document.querySelectorAll('.sub-tab-panel').forEach(p => p.classList.remove('active'));
    document.getElementById('subtab-' + subId).classList.add('active');
  });
});

// ── Benchmark data fetch ─────────────────────────────────────────────────────

async function fetchBenchmarks() {
  try {
    const resp = await fetch('/api/benchmarks', { cache: 'no-store' });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    benchmarksData = await resp.json();
    renderAllBenchmarks(benchmarksData);
  } catch (e) {
    document.querySelectorAll('.sub-tab-panel').forEach(p => {
      p.innerHTML = `<div class="bm-empty">Error loading benchmarks: ${he(e.message)}</div>`;
    });
  }
}

function renderOneshotModel(entry) {
  const details = [
    entry.target,
    entry.started,
    entry.run_id ? entry.run_id.slice(0, 8) : '',
  ].filter(Boolean);
  const detailHtml = details.length
    ? `<div style="font-size:10px;color:var(--text-dim);margin-top:3px">${details.map(he).join(' · ')}</div>`
    : '';
  return `<strong>${he(entry.model)}</strong>${detailHtml}`;
}

function renderAllBenchmarks(data) {
  document.getElementById('subtab-oneshot').innerHTML = renderOneshotTable(data.oneshot_table);
  document.getElementById('subtab-multiturn').innerHTML = renderMultiturnLangTable(data.multiturn_lang_table);
  document.getElementById('subtab-heatmap').innerHTML = renderHeatmap(data.oneshot_table);
  document.getElementById('subtab-quantization').innerHTML = renderQuantization(data.quant_comparison_table, data.comparison_table);
  document.getElementById('subtab-tokencost').innerHTML = renderTokenCost(data.token_cost_table, data.multiturn_leaderboard);
}

// ── Oneshot table ────────────────────────────────────────────────────────────

function renderOneshotTable(tbl) {
  if (!tbl || tbl.length === 0) return '<div class="bm-empty">No oneshot data</div>';
  const langs = ['python', 'javascript', 'go', 'rust', 'cpp', 'java'];
  const sorted = [...tbl].sort((a, b) => b.overall - a.overall);
  let html = '<div style="overflow-x:auto"><table class="data-table"><thead><tr>';
  html += '<th>Model</th><th>Params</th><th>Quant</th><th>Overall</th>';
  langs.forEach(l => { html += `<th>${l}</th>`; });
  html += '</tr></thead><tbody>';
  sorted.forEach(entry => {
    const overallColor = entry.overall >= 80 ? 'var(--green)' : entry.overall >= 50 ? 'var(--yellow)' : 'var(--red)';
    html += `<tr><td>${renderOneshotModel(entry)}</td>`;
    html += `<td class="num">${he(entry.params || '--')}</td>`;
    html += `<td>${he(entry.quant || '--')}</td>`;
    html += `<td class="num"><div style="color:${overallColor};font-weight:600">${entry.overall}%</div><div class="table-bar-wrap"><div class="table-bar" style="width:${entry.overall}%;background:${overallColor}"></div></div></td>`;
    langs.forEach(l => {
      const d = entry.per_lang[l];
      if (d != null) {
        const pct = d.pct;
        const color = pct >= 80 ? 'var(--green)' : pct >= 50 ? 'var(--yellow)' : 'var(--red)';
        html += `<td class="num"><div style="color:${color};font-weight:600">${pct}%</div><div class="table-bar-wrap"><div class="table-bar" style="width:${pct}%;background:${color}"></div></div><div style="font-size:10px;color:var(--text-dim);margin-top:2px">${d.ok}/${d.total}</div></td>`;
      } else {
        html += '<td class="num" style="color:var(--text-dim)">--</td>';
      }
    });
    html += '</tr>';
  });
  html += '</tbody></table></div>';
  return html;
}

// ── Multi-turn per-lang table ────────────────────────────────────────────────

function renderMultiturnLangTable(langTable) {
  if (!langTable || langTable.length === 0) return '<div class="bm-empty">No multi-turn data</div>';
  const langs = ['python', 'javascript', 'go', 'rust', 'cpp', 'java'];
  let html = '<div style="overflow-x:auto"><table class="data-table"><thead><tr>';
  html += '<th>Model</th><th>Params</th><th>Quant</th><th>pass@1</th><th>pass@2</th>';
  langs.forEach(l => { html += `<th>${l}</th>`; });
  html += '</tr></thead><tbody>';
  const sorted = [...langTable].sort((a, b) => parseFloat(b.pass2) - parseFloat(a.pass2));
  sorted.forEach(entry => {
    const p2color = parseFloat(entry.pass2) >= 80 ? 'var(--green)' : parseFloat(entry.pass2) >= 50 ? 'var(--yellow)' : 'var(--red)';
    html += `<tr><td><strong>${he(entry.model)}</strong></td>`;
    html += `<td class="num">${he(entry.params)}</td>`;
    html += `<td>${he(entry.quant)}</td>`;
    html += `<td class="num">${entry.pass1}</td>`;
    html += `<td class="num" style="color:${p2color};font-weight:600">${entry.pass2}</td>`;
    langs.forEach(l => {
      const v = entry.per_lang[l];
      if (v != null) {
        const color = v >= 80 ? 'var(--green)' : v >= 50 ? 'var(--yellow)' : 'var(--red)';
        html += `<td class="num"><div style="color:${color};font-weight:600">${v}</div></td>`;
      } else {
        html += '<td class="num" style="color:var(--text-dim)">--</td>';
      }
    });
    html += '</tr>';
  });
  html += '</tbody></table></div>';
  return html;
}

// ── Language heatmap (uses oneshot_table) ────────────────────────────────────

function renderHeatmap(tbl) {
  if (!tbl || tbl.length === 0) return '<div class="bm-empty">No oneshot data</div>';
  const langs = ['python', 'javascript', 'go', 'rust', 'cpp', 'java'];
  let html = '<div style="overflow-x:auto"><table class="data-table"><thead><tr>';
  html += '<th>Model</th>';
  langs.forEach(l => { html += `<th>${l}</th>`; });
  html += '<th>Overall</th></tr></thead><tbody>';
  const sorted = [...tbl].sort((a, b) => b.overall - a.overall);
  sorted.forEach(entry => {
    const overallColor = entry.overall >= 80 ? '0,230,118' : entry.overall >= 50 ? '255,202,40' : '255,82,82';
    const overallAlpha = (entry.overall / 100 * 0.35).toFixed(2);
    html += `<tr><td>${renderOneshotModel(entry)}</td>`;
    langs.forEach(l => {
      const d = entry.per_lang[l];
      if (d != null) {
        const pct = d.pct;
        const bgAlpha = (pct / 100 * 0.35).toFixed(2);
        const color = pct >= 80 ? '0,230,118' : pct >= 50 ? '255,202,40' : '255,82,82';
        html += `<td><div class="hm-cell" style="background:rgba(${color},${bgAlpha});color:rgba(${color},0.9)">${pct}%</div></td>`;
      } else {
        html += '<td style="text-align:center;color:var(--text-dim)">--</td>';
      }
    });
    html += `<td><div class="hm-cell" style="background:rgba(${overallColor},${overallAlpha});color:rgba(${overallColor},0.9)">${entry.overall}%</div></td>`;
    html += '</tr>';
  });
  html += '</tbody></table></div>';
  return html;
}

// ── Quantization comparison ──────────────────────────────────────────────────

function renderQuantization(quantTable, comparisonTable) {
  let html = '';

  if (quantTable && quantTable.length > 0) {
    html += '<div class="section-title" style="margin-bottom:8px;">Quantization Impact (pass@2)</div>';
    html += '<div style="overflow-x:auto;margin-bottom:20px"><table class="data-table"><thead><tr>';
    html += '<th>Model</th><th>Params</th><th>BF16</th><th>FP8</th><th>NVFP4</th><th>Delta</tr></thead><tbody>';
    quantTable.forEach(entry => {
      const parsePct = (v) => parseFloat(v) || 0;
      const bf16Val = parsePct(entry.bf16);
      const fp8Val = parsePct(entry.fp8);
      const nvfp4Val = parsePct(entry.nvfp4);
      const bf16Color = bf16Val >= 80 ? 'var(--green)' : bf16Val >= 50 ? 'var(--yellow)' : 'var(--red)';
      const fp8Color = fp8Val >= 80 ? 'var(--green)' : fp8Val >= 50 ? 'var(--yellow)' : 'var(--red)';
      const nvfp4Color = nvfp4Val >= 80 ? 'var(--green)' : nvfp4Val >= 50 ? 'var(--yellow)' : 'var(--red)';
      html += `<tr><td><strong>${he(entry.model)}</strong></td><td class="num">${he(entry.params)}</td>`;
      html += `<td class="num"><div style="color:${bf16Color};font-weight:600">${entry.bf16}</div></td>`;
      html += `<td class="num"><div style="color:${fp8Color};font-weight:600">${entry.fp8}</div></td>`;
      html += `<td class="num"><div style="color:${nvfp4Color};font-weight:600">${entry.nvfp4}</div></td>`;
      html += `<td class="num" style="color:var(--orange)">${he(entry.delta)}</td></tr>`;
    });
    html += '</tbody></table></div>';
  }

  if (comparisonTable && comparisonTable.length > 0) {
    html += '<div class="section-title" style="margin-bottom:8px;">DFlash vs Standard Comparison</div>';
    html += '<div style="overflow-x:auto"><table class="data-table"><thead><tr>';
    html += '<th>Model</th><th>Params</th><th>Config</th><th>Oneshot</th><th>Multi-turn</th></tr></thead><tbody>';
    comparisonTable.forEach(entry => {
      const osColor = entry.oneshot !== 'N/A' ? (parseFloat(entry.oneshot) >= 50 ? 'var(--green)' : 'var(--yellow)') : 'var(--text-dim)';
      const mtColor = entry.multiturn !== 'N/A' ? (parseFloat(entry.multiturn) >= 50 ? 'var(--green)' : 'var(--yellow)') : 'var(--text-dim)';
      html += `<tr><td><strong>${he(entry.model)}</strong></td><td class="num">${he(entry.params)}</td><td>${he(entry.config)}</td>`;
      html += `<td class="num" style="color:${osColor};font-weight:600">${entry.oneshot}</td>`;
      html += `<td class="num" style="color:${mtColor};font-weight:600">${entry.multiturn}</td></tr>`;
    });
    html += '</tbody></table></div>';
  }

  if (!html) return '<div class="bm-empty">No quantization data</div>';
  return html;
}

// ── Token cost ───────────────────────────────────────────────────────────────

function renderTokenCost(tbl, leaderboard) {
  let html = '';

  if (tbl && tbl.length > 0) {
    html += '<div class="section-title" style="margin-bottom:8px;">Oneshot Token Cost</div>';
    const langs = ['python', 'javascript', 'go', 'rust', 'cpp', 'java'];
    const sorted = [...tbl].sort((a, b) => b.avg - a.avg);
    const maxAvg = Math.max(...sorted.map(e => e.avg), 1);
    html += '<div style="overflow-x:auto;margin-bottom:24px"><table class="data-table"><thead><tr>';
    html += '<th>Model</th><th>Quant</th><th>Tests</th><th>Total Tokens</th><th>Avg/Test</th>';
    langs.forEach(l => { html += `<th>${l}</th>`; });
    html += '</tr></thead><tbody>';
    sorted.forEach(entry => {
      const pct = (entry.avg / maxAvg * 100).toFixed(1);
      html += `<tr><td><strong>${he(entry.model)}</strong></td>`;
      html += `<td>${he(entry.quant)}</td>`;
      html += `<td class="num">${entry.tests}</td>`;
      html += `<td class="num">${fmtNum(entry.total, 0)}</td>`;
      html += `<td class="num"><div style="font-weight:600">${fmtNum(entry.avg, 0)}</div><div class="table-bar-wrap"><div class="table-bar" style="width:${pct}%;background:var(--orange)"></div></div></td>`;
      langs.forEach(l => {
        const v = entry.per_lang[l];
        html += `<td class="num">${v != null ? fmtNum(v, 0) : '--'}</td>`;
      });
      html += '</tr>';
    });
    html += '</tbody></table></div>';
  }

  if (leaderboard && leaderboard.length > 0) {
    html += '<div class="section-title" style="margin-bottom:8px;">Multi-turn Token Cost</div>';
    html += '<div style="overflow-x:auto"><table class="data-table"><thead><tr>';
    html += '<th>Model</th><th>Quant</th><th>pass@1</th><th>pass@2</th><th>Total Tokens</th><th>Prompt</th><th>Completion</th><th>Avg/Test</th></tr></thead><tbody>';
    const sorted = [...leaderboard].sort((a, b) => parseFloat(b.pass2) - parseFloat(a.pass2));
    sorted.forEach(entry => {
      const p2color = parseFloat(entry.pass2) >= 80 ? 'var(--green)' : parseFloat(entry.pass2) >= 50 ? 'var(--yellow)' : 'var(--red)';
      html += `<tr><td><strong>${he(entry.model)}</strong></td><td>${he(entry.quant)}</td>`;
      html += `<td class="num">${entry.pass1}</td>`;
      html += `<td class="num" style="color:${p2color};font-weight:600">${entry.pass2}</td>`;
      html += `<td class="num">${fmtNum(entry.total_tokens, 0)}</td>`;
      html += `<td class="num">${fmtNum(entry.prompt_tokens, 0)}</td>`;
      html += `<td class="num">${fmtNum(entry.completion_tokens, 0)}</td>`;
      html += `<td class="num">${fmtNum(entry.avg_per_test, 0)}</td></tr>`;
    });
    html += '</tbody></table></div>';
  }

  if (!html) return '<div class="bm-empty">No token cost data</div>';
  return html;
}
