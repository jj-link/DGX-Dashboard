// ── Utilities ────────────────────────────────────────────────────────────────

function fmtBytes(bytes) {
  if (bytes == null || bytes === 0) return 'N/A';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let i = 0;
  let v = bytes;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
  return v.toFixed(1) + ' ' + units[i];
}

function fmtNum(n, decimals = 0) {
  if (n == null) return 'N/A';
  return Number(n).toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
}

function he(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function gaugeColor(pct) {
  if (pct == null) return '#888';
  if (pct < 50) return '#00e676';
  if (pct < 80) return '#ffca28';
  return '#ff5252';
}

function tempColor(t) {
  if (t == null) return '#888';
  if (t < 60) return '#00e676';
  if (t < 75) return '#ffca28';
  return '#ff5252';
}

function svgGauge(pct, color) {
  const r = 48, circ = 2 * Math.PI * r;
  const offset = circ - (pct / 100) * circ;
  return `<svg viewBox="0 0 120 120">
    <circle class="gauge-bg" cx="60" cy="60" r="${r}"/>
    <circle class="gauge-fill" cx="60" cy="60" r="${r}"
      stroke="${color}"
      stroke-dasharray="${circ}"
      stroke-dashoffset="${offset}"/>
  </svg>`;
}

// ── Render GPU card ──────────────────────────────────────────────────────────

function renderGPU(gpu) {
  const util = gpu['utilization.gpu [%]'] ?? null;
  const memUtil = gpu['utilization.memory [%]'] ?? null;
  const memUsed = gpu['memory.used [MiB]'];
  const memTotal = gpu['memory.total [MiB]'];
  const temp = gpu['temperature.gpu'];
  const power = gpu['power.draw [W]'];
  const pwrLimit = gpu['power.limit [W]'];
  const clockG = gpu['clocks.current.graphics [MHz]'];
  const clockM = gpu['clocks.current.memory [MHz]'];
  const name = gpu['name'] ?? 'Unknown';
  const idx = gpu['index'] ?? '?';
  const uuid = gpu['uuid'] ?? '';
  const host = gpu['host'] ?? 'local';

  const utilColor = gaugeColor(util);
  const tempCol = tempColor(temp);

  const memPct = (memUsed != null && memTotal != null) ? (memUsed / memTotal * 100) : null;
  const pwrPct = (power != null && pwrLimit != null) ? (power / pwrLimit * 100) : null;

  return `
  <div class="gpu-card">
    <div class="gpu-header">
      <div class="gpu-name">${name}</div>
      <div class="gpu-index">GPU ${idx}</div>
    </div>
    <div class="gpu-body">
      <div class="gauge">
        ${svgGauge(util ?? 0, utilColor)}
        <div class="gauge-center">
          <div class="gauge-value" style="color:${utilColor}">${util != null ? util : '--'}</div>
          <div class="gauge-label">Util %</div>
        </div>
      </div>
      <div class="gpu-metrics">
        <div class="metric-row">
          <span class="metric-label">Memory</span>
          <div class="metric-bar-container">
            <div class="metric-bar" style="width:${memPct ?? 0}%; background:${gaugeColor(memPct)}"></div>
          </div>
          <span class="metric-value">${memUsed != null && memTotal != null ? fmtNum(memUsed) + ' / ' + fmtNum(memTotal) + ' MiB' : 'N/A'}</span>
        </div>
        <div class="metric-row">
          <span class="metric-label">Temp</span>
          <div class="metric-bar-container">
            <div class="metric-bar" style="width:${temp != null ? temp : 0}%; background:${tempCol}"></div>
          </div>
          <span class="metric-value" style="color:${tempCol}">${temp != null ? temp + '°C' : 'N/A'}</span>
        </div>
        <div class="metric-row">
          <span class="metric-label">Power</span>
          <div class="metric-bar-container">
            <div class="metric-bar" style="width:${pwrPct ?? 0}%; background:var(--orange)"></div>
          </div>
          <span class="metric-value">${power != null ? power.toFixed(1) + 'W' : 'N/A'}${pwrLimit != null ? ' / ' + pwrLimit + 'W' : ''}</span>
        </div>
        <div class="metric-row">
          <span class="metric-label">Clocks</span>
          <span class="metric-value">${clockG != null ? clockG + ' MHz' : 'N/A'}${clockM != null ? ' / ' + clockM + ' MHz' : ''}</span>
        </div>
        <div class="metric-row">
          <span class="metric-label">Driver</span>
          <span class="metric-value">${gpu['driver_version'] ?? 'N/A'}</span>
        </div>
      </div>
    </div>
  </div>`;
}

// ── Render server card ───────────────────────────────────────────────────────

function renderServer(s) {
  const st = s.stats;
  const online = s.online;
  const models = (s.models || []).join(', ');

  let running = 0, queued = 0, throughput = 0;
  let kvCache = 0, weightGb = 0, kvCacheGb = 0;
  let specAcceptLen = 0, specAcceptRate = 0, version = '';
  let tokenUsage = 0, totalTokens = 0;
  let inputThroughput = 0;

  if (s.type === 'sglang') {
    running = st.running_requests ?? st.num_running_reqs;
    queued = st.queued_requests ?? st.num_waiting_reqs;
    throughput = st.gen_throughput ?? st.throughput;
    inputThroughput = st.input_throughput ?? 0;
    kvCache = st.token_usage ?? st.kv_cache_hit_rate ?? 0;
    weightGb = st.weight_gb ?? 0;
    kvCacheGb = st.kv_cache_gb ?? 0;
    specAcceptLen = st.spec_accept_length ?? 0;
    specAcceptRate = st.spec_accept_rate ?? 0;
    version = st.version ?? '';
    tokenUsage = st.token_usage ?? 0;
    totalTokens = st.total_tokens ?? 0;
  }

  const sparkKey = `srv-${s.name}-throughput`;
  pushSpark(sparkKey, throughput);
  const tputColor = 'var(--green)';
  // Show 0 for numeric display when idle
  const displayThroughput = (s.type === 'sglang' && (s.stats.running_requests ?? 0) === 0) ? 0 : throughput;

  const inputSparkKey = `srv-${s.name}-input`;
  pushSpark(inputSparkKey, inputThroughput);
  const inputColor = 'var(--cyan)';
  const displayInputThroughput = (s.type === 'sglang' && (s.stats.running_requests ?? 0) === 0) ? 0 : inputThroughput;

  return `
  <div class="server-card ${online ? '' : 'offline'}">
    <div class="server-header">
      <div class="server-type">${s.type}</div>
    </div>
    ${online ? `
    <div class="server-model">Model: ${models || 'Unknown'}</div>
    <div class="server-stats">
      <div class="server-stat">
        <div class="server-stat-value" style="color:${tputColor}">${displayThroughput != null ? displayThroughput.toFixed(2) : '--'}</div>
        <div class="server-stat-label">Gen tok/s</div>
        ${renderSpark(sparkKey, tputColor)}
      </div>
      <div class="server-stat">
        <div class="server-stat-value" style="color:${inputColor}">${displayInputThroughput != null ? displayInputThroughput.toFixed(1) : '--'}</div>
        <div class="server-stat-label">Input tok/s</div>
        ${renderSpark(inputSparkKey, inputColor)}
      </div>
      <div class="server-stat">
        <div class="server-stat-value">${running ?? '--'}</div>
        <div class="server-stat-label">Running</div>
      </div>
      <div class="server-stat">
        <div class="server-stat-value">${queued ?? '--'}</div>
        <div class="server-stat-label">Queued</div>
      </div>
      <div class="server-stat">
        <div class="server-stat-value">${(kvCache * 100).toFixed(1)}%</div>
        <div class="server-stat-label">KV Cache</div>
      </div>
      <div class="server-stat">
        <div class="server-stat-value">${weightGb ? weightGb.toFixed(1) + ' GB' : '--'}</div>
        <div class="server-stat-label">Weights</div>
      </div>
      <div class="server-stat">
        <div class="server-stat-value">${kvCacheGb ? kvCacheGb.toFixed(1) + ' GB' : '--'}</div>
        <div class="server-stat-label">KV Cache</div>
      </div>
      ${specAcceptLen > 0 ? `
      <div class="server-stat">
        <div class="server-stat-value">${specAcceptLen.toFixed(1)} tok</div>
        <div class="server-stat-label">Spec Accept</div>
      </div>` : ''}
      ${specAcceptRate > 0 ? `
      <div class="server-stat">
        <div class="server-stat-value">${specAcceptRate.toFixed(1)}%</div>
        <div class="server-stat-label">Accept Rate</div>
      </div>` : ''}
      ${version ? `
      <div class="server-stat">
        <div class="server-stat-value">${version}</div>
        <div class="server-stat-label">Version</div>
      </div>` : ''}
    </div>
    ` : `
    <div class="server-error">Offline: ${s.error || 'Unknown error'}</div>
    `}
  </div>`;
}

// ── Sparkline history ──
const sparkData = {};
function pushSpark(key, value, max = 30) {
  if (!sparkData[key]) sparkData[key] = [];
  sparkData[key].push(value);
  if (sparkData[key].length > max) sparkData[key].shift();
}

function renderSpark(key, color) {
  const data = sparkData[key] || [];
  if (data.length < 2) return '';
  const max = Math.max(...data, 1);
  return '<div class="sparkline">' +
    data.map(v => `<div class="sparkline-bar" style="height:${(v/max*100).toFixed(0)}%;background:${color}"></div>`).join('') +
    '</div>';
}

// ── Fetch and update ─────────────────────────────────────────────────────────

async function fetchStats() {
  try {
    const resp = await fetch('/api/stats');
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    updateDashboard(data);
    document.getElementById('status-dot').className = 'status-dot';
    document.getElementById('status-text').textContent = 'Live';
    flash();
  } catch (e) {
    document.getElementById('status-dot').className = 'status-dot error';
    document.getElementById('status-text').textContent = 'Error: ' + e.message;
  }
}

function updateDashboard(data) {
  // System
  const sys = data.system;
  document.getElementById('sys-hostname').textContent = sys.hostname || '--';
  document.getElementById('sys-uptime').textContent = sys.uptime || '--';

  if (sys.mem_total > 0) {
    const pct = (sys.mem_used / sys.mem_total * 100);
    document.getElementById('sys-mem').textContent =
      fmtBytes(sys.mem_used) + ' / ' + fmtBytes(sys.mem_total) + ' (' + pct.toFixed(0) + '%)';
    document.getElementById('sys-mem-bar').style.width = pct + '%';
    document.getElementById('sys-mem-bar').style.background = gaugeColor(pct);
  }

  // Timestamp
  const now = new Date();
  document.getElementById('last-update').textContent =
    now.toTimeString().split(' ')[0];

  // Group GPUs and servers by host
  const hosts = {};
  hosts['local'] = {
    name: sys.hostname || 'local',
    subtitle: 'This machine',
    gpus: [],
    servers: []
  };

  data.gpus.forEach(g => {
    const h = g.host || 'local';
    if (!hosts[h]) {
      hosts[h] = { name: h, subtitle: '', gpus: [], servers: [] };
    }
    hosts[h].gpus.push(g);
  });

  data.servers.forEach(s => {
    let h = 'local';
    if (s.name && s.name !== 'local') {
      h = s.name;
    }
    if (!hosts[h]) {
      hosts[h] = { name: h, subtitle: '', gpus: [], servers: [] };
    }
    hosts[h].servers.push(s);
  });

  // Render as horizontal sections: GPUs row, then servers row
  const container = document.getElementById('host-groups');
  let html = '';

  // GPU section
  const gpuHosts = Object.entries(hosts).filter(([, h]) => h.gpus.length > 0);
  if (gpuHosts.length > 0) {
    html += `<div class="section-row">`;
    gpuHosts.forEach(([key, host]) => {
      html += `<div class="section-col">`;
      html += `<div class="host-header">
        <div class="host-dot"></div>
        <span class="host-name">${host.name.toUpperCase()}</span>
        ${host.subtitle ? `<span class="host-subtitle">${host.subtitle}</span>` : ''}
      </div>`;
      html += `<div class="section-title" style="margin-bottom:8px;">GPUs</div>`;
      html += `<div class="gpu-grid">`;
      host.gpus.forEach((g, i) => { html += renderGPU(g); });
      html += `</div>`;
      html += `</div>`;
    });
    html += `</div>`;
  }

  // Server section
  const serverHosts = Object.entries(hosts).filter(([, h]) => h.servers.length > 0);
  if (serverHosts.length > 0) {
    html += `<div class="section-row">`;
    serverHosts.forEach(([key, host]) => {
      html += `<div class="section-col">`;
      html += `<div class="section-title" style="margin-bottom:8px;margin-top:12px;">Inference Servers</div>`;
      html += `<div class="server-grid">`;
      host.servers.forEach(s => { html += renderServer(s); });
      html += `</div>`;
      html += `</div>`;
    });
    html += `</div>`;
  }

  container.innerHTML = html;
}

function flash() {
  const el = document.getElementById('flash');
  el.classList.remove('active');
  void el.offsetWidth; // reflow
  el.classList.add('active');
}

const refreshIntervalMs = Number(document.body.dataset.refreshInterval) * 1000;
let pollTimer = setInterval(fetchStats, refreshIntervalMs);

fetchStats();
