// ── Unified control plane ────────────────────────────────────────────────────

(() => {
  'use strict';

  const TERMINAL_STATES = new Set([
    'succeeded', 'failed', 'timed_out', 'cancelled', 'interrupted',
    'launch_failed', 'cleanup_failed',
  ]);
  const POLL_MS = 3000;
  const STATUS_POLL_MS = 10000;
  const MAX_VISIBLE_LOG = 512 * 1024;

  let controlActive = false;
  let catalog = null;
  let statuses = new Map();
  let runs = [];
  let statusTimer = null;
  let runTimer = null;
  let servingSelectionInitialized = false;
  let selectedRunId = null;
  let selectedRunOffset = 0;
  let selectedRunTerminal = false;
  let runLogTimer = null;

  const byId = id => document.getElementById(id);
  const escapeHtml = value => String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');

  async function jsonRequest(url, options = {}) {
    const response = await fetch(url, options);
    let payload = null;
    try {
      payload = await response.json();
    } catch (_) {
      payload = null;
    }
    if (!response.ok) {
      const message = payload?.error || `HTTP ${response.status}`;
      const error = new Error(message);
      error.status = response.status;
      error.payload = payload;
      throw error;
    }
    return payload;
  }

  function mutationOptions(payload) {
    return {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Origin': window.location.origin,
      },
      body: JSON.stringify(payload),
    };
  }

  function setPolicyState(kind, text) {
    const lock = byId('control-lock');
    lock.classList.remove('enabled', 'disabled', 'error');
    lock.classList.add(kind);
    byId('control-lock-text').textContent = text;
  }

  function setControlsEnabled(enabled) {
    document.querySelectorAll('#tab-control button, #tab-control input, #tab-control select').forEach(element => {
      if (element.id !== 'run-log-close') element.disabled = !enabled;
    });
  }

  async function loadCatalog() {
    try {
      catalog = await jsonRequest('/api/control/catalog');
      if (!catalog.enabled) {
        setPolicyState('disabled', 'Controls disabled');
        setControlsEnabled(false);
        byId('control-targets').innerHTML = '<div class="control-empty">Read-only dashboard mode. Enable the authenticated control service to operate targets.</div>';
        return;
      }
      setPolicyState('enabled', 'Authenticated / armed');
      setControlsEnabled(true);
      populateTargetSelectors();
      syncServingSelectors();
      await Promise.all([fetchServingStatus(), fetchRuns()]);
    } catch (error) {
      setPolicyState('error', 'Control API unavailable');
      setControlsEnabled(false);
      byId('control-targets').innerHTML = `<div class="control-empty">${escapeHtml(error.message)}</div>`;
    }
  }

  function populateTargetSelectors() {
    const options = catalog.targets
      .map(target => `<option value="${escapeHtml(target)}">${escapeHtml(targetLabel(target))}</option>`)
      .join('');
    byId('serve-target').innerHTML = options;
    byId('bench-target').innerHTML = options;
  }

  function targetLabel(target) {
    return {
      local: 'Local · RTX 6000',
      spark1: 'Spark 1',
      spark2: 'Spark 2',
      spark3: 'Spark 3',
      cluster: 'Cluster · Spark 2 + 3',
    }[target] || target;
  }

  function recipesFor(target, engine = null) {
    return catalog.recipes.filter(recipe => recipe.target === target && (!engine || recipe.engine === engine));
  }

  function runningRecipe(target) {
    const status = statuses.get(target);
    return status?.state === 'running' ? status : null;
  }

  function selectedLaunchProfile(recipe = selectedRecipe()) {
    if (!recipe?.launch_profiles?.length) return null;
    return byId('serve-launch-profile').value || recipe.default_launch_profile;
  }

  function matchesRunningRecipe(recipe, running, launchProfile = selectedLaunchProfile(recipe)) {
    return !running || (
      recipe?.target === running.target
      && recipe?.engine === running.engine
      && recipe?.artifact === running.artifact
      && launchProfile === (running.launch_profile ?? null)
    );
  }


  function syncServingSelectors(preferRunning = false) {
    if (!catalog?.enabled) return;
    const target = byId('serve-target').value || catalog.targets[0];
    const running = preferRunning ? runningRecipe(target) : null;
    const available = recipesFor(target);
    const engines = [...new Set(available.map(recipe => recipe.engine))];
    const previousEngine = byId('serve-engine').value;
    byId('serve-engine').innerHTML = engines
      .map(engine => `<option value="${escapeHtml(engine)}">${escapeHtml(engine.toUpperCase())}</option>`)
      .join('');
    if (running && engines.includes(running.engine)) byId('serve-engine').value = running.engine;
    else if (engines.includes(previousEngine)) byId('serve-engine').value = previousEngine;

    const engine = byId('serve-engine').value;
    const recipes = recipesFor(target, engine);
    const previousArtifact = byId('serve-artifact').value;
    byId('serve-artifact').innerHTML = recipes
      .map(recipe => `<option value="${escapeHtml(recipe.artifact)}">${escapeHtml(recipe.artifact)}</option>`)
      .join('');
    if (running && recipes.some(recipe => recipe.artifact === running.artifact)) {
      byId('serve-artifact').value = running.artifact;
    } else if (recipes.some(recipe => recipe.artifact === previousArtifact)) {
      byId('serve-artifact').value = previousArtifact;
    }
    syncLaunchProfileSelector(selectedRecipe(), running, preferRunning);
  }

  function selectedRecipe() {
    if (!catalog?.enabled) return null;
    const target = byId('serve-target').value;
    const engine = byId('serve-engine').value;
    const artifact = byId('serve-artifact').value;
    return catalog.recipes.find(recipe => recipe.target === target && recipe.engine === engine && recipe.artifact === artifact) || null;
  }

  function syncLaunchProfileSelector(recipe, running = null, preferRunning = false) {
    const field = byId('serve-launch-profile-field');
    const select = byId('serve-launch-profile');
    const profiles = recipe?.launch_profiles || [];
    if (!profiles.length) {
      field.hidden = true;
      select.innerHTML = '';
      updateRecipeReadout();
      return;
    }

    const previous = select.value;
    const names = profiles.map(profile => profile.name);
    select.innerHTML = profiles
      .map(profile => `<option value="${escapeHtml(profile.name)}">${escapeHtml(profile.name.toUpperCase())}</option>`)
      .join('');
    const runningProfile = (
      preferRunning
      && running?.engine === recipe.engine
      && running?.artifact === recipe.artifact
      && names.includes(running.launch_profile)
    ) ? running.launch_profile : null;
    if (runningProfile) select.value = runningProfile;
    else if (names.includes(previous)) select.value = previous;
    else select.value = recipe.default_launch_profile;
    field.hidden = false;
    updateRecipeReadout();
  }

  function formatContextLength(value) {
    if (!Number.isFinite(value)) return '--';
    if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(2).replace(/\.?0+$/, '')}M`;
    return `${Math.round(value / 1000)}K`;
  }

  function renderProfileDetails(recipe) {
    const output = byId('serve-profile-meta');
    const selected = selectedLaunchProfile(recipe);
    const profile = recipe?.launch_profiles?.find(item => item.name === selected);
    if (!profile) {
      output.hidden = true;
      output.innerHTML = '';
      return;
    }
    output.innerHTML = `
      <span>KV cache<strong>${escapeHtml(profile.kv_cache_dtype)}</strong></span>
      <span>Context<strong>${escapeHtml(formatContextLength(profile.context_length))}</strong></span>
      <span>Sequences<strong>${escapeHtml(profile.max_sequences)}</strong></span>
      <span>Spec tokens<strong>MTP${escapeHtml(profile.speculative_tokens)}</strong></span>`;
    output.hidden = false;
  }

  function updateRecipeReadout() {
    const recipe = selectedRecipe();
    const state = statuses.get(recipe?.target);
    const running = runningRecipe(recipe?.target);
    if (!recipe) {
      byId('serve-recipe-meta').textContent = 'No recipe is available for this target and engine.';
      byId('serve-selection-state').textContent = 'NO RECIPE';
      renderProfileDetails(null);
      return;
    }
    renderProfileDetails(recipe);
    if (state?.state === 'conflict') {
      const profiles = (state.running_profiles || [])
        .map(item => item.launch_profile || `${item.engine}/${item.artifact}`)
        .join(', ');
      byId('serve-recipe-meta').textContent = `${targetLabel(recipe.target)} reports multiple running launch profiles: ${profiles}. Resolve the conflict outside this form before operating the cluster.`;
      byId('serve-selection-state').textContent = 'CONFLICT';
      return;
    }
    if (running && !matchesRunningRecipe(recipe, running)) {
      const activeProfile = running.launch_profile ? ` @ ${running.launch_profile}` : '';
      byId('serve-recipe-meta').textContent = `${targetLabel(recipe.target)} is occupied by ${running.served} · active recipe ${running.engine}/${running.artifact}${activeProfile}. Select and stop that exact lifecycle identity before starting another.`;
      byId('serve-selection-state').textContent = 'OCCUPIED';
      return;
    }
    const launchProfile = selectedLaunchProfile(recipe);
    const profileText = launchProfile ? ` · ${launchProfile} launch profile` : '';
    byId('serve-recipe-meta').textContent = `${recipe.served} · ${recipe.profile} target${profileText} · exact artifact ${recipe.artifact}`;
    byId('serve-selection-state').textContent = state?.state ? state.state.toUpperCase() : 'STATUS UNKNOWN';
  }

  async function fetchServingStatus() {
    if (!catalog?.enabled) return;
    try {
      const payload = await jsonRequest('/api/serving');
      statuses = new Map((payload.targets || []).map(item => [item.target, item]));
      renderTargets(false);
      if (!servingSelectionInitialized) {
        syncServingSelectors(true);
        servingSelectionInitialized = true;
      } else {
        updateRecipeReadout();
      }
    } catch (error) {
      renderTargets(true, error.message);
    }
  }

  function renderTargets(stale, message = '') {
    if (!catalog?.enabled) return;
    byId('control-targets').innerHTML = catalog.targets.map(target => {
      const status = statuses.get(target) || { state: stale ? 'error' : 'untracked' };
      const state = stale ? 'error' : status.state;
      const profile = status.launch_profile ? ` · ${status.launch_profile}` : '';
      const model = status.served ? `${status.served}${profile}` : (
        state === 'conflict' ? 'Multiple launch profiles detected' : 'No tracked recipe'
      );
      const endpoint = status.endpoint || (state === 'untracked' ? 'Awaiting first dashboard operation' : 'Endpoint resolved by adapter');
      const conflict = (status.running_profiles || [])
        .map(item => item.launch_profile || `${item.engine}/${item.artifact}`)
        .join(', ');
      const error = stale ? message : (
        state === 'conflict' ? `Running identities: ${conflict}` : status.error
      );
      return `<article class="target-card" data-state="${escapeHtml(state)}">
        <div class="target-card-head">
          <span class="target-name">${escapeHtml(targetLabel(target))}</span>
          <span class="state-chip ${escapeHtml(state)}">${escapeHtml(state)}</span>
        </div>
        <div class="target-detail"><strong>${escapeHtml(model)}</strong></div>
        <div class="target-detail">${escapeHtml(error || endpoint)}</div>
      </article>`;
    }).join('');
  }

  async function submitServing(action) {
    const recipe = selectedRecipe();
    if (!recipe) return;
    const status = statuses.get(recipe.target);
    if (status?.state === 'conflict') {
      showOperationError(new Error(`${targetLabel(recipe.target)} has multiple running launch profiles. Resolve the conflict before submitting another lifecycle action.`));
      return;
    }
    const launchProfile = selectedLaunchProfile(recipe);
    const running = runningRecipe(recipe.target);
    if (running && !matchesRunningRecipe(recipe, running, launchProfile)) {
      const activeProfile = running.launch_profile ? ` @ ${running.launch_profile}` : '';
      const instruction = action === 'start'
        ? 'Stop that active lifecycle identity before starting another.'
        : 'Select the active lifecycle identity before operating it.';
      showOperationError(new Error(`${targetLabel(recipe.target)} is running ${running.served} (${running.engine}/${running.artifact}${activeProfile}). ${instruction}`));
      return;
    }
    const profileText = launchProfile ? ` using ${launchProfile}` : '';
    if (action === 'stop' && !window.confirm(`Stop ${recipe.served}${profileText} on ${targetLabel(recipe.target)}?`)) return;
    setActionBusy(true);
    try {
      const created = await jsonRequest('/api/runs', mutationOptions({
        kind: 'serving',
        action,
        target: recipe.target,
        engine: recipe.engine,
        artifact: recipe.artifact,
        launch_profile: launchProfile,
      }));
      selectedRunId = created.id;
      await fetchRuns();
      await openRunLog(created.id);
    } catch (error) {
      showOperationError(error);
    } finally {
      setActionBusy(false);
    }
  }

  function benchmarkOptions() {
    const language = byId('bench-lang').value;
    const effort = byId('bench-reasoning-effort').value;
    const keywords = byId('bench-keywords').value
      .split(',')
      .map(value => value.trim())
      .filter(Boolean);
    return {
      lang: language || null,
      num_tests: Number(byId('bench-num-tests').value),
      keywords,
      max_tokens: Number(byId('bench-max-tokens').value),
      temperature: Number(byId('bench-temperature').value),
      timeout: Number(byId('bench-timeout').value),
      test_timeout: Number(byId('bench-test-timeout').value),
      concurrency: Number(byId('bench-concurrency').value),
      reasoning: byId('bench-reasoning').value,
      reasoning_effort: effort || null,
    };
  }

  async function submitBenchmark() {
    if (!document.querySelector('#tab-control input:invalid')) {
      const target = byId('bench-target').value;
      setActionBusy(true);
      try {
        const created = await jsonRequest('/api/runs', mutationOptions({
          kind: 'benchmark',
          benchmark: 'oneshot',
          target,
          options: benchmarkOptions(),
        }));
        selectedRunId = created.id;
        await fetchRuns();
        await openRunLog(created.id);
      } catch (error) {
        showOperationError(error);
      } finally {
        setActionBusy(false);
      }
      return;
    }
    document.querySelector('#tab-control input:invalid').reportValidity();
  }

  function setActionBusy(busy) {
    document.querySelectorAll('[data-serve-action], #bench-run').forEach(button => {
      button.disabled = busy || !catalog?.enabled;
    });
  }

  function showOperationError(error) {
    const occupant = error.payload?.occupying_run_id;
    const suffix = occupant ? ` · occupying run ${occupant}` : '';
    byId('run-poll-state').textContent = `ERROR · ${error.message}${suffix}`;
  }

  async function fetchRuns() {
    if (!catalog?.enabled) return;
    try {
      runs = await jsonRequest('/api/runs?limit=50');
      byId('run-poll-state').textContent = runs.some(run => !TERMINAL_STATES.has(run.state)) ? 'ACTIVE' : 'SYNCED';
      renderRuns();
      if (selectedRunId) {
        const selected = runs.find(run => run.id === selectedRunId);
        selectedRunTerminal = selected ? TERMINAL_STATES.has(selected.state) : true;
      }
    } catch (error) {
      byId('run-poll-state').textContent = `STALE · ${error.message}`;
    }
  }

  function requestSummary(run) {
    const value = run.request || {};
    if (run.kind === 'serving') {
      const profile = value.launch_profile ? ` @ ${value.launch_profile}` : '';
      return `${value.action} · ${value.target} · ${value.engine}/${value.artifact}${profile}`;
    }
    const lang = value.options?.lang || 'all six';
    return `oneshot · ${value.target} · ${lang}`;
  }

  function renderRuns() {
    if (!runs.length) {
      byId('run-history').innerHTML = '<div class="control-empty">No durable operations recorded.</div>';
      return;
    }
    byId('run-history').innerHTML = runs.map(run => {
      const active = !TERMINAL_STATES.has(run.state);
      const canCancel = active && run.state !== 'cancel_requested';
      const results = (run.results || []).map(result =>
        `<a href="${escapeHtml(result.url)}" target="_blank" rel="noopener">${escapeHtml(result.label)}</a>`
      ).join('');
      return `<article class="run-row">
        <div>
          <div class="run-id">${escapeHtml(run.kind.toUpperCase())} <code>${escapeHtml(run.id)}</code></div>
          <div class="run-request">${escapeHtml(requestSummary(run))}</div>
          ${results ? `<div class="run-results">${results}</div>` : ''}
        </div>
        <div>
          <span class="state-chip ${escapeHtml(run.state === 'running' ? 'running-run' : run.state)}">${escapeHtml(run.state)}</span>
          ${run.error_code ? `<div class="run-request">${escapeHtml(run.error_code)}</div>` : ''}
        </div>
        <div class="run-time">${escapeHtml(formatTimestamp(run.started_at || run.created_at))}</div>
        <div class="run-actions">
          <button type="button" class="control-button" data-run-log="${escapeHtml(run.id)}">Log</button>
          ${canCancel ? `<button type="button" class="control-button control-button-danger" data-run-cancel="${escapeHtml(run.id)}">Cancel</button>` : ''}
        </div>
      </article>`;
    }).join('');
  }

  function formatTimestamp(value) {
    if (!value) return '--';
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
  }

  async function cancelRun(runId) {
    try {
      await jsonRequest(`/api/runs/${encodeURIComponent(runId)}/cancel`, mutationOptions({}));
      await fetchRuns();
    } catch (error) {
      showOperationError(error);
    }
  }

  async function openRunLog(runId) {
    selectedRunId = runId;
    selectedRunOffset = 0;
    selectedRunTerminal = false;
    byId('run-log-title').textContent = runId;
    byId('run-log-output').textContent = '';
    byId('run-log-drawer').hidden = false;
    await fetchRunLog();
    clearInterval(runLogTimer);
    runLogTimer = setInterval(fetchRunLog, 1000);
  }

  async function fetchRunLog() {
    if (!selectedRunId || !controlActive) return;
    try {
      const payload = await jsonRequest(
        `/api/runs/${encodeURIComponent(selectedRunId)}/log?offset=${selectedRunOffset}&limit=65536`
      );
      selectedRunOffset = payload.next_offset;
      const output = byId('run-log-output');
      if (payload.data) {
        output.textContent = (output.textContent + payload.data).slice(-MAX_VISIBLE_LOG);
        output.scrollTop = output.scrollHeight;
      }
      if (payload.eof && selectedRunTerminal) {
        clearInterval(runLogTimer);
        runLogTimer = null;
      }
    } catch (error) {
      byId('run-log-output').textContent += `\n[log unavailable: ${error.message}]`;
      clearInterval(runLogTimer);
      runLogTimer = null;
    }
  }

  async function loadServingLogs() {
    const recipe = selectedRecipe();
    if (!recipe) return;
    const lines = byId('serve-log-lines').value;
    const launchProfile = selectedLaunchProfile(recipe);
    const output = byId('serve-log-output');
    output.textContent = 'Loading exact-container log…';
    const query = new URLSearchParams({ lines });
    if (launchProfile) query.set('launch_profile', launchProfile);
    try {
      const payload = await jsonRequest(
        `/api/serving/${encodeURIComponent(recipe.target)}/${encodeURIComponent(recipe.engine)}/${encodeURIComponent(recipe.artifact)}/logs?${query}`
      );
      output.textContent = payload.text || '[empty log]';
      if (payload.truncated) output.textContent += '\n[server response cap reached]';
    } catch (error) {
      output.textContent = `[log unavailable: ${error.message}]`;
    }
  }

  function startPolling() {
    clearInterval(statusTimer);
    clearInterval(runTimer);
    statusTimer = setInterval(fetchServingStatus, STATUS_POLL_MS);
    runTimer = setInterval(fetchRuns, POLL_MS);
  }

  function stopPolling() {
    clearInterval(statusTimer);
    clearInterval(runTimer);
    statusTimer = null;
    runTimer = null;
  }

  document.addEventListener('dashboard:control-active', () => {
    controlActive = true;
    if (!catalog) loadCatalog();
    else if (catalog.enabled) Promise.all([fetchServingStatus(), fetchRuns()]);
    startPolling();
  });

  document.querySelectorAll('.tab-btn').forEach(button => {
    button.addEventListener('click', () => {
      if (button.dataset.tab !== 'control') {
        controlActive = false;
        stopPolling();
      }
    });
  });

  byId('serve-target').addEventListener('change', () => syncServingSelectors(true));
  byId('serve-engine').addEventListener('change', () => syncServingSelectors(false));
  byId('serve-artifact').addEventListener('change', () => {
    syncLaunchProfileSelector(selectedRecipe());
  });
  byId('serve-launch-profile').addEventListener('change', updateRecipeReadout);
  document.querySelectorAll('[data-serve-action]').forEach(button => {
    button.addEventListener('click', () => submitServing(button.dataset.serveAction));
  });
  byId('serve-load-logs').addEventListener('click', loadServingLogs);
  byId('bench-run').addEventListener('click', submitBenchmark);
  byId('run-history').addEventListener('click', event => {
    const logButton = event.target.closest('[data-run-log]');
    if (logButton) openRunLog(logButton.dataset.runLog);
    const cancelButton = event.target.closest('[data-run-cancel]');
    if (cancelButton) cancelRun(cancelButton.dataset.runCancel);
  });
  byId('run-log-close').addEventListener('click', () => {
    selectedRunId = null;
    byId('run-log-drawer').hidden = true;
    clearInterval(runLogTimer);
    runLogTimer = null;
  });
})();
