const loginScreen = document.querySelector('#login-screen');
const loginForm = document.querySelector('#login-form');
const loginToken = document.querySelector('#login-token');
const loginStatus = document.querySelector('#login-status');
const consoleShell = document.querySelector('#console-shell');
const globalNav = document.querySelector('#global-nav');
const moduleNav = document.querySelector('#module-nav');
const viewKicker = document.querySelector('#view-kicker');
const viewTitle = document.querySelector('#view-title');
const globalView = document.querySelector('#global-view');
const moduleView = document.querySelector('#module-view');
const moduleHeading = document.querySelector('#module-heading');
const moduleDescription = document.querySelector('#module-description');
const moduleFields = document.querySelector('#module-fields');
const globalFields = document.querySelector('#global-fields');
const settingsForm = document.querySelector('#settings-form');
const runButton = document.querySelector('#run-button');
const trainQueueButton = document.querySelector('#train-queue-button');
const stopButton = document.querySelector('#stop-button');
const formStatus = document.querySelector('#form-status');
const jobStatus = document.querySelector('#job-status');
const runElapsed = document.querySelector('#run-elapsed');
const command = document.querySelector('#command');
const logs = document.querySelector('#logs');
const searchView = document.querySelector('#search-view');
const searchStart = document.querySelector('#search-start');
const searchStop = document.querySelector('#search-stop');
const searchNote = document.querySelector('#search-note');
const searchModes = ['search_random', 'search_tpe'];
const historyPanel = document.querySelector('#history-panel');
const historyRefresh = document.querySelector('#history-refresh');
const historyStatus = document.querySelector('#history-status');
const historyRows = document.querySelector('#history-rows');
const historyDetail = document.querySelector('#history-detail');
const historySummary = document.querySelector('#history-summary');
const historyLogs = document.querySelector('#history-logs');
const historyHead = document.querySelector('#history-head');
const groupResult = document.querySelector('#group-result');
const segmentResult = document.querySelector('#segment-result');
let historyColumns = [];
let historyRunning = false;
let historyStateTimer = null;
let historyVersion = 0;
let detailVersion = 0;
let segmentTimer = null;
const restartButton = document.querySelector('#restart-button');
const logoutButton = document.querySelector('#logout-button');
const fileDialog = document.querySelector('#file-dialog');
const fileDialogTitle = document.querySelector('#file-dialog-title');
const fileDialogClose = document.querySelector('#file-dialog-close');
const browserRoots = document.querySelector('#browser-roots');
const browserUp = document.querySelector('#browser-up');
const browserPath = document.querySelector('#browser-path');
const browserStatus = document.querySelector('#browser-status');
const browserEntries = document.querySelector('#browser-entries');
const browserSelectDirectory = document.querySelector('#browser-select-directory');
const browserCancel = document.querySelector('#browser-cancel');

let schemas = {};
let moduleOrder = [];
let selectedModule = null;
let schemasLoaded = false;
let currentJob = null;
let nextOffset = 0;
let pollTimer = null;
let elapsedTimer = null;
let elapsedStartedAt = null;
let pickerTarget = null;
let pickerMode = 'file';
let pickerRoot = 'home';
let pickerPath = '';
let pickerListing = null;
let formState = emptyFormState();
let stateSaveTimer = null;

const globalOptions = [
  { dest: 'seed', label: '随机种子', placeholder: '默认 42' },
  { dest: 'log_level', label: '日志等级', choices: ['DEBUG', 'INFO', 'WARNING', 'ERROR'], default: 'DEBUG' },
  { dest: 'console', label: '显示控制台日志', is_flag: true, default: true, help_text: '默认启用；后台线程会持续读取输出，不等待网页轮询。' },
];

const moduleDisplayDefaults = {
  preprocess: {
    altitude_unit: 'm',
    speed_unit: 'km/h',
    heading_unit: 'deg',
  },
};

const splitColumns = [
  { key: 'train', label: '训练' },
  { key: 'val', label: '验证' },
  { key: 'test', label: '测试' },
];
const splitDisplayDefaults = {
  partition_percentages: ['70', '15', '15'],
  normal_percentages: ['50', '50', '50'],
  anomaly_percentages: ['50', '50', '50'],
};
const splitRatioDestinations = new Set(['ratios', 'normal_ratios', 'anomaly_ratios']);

function fieldId(scope, dest) { return `${scope}-${dest}`; }

function createField(scope, spec) {
  const label = document.createElement('label');
  label.className = `field${spec.takes_multiple ? ' full' : ''}`;
  const id = fieldId(scope, spec.dest);
  if (spec.is_flag) {
    label.classList.add('checkbox');
    const input = document.createElement('input');
    input.type = 'checkbox'; input.id = id; input.name = spec.dest; input.checked = spec.default === true;
    label.append(input, document.createTextNode(spec.label || spec.option));
  } else {
    const title = document.createElement('span');
    title.textContent = spec.label || spec.option;
    if (spec.required) { const required = document.createElement('b'); required.textContent = ' *'; title.append(required); }
    const hasChoices = Array.isArray(spec.choices) && spec.choices.length > 0;
    const upstreamLabel = Array.isArray(spec.upstream_module) ? spec.upstream_module.join(' 或 ') : spec.upstream_module;
    const hasUpstream = typeof upstreamLabel === 'string' && upstreamLabel.length > 0;
    const input = hasChoices || hasUpstream ? document.createElement('select') : document.createElement('input');
    input.id = id; input.name = spec.dest; input.required = Boolean(spec.required);
    if (hasUpstream) {
      input.add(new Option(`正在读取已发布的 ${upstreamLabel} 运行…`, ''));
      input.disabled = true; label.append(title, input); void populateUpstreamRuns(input, spec);
    } else if (hasChoices) {
      const empty = new Option(spec.required ? '请选择' : '使用默认值', ''); input.add(empty);
      spec.choices.forEach((choice) => input.add(new Option(choice, choice)));
      if (spec.default !== undefined && spec.default !== null) input.value = String(spec.default);
      label.append(title, input);
    } else {
      input.type = spec.input_type || 'text';
      input.placeholder = spec.placeholder || (spec.takes_multiple ? '多个值以空格分隔' : '留空使用默认值');
      if (spec.default !== undefined && spec.default !== null) input.value = String(spec.default);
      label.append(title);
      if (spec.is_path) {
        const inputWithButton = document.createElement('div'); inputWithButton.className = 'input-with-button';
        const browse = document.createElement('button'); browse.type = 'button'; browse.className = 'secondary browse-button'; browse.textContent = '浏览…';
        browse.addEventListener('click', () => openFileBrowser(input, spec.path_mode || 'file', spec.label || spec.option));
        inputWithButton.append(input, browse); label.append(inputWithButton);
      } else label.append(input);
    }
  }
  if (spec.help_text) { const hint = document.createElement('small'); hint.className = 'hint'; hint.textContent = spec.help_text; label.append(hint); }
  return label;
}

function renderGlobal() {
  globalFields.replaceChildren(); globalOptions.forEach((spec) => globalFields.append(createField('global', spec)));
  applyFields('global', globalOptions, formState.global);
}

function renderModule() {
  moduleFields.replaceChildren();
  const implemented = isModuleImplemented(selectedModule);
  moduleDescription.hidden = !implemented;
  runButton.parentElement.hidden = !implemented;
  if (!implemented) return;
  const specs = schemas[selectedModule] || [];
  const visibleSpecs = specs.filter((spec) => !spec.auto_derived);
  moduleDescription.textContent = selectedModule === 'split'
    ? '先按来源航班划分训练、验证和测试集合，再记录未来 inject 应保持正常或加噪的航班计划。这里的“异常”尚未注入。'
    : '参数由模块 api.py 的实际 argparse 定义生成。';
  if (selectedModule === 'split') {
    renderSplitModule(visibleSpecs, moduleValuesForDisplay(selectedModule));
    return;
  }
  visibleSpecs.forEach((spec) => moduleFields.append(createField('module', spec)));
  applyFields('module', visibleSpecs, moduleValuesForDisplay(selectedModule));
  if (selectedModule === 'train') {
    const autoGo = document.createElement('label'); autoGo.className = 'checkbox';
    const input = document.createElement('input'); input.id = 'module-auto-go'; input.type = 'checkbox';
    input.checked = moduleValuesForDisplay(selectedModule).auto_go === true;
    const text = document.createElement('span'); text.textContent = 'auto_go：训练成功后依次执行 calibrate → detect(test) → evaluate，并优先于后续训练任务。';
    autoGo.append(input, text); moduleFields.append(autoGo);
  }
}

function isModuleImplemented(moduleName) { return Boolean(moduleName && schemas[moduleName]); }

function moduleValuesForDisplay(moduleName) {
  const saved = formState.modules[moduleName] || {};
  const values = { ...saved };
  Object.entries(moduleDisplayDefaults[moduleName] || {}).forEach(([dest, value]) => {
    if (values[dest] === undefined || values[dest] === '') values[dest] = value;
  });
  return values;
}

function splitPercentageValues(values, storedName, legacyName) {
  const stored = values[storedName];
  if (Array.isArray(stored) && stored.length === splitColumns.length) return stored.map((value) => String(value));
  const legacy = values[legacyName];
  if (Array.isArray(legacy) && legacy.length === splitColumns.length) {
    return legacy.map((value) => Number.isFinite(Number(value)) ? String(Number(value) * 100) : '');
  }
  return splitDisplayDefaults[storedName];
}

function splitInputId(section, split) { return `module-${section}-${split}`; }

function splitPercentageInput(section, split, value, label) {
  const input = document.createElement('input');
  input.id = splitInputId(section, split);
  input.type = 'number'; input.min = '0'; input.max = '100'; input.step = '0.1'; input.required = true;
  input.value = value; input.setAttribute('aria-label', label);
  return input;
}

function renderSplitModule(specs, values) {
  const regularSpecs = specs.filter((spec) => !splitRatioDestinations.has(spec.dest));
  regularSpecs.forEach((spec) => moduleFields.append(createField('module', spec)));
  applyFields('module', regularSpecs, values);

  const configuration = document.createElement('section'); configuration.className = 'split-configuration';
  const partitionTitle = document.createElement('h4'); partitionTitle.textContent = '划分比例（%）'; configuration.append(partitionTitle);
  const partitionGrid = document.createElement('div'); partitionGrid.className = 'split-share-grid';
  const partitionValues = splitPercentageValues(values, 'partition_percentages', 'ratios');
  splitColumns.forEach((column, index) => {
    const label = document.createElement('label'); label.textContent = column.label;
    label.append(splitPercentageInput('partition', column.key, partitionValues[index], `${column.label}集合比例（%）`));
    partitionGrid.append(label);
  });
  configuration.append(partitionGrid);

  const planTitle = document.createElement('h4'); planTitle.textContent = '各集合内的轨迹构成比例（%）'; configuration.append(planTitle);
  const table = document.createElement('table'); table.className = 'split-plan-table';
  const header = table.insertRow(); header.insertCell().textContent = '';
  splitColumns.forEach((column) => { const cell = document.createElement('th'); cell.scope = 'col'; cell.textContent = column.label; header.append(cell); });
  const normalValues = splitPercentageValues(values, 'normal_percentages', 'normal_ratios');
  const anomalyValues = splitPercentageValues(values, 'anomaly_percentages', 'anomaly_ratios');
  [['normal', '正常', normalValues], ['anomaly', '异常', anomalyValues]].forEach(([kind, label, percentages]) => {
    const row = table.insertRow(); const heading = document.createElement('th'); heading.scope = 'row'; heading.textContent = label; row.append(heading);
    splitColumns.forEach((column, index) => {
      const cell = row.insertCell(); cell.append(splitPercentageInput(String(kind), column.key, percentages[index], `${String(label)} ${column.label}比例（%）`));
    });
  });
  configuration.append(table);
  const note = document.createElement('small'); note.className = 'hint';
  note.textContent = '上方三项必须合计 100%；下表每一列的正常与异常必须合计 100%。split 只保存注入计划，inject 才实际生成异常轨迹和记录标签。';
  configuration.append(note); moduleFields.append(configuration);
}

function renderNavigation() {
  moduleNav.replaceChildren();
  moduleOrder.forEach((name) => {
    const item = document.createElement('button'); item.type = 'button'; item.className = 'nav-item'; item.textContent = name;
    item.dataset.module = name; item.addEventListener('click', () => selectModule(name)); moduleNav.append(item);
  });
}

let trainMonitorTimer = null;
let trainMonitorVersion = 0;
const trainMonitorPanel = document.querySelector('#train-monitor');
const trainMonitorRows = document.querySelector('#train-monitor-rows');
const trainMonitorStatus = document.querySelector('#train-monitor-status');

function stopTrainMonitor() {
  trainMonitorVersion += 1;
  if (trainMonitorTimer) clearInterval(trainMonitorTimer);
  trainMonitorTimer = null;
  trainMonitorPanel.hidden = true;
}

function startTrainMonitor(artifact = null) {
  stopTrainMonitor();
  trainMonitorPanel.hidden = false;
  trainMonitorRows.replaceChildren();
  const version = trainMonitorVersion;
  const rows = new Map();
  let busy = false;
  const number = (value, digits = 0) => value == null ? '—' : Number(value).toFixed(digits);
  const states = {queued: '排队', initializing: '初始化', training: '训练', validating: '验证', saving: '保存', completed: '完成', failed: '失败', interrupted: '已中断'};
  async function refresh() {
    if (busy || document.hidden) return;
    busy = true;
    try {
      const data = await apiRequest(`/api/train/progress${artifact ? `?artifact=${encodeURIComponent(artifact)}` : ''}`);
      if (version !== trainMonitorVersion || selectedModule !== 'train') return;
      trainMonitorStatus.textContent = data.run ? `${data.run.dataset_name} · run ${data.run.run_id} · ${data.run.state}${data.mode === 'legacy_logs' ? ' · 旧运行：仅显示日志确认的进度' : ' · 每秒刷新'}` : '暂无训练运行';
      const seen = new Set();
      const gpuMap = new Map((data.gpus || []).map((gpu) => [gpu.device, gpu]));
      for (const item of (data.experts || []).sort((a, b) => a.group_id.localeCompare(b.group_id))) {
        seen.add(item.group_id);
        let row = rows.get(item.group_id);
        if (!row) {
          row = document.createElement('tr');
          for (let i = 0; i < 11; i += 1) row.append(document.createElement('td'));
          rows.set(item.group_id, row); trainMonitorRows.append(row);
        }
        const active = !['completed', 'failed', 'interrupted', 'queued'].includes(item.state) && data.run.state === 'running';
        const elapsed = item.started_at && active ? data.server_time - item.started_at : item.elapsed_seconds;
        const gpu = gpuMap.get(item.device);
        const memory = item.memory_allocated_mib != null ? `${number(item.memory_allocated_mib)}/${number(item.memory_reserved_mib)}` : (active && gpu ? `整卡 ${number(gpu.memory_used_mib)}` : '—');
        const seconds = elapsed == null ? null : Math.max(0, Math.floor(elapsed));
        const duration = seconds == null ? '—' : `${Math.floor(seconds / 3600)}:${String(Math.floor(seconds / 60) % 60).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`;
        const state = `${states[item.state] || item.state}${item.early_stopped ? '（早停）' : ''}`;
        const values = [item.group_id, state, `${item.epoch ?? '—'}/${item.epochs ?? '—'}`,
          item.total == null ? '—' : `${item.completed ?? 0}/${item.total}`, item.device || '—', memory,
          gpu ? `${number(gpu.utilization)}%` : '—', duration, number(item.train_loss, 5), number(item.validation_loss, 5),
          item.updated_at ? `${number(Math.max(0, data.server_time - item.updated_at))}秒` : '—'];
        values.forEach((value, index) => { if (row.cells[index].textContent !== value) row.cells[index].textContent = value; });
      }
      for (const [key, row] of rows) if (!seen.has(key)) { row.remove(); rows.delete(key); }
    } catch (error) {
      if (version === trainMonitorVersion) trainMonitorStatus.textContent = `进度暂不可用：${error.message}`;
    } finally { busy = false; }
  }
  void refresh(); trainMonitorTimer = setInterval(refresh, 1000);
}

document.querySelector('#train-monitor-latest').addEventListener('click', () => startTrainMonitor());

function showGlobal() {
  searchView.hidden = true;
  document.querySelectorAll('#search-random-nav,#search-tpe-nav').forEach(item => item.classList.remove('selected'));
  stopTrainMonitor();
  captureFormState();
  clearTaskOutput();
  clearHistory();
  selectedModule = null; globalView.hidden = false; moduleView.hidden = true;
  viewKicker.textContent = '全局运行配置'; viewTitle.textContent = '总体设置'; formStatus.textContent = '';
  globalNav.classList.add('selected'); moduleNav.querySelectorAll('.nav-item').forEach((item) => item.classList.remove('selected'));
  formState.selected_module = null; scheduleFormStateSave();
}

function selectModule(name) {
  searchView.hidden = true;
  document.querySelectorAll('#search-random-nav,#search-tpe-nav').forEach(item => item.classList.remove('selected'));
  stopTrainMonitor();
  captureFormState();
  selectedModule = name; globalView.hidden = true; moduleView.hidden = !isModuleImplemented(name);
  viewKicker.textContent = '项目模块'; viewTitle.textContent = name; moduleHeading.textContent = `${name} 模块设置`; formStatus.textContent = '';
  globalNav.classList.remove('selected');
  moduleNav.querySelectorAll('.nav-item').forEach((item) => item.classList.toggle('selected', item.dataset.module === name));
  renderModule();
  formState.selected_module = name; scheduleFormStateSave();
  void loadModuleOutput(name);
  void loadHistory(name);
  if (name === 'train') startTrainMonitor();
}

function readFields(scope, specs) {
  const values = {};
  specs.forEach((spec) => {
    const element = document.querySelector(`#${fieldId(scope, spec.dest)}`);
    if (spec.is_flag) { values[spec.dest] = element.checked; return; }
    const value = element.value.trim();
    if (spec.takes_multiple) values[spec.dest] = value ? value.split(/\s+/) : null;
    else values[spec.dest] = value;
  });
  return values;
}

function readSplitFormState() {
  const regularSpecs = (schemas.split || []).filter((spec) => !splitRatioDestinations.has(spec.dest));
  const values = readFields('module', regularSpecs);
  [['partition_percentages', 'partition'], ['normal_percentages', 'normal'], ['anomaly_percentages', 'anomaly']].forEach(([name, section]) => {
    values[name] = splitColumns.map((column) => document.querySelector(`#${splitInputId(section, column.key)}`).value.trim());
  });
  return values;
}

function percentageArguments(percentages, label, requireTotal = true) {
  if (!Array.isArray(percentages) || percentages.length !== splitColumns.length) throw new Error(`${label}必须填写训练、验证和测试三项。`);
  const numbers = percentages.map((value) => Number(value));
  if (!numbers.every((value) => Number.isFinite(value) && value >= 0 && value <= 100)) throw new Error(`${label}必须是 0 到 100 之间的数值。`);
  if (requireTotal && Math.abs(numbers.reduce((sum, value) => sum + value, 0) - 100) > 1e-8) throw new Error(`${label}必须合计 100%。`);
  return numbers.map((value) => String(value / 100));
}

function splitModuleArguments() {
  const state = readSplitFormState();
  const partition = percentageArguments(state.partition_percentages, '划分比例');
  if (partition.some((value) => Number(value) <= 0)) throw new Error('训练、验证和测试的划分比例必须均大于 0%。');
  const normal = percentageArguments(state.normal_percentages, '正常轨迹比例', false);
  const anomaly = percentageArguments(state.anomaly_percentages, '异常轨迹比例', false);
  splitColumns.forEach((column, index) => {
    if (Math.abs(Number(normal[index]) + Number(anomaly[index]) - 1) > 1e-8) {
      throw new Error(`${column.label}集合中的正常与异常比例必须合计 100%。`);
    }
  });
  return {
    input: state.input,
    ratios: partition,
    normal_ratios: normal,
    anomaly_ratios: anomaly,
  };
}

function readModuleFormState() {
  const values = selectedModule === 'split'
    ? readSplitFormState()
    : readFields('module', (schemas[selectedModule] || []).filter((spec) => !spec.auto_derived));
  if (selectedModule === 'train') values.auto_go = document.querySelector('#module-auto-go')?.checked === true;
  return values;
}

function emptyFormState() { return { global: {}, modules: {}, selected_module: null }; }

function normalizeFormState(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return emptyFormState();
  const global = value.global && typeof value.global === 'object' && !Array.isArray(value.global) ? value.global : {};
  const modules = value.modules && typeof value.modules === 'object' && !Array.isArray(value.modules) ? value.modules : {};
  const selectedModuleValue = typeof value.selected_module === 'string' ? value.selected_module : null;
  return { global, modules, selected_module: selectedModuleValue };
}

function applyFields(scope, specs, values) {
  specs.forEach((spec) => {
    const element = document.querySelector(`#${fieldId(scope, spec.dest)}`);
    if (!element || !(spec.dest in values)) return;
    const value = values[spec.dest];
    if (spec.is_flag) { element.checked = value === true; return; }
    if (spec.upstream_module) {
      element.dataset.savedValue = typeof value === 'string' ? value : '';
      if (Array.from(element.options).some((option) => option.value === element.dataset.savedValue)) element.value = element.dataset.savedValue;
      return;
    }
    if (spec.takes_multiple) { element.value = Array.isArray(value) ? value.join(' ') : ''; return; }
    if (typeof value === 'string' || typeof value === 'number') element.value = String(value);
  });
}

async function populateUpstreamRuns(select, spec) {
  try {
    const data = await apiRequest(`/api/upstream-runs?module=${encodeURIComponent(selectedModule)}&argument=${encodeURIComponent(spec.dest)}`);
    select.replaceChildren();
    const upstreamLabel = Array.isArray(spec.upstream_module) ? spec.upstream_module.join(' 或 ') : spec.upstream_module;
    select.add(new Option(`请选择已发布的 ${upstreamLabel} 运行`, ''));
    data.runs.forEach((run) => select.add(new Option(run.label, run.manifest)));
    select.disabled = false;
    if (select.dataset.savedValue && Array.from(select.options).some((option) => option.value === select.dataset.savedValue)) {
      select.value = select.dataset.savedValue;
    }
  } catch (error) {
    select.replaceChildren(); select.add(new Option(`读取上游运行失败：${error.message}`, '')); select.disabled = false;
  }
}

function captureFormState() {
  if (document.querySelector(`#${fieldId('global', 'seed')}`)) formState.global = readFields('global', globalOptions);
  if (selectedModule && !searchModes.includes(selectedModule) && moduleFields.querySelector('[name]')) {
    formState.modules[selectedModule] = readModuleFormState();
  }
}

function scheduleFormStateSave() {
  captureFormState();
  if (stateSaveTimer) clearTimeout(stateSaveTimer);
  stateSaveTimer = setTimeout(() => { void saveFormState(); }, 250);
}

async function saveFormState() {
  stateSaveTimer = null; captureFormState();
  try {
    await apiRequest('/api/form-state', {
      method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ state: formState }),
    });
  } catch (error) {
    if (!loginScreen.hidden) return;
    formStatus.textContent = `配置暂存失败：${error.message}`;
  }
}

function appendLines(lines) {
  if (lines.length) { logs.textContent += `${lines.join('\n')}\n`; logs.scrollTop = logs.scrollHeight; }
}

function stopPolling() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = null;
}

function formatElapsed(milliseconds) {
  const seconds = Math.max(0, Math.floor(milliseconds / 1000));
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`;
}

function renderElapsed(endedAt = null) {
  if (elapsedStartedAt === null) return;
  const end = endedAt === null ? Date.now() : endedAt;
  runElapsed.textContent = `运行计时 ${formatElapsed(end - elapsedStartedAt)}`;
  runElapsed.hidden = false;
}

function startElapsed(startedAt = Date.now()) {
  if (!Number.isFinite(startedAt)) return;
  if (elapsedTimer) clearInterval(elapsedTimer);
  elapsedStartedAt = startedAt;
  renderElapsed();
  elapsedTimer = setInterval(renderElapsed, 1000);
}

function stopElapsed(endedAt = Date.now()) {
  if (elapsedTimer) clearInterval(elapsedTimer);
  elapsedTimer = null;
  renderElapsed(endedAt);
}

function clearElapsed() {
  if (elapsedTimer) clearInterval(elapsedTimer);
  elapsedTimer = null; elapsedStartedAt = null;
  runElapsed.textContent = ''; runElapsed.hidden = true;
}

function syncElapsed(job, status) {
  const startedAt = Number(job?.started_at) * 1000;
  const endedAt = Number(job?.ended_at) * 1000;
  if (!Number.isFinite(startedAt)) return;
  if (status === 'running' || status === 'stopping') startElapsed(startedAt);
  else stopElapsed(Number.isFinite(endedAt) ? endedAt : Date.now());
}

function setJobControls(status = '') {
  const active = status === 'running' || status === 'stopping';
  runButton.disabled = active;
  trainQueueButton.hidden = selectedModule !== 'train';
  trainQueueButton.disabled = selectedModule !== 'train';
  searchStart.disabled = active;
  historyRefresh.disabled = active || historyRunning;
  stopButton.hidden = false;
  stopButton.disabled = !active || status === 'stopping';
  searchStop.disabled = stopButton.disabled;
}

function clearTaskOutput() {
  stopPolling(); currentJob = null; nextOffset = 0;
  stopButton.textContent = '停止运行';
  clearElapsed();
  command.textContent = ''; logs.textContent = ''; jobStatus.textContent = ''; setJobControls();
}

async function loadModuleOutput(moduleName) {
  clearTaskOutput();
  try {
    const data = await apiRequest(`/api/jobs/latest?module=${encodeURIComponent(moduleName)}`);
    if (selectedModule !== moduleName) return;
    const job = data.job;
    if (!job) { if (data.logs) logs.textContent = data.logs; return; }
    currentJob = job.job_id; command.textContent = job.command.join(' '); jobStatus.textContent = job.status; setJobControls(job.status); syncElapsed(job, job.status);
    stopButton.textContent = job.module_name?.startsWith('search_') ? '停止搜索及训练' : '停止运行';
    await pollJob();
    if (currentJob === job.job_id && ['running', 'stopping'].includes(jobStatus.textContent)) pollTimer = setInterval(pollJob, 700);
  } catch (error) {
    if (selectedModule === moduleName) formStatus.textContent = `读取任务输出失败：${error.message}`;
  }
}

async function apiRequest(path, options = {}) {
  const response = await fetch(path, { credentials: 'same-origin', ...options });
  const data = await response.json().catch(() => ({}));
  if (response.status === 401 && path !== '/api/auth/login') showLogin(data.error || '会话已过期，请重新登录。');
  if (!response.ok) throw new Error(data.error || '服务器请求失败');
  return data;
}

async function pollJob() {
  if (!currentJob) return;
  const jobId = currentJob;
  try {
    const data = await apiRequest(`/api/jobs/${jobId}/logs?after=${nextOffset}`);
    if (currentJob !== jobId) return;
    if (data.truncated) appendLines(['[web-console] 较早日志已超过浏览器缓存上限。']);
    appendLines(data.lines); nextOffset = data.next_offset; jobStatus.textContent = data.status; setJobControls(data.status);
    if (!['running', 'stopping'].includes(data.status)) { historyRunning = false; stopElapsed(); stopPolling(); void loadHistory(selectedModule); }
  } catch (error) { jobStatus.textContent = `轮询失败：${error.message}`; stopPolling(); }
}

function clearHistory() {
  historyVersion += 1; detailVersion += 1; historyRunning = false;
  if (historyStateTimer) clearInterval(historyStateTimer); historyStateTimer = null;
  groupResult.hidden = true; groupResult.replaceChildren();
  if (segmentTimer) clearInterval(segmentTimer); segmentTimer = null;
  segmentResult.hidden = true; segmentResult.replaceChildren();
  historyPanel.hidden = true; historyRows.replaceChildren(); historyDetail.hidden = true;
  historyStatus.textContent = ''; historySummary.textContent = ''; historyLogs.textContent = '';
}

async function loadHistory(moduleName, manual = false) {
  if (manual && historyRefresh.disabled) return;
  if (!isModuleImplemented(moduleName)) { clearHistory(); return; }
  const version = ++historyVersion; detailVersion += 1;
  historyColumns = (schemas[moduleName] || []).filter(item => !item.auto_derived).map(item => ({key:item.dest, label:item.option}));
  renderHistoryHead();
  if (historyStateTimer) clearInterval(historyStateTimer); historyStateTimer = null;
  groupResult.hidden = true; groupResult.replaceChildren();
  if (segmentTimer) clearInterval(segmentTimer); segmentTimer = null;
  segmentResult.hidden = true; segmentResult.replaceChildren();
  historyRefresh.disabled = true;
  historyPanel.hidden = false; historyStatus.textContent = '正在读取历史运行…'; historyRows.replaceChildren();
  historyDetail.hidden = true; historySummary.textContent = ''; historyLogs.textContent = '';
  try {
    const data = await apiRequest(`/api/history?module=${encodeURIComponent(moduleName)}${manual ? '&refresh=1' : ''}`);
    if (selectedModule !== moduleName || version !== historyVersion) return;
    historyColumns = data.columns || []; historyRunning = Boolean(data.running);
    renderHistoryHead();
    const runs = Array.isArray(data.runs) ? data.runs : [];
    historyStatus.textContent = runs.length ? `共 ${runs.length} 条运行记录。` : '该模块尚无落盘运行。';
    runs.forEach((run) => appendHistoryRow(moduleName, run));
    setJobControls(jobStatus.textContent);
    const activeSegment = moduleName === 'segment' ? runs.find((run) => run.state === 'running') : null;
    if (activeSegment?.artifact) void loadHistoryDetail(moduleName, activeSegment.artifact);
    if (historyRunning) watchHistoryCompletion(moduleName);
  } catch (error) {
    if (selectedModule === moduleName && version === historyVersion) { historyStatus.textContent = `读取历史运行失败：${error.message}`; setJobControls(jobStatus.textContent); }
  }
}

function watchHistoryCompletion(moduleName) {
  let busy = false;
  historyStateTimer = setInterval(async () => {
    if (busy || document.hidden || selectedModule !== moduleName) return;
    busy = true;
    try {
      const state = await apiRequest(`/api/history/state?module=${encodeURIComponent(moduleName)}`);
      if (selectedModule === moduleName && !state.running) { historyRunning = false; void loadHistory(moduleName); }
    } catch (error) { historyStatus.textContent = error.message; }
    finally { busy = false; }
  }, 2000);
}

function renderHistoryHead() {
  historyHead.replaceChildren();
  ['数据集', '运行', '时间', '状态', '随机种子', ...historyColumns.map((item) => item.label), '操作'].forEach((label) => {
    const cell = document.createElement('th'); cell.textContent = label; historyHead.append(cell);
  });
}

function appendHistoryRow(moduleName, run) {
  if (!run || typeof run !== 'object') return;
  const row = document.createElement('tr');
  [run.dataset_name, run.run_name, run.run_timestamp, run.state, run.seed,
    ...historyColumns.map((item) => run.parameters?.[item.key])].forEach((value) => {
    const cell = document.createElement('td');
    const full = value === undefined ? '未记录' : value === null ? '—' : typeof value === 'object' ? JSON.stringify(value) : String(value);
    cell.textContent = full.length > 65 ? `${full.slice(0, 62)}…` : full; cell.title = full; row.append(cell);
  });
  const actions = document.createElement('td');
  const detailButton = document.createElement('button'); detailButton.type = 'button'; detailButton.className = 'secondary'; detailButton.textContent = '查看';
  detailButton.addEventListener('click', () => { void loadHistoryDetail(moduleName, run.artifact); });
  const deleteButton = document.createElement('button'); deleteButton.type = 'button'; deleteButton.className = 'secondary'; deleteButton.textContent = '删除';
  deleteButton.addEventListener('click', () => { void deleteHistoryRun(moduleName, run.artifact); });
  detailButton.disabled = !run.artifact; deleteButton.disabled = !run.artifact || run.state === 'running';
  actions.append(detailButton, document.createTextNode(' '), deleteButton); row.append(actions); historyRows.append(row);
}

async function loadHistoryDetail(moduleName, artifact) {
  const version = ++detailVersion; groupResult.hidden = true; groupResult.replaceChildren();
  if (segmentTimer) clearInterval(segmentTimer); segmentTimer = null;
  segmentResult.hidden = true; segmentResult.replaceChildren();
  if (moduleName === 'train') startTrainMonitor(artifact);
  try {
    const data = await apiRequest(`/api/history/detail?module=${encodeURIComponent(moduleName)}&artifact=${encodeURIComponent(artifact)}`);
    if (selectedModule !== moduleName || version !== detailVersion) return null;
    const result = data.summary || data.failure || { message: '该运行尚未生成摘要。' };
    historySummary.textContent = JSON.stringify(result, null, 2);
    historyLogs.textContent = `${data.logs_truncated ? '[仅显示日志尾部]\n' : ''}${data.logs || '未找到对应日志。'}`;
    historyDetail.hidden = false;
    if (moduleName === 'group') {
      groupResult.hidden = false; groupResult.textContent = '正在读取分组统计与轨迹…';
      apiRequest(`/api/group/result?artifact=${encodeURIComponent(artifact)}`).then((result) => {
        if (selectedModule === moduleName && version === detailVersion) renderGroupResult(groupResult, result);
      }).catch((error) => { if (version === detailVersion) groupResult.textContent = `分组展示失败：${error.message}`; });
    }
    if (moduleName === 'segment') {
      segmentResult.hidden = false;
      const refresh = async () => {
        try {
          const result = await apiRequest(`/api/segment/result?artifact=${encodeURIComponent(artifact)}`);
          if (selectedModule === 'segment' && version === detailVersion) renderSegmentResult(segmentResult, result);
        } catch (error) { if (version === detailVersion) segmentResult.textContent = `阶段进度读取失败：${error.message}`; }
      };
      await refresh();
      if (data.entry?.state === 'running') segmentTimer = setInterval(refresh, 1000);
    }
    return data;
  } catch (error) {
    if (selectedModule === moduleName) historyStatus.textContent = `读取运行详情失败：${error.message}`;
    return null;
  }
}

async function deleteHistoryRun(moduleName, artifact) {
  const detail = await loadHistoryDetail(moduleName, artifact);
  if (!detail) return;
  const preview = Array.isArray(detail.delete_preview) ? detail.delete_preview : [];
  const message = `删除此运行及其 ${Math.max(0, preview.length - 1)} 个关联下游产物？\n\n${preview.join('\n')}`;
  if (!confirm(message)) return;
  try {
    const data = await apiRequest(`/api/history?module=${encodeURIComponent(moduleName)}&artifact=${encodeURIComponent(artifact)}`, { method: 'DELETE' });
    historyStatus.textContent = `已删除 ${Array.isArray(data.deleted) ? data.deleted.length : 0} 个运行产物。`;
    historyDetail.hidden = true; await loadHistory(moduleName);
  } catch (error) { historyStatus.textContent = `删除失败：${error.message}`; }
}

settingsForm.addEventListener('submit', async (event) => {
  event.preventDefault(); formStatus.textContent = '';
  if (!selectedModule) { formStatus.textContent = '请先在左侧选择一个模块。'; return; }
  if (!isModuleImplemented(selectedModule)) return;
  let moduleArguments;
  try { moduleArguments = selectedModule === 'split' ? splitModuleArguments() : readModuleFormState(); }
  catch (error) { formStatus.textContent = error.message; return; }
  await saveFormState();
  const autoGo = selectedModule === 'train' && moduleArguments.auto_go === true;
  delete moduleArguments.auto_go;
  const payload = { module: selectedModule, global: readFields('global', globalOptions), module_args: moduleArguments };
  try {
    clearTaskOutput(); startElapsed(); setJobControls('running'); jobStatus.textContent = '正在启动';
    if (selectedModule === 'train') payload.auto_go = autoGo;
    const data = await apiRequest(selectedModule === 'train' ? '/api/train/queue' : '/api/jobs', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
    if (selectedModule === 'train') {
      const task = data.task || {};
      formStatus.textContent = task.auto_go ? '训练任务已加入队列；成功后将自动执行后续三步。' : '训练任务已加入队列。';
      jobStatus.textContent = '已排队'; setJobControls();
      window.setTimeout(() => { if (selectedModule === 'train') void loadModuleOutput('train'); }, 300);
      return;
    }
    currentJob = data.job_id; command.textContent = data.command.join(' '); jobStatus.textContent = data.status; setJobControls(data.status);
    if (data.history_entry) { appendHistoryRow(selectedModule, {...data.history_entry, state:data.status}); historyRows.prepend(historyRows.lastElementChild); }
    if (selectedModule === 'segment') window.setTimeout(() => { void loadHistory('segment'); }, 500);
    if (selectedModule === 'train') startTrainMonitor();
    stopPolling(); await pollJob(); pollTimer = setInterval(pollJob, 700);
  } catch (error) { formStatus.textContent = error.message; setJobControls(); jobStatus.textContent = '未启动'; }
});

trainQueueButton.addEventListener('click', async () => {
  if (selectedModule !== 'train') return;
  formStatus.textContent = '';
  let moduleArguments;
  try { moduleArguments = readModuleFormState(); }
  catch (error) { formStatus.textContent = error.message; return; }
  const autoGo = moduleArguments.auto_go === true;
  delete moduleArguments.auto_go;
  await saveFormState();
  const payload = { module: 'train', global: readFields('global', globalOptions), module_args: moduleArguments, auto_go: autoGo };
  try {
    const data = await apiRequest('/api/train/queue', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
    formStatus.textContent = data.task?.auto_go ? '训练任务已加入队列；成功后将自动执行后续三步。' : '训练任务已加入队列。';
    window.setTimeout(() => { if (selectedModule === 'train') void loadModuleOutput('train'); }, 300);
  } catch (error) { formStatus.textContent = error.message; }
});

async function stopCurrentJob() {
  if (!currentJob || !confirm('停止当前任务及全部子进程？若由参数搜索启动，将停止整轮搜索，避免继续下一组。本次尚未发布的产物会被清理，已完成产物保留。')) return;
  formStatus.textContent = '';
  try {
    const data = await apiRequest(`/api/jobs/${encodeURIComponent(currentJob)}/stop`, { method: 'POST' });
    jobStatus.textContent = data.status; setJobControls(data.status);
  } catch (error) { formStatus.textContent = `停止失败：${error.message}`; appendLines([`停止失败：${error.message}`]); }
}
stopButton.addEventListener('click', stopCurrentJob);
searchStop.addEventListener('click', stopCurrentJob);

async function loadPicker(path = '') {
  browserStatus.textContent = '正在读取目录…'; browserEntries.replaceChildren();
  try {
    const data = await apiRequest(`/api/files?root=${encodeURIComponent(pickerRoot)}&path=${encodeURIComponent(path)}`);
    pickerPath = data.path; pickerListing = data; browserStatus.textContent = data.truncated ? '目录项较多，仅显示前 500 项。' : '';
    browserPath.textContent = data.path ? `${data.root_path}/${data.path}` : data.root_path;
    browserUp.disabled = data.parent === null;
    browserRoots.querySelectorAll('button').forEach((button) => button.classList.toggle('selected', button.dataset.root === pickerRoot));
    browserSelectDirectory.hidden = pickerMode !== 'directory';
    data.entries.forEach((entry) => {
      const row = document.createElement('div'); row.className = 'browser-entry';
      const name = document.createElement('span'); name.className = 'browser-name'; name.textContent = `${entry.kind === 'directory' ? '▸' : '·'} ${entry.name}`;
      const action = document.createElement('button'); action.type = 'button'; action.className = 'secondary';
      if (entry.kind === 'directory') { action.textContent = '打开'; action.addEventListener('click', () => loadPicker(entry.path)); }
      else if (pickerMode === 'directory') { action.textContent = '文件'; action.disabled = true; }
      else { action.textContent = '选择'; action.addEventListener('click', () => chooseFile(entry.absolute_path)); }
      row.append(name, action); browserEntries.append(row);
    });
  } catch (error) { browserStatus.textContent = error.message; }
}

function openFileBrowser(target, mode, title) {
  pickerTarget = target; pickerMode = mode; pickerRoot = 'home'; pickerPath = ''; pickerListing = null;
  fileDialogTitle.textContent = mode === 'directory' ? `选择目录：${title}` : `选择输入文件：${title}`;
  fileDialog.showModal(); loadPicker('');
}

function chooseFile(path) {
  if (pickerTarget) pickerTarget.value = path;
  fileDialog.close(); pickerTarget = null; scheduleFormStateSave();
}

loginForm.addEventListener('submit', async (event) => {
  event.preventDefault(); loginStatus.textContent = '';
  try {
    await apiRequest('/api/auth/login', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ token: loginToken.value }) });
    loginToken.value = ''; await showConsole();
  } catch (error) { loginStatus.textContent = error.message; loginToken.select(); }
});

logoutButton.addEventListener('click', async () => {
  try { await apiRequest('/api/auth/logout', { method: 'POST' }); } catch (error) { /* 会话失效时仍回到登录页。 */ }
  showLogin('已退出登录。');
});

restartButton.addEventListener('click', async () => {
  if (!window.confirm('重启会关闭当前控制台，5 秒后按服务器上的最新代码重新启动；登录会话将失效。继续吗？')) return;
  restartButton.disabled = true;
  restartButton.textContent = '正在重启…';
  try {
    const result = await apiRequest('/api/server/restart', { method: 'POST' });
    restartButton.textContent = '5 秒后重新登录';
    window.setTimeout(() => window.location.reload(), (Number(result.delay_seconds) + 1) * 1000);
  } catch (error) {
    restartButton.disabled = false;
    restartButton.textContent = '重启控制台';
    window.alert(`无法重启控制台：${error.message}`);
  }
});

globalNav.addEventListener('click', showGlobal);
function selectSearch(mode) {
  captureFormState(); stopTrainMonitor(); clearHistory(); clearTaskOutput();
  selectedModule = 'search_' + mode;
  globalView.hidden = true; moduleView.hidden = true; searchView.hidden = false;
  globalNav.classList.remove('selected'); moduleNav.querySelectorAll('.nav-item').forEach(item => item.classList.remove('selected'));
  document.querySelector('#search-random-nav').classList.toggle('selected', mode === 'random');
  document.querySelector('#search-tpe-nav').classList.toggle('selected', mode === 'tpe');
  viewKicker.textContent = '参数寻优'; viewTitle.textContent = mode === 'random' ? '随机搜索' : 'TPE';
  searchNote.textContent = mode === 'random' ? '按服务器 configs/parameter_search.json 随机尝试20组；每组顺序执行 train → calibrate → detect(val) → evaluate。重复启动会继续未完成的组合预算。' : '在同一研究的20组随机结果基础上继续TPE；默认追加20组，仍使用验证集评分。';
  formState.selected_module = selectedModule; scheduleFormStateSave(); void loadModuleOutput(selectedModule);
}
document.querySelector('#search-random-nav').addEventListener('click', () => selectSearch('random'));
document.querySelector('#search-tpe-nav').addEventListener('click', () => selectSearch('tpe'));
searchStart.addEventListener('click', async () => {
  const name = selectedModule;
  if (!searchModes.includes(name)) return;
  clearTaskOutput(); startElapsed(); setJobControls('running');
  try {
    const data = await apiRequest(`/api/search/${name.slice(7)}`, {method:'POST'});
    if (selectedModule !== name) return;
    currentJob = data.job_id; command.textContent = data.command.join(' '); jobStatus.textContent = data.status;
    setJobControls(data.status); await pollJob();
    if (['running','stopping'].includes(jobStatus.textContent)) pollTimer = setInterval(pollJob,700);
  } catch (error) { logs.textContent = `启动失败：${error.message}`; jobStatus.textContent = '未启动'; stopElapsed(); setJobControls(); }
});
historyRefresh.addEventListener('click', () => { if (selectedModule) void loadHistory(selectedModule, true); });
settingsForm.addEventListener('input', scheduleFormStateSave);
settingsForm.addEventListener('change', scheduleFormStateSave);
browserRoots.addEventListener('click', (event) => { const button = event.target.closest('button[data-root]'); if (button) { pickerRoot = button.dataset.root; loadPicker(''); } });
browserUp.addEventListener('click', () => { if (pickerListing && pickerListing.parent !== null) loadPicker(pickerListing.parent === '.' ? '' : pickerListing.parent); });
browserSelectDirectory.addEventListener('click', () => { if (pickerListing) chooseFile(pickerListing.path ? `${pickerListing.root_path}/${pickerListing.path}` : pickerListing.root_path); });
fileDialogClose.addEventListener('click', () => fileDialog.close());
browserCancel.addEventListener('click', () => fileDialog.close());

function showLogin(message = '') {
  stopTrainMonitor();
  clearHistory();
  clearTaskOutput(); selectedModule = null; schemasLoaded = false; formState = emptyFormState();
  if (stateSaveTimer) clearTimeout(stateSaveTimer); stateSaveTimer = null;
  loginScreen.hidden = false; consoleShell.hidden = true; loginStatus.textContent = message; loginToken.focus();
}

async function showConsole() {
  loginScreen.hidden = true; consoleShell.hidden = false;
  if (!schemasLoaded) {
    const [schemaData, stateData] = await Promise.all([apiRequest('/api/schema'), apiRequest('/api/form-state')]);
    schemas = schemaData.modules;
    moduleOrder = Array.isArray(schemaData.module_order)
      ? schemaData.module_order.filter((name) => typeof name === 'string') : Object.keys(schemas);
    formState = normalizeFormState(stateData.state); schemasLoaded = true; renderGlobal(); renderNavigation();
  }
  if (searchModes.includes(formState.selected_module)) selectSearch(formState.selected_module.slice(7));
  else if (formState.selected_module && moduleOrder.includes(formState.selected_module)) selectModule(formState.selected_module);
  else showGlobal();
}

async function initialize() {
  const response = await fetch('/api/auth/status', { credentials: 'same-origin' });
  const data = await response.json();
  if (data.authenticated) await showConsole(); else showLogin();
}

initialize().catch((error) => showLogin(`页面初始化失败：${error.message}`));
