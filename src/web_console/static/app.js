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
const moduleFields = document.querySelector('#module-fields');
const globalFields = document.querySelector('#global-fields');
const settingsForm = document.querySelector('#settings-form');
const runButton = document.querySelector('#run-button');
const formStatus = document.querySelector('#form-status');
const jobStatus = document.querySelector('#job-status');
const command = document.querySelector('#command');
const logs = document.querySelector('#logs');
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
let selectedModule = null;
let schemasLoaded = false;
let currentJob = null;
let nextOffset = 0;
let pollTimer = null;
let pickerTarget = null;
let pickerMode = 'file';
let pickerRoot = 'home';
let pickerPath = '';
let pickerListing = null;
let formState = emptyFormState();
let stateSaveTimer = null;

const globalOptions = [
  { dest: 'seed', label: '随机种子', placeholder: '默认 42' },
  { dest: 'log_level', label: '日志等级', choices: ['DEBUG', 'INFO', 'WARNING', 'ERROR'], default: 'INFO' },
  { dest: 'console', label: '显示控制台日志', is_flag: true, help_text: '启用后传入 --console；后台线程会持续读取输出，不等待网页轮询。' },
];

const moduleDisplayDefaults = {
  preprocess: {
    altitude_unit: 'm',
    speed_unit: 'km/h',
    heading_unit: 'deg',
  },
};

function fieldId(scope, dest) { return `${scope}-${dest}`; }

function createField(scope, spec) {
  const label = document.createElement('label');
  label.className = `field${spec.takes_multiple ? ' full' : ''}`;
  const id = fieldId(scope, spec.dest);
  if (spec.is_flag) {
    label.classList.add('checkbox');
    const input = document.createElement('input');
    input.type = 'checkbox'; input.id = id; input.name = spec.dest;
    label.append(input, document.createTextNode(spec.label || spec.option));
  } else {
    const title = document.createElement('span');
    title.textContent = spec.label || spec.option;
    if (spec.required) { const required = document.createElement('b'); required.textContent = ' *'; title.append(required); }
    const hasChoices = Array.isArray(spec.choices) && spec.choices.length > 0;
    const input = hasChoices ? document.createElement('select') : document.createElement('input');
    input.id = id; input.name = spec.dest; input.required = Boolean(spec.required);
    if (hasChoices) {
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
  const specs = schemas[selectedModule] || [];
  specs.forEach((spec) => moduleFields.append(createField('module', spec)));
  applyFields('module', specs, moduleValuesForDisplay(selectedModule));
}

function moduleValuesForDisplay(moduleName) {
  const saved = formState.modules[moduleName] || {};
  const values = { ...saved };
  Object.entries(moduleDisplayDefaults[moduleName] || {}).forEach(([dest, value]) => {
    if (values[dest] === undefined || values[dest] === '') values[dest] = value;
  });
  return values;
}

function renderNavigation() {
  moduleNav.replaceChildren();
  Object.keys(schemas).forEach((name) => {
    const item = document.createElement('button'); item.type = 'button'; item.className = 'nav-item'; item.textContent = name;
    item.dataset.module = name; item.addEventListener('click', () => selectModule(name)); moduleNav.append(item);
  });
}

function showGlobal() {
  captureFormState();
  clearTaskOutput();
  selectedModule = null; globalView.hidden = false; moduleView.hidden = true;
  viewKicker.textContent = '全局运行配置'; viewTitle.textContent = '总体设置'; formStatus.textContent = '';
  globalNav.classList.add('selected'); moduleNav.querySelectorAll('.nav-item').forEach((item) => item.classList.remove('selected'));
  formState.selected_module = null; scheduleFormStateSave();
}

function selectModule(name) {
  captureFormState();
  selectedModule = name; globalView.hidden = true; moduleView.hidden = false;
  viewKicker.textContent = '项目模块'; viewTitle.textContent = name; moduleHeading.textContent = `${name} 模块设置`; formStatus.textContent = '';
  globalNav.classList.remove('selected');
  moduleNav.querySelectorAll('.nav-item').forEach((item) => item.classList.toggle('selected', item.dataset.module === name));
  renderModule();
  formState.selected_module = name; scheduleFormStateSave();
  void loadModuleOutput(name);
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
    if (spec.takes_multiple) { element.value = Array.isArray(value) ? value.join(' ') : ''; return; }
    if (typeof value === 'string' || typeof value === 'number') element.value = String(value);
  });
}

function captureFormState() {
  if (document.querySelector(`#${fieldId('global', 'seed')}`)) formState.global = readFields('global', globalOptions);
  if (selectedModule && document.querySelector(`#${fieldId('module', (schemas[selectedModule] || [])[0]?.dest || '')}`)) {
    formState.modules[selectedModule] = readFields('module', schemas[selectedModule] || []);
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

function clearTaskOutput() {
  stopPolling(); currentJob = null; nextOffset = 0;
  command.textContent = ''; logs.textContent = ''; jobStatus.textContent = ''; runButton.disabled = false;
}

async function loadModuleOutput(moduleName) {
  clearTaskOutput();
  try {
    const data = await apiRequest(`/api/jobs/latest?module=${encodeURIComponent(moduleName)}`);
    if (selectedModule !== moduleName) return;
    const job = data.job;
    if (!job) return;
    currentJob = job.job_id; command.textContent = job.command.join(' '); jobStatus.textContent = job.status;
    await pollJob();
    if (currentJob === job.job_id && jobStatus.textContent === 'running') pollTimer = setInterval(pollJob, 700);
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
    appendLines(data.lines); nextOffset = data.next_offset; jobStatus.textContent = data.status;
    if (data.status !== 'running') { runButton.disabled = false; stopPolling(); }
  } catch (error) { jobStatus.textContent = `轮询失败：${error.message}`; stopPolling(); }
}

settingsForm.addEventListener('submit', async (event) => {
  event.preventDefault(); formStatus.textContent = '';
  if (!selectedModule) { formStatus.textContent = '请先在左侧选择一个模块。'; return; }
  await saveFormState();
  const payload = { module: selectedModule, global: readFields('global', globalOptions), module_args: readFields('module', schemas[selectedModule] || []) };
  try {
    clearTaskOutput(); runButton.disabled = true; jobStatus.textContent = '正在启动';
    const data = await apiRequest('/api/jobs', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
    currentJob = data.job_id; command.textContent = data.command.join(' '); jobStatus.textContent = data.status;
    stopPolling(); await pollJob(); pollTimer = setInterval(pollJob, 700);
  } catch (error) { formStatus.textContent = error.message; runButton.disabled = false; jobStatus.textContent = '未启动'; }
});

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

globalNav.addEventListener('click', showGlobal);
settingsForm.addEventListener('input', scheduleFormStateSave);
settingsForm.addEventListener('change', scheduleFormStateSave);
browserRoots.addEventListener('click', (event) => { const button = event.target.closest('button[data-root]'); if (button) { pickerRoot = button.dataset.root; loadPicker(''); } });
browserUp.addEventListener('click', () => { if (pickerListing && pickerListing.parent !== null) loadPicker(pickerListing.parent === '.' ? '' : pickerListing.parent); });
browserSelectDirectory.addEventListener('click', () => { if (pickerListing) chooseFile(pickerListing.path ? `${pickerListing.root_path}/${pickerListing.path}` : pickerListing.root_path); });
fileDialogClose.addEventListener('click', () => fileDialog.close());
browserCancel.addEventListener('click', () => fileDialog.close());

function showLogin(message = '') {
  clearTaskOutput(); selectedModule = null; schemasLoaded = false; formState = emptyFormState();
  if (stateSaveTimer) clearTimeout(stateSaveTimer); stateSaveTimer = null;
  loginScreen.hidden = false; consoleShell.hidden = true; loginStatus.textContent = message; loginToken.focus();
}

async function showConsole() {
  loginScreen.hidden = true; consoleShell.hidden = false;
  if (!schemasLoaded) {
    const [schemaData, stateData] = await Promise.all([apiRequest('/api/schema'), apiRequest('/api/form-state')]);
    schemas = schemaData.modules; formState = normalizeFormState(stateData.state); schemasLoaded = true; renderGlobal(); renderNavigation();
  }
  if (formState.selected_module && schemas[formState.selected_module]) selectModule(formState.selected_module);
  else showGlobal();
}

async function initialize() {
  const response = await fetch('/api/auth/status', { credentials: 'same-origin' });
  const data = await response.json();
  if (data.authenticated) await showConsole(); else showLogin();
}

initialize().catch((error) => showLogin(`页面初始化失败：${error.message}`));
