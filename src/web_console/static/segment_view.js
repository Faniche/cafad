let segmentArtifact = null;
const selectedPhaseFilters = new Set();
const selectedSegments = new Set();
let flightPlotRequest = 0;
const flightPlotDialog = document.querySelector('#flight-plot-dialog');
const flightPlotTitle = document.querySelector('#flight-plot-title');
const flightPlotStatus = document.querySelector('#flight-plot-status');
const flightPlotFrame = document.querySelector('#flight-plot-frame');

function closeFlightPlot() {
  flightPlotRequest += 1;
  if (flightPlotDialog.open) flightPlotDialog.close();
  flightPlotFrame.removeAttribute('src'); flightPlotFrame.hidden = true;
}

async function showFlightPlot(artifact, flightId) {
  const request = ++flightPlotRequest;
  flightPlotTitle.textContent = `航班 ${flightId.slice(0, 8)} 的三维轨迹`;
  flightPlotStatus.textContent = '正在读取或生成可拖动的三维轨迹…'; flightPlotFrame.hidden = true;
  if (!flightPlotDialog.open) flightPlotDialog.showModal();
  const address = `/api/segment/flight-plot-interactive?artifact=${encodeURIComponent(artifact)}&flight_id=${encodeURIComponent(flightId)}`;
  try {
    const response = await fetch(address);
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.error || `请求失败：${response.status}`);
    }
    if (request !== flightPlotRequest) return;
    flightPlotFrame.onload = () => {
      if (request !== flightPlotRequest) return;
      flightPlotFrame.hidden = false;
      flightPlotStatus.textContent = '可拖动旋转、滚轮缩放；★ 为起点，➤ 为终点。颜色在所有阶段图中固定：GND 灰、CL 绿、LVL 蓝、CR 紫、DE 橙、NA 红。首次生成后保存在本次运行目录，后续点击直接加载。';
    };
    flightPlotFrame.src = address;
  } catch (error) { flightPlotStatus.textContent = `三维图生成失败：${error.message}`; }
}

document.querySelector('#flight-plot-close').addEventListener('click', closeFlightPlot);
flightPlotDialog.addEventListener('click', (event) => { if (event.target === flightPlotDialog) closeFlightPlot(); });

function clearSegmentSelection() {
  selectedSegments.clear();
}

function renderSegmentResult(container, data, artifact, onPhaseChange) {
  if (segmentArtifact !== artifact) {
    segmentArtifact = artifact; selectedPhaseFilters.clear(); selectedSegments.clear();
  }
  container.replaceChildren();
  const rows = Array.isArray(data.rows) ? data.rows : [];
  const phases = Array.from(new Set((data.allowed_phases || rows.map((row) => row.phase)).filter(Boolean)));
  const editable = data.editable === true;
  const controls = document.createElement('div'); controls.className = 'actions';
  const filterGroup = document.createElement('fieldset'); filterGroup.className = 'phase-filter-options';
  const filterLegend = document.createElement('legend'); filterLegend.textContent = '阶段筛选（可多选；未勾选显示全部）';
  filterGroup.append(filterLegend);
  phases.forEach((phase) => {
    const filterLabel = document.createElement('label');
    const filter = document.createElement('input'); filter.type = 'checkbox'; filter.value = phase;
    filter.checked = selectedPhaseFilters.has(phase);
    filter.addEventListener('change', () => {
      if (filter.checked) selectedPhaseFilters.add(phase); else selectedPhaseFilters.delete(phase);
      renderSegmentResult(container, data, artifact, onPhaseChange);
    });
    filterLabel.append(filter, document.createTextNode(phase)); filterGroup.append(filterLabel);
  });
  controls.append(filterGroup);
  if (editable) {
    const batchLabel = document.createElement('label'); batchLabel.className = 'field'; batchLabel.textContent = '批量改为';
    const batchPhase = document.createElement('select'); phases.forEach((phase) => {
      const option = document.createElement('option'); option.value = phase; option.textContent = phase; batchPhase.append(option);
    });
    batchLabel.append(batchPhase);
    const batchButton = document.createElement('button'); batchButton.type = 'button'; batchButton.className = 'secondary'; batchButton.textContent = `批量更改（${selectedSegments.size}）`;
    batchButton.disabled = selectedSegments.size === 0;
    batchButton.addEventListener('click', async () => {
      const identifiers = Array.from(selectedSegments);
      if (!confirm(`将选中的 ${identifiers.length} 个片段改为 ${batchPhase.value}？`)) return;
      batchButton.disabled = true;
      try { await onPhaseChange(identifiers, batchPhase.value); }
      catch (error) { window.alert(`修改失败：${error.message}`); batchButton.disabled = false; }
    });
    controls.append(batchLabel, batchButton);
  }
  const note = document.createElement('p'); note.className = 'inline-note';
  const shownRows = rows.filter((row) => !selectedPhaseFilters.size || selectedPhaseFilters.has(row.phase));
  note.textContent = data.available ? `已提交 ${rows.length} 个阶段片段；${selectedPhaseFilters.size ? `当前显示 ${shownRows.length} 条。` : '运行中每秒刷新。'}` : '尚未有已完成航班。';
  if (!editable && data.edit_reason) note.textContent += ` 标签不可编辑：${data.edit_reason}。`;
  const labels = ['航班 ID', 'ICAO', 'CALL', '阶段', '点数', '起始时间', '终止时间', '时长 (s)', '起始经纬度', '终止经纬度', '跨度 (km)', '起始高度 (m)', '终止高度 (m)', '选择'];
  const table = document.createElement('table'); table.className = 'history-table';
  const head = table.createTHead().insertRow(); labels.forEach((label) => { const cell = document.createElement('th'); cell.textContent = label; head.append(cell); });
  const body = table.createTBody();
  const number = (value, digits = 3) => value == null ? '' : Number(value).toFixed(digits);
  const groups = [];
  shownRows.forEach((row) => {
    const flight = String(row.flight_id ?? `${row.icao ?? ''}\u0000${row.call ?? ''}`);
    if (!groups.length || groups.at(-1).flight !== flight) groups.push({ flight, rows: [] });
    groups.at(-1).rows.push(row);
  });
  groups.forEach((group, flightBlock) => {
    group.rows.forEach((row, index) => {
      const line = body.insertRow(); line.className = flightBlock % 2 === 0 ? 'segment-flight-even' : 'segment-flight-odd';
      if (index === 0) {
        const identity = line.insertCell(); identity.rowSpan = group.rows.length; identity.className = 'segment-flight-id';
        const flightButton = document.createElement('button'); flightButton.type = 'button'; flightButton.className = 'secondary';
        flightButton.textContent = group.flight.slice(0, 8); flightButton.title = `查看完整航班 ${group.flight} 的三维轨迹`;
        flightButton.addEventListener('click', () => { void showFlightPlot(artifact, group.flight); }); identity.append(flightButton);
      }
      [row.icao, row.call].forEach((value) => { line.insertCell().textContent = String(value ?? ''); });
      const phaseCell = line.insertCell();
      if (editable) {
        const editor = document.createElement('select'); phases.forEach((phase) => {
          const option = document.createElement('option'); option.value = phase; option.textContent = phase; editor.append(option);
        });
        editor.value = row.phase; editor.addEventListener('change', async () => {
          editor.disabled = true;
          try { await onPhaseChange([String(row.segment_id)], editor.value); }
          catch (error) { window.alert(`修改失败：${error.message}`); editor.disabled = false; }
        });
        phaseCell.append(editor);
      } else phaseCell.textContent = String(row.phase ?? '');
      [row.points, row.start_time, row.end_time, number(row.duration_seconds),
        `${number(row.start_longitude_deg, 5)}, ${number(row.start_latitude_deg, 5)}`,
        `${number(row.end_longitude_deg, 5)}, ${number(row.end_latitude_deg, 5)}`,
        number(row.span_km), number(row.start_altitude_m), number(row.end_altitude_m)].forEach((value) => { line.insertCell().textContent = String(value ?? ''); });
      const selectionCell = line.insertCell();
      const checkbox = document.createElement('input'); checkbox.type = 'checkbox'; checkbox.disabled = !editable; checkbox.checked = selectedSegments.has(String(row.segment_id));
      checkbox.addEventListener('change', () => {
        if (checkbox.checked) selectedSegments.add(String(row.segment_id)); else selectedSegments.delete(String(row.segment_id));
        renderSegmentResult(container, data, artifact, onPhaseChange);
      });
      selectionCell.append(checkbox);
    });
  });
  const scroll = document.createElement('div'); scroll.className = 'table-scroll'; scroll.append(table);
  container.append(controls, note, scroll);
}
