let segmentArtifact = null;
const selectedCells = new Set();
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

async function showTrajectoryPlot(title, address, completedText) {
  const request = ++flightPlotRequest;
  flightPlotTitle.textContent = title;
  flightPlotStatus.textContent = '正在读取或生成可拖动的三维轨迹…'; flightPlotFrame.hidden = true;
  if (!flightPlotDialog.open) flightPlotDialog.showModal();
  try {
    const response = await fetch(address);
    if (!response.ok) throw new Error((await response.json().catch(() => ({}))).error || `请求失败：${response.status}`);
    if (request !== flightPlotRequest) return;
    flightPlotFrame.onload = () => {
      if (request !== flightPlotRequest) return;
      flightPlotFrame.hidden = false;
      flightPlotStatus.textContent = completedText;
    };
    flightPlotFrame.src = address;
  } catch (error) { flightPlotStatus.textContent = `三维图生成失败：${error.message}`; }
}

async function showFlightPlot(artifact, flightId) {
  const address = `/api/segment/flight-plot-interactive?artifact=${encodeURIComponent(artifact)}&flight_id=${encodeURIComponent(flightId)}`;
  return showTrajectoryPlot(`航班 ${flightId.slice(0, 8)} 的三维轨迹`, address,
    '可拖动旋转、滚轮缩放；★ 为起点，➤ 为终点。颜色按硬时空单元固定；首次生成后缓存到本次运行目录。');
}

document.querySelector('#flight-plot-close').addEventListener('click', closeFlightPlot);
flightPlotDialog.addEventListener('click', (event) => { if (event.target === flightPlotDialog) closeFlightPlot(); });

function renderSegmentResult(container, data, artifact) {
  if (segmentArtifact !== artifact) { segmentArtifact = artifact; selectedCells.clear(); }
  container.replaceChildren();
  const rows = Array.isArray(data.rows) ? data.rows : [];
  const cells = Array.from(new Set(rows.map((row) => row.cell_key).filter(Boolean))).sort();
  const controls = document.createElement('fieldset'); controls.className = 'hard-cell-filter-options';
  const legend = document.createElement('legend'); legend.textContent = '硬时空单元筛选（可多选；未勾选显示全部）'; controls.append(legend);
  cells.forEach((cell) => {
    const label = document.createElement('label');
    const input = document.createElement('input'); input.type = 'checkbox'; input.value = cell; input.checked = selectedCells.has(cell);
    input.addEventListener('change', () => { if (input.checked) selectedCells.add(cell); else selectedCells.delete(cell); renderSegmentResult(container, data, artifact); });
    label.append(input, document.createTextNode(cell)); controls.append(label);
  });
  const shownRows = rows.filter((row) => !selectedCells.size || selectedCells.has(row.cell_key));
  const note = document.createElement('p'); note.className = 'inline-note';
  note.textContent = data.available ? `已提交 ${rows.length} 个硬时空片段；${selectedCells.size ? `当前显示 ${shownRows.length} 条。` : '运行中每秒刷新。'}` : '尚未有已完成航班。';
  const labels = ['航班 ID', 'ICAO', 'CALL', '硬单元', '高度层', '时间桶', '经度格', '纬度格', '点数', '起始时间', '终止时间', '时长 (s)', '起始经纬度', '终止经纬度', '跨度 (km)', '起始高度 (m)', '终止高度 (m)'];
  const table = document.createElement('table'); table.className = 'history-table';
  const head = table.createTHead().insertRow(); labels.forEach((label) => { const cell = document.createElement('th'); cell.textContent = label; head.append(cell); });
  const body = table.createTBody(); const number = (value, digits = 3) => value == null ? '' : Number(value).toFixed(digits);
  const flights = [];
  shownRows.forEach((row) => { const id = String(row.flight_id ?? ''); if (!flights.length || flights.at(-1).id !== id) flights.push({ id, rows: [] }); flights.at(-1).rows.push(row); });
  flights.forEach((flight, flightIndex) => flight.rows.forEach((row, index) => {
    const line = body.insertRow(); line.className = flightIndex % 2 === 0 ? 'segment-flight-even' : 'segment-flight-odd';
    if (index === 0) {
      const identity = line.insertCell(); identity.rowSpan = flight.rows.length; identity.className = 'segment-flight-id';
      const button = document.createElement('button'); button.type = 'button'; button.className = 'secondary'; button.textContent = flight.id.slice(0, 8);
      button.title = `查看完整航班 ${flight.id} 的三维轨迹`; button.addEventListener('click', () => { void showFlightPlot(artifact, flight.id); }); identity.append(button);
    }
    [row.icao, row.call, row.cell_key, row.height_band, row.time_bucket, row.longitude_cell, row.latitude_cell,
     row.points, row.start_time, row.end_time, number(row.duration_seconds),
     `${number(row.start_longitude_deg, 5)}, ${number(row.start_latitude_deg, 5)}`,
     `${number(row.end_longitude_deg, 5)}, ${number(row.end_latitude_deg, 5)}`,
     number(row.span_km), number(row.start_altitude_m), number(row.end_altitude_m)].forEach((value) => { line.insertCell().textContent = String(value ?? ''); });
  }));
  const scroll = document.createElement('div'); scroll.className = 'table-scroll'; scroll.append(table);
  container.append(controls, note, scroll);
}
