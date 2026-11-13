function updateSegmentMergePanel(panel, rows, state, options) {
  panel.replaceChildren();
  const selected = rows.filter((row) => state.ids.has(String(row.segment_id)));
  const heading = document.createElement('h4'); heading.textContent = '手动合并设置'; panel.append(heading);
  if (!selected.length) {
    const note = document.createElement('p'); note.className = 'inline-note'; note.textContent = '勾选同一航班中连续的阶段片段后，在此选择合并后的阶段标签。'; panel.append(note); return;
  }
  const flightIds = new Set(selected.map((row) => String(row.flight_id)));
  const positions = selected.map((row) => rows.indexOf(row));
  const contiguous = Math.max(...positions) - Math.min(...positions) + 1 === positions.length;
  if (flightIds.size !== 1 || !contiguous) {
    const note = document.createElement('p'); note.className = 'form-status';
    note.textContent = '只能合并同一航班中时间连续的片段；请调整勾选范围。'; panel.append(note); return;
  }
  const note = document.createElement('p'); note.className = 'inline-note';
  note.textContent = `已选择 ${selected.length} 个连续片段；合并后阶段标签：`; panel.append(note);
  const phases = [...new Set(selected.map((row) => String(row.phase)))];
  if (!phases.includes(state.phase)) state.phase = phases[0];
  const choices = document.createElement('div'); choices.className = 'segment-merge-choices';
  phases.forEach((phase) => {
    const label = document.createElement('label'); const input = document.createElement('input');
    input.type = 'radio'; input.name = 'segment-merge-phase'; input.value = phase; input.checked = state.phase === phase;
    input.addEventListener('change', () => { state.phase = phase; updateSegmentMergePanel(panel, rows, state, options); });
    label.append(input, document.createTextNode(phase)); choices.append(label);
  });
  panel.append(choices);
  const preview = document.createElement('p'); preview.className = 'inline-note';
  preview.textContent = `拟合并为 ${state.phase}。确认后会同步更新该运行的片段、点映射、阶段序列审计和展示记录。`;
  panel.append(preview);
  const submit = document.createElement('button'); submit.type = 'button'; submit.textContent = '确认合并';
  submit.disabled = !options.canMerge;
  submit.addEventListener('click', async () => {
    submit.disabled = true; preview.textContent = '正在写入合并结果…';
    try {
      await options.merge({ segment_ids: selected.map((row) => String(row.segment_id)), phase: state.phase });
      state.ids.clear(); state.phase = '';
    } catch (error) { preview.textContent = `合并失败：${error.message}`; submit.disabled = false; }
  });
  panel.append(submit);
  if (!options.canMerge) {
    const status = document.createElement('p'); status.className = 'form-status';
    status.textContent = '仅已完成的 segment 运行可以手动合并。'; panel.append(status);
  }
}

function renderSegmentResult(container, data, options = {}) {
  container.replaceChildren();
  const labels = ['航班 ID', 'ICAO', 'CALL', '阶段', '点数', '起始时间', '终止时间', '时长 (s)', '起始经纬度', '终止经纬度', '跨度 (km)', '起始高度 (m)', '终止高度 (m)', '合并'];
  const table = document.createElement('table'); table.className = 'history-table';
  const head = table.createTHead().insertRow(); labels.forEach((label) => { const cell = document.createElement('th'); cell.textContent = label; head.append(cell); });
  const body = table.createTBody();
  const number = (value, digits = 3) => value == null ? '' : Number(value).toFixed(digits);
  const state = container._segmentMergeState || { ids: new Set(), phase: '' };
  const rows = data.rows || [];
  const available = new Set(rows.map((row) => String(row.segment_id)));
  state.ids.forEach((segmentId) => { if (!available.has(segmentId)) state.ids.delete(segmentId); });
  container._segmentMergeState = state;
  const groups = [];
  rows.forEach((row) => {
    const flight = String(row.flight_id ?? `${row.icao ?? ''}\u0000${row.call ?? ''}`);
    if (!groups.length || groups.at(-1).flight !== flight) groups.push({ flight, rows: [] });
    groups.at(-1).rows.push(row);
  });
  groups.forEach((group, flightBlock) => {
    group.rows.forEach((row, index) => {
      const cells = [row.icao, row.call, row.phase, row.points, row.start_time, row.end_time,
        number(row.duration_seconds), `${number(row.start_longitude_deg, 5)}, ${number(row.start_latitude_deg, 5)}`,
        `${number(row.end_longitude_deg, 5)}, ${number(row.end_latitude_deg, 5)}`,
        number(row.span_km), number(row.start_altitude_m), number(row.end_altitude_m)];
      const line = body.insertRow();
      line.className = flightBlock % 2 === 0 ? 'segment-flight-even' : 'segment-flight-odd';
      if (index === 0) {
        const identity = line.insertCell();
        identity.rowSpan = group.rows.length;
        identity.className = 'segment-flight-id';
        identity.textContent = group.flight.slice(0, 8);
      }
      cells.forEach((value) => { line.insertCell().textContent = String(value ?? ''); });
      const select = document.createElement('input'); select.type = 'checkbox'; select.checked = state.ids.has(String(row.segment_id));
      select.addEventListener('change', () => {
        if (select.checked) state.ids.add(String(row.segment_id)); else state.ids.delete(String(row.segment_id));
        updateSegmentMergePanel(mergePanel, rows, state, options);
      });
      line.insertCell().append(select);
    });
  });
  const note = document.createElement('p'); note.className = 'inline-note';
  note.textContent = data.available ? `已提交 ${rows.length} 个阶段片段；运行中每秒刷新。` : '尚未有已完成航班。';
  const scroll = document.createElement('div'); scroll.className = 'table-scroll'; scroll.append(table);
  const mergePanel = document.createElement('aside'); mergePanel.className = 'segment-merge-panel';
  const layout = document.createElement('div'); layout.className = 'segment-merge-layout'; layout.append(scroll, mergePanel);
  container.append(note, layout); updateSegmentMergePanel(mergePanel, rows, state, options);
}
