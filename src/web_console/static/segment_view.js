function renderSegmentResult(container, data) {
  container.replaceChildren();
  const labels = ['ICAO', 'CALL', '阶段', '点数', '起始时间', '终止时间', '时长 (s)', '起始经纬度', '终止经纬度', '跨度 (km)', '起始高度 (m)', '终止高度 (m)'];
  const table = document.createElement('table'); table.className = 'history-table';
  const head = table.createTHead().insertRow(); labels.forEach((label) => { const cell = document.createElement('th'); cell.textContent = label; head.append(cell); });
  const body = table.createTBody();
  const number = (value, digits = 3) => value == null ? '' : Number(value).toFixed(digits);
  (data.rows || []).forEach((row) => {
    const cells = [row.icao, row.call, row.phase, row.points, row.start_time, row.end_time,
      number(row.duration_seconds), `${number(row.start_longitude_deg, 5)}, ${number(row.start_latitude_deg, 5)}`,
      `${number(row.end_longitude_deg, 5)}, ${number(row.end_latitude_deg, 5)}`,
      number(row.span_km), number(row.start_altitude_m), number(row.end_altitude_m)];
    const line = body.insertRow(); cells.forEach((value) => { line.insertCell().textContent = String(value ?? ''); });
  });
  const note = document.createElement('p'); note.className = 'inline-note';
  note.textContent = data.available ? `已提交 ${data.rows.length} 个阶段片段；运行中每秒刷新。` : '尚未有已完成航班。';
  const scroll = document.createElement('div'); scroll.className = 'table-scroll'; scroll.append(table);
  container.append(note, scroll);
}
