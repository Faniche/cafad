// group_view.js | 2026-11-09 | 无外部依赖的分组统计与可旋转三维范围预览 | Codex
function renderGroupResult(container, data) {
  container.replaceChildren();
  const note = document.createElement('p'); note.textContent = data.note; container.append(note);
  const groups = data.groups || [];
  if (!groups.length) return;
  const colors = new Map(groups.map((g, i) => [g.group_id, `hsl(${i * 137.508 % 360} 75% 65%)`]));
  const columns = [['group_id', '分组'], ['segments', '片段数量'], ['longest', '最长片段点数'], ['shortest', '最短片段点数'],
    ['earliest', '数据点最早时间（UTC）'], ['latest', '数据点最晚时间（UTC）'], ['min_alt', '空域最低高度（m）'],
    ['max_alt', '空域最高高度（m）'], ['min_lon', '最小经度'], ['max_lon', '最大经度'], ['min_lat', '最小纬度'], ['max_lat', '最大纬度']];
  const scroll = document.createElement('div'); scroll.className = 'table-scroll';
  const table = document.createElement('table'); table.className = 'history-table';
  const header = table.createTHead().insertRow();
  columns.forEach(([, label]) => { const cell = document.createElement('th'); cell.textContent = label; header.append(cell); });
  const body = table.createTBody();
  const controls = document.createElement('p');
  const select = document.createElement('select'); select.setAttribute('aria-label', '选择分组');
  select.add(new Option('显示全部分组', ''));
  groups.forEach((group) => select.add(new Option(group.group_id, group.group_id)));
  const reset = document.createElement('button'); reset.type = 'button'; reset.textContent = '重置视角'; reset.className = 'secondary';
  controls.append(select, document.createTextNode(' 拖动旋转，滚轮缩放；点击分组查看成员。 '), reset);
  const canvas = document.createElement('canvas'); canvas.setAttribute('aria-label', '经度、纬度、几何高度三维轨迹和分组范围');
  const memberPanel = document.createElement('details');
  const memberTitle = document.createElement('summary'); memberTitle.textContent = '选择分组后查看片段记录';
  const memberBody = document.createElement('div'); memberBody.className = 'table-scroll'; memberPanel.append(memberTitle, memberBody);
  for (const group of groups) {
    const row = body.insertRow(); row.style.cursor = 'pointer';
    columns.forEach(([key]) => { const cell = row.insertCell(); cell.textContent = String(group[key]); });
    row.cells[0].style.color = colors.get(group.group_id);
    row.addEventListener('click', () => { select.value = group.group_id; chooseGroup(); });
  }
  scroll.append(table); container.append(scroll, controls, canvas, memberPanel);
  let yaw = -0.65, pitch = 0.55, zoom = 1, dragging = null, scheduled = false;
  const bounds = ['lon', 'lat', 'alt'].map((axis) => [Math.min(...groups.map(g => g[`min_${axis}`])), Math.max(...groups.map(g => g[`max_${axis}`]))]);
  // 范围也包括灰色背景片段，避免验证/测试轨迹落在坐标轴之外。
  for (const path of data.trajectories || []) for (const point of path.points) point.forEach((v, i) => {
    bounds[i][0] = Math.min(bounds[i][0], v); bounds[i][1] = Math.max(bounds[i][1], v);
  });
  const normalized = (p) => p.map((v, i) => (v - bounds[i][0]) / (bounds[i][1] - bounds[i][0] || 1) * 2 - 1);
  const paths = (data.trajectories || []).map(path => ({ ...path, points: path.points.map(normalized) }));
  const edges = [[0,1],[0,2],[0,4],[1,3],[1,5],[2,3],[2,6],[3,7],[4,5],[4,6],[5,7],[6,7]];
  const boxes = groups.map(group => ({id: group.group_id, corners: Array.from({length:8}, (_, i) => normalized([
    group[(i & 1) ? 'max_lon' : 'min_lon'], group[(i & 2) ? 'max_lat' : 'min_lat'], group[(i & 4) ? 'max_alt' : 'min_alt']]))}));
  function draw() {
    scheduled = false;
    const width = Math.max(320, canvas.clientWidth), height = 460, ratio = window.devicePixelRatio || 1;
    canvas.width = width * ratio; canvas.height = height * ratio;
    const ctx = canvas.getContext('2d'); ctx.scale(ratio, ratio); ctx.clearRect(0, 0, width, height);
    const scale = Math.min(width * 0.25, height * 0.30) * zoom;
    function project([x,y,z]) {
      const rx = x * Math.cos(yaw) - y * Math.sin(yaw), ry = x * Math.sin(yaw) + y * Math.cos(yaw);
      return [width / 2 + rx * scale, height / 2 - (z * Math.cos(pitch) - ry * Math.sin(pitch)) * scale];
    }
    function line(points, color, alpha, thickness = 1) {
      if (!points.length) return;
      ctx.strokeStyle = color; ctx.globalAlpha = alpha; ctx.lineWidth = thickness; ctx.beginPath();
      points.forEach((point, i) => { const [x,y] = project(point); if (i) ctx.lineTo(x,y); else ctx.moveTo(x,y); }); ctx.stroke();
      if (points.length === 1) { ctx.fillStyle = color; ctx.beginPath(); ctx.arc(...project(points[0]), 2, 0, Math.PI*2); ctx.fill(); }
    }
    for (const path of paths) {
      const chosen = !select.value || path.group_id === select.value;
      line(path.points, colors.get(path.group_id) || '#94a3b8', path.group_id ? (chosen ? 0.65 : 0.06) : 0.08);
    }
    for (const box of boxes) {
      if (select.value && box.id !== select.value) continue;
      ctx.fillStyle = colors.get(box.id); ctx.globalAlpha = 0.06;
      for (const face of [[0,1,3,2],[4,5,7,6],[0,1,5,4],[2,3,7,6],[0,2,6,4],[1,3,7,5]]) {
        ctx.beginPath(); face.forEach((index, i) => { const [x,y] = project(box.corners[index]); if (i) ctx.lineTo(x,y); else ctx.moveTo(x,y); });
        ctx.closePath(); ctx.fill();
      }
      edges.forEach(([a,b]) => line([box.corners[a], box.corners[b]], colors.get(box.id), 0.55));
      ctx.globalAlpha = 1; ctx.fillStyle = colors.get(box.id); ctx.font = '12px sans-serif';
      ctx.fillText(box.id, ...project(box.corners[7]));
    }
    const origin = [-1,-1,-1];
    ['经度 °', '纬度 °', '几何高度 m'].forEach((label, axis) => {
      const end = [...origin]; end[axis] = 1.25; line([origin, end], '#e2e8f0', 1, 1.5);
      ctx.fillStyle = '#e2e8f0'; ctx.font = '12px sans-serif'; ctx.fillText(label, ...project(end));
      for (const fraction of [0,0.5,1]) {
        const at = [...origin]; at[axis] = fraction * 2 - 1;
        const [x,y] = project(at); const value = bounds[axis][0] + fraction * (bounds[axis][1] - bounds[axis][0]);
        const offset = [[6,-8],[-42,20],[-32,-6]][axis];
        ctx.fillText(value.toFixed(axis === 2 ? 0 : 2), x + offset[0], y + offset[1]);
      }
    });
    ctx.globalAlpha = 1;
  }
  function requestDraw() { if (!scheduled) { scheduled = true; requestAnimationFrame(draw); } }
  function chooseGroup() {
    memberBody.replaceChildren();
    const group = groups.find(item => item.group_id === select.value);
    memberTitle.textContent = group ? `${group.group_id}：${group.segments} 个正常训练片段` : '选择分组后查看片段记录';
    if (group) {
      const records = document.createElement('table'); records.className = 'history-table';
      const head = records.createTHead().insertRow(); ['片段 ID', '航班 ID', '点数'].forEach(text => { const cell = document.createElement('th'); cell.textContent = text; head.append(cell); });
      const rows = records.createTBody(); group.members.forEach(item => { const row = rows.insertRow(); [item.segment_id,item.flight_id,item.points].forEach(value => { row.insertCell().textContent = String(value); }); });
      memberBody.append(records); memberPanel.open = true;
    }
    requestDraw();
  }
  select.addEventListener('change', chooseGroup);
  reset.addEventListener('click', () => { yaw = -0.65; pitch = 0.55; zoom = 1; requestDraw(); });
  canvas.addEventListener('pointerdown', e => { dragging = [e.clientX,e.clientY]; canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener('pointermove', e => { if (!dragging) return; yaw += (e.clientX-dragging[0])*0.008; pitch = Math.max(-1.4,Math.min(1.4,pitch+(e.clientY-dragging[1])*0.008)); dragging=[e.clientX,e.clientY]; requestDraw(); });
  canvas.addEventListener('pointerup', () => { dragging = null; }); canvas.addEventListener('pointercancel', () => { dragging = null; });
  canvas.addEventListener('wheel', e => { e.preventDefault(); zoom = Math.max(0.3,Math.min(4,zoom*Math.exp(-e.deltaY*0.001))); requestDraw(); }, {passive:false});
  requestDraw();
}
