// group_view.js | 2026-11-15 | 分组统计表及按需三维组内轨迹预览 | Codex
function renderGroupResult(container, data, artifact) {
  container.replaceChildren();
  const note = document.createElement('p'); note.textContent = data.note; container.append(note);
  const groups = data.groups || [];
  if (!groups.length) return;
  const columns = [['group_id', '分组'], ['segments', '片段数量'], ['longest', '最长片段点数'], ['shortest', '最短片段点数'],
    ['earliest', '数据点最早时间（UTC）'], ['latest', '数据点最晚时间（UTC）'], ['min_alt', '空域最低高度（m）'],
    ['max_alt', '空域最高高度（m）'], ['min_lon', '最小经度'], ['max_lon', '最大经度'], ['min_lat', '最小纬度'], ['max_lat', '最大纬度']];
  const memberPanel = document.createElement('details');
  const memberTitle = document.createElement('summary'); memberTitle.textContent = '点击分组 ID 查看组内片段记录和三维轨迹';
  const memberBody = document.createElement('div'); memberBody.className = 'table-scroll'; memberPanel.append(memberTitle, memberBody);
  function showMembers(group) {
    memberBody.replaceChildren(); memberTitle.textContent = `${group.group_id}：${group.segments} 个正常训练片段`;
    const records = document.createElement('table'); records.className = 'history-table';
    const head = records.createTHead().insertRow(); ['片段 ID', '航班 ID', '点数'].forEach((text) => { const cell = document.createElement('th'); cell.textContent = text; head.append(cell); });
    const rows = records.createTBody(); group.members.forEach((item) => { const row = rows.insertRow(); [item.segment_id, item.flight_id, item.points].forEach((value) => { row.insertCell().textContent = String(value); }); });
    memberBody.append(records); memberPanel.open = true;
  }
  const scroll = document.createElement('div'); scroll.className = 'table-scroll';
  const table = document.createElement('table'); table.className = 'history-table';
  const header = table.createTHead().insertRow(); columns.forEach(([, label]) => { const cell = document.createElement('th'); cell.textContent = label; header.append(cell); });
  const body = table.createTBody();
  groups.forEach((group) => {
    const row = body.insertRow();
    const groupCell = row.insertCell(); const button = document.createElement('button'); button.type = 'button'; button.className = 'secondary'; button.textContent = group.group_id;
    button.title = `查看 ${group.group_id} 内各片段的三维轨迹`; button.addEventListener('click', () => {
      showMembers(group);
      const address = `/api/group/trajectory-plot-interactive?artifact=${encodeURIComponent(artifact)}&group_id=${encodeURIComponent(group.group_id)}`;
      void showTrajectoryPlot(`分组 ${group.group_id} 的三维轨迹`, address,
        '可拖动旋转、滚轮缩放；组内每条片段轨迹使用独立颜色，首次生成后缓存到本次运行目录。');
    });
    groupCell.append(button);
    columns.slice(1).forEach(([key]) => { row.insertCell().textContent = String(group[key]); });
  });
  scroll.append(table); container.append(scroll, memberPanel);
}
