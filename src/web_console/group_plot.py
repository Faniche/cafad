"""group_plot.py | 2026-11-15 | 单一硬分组的交互式三维轨迹预览 | Codex."""
from __future__ import annotations

import hashlib
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as plotly_io

from .group_view import _read, group_source
from .segment_plot import MINIMUM_ALTITUDE_AXIS_UPPER_METERS


MAX_POINTS_PER_SEGMENT = 800
_CACHE_MARKER = b'<!-- CAFAD group trajectory plot v1 -->\n'


def group_trajectory_html(project_root: Path, artifact: str, group_id: str) -> bytes:
    """按需生成指定分组的 Plotly 页面，并在 group 运行目录缓存结果。"""
    if not group_id or len(group_id) > 256:
        raise ValueError('group_id 无效')
    directory, source = group_source(project_root, artifact)
    if directory is None or source is None:
        raise ValueError('分组尚未成功发布')
    cached = _cache_path(directory, group_id)
    if cached.is_file():
        page = cached.read_bytes()
        if page.startswith(_CACHE_MARKER):
            return page
    members = _read(directory / 'data', 'routing_segments',
                    columns=['group_id', 'segment_id', 'flight_id'],
                    dtype={'group_id': str, 'segment_id': str, 'flight_id': str})
    selected = members.loc[members['group_id'].eq(group_id), ['segment_id', 'flight_id']].drop_duplicates('segment_id')
    if selected.empty:
        raise ValueError('分组不存在或没有参考片段')
    paths = _read_group_paths(source, selected)
    page = _CACHE_MARKER + _draw_group_html(group_id, paths)
    _write_cached(cached, page)
    return page


def _cache_path(directory: Path, group_id: str) -> Path:
    """使用哈希缓存名，避免分组键成为路径的一部分。"""
    digest = hashlib.sha256(group_id.encode('utf-8')).hexdigest()[:20]
    return directory / 'group_trajectory_html' / f'{digest}.html'


def _read_group_paths(source: Path, members: pd.DataFrame) -> list[tuple[str, str, np.ndarray]]:
    """仅联结目标分组成员、点映射与观测；Parquet 路径交给 DuckDB 下推筛选。"""
    mapping_path = source / 'point_segment_map.parquet'
    observation_path = source / 'input_observations.parquet'
    if mapping_path.is_file() and observation_path.is_file():
        with duckdb.connect(':memory:') as database:
            database.register('selected_members', members)
            points = database.execute(
                'SELECT members.segment_id, members.flight_id, mapping.time_ns, '
                'observations.longitude_deg, observations.latitude_deg, observations.geometric_altitude_m '
                'FROM selected_members AS members '
                'JOIN read_parquet(?) AS mapping USING (segment_id) '
                'JOIN read_parquet(?) AS observations USING (record_id) '
                'WHERE mapping.time_ns IS NOT NULL AND observations.longitude_deg IS NOT NULL '
                'AND observations.latitude_deg IS NOT NULL AND observations.geometric_altitude_m IS NOT NULL '
                'ORDER BY members.segment_id, mapping.time_ns',
                [str(mapping_path), str(observation_path)],
            ).fetchdf()
    else:
        mapping = _read(source, 'point_segment_map', columns=['record_id', 'segment_id', 'time_ns'],
                        dtype={'record_id': str, 'segment_id': str})
        observations = _read(source, 'input_observations',
                             columns=['record_id', 'longitude_deg', 'latitude_deg', 'geometric_altitude_m'],
                             dtype={'record_id': str})
        points = (mapping.loc[mapping['segment_id'].isin(members['segment_id'])]
                  .merge(members, on='segment_id', how='inner', validate='many_to_one')
                  .merge(observations, on='record_id', how='inner', validate='one_to_one')
                  .dropna(subset=['time_ns', 'longitude_deg', 'latitude_deg', 'geometric_altitude_m'])
                  .sort_values(['segment_id', 'time_ns'], kind='stable'))
    if points.empty:
        raise ValueError('分组成员没有可绘制的经纬度或几何高度')
    paths: list[tuple[str, str, np.ndarray]] = []
    for segment_id, frame in points.groupby('segment_id', sort=True):
        path = frame[['longitude_deg', 'latitude_deg', 'geometric_altitude_m']].to_numpy(dtype=float)
        if len(path) > MAX_POINTS_PER_SEGMENT:
            path = path[np.linspace(0, len(path) - 1, MAX_POINTS_PER_SEGMENT, dtype=int)]
        paths.append((str(segment_id), str(frame['flight_id'].iloc[0]), path))
    return paths


def _trajectory_color(segment_id: str) -> str:
    """由稳定片段 ID 产生确定颜色，重开页面不会改变组内颜色对应。"""
    hue = int(hashlib.sha256(segment_id.encode('utf-8')).hexdigest()[:8], 16) % 360
    return f'hsl({hue}, 70%, 48%)'


def _draw_group_html(group_id: str, paths: list[tuple[str, str, np.ndarray]]) -> bytes:
    """绘制各片段独立着色的可旋转三维路径。"""
    figure = go.Figure()
    all_altitudes = []
    for segment_id, flight_id, path in paths:
        all_altitudes.append(path[:, 2])
        figure.add_trace(go.Scatter3d(
            x=path[:, 0], y=path[:, 1], z=path[:, 2], mode='lines',
            name=f'{flight_id[:8]} / {segment_id[:8]}',
            line={'color': _trajectory_color(segment_id), 'width': 4},
            hovertemplate=f'航班 {flight_id}<br>片段 {segment_id}<extra></extra>',
        ))
    altitude_upper = max(MINIMUM_ALTITUDE_AXIS_UPPER_METERS, float(np.max(np.concatenate(all_altitudes))))
    figure.update_layout(
        title=f'Group {group_id}: {len(paths)} trajectories', margin={'l': 0, 'r': 0, 'b': 0, 't': 42},
        scene={'dragmode': 'orbit', 'xaxis_title': 'Longitude (deg)', 'yaxis_title': 'Latitude (deg)',
               'zaxis': {'title': 'Geometric altitude (m)', 'range': [0.0, altitude_upper]}},
    )
    return plotly_io.to_html(
        figure, include_plotlyjs='/api/segment/plotly.min.js', full_html=True,
        config={'scrollZoom': True, 'displaylogo': False, 'responsive': True},
    ).encode('utf-8')


def _write_cached(path: Path, value: bytes) -> None:
    """原子替换缓存，避免并发点击读到半写页面。"""
    path.parent.mkdir(exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_bytes(value)
    temporary.replace(path)
