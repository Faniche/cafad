"""segment_plot.py | 2026-11-15 | 单航班阶段着色静态及交互式三维预览 | Codex."""
from __future__ import annotations

from io import BytesIO
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
import re

import duckdb
import matplotlib
import numpy as np
import plotly.graph_objects as go
import plotly.io as plotly_io

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from src.utils.phase_palette import phase_color
from .history import _artifact_directory


MAX_POINTS_PER_PHASE = 600
_FLIGHT_ID = re.compile(r'[A-Za-z0-9_-]+\Z')
_CACHE_MARKER = b'<!-- CAFAD phase trajectory plot v4 -->\n'


def segment_flight_svg(project_root: Path, artifact: str, flight_id: str) -> bytes:
    """生成或读取统一阶段配色的静态 SVG，供离线查看和接口兼容使用。"""
    directory = _validated_directory(project_root, artifact, flight_id)
    cached = _flight_svg_path(directory, flight_id)
    if cached.is_file():
        image = cached.read_bytes()
        if image.startswith(_CACHE_MARKER):
            return image
    phase_paths, flight_path = _read_flight_paths(directory, flight_id)
    image = _CACHE_MARKER + _draw_flight_svg(flight_id, phase_paths, flight_path)
    _write_cached(cached, image)
    return image


def segment_flight_html(project_root: Path, artifact: str, flight_id: str) -> bytes:
    """生成或读取可拖动旋转的单航班 Plotly 三维 HTML。"""
    directory = _validated_directory(project_root, artifact, flight_id)
    cached = _flight_html_path(directory, flight_id)
    if cached.is_file():
        page = cached.read_bytes()
        if _CACHE_MARKER in page[:512]:
            return page
    phase_paths, flight_path = _read_flight_paths(directory, flight_id)
    page = _CACHE_MARKER + _draw_flight_html(flight_id, phase_paths, flight_path)
    _write_cached(cached, page)
    return page


@lru_cache(maxsize=1)
def plotly_javascript() -> bytes:
    """读取一次已安装 Plotly 的浏览器脚本，避免每个航班缓存 3 MB 以上副本。"""
    return files('plotly.package_data').joinpath('plotly.min.js').read_bytes()


def invalidate_segment_flight_svgs(directory: Path, flight_ids: set[str]) -> None:
    """标签人工修订后同时删除静态和交互式预览缓存。"""
    for flight_id in flight_ids:
        if _FLIGHT_ID.fullmatch(flight_id):
            _flight_svg_path(directory, flight_id).unlink(missing_ok=True)
            _flight_html_path(directory, flight_id).unlink(missing_ok=True)


def _validated_directory(project_root: Path, artifact: str, flight_id: str) -> Path:
    """统一校验 artifact 与稳定 flight ID，防止预览接口越界读取。"""
    if not _FLIGHT_ID.fullmatch(flight_id) or len(flight_id) > 128:
        raise ValueError('flight_id 无效')
    return _artifact_directory(project_root, 'segment', artifact)


def _flight_svg_path(directory: Path, flight_id: str) -> Path:
    """构造当前运行目录下的静态 SVG 缓存路径。"""
    return directory / 'flight_trajectory_svg' / f'{flight_id}.svg'


def _flight_html_path(directory: Path, flight_id: str) -> Path:
    """构造当前运行目录下的交互式 HTML 缓存路径。"""
    return directory / 'flight_trajectory_html' / f'{flight_id}.html'


def _write_cached(path: Path, value: bytes) -> None:
    """原子替换预览缓存，避免并发浏览时读到半写入文件。"""
    path.parent.mkdir(exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_bytes(value)
    temporary.replace(path)


def _read_flight_paths(directory: Path, flight_id: str) -> tuple[dict[str, list[np.ndarray]], np.ndarray]:
    """一次 DuckDB 查询取得分段路径及整条轨迹的时序起止点。"""
    data_dir = directory / 'data'
    mapping = data_dir / 'point_segment_map.parquet'
    observations = data_dir / 'input_observations.parquet'
    if not mapping.is_file() or not observations.is_file():
        raise ValueError('该 segment 运行尚未发布可绘图的点映射或观测表')
    with duckdb.connect(':memory:') as database:
        points = database.execute(
            'SELECT mapping.segment_id, mapping.phase, mapping.time_ns, '
            'observations.longitude_deg, observations.latitude_deg, observations.geometric_altitude_m '
            'FROM read_parquet(?) AS mapping JOIN read_parquet(?) AS observations USING (record_id) '
            'WHERE mapping.flight_id = ? ORDER BY mapping.time_ns, mapping.segment_id',
            [str(mapping), str(observations), flight_id],
        ).fetchall()
    if not points:
        raise ValueError('航班没有可绘制的阶段观测')
    by_segment: dict[tuple[str, str], list[tuple[float, float, float, float]]] = {}
    ordered: list[tuple[float, float, float, float]] = []
    for segment_id, phase, time_ns, longitude, latitude, altitude in points:
        if None in (time_ns, longitude, latitude, altitude):
            continue
        value = (float(time_ns), float(longitude), float(latitude), float(altitude))
        by_segment.setdefault((str(phase), str(segment_id)), []).append(value)
        ordered.append(value)
    if not ordered:
        raise ValueError('航班经纬度或几何高度均无有效值，无法绘图')
    phase_paths: dict[str, list[np.ndarray]] = {}
    for (phase, _segment_id), values in by_segment.items():
        path = np.asarray(values, dtype=float)[:, 1:4]
        if len(path) > MAX_POINTS_PER_PHASE:
            path = path[np.linspace(0, len(path) - 1, MAX_POINTS_PER_PHASE, dtype=int)]
        phase_paths.setdefault(phase, []).append(path)
    flight_path = np.asarray(ordered, dtype=float)
    flight_path = flight_path[np.argsort(flight_path[:, 0], kind='stable')][:, 1:4]
    return phase_paths, flight_path


def _draw_flight_svg(flight_id: str, phase_paths: dict[str, list[np.ndarray]], flight_path: np.ndarray) -> bytes:
    """绘制静态 SVG；颜色固定，首点为小星标、终段为方向箭头。"""
    figure = plt.figure(figsize=(10, 10), constrained_layout=True)
    axis = figure.add_subplot(projection='3d')
    phases = sorted(phase_paths)
    all_points: list[np.ndarray] = []
    for phase in phases:
        paths = phase_paths[phase]
        axis.add_collection3d(Line3DCollection(paths, colors=[phase_color(phase)], linewidths=1.15, alpha=0.88))
        all_points.extend(paths)
    coordinates = np.concatenate(all_points)
    axis.auto_scale_xyz(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2])
    _add_svg_end_markers(axis, flight_path)
    axis.set_xlabel('Longitude (deg)')
    axis.set_ylabel('Latitude (deg)')
    axis.set_zlabel('Geometric altitude (m)')
    axis.set_title(f'Flight {flight_id[:12]}: {len(all_points)} phase paths')
    axis.legend([Line2D([0], [0], color=phase_color(phase), lw=2) for phase in phases], phases,
                title='Phase', loc='best', fontsize=8)
    output = BytesIO()
    figure.savefig(output, format='svg')
    plt.close(figure)
    return output.getvalue()


def _add_svg_end_markers(axis: object, flight_path: np.ndarray) -> None:
    """加入不遮挡航迹的小起点星标和最终运动方向箭头。"""
    start = flight_path[0]
    axis.scatter(*start, marker='*', s=26, c='#111827', edgecolors='white', linewidths=0.35, depthshade=False)
    if len(flight_path) < 2:
        return
    previous, end = flight_path[-2], flight_path[-1]
    direction = end - previous
    if not np.any(direction):
        return
    axis.quiver(*previous, *direction, color='#111827', linewidth=0.75, arrow_length_ratio=0.12, normalize=False)


def _draw_flight_html(flight_id: str, phase_paths: dict[str, list[np.ndarray]], flight_path: np.ndarray) -> bytes:
    """生成浏览器原生旋转／缩放的 Plotly 三维页面，沿用静态图阶段色。"""
    figure = go.Figure()
    for phase in sorted(phase_paths):
        x_values: list[float | None] = []
        y_values: list[float | None] = []
        z_values: list[float | None] = []
        for path in phase_paths[phase]:
            x_values.extend(path[:, 0].tolist() + [None])
            y_values.extend(path[:, 1].tolist() + [None])
            z_values.extend(path[:, 2].tolist() + [None])
        figure.add_trace(go.Scatter3d(
            x=x_values, y=y_values, z=z_values, mode='lines', name=phase,
            line={'color': phase_color(phase), 'width': 4}, hovertemplate=f'{phase}<extra></extra>',
        ))
    figure.add_trace(go.Scatter3d(
        x=[flight_path[0, 0]], y=[flight_path[0, 1]], z=[flight_path[0, 2]], mode='text',
        text=['★'], textfont={'color': '#111827', 'size': 14}, name='Start', hovertemplate='Start<extra></extra>',
    ))
    figure.add_trace(go.Scatter3d(
        x=[flight_path[-1, 0]], y=[flight_path[-1, 1]], z=[flight_path[-1, 2]], mode='text',
        text=['➤'], textfont={'color': '#111827', 'size': 14}, name='End', hovertemplate='End<extra></extra>',
    ))
    figure.update_layout(
        title=f'Flight {flight_id[:12]}: phase paths',
        margin={'l': 0, 'r': 0, 'b': 0, 't': 42},
        legend={'title': {'text': 'Phase'}},
        scene={
            'dragmode': 'orbit',
            'xaxis_title': 'Longitude (deg)', 'yaxis_title': 'Latitude (deg)',
            'zaxis_title': 'Geometric altitude (m)',
        },
    )
    page = plotly_io.to_html(
        figure, include_plotlyjs='/api/segment/plotly.min.js', full_html=True,
        config={'scrollZoom': True, 'displaylogo': False, 'responsive': True},
    )
    return page.encode('utf-8')
