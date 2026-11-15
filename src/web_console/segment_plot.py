"""segment_plot.py | 2026-11-14 | 单航班阶段着色三维 SVG 预览 | Codex."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import re

import duckdb
import matplotlib
import numpy as np

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from .history import _artifact_directory


MAX_POINTS_PER_PHASE = 600
_FLIGHT_ID = re.compile(r'[A-Za-z0-9_-]+\Z')


def segment_flight_svg(project_root: Path, artifact: str, flight_id: str) -> bytes:
    """读取一个已分段航班的观测，以最终阶段标签着色生成只读三维 SVG。"""
    if not _FLIGHT_ID.fullmatch(flight_id) or len(flight_id) > 128:
        raise ValueError('flight_id 无效')
    directory = _artifact_directory(project_root, 'segment', artifact)
    cached = _flight_svg_path(directory, flight_id)
    if cached.is_file():
        return cached.read_bytes()
    data_dir = directory / 'data'
    mapping = data_dir / 'point_segment_map.parquet'
    observations = data_dir / 'input_observations.parquet'
    if not mapping.is_file() or not observations.is_file():
        raise ValueError('该 segment 运行尚未发布可绘图的点映射或观测表')
    image = _render_flight(mapping, observations, flight_id)
    cached.parent.mkdir(exist_ok=True)
    temporary = cached.with_suffix('.svg.tmp')
    temporary.write_bytes(image)
    temporary.replace(cached)
    return image


def invalidate_segment_flight_svgs(directory: Path, flight_ids: set[str]) -> None:
    """阶段标签人工修订后删除受影响航班的预览缓存，避免颜色与当前标签不一致。"""
    for flight_id in flight_ids:
        if _FLIGHT_ID.fullmatch(flight_id):
            _flight_svg_path(directory, flight_id).unlink(missing_ok=True)


def _flight_svg_path(directory: Path, flight_id: str) -> Path:
    """以受限的稳定 flight_id 构造当前运行目录下的缓存 SVG 文件名。"""
    return directory / 'flight_trajectory_svg' / f'{flight_id}.svg'


def _render_flight(mapping: Path, observations: Path, flight_id: str) -> bytes:
    """首次点击时以 DuckDB 查询所选航班并渲染图像。"""
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
    phase_paths: dict[str, list[np.ndarray]] = {}
    by_segment: dict[tuple[str, str], list[tuple[float, float, float]]] = {}
    for segment_id, phase, _time_ns, longitude, latitude, altitude in points:
        if None in (longitude, latitude, altitude):
            continue
        by_segment.setdefault((str(phase), str(segment_id)), []).append((float(longitude), float(latitude), float(altitude)))
    for (phase, _segment_id), values in by_segment.items():
        path = np.asarray(values, dtype=float)
        if len(path) > MAX_POINTS_PER_PHASE:
            path = path[np.linspace(0, len(path) - 1, MAX_POINTS_PER_PHASE, dtype=int)]
        phase_paths.setdefault(phase, []).append(path)
    if not phase_paths:
        raise ValueError('航班经纬度或几何高度均无有效值，无法绘图')
    return _draw_flight(flight_id, phase_paths)


def _draw_flight(flight_id: str, phase_paths: dict[str, list[np.ndarray]]) -> bytes:
    """以阶段为图例分组绘制经度、纬度、几何高度三轴路径。"""
    figure = plt.figure(figsize=(10, 10), constrained_layout=True)
    axis = figure.add_subplot(projection='3d')
    phases = sorted(phase_paths)
    palette = {phase: plt.get_cmap('tab20')(index % 20) for index, phase in enumerate(phases)}
    all_points = []
    for phase in phases:
        paths = phase_paths[phase]
        axis.add_collection3d(Line3DCollection(paths, colors=[palette[phase]], linewidths=1.15, alpha=0.88))
        all_points.extend(paths)
    coordinates = np.concatenate(all_points)
    axis.auto_scale_xyz(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2])
    axis.set_xlabel('Longitude (deg)')
    axis.set_ylabel('Latitude (deg)')
    axis.set_zlabel('Geometric altitude (m)')
    axis.set_title(f'Flight {flight_id[:12]}: {len(all_points)} phase paths')
    axis.legend([Line2D([0], [0], color=palette[phase], lw=2) for phase in phases], phases,
                title='Phase', loc='best', fontsize=8)
    output = BytesIO()
    figure.savefig(output, format='svg')
    plt.close(figure)
    return output.getvalue()
