"""local_plot.py | 2026-11-11 | 本地分组轨迹三维绘图数据读取与 SVG 渲染 | Codex."""
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import as_completed
from dataclasses import dataclass
from io import BytesIO
import multiprocessing as mp
import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from src.utils.process_pool import InterruptibleProcessPoolExecutor

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection


MAX_POINTS_PER_PATH = 300
MAX_TOTAL_DISPLAY_POINTS = 50_000
PARALLEL_FINALIZE_MINIMUM_PATHS = 32
PARALLEL_FINALIZE_MAX_POINTS = 1_000_000
OBSERVATION_COLUMNS = ('record_id', 'flight_id', 'time_ns', 'longitude_deg', 'latitude_deg', 'geometric_altitude_m')


@dataclass(frozen=True)
class GroupIndex:
    """已选 group 结果中可用于本地筛选的最小索引。"""

    routing_path: Path
    point_map_path: Path
    members: pd.DataFrame

    @property
    def group_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.members['group_id'].unique()))


def locate_routing_metadata(path: Path) -> Path:
    """接受 routing_segments.csv 或 group 运行 manifest，并定位成员描述 CSV。"""
    selected = path.expanduser().resolve()
    if selected.name == 'routing_segments.csv':
        candidate = selected
    elif selected.name == 'manifest.json':
        candidate = selected.parent / 'data' / 'routing_segments.csv'
    else:
        candidate = selected.parent / 'routing_segments.csv'
    if not candidate.is_file():
        raise ValueError('分组元数据应为 routing_segments.csv，或其所属 group 运行的 manifest.json')
    return candidate


def locate_point_map(dataset_path: Path, routing_path: Path) -> Path:
    """按本地镜像目录优先级定位 segment 的 record_id 到 segment_id 映射。"""
    candidates = [dataset_path.parent / 'point_segment_map.csv', routing_path.parent / 'point_segment_map.csv']
    group_manifest = routing_path.parent.parent / 'manifest.json'
    if group_manifest.is_file():
        try:
            import json
            manifest = json.loads(group_manifest.read_text(encoding='utf-8'))
            for item in manifest.get('dependencies', []):
                value = item.get('manifest') if isinstance(item, dict) else None
                if not isinstance(value, str):
                    continue
                remote = Path(value)
                if remote.is_file():
                    candidates.append(remote.parent / 'data' / 'point_segment_map.csv')
                    continue
                parts = remote.parts
                if 'data' in parts:
                    data_root = next((parent for parent in routing_path.parents if parent.name == 'data'), None)
                    if data_root is not None:
                        candidates.append(data_root.joinpath(*parts[parts.index('data') + 1:]).parent / 'data' /
                                          'point_segment_map.csv')
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise ValueError('找不到 point_segment_map.csv。请选择 segment 运行 data/input_observations.csv，或保持 data 目录镜像结构。')


def load_group_index(dataset_path: Path, metadata_path: Path) -> GroupIndex:
    """加载并校验分组成员，且只保留绘图实际使用的字段。"""
    routing_path = locate_routing_metadata(metadata_path)
    members = pd.read_csv(routing_path, usecols=['segment_id', 'group_id', 'flight_id'], dtype='string')
    if members.empty or members.isna().any().any() or members['segment_id'].duplicated().any():
        raise ValueError('routing_segments.csv 为空、字段缺失或同一片段被分配到多个组')
    return GroupIndex(routing_path, locate_point_map(dataset_path, routing_path), members)


def segment_choices(index: GroupIndex, group_id: str | None) -> tuple[str, ...]:
    """返回某组内可独立查看的片段；未选组时不罗列全部片段。"""
    if group_id is None:
        return ()
    values = index.members.loc[index.members['group_id'].eq(group_id), 'segment_id'].astype(str)
    return tuple(sorted(values))


def render_plot(dataset_path: Path, index: GroupIndex, group_id: str | None, segment_id: str | None,
                whole_flight: bool, workers: int = 1) -> tuple[bytes, dict[str, object]]:
    """筛选数据并由 Matplotlib 生成 SVG；前端只嵌入该静态图像。"""
    selected = index.members
    if group_id is not None:
        selected = selected.loc[selected['group_id'].eq(group_id)]
    if segment_id is not None:
        selected = selected.loc[selected['segment_id'].eq(segment_id)]
    if selected.empty:
        raise ValueError('当前分组或片段没有可展示成员')
    paths = (_collect_flight_paths(dataset_path, selected['flight_id'].astype(str).unique(), workers) if whole_flight
             else _collect_segment_paths(dataset_path, index.point_map_path, selected['segment_id'].astype(str).unique(), workers))
    if not paths:
        raise ValueError('所选成员没有与数据集中的有效经纬高记录对齐')
    color_by_segment = index.members.set_index('segment_id')['group_id'].astype(str).to_dict()
    displayed = _display_paths(paths)
    image = _draw(displayed, color_by_segment, whole_flight)
    return image, {
        'group_count': int(selected['group_id'].nunique()), 'segment_count': int(selected['segment_id'].nunique()),
        'flight_count': int(selected['flight_id'].nunique()), 'path_count': len(paths),
        'display_points': int(sum(len(values) for values in displayed.values())),
        'display_point_budget': MAX_TOTAL_DISPLAY_POINTS,
        'mode': '整条轨迹' if whole_flight else '阶段片段', 'routing_file': str(index.routing_path),
        'configured_workers': workers,
    }


def _collect_segment_paths(dataset_path: Path, point_map_path: Path, segment_ids: np.ndarray,
                           workers: int) -> dict[str, np.ndarray]:
    """仅建立已选分组成员的 record_id 索引，避免把完整映射表常驻内存。"""
    wanted = set(segment_ids.tolist())
    record_to_segment: dict[str, str] = {}
    for chunk in pd.read_csv(point_map_path, usecols=['record_id', 'segment_id'], dtype='string', chunksize=200_000):
        chosen = chunk.loc[chunk['segment_id'].isin(wanted)]
        record_to_segment.update(zip(chosen['record_id'].astype(str), chosen['segment_id'].astype(str), strict=True))
    return _collect_paths(dataset_path, record_to_segment, by_record=True, workers=workers)


def _collect_flight_paths(dataset_path: Path, flight_ids: np.ndarray, workers: int) -> dict[str, np.ndarray]:
    """按分组成员所属航班收集完整飞行轨迹。"""
    return _collect_paths(dataset_path, set(flight_ids.tolist()), by_record=False, workers=workers)


def _collect_paths(dataset_path: Path, selected: dict[str, str] | set[str], *, by_record: bool,
                   workers: int) -> dict[str, np.ndarray]:
    """分块读取观测，按路径动态降采样并保持时间顺序。"""
    paths: dict[str, list[np.ndarray]] = defaultdict(list)
    for chunk in pd.read_csv(dataset_path, usecols=OBSERVATION_COLUMNS, dtype={'record_id': 'string', 'flight_id': 'string'},
                             chunksize=200_000):
        numeric = chunk[['time_ns', 'longitude_deg', 'latitude_deg', 'geometric_altitude_m']].apply(pd.to_numeric, errors='coerce')
        valid = numeric.notna().all(axis=1)
        if by_record:
            keys = chunk['record_id'].astype(str)
            selected_rows = chunk.loc[valid & keys.isin(selected)].copy()
            selected_rows['_path'] = selected_rows['record_id'].astype(str).map(selected)
        else:
            selected_rows = chunk.loc[valid & chunk['flight_id'].astype(str).isin(selected)].copy()
            selected_rows['_path'] = selected_rows['flight_id'].astype(str)
        if selected_rows.empty:
            continue
        selected_rows[['time_ns', 'longitude_deg', 'latitude_deg', 'geometric_altitude_m']] = numeric.loc[selected_rows.index]
        for path_id, frame in selected_rows.groupby('_path', sort=False):
            paths[str(path_id)].append(frame[['time_ns', 'longitude_deg', 'latitude_deg', 'geometric_altitude_m']].to_numpy(dtype=float))
    return _finalize_paths(paths, workers)


def _finalize_path(item: tuple[str, list[np.ndarray]]) -> tuple[str, np.ndarray]:
    """在一个独立任务中合并、稳定排序并降采样单条展示路径。"""
    path_id, chunks = item
    values = np.concatenate(chunks)
    values = values[np.argsort(values[:, 0], kind='stable')]
    return path_id, _downsample(values)


def _finalize_paths(paths: dict[str, list[np.ndarray]], workers: int) -> dict[str, np.ndarray]:
    """有足够多且规模受控的路径时并行后处理，避免复制完整原始 CSV。"""
    total_points = sum(sum(len(chunk) for chunk in chunks) for chunks in paths.values())
    parallel = min(workers, len(paths))
    if parallel < 2 or len(paths) < PARALLEL_FINALIZE_MINIMUM_PATHS or total_points > PARALLEL_FINALIZE_MAX_POINTS:
        return dict(_finalize_path(item) for item in paths.items())
    context = mp.get_context('spawn' if os.name == 'nt' else 'fork')
    result: dict[str, np.ndarray] = {}
    with InterruptibleProcessPoolExecutor(max_workers=parallel, mp_context=context) as pool:
        futures = [pool.submit(_finalize_path, item) for item in paths.items()]
        for future in as_completed(futures):
            path_id, values = future.result()
            result[path_id] = values
    return result


def _downsample(values: np.ndarray) -> np.ndarray:
    """显示层最多保留固定点数；不影响原始数据或分组统计。"""
    if len(values) <= MAX_POINTS_PER_PATH:
        return values
    return values[np.linspace(0, len(values) - 1, MAX_POINTS_PER_PATH, dtype=int)]


def _display_paths(paths: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """按总展示点预算二次采样，大量片段的总览不生成巨型 SVG。"""
    per_path = max(2, min(MAX_POINTS_PER_PATH, MAX_TOTAL_DISPLAY_POINTS // len(paths)))
    return {
        path_id: values if len(values) <= per_path else values[np.linspace(0, len(values) - 1, per_path, dtype=int)]
        for path_id, values in paths.items()
    }


def _draw(paths: dict[str, np.ndarray], color_by_segment: dict[str, str], whole_flight: bool) -> bytes:
    """按组批量绘制经度、纬度、几何高度三轴折线，并返回 SVG 字节。"""
    figure = plt.figure(figsize=(10, 7), constrained_layout=True)
    axis = figure.add_subplot(projection='3d')
    groups = sorted(set(color_by_segment.values()))
    palette = {group: plt.get_cmap('tab20')(number % 20) for number, group in enumerate(groups)}
    grouped_paths: dict[str, list[np.ndarray]] = defaultdict(list)
    for number, (path_id, values) in enumerate(paths.items()):
        group = color_by_segment.get(path_id)
        key = group if group is not None else f'unassigned-{number % 20}'
        grouped_paths[key].append(values[:, 1:4])
    for number, (group, segments) in enumerate(grouped_paths.items()):
        color = palette.get(group, plt.get_cmap('tab20')(number % 20))
        axis.add_collection3d(Line3DCollection(segments, colors=[color], linewidths=0.7, alpha=0.75))
    coordinates = np.concatenate(list(paths.values()))[:, 1:4]
    axis.auto_scale_xyz(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2])
    axis.set_xlabel('Longitude (deg)')
    axis.set_ylabel('Latitude (deg)')
    axis.set_zlabel('Geometric altitude (m)')
    axis.set_title(f"{'Whole flights' if whole_flight else 'Flight-phase segments'}: {len(paths)} paths")
    buffer = BytesIO()
    figure.savefig(buffer, format='svg')
    plt.close(figure)
    return buffer.getvalue()
