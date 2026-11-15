"""group_distance_experiment.py | 2026-11-12 | HDBSCAN 时空距离离线诊断 | Codex.

只读一个 segment 产物，比较现行 13 维距离和阶段内时空门控距离；不发布 group 产物，
也不修改训练路由。
"""
from __future__ import annotations

import argparse
from concurrent.futures import as_completed
import json
from dataclasses import dataclass
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN

from src.app.tables import read_table
from src.modules.group.api import _read_segment_store
from src.modules.group.features import FEATURE_NAMES, build_segment_features, fit_feature_scaler, transform_features
from src.utils.process_pool import InterruptibleProcessPoolExecutor


KINEMATIC_WEIGHTS = {
    'mean_geometric_altitude_m': 1.0,
    'altitude_range_m': 0.5,
    'mean_speed_mps': 0.7,
    'std_speed_mps': 0.3,
    'mean_vertical_rate_mps': 1.0,
    'mean_acceleration_mps2': 0.2,
    'mean_heading_sin': 0.25,
    'mean_heading_cos': 0.25,
    'mean_abs_turn_rate_radps': 0.2,
    'duration_s': 0.3,
}


@dataclass(frozen=True)
class Candidate:
    minimum_size: int
    minimum_samples: int
    labels: np.ndarray
    probabilities: np.ndarray
    groups: tuple[tuple[int, np.ndarray], ...]
    coverage: float
    probability: float


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='比较 HDBSCAN 现行与时空门控距离，不修改正式数据。')
    parser.add_argument('--segment-manifest', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--space-scales-km', nargs='+', type=float, default=(10., 20., 50.))
    parser.add_argument('--time-scales-seconds', nargs='+', type=float, default=(300., 600., 1200.))
    parser.add_argument('--space-gate-km', type=float, default=50.)
    parser.add_argument('--time-gate-seconds', type=float, default=1800.)
    parser.add_argument('--min-cluster-sizes', nargs='+', type=int, default=(4, 8, 12, 16))
    parser.add_argument('--min-samples-values', nargs='+', type=int, default=(1, 2, 4))
    parser.add_argument('--min-group-members', type=int, default=12)
    parser.add_argument('--min-reference-flights', type=int, default=2)
    parser.add_argument('--workers', type=int, default=30, help='距离配置级诊断进程数上限（默认 30）。')
    return parser


def _haversine_matrix(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    """返回每对点的球面距离，单位 km。"""
    lat = np.deg2rad(latitude)[:, None]
    lon = np.deg2rad(longitude)[:, None]
    dlat, dlon = lat - lat.T, lon - lon.T
    a = np.sin(dlat / 2.) ** 2 + np.cos(lat) * np.cos(lat.T) * np.sin(dlon / 2.) ** 2
    return 6371.0088 * 2. * np.arctan2(np.sqrt(np.maximum(a, 0.)), np.sqrt(np.maximum(1. - a, 0.)))


def _candidate_pairs(arguments: argparse.Namespace) -> tuple[tuple[int, int], ...]:
    return tuple((size, samples) for size in sorted(set(arguments.min_cluster_sizes))
                 for samples in sorted(set(arguments.min_samples_values)) if 1 <= samples <= size)


def _fit_candidates(matrix: np.ndarray, phase_rows: pd.DataFrame, arguments: argparse.Namespace) -> tuple[Candidate, list[dict[str, object]]]:
    """与正式模块同样按有效成员和不同参考航班过滤，但不施加专家数下限。"""
    rows: list[dict[str, object]] = []
    candidates: list[Candidate] = []
    for minimum_size, minimum_samples in _candidate_pairs(arguments):
        if minimum_size > len(phase_rows):
            continue
        fitted = HDBSCAN(min_cluster_size=minimum_size, min_samples=minimum_samples, metric='precomputed',
                         algorithm='brute', n_jobs=1, cluster_selection_method='eom', allow_single_cluster=True,
                         copy=False).fit(matrix.copy())
        labels, probabilities = np.asarray(fitted.labels_), np.asarray(fitted.probabilities_)
        valid: list[tuple[int, np.ndarray]] = []
        for label in sorted(int(item) for item in set(labels) if item >= 0):
            positions = np.flatnonzero(labels == label)
            members = phase_rows.iloc[positions]
            if (len(members) >= arguments.min_group_members
                    and members['flight_id'].nunique() >= arguments.min_reference_flights):
                valid.append((label, positions))
        selected = np.concatenate([positions for _, positions in valid]) if valid else np.array([], dtype=np.int64)
        coverage = float(len(selected) / len(phase_rows))
        probability = float(probabilities[selected].mean()) if len(selected) else 0.0
        candidate = Candidate(minimum_size, minimum_samples, labels, probabilities, tuple(valid), coverage, probability)
        candidates.append(candidate)
        rows.append({'min_cluster_size': minimum_size, 'min_samples': minimum_samples,
                     'effective_groups': len(valid), 'coverage': coverage, 'mean_membership_probability': probability})
    if not candidates:
        raise ValueError('没有可运行的 HDBSCAN 参数组合')
    return max(candidates, key=lambda item: (item.coverage, item.probability, item.minimum_size, item.minimum_samples)), rows


def _spatiotemporal_matrix(rows: pd.DataFrame, standardized: pd.DataFrame, space_scale_km: float,
                           time_scale_seconds: float, space_gate_km: float, time_gate_seconds: float) -> np.ndarray:
    """同阶段的实验距离：起终位置、中心时间为主项，运动学为辅助项。"""
    start = _haversine_matrix(rows['start_latitude_deg'].to_numpy(float), rows['start_longitude_deg'].to_numpy(float))
    end = _haversine_matrix(rows['end_latitude_deg'].to_numpy(float), rows['end_longitude_deg'].to_numpy(float))
    space = np.sqrt((start ** 2 + end ** 2) / 2.)
    elapsed = np.abs(rows['center_time_s'].to_numpy(float)[:, None] - rows['center_time_s'].to_numpy(float)[None, :])
    squared = 4.0 * (space / space_scale_km) ** 2 + 3.0 * (elapsed / time_scale_seconds) ** 2
    for feature, weight in KINEMATIC_WEIGHTS.items():
        delta = standardized[feature].to_numpy(float)[:, None] - standardized[feature].to_numpy(float)[None, :]
        squared += weight * delta ** 2
    matrix = np.sqrt(np.maximum(squared, 0.0))
    # 有限大值只用于诊断中阻断候选边；正式实现若采用该规则，应改为连通分量内分别聚类。
    matrix[(space > space_gate_km) | (elapsed > time_gate_seconds)] = 1_000_000.0
    np.fill_diagonal(matrix, 0.0)
    return matrix


def _hard_cells(rows: pd.DataFrame, space_scale_km: float, time_scale_seconds: float) -> list[tuple[str, pd.DataFrame]]:
    """按中心位置和中心时间的固定格分区，阻断 HDBSCAN 的近邻链跨时空扩张。"""
    degree = space_scale_km / 111.32
    values = rows.copy()
    values['_time_cell'] = np.floor(values['center_time_s'].to_numpy(float) / time_scale_seconds).astype(np.int64)
    values['_lat_cell'] = np.floor((values['mean_latitude_deg'].to_numpy(float) + 90.0) / degree).astype(np.int64)
    longitude = np.mod(values['mean_longitude_deg'].to_numpy(float) + 180.0, 360.0)
    values['_lon_cell'] = np.floor(longitude / degree).astype(np.int64)
    return [(f't{time_bin}_y{latitude}_x{longitude}', part.drop(columns=['_time_cell', '_lat_cell', '_lon_cell']))
            for (time_bin, latitude, longitude), part in values.groupby(['_time_cell', '_lat_cell', '_lon_cell'], sort=True, observed=True)]


def _circular_span(values: pd.Series) -> float:
    radians = np.sort(np.deg2rad(np.mod(values.to_numpy(float), 360.0)))
    if len(radians) < 2:
        return 0.0
    gaps = np.diff(np.r_[radians, radians[0] + 2 * np.pi])
    return float(np.rad2deg(2 * np.pi - gaps.max()))


def _group_statistics(assignments: pd.DataFrame, segments: pd.DataFrame, values: pd.DataFrame) -> pd.DataFrame:
    joined = assignments.merge(segments[['segment_id']], on='segment_id', how='left', validate='many_to_one')
    points = assignments[['config', 'group_id', 'segment_id']].merge(values, on='segment_id', how='left', validate='many_to_many')
    rows: list[dict[str, object]] = []
    for (config, group_id), members in joined.groupby(['config', 'group_id'], sort=True):
        group_points = points.loc[(points['config'] == config) & (points['group_id'] == group_id)]
        times = group_points['time_ns'].to_numpy(np.int64)
        rows.append({
            'config': config, 'group_id': group_id, 'phase': members['phase'].iloc[0],
            'segments': len(members), 'flights': members['flight_id'].nunique(),
            'time_start': group_points['time_ns'].min(), 'time_end': group_points['time_ns'].max(),
            'time_span_seconds': float((times.max() - times.min()) / 1_000_000_000.0),
            'longitude_min': group_points['longitude_deg'].min(), 'longitude_max': group_points['longitude_deg'].max(),
            'latitude_min': group_points['latitude_deg'].min(), 'latitude_max': group_points['latitude_deg'].max(),
            'altitude_min_m': group_points['geometric_altitude_m'].min(), 'altitude_max_m': group_points['geometric_altitude_m'].max(),
            'speed_min_mps': group_points['speed_mps'].min(), 'speed_max_mps': group_points['speed_mps'].max(),
            'heading_circular_span_deg': _circular_span(group_points['heading_deg']),
        })
    return pd.DataFrame(rows)


def _run_phase_configuration(config_name: str, mode: str, space_scale: float | None, time_scale: float | None,
                             phase: str, phase_rows: pd.DataFrame, arguments: argparse.Namespace
                             ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """一个子进程独立跑一个“距离配置 × 阶段”，HDBSCAN 内部始终单线程。"""
    selections: list[dict[str, object]] = []
    assignments: list[dict[str, object]] = []
    phase_rows = phase_rows.reset_index(drop=True)
    partitions = [('all', phase_rows)] if mode != 'hard_cells' else _hard_cells(phase_rows, float(space_scale), float(time_scale))
    for partition, members in partitions:
        if len(members) < min(arguments.min_cluster_sizes):
            continue
        if mode == 'baseline':
            vector = members[[f'{name}_z' for name in FEATURE_NAMES]].to_numpy(float)
            norms = np.einsum('ij,ij->i', vector, vector)
            matrix = np.sqrt(np.maximum(norms[:, None] + norms[None, :] - 2 * vector @ vector.T, 0.0))
            np.fill_diagonal(matrix, 0.0)
        else:
            standardized = members[[f'{name}_z' for name in FEATURE_NAMES]].rename(
                columns={f'{name}_z': name for name in FEATURE_NAMES})
            gate_space = arguments.space_gate_km if mode == 'soft_gate' else float('inf')
            gate_time = arguments.time_gate_seconds if mode == 'soft_gate' else float('inf')
            matrix = _spatiotemporal_matrix(members, standardized, float(space_scale), float(time_scale), gate_space, gate_time)
        selected, candidates = _fit_candidates(matrix, members, arguments)
        for row in candidates:
            selections.append({'config': config_name, 'phase': phase, 'partition': partition,
                               'selected': row['min_cluster_size'] == selected.minimum_size
                               and row['min_samples'] == selected.minimum_samples, **row})
        for label, positions in selected.groups:
            group_id = f'{config_name}:{phase}:{partition}:{label}'
            for position in positions:
                member = members.iloc[int(position)]
                assignments.append({'config': config_name, 'group_id': group_id, 'phase': phase,
                                    'segment_id': member['segment_id'], 'flight_id': member['flight_id'],
                                    'membership_probability': float(selected.probabilities[int(position)])})
    return selections, assignments


def run_experiment(arguments: argparse.Namespace) -> dict[str, object]:
    if (min((*arguments.space_scales_km, *arguments.time_scales_seconds, arguments.space_gate_km,
            arguments.time_gate_seconds)) <= 0 or arguments.workers < 1):
        raise ValueError('空间/时间尺度与门控必须为正')
    output = arguments.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = arguments.segment_manifest.resolve()
    segments, values = _read_segment_store(manifest_path)
    geometry = read_table(
        manifest_path.parent / 'data', 'segments',
        columns={'segment_id', 'start_longitude_deg', 'start_latitude_deg', 'end_longitude_deg', 'end_latitude_deg'},
        dtype={'segment_id': 'string'},
    )
    segments = segments.merge(geometry, on='segment_id', how='left', validate='one_to_one')
    if segments[['start_longitude_deg', 'start_latitude_deg', 'end_longitude_deg', 'end_latitude_deg']].isna().any().any():
        raise ValueError('时空距离诊断需要完整的阶段起终经纬度')
    recognized = segments.loc[segments['phase'].ne('NA')].copy()
    normal = recognized.loc[recognized['split'].eq('train') & recognized['segment_label'].eq(0)].copy()
    origin = int(values['time_ns'].min())
    features = build_segment_features(recognized, values.loc[values['segment_id'].isin(recognized['segment_id'])], origin)
    scaler = fit_feature_scaler(features.loc[features['segment_id'].isin(normal['segment_id'].astype(str))], np.ones(len(FEATURE_NAMES)), origin)
    standardized = transform_features(features, scaler)
    feature_rows = recognized.merge(features, on='segment_id', validate='one_to_one').merge(
        standardized.add_suffix('_z').assign(segment_id=features['segment_id'].astype(str)), on='segment_id', validate='one_to_one')
    configurations: list[tuple[str, str, float | None, float | None]] = [('baseline_13d', 'baseline', None, None)]
    configurations.extend((f'st_R{space:g}km_T{time:g}s', 'soft_gate', space, time)
                          for space in arguments.space_scales_km for time in arguments.time_scales_seconds)
    configurations.extend((f'cell_R{space:g}km_T{time:g}s', 'hard_cells', space, time)
                          for space in arguments.space_scales_km for time in arguments.time_scales_seconds)
    normal_rows = feature_rows.loc[feature_rows['segment_id'].isin(normal['segment_id'].astype(str))].copy()
    selections: list[dict[str, object]] = []
    assignment_rows: list[dict[str, object]] = []
    phase_inputs = [(str(phase), part.copy()) for phase, part in normal_rows.groupby('phase', sort=True, observed=True)]
    task_count = len(configurations) * len(phase_inputs)
    with InterruptibleProcessPoolExecutor(max_workers=min(arguments.workers, task_count),
                                          mp_context=mp.get_context('spawn')) as pool:
        futures = [pool.submit(_run_phase_configuration, name, mode, space, time, phase, phase_rows, arguments)
                   for name, mode, space, time in configurations for phase, phase_rows in phase_inputs]
        for future in as_completed(futures):
            selection, assignment = future.result()
            selections.extend(selection)
            assignment_rows.extend(assignment)
    selection_frame = pd.DataFrame(selections)
    assignments = pd.DataFrame(assignment_rows)
    statistics = _group_statistics(assignments, feature_rows, values) if not assignments.empty else pd.DataFrame()
    selection_frame.to_csv(output / 'hdbscan_candidates.csv', index=False)
    assignments.to_csv(output / 'group_members.csv', index=False)
    statistics.to_csv(output / 'group_statistics.csv', index=False)
    summary = {
        'segment_manifest': str(arguments.segment_manifest.resolve()), 'configurations': [item[0] for item in configurations],
        'normal_training_segments': int(len(normal)), 'selected_rows': int(selection_frame['selected'].sum()),
        'groups': int(len(statistics)), 'members': int(len(assignments)),
        'notes': '每个 phase 独立拟合；时空版本以有限大距离阻断超出门控的边，仅用于诊断。',
    }
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (output / 'README.md').write_text(
        '# HDBSCAN 分组距离诊断\n\n'
        '`baseline_13d` 复现当前 13 维标准化距离。`st_R*km_T*s` 使用起终点 Haversine 空间距离、'
        '中心时间距离和较低权重的运动学差异，但仍可能出现近邻链。`cell_R*km_T*s` 先以中心位置和'
        '中心时间硬分区，再在格内 HDBSCAN；不同阶段从不放入同一矩阵。'
        '`group_statistics.csv` 记录每组时间跨度、经纬度/高度/速度范围及航向最小圆弧跨度。\n', encoding='utf-8')
    return summary


if __name__ == '__main__':
    print(json.dumps(run_experiment(_parser().parse_args()), ensure_ascii=False, indent=2))
