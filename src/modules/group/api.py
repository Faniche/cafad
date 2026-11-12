"""api.py | 2026-11-09 | 训练正常片段 HDBSCAN 分组及训练路由 | Codex."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN

from src.app.arguments import ProjectArgumentParser
from src.app.artifacts import read_manifest, write_json
from src.app.contracts import ModuleResult, RunContext, ValidationReport
from src.app.tables import read_table, write_table

from .features import (FEATURE_NAMES, FEATURE_WEIGHT_ARGUMENTS, build_segment_features, centroids_by_group,
                       fit_feature_scaler, transform_features, weighted_distance_matrix, weights_from_arguments)


GROUP_COLUMNS = ('group_id', 'expert_id', 'phase', 'cluster_label', 'members', 'routing_members',
                 'reference_flights', 'min_cluster_size', 'min_samples', 'mean_membership_probability')
ROUTING_COLUMNS = ('group_id', 'segment_id', 'flight_id', 'phase', 'membership_probability')
REFERENCE_COLUMNS = ('group_id', 'segment_id', 'flight_id', 'phase', 'ordinal')
ASSIGNMENT_COLUMNS = ('segment_id', 'flight_id', 'phase', 'segment_label', 'group_id', 'expert_id', 'distance', 'route_mode')
EXCLUSION_COLUMNS = ('segment_id', 'flight_id', 'phase', 'segment_label', 'reason')
SELECTION_COLUMNS = ('phase', 'min_cluster_size', 'min_samples', 'clusters_before_reference_filter',
                     'effective_groups', 'covered_segments', 'normal_segments', 'coverage', 'largest_group_fraction',
                     'mean_membership_probability', 'minimum_effective_groups', 'selected')


@dataclass(frozen=True)
class HdbscanCandidate:
    """保存一个阶段中一次 HDBSCAN 拟合及参考约束后的可比较结果。"""

    phase: str
    min_cluster_size: int
    min_samples: int
    labels: np.ndarray
    probabilities: np.ndarray
    effective_clusters: tuple[tuple[int, np.ndarray, np.ndarray], ...]
    clusters_before_reference_filter: int
    coverage: float
    largest_group_fraction: float
    mean_membership_probability: float


def build_parser() -> argparse.ArgumentParser:
    """定义训练正常片段 HDBSCAN 建组与自动候选选择参数。"""
    parser = ProjectArgumentParser(prog='main.py -m group', description='仅以训练正常阶段片段拟合 HDBSCAN 正常参考组。')
    parser.add_argument('--input', required=True, type=Path, help='segment 模块已发布的 manifest.json')
    for _, destination, help_text in FEATURE_WEIGHT_ARGUMENTS:
        parser.add_argument('--' + destination.replace('_', '-'), dest=destination, type=float, default=1.0,
                            help=f'{help_text}（默认 1.0；0 表示不参与距离）')
    parser.add_argument('--min-cluster-sizes', nargs='+', type=int, default=(4, 8, 12, 16, 24, 32, 48, 64),
                        help='自动搜索的 HDBSCAN 最小稳定簇片段数（默认：4 8 12 16 24 32 48 64）')
    parser.add_argument('--min-samples-values', nargs='+', type=int, default=(1, 2, 4, 8, 12, 16),
                        help='自动搜索的 HDBSCAN 局部密度保守度（默认：1 2 4 8 12 16）')
    parser.add_argument('--min-group-members', type=int, default=12,
                        help='有效组至少保留的正常训练片段数（默认 12，保证专家具备基本训练样本）')
    parser.add_argument('--max-group-fraction', type=float, default=1.0,
                        help='候选优先条件：单个有效组最多覆盖本阶段正常片段的比例（默认 1.0，即不强制拆分稳定大组）')
    parser.add_argument('--min-reference-flights', type=int, default=2, help='有效组同源去重后的最小参考航班数')
    parser.add_argument('--distance-block-size', type=int, default=256, help='构造预计算加权欧氏距离矩阵的行块大小')
    return parser


def _project_path(context: RunContext, value: Path) -> Path:
    return value if value.is_absolute() else context.project_root / value


def _read_segment_store(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取已验收 S0，按稳定 record_id 合并六维观测。"""
    manifest = read_manifest(path, expected_module='segment')
    data_dir = path.parent / 'data'
    required_segments = {'segment_id', 'flight_id', 'phase', 'ordinal', 'segment_label', 'split'}
    required_points = {'segment_id', 'record_id', 'flight_id', 'time_ns'}
    required_observations = {'record_id', 'time_ns', 'longitude_deg', 'latitude_deg', 'geometric_altitude_m',
                             'barometric_altitude_m', 'heading_deg', 'speed_mps'}
    segments = read_table(data_dir, 'segments', columns=required_segments,
                          dtype={'segment_id': 'string', 'flight_id': 'string'})
    points = read_table(data_dir, 'point_segment_map', columns=required_points,
                        dtype={'segment_id': 'string', 'record_id': 'string', 'flight_id': 'string'})
    observations = read_table(data_dir, 'input_observations', columns=required_observations,
                              dtype={'record_id': 'string'})
    if not required_segments.issubset(segments.columns) or not required_points.issubset(points.columns) or not required_observations.issubset(observations.columns):
        raise ValueError('segment 产物缺少分组所需的片段、映射或观测字段')
    values = points.merge(observations[['record_id', 'longitude_deg', 'latitude_deg', 'geometric_altitude_m',
                                        'barometric_altitude_m', 'heading_deg', 'speed_mps']], on='record_id',
                          how='left', validate='one_to_one')
    if values[['longitude_deg', 'latitude_deg', 'geometric_altitude_m', 'barometric_altitude_m', 'heading_deg', 'speed_mps']].isna().any().any() or values['record_id'].duplicated().any():
        raise ValueError('片段记录无法与最终观测一一对应')
    if manifest.get('dataset_id') is None or segments['segment_id'].duplicated().any():
        raise ValueError('segment 清单或片段 ID 无效')
    return segments, values


def _candidate_pairs(args: argparse.Namespace) -> tuple[tuple[int, int], ...]:
    """返回去重且稳定排序的有限 HDBSCAN 候选网格，避免密度门槛高于最小稳定簇规模。"""
    return tuple((size, samples) for size in sorted(set(args.min_cluster_sizes))
           for samples in sorted(set(args.min_samples_values)) if samples <= size)


def _fit_candidate(phase: str, phase_rows: pd.DataFrame, matrix: np.ndarray, min_cluster_size: int,
                   min_samples: int, min_group_members: int, min_reference_flights: int) -> HdbscanCandidate:
    """拟合一次 HDBSCAN，并以同源去重后的参考数筛选真正可用的簇。"""
    fitted = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric='precomputed',
                     algorithm='brute', n_jobs=1, cluster_selection_method='eom', allow_single_cluster=True,
                     copy=False).fit(matrix.copy())
    labels = np.asarray(fitted.labels_, dtype=np.int64)
    probabilities = np.asarray(fitted.probabilities_, dtype=np.float64)
    effective: list[tuple[int, np.ndarray, np.ndarray]] = []
    cluster_labels = sorted(int(label) for label in set(labels) if label >= 0)
    for label in cluster_labels:
        positions = np.flatnonzero(labels == label)
        members = phase_rows.iloc[positions]
        references = (members.sort_values(['flight_id', 'ordinal', 'segment_id'], kind='stable')
                      .drop_duplicates('flight_id', keep='first'))
        if len(members) >= min_group_members and references['flight_id'].nunique() >= min_reference_flights:
            effective.append((label, positions, references['segment_id'].astype(str).to_numpy()))
    covered = int(sum(len(item[1]) for item in effective))
    group_sizes = [len(item[1]) for item in effective]
    member_positions = np.concatenate([item[1] for item in effective]) if effective else np.array([], dtype=np.int64)
    member_probabilities = probabilities[member_positions]
    return HdbscanCandidate(phase=phase, min_cluster_size=min_cluster_size, min_samples=min_samples,
                            labels=labels, probabilities=probabilities, effective_clusters=tuple(effective),
                            clusters_before_reference_filter=len(cluster_labels), coverage=covered / len(phase_rows),
                            largest_group_fraction=(max(group_sizes) / len(phase_rows) if group_sizes else 0.0),
                            mean_membership_probability=(float(member_probabilities.mean()) if len(member_probabilities) else 0.0))


def _select_candidate(candidates: list[HdbscanCandidate], max_group_fraction: float) -> HdbscanCandidate:
    """覆盖率优先选择可用候选；优先回避单一超大组，所有候选都超限时仍保留最佳可审计结果。"""
    if not candidates:
        raise ValueError('当前阶段没有可运行的 HDBSCAN 候选配置')
    return max(candidates, key=lambda item: (item.largest_group_fraction <= max_group_fraction,
                                              item.coverage, item.mean_membership_probability,
                                              item.min_cluster_size, item.min_samples, -item.largest_group_fraction))


def _select_candidates_with_group_floor(
    candidates_by_phase: list[tuple[str, list[HdbscanCandidate]]],
    max_group_fraction: float,
    minimum_effective_groups: int,
) -> list[HdbscanCandidate]:
    """联合选择各阶段候选，使有效组总数不低于最长正常训练轨迹的片段数。"""
    selected = [_select_candidate(candidates, max_group_fraction) for _, candidates in candidates_by_phase]
    while sum(len(candidate.effective_clusters) for candidate in selected) < minimum_effective_groups:
        replacements: list[tuple[bool, float, int, float, int, int, int, int, int]] = []
        for phase_index, (_, candidates) in enumerate(candidates_by_phase):
            current = selected[phase_index]
            current_groups = len(current.effective_clusters)
            for candidate_index, candidate in enumerate(candidates):
                gain = len(candidate.effective_clusters) - current_groups
                if gain <= 0:
                    continue
                coverage_loss_per_group = (current.coverage - candidate.coverage) / gain
                replacements.append((candidate.largest_group_fraction <= max_group_fraction,
                                     -coverage_loss_per_group, gain, candidate.mean_membership_probability,
                                     candidate.min_cluster_size, candidate.min_samples, -phase_index,
                                     phase_index, candidate_index))
        if not replacements:
            selected_groups = sum(len(candidate.effective_clusters) for candidate in selected)
            raise ValueError(
                f'HDBSCAN 候选最多只能形成 {selected_groups} 个有效组，低于最长正常训练轨迹所需的 '
                f'{minimum_effective_groups} 个；请降低 --min-cluster-sizes 或 --min-group-members，或调整片段距离参数'
            )
        _, _, _, _, _, _, _, phase_index, candidate_index = max(replacements, key=lambda item: item[:7])
        selected[phase_index] = candidates_by_phase[phase_index][1][candidate_index]
    return selected


def _selection_row(candidate: HdbscanCandidate, normal_count: int, minimum_effective_groups: int,
                   selected: bool) -> dict[str, object]:
    return {'phase': candidate.phase, 'min_cluster_size': candidate.min_cluster_size, 'min_samples': candidate.min_samples,
            'clusters_before_reference_filter': candidate.clusters_before_reference_filter,
            'effective_groups': len(candidate.effective_clusters), 'covered_segments': round(candidate.coverage * normal_count),
            'normal_segments': normal_count, 'coverage': candidate.coverage,
            'largest_group_fraction': candidate.largest_group_fraction,
            'mean_membership_probability': candidate.mean_membership_probability,
            'minimum_effective_groups': minimum_effective_groups, 'selected': selected}


def _fit_groups(normal: pd.DataFrame, vectors_by_segment: pd.DataFrame, args: argparse.Namespace,
                logger: logging.Logger, minimum_effective_groups: int
                ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str], pd.DataFrame, pd.DataFrame]:
    """按阶段自动选择覆盖率优先的 HDBSCAN 配置，并保留合格正常参考组。"""
    group_rows: list[dict[str, object]] = []
    routing_rows: list[dict[str, object]] = []
    reference_rows: list[dict[str, object]] = []
    assignments: dict[str, str] = {}
    exclusions: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    provisional: list[tuple[HdbscanCandidate, int, np.ndarray, np.ndarray]] = []
    phase_candidates: list[tuple[str, pd.DataFrame, list[HdbscanCandidate]]] = []
    for phase, phase_rows in normal.groupby('phase', sort=True, observed=True):
        ids = phase_rows['segment_id'].astype(str).tolist()
        candidates: list[HdbscanCandidate] = []
        if len(ids) >= min(args.min_cluster_sizes):
            logger.debug('开始构造加权标准化特征距离矩阵：phase=%s；normal_segments=%d；features=%s',
                         phase, len(ids), FEATURE_NAMES)
            matrix = weighted_distance_matrix(
                vectors_by_segment.loc[ids, list(FEATURE_NAMES)].to_numpy(dtype=np.float64), args.distance_block_size,
            )
            for min_cluster_size, min_samples in _candidate_pairs(args):
                if min_cluster_size <= len(ids):
                    started = time.monotonic()
                    logger.debug('开始 HDBSCAN 候选：phase=%s；min_cluster_size=%d；min_samples=%d',
                                 phase, min_cluster_size, min_samples)
                    candidates.append(_fit_candidate(str(phase), phase_rows, matrix, min_cluster_size, min_samples,
                                                     args.min_group_members, args.min_reference_flights))
                    logger.debug('完成 HDBSCAN 候选：phase=%s；min_cluster_size=%d；min_samples=%d；耗时=%.2fs；coverage=%.4f',
                                 phase, min_cluster_size, min_samples, time.monotonic() - started, candidates[-1].coverage)
        if not candidates:
            exclusions.extend(_exclusion_rows(phase_rows, 'hdbscan_insufficient_phase'))
            continue
        phase_candidates.append((str(phase), phase_rows, candidates))
    selected_candidates = _select_candidates_with_group_floor(
        [(phase, candidates) for phase, _, candidates in phase_candidates], args.max_group_fraction,
        minimum_effective_groups,
    )
    for (phase, phase_rows, candidates), selected in zip(phase_candidates, selected_candidates, strict=True):
        logger.debug('阶段 HDBSCAN 选择：phase=%s；min_cluster_size=%d；min_samples=%d；coverage=%.4f；groups=%d；largest_fraction=%.4f；总组数下限=%d',
                     phase, selected.min_cluster_size, selected.min_samples, selected.coverage,
                     len(selected.effective_clusters), selected.largest_group_fraction, minimum_effective_groups)
        selection_rows.extend(_selection_row(candidate, len(phase_rows), minimum_effective_groups, candidate is selected)
                              for candidate in candidates)
        provisional.extend((selected, label, positions, reference_ids)
                           for label, positions, reference_ids in selected.effective_clusters)
        selected_positions = set(np.concatenate([item[1] for item in selected.effective_clusters]).tolist()) if selected.effective_clusters else set()
        excluded_rows = phase_rows.iloc[sorted(set(range(len(phase_rows))) - selected_positions)]
        exclusions.extend(_exclusion_rows(excluded_rows, 'hdbscan_noise_or_deleted_group'))
    for expert_number, (candidate, label, positions, reference_ids) in enumerate(provisional):
        members = normal.loc[normal['phase'].eq(candidate.phase)].iloc[positions]
        group_id = f'g{expert_number:04d}'
        probabilities = candidate.probabilities[positions]
        group_rows.append({'group_id': group_id, 'expert_id': expert_number, 'phase': candidate.phase, 'cluster_label': label,
                           'members': len(members), 'routing_members': len(members), 'reference_flights': len(reference_ids),
                           'min_cluster_size': candidate.min_cluster_size, 'min_samples': candidate.min_samples,
                           'mean_membership_probability': float(probabilities.mean())})
        for position, row in enumerate(members.itertuples(index=False)):
            segment_id = str(row.segment_id)
            assignments[segment_id] = group_id
            routing_rows.append({'group_id': group_id, 'segment_id': segment_id, 'flight_id': str(row.flight_id),
                                 'phase': candidate.phase, 'membership_probability': float(probabilities[position])})
        for reference_id in reference_ids:
            row = members.loc[members['segment_id'].astype(str).eq(reference_id)].iloc[0]
            reference_rows.append({'group_id': group_id, 'segment_id': reference_id, 'flight_id': str(row.flight_id),
                                   'phase': candidate.phase, 'ordinal': int(row.ordinal)})
    normal_ids = set(normal['segment_id'].astype(str))
    kept_ids = set(assignments)
    existing_exclusions = {item['segment_id'] for item in exclusions}
    for row in normal.loc[normal['segment_id'].astype(str).isin(normal_ids - kept_ids)].itertuples(index=False):
        if str(row.segment_id) not in existing_exclusions:
            exclusions.append({'segment_id': str(row.segment_id), 'flight_id': str(row.flight_id), 'phase': str(row.phase),
                               'segment_label': int(row.segment_label), 'reason': 'hdbscan_deleted_group'})
    return (pd.DataFrame(group_rows, columns=GROUP_COLUMNS), pd.DataFrame(routing_rows, columns=ROUTING_COLUMNS),
            pd.DataFrame(reference_rows, columns=REFERENCE_COLUMNS), assignments,
            pd.DataFrame(exclusions, columns=EXCLUSION_COLUMNS), pd.DataFrame(selection_rows, columns=SELECTION_COLUMNS))


def _exclusion_rows(rows: pd.DataFrame, reason: str) -> list[dict[str, object]]:
    return [{'segment_id': str(row.segment_id), 'flight_id': str(row.flight_id), 'phase': str(row.phase),
             'segment_label': int(row.segment_label), 'reason': reason} for row in rows.itertuples(index=False)]


def _nearest_group(segment_id: str, phase: str, phase_by_group: dict[str, str],
                   centers: dict[str, np.ndarray], vectors_by_segment: pd.DataFrame) -> tuple[str, float, str]:
    """以冻结组中心路由一个片段；优先同阶段，否则退回全部正常组。"""
    same_phase = [group_id for group_id, group_phase in phase_by_group.items() if group_phase == phase]
    candidates = same_phase or list(centers)
    if not candidates:
        raise ValueError('没有有效正常组，无法路由片段')
    vector = vectors_by_segment.loc[segment_id, list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
    distances = {group_id: float(np.linalg.norm(vector - centers[group_id])) for group_id in candidates}
    group_id = min(distances, key=lambda item: (distances[item], item))
    return group_id, distances[group_id], 'same_phase' if same_phase else 'cross_phase'


def _route_training_targets(segments: pd.DataFrame, core: pd.DataFrame, vectors_by_segment: pd.DataFrame,
                            normal_assignments: dict[str, str]) -> pd.DataFrame:
    """正常沿用唯一簇；异常以注入后特征向量到冻结组中心的距离路由。"""
    phase_by_group = {str(rows['group_id'].iloc[0]): str(rows['phase'].iloc[0])
                      for _, rows in core.groupby('group_id', sort=False, observed=True)}
    centers = centroids_by_group(core, vectors_by_segment)
    assignments: list[dict[str, object]] = []
    for row in segments.loc[segments['split'].eq('train')].itertuples(index=False):
        segment_id = str(row.segment_id)
        if int(row.segment_label) == 0:
            group_id = normal_assignments.get(segment_id)
            if group_id is None:
                continue
            assignments.append(_assignment_row(row, group_id, 0.0, 'normal_cluster'))
            continue
        group_id, distance, scope = _nearest_group(segment_id, str(row.phase), phase_by_group, centers, vectors_by_segment)
        assignments.append(_assignment_row(row, group_id, distance, f'abnormal_nearest_{scope}'))
    return pd.DataFrame(assignments, columns=ASSIGNMENT_COLUMNS)


def _route_validation_targets(segments: pd.DataFrame, core: pd.DataFrame, vectors_by_segment: pd.DataFrame) -> pd.DataFrame:
    """验证片段不参与 HDBSCAN，只按同一冻结检索规则得到单专家检查点评价归属。"""
    phase_by_group = {str(rows['group_id'].iloc[0]): str(rows['phase'].iloc[0])
                      for _, rows in core.groupby('group_id', sort=False, observed=True)}
    centers = centroids_by_group(core, vectors_by_segment)
    rows: list[dict[str, object]] = []
    for row in segments.loc[segments['split'].eq('val')].itertuples(index=False):
        group_id, distance, scope = _nearest_group(str(row.segment_id), str(row.phase), phase_by_group, centers,
                                                   vectors_by_segment)
        rows.append(_assignment_row(row, group_id, distance, f'validation_nearest_{scope}'))
    return pd.DataFrame(rows, columns=ASSIGNMENT_COLUMNS)


def _assignment_row(row: object, group_id: str, distance: float, mode: str) -> dict[str, object]:
    return {'segment_id': str(row.segment_id), 'flight_id': str(row.flight_id), 'phase': str(row.phase),
            'segment_label': int(row.segment_label), 'group_id': group_id, 'expert_id': int(group_id[1:]),
            'distance': float(distance), 'route_mode': mode}


def run(context: RunContext, args: argparse.Namespace) -> ModuleResult:
    """以训练正常片段固定 HDBSCAN 组结构，并为所有训练目标生成唯一专家归属。"""
    if (not args.min_cluster_sizes or min(args.min_cluster_sizes) < 2 or
            not args.min_samples_values or min(args.min_samples_values) < 1 or args.min_group_members < 2 or
            not 0 < args.max_group_fraction <= 1 or
            args.min_reference_flights < 2 or args.distance_block_size < 1):
        raise ValueError('HDBSCAN 候选、min-group-members、max-group-fraction、min-reference-flights 和 distance-block-size 必须有效；最小成员及参考航班数至少为 2')
    input_manifest = _project_path(context, args.input).resolve()
    segments, values = _read_segment_store(input_manifest)
    recognized = segments.loc[~segments['phase'].eq('NA')].copy()
    normal = recognized.loc[recognized['split'].eq('train') & recognized['segment_label'].eq(0)].copy()
    if normal.empty:
        raise ValueError('没有可用于建立正常训练分组的非 NA 片段')
    segments_per_normal_flight = normal.groupby('flight_id', observed=True).size()
    minimum_effective_groups = int(segments_per_normal_flight.max())
    longest_normal_flight = str(segments_per_normal_flight.idxmax())
    context.logger.debug('分组输入读取完成：segments=%d；observations=%d；recognized=%d；normal_train=%d；phases=%s',
                         len(segments), len(values), len(recognized), len(normal),
                         normal.groupby('phase', observed=True).size().to_dict())
    time_origin_ns = int(values['time_ns'].min())
    features = build_segment_features(recognized, values.loc[values['segment_id'].isin(recognized['segment_id'])], time_origin_ns)
    weights = weights_from_arguments(args)
    scaler = fit_feature_scaler(features.loc[features['segment_id'].isin(normal['segment_id'].astype(str))], weights,
                                time_origin_ns)
    transformed = transform_features(features, scaler)
    transformed.insert(0, 'segment_id', features['segment_id'].astype(str))
    vectors_by_segment = transformed.set_index('segment_id', verify_integrity=True)
    candidates = _candidate_pairs(args)
    context.logger.info('开始对训练集正常片段进行 HDBSCAN 建组：normal=%d；recognized=%d；features=%d；candidates=%s；min_group_members=%d；max_group_fraction=%g；最长正常航班=%s；组数下限=%d',
                        len(normal), len(recognized), len(FEATURE_NAMES), candidates, args.min_group_members, args.max_group_fraction,
                        longest_normal_flight, minimum_effective_groups)
    groups, core, references, normal_assignments, exclusions, selection = _fit_groups(
        normal, vectors_by_segment, args, context.logger, minimum_effective_groups,
    )
    context.logger.debug('HDBSCAN 建组原始结果：groups=%d；routing_members=%d；references=%d；excluded=%d；selection=%s',
                         len(groups), len(core), len(references), len(exclusions), selection.loc[selection['selected']].to_dict('records'))
    if groups.empty or core.empty or references.empty:
        raise ValueError('HDBSCAN 后没有满足最小正常参考航班数的有效组')
    assignments = _route_training_targets(recognized, core, vectors_by_segment, normal_assignments)
    if assignments.empty or assignments['segment_id'].duplicated().any() or assignments['group_id'].isna().any():
        raise ValueError('训练目标唯一归属为空、重复或不完整')
    validation_assignments = _route_validation_targets(recognized, core, vectors_by_segment)
    context.logger.debug('冻结组路由完成：train_targets=%d；validation_targets=%d；route_modes=%s',
                         len(assignments), len(validation_assignments), assignments['route_mode'].value_counts().to_dict())
    data_dir = context.staging_dir / 'data'
    data_dir.mkdir()
    write_table(groups, data_dir, 'groups')
    write_table(core, data_dir, 'routing_segments')
    write_table(references, data_dir, 'references')
    write_table(assignments, data_dir, 'train_assignments')
    write_table(validation_assignments, data_dir, 'validation_assignments')
    write_table(exclusions, data_dir, 'exclusions')
    write_table(selection, data_dir, 'hdbscan_selection')
    write_table(features, data_dir, 'segment_features')
    write_json(data_dir / 'feature_scaler.json', scaler.as_dict())
    normal_coverage = len(normal_assignments) / len(normal)
    summary = {'input_manifest': str(input_manifest), 'clustering_algorithm': 'hdbscan',
               'feature_distance': 'weighted_euclidean_on_standardized_segment_features',
               'feature_names': list(FEATURE_NAMES), 'feature_weights': dict(zip(FEATURE_NAMES, weights.tolist(), strict=True)),
               'normal_training_segments': int(len(normal)), 'normal_training_coverage': normal_coverage,
               'recognized_segments': int(len(recognized)), 'effective_groups': int(len(groups)), 'routing_segments': int(len(core)),
               'minimum_effective_groups': minimum_effective_groups, 'longest_normal_training_flight': longest_normal_flight,
               'reference_segments': int(len(references)), 'train_targets': int(len(assignments)),
               'abnormal_train_targets': int(assignments['segment_label'].sum()), 'validation_targets': int(len(validation_assignments)),
               'excluded_normal_segments': int(len(exclusions))}
    context.logger.info('分组完成：有效组=%d（下限=%d）；参考=%d；正常覆盖率=%.2f%%；训练目标=%d（异常=%d）；排除正常=%d',
                        summary['effective_groups'], summary['minimum_effective_groups'], summary['reference_segments'], normal_coverage * 100,
                        summary['train_targets'], summary['abnormal_train_targets'], summary['excluded_normal_segments'])
    return ModuleResult(summary=summary, dependencies=(input_manifest,))


def validate_outputs(context: RunContext, args: argparse.Namespace, result: ModuleResult) -> ValidationReport:
    """重新读取 S1，核验 HDBSCAN 参考、候选选择和训练唯一归属。"""
    data_dir = context.staging_dir / 'data'
    groups = read_table(data_dir, 'groups', dtype={'group_id': 'string'})
    core = read_table(data_dir, 'routing_segments', dtype={'group_id': 'string', 'segment_id': 'string', 'flight_id': 'string'})
    references = read_table(data_dir, 'references', dtype={'group_id': 'string', 'segment_id': 'string', 'flight_id': 'string'})
    assignments = read_table(data_dir, 'train_assignments', dtype={'segment_id': 'string', 'group_id': 'string'})
    validation = read_table(data_dir, 'validation_assignments', dtype={'segment_id': 'string', 'group_id': 'string'})
    exclusions = read_table(data_dir, 'exclusions', dtype={'segment_id': 'string'})
    selection = read_table(data_dir, 'hdbscan_selection')
    if groups.empty or groups['group_id'].duplicated().any() or core.empty or references.empty:
        raise ValueError('有效组、路由片段或参考片段为空')
    minimum_effective_groups = int(result.summary['minimum_effective_groups'])
    if len(groups) < minimum_effective_groups:
        raise ValueError('有效组数低于最长正常训练轨迹的阶段片段数下限')
    if not set(core['group_id']).issubset(set(groups['group_id'])) or not set(references['group_id']).issubset(set(groups['group_id'])):
        raise ValueError('路由或参考片段指向不存在的组')
    if references.duplicated(['group_id', 'flight_id']).any() or references.groupby('group_id')['flight_id'].nunique().lt(args.min_reference_flights).any():
        raise ValueError('正常参考没有按同源航班去重或数量不足')
    if (selection.empty or selection.groupby('phase')['selected'].sum().ne(1).any()
            or not selection['coverage'].between(0, 1).all()
            or selection['minimum_effective_groups'].ne(minimum_effective_groups).any()):
        raise ValueError('HDBSCAN 候选选择记录缺失或无效')
    if assignments.empty or assignments['segment_id'].duplicated().any() or not set(assignments['group_id']).issubset(set(groups['group_id'])):
        raise ValueError('训练目标没有唯一有效专家归属')
    if validation['segment_id'].duplicated().any() or not set(validation['group_id']).issubset(set(groups['group_id'])):
        raise ValueError('验证片段没有唯一有效专家归属')
    if set(assignments.loc[assignments['segment_label'].eq(0), 'segment_id']) & set(exclusions['segment_id']):
        raise ValueError('被排除正常片段不能同时成为训练目标')
    return ValidationReport(checks={'groups': int(len(groups)), 'routing_segments': int(len(core)), 'references': int(len(references)),
                                    'hdbscan_selection_rows': int(len(selection)), 'minimum_effective_groups': minimum_effective_groups,
                                    'train_targets': int(len(assignments)),
                                    'normal_reference_deduplicated': True, 'validation_targets': int(len(validation)),
                                    'unique_training_assignment': True})
