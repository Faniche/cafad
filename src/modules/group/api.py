"""api.py | 2026-11-08 | 训练正常片段 DBSCAN 分组及训练路由 | Codex."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from src.app.arguments import ProjectArgumentParser
from src.app.artifacts import read_manifest
from src.app.contracts import ModuleResult, RunContext, ValidationReport

from .distance import DistanceConfig, load_distance_config, sample_anchors, segment_distance


GROUP_COLUMNS = ('group_id', 'expert_id', 'phase', 'cluster_label', 'members', 'core_members', 'reference_flights')
CORE_COLUMNS = ('group_id', 'segment_id', 'flight_id', 'phase')
REFERENCE_COLUMNS = ('group_id', 'segment_id', 'flight_id', 'phase', 'ordinal')
ASSIGNMENT_COLUMNS = ('segment_id', 'flight_id', 'phase', 'segment_label', 'group_id', 'expert_id', 'distance', 'route_mode')
EXCLUSION_COLUMNS = ('segment_id', 'flight_id', 'phase', 'segment_label', 'reason')


def build_parser() -> argparse.ArgumentParser:
    """定义训练片段建组参数。"""
    parser = ProjectArgumentParser(prog='main.py -m group', description='仅以训练正常阶段片段拟合 DBSCAN 正常参考组。')
    parser.add_argument('--input', required=True, type=Path, help='segment 模块已发布的 manifest.json')
    parser.add_argument('--distance-config', required=True, type=Path, help='schema_version=1 的片段距离 JSON/YAML')
    parser.add_argument('--anchors', type=int, default=8, help='每片段实际锚点的最大数量')
    parser.add_argument('--eps', type=float, default=1.0,
                        help='预计算片段距离 DBSCAN 邻域半径（启动默认 1.0，须用验证集调参）')
    parser.add_argument('--min-samples', type=int, default=5,
                        help='DBSCAN 每簇最小片段邻域数量（含自身；启动默认 5，须用验证集调参）')
    parser.add_argument('--min-reference-flights', type=int, default=2, help='有效组同源去重后的最小参考航班数')
    parser.add_argument('--distance-block-size', type=int, default=256, help='构造完整距离矩阵的行块大小')
    return parser


def _project_path(context: RunContext, value: Path) -> Path:
    return value if value.is_absolute() else context.project_root / value


def _read_segment_store(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取已验收 S0，按稳定 record_id 合并六维观测。"""
    manifest = read_manifest(path, expected_module='segment')
    data_dir = path.parent / 'data'
    required_segments = {'segment_id', 'flight_id', 'phase', 'ordinal', 'segment_label', 'split'}
    required_points = {'segment_id', 'record_id', 'flight_id', 'time_ns'}
    required_observations = {'record_id', 'time_ns', 'longitude_deg', 'latitude_deg', 'heading_deg'}
    segments = pd.read_csv(data_dir / 'segments.csv', usecols=list(required_segments),
                           dtype={'segment_id': 'string', 'flight_id': 'string'})
    points = pd.read_csv(data_dir / 'point_segment_map.csv', usecols=list(required_points),
                         dtype={'segment_id': 'string', 'record_id': 'string', 'flight_id': 'string'})
    observations = pd.read_csv(data_dir / 'input_observations.csv', usecols=list(required_observations),
                               dtype={'record_id': 'string'})
    if not required_segments.issubset(segments.columns) or not required_points.issubset(points.columns) or not required_observations.issubset(observations.columns):
        raise ValueError('segment 产物缺少分组所需的片段、映射或观测字段')
    values = points.merge(observations[['record_id', 'longitude_deg', 'latitude_deg', 'heading_deg']], on='record_id', how='left', validate='one_to_one')
    if values[['longitude_deg', 'latitude_deg', 'heading_deg']].isna().any().any() or values['record_id'].duplicated().any():
        raise ValueError('片段记录无法与最终观测一一对应')
    if manifest.get('dataset_id') is None or segments['segment_id'].duplicated().any():
        raise ValueError('segment 清单或片段 ID 无效')
    return segments, values


def _anchors_for_segments(segments: pd.DataFrame, values: pd.DataFrame, maximum: int) -> dict[str, np.ndarray]:
    """按已保存阶段片段建立锚点；仅观测值参与，不读取干净副本。"""
    anchors: dict[str, np.ndarray] = {}
    for segment_id, rows in values.groupby('segment_id', sort=False, observed=True):
        ordered = rows.sort_values('time_ns', kind='stable')
        anchors[str(segment_id)] = sample_anchors(ordered['time_ns'].to_numpy(), ordered['longitude_deg'].to_numpy(),
                                                  ordered['latitude_deg'].to_numpy(), ordered['heading_deg'].to_numpy(), maximum)
    if set(anchors) != set(segments['segment_id'].astype(str)):
        raise ValueError('存在没有有效观测锚点的阶段片段')
    return anchors


def _distance_matrix(segment_ids: list[str], anchors: dict[str, np.ndarray], config: DistanceConfig, block_size: int) -> np.ndarray:
    """分块构造对称 float64 完整矩阵；分块仅限制临时计算，不隐藏最终矩阵成本。"""
    count = len(segment_ids)
    matrix = np.zeros((count, count), dtype=np.float64)
    for start in range(0, count, block_size):
        for left in range(start, min(start + block_size, count)):
            for right in range(left + 1, count):
                value = segment_distance(anchors[segment_ids[left]], anchors[segment_ids[right]], config)
                matrix[left, right] = matrix[right, left] = value
    return matrix


def _fit_groups(normal: pd.DataFrame, anchors: dict[str, np.ndarray], args: argparse.Namespace,
                config: DistanceConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str], pd.DataFrame]:
    """按阶段 DBSCAN，保留正常参考去重后仍具备多个航班的组。"""
    group_rows: list[dict[str, object]] = []
    core_rows: list[dict[str, object]] = []
    reference_rows: list[dict[str, object]] = []
    assignments: dict[str, str] = {}
    exclusions: list[dict[str, object]] = []
    provisional: list[tuple[str, str, pd.DataFrame, np.ndarray, np.ndarray]] = []
    for phase, phase_rows in normal.groupby('phase', sort=True, observed=True):
        ids = phase_rows['segment_id'].astype(str).tolist()
        if len(ids) < args.min_samples:
            exclusions.extend(_exclusion_rows(phase_rows, 'dbscan_noise_or_insufficient_phase'))
            continue
        matrix = _distance_matrix(ids, anchors, config, args.distance_block_size)
        fitted = DBSCAN(eps=args.eps, min_samples=args.min_samples, metric='precomputed', algorithm='brute', n_jobs=1).fit(matrix)
        for label in sorted(set(int(value) for value in fitted.labels_ if value >= 0)):
            members = phase_rows.iloc[np.flatnonzero(fitted.labels_ == label)].copy()
            core_positions = fitted.core_sample_indices_[fitted.labels_[fitted.core_sample_indices_] == label]
            core_ids = np.asarray(ids, dtype=object)[core_positions]
            deduplicated = (members.sort_values(['flight_id', 'ordinal', 'segment_id'], kind='stable')
                             .drop_duplicates('flight_id', keep='first'))
            if deduplicated['flight_id'].nunique() < args.min_reference_flights:
                exclusions.extend(_exclusion_rows(members, 'deleted_group_insufficient_reference_flights'))
                continue
            provisional.append((str(phase), str(label), members, core_ids, deduplicated['segment_id'].astype(str).to_numpy()))
        noise = phase_rows.iloc[np.flatnonzero(fitted.labels_ < 0)]
        exclusions.extend(_exclusion_rows(noise, 'dbscan_noise'))
    for expert_number, (phase, label, members, core_ids, reference_ids) in enumerate(provisional):
        group_id = f'g{expert_number:04d}'
        group_rows.append({'group_id': group_id, 'expert_id': expert_number, 'phase': phase, 'cluster_label': int(label),
                           'members': len(members), 'core_members': len(core_ids), 'reference_flights': len(reference_ids)})
        for row in members.itertuples(index=False):
            assignments[str(row.segment_id)] = group_id
        core_set = set(str(value) for value in core_ids)
        for row in members.itertuples(index=False):
            if str(row.segment_id) in core_set:
                core_rows.append({'group_id': group_id, 'segment_id': str(row.segment_id), 'flight_id': str(row.flight_id), 'phase': phase})
        for reference_id in reference_ids:
            row = members.loc[members['segment_id'].astype(str).eq(reference_id)].iloc[0]
            reference_rows.append({'group_id': group_id, 'segment_id': reference_id, 'flight_id': str(row.flight_id),
                                   'phase': phase, 'ordinal': int(row.ordinal)})
    normal_ids = set(normal['segment_id'].astype(str))
    kept_ids = set(assignments)
    for row in normal.loc[normal['segment_id'].astype(str).isin(normal_ids - kept_ids)].itertuples(index=False):
        if not any(item['segment_id'] == str(row.segment_id) for item in exclusions):
            exclusions.append({'segment_id': str(row.segment_id), 'flight_id': str(row.flight_id), 'phase': str(row.phase),
                               'segment_label': int(row.segment_label), 'reason': 'deleted_group'})
    return (pd.DataFrame(group_rows, columns=GROUP_COLUMNS), pd.DataFrame(core_rows, columns=CORE_COLUMNS),
            pd.DataFrame(reference_rows, columns=REFERENCE_COLUMNS), assignments, pd.DataFrame(exclusions, columns=EXCLUSION_COLUMNS))


def _exclusion_rows(rows: pd.DataFrame, reason: str) -> list[dict[str, object]]:
    return [{'segment_id': str(row.segment_id), 'flight_id': str(row.flight_id), 'phase': str(row.phase),
             'segment_label': int(row.segment_label), 'reason': reason} for row in rows.itertuples(index=False)]


def _route_training_targets(segments: pd.DataFrame, core: pd.DataFrame, anchors: dict[str, np.ndarray],
                            config: DistanceConfig, normal_assignments: dict[str, str]) -> pd.DataFrame:
    """正常沿用唯一簇；异常用注入后的观测锚点路由到最近有效组。"""
    cores_by_group = {group_id: rows['segment_id'].astype(str).tolist()
                      for group_id, rows in core.groupby('group_id', sort=False, observed=True)}
    phase_by_group = {str(rows['group_id'].iloc[0]): str(rows['phase'].iloc[0])
                      for _, rows in core.groupby('group_id', sort=False, observed=True)}
    assignments: list[dict[str, object]] = []
    for row in segments.loc[segments['split'].eq('train')].itertuples(index=False):
        segment_id = str(row.segment_id)
        if int(row.segment_label) == 0:
            group_id = normal_assignments.get(segment_id)
            if group_id is None:
                continue
            assignments.append(_assignment_row(row, group_id, 0.0, 'normal_cluster'))
            continue
        same_phase = [group_id for group_id, phase in phase_by_group.items() if phase == str(row.phase)]
        candidates = same_phase or list(cores_by_group)
        if not candidates:
            raise ValueError('没有有效正常组，无法路由异常训练目标')
        distances = {group_id: min(segment_distance(anchors[segment_id], anchors[core_id], config)
                                   for core_id in cores_by_group[group_id]) for group_id in candidates}
        group_id = min(distances, key=lambda item: (distances[item], item))
        assignments.append(_assignment_row(row, group_id, distances[group_id],
                                           'abnormal_nearest_same_phase' if same_phase else 'abnormal_nearest_cross_phase'))
    return pd.DataFrame(assignments, columns=ASSIGNMENT_COLUMNS)


def _route_validation_targets(segments: pd.DataFrame, core: pd.DataFrame, anchors: dict[str, np.ndarray],
                              config: DistanceConfig) -> pd.DataFrame:
    """验证片段不参与 DBSCAN，只按同一冻结检索规则得到单专家检查点评价归属。"""
    cores_by_group = {group_id: rows['segment_id'].astype(str).tolist()
                      for group_id, rows in core.groupby('group_id', sort=False, observed=True)}
    phase_by_group = {str(rows['group_id'].iloc[0]): str(rows['phase'].iloc[0])
                      for _, rows in core.groupby('group_id', sort=False, observed=True)}
    rows: list[dict[str, object]] = []
    for row in segments.loc[segments['split'].eq('val')].itertuples(index=False):
        same_phase = [group_id for group_id, phase in phase_by_group.items() if phase == str(row.phase)]
        candidates = same_phase or list(cores_by_group)
        distances = {group_id: min(segment_distance(anchors[str(row.segment_id)], anchors[core_id], config)
                                   for core_id in cores_by_group[group_id]) for group_id in candidates}
        group_id = min(distances, key=lambda item: (distances[item], item))
        rows.append(_assignment_row(row, group_id, distances[group_id],
                                    'validation_nearest_same_phase' if same_phase else 'validation_nearest_cross_phase'))
    return pd.DataFrame(rows, columns=ASSIGNMENT_COLUMNS)


def _assignment_row(row: object, group_id: str, distance: float, mode: str) -> dict[str, object]:
    return {'segment_id': str(row.segment_id), 'flight_id': str(row.flight_id), 'phase': str(row.phase),
            'segment_label': int(row.segment_label), 'group_id': group_id, 'expert_id': int(group_id[1:]),
            'distance': float(distance), 'route_mode': mode}


def run(context: RunContext, args: argparse.Namespace) -> ModuleResult:
    """以训练正常片段固定组结构，并为所有训练目标生成唯一专家归属。"""
    if args.anchors < 1 or args.eps <= 0 or args.min_samples < 1 or args.min_reference_flights < 2 or args.distance_block_size < 1:
        raise ValueError('anchors、eps、min-samples、min-reference-flights 和 distance-block-size 必须为有效正值；最小参考航班数至少为 2')
    input_manifest = _project_path(context, args.input).resolve()
    distance_path = _project_path(context, args.distance_config).resolve()
    config = load_distance_config(distance_path)
    segments, values = _read_segment_store(input_manifest)
    recognized = segments.loc[~segments['phase'].eq('NA')].copy()
    normal = recognized.loc[recognized['split'].eq('train') & recognized['segment_label'].eq(0)].copy()
    if normal.empty:
        raise ValueError('没有可用于建立正常训练分组的非 NA 片段')
    context.logger.debug('分组输入读取完成：segments=%d；observations=%d；recognized=%d；normal_train=%d；phases=%s',
                         len(segments), len(values), len(recognized), len(normal),
                         normal.groupby('phase', observed=True).size().to_dict())
    anchors = _anchors_for_segments(recognized, values.loc[values['segment_id'].isin(recognized['segment_id'])], args.anchors)
    context.logger.debug('已建立片段锚点：segments=%d；每段最多 anchors=%d；distance_config=%s',
                         len(anchors), args.anchors, distance_path)
    context.logger.info('开始对训练集正常片段进行 DBSCAN 建组：normal=%d；recognized=%d；anchors≤%d；eps=%g；min_samples=%d',
                        len(normal), len(recognized), args.anchors, args.eps, args.min_samples)
    groups, core, references, normal_assignments, exclusions = _fit_groups(normal, anchors, args, config)
    context.logger.debug('DBSCAN 建组原始结果：groups=%d；core=%d；references=%d；excluded=%d',
                         len(groups), len(core), len(references), len(exclusions))
    if groups.empty or core.empty or references.empty:
        raise ValueError('DBSCAN 后没有满足最小正常参考航班数的有效组')
    assignments = _route_training_targets(recognized, core, anchors, config, normal_assignments)
    if assignments.empty or assignments['segment_id'].duplicated().any() or assignments['group_id'].isna().any():
        raise ValueError('训练目标唯一归属为空、重复或不完整')
    validation_assignments = _route_validation_targets(recognized, core, anchors, config)
    context.logger.debug('冻结组路由完成：train_targets=%d；validation_targets=%d；route_modes=%s',
                         len(assignments), len(validation_assignments), assignments['route_mode'].value_counts().to_dict())
    data_dir = context.staging_dir / 'data'
    data_dir.mkdir()
    groups.to_csv(data_dir / 'groups.csv', index=False, encoding='utf-8')
    core.to_csv(data_dir / 'core_segments.csv', index=False, encoding='utf-8')
    references.to_csv(data_dir / 'references.csv', index=False, encoding='utf-8')
    assignments.to_csv(data_dir / 'train_assignments.csv', index=False, encoding='utf-8')
    validation_assignments.to_csv(data_dir / 'validation_assignments.csv', index=False, encoding='utf-8')
    exclusions.to_csv(data_dir / 'exclusions.csv', index=False, encoding='utf-8')
    (context.staging_dir / 'distance_config.yaml').write_bytes(distance_path.read_bytes())
    summary = {'input_manifest': str(input_manifest), 'distance_config': str(distance_path), 'normal_training_segments': int(len(normal)),
               'recognized_segments': int(len(recognized)), 'effective_groups': int(len(groups)), 'core_segments': int(len(core)),
               'reference_segments': int(len(references)), 'train_targets': int(len(assignments)),
               'abnormal_train_targets': int(assignments['segment_label'].sum()), 'validation_targets': int(len(validation_assignments)),
               'excluded_normal_segments': int(len(exclusions))}
    context.logger.info('分组完成：有效组=%d；参考=%d；训练目标=%d（异常=%d）；排除正常=%d', summary['effective_groups'],
                        summary['reference_segments'], summary['train_targets'], summary['abnormal_train_targets'], summary['excluded_normal_segments'])
    return ModuleResult(summary=summary, dependencies=(input_manifest,))


def validate_outputs(context: RunContext, args: argparse.Namespace, result: ModuleResult) -> ValidationReport:
    """重新读取 S1，核验参考全正常、同源去重和训练唯一归属。"""
    data_dir = context.staging_dir / 'data'
    groups = pd.read_csv(data_dir / 'groups.csv', dtype={'group_id': 'string'})
    core = pd.read_csv(data_dir / 'core_segments.csv', dtype={'group_id': 'string', 'segment_id': 'string', 'flight_id': 'string'})
    references = pd.read_csv(data_dir / 'references.csv', dtype={'group_id': 'string', 'segment_id': 'string', 'flight_id': 'string'})
    assignments = pd.read_csv(data_dir / 'train_assignments.csv', dtype={'segment_id': 'string', 'group_id': 'string'})
    validation = pd.read_csv(data_dir / 'validation_assignments.csv', dtype={'segment_id': 'string', 'group_id': 'string'})
    exclusions = pd.read_csv(data_dir / 'exclusions.csv', dtype={'segment_id': 'string'})
    if groups.empty or groups['group_id'].duplicated().any() or core.empty or references.empty:
        raise ValueError('有效组、核心片段或参考片段为空')
    if not set(core['group_id']).issubset(set(groups['group_id'])) or not set(references['group_id']).issubset(set(groups['group_id'])):
        raise ValueError('核心或参考片段指向不存在的组')
    if references.duplicated(['group_id', 'flight_id']).any() or references.groupby('group_id')['flight_id'].nunique().lt(args.min_reference_flights).any():
        raise ValueError('正常参考没有按同源航班去重或数量不足')
    if assignments.empty or assignments['segment_id'].duplicated().any() or not set(assignments['group_id']).issubset(set(groups['group_id'])):
        raise ValueError('训练目标没有唯一有效专家归属')
    if validation['segment_id'].duplicated().any() or not set(validation['group_id']).issubset(set(groups['group_id'])):
        raise ValueError('验证片段没有唯一有效专家归属')
    if set(assignments.loc[assignments['segment_label'].eq(0), 'segment_id']) & set(exclusions['segment_id']):
        raise ValueError('被排除正常片段不能同时成为训练目标')
    return ValidationReport(checks={'groups': int(len(groups)), 'core_segments': int(len(core)), 'references': int(len(references)),
                                    'train_targets': int(len(assignments)), 'normal_reference_deduplicated': True,
                                    'validation_targets': int(len(validation)), 'unique_training_assignment': True})
