"""api.py | 2026-11-15 | 硬时空单元的确定性专家分组与冻结路由 | Codex."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.app.arguments import ProjectArgumentParser
from src.app.artifacts import read_manifest
from src.app.contracts import ModuleResult, RunContext, ValidationReport
from src.app.tables import read_table, write_table


GROUP_COLUMNS = ('group_id', 'expert_id', 'cell_key', 'height_band', 'time_bucket', 'longitude_cell', 'latitude_cell',
                 'members', 'routing_members', 'reference_flights')
ROUTING_COLUMNS = ('group_id', 'segment_id', 'flight_id', 'cell_key')
REFERENCE_COLUMNS = ('group_id', 'segment_id', 'flight_id', 'cell_key', 'ordinal')
ASSIGNMENT_COLUMNS = ('segment_id', 'flight_id', 'cell_key', 'segment_label', 'group_id', 'expert_id', 'route_mode')
EXCLUSION_COLUMNS = ('segment_id', 'flight_id', 'cell_key', 'segment_label', 'reason')


def build_parser() -> argparse.ArgumentParser:
    """定义硬分组的最小正常参考约束，不包含 HDBSCAN 或距离超参数。"""
    parser = ProjectArgumentParser(prog='main.py -m group', description='以硬时空单元直接构造正常参考组和唯一专家路由。')
    parser.add_argument('--input', required=True, type=Path, help='segment 模块已发布的 manifest.json')
    parser.add_argument('--min-group-members', type=int, default=2, help='保留专家组所需的最少正常训练片段数（默认 2）')
    parser.add_argument('--min-reference-flights', type=int, default=2, help='保留专家组所需的最少不同正常训练航班数（默认 2）')
    return parser


def _project_path(context: RunContext, value: Path) -> Path:
    return value if value.is_absolute() else context.project_root / value


def read_segment_store_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取硬分段边界和点映射；保留此接口供训练、注入和检测追溯。"""
    required_segments = {'segment_id', 'flight_id', 'ordinal', 'cell_key', 'height_band', 'time_bucket',
                         'longitude_cell', 'latitude_cell', 'segment_label', 'split', 'injection_plan'}
    segments = read_table(data_dir, 'segments', dtype={'segment_id': 'string', 'flight_id': 'string'})
    mapping = read_table(data_dir, 'point_segment_map', dtype={'segment_id': 'string', 'record_id': 'string', 'flight_id': 'string'})
    if (not required_segments.issubset(segments.columns) or segments['segment_id'].duplicated().any()
            or mapping['record_id'].duplicated().any() or not set(mapping['segment_id']).issubset(set(segments['segment_id']))):
        raise ValueError('segment 产物缺少硬分组所需字段，或片段／记录 ID 不唯一')
    return segments, mapping


def _read_segment_store(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = read_manifest(path, expected_module='segment')
    if manifest.get('dataset_id') is None:
        raise ValueError('segment 清单数据集身份无效')
    return read_segment_store_data(path.parent / 'data')


def _references(normal: pd.DataFrame) -> pd.DataFrame:
    """同一专家中每个原始航班只留最早一个正常片段作为记忆参考。"""
    return (normal.sort_values(['cell_key', 'flight_id', 'ordinal', 'segment_id'], kind='stable')
            .drop_duplicates(['cell_key', 'flight_id'], keep='first'))


def _assign(rows: pd.DataFrame, groups: pd.DataFrame, route_mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按同一 cell_key 精确路由；未被有效正常组覆盖的片段显式排除。"""
    lookup = groups.set_index('cell_key')[['group_id', 'expert_id']]
    merged = rows.merge(lookup, on='cell_key', how='left', validate='many_to_one')
    assigned = merged.loc[merged['group_id'].notna(), ['segment_id', 'flight_id', 'cell_key', 'segment_label', 'group_id', 'expert_id']].copy()
    assigned['route_mode'] = route_mode
    excluded = merged.loc[merged['group_id'].isna(), ['segment_id', 'flight_id', 'cell_key', 'segment_label']].copy()
    excluded['reason'] = 'hard_cell_without_normal_reference'
    return assigned.loc[:, ASSIGNMENT_COLUMNS], excluded.loc[:, EXCLUSION_COLUMNS]


def run(context: RunContext, args: argparse.Namespace) -> ModuleResult:
    """用计划正常的训练片段建立硬组，并冻结 train／val 片段的精确单专家路由。"""
    if args.min_group_members < 1 or args.min_reference_flights < 1:
        raise ValueError('min-group-members 和 min-reference-flights 必须为正整数')
    input_manifest = _project_path(context, args.input).resolve()
    segments, _ = _read_segment_store(input_manifest)
    normal = segments.loc[segments['split'].eq('train') & segments['injection_plan'].eq('normal')].copy()
    if normal.empty:
        raise ValueError('没有计划正常的训练片段可建立硬时空专家组')
    references = _references(normal)
    statistics = normal.groupby('cell_key', observed=True).agg(
        members=('segment_id', 'size'), height_band=('height_band', 'first'), time_bucket=('time_bucket', 'first'),
        longitude_cell=('longitude_cell', 'first'), latitude_cell=('latitude_cell', 'first')).reset_index()
    statistics['reference_flights'] = statistics['cell_key'].map(references.groupby('cell_key', observed=True)['flight_id'].nunique())
    groups = statistics.loc[(statistics['members'] >= args.min_group_members)
                            & (statistics['reference_flights'] >= args.min_reference_flights)].copy()
    groups = groups.sort_values(['time_bucket', 'height_band', 'longitude_cell', 'latitude_cell', 'cell_key'], kind='stable').reset_index(drop=True)
    if groups.empty:
        raise ValueError('没有硬时空单元同时满足最少正常片段和不同参考航班数；请减小最小约束或加粗网格／时间桶')
    groups.insert(0, 'expert_id', range(len(groups)))
    groups.insert(0, 'group_id', groups['cell_key'].astype(str))
    groups['routing_members'] = groups['members']
    groups = groups.loc[:, GROUP_COLUMNS]
    routing = normal.merge(groups[['group_id', 'cell_key']], on='cell_key', how='inner', validate='many_to_one')
    routing = routing.loc[:, ['group_id', 'segment_id', 'flight_id', 'cell_key']]
    references = references.merge(groups[['group_id', 'cell_key']], on='cell_key', how='inner', validate='many_to_one')
    references = references.loc[:, ['group_id', 'segment_id', 'flight_id', 'cell_key', 'ordinal']]
    train, train_exclusions = _assign(segments.loc[segments['split'].eq('train')], groups, 'hard_exact')
    validation, validation_exclusions = _assign(segments.loc[segments['split'].eq('val')], groups, 'hard_exact')
    exclusions = pd.concat((train_exclusions, validation_exclusions), ignore_index=True)
    data = context.staging_dir / 'data'; data.mkdir()
    write_table(groups, data, 'groups')
    write_table(routing, data, 'routing_segments')
    write_table(references, data, 'references')
    write_table(train, data, 'train_assignments')
    write_table(validation, data, 'validation_assignments')
    write_table(exclusions, data, 'exclusions')
    normal_covered = int(train.loc[train['segment_label'].eq(0), 'segment_id'].nunique())
    summary = {'input_manifest': str(input_manifest), 'grouping_algorithm': 'hard_height_time_grid',
               'normal_training_segments': int(len(normal)), 'normal_training_coverage': normal_covered / len(normal),
               'effective_groups': int(len(groups)), 'reference_segments': int(len(references)),
               'train_targets': int(len(train)), 'validation_targets': int(len(validation)),
               'excluded_segments': int(len(exclusions)), 'min_group_members': args.min_group_members,
               'min_reference_flights': args.min_reference_flights}
    context.logger.info('硬分组完成：groups=%d；normal=%d；normal_coverage=%.2f%%；references=%d；train=%d；val=%d；excluded=%d',
                        len(groups), len(normal), summary['normal_training_coverage'] * 100, len(references), len(train), len(validation), len(exclusions))
    return ModuleResult(summary=summary, dependencies=(input_manifest,))


def validate_outputs(context: RunContext, args: argparse.Namespace, result: ModuleResult) -> ValidationReport:
    """验证每片段至多一个组、参考同源去重及有效硬单元约束。"""
    data = context.staging_dir / 'data'
    groups = read_table(data, 'groups', dtype={'group_id': 'string'})
    routing = read_table(data, 'routing_segments', dtype={'group_id': 'string', 'segment_id': 'string', 'flight_id': 'string'})
    references = read_table(data, 'references', dtype={'group_id': 'string', 'segment_id': 'string', 'flight_id': 'string'})
    train = read_table(data, 'train_assignments', dtype={'group_id': 'string', 'segment_id': 'string'})
    validation = read_table(data, 'validation_assignments', dtype={'group_id': 'string', 'segment_id': 'string'})
    if (groups.empty or groups['group_id'].duplicated().any() or groups['cell_key'].duplicated().any()
            or routing['segment_id'].duplicated().any() or references.duplicated(['group_id', 'flight_id']).any()
            or references.groupby('group_id')['flight_id'].nunique().lt(args.min_reference_flights).any()
            or train['segment_id'].duplicated().any() or validation['segment_id'].duplicated().any()
            or not set(train['group_id']).issubset(set(groups['group_id']))
            or not set(validation['group_id']).issubset(set(groups['group_id']))):
        raise ValueError('硬分组的唯一归属、参考航班或最小组约束无效')
    return ValidationReport(checks={'groups': int(len(groups)), 'references': int(len(references)),
                                    'train_targets': int(len(train)), 'validation_targets': int(len(validation)),
                                    'unique_hard_cell_assignment': True, 'normal_reference_deduplicated': True})
