"""api.py | 2026-11-15 | 硬单元局部合并的确定性专家分组与冻结路由 | Codex."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.app.arguments import ProjectArgumentParser
from src.app.artifacts import read_manifest
from src.app.contracts import ModuleResult, RunContext, ValidationReport
from src.app.tables import read_table, write_table
from .merge import CELL_GROUP_COLUMNS, MERGE_EVENT_COLUMNS, merge_hard_cells


GROUP_COLUMNS = ('group_id', 'expert_id', 'cell_key', 'height_band', 'time_bucket', 'longitude_cell', 'latitude_cell',
                 'members', 'routing_members', 'reference_flights', 'base_cells', 'min_time_bucket', 'max_time_bucket',
                 'min_longitude_cell', 'max_longitude_cell', 'min_latitude_cell', 'max_latitude_cell')
ROUTING_COLUMNS = ('group_id', 'segment_id', 'flight_id', 'cell_key')
REFERENCE_COLUMNS = ('group_id', 'segment_id', 'flight_id', 'cell_key', 'ordinal')
ASSIGNMENT_COLUMNS = ('segment_id', 'flight_id', 'cell_key', 'segment_label', 'group_id', 'expert_id', 'route_mode')
EXCLUSION_COLUMNS = ('segment_id', 'flight_id', 'cell_key', 'segment_label', 'reason')


def build_parser() -> argparse.ArgumentParser:
    """定义硬分组的最低支持、目标支持及局部合并边界。"""
    parser = ProjectArgumentParser(prog='main.py -m group', description='合并相邻稀疏硬时空单元，构造正常参考组和唯一专家路由。')
    parser.add_argument('--input', required=True, type=Path, help='segment 模块已发布的 manifest.json')
    parser.add_argument('--min-group-members', type=int, default=2, help='保留专家组所需的最少正常训练片段数（默认 2）')
    parser.add_argument('--min-reference-flights', type=int, default=2, help='保留专家组所需的最少不同正常训练航班数（默认 2）')
    parser.add_argument('--merge-target-members', type=int, default=8, help='稀疏单元优先合并至的正常训练片段数（默认 8）')
    parser.add_argument('--merge-target-reference-flights', type=int, default=4,
                        help='稀疏单元优先合并至的不同正常训练航班数（默认 4）')
    parser.add_argument('--merge-max-time-buckets', type=int, default=2,
                        help='一个合并组最多覆盖的连续时间桶数（默认 2）')
    parser.add_argument('--merge-max-grid-cells', type=int, default=3,
                        help='一个合并组在经度或纬度方向最多覆盖的连续网格数（默认 3）')
    return parser


def _project_path(context: RunContext, value: Path) -> Path:
    """将用户相对输入解析到唯一项目根目录。"""
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
    """校验上游身份后返回 segment 数据表。"""
    manifest = read_manifest(path, expected_module='segment')
    if manifest.get('dataset_id') is None:
        raise ValueError('segment 清单数据集身份无效')
    return read_segment_store_data(path.parent / 'data')


def _references(normal: pd.DataFrame) -> pd.DataFrame:
    """同一基础单元中每个原始航班只留最早一个正常片段。"""
    return (normal.sort_values(['cell_key', 'flight_id', 'ordinal', 'segment_id'], kind='stable')
            .drop_duplicates(['cell_key', 'flight_id'], keep='first'))


def _group_references(rows: pd.DataFrame) -> pd.DataFrame:
    """对合并后的组再次按航班去重，避免同航班跨基础单元重复进入一份记忆。"""
    return (rows.sort_values(['group_id', 'flight_id', 'ordinal', 'segment_id'], kind='stable')
            .drop_duplicates(['group_id', 'flight_id'], keep='first'))


def _assign(rows: pd.DataFrame, cell_groups: pd.DataFrame, route_mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按基础 cell_key 映射至唯一最终组；未被有效正常组覆盖的片段显式排除。"""
    lookup = cell_groups.loc[cell_groups['group_id'].notna()].set_index('cell_key')[['group_id', 'expert_id']]
    merged = rows.merge(lookup, on='cell_key', how='left', validate='many_to_one')
    assigned = merged.loc[merged['group_id'].notna(), ['segment_id', 'flight_id', 'cell_key', 'segment_label', 'group_id', 'expert_id']].copy()
    assigned['route_mode'] = route_mode
    excluded = merged.loc[merged['group_id'].isna(), ['segment_id', 'flight_id', 'cell_key', 'segment_label']].copy()
    excluded['reason'] = 'hard_cell_without_normal_reference'
    return assigned.loc[:, ASSIGNMENT_COLUMNS], excluded.loc[:, EXCLUSION_COLUMNS]


def run(context: RunContext, args: argparse.Namespace) -> ModuleResult:
    """用计划正常训练片段合并相邻稀疏单元，并冻结 train／val 的唯一专家路由。"""
    input_manifest = _project_path(context, args.input).resolve()
    segments, _ = _read_segment_store(input_manifest)
    normal = segments.loc[segments['split'].eq('train') & segments['injection_plan'].eq('normal')].copy()
    if normal.empty:
        raise ValueError('没有计划正常的训练片段可建立硬时空专家组')
    cell_groups, groups, merge_events = merge_hard_cells(
        normal, minimum_members=args.min_group_members, minimum_flights=args.min_reference_flights,
        target_members=args.merge_target_members, target_flights=args.merge_target_reference_flights,
        max_time_buckets=args.merge_max_time_buckets, max_grid_cells=args.merge_max_grid_cells,
    )
    groups.insert(1, 'expert_id', range(len(groups)))
    groups['routing_members'] = groups['members']
    groups = groups.loc[:, GROUP_COLUMNS]
    cell_groups = cell_groups.merge(groups[['group_id', 'expert_id']], on='group_id', how='left', validate='many_to_one')
    available_cells = cell_groups.loc[cell_groups['group_id'].notna(), ['group_id', 'cell_key']]
    routing = normal.merge(available_cells, on='cell_key', how='inner', validate='many_to_one')
    routing = routing.loc[:, ROUTING_COLUMNS]
    references = _group_references(routing.merge(normal[['segment_id', 'ordinal']], on='segment_id', validate='one_to_one'))
    references = references.loc[:, REFERENCE_COLUMNS]
    train, train_exclusions = _assign(segments.loc[segments['split'].eq('train')], cell_groups, 'hard_exact')
    validation, validation_exclusions = _assign(segments.loc[segments['split'].eq('val')], cell_groups, 'hard_exact')
    exclusions = pd.concat((train_exclusions, validation_exclusions), ignore_index=True)
    data = context.staging_dir / 'data'; data.mkdir()
    write_table(groups, data, 'groups')
    write_table(cell_groups.loc[:, (*CELL_GROUP_COLUMNS, 'expert_id')], data, 'cell_group_map')
    write_table(merge_events.loc[:, MERGE_EVENT_COLUMNS], data, 'merge_events')
    write_table(routing, data, 'routing_segments')
    write_table(references, data, 'references')
    write_table(train, data, 'train_assignments')
    write_table(validation, data, 'validation_assignments')
    write_table(exclusions, data, 'exclusions')
    normal_covered = int(train.loc[train['segment_label'].eq(0), 'segment_id'].nunique())
    summary = {
        'input_manifest': str(input_manifest), 'grouping_algorithm': 'constrained_hard_cell_merge',
        'normal_training_segments': int(len(normal)), 'normal_training_coverage': normal_covered / len(normal),
        'effective_groups': int(len(groups)), 'reference_segments': int(len(references)),
        'train_targets': int(len(train)), 'validation_targets': int(len(validation)),
        'excluded_segments': int(len(exclusions)), 'min_group_members': args.min_group_members,
        'min_reference_flights': args.min_reference_flights, 'merge_target_members': args.merge_target_members,
        'merge_target_reference_flights': args.merge_target_reference_flights,
        'merge_max_time_buckets': args.merge_max_time_buckets, 'merge_max_grid_cells': args.merge_max_grid_cells,
        'base_cells': int(len(cell_groups)), 'merged_groups': int((groups['base_cells'] > 1).sum()),
        'merge_events': int(len(merge_events)),
    }
    context.logger.info('硬单元合并完成：groups=%d；base_cells=%d；merge_events=%d；normal=%d；coverage=%.2f%%；references=%d；train=%d；val=%d；excluded=%d',
                        len(groups), len(cell_groups), len(merge_events), len(normal), summary['normal_training_coverage'] * 100,
                        len(references), len(train), len(validation), len(exclusions))
    return ModuleResult(summary=summary, dependencies=(input_manifest,))


def validate_outputs(context: RunContext, args: argparse.Namespace, result: ModuleResult) -> ValidationReport:
    """验证唯一归属、合并组参考去重、最低约束及基础单元映射完整性。"""
    data = context.staging_dir / 'data'
    groups = read_table(data, 'groups', dtype={'group_id': 'string'})
    cell_groups = read_table(data, 'cell_group_map', dtype={'group_id': 'string', 'cell_key': 'string'})
    routing = read_table(data, 'routing_segments', dtype={'group_id': 'string', 'segment_id': 'string', 'flight_id': 'string'})
    references = read_table(data, 'references', dtype={'group_id': 'string', 'segment_id': 'string', 'flight_id': 'string'})
    train = read_table(data, 'train_assignments', dtype={'group_id': 'string', 'segment_id': 'string'})
    validation = read_table(data, 'validation_assignments', dtype={'group_id': 'string', 'segment_id': 'string'})
    mapped = cell_groups.loc[cell_groups['group_id'].notna()]
    if (groups.empty or groups['group_id'].duplicated().any() or groups['cell_key'].duplicated().any()
            or cell_groups['cell_key'].duplicated().any() or not set(mapped['group_id']).issubset(set(groups['group_id']))
            or routing['segment_id'].duplicated().any() or references.duplicated(['group_id', 'flight_id']).any()
            or references.groupby('group_id')['flight_id'].nunique().lt(args.min_reference_flights).any()
            or train['segment_id'].duplicated().any() or validation['segment_id'].duplicated().any()
            or not set(train['group_id']).issubset(set(groups['group_id']))
            or not set(validation['group_id']).issubset(set(groups['group_id']))):
        raise ValueError('硬单元合并后的唯一归属、参考航班或最小组约束无效')
    return ValidationReport(checks={
        'groups': int(len(groups)), 'references': int(len(references)), 'base_cells': int(len(cell_groups)),
        'train_targets': int(len(train)), 'validation_targets': int(len(validation)),
        'unique_hard_cell_assignment': True, 'normal_reference_deduplicated': True,
    })
