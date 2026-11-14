"""api.py | 2026-11-03 | segment 模块参数、阶段识别与落盘验收 | Codex."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from src.app.arguments import ProjectArgumentParser
from src.app.artifacts import read_manifest
from src.app.contracts import ModuleResult, RunContext, ValidationReport
from src.app.tables import export_csv, read_table, write_table
from src.data_processing.phase_pipeline import PipelineConfig, process_cleaned_csv
from src.data_processing.sun_phase import PHASES as SUN_PHASES


def build_parser() -> argparse.ArgumentParser:
    """定义阶段识别模块参数。"""
    parser = ProjectArgumentParser(prog='main.py -m segment', description='对训练／验证集的干净观测执行阶段识别并生成片段。')
    parser.add_argument('--input', required=True, type=Path, help='split 模块已发布的 manifest.json')
    parser.add_argument('--flight-gap-seconds', type=float,
                        help='来源航班断开阈值；省略时继承 preprocess 记录的值')
    parser.add_argument('--window-seconds', type=float, default=60.0, help='Sun 模糊规则时间窗（秒）')
    parser.add_argument('--chunk-rows', type=int, default=200_000, help='Sun 临时 CSV 分块行数')
    parser.add_argument('--buckets', type=int, default=128, help='Sun 按航班散列的临时桶数')
    return parser


def _project_path(context: RunContext, value: Path) -> Path:
    return value if value.is_absolute() else context.project_root / value


def _upstream_gap(manifest_path: Path) -> float:
    """从 preprocess 或 inject 摘要继承已经冻结的航班断开阈值。"""
    summary_path = manifest_path.parent / 'summary.json'
    try:
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f'上游摘要无法读取：{summary_path}') from exc
    if isinstance(summary, dict) and 'flight_gap_seconds' in summary:
        gap = float(summary['flight_gap_seconds'])
        if np.isfinite(gap) and gap > 0:
            return gap
    config_path = manifest_path.parent / 'resolved_config.json'
    value = json.loads(config_path.read_text(encoding='utf-8'))
    module = value.get('module')
    if not isinstance(module, dict) or 'flight_gap_seconds' not in module:
        raise ValueError('上游产物没有记录 flight_gap_seconds；请显式传入 --flight-gap-seconds')
    gap = float(module['flight_gap_seconds'])
    if not np.isfinite(gap) or gap <= 0:
        raise ValueError('上游 flight_gap_seconds 无效')
    return gap


def _preprocess_manifest(split_manifest: Path) -> Path:
    """从唯一直接 split 输入恢复其 preprocess 祖先，不要求用户重复指定路径。"""
    manifest = read_manifest(split_manifest, expected_module='split')
    matches = [Path(item['manifest']) for item in manifest.get('dependencies', [])
               if read_manifest(Path(item['manifest'])).get('module') == 'preprocess']
    if len(matches) != 1:
        raise ValueError('split 清单无法唯一溯源 preprocess 输入')
    return matches[0]


def _write_segment_tables(data_dir: Path, phase_dir: Path) -> dict[str, int]:
    """回接可识别记录，并为按航班排除的数值失败轨迹保存稳定 ID 报告。"""
    observations = read_table(data_dir, 'input_observations', dtype={'record_id': 'string', 'flight_id': 'string'})
    points = pd.read_csv(phase_dir / 'points.csv', dtype={'flight_id': 'string', 'segment_id': 'string'})
    unrecognized = pd.read_csv(phase_dir / 'unrecognized_flights.csv', dtype={'flight_id': 'string'})
    unrecognized_points = pd.read_csv(phase_dir / 'unrecognized_points.csv', dtype={'flight_id': 'string'})
    observations['_pipeline_row_index'] = np.arange(len(observations), dtype=np.int64)
    has_split = 'split' in observations.columns
    has_plan = 'injection_plan' in observations.columns
    observation_columns = ['record_id', 'flight_id', 'time_ns', 'injected_label', '_pipeline_row_index']
    if has_split:
        observation_columns.append('split')
    if has_plan:
        observation_columns.append('injection_plan')
    mapping = points.merge(observations[observation_columns],
                           left_on='source_row_index', right_on='_pipeline_row_index', how='left',
                           suffixes=('_phase', '_observation'), validate='one_to_one')
    if (mapping['record_id'].isna().any()
            or not mapping['flight_id_observation'].eq(mapping['flight_id_phase']).all()
            or not mapping['time_ns_observation'].eq(mapping['time_ns_phase']).all()):
        raise RuntimeError('阶段输出无法与规范观测按稳定顺序对齐')
    mapping = mapping.rename(columns={'flight_id_phase': 'flight_id', 'time_ns_phase': 'time_ns'})
    mapping_columns = ['record_id', 'flight_id', 'time_ns', 'phase', 'segment_id', 'injected_label']
    if has_split:
        mapping_columns.append('split')
    if has_plan:
        mapping_columns.append('injection_plan')
    mapping = mapping[mapping_columns]
    if mapping['record_id'].duplicated().any() or mapping['segment_id'].isna().any():
        raise RuntimeError('点到片段映射不完整')
    excluded_columns = ['record_id', 'flight_id', 'time_ns', '_pipeline_row_index']
    if has_split:
        excluded_columns.append('split')
    excluded = unrecognized_points.merge(
        observations[excluded_columns],
        left_on='source_row_index', right_on='_pipeline_row_index', how='left', suffixes=('_report', ''),
        validate='one_to_one',
    )
    if (excluded['record_id'].isna().any() or excluded['record_id'].duplicated().any()
            or not excluded['flight_id_report'].eq(excluded['flight_id']).all()):
        raise RuntimeError('不可识别轨迹报告无法与规范观测按稳定顺序对齐')
    if unrecognized['flight_id'].duplicated().any() or int(unrecognized['points'].sum()) != len(excluded):
        raise RuntimeError('不可识别轨迹汇总与点报告不一致')
    excluded = excluded.merge(unrecognized[['flight_id', 'reason']], on='flight_id', how='left', validate='many_to_one')
    if excluded['reason'].isna().any() or set(mapping['record_id']) & set(excluded['record_id']):
        raise RuntimeError('可识别与不可识别记录不能重叠')
    if len(mapping) + len(excluded) != len(observations):
        raise RuntimeError('阶段结果没有覆盖全部规范观测')
    labels = mapping.groupby('segment_id', sort=False)['injected_label'].max().astype(np.int8).rename('segment_label')
    segments = pd.read_csv(phase_dir / 'segments.csv', dtype={'segment_id': 'string', 'flight_id': 'string'})
    segments = segments.merge(labels, on='segment_id', how='left', validate='one_to_one')
    if has_split:
        segment_splits = mapping[['segment_id', 'split']].drop_duplicates()
        if segment_splits['segment_id'].duplicated().any():
            raise RuntimeError('同一阶段片段不能跨 split')
        segments = segments.merge(segment_splits, on='segment_id', how='left', validate='one_to_one')
    if has_plan:
        segment_plans = mapping[['segment_id', 'injection_plan']].drop_duplicates()
        if segment_plans['segment_id'].duplicated().any():
            raise RuntimeError('同一阶段片段不能跨注入计划')
        segments = segments.merge(segment_plans, on='segment_id', how='left', validate='one_to_one')
    if segments['segment_label'].isna().any() or not set(segments['segment_id']).issubset(set(mapping['segment_id'])):
        raise RuntimeError('片段标签或映射不完整')
    write_table(mapping, data_dir, 'point_segment_map')
    write_table(segments, data_dir, 'segments')
    write_table(unrecognized, data_dir, 'unrecognized_trajectories')
    excluded_output_columns = ['record_id', 'flight_id', 'time_ns', 'reason']
    if has_split:
        excluded_output_columns.append('split')
    write_table(excluded[excluded_output_columns], data_dir, 'unrecognized_point_map')
    return {'points': int(len(mapping)), 'segments': int(len(segments)),
            'anomalous_segments': int(segments['segment_label'].sum()),
            'unrecognized_flights': int(len(unrecognized)), 'unrecognized_points': int(len(excluded))}


def _sequence_category(phases: tuple[str, ...]) -> str:
    """仅审计 Sun 原始序列，不在未核验前擅自合并或改写阶段标签。"""
    if not phases or set(phases) == {'GND'}:
        return 'ground_only'
    first_gnd, last_gnd = phases[0] == 'GND', phases[-1] == 'GND'
    if first_gnd and last_gnd:
        return 'complete_or_ground_return'
    if first_gnd:
        return 'departure_side'
    if last_gnd:
        return 'arrival_side'
    if set(phases) == {'CR'}:
        return 'cruise_only'
    if 'CR' not in phases and 'LVL' in phases:
        return 'partial_level_flight'
    return 'airborne_or_unresolved'


def write_sequence_review(data_dir: Path) -> dict[str, int]:
    """输出按航班排列的 Sun 原始阶段序列及其语义审计分类。"""
    segments = read_table(data_dir, 'segments', dtype={'segment_id': 'string', 'flight_id': 'string'})
    rows: list[dict[str, object]] = []
    for flight_id, frame in segments.sort_values(['flight_id', 'ordinal'], kind='stable').groupby('flight_id', sort=False):
        phases = tuple(frame['phase'].astype(str))
        category = _sequence_category(phases)
        rows.append({'flight_id': str(flight_id), 'icao': str(frame['icao'].iloc[0]), 'call': str(frame['call'].iloc[0]),
                     'segments': int(len(frame)), 'phase_sequence': '>'.join(phases),
                     'semantic_category': category,
                     'starts_ground': phases[0] == 'GND', 'ends_ground': phases[-1] == 'GND'})
    review = pd.DataFrame(rows)
    write_table(review, data_dir, 'flight_phase_sequences')
    return {str(key): int(value) for key, value in review['semantic_category'].value_counts().items()}


def segment_observations(observations: pd.DataFrame, data_dir: Path, args: argparse.Namespace,
                         workers: int, logger: object, flight_gap_seconds: float) -> dict[str, object]:
    """在指定目录分段一份已带 split/标签的观测表，供正式 segment 与 detect 测试复用。"""
    required = {'record_id', 'flight_id', 'icao', 'call', 'time_ns', 'longitude_deg', 'latitude_deg',
                'geometric_altitude_m', 'barometric_altitude_m', 'heading_deg', 'speed_mps',
                'injected_label', 'split'}
    if observations.empty or not required.issubset(observations.columns):
        raise ValueError('阶段识别输入缺少规范观测、分区或逐点标签字段')
    data_dir.mkdir(parents=True, exist_ok=True)
    write_table(observations, data_dir, 'input_observations')
    phase_dir = data_dir / 'phase'
    staged_source = data_dir / '.input_observations.runtime.csv'
    export_csv(data_dir, 'input_observations', staged_source)
    config = PipelineConfig(altitude_unit='m', speed_unit='m/s', workers=workers,
                            chunk_rows=args.chunk_rows, buckets=args.buckets, window_seconds=args.window_seconds,
                            flight_gap_seconds=flight_gap_seconds, timestamp_unit='ns')
    logger.debug('Sun 阶段识别运行：observations=%d；gap=%s；window=%s；chunk_rows=%d；buckets=%d；workers=%d',
                 len(observations), flight_gap_seconds, args.window_seconds, args.chunk_rows, args.buckets, workers)
    try:
        summary = process_cleaned_csv(staged_source, phase_dir, config,
                                      column_overrides={'icao': 'icao', 'call': 'call', 'time': 'time_ns',
                                                        'altitude': 'geometric_altitude_m', 'speed': 'speed_mps',
                                                        'longitude': 'longitude_deg', 'latitude': 'latitude_deg'},
                                      progress_database=data_dir / 'segment_progress.sqlite', logger=logger)
    finally:
        staged_source.unlink(missing_ok=True)
    linked = _write_segment_tables(data_dir, data_dir / 'phase')
    shutil.rmtree(data_dir / 'phase')
    categories = write_sequence_review(data_dir)
    return {**summary, **linked, 'segmenter': 'sun', 'phase_labels': list(SUN_PHASES),
            'sequence_categories': categories}


def run(context: RunContext, args: argparse.Namespace) -> ModuleResult:
    """仅对训练／验证集的干净轨迹执行选定阶段识别，冻结后不受注入扰动。"""
    input_manifest = _project_path(context, args.input).resolve()
    manifest = read_manifest(input_manifest, expected_module='split')
    if manifest.get('dataset_id') != context.dataset_id:
        raise ValueError('上游数据集身份与本次从 --input 自动识别的身份不一致')
    preprocess = _preprocess_manifest(input_manifest)
    observations = read_table(preprocess.parent / 'data', 'observations',
                              dtype={'record_id': 'string', 'flight_id': 'string', 'icao': 'string', 'call': 'string'})
    splits = read_table(input_manifest.parent / 'data', 'splits', dtype={'flight_id': 'string'})
    if observations.empty or splits.empty or splits['flight_id'].duplicated().any():
        raise ValueError('preprocess 观测或 split 航班计划为空／重复')
    observations = observations.merge(splits[['flight_id', 'split', 'injection_plan']], on='flight_id',
                                      how='inner', validate='many_to_one')
    observations = observations.loc[observations['split'].isin(('train', 'val'))].copy()
    observations['injected_label'] = np.int8(0)
    if observations.empty:
        raise ValueError('split 后没有训练或验证观测可供干净分段')
    gap = args.flight_gap_seconds if args.flight_gap_seconds is not None else _upstream_gap(preprocess)
    data_dir = context.staging_dir / 'data'
    summary = segment_observations(observations, data_dir, args, context.workers, context.logger, gap)
    combined: dict[str, object] = {'input_manifest': str(input_manifest), 'preprocess_manifest': str(preprocess),
                                   'flight_gap_seconds': gap, 'window_seconds': args.window_seconds,
                                   'segmenter': 'sun', **summary}
    context.logger.info('阶段片段完成：可识别点=%d，片段=%d，异常片段=%d，不可识别航班=%d／点=%d',
                        summary['points'], summary['segments'], summary['anomalous_segments'],
                        summary['unrecognized_flights'], summary['unrecognized_points'])
    return ModuleResult(summary=combined, dependencies=(input_manifest,))


def validate_outputs(context: RunContext, args: argparse.Namespace, result: ModuleResult) -> ValidationReport:
    """检查记录守恒、唯一映射、片段标签 OR 及阶段输出完整性。"""
    data_dir = context.staging_dir / 'data'
    observations = read_table(data_dir, 'input_observations', dtype={'record_id': 'string', 'flight_id': 'string'})
    mapping = read_table(data_dir, 'point_segment_map', dtype={'record_id': 'string', 'flight_id': 'string', 'segment_id': 'string'})
    segments = read_table(data_dir, 'segments', dtype={'segment_id': 'string', 'flight_id': 'string'})
    excluded = read_table(data_dir, 'unrecognized_point_map', dtype={'record_id': 'string', 'flight_id': 'string'})
    unrecognized = read_table(data_dir, 'unrecognized_trajectories', dtype={'flight_id': 'string'})
    if (mapping['record_id'].duplicated().any() or excluded['record_id'].duplicated().any()
            or len(mapping) + len(excluded) != len(observations)
            or set(mapping['record_id']) & set(excluded['record_id'])):
        raise ValueError('可识别与不可识别阶段结果没有一一覆盖全部观测')
    if mapping['segment_id'].isna().any() or set(mapping['segment_id']) != set(segments['segment_id']):
        raise ValueError('映射与片段表的 segment_id 不一致')
    if unrecognized['flight_id'].duplicated().any() or set(excluded['flight_id']) != set(unrecognized['flight_id']):
        raise ValueError('不可识别轨迹与点报告不一致')
    expected = mapping.groupby('segment_id')['injected_label'].max().astype(np.int8)
    actual = segments.set_index('segment_id')['segment_label'].astype(np.int8)
    if not expected.equals(actual.reindex(expected.index)):
        raise ValueError('片段标签不等于记录标签的 OR')
    if not set(segments['phase']).issubset(set(SUN_PHASES)):
        raise ValueError('存在未知阶段标签')
    if 'split' in observations.columns:
        if (mapping['split'].isna().any() or segments['split'].isna().any()
                or set(mapping['split']) - {'train', 'val', 'test'}
                or not mapping.groupby('segment_id')['split'].nunique().eq(1).all()):
            raise ValueError('注入后的 split 未正确传递到阶段输出')
    if not set(observations['split']).issubset({'train', 'val'}):
        raise ValueError('正式 segment 只能处理训练／验证集；测试集应由 detect 动态分段')
    return ValidationReport(checks={'observations': int(len(observations)), 'recognized_points': int(len(mapping)),
                                    'unrecognized_points': int(len(excluded)), 'segments': int(len(segments)),
                                    'point_conservation': True, 'segment_label_or_verified': True})
