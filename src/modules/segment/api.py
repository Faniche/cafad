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


def build_parser() -> argparse.ArgumentParser:
    """定义阶段识别模块参数。"""
    parser = ProjectArgumentParser(prog='main.py -m segment', description='对最终观测执行 Sun 阶段识别并生成片段。')
    parser.add_argument('--input', required=True, type=Path, help='preprocess 或未来 inject 模块已发布的 manifest.json')
    parser.add_argument('--flight-gap-seconds', type=float,
                        help='来源航班断开阈值；省略时继承 preprocess 记录的值')
    parser.add_argument('--window-seconds', type=float, default=60.0)
    parser.add_argument('--chunk-rows', type=int, default=200_000)
    parser.add_argument('--buckets', type=int, default=128)
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


def _write_segment_tables(data_dir: Path, phase_dir: Path) -> dict[str, int]:
    """回接可识别记录，并为按航班排除的数值失败轨迹保存稳定 ID 报告。"""
    observations = read_table(data_dir, 'input_observations', dtype={'record_id': 'string', 'flight_id': 'string'})
    points = pd.read_csv(phase_dir / 'points.csv', dtype={'flight_id': 'string', 'segment_id': 'string'})
    unrecognized = pd.read_csv(phase_dir / 'unrecognized_flights.csv', dtype={'flight_id': 'string'})
    unrecognized_points = pd.read_csv(phase_dir / 'unrecognized_points.csv', dtype={'flight_id': 'string'})
    observations['_pipeline_row_index'] = np.arange(len(observations), dtype=np.int64)
    has_split = 'split' in observations.columns
    observation_columns = ['record_id', 'flight_id', 'time_ns', 'injected_label', '_pipeline_row_index']
    if has_split:
        observation_columns.append('split')
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


def run(context: RunContext, args: argparse.Namespace) -> ModuleResult:
    """运行既有 Sun 数值实现，并将其结果转换为可独立读取的 S0。"""
    input_manifest = _project_path(context, args.input).resolve()
    manifest = read_manifest(input_manifest)
    if manifest.get('module') not in {'preprocess', 'inject'}:
        raise ValueError('segment 只能读取 preprocess 或 inject 的最终观测')
    if manifest.get('dataset_id') != context.dataset_id:
        raise ValueError('上游数据集身份与本次从 --input 自动识别的身份不一致')
    source_dir = input_manifest.parent / 'data'
    source = source_dir / 'observations.parquet'
    if not source.is_file():
        raise FileNotFoundError(f'上游缺少规范观测：{source}')
    gap = args.flight_gap_seconds if args.flight_gap_seconds is not None else _upstream_gap(input_manifest)
    data_dir = context.staging_dir / 'data'
    data_dir.mkdir()
    shutil.copy2(source, data_dir / 'input_observations.parquet')
    staged_source = data_dir / '.input_observations.runtime.csv'
    export_csv(data_dir, 'input_observations', staged_source)
    config = PipelineConfig(altitude_unit='m', speed_unit='m/s', workers=context.workers,
                            chunk_rows=args.chunk_rows, buckets=args.buckets, window_seconds=args.window_seconds,
                            flight_gap_seconds=gap, timestamp_unit='ns')
    context.logger.debug('阶段识别输入：manifest=%s；gap=%s；window=%s；chunk_rows=%d；buckets=%d；workers=%d',
                         input_manifest, gap, args.window_seconds, args.chunk_rows, args.buckets, context.workers)
    summary = process_cleaned_csv(staged_source, data_dir / 'phase', config,
                                  column_overrides={'icao': 'icao', 'call': 'call', 'time': 'time_ns',
                                                    'altitude': 'geometric_altitude_m', 'speed': 'speed_mps',
                                                    'longitude': 'longitude_deg', 'latitude': 'latitude_deg'},
                                  progress_database=data_dir / 'segment_progress.sqlite',
                                  logger=context.logger)
    linked = _write_segment_tables(data_dir, data_dir / 'phase')
    staged_source.unlink()
    shutil.rmtree(data_dir / 'phase')
    combined: dict[str, object] = {'input_manifest': str(input_manifest), 'flight_gap_seconds': gap,
                                   'window_seconds': args.window_seconds, **summary, **linked}
    context.logger.info('阶段片段完成：可识别点=%d，片段=%d，异常片段=%d，不可识别航班=%d／点=%d',
                        linked['points'], linked['segments'], linked['anomalous_segments'],
                        linked['unrecognized_flights'], linked['unrecognized_points'])
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
    if not set(segments['phase']).issubset({'GND', 'CL', 'DE', 'CR', 'LVL', 'NA'}):
        raise ValueError('存在未知阶段标签')
    if 'split' in observations.columns:
        if (mapping['split'].isna().any() or segments['split'].isna().any()
                or set(mapping['split']) - {'train', 'val', 'test'}
                or not mapping.groupby('segment_id')['split'].nunique().eq(1).all()):
            raise ValueError('注入后的 split 未正确传递到阶段输出')
    return ValidationReport(checks={'observations': int(len(observations)), 'recognized_points': int(len(mapping)),
                                    'unrecognized_points': int(len(excluded)), 'segments': int(len(segments)),
                                    'point_conservation': True, 'segment_label_or_verified': True})
