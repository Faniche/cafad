"""api.py | 2026-11-09 | 消费校准清单，冻结阈值检测指定分区 | Codex."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.app.arguments import ProjectArgumentParser
from src.app.artifacts import read_manifest
from src.app.contracts import RunContext, ModuleResult, ValidationReport
from src.app.tables import read_table, write_table
from .inference import score_partition, upstream
from .fusion import make_predictions


def build_parser() -> argparse.ArgumentParser:
    parser = ProjectArgumentParser(prog='main.py -m detect', description='使用固定系统生成片段与航班异常判别。')
    parser.add_argument('--input', required=True, type=Path, help='calibrate 已发布的 manifest.json')
    parser.add_argument('--split', choices=('val', 'test'), default='test')
    parser.add_argument('--gpus', default='0,1,2,3', help='参与评分的 GPU，例如 0,1,2,3；空闲设备领取任务')
    parser.add_argument('--score-task-size', type=int, default=32, help='每个专家评分任务的片段数，批内逐片段前向')
    return parser


def run(context: RunContext, args: argparse.Namespace) -> ModuleResult:
    calibrated = (context.project_root / args.input).resolve()
    read_manifest(calibrated, expected_module='calibrate')
    bundle = json.loads((calibrated.parent / 'system_bundle.json').read_text(encoding='utf-8'))
    trained = upstream(calibrated, 'train')
    if trained.resolve() != Path(bundle['train_manifest']).resolve():
        raise ValueError('校准系统模型与依赖不一致')
    scores, pairs, _ = score_partition(trained, args.split, bundle['top_r'], bundle['temperature'], args.gpus,
                                     context.logger, context.workers, args.score_task_size)
    segments, flights = make_predictions(scores, bundle['threshold'])
    data = context.staging_dir / 'data'; data.mkdir()
    write_table(pairs, data, 'expert_scores')
    write_table(segments, data, 'segment_predictions')
    write_table(flights, data, 'flight_predictions')
    context.logger.info('检测完成：segments=%d；flights=%d；异常片段=%d；异常航班=%d',
                        len(segments), len(flights), segments['prediction'].sum(), flights['prediction'].sum())
    return ModuleResult(summary={'segments': len(segments), 'flights': len(flights), 'threshold': bundle['threshold'],
                                 'split': args.split, 'abnormal_segments': int(segments['prediction'].sum())}, dependencies=(calibrated,))


def validate_outputs(context: RunContext, args: argparse.Namespace, result: ModuleResult) -> ValidationReport:
    segments = read_table(context.staging_dir / 'data', 'segment_predictions')
    flights = read_table(context.staging_dir / 'data', 'flight_predictions', dtype={'flight_id': str})
    if segments.empty or segments['segment_id'].duplicated().any() or not np.isfinite(segments['score']).all():
        raise ValueError('检测片段为空、重复或评分非有限')
    predicted, expected = make_predictions(segments, result.summary['threshold'])
    if not predicted['prediction'].equals(segments['prediction']) or not expected['prediction'].equals(flights['prediction']):
        raise ValueError('片段阈值或轨迹 OR 复算失败')
    if len(segments) != result.summary['segments'] or not segments['split'].eq(args.split).all():
        raise ValueError('检测覆盖数量或分区不一致')
    return ValidationReport(checks={'segments': len(segments), 'flight_or_verified': True, 'finite_scores': True})
