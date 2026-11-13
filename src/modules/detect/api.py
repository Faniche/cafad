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
from src.modules.train.api import _group_manifest, _segment_manifest
from src.modules.segment.api import segment_observations


def build_parser() -> argparse.ArgumentParser:
    parser = ProjectArgumentParser(prog='main.py -m detect', description='对测试集先动态分段，再使用固定系统生成异常判别。')
    parser.add_argument('--input', required=True, type=Path, help='calibrate 已发布的 manifest.json')
    parser.add_argument('--split', choices=('test',), default='test', help='测试集固定由 detect 动态分段；验证仅由 calibrate 使用')
    parser.add_argument('--gpus', default='0,1,2,3', help='参与评分的 GPU，例如 0,1,2,3；空闲设备领取任务')
    parser.add_argument('--score-task-size', type=int, default=32, help='每个专家评分任务的片段数，批内逐片段前向')
    return parser


def _test_segmentation(context: RunContext, trained: Path, output: Path) -> dict[str, object]:
    """依据训练时冻结的 Sun 配置，对从未参与 group 的测试观测动态生成阶段片段。"""
    injected = upstream(trained, 'inject')
    grouped = _group_manifest(injected)
    segmented = _segment_manifest(grouped)
    config = json.loads((segmented.parent / 'resolved_config.json').read_text(encoding='utf-8')).get('module', {})
    segment_summary = json.loads((segmented.parent / 'summary.json').read_text(encoding='utf-8'))
    required = ('window_seconds', 'chunk_rows', 'buckets')
    if not isinstance(config, dict) or any(name not in config for name in required):
        raise ValueError('训练祖先 segment 未记录完整 Sun 分段配置')
    source = read_table(injected.parent / 'data', 'observations',
                        dtype={'record_id': 'string', 'flight_id': 'string'})
    test = source.loc[source['split'].eq('test')].copy()
    if test.empty:
        raise ValueError('inject 产物没有测试集观测')
    runtime = argparse.Namespace(window_seconds=float(config['window_seconds']),
                                 chunk_rows=int(config['chunk_rows']), buckets=int(config['buckets']))
    summary = segment_observations(test, output, runtime, context.workers, context.logger,
                                   float(segment_summary.get('flight_gap_seconds') or config.get('flight_gap_seconds') or 1800.0))
    context.logger.info('测试集动态分段完成：points=%d；segments=%d；sequence_categories=%s',
                        summary['points'], summary['segments'], summary['sequence_categories'])
    return summary


def run(context: RunContext, args: argparse.Namespace) -> ModuleResult:
    calibrated = (context.project_root / args.input).resolve()
    read_manifest(calibrated, expected_module='calibrate')
    bundle = json.loads((calibrated.parent / 'system_bundle.json').read_text(encoding='utf-8'))
    trained = upstream(calibrated, 'train')
    if trained.resolve() != Path(bundle['train_manifest']).resolve():
        raise ValueError('校准系统模型与依赖不一致')
    data = context.staging_dir / 'data'; data.mkdir()
    test_data = data / 'test_segmentation'
    segment_summary = _test_segmentation(context, trained, test_data)
    scores, pairs, _ = score_partition(trained, args.split, bundle['top_r'], bundle['temperature'], args.gpus,
                                        context.logger, context.workers, args.score_task_size, test_data)
    segments, flights = make_predictions(scores, bundle['threshold'])
    write_table(pairs, data, 'expert_scores')
    write_table(segments, data, 'segment_predictions')
    write_table(flights, data, 'flight_predictions')
    context.logger.info('检测完成：segments=%d；flights=%d；异常片段=%d；异常航班=%d',
                        len(segments), len(flights), segments['prediction'].sum(), flights['prediction'].sum())
    return ModuleResult(summary={'segments': len(segments), 'flights': len(flights), 'threshold': bundle['threshold'],
                                 'split': args.split, 'abnormal_segments': int(segments['prediction'].sum()),
                                 'test_segmentation': segment_summary}, dependencies=(calibrated,))


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
