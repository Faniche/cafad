"""api.py | 2026-11-09 | 从 detect 追溯标签，评价片段与航班检测 | Codex."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.app.arguments import ProjectArgumentParser
from src.app.artifacts import read_manifest, write_json
from src.app.contracts import RunContext, ModuleResult, ValidationReport
from src.app.tables import read_table, write_table
from src.modules.detect.inference import upstream
from src.modules.detect.fusion import make_predictions
from .metrics import binary_metrics, align_segments


def build_parser() -> argparse.ArgumentParser:
    parser = ProjectArgumentParser(prog='main.py -m evaluate', description='对已保存预测计算片段与航班指标，不改变阈值。')
    parser.add_argument('--input', required=True, type=Path, help='detect 已发布的 manifest.json；自动追溯标签')
    return parser


def read_evaluation(detected: Path) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    read_manifest(detected, expected_module='detect')
    calibrated = upstream(detected, 'calibrate')
    trained = upstream(calibrated, 'train')
    grouped = upstream(trained, 'group')
    segmented = upstream(grouped, 'segment')
    config = json.loads((detected.parent / 'resolved_config.json').read_text(encoding='utf-8'))
    split = config['module']['split']
    if split not in ('val', 'test'):
        raise ValueError('评价分区必须为 val 或 test')
    truth = read_table(segmented.parent / 'data', 'segments', keep_default_na=False,
                       dtype={'segment_id': str, 'flight_id': str},
                       columns=['segment_id', 'flight_id', 'phase', 'split', 'segment_label'])
    truth = truth.loc[truth.split.eq(split)]
    predictions = read_table(detected.parent / 'data', 'segment_predictions', keep_default_na=False,
                             dtype={'segment_id': str, 'flight_id': str})
    joined = align_segments(predictions, truth)
    bundle = json.loads((calibrated.parent / 'system_bundle.json').read_text(encoding='utf-8'))
    expected, flights = make_predictions(joined, bundle['threshold'])
    if not expected.prediction.equals(joined.prediction):
        raise ValueError('预测不符合校准的固定阈值')
    saved_flights = read_table(detected.parent / 'data', 'flight_predictions', dtype={'flight_id': str})
    if saved_flights.flight_id.duplicated().any() or set(saved_flights.flight_id) != set(flights.flight_id):
        raise ValueError('航班预测 ID 不完整或重复')
    check = flights.merge(saved_flights, on='flight_id', validate='one_to_one', suffixes=('', '_saved'))
    if not np.array_equal(check.prediction, check.prediction_saved) or not np.allclose(check.score, check.score_saved):
        raise ValueError('保存的航班预测不符合片段 OR')
    flight_truth = truth.groupby('flight_id').segment_label.max().rename('label').reset_index()
    return joined, flights.merge(flight_truth, on='flight_id', validate='one_to_one'), split


def run(context: RunContext, args: argparse.Namespace) -> ModuleResult:
    detected = (context.project_root / args.input).resolve()
    segments, flights, split = read_evaluation(detected)
    metrics = {'split': split, 'scope': 'segment 模块已产出的完整分区；不含此前无法识别的航班',
               'segment': binary_metrics(segments.segment_label.to_numpy(), segments.prediction.to_numpy()),
               'trajectory': binary_metrics(flights.label.to_numpy(), flights.prediction.to_numpy())}
    data = context.staging_dir / 'data'; data.mkdir()
    write_json(data / 'metrics.json', metrics)
    write_table(segments, data, 'segment_results')
    write_table(flights, data, 'flight_results')
    for field, filename in (('phase', 'metrics_by_phase.csv'), ('route_mode', 'metrics_by_route.csv')):
        keys = sorted(set(segments[field]) | ({'same_phase', 'fallback'} if field == 'route_mode' else set()))
        rows = []
        for key in keys:
            frame = segments.loc[segments[field].eq(key)]
            rows.append({field: key, **binary_metrics(frame.segment_label.to_numpy(), frame.prediction.to_numpy())})
        write_table(pd.DataFrame(rows), data, filename.removesuffix('.csv'))
    context.logger.info('评价完成：split=%s；片段F1=%.6f；航班F1=%.6f；segments=%d；flights=%d',
                        split, metrics['segment']['f1'], metrics['trajectory']['f1'], len(segments), len(flights))
    return ModuleResult(summary=metrics, dependencies=(detected,))


def validate_outputs(context: RunContext, args: argparse.Namespace, result: ModuleResult) -> ValidationReport:
    data = context.staging_dir / 'data'
    metrics = json.loads((data / 'metrics.json').read_text(encoding='utf-8'))
    for level, table, label in (('segment', 'segment_results', 'segment_label'), ('trajectory', 'flight_results', 'label')):
        frame = read_table(data, table)
        if binary_metrics(frame[label].to_numpy(), frame.prediction.to_numpy()) != metrics[level]:
            raise ValueError('评价指标复算不一致')
    return ValidationReport(checks={'metrics_recomputed': True, 'segment_samples': metrics['segment']['samples'],
                                    'trajectory_samples': metrics['trajectory']['samples']})
