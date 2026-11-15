"""api.py | 2026-11-09 | 从 train 追溯完整验证集并冻结系统 | Codex."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.app.arguments import ProjectArgumentParser
from src.app.artifacts import write_json
from src.app.contracts import RunContext, ModuleResult, ValidationReport
from src.app.tables import read_table, write_table
from src.modules.detect.inference import score_partition
from .threshold import select_segment_threshold


def build_parser() -> argparse.ArgumentParser:
    parser = ProjectArgumentParser(prog='main.py -m calibrate', description='用完整验证集固定检测阈值。')
    parser.add_argument('--input', required=True, type=Path, help='train 已发布的 manifest.json')
    parser.add_argument('--gpus', default='0,1,2,3', help='参与评分的 GPU，例如 0,1,2,3；空闲设备领取任务')
    parser.add_argument('--score-task-size', type=int, default=32, help='每个专家评分任务的片段数，批内逐片段前向')
    return parser


def run(context: RunContext, args: argparse.Namespace) -> ModuleResult:
    trained = (context.project_root / args.input).resolve()
    scores, pairs, _ = score_partition(trained, 'val', args.gpus,
                                        context.logger, context.workers, args.score_task_size)
    threshold, scan = select_segment_threshold(scores['score'].to_numpy(), scores['segment_label'].to_numpy())
    data = context.staging_dir / 'data'; data.mkdir()
    write_table(scores, data, 'val_scores')
    write_table(pairs, data, 'val_expert_scores')
    write_table(scan, data, 'threshold_scan')
    write_json(context.staging_dir / 'system_bundle.json', {'schema_version': 1, 'train_manifest': str(trained),
               'threshold': threshold, 'fusion': 'hard-cell-exact',
               'fallback': 'majority-interval-all-experts', 'comparison': '>='})
    context.logger.info('校准完成：validation=%d；threshold=%.8f；F1=%.6f', len(scores), threshold, scan['f1'].max())
    return ModuleResult(summary={'validation_segments': len(scores), 'threshold': threshold, 'validation_f1': float(scan['f1'].max())},
                        dependencies=(trained,))


def validate_outputs(context: RunContext, args: argparse.Namespace, result: ModuleResult) -> ValidationReport:
    scores = read_table(context.staging_dir / 'data', 'val_scores')
    threshold, _ = select_segment_threshold(scores['score'].to_numpy(), scores['segment_label'].to_numpy())
    bundle = json.loads((context.staging_dir / 'system_bundle.json').read_text(encoding='utf-8'))
    if scores['segment_id'].duplicated().any() or not scores['split'].eq('val').all() or not np.isclose(threshold, bundle['threshold']):
        raise ValueError('校准 ID、分区或阈值复算不一致')
    return ValidationReport(checks={'val_only': True, 'threshold_recomputed': True, 'segments': len(scores)})
