"""api.py | 2026-11-15 | 硬时空网格切分模块接口 | Codex."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.app.arguments import ProjectArgumentParser
from src.app.artifacts import read_manifest
from src.app.contracts import ModuleResult, RunContext, ValidationReport
from src.app.tables import read_table

from .hard_rules import HardSegmentConfig, segment_observations


def build_parser() -> argparse.ArgumentParser:
    """定义不依赖 Sun 或学习型聚类的确定性时空切分参数。"""
    parser = ProjectArgumentParser(prog='main.py -m segment', description='按高度层、时间桶和经纬度网格切分训练／验证轨迹。')
    parser.add_argument('--input', required=True, type=Path, help='split 模块已发布的 manifest.json')
    parser.add_argument('--lower-height-m', type=float, default=3000.0, help='低／中高度层边界（m，默认 3000）')
    parser.add_argument('--upper-height-m', type=float, default=6000.0, help='中／高高度层边界（m，默认 6000）')
    parser.add_argument('--time-bucket-seconds', type=float, default=1200.0, help='相对数据集最早时刻的时间桶宽度（秒，默认 1200）')
    parser.add_argument('--grid-degrees', type=float, default=0.5, help='经纬度方格边长（度，默认 0.5）')
    return parser


def _project_path(context: RunContext, value: Path) -> Path:
    return value if value.is_absolute() else context.project_root / value


def _preprocess_manifest(split_manifest: Path) -> Path:
    manifest = read_manifest(split_manifest, expected_module='split')
    matches = [Path(item['manifest']) for item in manifest.get('dependencies', [])
               if read_manifest(Path(item['manifest'])).get('module') == 'preprocess']
    if len(matches) != 1:
        raise ValueError('split 清单无法唯一溯源 preprocess 输入')
    return matches[0]


def hard_config(args: argparse.Namespace) -> HardSegmentConfig:
    """从统一 CLI 参数构造并校验可由 detect 重放的规则配置。"""
    config = HardSegmentConfig(args.lower_height_m, args.upper_height_m, args.time_bucket_seconds, args.grid_degrees)
    config.validate()
    return config


def run(context: RunContext, args: argparse.Namespace) -> ModuleResult:
    """冻结训练／验证的干净硬分段；测试集由 detect 重放同一规则独立切分。"""
    input_manifest = _project_path(context, args.input).resolve()
    split_manifest = read_manifest(input_manifest, expected_module='split')
    if split_manifest.get('dataset_id') != context.dataset_id:
        raise ValueError('上游数据集身份与本次运行不一致')
    preprocess = _preprocess_manifest(input_manifest)
    all_observations = read_table(preprocess.parent / 'data', 'observations', dtype={'record_id': 'string', 'flight_id': 'string', 'icao': 'string', 'call': 'string'})
    splits = read_table(input_manifest.parent / 'data', 'splits', dtype={'flight_id': 'string'})
    observations = all_observations.merge(splits[['flight_id', 'split', 'injection_plan']], on='flight_id', how='inner', validate='many_to_one')
    observations = observations.loc[observations['split'].isin(('train', 'val'))].copy()
    observations['injected_label'] = np.int8(0)
    if observations.empty:
        raise ValueError('split 后没有训练或验证观测可供硬分段')
    config = hard_config(args)
    summary = segment_observations(observations, context.staging_dir / 'data', config,
                                   int(all_observations['time_ns'].min()), context.logger)
    result = {'input_manifest': str(input_manifest), 'preprocess_manifest': str(preprocess), **summary}
    context.logger.info('硬分段发布：片段=%d；唯一规则单元=%d；高度=[0,%gm)、[%gm,%gm)、[%gm,+∞)；时间桶=%gs；网格=%g°',
                        summary['segments'], summary['cells'], config.lower_height_m, config.lower_height_m,
                        config.upper_height_m, config.upper_height_m, config.time_bucket_seconds, config.grid_degrees)
    return ModuleResult(summary=result, dependencies=(input_manifest,))


def validate_outputs(context: RunContext, args: argparse.Namespace, result: ModuleResult) -> ValidationReport:
    """检查记录守恒、每片段唯一硬单元以及训练／验证边界。"""
    data = context.staging_dir / 'data'
    observations = read_table(data, 'input_observations', dtype={'record_id': 'string', 'flight_id': 'string'})
    mapping = read_table(data, 'point_segment_map', dtype={'record_id': 'string', 'segment_id': 'string'})
    segments = read_table(data, 'segments', dtype={'segment_id': 'string', 'flight_id': 'string'})
    required = {'cell_key', 'height_band', 'time_bucket', 'longitude_cell', 'latitude_cell'}
    if (mapping['record_id'].duplicated().any() or len(mapping) != len(observations)
            or mapping['segment_id'].isna().any() or set(mapping['segment_id']) != set(segments['segment_id'])
            or not required.issubset(segments.columns)
            or not mapping.groupby('segment_id')['cell_key'].nunique().eq(1).all()):
        raise ValueError('硬分段记录守恒、片段或规则单元约束失败')
    if not set(observations['split']).issubset({'train', 'val'}):
        raise ValueError('正式 segment 只能处理训练／验证集')
    return ValidationReport(checks={'observations': int(len(observations)), 'segments': int(len(segments)),
                                    'cells': int(segments['cell_key'].nunique()), 'point_conservation': True,
                                    'one_hard_cell_per_segment': True})
