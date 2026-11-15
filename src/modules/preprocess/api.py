"""api.py | 2026-11-03 | preprocess 模块参数、运行与落盘验收 | Codex."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.app.arguments import ProjectArgumentParser
from src.app.contracts import ModuleResult, RunContext, ValidationReport
from src.app.tables import read_table
from .canonical import CANONICAL_COLUMNS, preprocess_adsb_csv


def build_parser() -> argparse.ArgumentParser:
    """定义原始 ADS-B CSV 预处理所需的模块参数。"""
    parser = ProjectArgumentParser(prog='main.py -m preprocess', description='清洗一个 ADS-B CSV 并生成规范观测。')
    parser.add_argument('--format', required=True, choices=('adsb-csv',),
                        help='原始数据适配器；当前正式入口仅支持标准 ADS-B CSV')
    parser.add_argument('--input', required=True, type=Path, help='原始 ADS-B CSV')
    parser.add_argument('--altitude-unit', required=True, choices=('m', 'ft'))
    parser.add_argument('--speed-unit', required=True, choices=('m/s', 'kt', 'km/h'))
    parser.add_argument('--heading-unit', required=True, choices=('deg', 'rad'))
    parser.add_argument('--timestamp-unit', choices=('s', 'ms', 'us', 'ns'),
                        help='仅数值时间戳需要，字符串时间戳不得猜测单位')
    parser.add_argument('--flight-gap-seconds', type=float, default=1800.0,
                        help='同 ICAO/CALL 相邻观测超过该时间间隔时拆分来源航班')
    parser.add_argument('--minimum-flight-records', type=int, default=2,
                        help='连续航班会话至少保留的记录数；默认 2，仅剔除孤立单点会话')
    for name in ('icao', 'call', 'time', 'longitude', 'latitude', 'geometric_altitude',
                 'barometric_altitude', 'heading', 'speed'):
        parser.add_argument(f'--{name.replace("_", "-")}-column', dest=f'{name}_column')
    return parser


def _project_path(context: RunContext, value: Path) -> Path:
    """项目内相对路径从根目录解析，外部绝对路径保持原样。"""
    return value if value.is_absolute() else context.project_root / value


def run(context: RunContext, args: argparse.Namespace) -> ModuleResult:
    """执行 CSV 清洗并发布前写入 data 目录。"""
    source = _project_path(context, args.input).resolve()
    overrides = {name: getattr(args, f'{name}_column') for name in ('icao', 'call', 'time', 'longitude', 'latitude',
                 'geometric_altitude', 'barometric_altitude', 'heading', 'speed') if getattr(args, f'{name}_column')}
    context.logger.debug('预处理输入：%s；列覆盖：%s；单位 altitude=%s speed=%s heading=%s timestamp=%s',
                         source, overrides, args.altitude_unit, args.speed_unit, args.heading_unit, args.timestamp_unit)
    summary = preprocess_adsb_csv(source, context.staging_dir / 'data', column_overrides=overrides,
                                  altitude_unit=args.altitude_unit, speed_unit=args.speed_unit,
                                  heading_unit=args.heading_unit, timestamp_unit=args.timestamp_unit,
                                  flight_gap_seconds=args.flight_gap_seconds,
                                  minimum_flight_records=args.minimum_flight_records)
    context.logger.info('预处理完成：输入=%d，保留=%d，剔除=%d，来源航班=%d，孤立会话=%d',
                        summary['input_rows'], summary['kept_rows'], summary['rejected_rows'], summary['flights'],
                        summary['isolated_flight_sessions'])
    return ModuleResult(summary=summary)


def validate_outputs(context: RunContext, args: argparse.Namespace, result: ModuleResult) -> ValidationReport:
    """重新读取 CSV，检查字段、身份唯一性、时间去重和航班外键。"""
    data = context.staging_dir / 'data'
    observations = read_table(data, 'observations', dtype={'record_id': 'string', 'flight_id': 'string', 'icao': 'string', 'call': 'string'})
    flights = read_table(data, 'flights', dtype={'flight_id': 'string', 'icao': 'string', 'call': 'string'})
    if tuple(observations.columns) != CANONICAL_COLUMNS:
        raise ValueError('规范观测字段或顺序不符合约定')
    if observations.empty or observations['record_id'].duplicated().any():
        raise ValueError('规范观测为空或 record_id 不唯一')
    if observations.duplicated(['icao', 'call', 'time_ns']).any():
        raise ValueError('同一 ICAO/CALL 内仍存在重复时间戳')
    if observations['flight_id'].isna().any() or not set(observations['flight_id']).issubset(set(flights['flight_id'])):
        raise ValueError('观测与来源航班表的关系不完整')
    if observations['injected_label'].ne(0).any():
        raise ValueError('preprocess 产物不应含注入标签')
    rejections = read_table(data, 'rejections')
    if len(observations) + len(rejections) != int(result.summary['input_rows']):
        raise ValueError('清洗前后记录数不守恒')
    return ValidationReport(checks={'observations': int(len(observations)), 'flights': int(len(flights)),
                                    'rejections': int(len(rejections)), 'unique_record_ids': True,
                                    'unique_identity_timestamps': True, 'row_conservation': True})
