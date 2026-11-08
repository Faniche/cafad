"""canonical.py | 2026-11-03 | ADS-B CSV 清洗、单位统一与稳定身份生成 | Codex."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


CANONICAL_COLUMNS = (
    'record_id', 'source_row_index', 'flight_id', 'icao', 'call', 'time_ns',
    'longitude_deg', 'latitude_deg', 'geometric_altitude_m', 'barometric_altitude_m',
    'heading_deg', 'speed_mps', 'injected_label',
)

ALIASES: dict[str, tuple[str, ...]] = {
    'icao': ('icao号', 'icao', 'icao24', 'aircraft_id'),
    'call': ('call号', 'call', 'callsign'),
    'time': ('时间', 'time', 'timestamp'),
    'longitude': ('经度', 'longitude', 'lon'),
    'latitude': ('纬度', 'latitude', 'lat'),
    'geometric_altitude': ('水平高度', 'geometric_altitude', 'gnss_altitude', 'altitude'),
    'barometric_altitude': ('气压高度', 'barometric_altitude', 'baro_altitude'),
    'heading': ('航向', 'heading', 'track', 'course'),
    'speed': ('速度', 'speed', 'groundspeed'),
}


def sha256_file(path: Path) -> str:
    """对原始输入计算可追溯的文件哈希。"""
    digest = hashlib.sha256()
    with path.open('rb') as reader:
        while block := reader.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def resolve_columns(columns: list[str], overrides: Mapping[str, str]) -> dict[str, str]:
    """根据显式列名或唯一别名选取九个必需原始字段。"""
    selected: dict[str, str] = {}
    for logical_name, aliases in ALIASES.items():
        candidates = [overrides[logical_name]] if logical_name in overrides else [name for name in aliases if name in columns]
        if len(candidates) != 1 or candidates[0] not in columns:
            raise ValueError(f'无法唯一识别 {logical_name} 列；请使用 --{logical_name.replace("_", "-")}-column 显式指定')
        selected[logical_name] = candidates[0]
    if len(set(selected.values())) != len(selected):
        raise ValueError('每个逻辑字段必须来自不同的输入列')
    return selected


def _parse_times_or_mark_invalid(values: pd.Series, timestamp_unit: str | None) -> tuple[np.ndarray, np.ndarray]:
    """解析时间并返回纳秒值与无效掩码，缺失或坏值由清洗记录而非中断全表。"""
    text = values.astype('string').str.strip()
    invalid = text.eq('').to_numpy(dtype=bool)
    time_ns = np.zeros(len(text), dtype=np.int64)
    positions = np.flatnonzero(~invalid)
    if not len(positions):
        return time_ns, invalid
    candidates = text.iloc[positions]
    if timestamp_unit is None:
        numeric_like = candidates.str.fullmatch(r'[+-]?\d+(?:\.\d+)?').to_numpy(dtype=bool)
        parsed = pd.to_datetime(
            candidates.mask(numeric_like), format='mixed', utc=True, errors='coerce',
        )
    elif candidates.str.fullmatch(r'[+-]?\d+').all():
        numeric = pd.to_numeric(candidates, errors='coerce').to_numpy(dtype=np.int64)
        parsed = pd.to_datetime(numeric, unit=timestamp_unit, utc=True, errors='coerce')
    else:
        numeric = pd.to_numeric(candidates, errors='coerce')
        parsed = pd.to_datetime(numeric, unit=timestamp_unit, utc=True, errors='coerce')
    parsed_ns = pd.DatetimeIndex(parsed).as_unit('ns').asi8
    invalid_candidates = parsed_ns == np.iinfo(np.int64).min
    invalid[positions[invalid_candidates]] = True
    time_ns[positions[~invalid_candidates]] = parsed_ns[~invalid_candidates]
    return time_ns, invalid


def _flight_ids(frame: pd.DataFrame, flight_gap_seconds: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按 ICAO、CALL 与时间间隔写入稳定 flight_id 和航班表。"""
    result = frame.sort_values(['icao', 'call', 'time_ns', 'source_row_index'], kind='stable').copy()
    flight_values: list[str] = []
    flight_rows: list[dict[str, object]] = []
    for (icao, call), group in result.groupby(['icao', 'call'], sort=False, observed=True):
        times = group['time_ns'].to_numpy(dtype=np.int64)
        starts = np.r_[0, np.flatnonzero(np.diff(times) > flight_gap_seconds * 1_000_000_000) + 1]
        ends = np.r_[starts[1:], len(group)]
        for ordinal, (start, end) in enumerate(zip(starts, ends)):
            identity = json.dumps([str(icao), str(call), int(times[start])], ensure_ascii=False)
            flight_id = hashlib.sha256(identity.encode('utf-8')).hexdigest()[:24]
            flight_values.extend([flight_id] * int(end - start))
            flight_rows.append({'flight_id': flight_id, 'icao': str(icao), 'call': str(call),
                                'flight_ordinal_for_identity': ordinal, 'records': int(end - start),
                                'start_time_ns': int(times[start]), 'end_time_ns': int(times[end - 1])})
    result['flight_id'] = flight_values
    return result, pd.DataFrame(flight_rows)


def preprocess_adsb_csv(source: Path, destination: Path, *, column_overrides: Mapping[str, str],
                        altitude_unit: str, speed_unit: str, heading_unit: str,
                        timestamp_unit: str | None, flight_gap_seconds: float) -> dict[str, object]:
    """清洗一个 ADS-B CSV，写出规范观测、拒绝记录和来源航班表。"""
    if not source.is_file():
        raise FileNotFoundError(f'找不到输入 CSV：{source}')
    if altitude_unit not in {'m', 'ft'} or speed_unit not in {'m/s', 'kt', 'km/h'} or heading_unit not in {'deg', 'rad'}:
        raise ValueError('高度、速度和航向单位必须来自支持的明确枚举')
    if timestamp_unit not in {None, 's', 'ms', 'us', 'ns'}:
        raise ValueError('数值时间戳单位必须为 s、ms、us 或 ns')
    if not np.isfinite(flight_gap_seconds) or flight_gap_seconds <= 0:
        raise ValueError('flight_gap_seconds 必须为正的有限数')
    source_hash = sha256_file(source)
    raw = pd.read_csv(source, dtype='string', keep_default_na=False)
    if raw.empty:
        raise ValueError('输入 CSV 没有观测记录')
    selected = resolve_columns(raw.columns.tolist(), column_overrides)
    frame = pd.DataFrame({key: raw[value] for key, value in selected.items()})
    frame['source_row_index'] = np.arange(len(frame), dtype=np.int64)
    for key in ('icao', 'call'):
        frame[key] = frame[key].astype('string').str.strip()
    missing_identity = frame['icao'].eq('') | frame['call'].eq('')
    numeric_keys = ('longitude', 'latitude', 'geometric_altitude', 'barometric_altitude', 'heading', 'speed')
    for key in numeric_keys:
        frame[key] = pd.to_numeric(frame[key], errors='coerce')
    frame['time_ns'], invalid_time = _parse_times_or_mark_invalid(frame.pop('time'), timestamp_unit)
    numeric = frame.loc[:, list(numeric_keys)].to_numpy(dtype=np.float64)
    invalid_numeric = ~np.isfinite(numeric).all(axis=1)
    invalid_coordinates = ~frame['latitude'].between(-90.0, 90.0).to_numpy() | ~frame['longitude'].between(-180.0, 180.0).to_numpy()
    invalid_heading = frame['heading'].to_numpy(dtype=np.float64) < 0.0
    # 地面等待或起飞前静止的 0 m/s 是合法的；负地速不是。若原始字段
    # 实际表示有符号的其他量，调用方必须通过字段映射改用真正的速度列。
    invalid_speed = frame['speed'].to_numpy(dtype=np.float64) < 0.0
    reason = np.full(len(frame), '', dtype=object)
    reason[missing_identity.to_numpy()] = 'missing_identity'
    reason[(reason == '') & invalid_time] = 'invalid_time'
    reason[(reason == '') & invalid_numeric] = 'nonfinite_required_value'
    reason[(reason == '') & invalid_coordinates] = 'invalid_coordinate'
    reason[(reason == '') & invalid_heading] = 'invalid_heading'
    reason[(reason == '') & invalid_speed] = 'negative_speed'
    frame = frame.loc[reason == ''].copy()
    if frame.empty:
        raise ValueError('清洗后没有保留任何有效记录')
    if altitude_unit == 'ft':
        frame[['geometric_altitude', 'barometric_altitude']] *= 0.3048
    if speed_unit == 'kt':
        frame['speed'] *= 1852.0 / 3600.0
    elif speed_unit == 'km/h':
        frame['speed'] /= 3.6
    if heading_unit == 'rad':
        frame['heading'] = np.degrees(frame['heading'])
    frame['heading'] = np.mod(frame['heading'].to_numpy(dtype=np.float64), 360.0)
    frame = frame.sort_values(['icao', 'call', 'time_ns', 'source_row_index'], kind='stable')
    duplicate = frame.duplicated(['icao', 'call', 'time_ns'], keep='first').to_numpy()
    duplicate_rejections = pd.DataFrame({
        'source_row_index': frame.loc[duplicate, 'source_row_index'].to_numpy(dtype=np.int64),
        'reason': 'duplicate_timestamp_within_identity',
    })
    frame = frame.loc[~duplicate].copy()
    frame, flights = _flight_ids(frame, flight_gap_seconds)
    frame['record_id'] = [hashlib.sha256(f'{source_hash}:{value}'.encode('utf-8')).hexdigest()[:32]
                          for value in frame['source_row_index'].to_numpy(dtype=np.int64)]
    observations = pd.DataFrame({
        'record_id': frame['record_id'], 'source_row_index': frame['source_row_index'], 'flight_id': frame['flight_id'],
        'icao': frame['icao'], 'call': frame['call'], 'time_ns': frame['time_ns'],
        'longitude_deg': frame['longitude'], 'latitude_deg': frame['latitude'],
        'geometric_altitude_m': frame['geometric_altitude'], 'barometric_altitude_m': frame['barometric_altitude'],
        'heading_deg': frame['heading'], 'speed_mps': frame['speed'], 'injected_label': np.zeros(len(frame), dtype=np.int8),
    }, columns=CANONICAL_COLUMNS)
    if observations['record_id'].duplicated().any() or observations['flight_id'].isna().any():
        raise RuntimeError('稳定记录或航班身份生成失败')
    destination.mkdir(parents=True, exist_ok=False)
    observations.to_csv(destination / 'observations.csv', index=False, encoding='utf-8')
    flights.to_csv(destination / 'flights.csv', index=False, encoding='utf-8')
    invalid_rejections = pd.DataFrame({
        'source_row_index': np.arange(len(reason), dtype=np.int64)[reason != ''],
        'reason': reason[reason != ''],
    })
    rejection_table = pd.concat((invalid_rejections, duplicate_rejections), ignore_index=True)
    if rejection_table['source_row_index'].duplicated().any():
        raise RuntimeError('同一原始记录不能被写入多次删除清单')
    if len(observations) + len(rejection_table) != len(raw):
        raise RuntimeError('清洗前后记录数不守恒')
    rejection_table.to_csv(destination / 'rejections.csv', index=False, encoding='utf-8')
    summary: dict[str, object] = {
        'input': str(source.resolve()), 'input_sha256': source_hash, 'input_rows': int(len(raw)),
        'kept_rows': int(len(observations)), 'rejected_rows': int(len(rejection_table)),
        'flights': int(len(flights)), 'columns': selected,
        'units': {'altitude': 'm', 'speed': 'm/s', 'heading': 'degrees'},
        'flight_gap_seconds': flight_gap_seconds,
    }
    return summary
