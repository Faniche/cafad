"""hard_rules.py | 2026-11-15 | 高度、时间和空间网格的确定性轨迹切分 | Codex."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from src.app.tables import write_table


POINT_COLUMNS = ('record_id', 'flight_id', 'time_ns', 'cell_key', 'segment_id', 'injected_label', 'split', 'injection_plan')
SEGMENT_COLUMNS = ('segment_id', 'flight_id', 'icao', 'call', 'ordinal', 'cell_key', 'height_band', 'time_bucket',
                   'longitude_cell', 'latitude_cell', 'points', 'start_time', 'end_time', 'duration_seconds',
                   'start_longitude_deg', 'start_latitude_deg', 'end_longitude_deg', 'end_latitude_deg', 'span_km',
                   'start_altitude_m', 'end_altitude_m', 'segment_label', 'split', 'injection_plan')


@dataclass(frozen=True)
class HardSegmentConfig:
    """冻结硬分组边界；相同 cell_key 的片段属于同一候选专家组。"""

    lower_height_m: float = 3000.0
    upper_height_m: float = 6000.0
    low_time_bucket_seconds: float = 1200.0
    middle_time_bucket_seconds: float = 1800.0
    high_time_bucket_seconds: float = 7200.0
    low_grid_degrees: float = 0.5
    middle_grid_degrees: float = 3.0
    high_grid_degrees: float = 12.0

    def validate(self) -> None:
        values = [self.lower_height_m, self.upper_height_m, self.low_time_bucket_seconds,
                  self.middle_time_bucket_seconds, self.high_time_bucket_seconds,
                  self.low_grid_degrees, self.middle_grid_degrees, self.high_grid_degrees]
        if not (np.isfinite(values).all() and 0.0 <= self.lower_height_m < self.upper_height_m
                and all(value > 0.0 for value in values[2:])):
            raise ValueError('高度边界及各高度层的时间桶、空间网格必须为有效正数，且 lower < upper')

    def time_bucket_widths(self) -> np.ndarray:
        """按低、中、高高度层给出时间桶宽度，索引与 height_band 一一对应。"""
        return np.asarray((self.low_time_bucket_seconds, self.middle_time_bucket_seconds,
                           self.high_time_bucket_seconds), dtype=np.float64)

    def grid_widths(self) -> np.ndarray:
        """按低、中、高高度层给出经纬网格边长，索引与 height_band 一一对应。"""
        return np.asarray((self.low_grid_degrees, self.middle_grid_degrees,
                           self.high_grid_degrees), dtype=np.float64)

    def as_dict(self) -> dict[str, float]:
        return {'lower_height_m': self.lower_height_m, 'upper_height_m': self.upper_height_m,
                'low_time_bucket_seconds': self.low_time_bucket_seconds,
                'middle_time_bucket_seconds': self.middle_time_bucket_seconds,
                'high_time_bucket_seconds': self.high_time_bucket_seconds,
                'low_grid_degrees': self.low_grid_degrees,
                'middle_grid_degrees': self.middle_grid_degrees,
                'high_grid_degrees': self.high_grid_degrees}


def _haversine_km(longitudes: np.ndarray, latitudes: np.ndarray) -> float:
    """计算片段首末点球面距离；切分元数据仅用于展示，不替代实际航程。"""
    if len(longitudes) < 2:
        return 0.0
    lon = np.radians([longitudes[0], longitudes[-1]])
    lat = np.radians([latitudes[0], latitudes[-1]])
    value = np.sin((lat[1] - lat[0]) / 2.0) ** 2
    value += np.cos(lat[0]) * np.cos(lat[1]) * np.sin((lon[1] - lon[0]) / 2.0) ** 2
    return float(6371.0088 * 2.0 * np.arcsin(np.sqrt(value)))


def assign_cells(observations: pd.DataFrame, config: HardSegmentConfig, time_origin_ns: int) -> pd.DataFrame:
    """为每条观测赋唯一高度层、时间桶和经纬度网格；不使用阶段识别或聚类。"""
    required = {'time_ns', 'longitude_deg', 'latitude_deg', 'geometric_altitude_m'}
    if not required.issubset(observations.columns):
        raise ValueError('硬分段输入缺少时间、经纬度或几何高度字段')
    config.validate()
    result = observations.copy()
    altitude = result['geometric_altitude_m'].to_numpy(dtype=np.float64)
    bands = np.select((altitude < config.lower_height_m, altitude < config.upper_height_m), (0, 1), default=2).astype(np.int8)
    time_widths = config.time_bucket_widths()[bands]
    grid_widths = config.grid_widths()[bands]
    result['height_band'] = bands
    result['time_bucket'] = np.floor((result['time_ns'].to_numpy(dtype=np.int64) - int(time_origin_ns)) / (time_widths * 1e9)).astype(np.int64)
    result['longitude_cell'] = np.floor((result['longitude_deg'].to_numpy(dtype=np.float64) + 180.0) / grid_widths).astype(np.int32)
    result['latitude_cell'] = np.floor((result['latitude_deg'].to_numpy(dtype=np.float64) + 90.0) / grid_widths).astype(np.int32)
    result['cell_key'] = ['h%d_t%d_x%d_y%d' % values for values in zip(result['height_band'], result['time_bucket'], result['longitude_cell'], result['latitude_cell'], strict=True)]
    return result


def _segment_id(flight_id: str, ordinal: int, cell_key: str, start_time_ns: int) -> str:
    return hashlib.sha256(f'{flight_id}|{ordinal}|{cell_key}|{start_time_ns}'.encode('utf-8')).hexdigest()[:24]


def _flight_segments(frame: pd.DataFrame) -> tuple[list[dict[str, object]], list[pd.DataFrame]]:
    """只在同一航班的连续 cell_key 不同时断开，故每条模型片段有唯一硬组键。"""
    ordered = frame.sort_values(['time_ns', 'record_id'], kind='stable').reset_index(drop=True)
    switches = ordered['cell_key'].ne(ordered['cell_key'].shift()).to_numpy()
    starts = np.r_[0, np.flatnonzero(switches[1:]) + 1, len(ordered)]
    segment_rows: list[dict[str, object]] = []
    mapping_frames: list[pd.DataFrame] = []
    for ordinal, (start, stop) in enumerate(zip(starts[:-1], starts[1:], strict=True)):
        part = ordered.iloc[start:stop]
        first = part.iloc[0]
        last = part.iloc[-1]
        segment_id = _segment_id(str(first.flight_id), ordinal, str(first.cell_key), int(first.time_ns))
        segment_rows.append({
            'segment_id': segment_id, 'flight_id': str(first.flight_id), 'icao': str(first.icao), 'call': str(first.call),
            'ordinal': ordinal, 'cell_key': str(first.cell_key), 'height_band': int(first.height_band),
            'time_bucket': int(first.time_bucket), 'longitude_cell': int(first.longitude_cell), 'latitude_cell': int(first.latitude_cell),
            'points': int(len(part)), 'start_time': pd.Timestamp(int(first.time_ns), unit='ns', tz='UTC').isoformat(),
            'end_time': pd.Timestamp(int(last.time_ns), unit='ns', tz='UTC').isoformat(),
            'duration_seconds': (int(last.time_ns) - int(first.time_ns)) / 1e9,
            'start_longitude_deg': float(first.longitude_deg), 'start_latitude_deg': float(first.latitude_deg),
            'end_longitude_deg': float(last.longitude_deg), 'end_latitude_deg': float(last.latitude_deg),
            'span_km': _haversine_km(part['longitude_deg'].to_numpy(dtype=np.float64), part['latitude_deg'].to_numpy(dtype=np.float64)),
            'start_altitude_m': float(first.geometric_altitude_m), 'end_altitude_m': float(last.geometric_altitude_m),
            'segment_label': int(part['injected_label'].max()), 'split': str(first.split), 'injection_plan': str(first.injection_plan),
        })
        point_map = part.loc[:, ['record_id', 'flight_id', 'time_ns', 'cell_key', 'injected_label', 'split', 'injection_plan']].copy()
        point_map.insert(4, 'segment_id', segment_id)
        mapping_frames.append(point_map.loc[:, POINT_COLUMNS])
    return segment_rows, mapping_frames


def _open_progress(path: Path) -> sqlite3.Connection:
    """进度库仅由父进程写入，避免多进程/网页同时写 SQLite。"""
    database = sqlite3.connect(path)
    database.execute('PRAGMA journal_mode=WAL')
    database.execute('''CREATE TABLE segments (
        segment_id TEXT PRIMARY KEY, flight_id TEXT NOT NULL, icao TEXT NOT NULL, call TEXT NOT NULL,
        cell_key TEXT NOT NULL, height_band INTEGER NOT NULL, time_bucket INTEGER NOT NULL,
        longitude_cell INTEGER NOT NULL, latitude_cell INTEGER NOT NULL, points INTEGER NOT NULL,
        start_time TEXT NOT NULL, end_time TEXT NOT NULL, duration_seconds REAL NOT NULL,
        start_longitude_deg REAL NOT NULL, start_latitude_deg REAL NOT NULL,
        end_longitude_deg REAL NOT NULL, end_latitude_deg REAL NOT NULL, span_km REAL NOT NULL,
        start_altitude_m REAL NOT NULL, end_altitude_m REAL NOT NULL
    )''')
    return database


def _write_progress(database: sqlite3.Connection, rows: list[dict[str, object]]) -> None:
    database.executemany('''INSERT INTO segments VALUES (
        :segment_id, :flight_id, :icao, :call, :cell_key, :height_band, :time_bucket, :longitude_cell, :latitude_cell,
        :points, :start_time, :end_time, :duration_seconds, :start_longitude_deg, :start_latitude_deg,
        :end_longitude_deg, :end_latitude_deg, :span_km, :start_altitude_m, :end_altitude_m)''', rows)
    database.commit()


def segment_observations(observations: pd.DataFrame, data_dir: Path, config: HardSegmentConfig, time_origin_ns: int,
                         logger: logging.Logger) -> dict[str, object]:
    """切分一份带 split 与逐点标签的观测，并持久化模型片段、点映射及实时展示表。"""
    required = {'record_id', 'flight_id', 'icao', 'call', 'time_ns', 'longitude_deg', 'latitude_deg',
                'geometric_altitude_m', 'injected_label', 'split', 'injection_plan'}
    if observations.empty or not required.issubset(observations.columns):
        raise ValueError('硬分段输入缺少规范观测、split、计划或逐点标签字段')
    if observations['record_id'].duplicated().any():
        raise ValueError('硬分段输入 record_id 不得重复')
    data_dir.mkdir(parents=True, exist_ok=True)
    assigned = assign_cells(observations, config, time_origin_ns)
    write_table(assigned, data_dir, 'input_observations')
    segment_rows: list[dict[str, object]] = []
    mapping_frames: list[pd.DataFrame] = []
    progress = _open_progress(data_dir / 'segment_progress.sqlite')
    try:
        for _, flight in assigned.groupby('flight_id', sort=True, observed=True):
            rows, mapping = _flight_segments(flight)
            _write_progress(progress, rows)
            segment_rows.extend(rows)
            mapping_frames.extend(mapping)
    finally:
        progress.close()
    segments = pd.DataFrame(segment_rows, columns=SEGMENT_COLUMNS)
    mapping = pd.concat(mapping_frames, ignore_index=True).loc[:, POINT_COLUMNS]
    if (segments.empty or mapping['record_id'].duplicated().any() or len(mapping) != len(assigned)
            or not mapping.groupby('segment_id')['cell_key'].nunique().eq(1).all()):
        raise RuntimeError('硬分段没有保持记录守恒或片段唯一硬组键')
    write_table(mapping, data_dir, 'point_segment_map')
    write_table(segments, data_dir, 'segments')
    review = segments.groupby('flight_id', observed=True).agg(icao=('icao', 'first'), call=('call', 'first'),
        segments=('segment_id', 'size'), cells=('cell_key', lambda values: '>'.join(values.astype(str))),
        starts_low=('height_band', lambda values: bool(values.iloc[0] == 0)),
        ends_low=('height_band', lambda values: bool(values.iloc[-1] == 0))).reset_index()
    write_table(review, data_dir, 'flight_segment_sequences')
    logger.info('硬规则切分完成：points=%d；segments=%d；flights=%d；cells=%d', len(mapping), len(segments), review.shape[0], segments['cell_key'].nunique())
    return {'points': int(len(mapping)), 'segments': int(len(segments)), 'flights': int(len(review)),
            'cells': int(segments['cell_key'].nunique()), 'anomalous_segments': int(segments['segment_label'].sum()),
            'segmenter': 'hard_grid', 'time_origin_ns': int(time_origin_ns), 'hard_rules': config.as_dict()}
