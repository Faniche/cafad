"""Bounded-memory, one-flight-per-task processing of initially cleaned CSVs."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import FIRST_COMPLETED, wait
from contextlib import closing
from src.utils.process_pool import InterruptibleProcessPoolExecutor as ProcessPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import json
import logging
import math
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sqlite3
from statistics import fmean, median
import tempfile
import time

import numpy as np
import pandas as pd

from .sun_phase import PHASES, identify_sun_phases
from ..utils.logging_utils import close_project_logging, setup_project_logging


ALIASES = {
    'icao': ('icao号', 'icao', 'icao24', 'aircraft_id'),
    'call': ('call号', 'call', 'callsign'),
    'time': ('时间', 'time', 'timestamp'),
    'altitude': ('水平高度', 'altitude', 'horizontal_altitude'),
    'speed': ('速度', 'speed', 'groundspeed'),
}
LOCATION_ALIASES = {
    'longitude': ('经度', 'longitude', 'lon', 'longitude_deg'),
    'latitude': ('纬度', 'latitude', 'lat', 'latitude_deg'),
}
NATIVE_THREAD_KEYS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                      'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')
POINT_COLUMNS = ['source_row_index', 'flight_id', 'icao', 'call', 'time_ns', 'phase', 'segment_id']
SEGMENT_COLUMNS = ['segment_id', 'flight_id', 'icao', 'call', 'ordinal', 'phase', 'points', 'start_time',
                   'end_time', 'duration_seconds', 'start_longitude_deg', 'start_latitude_deg',
                   'end_longitude_deg', 'end_latitude_deg', 'span_km', 'start_altitude_m', 'end_altitude_m',
                   'first_source_row_index', 'last_source_row_index']
FLIGHT_COLUMNS = ['flight_id', 'icao', 'call', 'points', 'phase_types', 'duplicate_time_points',
                  'unique_time_points', 'start_time', 'end_time', 'worker_pid', 'seconds', 'segments',
                  'phase_counts']
UNRECOGNIZED_FLIGHT_COLUMNS = ['flight_id', 'icao', 'call', 'points', 'duration_seconds', 'start_time',
                               'end_time', 'start_longitude_deg', 'start_latitude_deg', 'end_longitude_deg',
                               'end_latitude_deg', 'reason']
UNRECOGNIZED_POINT_COLUMNS = ['source_row_index', 'flight_id']


def configure_worker_threads() -> None:
    """Set before numerical imports/spawn to avoid workers times BLAS threads."""
    for key in NATIVE_THREAD_KEYS:
        os.environ[key] = '1'


@dataclass(frozen=True)
class PipelineConfig:
    altitude_unit: str
    speed_unit: str
    workers: int = 1
    chunk_rows: int = 200_000
    buckets: int = 128
    window_seconds: float = 60.
    flight_gap_seconds: float = 1800.
    timestamp_unit: str | None = None
    vertical_rate_column: str | None = None
    vertical_rate_unit: str | None = None

    def validate(self) -> None:
        if self.altitude_unit not in {'m', 'ft'} or self.speed_unit not in {'m/s', 'kt', 'km/h'}:
            raise ValueError('Explicit supported altitude/speed units are required')
        if min(self.workers, self.chunk_rows, self.buckets) < 1:
            raise ValueError('workers, chunk_rows and buckets must be positive')
        if any(not math.isfinite(x) or x <= 0 for x in (self.window_seconds, self.flight_gap_seconds)):
            raise ValueError('Window and flight-gap durations must be positive')
        if self.timestamp_unit not in {None, 's', 'ms', 'us', 'ns'}:
            raise ValueError('Invalid numeric timestamp unit')
        if self.vertical_rate_column and self.vertical_rate_unit not in {'ft/min', 'm/s', 'ft/s'}:
            raise ValueError('A supplied vertical-rate column requires an explicit vertical-rate unit')
        if self.vertical_rate_unit and not self.vertical_rate_column:
            raise ValueError('vertical_rate_unit requires vertical_rate_column')


@dataclass
class FlightTask:
    flight_id: str
    icao: str
    call: str
    row_indices: np.ndarray
    timestamps_ns: np.ndarray
    altitude: np.ndarray
    speed: np.ndarray
    vertical_rate: np.ndarray | None
    longitude: np.ndarray | None = None
    latitude: np.ndarray | None = None


def _resolve_columns(source: Path, overrides: dict[str, str], config: PipelineConfig) -> dict[str, str]:
    header = pd.read_csv(source, nrows=0).columns.tolist()
    selected = {}
    for key, aliases in ALIASES.items():
        candidates = [overrides[key]] if key in overrides else [name for name in aliases if name in header]
        if len(candidates) != 1 or candidates[0] not in header:
            raise ValueError(f'Cannot identify unique {key} column: {candidates}; use an explicit column option')
        selected[key] = candidates[0]
    for key, aliases in LOCATION_ALIASES.items():
        candidates = [overrides[key]] if key in overrides else [name for name in aliases if name in header]
        if not candidates:
            continue
        if len(candidates) != 1 or candidates[0] not in header:
            raise ValueError(f'Cannot identify unique optional {key} column: {candidates}')
        selected[key] = candidates[0]
    if config.vertical_rate_column:
        if config.vertical_rate_column not in header:
            raise ValueError(f'Missing vertical-rate column: {config.vertical_rate_column}')
        selected['vertical_rate'] = config.vertical_rate_column
    if len(set(selected.values())) != len(selected):
        raise ValueError('Each logical input requires its own CSV column')
    return selected


def _parse_times(values: pd.Series, unit: str | None) -> np.ndarray:
    text = values.astype('string').str.strip()
    if unit:
        # Preserve integer epoch nanoseconds; never pass them through float64.
        if text.str.fullmatch(r'[+-]?\d+').all():
            numeric = pd.to_numeric(text, errors='raise').to_numpy(dtype=np.int64)
        else:
            numeric = pd.to_numeric(text, errors='raise').to_numpy(dtype=np.float64)
        parsed = pd.to_datetime(numeric, unit=unit, utc=True, errors='raise')
    else:
        if text.str.fullmatch(r'[+-]?\d+(?:\.\d+)?').any():
            raise ValueError('Numeric timestamps require --timestamp-unit; epoch units are never guessed')
        text = text.str.replace(r'(\d{2}:\d{2}:\d{2}) (\d{1,9})$', r'\1.\2', regex=True)
        parsed = pd.to_datetime(text, format='mixed', utc=True, errors='raise')
    if pd.isna(parsed).any():
        raise ValueError('Missing timestamp in initially cleaned input')
    return np.asarray(pd.DatetimeIndex(parsed).as_unit('ns').asi8, dtype=np.int64)


def _partition_csv(source: Path, workspace: Path, columns: dict, config: PipelineConfig,
                   logger: logging.Logger) -> tuple[list[Path], int]:
    """Hash partition IDs so even unsorted/chunk-spanning flights stay together."""
    paths = [workspace / f'bucket_{i:04d}.csv' for i in range(config.buckets)]
    seen = set()
    offset = 0
    numeric = ['altitude', 'speed'] + [key for key in ('longitude', 'latitude') if key in columns]
    if config.vertical_rate_column:
        numeric.append('vertical_rate')
    with pd.read_csv(source, usecols=list(columns.values()), dtype='string',
                     keep_default_na=False, chunksize=config.chunk_rows) as reader:
        for chunk in reader:
            chunk = chunk.rename(columns={value: key for key, value in columns.items()})
            for key in ('icao', 'call'):
                chunk[key] = chunk[key].str.strip()
                if chunk[key].eq('').any():
                    raise ValueError(f'Missing {key} in input rows {offset}..{offset + len(chunk) - 1}')
            chunk['time_ns'] = _parse_times(chunk.pop('time'), config.timestamp_unit)
            for key in numeric:
                chunk[key] = pd.to_numeric(chunk[key], errors='raise').to_numpy(dtype=np.float64)
                if not np.isfinite(chunk[key]).all():
                    raise ValueError(f'Nonfinite {key} in initially cleaned input')
            chunk['source_row_index'] = np.arange(offset, offset + len(chunk), dtype=np.int64)
            hashes = pd.util.hash_pandas_object(chunk[['icao', 'call']], index=False).to_numpy(dtype=np.uint64)
            for bucket, part in chunk.groupby(hashes % config.buckets, sort=False):
                bucket = int(bucket)
                part.to_csv(paths[bucket], mode='a', header=bucket not in seen, index=False, float_format='%.17g')
                seen.add(bucket)
            offset += len(chunk)
            logger.info('读取并分桶：%s 个观测点，%s 个非空桶', f'{offset:,}', len(seen))
    if offset == 0:
        raise ValueError('Input CSV has no observations')
    return [paths[i] for i in sorted(seen)], offset


def _iter_flights(bucket_paths: list[Path], config: PipelineConfig):
    for path in bucket_paths:
        frame = pd.read_csv(path, dtype={'icao': 'string', 'call': 'string',
                                        'time_ns': 'int64', 'source_row_index': 'int64'},
                            keep_default_na=False, float_precision='round_trip')
        frame = frame.sort_values(['icao', 'call', 'time_ns', 'source_row_index'], kind='stable')
        for (icao, call), group in frame.groupby(['icao', 'call'], sort=False):
            times = group.time_ns.to_numpy(dtype=np.int64)
            cuts = np.r_[0, np.flatnonzero(np.diff(times) > config.flight_gap_seconds * 1e9) + 1, len(group)]
            for start, end in zip(cuts[:-1], cuts[1:]):
                part = group.iloc[start:end]
                identity = json.dumps([str(icao), str(call), int(times[start])], ensure_ascii=False)
                flight_id = hashlib.sha256(identity.encode('utf-8')).hexdigest()[:24]
                yield FlightTask(
                    flight_id=flight_id,
                    icao=str(icao),
                    call=str(call),
                    row_indices=part.source_row_index.to_numpy(dtype=np.int64, copy=True),
                    timestamps_ns=part.time_ns.to_numpy(dtype=np.int64, copy=True),
                    altitude=part.altitude.to_numpy(dtype=np.float64, copy=True),
                    speed=part.speed.to_numpy(dtype=np.float64, copy=True),
                    vertical_rate=part.vertical_rate.to_numpy(dtype=np.float64, copy=True) if config.vertical_rate_column else None,
                    longitude=part.longitude.to_numpy(dtype=np.float64, copy=True) if 'longitude' in part else None,
                    latitude=part.latitude.to_numpy(dtype=np.float64, copy=True) if 'latitude' in part else None,
                )


def _iso(timestamp_ns: int) -> str:
    return pd.Timestamp(int(timestamp_ns), tz='UTC').isoformat()


def _span_km(task: FlightTask, start: int, end: int) -> float | None:
    """返回片段首末点球面直线距离；缺失位置时不伪造数值。"""
    if task.longitude is None or task.latitude is None:
        return None
    latitude = np.deg2rad([task.latitude[start], task.latitude[end - 1]])
    longitude = np.deg2rad([task.longitude[start], task.longitude[end - 1]])
    value = np.sin((latitude[1] - latitude[0]) / 2) ** 2
    value += np.cos(latitude[0]) * np.cos(latitude[1]) * np.sin((longitude[1] - longitude[0]) / 2) ** 2
    return float(6371.0088 * 2 * np.arcsin(np.sqrt(value)))


def _progress_database(path: Path) -> sqlite3.Connection:
    """创建阶段进度数据库；仅父进程持有该连接并串行写入。"""
    database = sqlite3.connect(path)
    database.execute('PRAGMA journal_mode=WAL')
    database.execute('''CREATE TABLE IF NOT EXISTS segments (
        segment_id TEXT PRIMARY KEY, flight_id TEXT NOT NULL, icao TEXT NOT NULL, call TEXT NOT NULL,
        phase TEXT NOT NULL, points INTEGER NOT NULL, start_time TEXT NOT NULL, end_time TEXT NOT NULL,
        duration_seconds REAL NOT NULL, start_longitude_deg REAL, start_latitude_deg REAL,
        end_longitude_deg REAL, end_latitude_deg REAL, span_km REAL, start_altitude_m REAL, end_altitude_m REAL
    )''')
    return database


def _write_progress(database: sqlite3.Connection, segments: list[dict]) -> None:
    """由主进程一次提交一个航班的片段，避免工作进程并发写 SQLite。"""
    database.executemany(
        '''INSERT INTO segments VALUES (:segment_id, :flight_id, :icao, :call, :phase, :points, :start_time, :end_time,
           :duration_seconds, :start_longitude_deg, :start_latitude_deg, :end_longitude_deg, :end_latitude_deg,
           :span_km, :start_altitude_m, :end_altitude_m)''',
        segments,
    )
    database.commit()


def write_segment_progress(path: Path, segments: list[dict]) -> None:
    """将一条航班的片段进度写入 SQLite；供其他分段器复用父进程单写入约束。"""
    with closing(_progress_database(path)) as database:
        _write_progress(database, segments)


def _unrecognized_result(task: FlightTask, reason: str, workspace: str) -> dict:
    """保存单条数值不可识别轨迹的最小追溯信息，不伪造阶段或片段。"""
    def coordinate(values: np.ndarray | None, index: int) -> float | None:
        return None if values is None else float(values[index])

    points_path = Path(workspace) / f'unrecognized_points_{task.flight_id}.csv'
    pd.DataFrame({'source_row_index': task.row_indices, 'flight_id': task.flight_id}).to_csv(points_path, index=False)
    return {
        'status': 'unrecognized',
        'unrecognized': {
            'flight_id': task.flight_id,
            'icao': task.icao,
            'call': task.call,
            'points': int(len(task.row_indices)),
            'duration_seconds': (int(task.timestamps_ns[-1]) - int(task.timestamps_ns[0])) / 1e9,
            'start_time': _iso(task.timestamps_ns[0]),
            'end_time': _iso(task.timestamps_ns[-1]),
            'start_longitude_deg': coordinate(task.longitude, 0),
            'start_latitude_deg': coordinate(task.latitude, 0),
            'end_longitude_deg': coordinate(task.longitude, -1),
            'end_latitude_deg': coordinate(task.latitude, -1),
            'reason': reason,
        },
        'unrecognized_points_path': str(points_path),
    }


def _process_flight(task: FlightTask, config: PipelineConfig, workspace: str) -> dict:
    """A child owns one flight and one private result file; no shared CSV writes."""
    started = time.monotonic()
    unique_times, starts, inverse, counts = np.unique(task.timestamps_ns, return_index=True,
                                                     return_inverse=True, return_counts=True)
    average = lambda values: np.add.reduceat(values, starts) / counts
    altitude = average(task.altitude) * (1. / .3048 if config.altitude_unit == 'm' else 1.)
    speed = average(task.speed) * {'m/s': 3600. / 1852., 'kt': 1., 'km/h': 1000. / 1852.}[config.speed_unit]
    rate = None
    if task.vertical_rate is not None:
        rate = average(task.vertical_rate) * {'ft/min': 1., 'm/s': 60. / .3048, 'ft/s': 60.}[config.vertical_rate_unit]
    relative = (unique_times - unique_times[0]).astype(np.float64) / 1e9
    try:
        unique_labels = identify_sun_phases(relative, altitude, speed, rate, window_seconds=config.window_seconds)
    except ValueError as exc:
        return _unrecognized_result(task, f'{type(exc).__name__}: {exc}', workspace)
    labels = unique_labels[inverse]
    cuts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
    segment_ids = np.empty(len(labels), dtype=object)
    segments = []
    for ordinal, (start, end) in enumerate(zip(cuts[:-1], cuts[1:])):
        segment_id = f'{task.flight_id}:{ordinal}'
        segment_ids[start:end] = segment_id
        segments.append({'segment_id': segment_id, 'flight_id': task.flight_id,
                         'icao': task.icao, 'call': task.call, 'ordinal': ordinal,
                         'phase': str(labels[start]), 'points': int(end - start),
                         'start_time': _iso(task.timestamps_ns[start]),
                         'end_time': _iso(task.timestamps_ns[end - 1]),
                         'duration_seconds': (int(task.timestamps_ns[end - 1]) - int(task.timestamps_ns[start])) / 1e9,
                         'start_longitude_deg': None if task.longitude is None else float(task.longitude[start]),
                         'start_latitude_deg': None if task.latitude is None else float(task.latitude[start]),
                         'end_longitude_deg': None if task.longitude is None else float(task.longitude[end - 1]),
                         'end_latitude_deg': None if task.latitude is None else float(task.latitude[end - 1]),
                         'span_km': _span_km(task, start, end),
                         'start_altitude_m': float(task.altitude[start]),
                         'end_altitude_m': float(task.altitude[end - 1]),
                         'first_source_row_index': int(task.row_indices[start]),
                         'last_source_row_index': int(task.row_indices[end - 1])})
    points_path = Path(workspace) / f'points_{task.flight_id}.csv'
    pd.DataFrame({'source_row_index': task.row_indices, 'flight_id': task.flight_id,
                  'icao': task.icao, 'call': task.call, 'time_ns': task.timestamps_ns,
                  'phase': labels, 'segment_id': segment_ids}).to_csv(points_path, index=False)
    return {'status': 'recognized', 'flight_id': task.flight_id, 'icao': task.icao, 'call': task.call,
            'points': len(labels), 'segments': segments,
            'phase_types': len(set(labels.tolist())), 'duplicate_time_points': int(len(labels) - len(unique_times)),
            'unique_time_points': len(unique_times), 'points_path': str(points_path),
            'start_time': _iso(task.timestamps_ns[0]), 'end_time': _iso(task.timestamps_ns[-1]),
            'phase_counts': dict(Counter(segment['phase'] for segment in segments)),
            'worker_pid': os.getpid(), 'seconds': time.monotonic() - started}


def _distribution(values: list[int]) -> dict:
    if not values:
        return {'count': 0, 'min': None, 'max': None, 'mean': None, 'median': None}
    return {'count': len(values), 'min': min(values), 'max': max(values),
            'mean': fmean(values), 'median': median(values)}


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def process_cleaned_csv(source: str | Path, output: str | Path, config: PipelineConfig,
                        *, column_overrides: dict[str, str] | None = None,
                        progress_database: str | Path | None = None,
                        logger: logging.Logger | None = None) -> dict:
    """Write point membership, per-flight/segment tables and aggregate statistics.

    ICAO + callsign + a configurable temporal gap define inferred flight sessions.
    Duplicate times are averaged for identification only. Recognized rows map
    back to their original source_row_index; a per-flight numerical failure is
    retained separately in the unrecognized reports rather than mislabelled.
    """
    import csv

    owns_logger = logger is None
    if owns_logger:
        logger, _ = setup_project_logging(console=False)
    log_path = getattr(logger, 'project_log_path', None)
    output_created = False
    started = time.monotonic()
    processed = submitted = unrecognized_flights = unrecognized_points = 0
    flight_points, segment_counts, segment_points, phase_types = [], [], [], []
    phase_segment_counts, phase_point_counts = Counter(), Counter()
    worker_pids = set()
    duplicate_points = short_flights = 0
    try:
        config.validate()
        configure_worker_threads()
        source, output = Path(source).resolve(), Path(output).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        columns = _resolve_columns(source, column_overrides or {}, config)
        source_stat = source.stat()
        output.mkdir(parents=True, exist_ok=False)
        output_created = True
        _write_json(output / 'configuration.json', {'input': str(source), 'input_bytes': source_stat.st_size,
                    'input_mtime_ns': source_stat.st_mtime_ns, 'config': asdict(config), 'columns': columns,
                    'log_file': str(log_path) if log_path is not None else None,
                    'input_stage': 'initially_cleaned_before_any_new_noise_injection',
                    'flight_definition': 'ICAO + callsign; split when adjacent time gap exceeds flight_gap_seconds',
                    'naive_timestamp_convention': 'UTC for storage; source timezone is not independently certified',
                    'method': 'Sun 2017 author-rule reimplementation; final window and short-flight handling fixed',
                    'source_reference': 'https://github.com/junzis/flight-data-processor/blob/master/flightphase.py',
                    'queue': 'ProcessPoolExecutor synchronized multiprocessing queues; bounded to 2 * workers',
                    'native_threads_per_worker': 1})
        logger.info('开始 Sun 阶段识别：workers=%d，chunk_rows=%d，buckets=%d', config.workers, config.chunk_rows, config.buckets)
        progress_path = Path(progress_database).resolve() if progress_database is not None else output / 'segment_progress.sqlite'
        with tempfile.TemporaryDirectory(prefix='_sun_work_', dir=output) as directory:
            workspace = Path(directory)
            bucket_paths, input_points = _partition_csv(source, workspace, columns, config, logger)
            current = source.stat()
            if (current.st_size, current.st_mtime_ns) != (source_stat.st_size, source_stat.st_mtime_ns):
                raise RuntimeError('Input CSV changed while it was being read')
            tasks = iter(_iter_flights(bucket_paths, config))
            with closing(_progress_database(progress_path)) as progress, \
                 (output / 'points.csv').open('w', encoding='utf-8', newline='') as points_file, \
                 (output / 'segments.csv').open('w', encoding='utf-8', newline='') as segments_file, \
                 (output / 'flights.csv').open('w', encoding='utf-8', newline='') as flights_file, \
                 (output / 'unrecognized_flights.csv').open('w', encoding='utf-8', newline='') as unrecognized_file, \
                 (output / 'unrecognized_points.csv').open('w', encoding='utf-8', newline='') as unrecognized_points_file:
                csv.writer(points_file).writerow(POINT_COLUMNS)
                csv.DictWriter(segments_file, fieldnames=SEGMENT_COLUMNS).writeheader()
                csv.DictWriter(flights_file, fieldnames=FLIGHT_COLUMNS).writeheader()
                unrecognized_writer = csv.DictWriter(unrecognized_file, fieldnames=UNRECOGNIZED_FLIGHT_COLUMNS)
                unrecognized_points_writer = csv.writer(unrecognized_points_file)
                unrecognized_writer.writeheader()
                unrecognized_points_writer.writerow(UNRECOGNIZED_POINT_COLUMNS)
                # Each submit is one complete flight. Executor's IPC queues lock
                # their own put/get operations; workers never race on a list or
                # access the input CSV. Only the parent mutates pending/writers.
                with ProcessPoolExecutor(max_workers=config.workers, mp_context=mp.get_context('spawn'),
                                         initializer=configure_worker_threads) as pool:
                    pending = {}

                    def enqueue() -> bool:
                        nonlocal submitted
                        try:
                            task = next(tasks)
                        except StopIteration:
                            return False
                        future = pool.submit(_process_flight, task, config, directory)
                        pending[future] = task.flight_id
                        submitted += 1
                        return True

                    for _ in range(config.workers * 2):
                        if not enqueue():
                            break
                    while pending:
                        completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                        for future in completed:
                            identity = pending.pop(future)
                            try:
                                result = future.result()
                            except Exception as exc:
                                for queued in pending:
                                    queued.cancel()
                                raise RuntimeError(f'Flight {identity} failed: {exc}') from exc
                            if result['status'] == 'unrecognized':
                                unrecognized_writer.writerow(result['unrecognized'])
                                with Path(result['unrecognized_points_path']).open('r', encoding='utf-8', newline='') as rows:
                                    next(rows)
                                    shutil.copyfileobj(rows, unrecognized_points_file)
                                Path(result['unrecognized_points_path']).unlink()
                                report = result['unrecognized']
                                unrecognized_flights += 1
                                unrecognized_points += int(report['points'])
                                logger.warning('跳过不可识别轨迹 [ICAO=%s CALL=%s id=%s]：%d 点，%.3f 秒；原因=%s',
                                               report['icao'], report['call'], identity, report['points'],
                                               report['duration_seconds'], report['reason'])
                                enqueue()
                                continue
                            with Path(result['points_path']).open('r', encoding='utf-8', newline='') as rows:
                                next(rows)  # Parent writes the only consolidated header.
                                shutil.copyfileobj(rows, points_file)
                            Path(result['points_path']).unlink()
                            segments = result['segments']
                            csv.DictWriter(segments_file, fieldnames=SEGMENT_COLUMNS).writerows(segments)
                            _write_progress(progress, segments)
                            flight = {key: result[key] for key in ('flight_id', 'icao', 'call', 'points', 'phase_types',
                                      'duplicate_time_points', 'unique_time_points', 'start_time', 'end_time', 'worker_pid', 'seconds')}
                            flight['segments'] = len(segments)
                            flight['phase_counts'] = json.dumps(result['phase_counts'], sort_keys=True)
                            csv.DictWriter(flights_file, fieldnames=FLIGHT_COLUMNS).writerow(flight)
                            flight_points.append(result['points'])
                            segment_counts.append(len(segments))
                            phase_types.append(result['phase_types'])
                            segment_points.extend(segment['points'] for segment in segments)
                            phase_segment_counts.update(result['phase_counts'])
                            for segment in segments:
                                phase_point_counts[segment['phase']] += segment['points']
                            duplicate_points += result['duplicate_time_points']
                            short_flights += result['unique_time_points'] < 4
                            worker_pids.add(result['worker_pid'])
                            processed += 1
                            logger.info('已处理 %d 个航班 [ICAO=%s CALL=%s id=%s]：%d 点 -> %d 个阶段片段，%s，worker=%d',
                                        processed, result['icao'], result['call'], identity, result['points'],
                                        len(segments), result['phase_counts'], result['worker_pid'])
                            enqueue()
        identified_points = sum(flight_points)
        if processed + unrecognized_flights != submitted:
            raise RuntimeError('Flight accounting failed')
        if identified_points != sum(segment_points) or identified_points + unrecognized_points != input_points:
            raise RuntimeError('Point conservation failed')
        summary = {'status': 'completed', 'input': str(source), 'output': str(output),
                   'log_file': str(log_path) if log_path is not None else None,
                   'input_points': input_points, 'processed_flights': processed,
                   'unrecognized_flights': unrecognized_flights, 'unrecognized_points': unrecognized_points,
                   'submitted_flights': submitted,
                   'total_segments': len(segment_points), 'configured_workers': config.workers,
                   'observed_worker_processes': len(worker_pids), 'worker_pids': sorted(worker_pids),
                   'flight_point_count': _distribution(flight_points),
                   'segments_per_flight': _distribution(segment_counts),
                   'phase_types_per_flight': _distribution(phase_types),
                   'segment_point_count': _distribution(segment_points),
                   'phase_segment_counts': {phase: phase_segment_counts[phase] for phase in PHASES},
                   'phase_point_counts': {phase: phase_point_counts[phase] for phase in PHASES},
                   'duplicate_timestamp_rows_retained': duplicate_points,
                   'flights_with_fewer_than_four_unique_times': short_flights,
                   'point_conservation_verified': True, 'elapsed_seconds': time.monotonic() - started}
        _write_json(output / 'summary.json', summary)
        logger.info('完成：%d 个可识别航班，%d 个不可识别航班，%d 个阶段片段，%d 个输入点。航班点数=%s；每航班片段数=%s；片段点数=%s',
                    processed, unrecognized_flights, len(segment_points), input_points, summary['flight_point_count'],
                    summary['segments_per_flight'], summary['segment_point_count'])
        return summary
    except Exception as exc:
        logger.exception('阶段识别失败，当前输出不能视为完整结果')
        if output_created:
            try:
                _write_json(output / 'failure.json', {'status': 'failed', 'error': str(exc),
                            'log_file': str(log_path) if log_path is not None else None,
                            'processed_flights': processed, 'submitted_flights': submitted})
            except OSError:
                logger.exception('无法写入结果目录中的 failure.json，请查看当前日志')
        raise
    finally:
        if owns_logger:
            close_project_logging(logger)
