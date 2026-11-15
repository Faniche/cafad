"""Normalize UMN files to the nine ADS-B columns through a synchronized task queue.

No calendar dates are inferred from relative time. Numerical imports happen only
after the parent has set each native numerical library's thread budget to one.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import logging
from logging.handlers import QueueHandler, QueueListener
import math
import multiprocessing as mp
import os
from pathlib import Path
from queue import Empty
import shutil
import time

from src.utils.env import Env
from src.utils.logging_utils import close_project_logging, setup_project_logging
from src.utils.process_control import terminate_workers


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_COLUMNS = ['纬度', '经度', '水平高度', '气压高度', '航向', '速度', 'call号', 'icao号', '时间']
NATIVE_THREAD_KEYS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                     'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')
LOGGER_NAME = 'umn_preprocessing'


@dataclass(frozen=True)
class UMNOptions:
    workers: int | None = None
    chunk_rows: int = 100_000
    pattern: str = '*.csv'
    baro_tolerance_seconds: float = 1.0
    min_track_speed: float = 0.5
    allow_missing_baro: bool = False
    reject_zero_position: bool = True
    merge: bool = True

    def validate(self):
        if self.workers is not None and (isinstance(self.workers, bool) or not isinstance(self.workers, int) or self.workers < 1):
            raise ValueError('workers must be a positive integer')
        if isinstance(self.chunk_rows, bool) or not isinstance(self.chunk_rows, int) or self.chunk_rows < 1:
            raise ValueError('chunk_rows must be positive')
        if not self.pattern or '/' in self.pattern or '\\' in self.pattern or '**' in self.pattern:
            raise ValueError('pattern must select files directly inside the input directory')
        if any(not math.isfinite(v) or v < 0 for v in (self.baro_tolerance_seconds, self.min_track_speed)):
            raise ValueError('Time tolerance and minimum track speed must be finite and nonnegative')


def _schema(columns):
    """Map by documented source fields; never treat IAS or Euler yaw as GPS motion."""
    columns = set(columns)
    if 'gps.time' in columns:
        name = 'vireo_raw_gps'
        fields = dict(time='gps.time', lat='gps.latitude', lon='gps.longitude', alt='gps.altitude_gps',
                      north='gps.Vn', east='gps.Ve', baro='air.altitude_baro', baro_time='air.time')
        position = 'GPS geographic degrees'
        datum = 'source air.altitude_baro; source barometric reference, no datum conversion'
    elif 'dtc.time' in columns:
        name = 'official_synchronized_ekf'
        fields = dict(time='dtc.time', lat='dtc.ekf.lat', lon='dtc.ekf.lon', alt='dtc.ekf.alt_ekf',
                      north='dtc.ekf.Vn', east='dtc.ekf.Ve', baro='dtc.air.pitot.alt_baro')
        position = 'official synchronized EKF estimate in degrees; not a raw GPS fix'
        datum = 'source dtc.air.pitot.alt_baro is AGL metres; not converted to standard pressure altitude'
    else:
        name = 'legacy_gps'
        fields = dict(time='time', lat='lat', lon='lon', alt='alt', north='vn', east='ve', baro='h')
        position = 'GPS geographic degrees'
        datum = 'source h in metres; dictionary describes AGL, no datum conversion'
    missing = [value for value in fields.values() if value not in columns]
    if missing:
        raise ValueError(f'Unsupported/incomplete {name} schema; missing fields: {missing}')
    return {'name': name, 'fields': fields, 'position_source': position, 'barometric_reference': datum}


def _identity(path, source_ids):
    if len(source_ids) > 1:
        raise ValueError(f'One input file must represent one record; found {len(source_ids)} _record_id values')
    record_id = next(iter(source_ids)) if source_ids else hashlib.sha256(path.stem.encode('utf-8')).hexdigest()
    # A namespace avoids collision with six-character real ICAO addresses. A
    # different mission on the same airframe intentionally gets a different ID.
    icao = 'UMN_' + record_id
    call = path.stem.split('__', 1)[0]
    return record_id, call, icao


def process_file(source: Path, destination: Path, provenance: Path, options: UMNOptions,
                 logger: logging.Logger) -> dict:
    """One worker owns the file and all its private output paths."""
    import numpy as np
    import pandas as pd

    started = time.monotonic()
    source = Path(source).resolve()
    header = pd.read_csv(source, nrows=0, encoding='utf-8-sig').columns.tolist()
    schema = _schema(header)
    fields = schema['fields']
    gps_columns = ['time', 'lat', 'lon', 'alt', 'north', 'east', 'baro']
    selected = set(fields.values())
    if '_record_id' in header:
        selected.add('_record_id')
    ids, point_parts, baro_parts = set(), [], []
    source_rows = 0
    # Select only required fields, rather than loading 50-150 raw channels into
    # each of the configured 44 processes. No source file is shared for writing.
    with pd.read_csv(source, usecols=lambda c: c in selected, encoding='utf-8-sig',
                     dtype={'_record_id': 'string'}, chunksize=options.chunk_rows,
                     float_precision='round_trip') as reader:
        for chunk in reader:
            if '_record_id' in chunk:
                ids.update(chunk['_record_id'].dropna().astype(str).str.strip().loc[lambda x: x.ne('')].unique())
            for column in fields.values():
                chunk[column] = pd.to_numeric(chunk[column], errors='raise').astype(np.float64)
            row_numbers = np.arange(source_rows, source_rows + len(chunk), dtype=np.int64)
            points = pd.DataFrame({key: chunk[value].to_numpy() for key, value in fields.items() if key != 'baro_time'})
            points['source_row_index'] = row_numbers
            if 'baro_time' in fields:
                # Only GPS packet rows are location observations in the union of
                # asynchronous streams. Other packet rows are not fake GPS fixes.
                point_parts.append(points.loc[chunk[fields['time']].notna().to_numpy()].copy())
                baro = pd.DataFrame({'baro_time': chunk[fields['baro_time']].to_numpy(),
                                     'baro': chunk[fields['baro']].to_numpy(), 'baro_source_row_index': row_numbers})
                good_baro = np.isfinite(baro['baro_time']) & np.isfinite(baro['baro'])
                baro_parts.append(baro.loc[good_baro].copy())
            else:
                points['baro_source_row_index'] = row_numbers
                points['baro_age_seconds'] = 0.0
                point_parts.append(points)
            source_rows += len(chunk)
    points = pd.concat(point_parts, ignore_index=True) if point_parts else pd.DataFrame(columns=gps_columns + ['source_row_index'])
    if 'baro_time' not in fields and not len(points):
        points['baro_source_row_index'] = np.empty(0, dtype=float)
        points['baro_age_seconds'] = np.empty(0, dtype=float)
    record_id, call, icao = _identity(source, ids)
    candidate_rows = len(points)
    finite_time = np.isfinite(points['time'].to_numpy(dtype=float))
    bad_time = int((~finite_time).sum())
    points = points.loc[finite_time].sort_values(['time', 'source_row_index'], kind='stable').reset_index(drop=True)
    if 'baro_time' in fields:
        baro = pd.concat(baro_parts, ignore_index=True) if baro_parts else pd.DataFrame(columns=['baro_time', 'baro', 'baro_source_row_index'])
        # A causal bounded as-of join is explicit preprocessing, not an unbounded
        # forward fill or interpolation. Ties use the last source observation.
        points = points.drop(columns='baro')
        if len(baro) and len(points):
            baro = baro.sort_values(['baro_time', 'baro_source_row_index'], kind='stable').drop_duplicates('baro_time', keep='last')
            points = pd.merge_asof(points, baro, left_on='time', right_on='baro_time',
                                   direction='backward', tolerance=options.baro_tolerance_seconds)
            points['baro_age_seconds'] = points['time'] - points['baro_time']
        else:
            for column in ('baro', 'baro_time', 'baro_source_row_index', 'baro_age_seconds'):
                points[column] = np.nan
    numeric = points[['lat', 'lon', 'alt', 'north', 'east']].to_numpy(dtype=float)
    bad_physical = ~np.isfinite(numeric).all(axis=1)
    bad_coordinates = ~points['lat'].between(-90, 90) | ~points['lon'].between(-180, 180)
    zero_position = (points['lat'].eq(0) & points['lon'].eq(0)).to_numpy() if options.reject_zero_position else np.zeros(len(points), dtype=bool)
    bad_baro = ~np.isfinite(points['baro'].to_numpy(dtype=float))
    reject = bad_physical | bad_coordinates.to_numpy() | zero_position
    if not options.allow_missing_baro:
        reject |= bad_baro
    quality = {'invalid_time_rows': bad_time, 'nonfinite_gps_motion_rows': int(bad_physical.sum()),
               'out_of_range_coordinate_rows': int(bad_coordinates.sum()), 'zero_position_rows': int(zero_position.sum()),
               'unavailable_baro_rows': int(bad_baro.sum()),
               'rejected_rows': int(reject.sum()) + bad_time,
               'non_position_packet_rows': source_rows - candidate_rows}
    points = points.loc[~reject].reset_index(drop=True)
    speed = np.hypot(points['north'].to_numpy(dtype=float), points['east'].to_numpy(dtype=float))
    heading = np.mod(np.degrees(np.arctan2(points['east'].to_numpy(dtype=float), points['north'].to_numpy(dtype=float))), 360.0)
    heading = np.where(speed >= max(options.min_track_speed, np.finfo(float).tiny), heading, -1.0)
    output = pd.DataFrame({'纬度': points['lat'], '经度': points['lon'], '水平高度': points['alt'],
                           '气压高度': points['baro'], '航向': heading, '速度': speed,
                           'call号': call, 'icao号': icao, '时间': points['time']}, columns=OUTPUT_COLUMNS)
    # ID columns must exist even for a header-only file.
    mapping = pd.DataFrame({'output_row_index': np.arange(len(output), dtype=np.int64),
                            'source_row_index': points['source_row_index'],
                            'baro_source_row_index': points['baro_source_row_index'],
                            'baro_age_seconds': points['baro_age_seconds']})
    destination.parent.mkdir(parents=True, exist_ok=True)
    provenance.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f'.{os.getpid()}.part')
    mapping_path = provenance.with_suffix('.rows.csv')
    mapping_tmp = mapping_path.with_name(mapping_path.name + f'.{os.getpid()}.part')
    try:
        output.to_csv(temporary, index=False, encoding='utf-8-sig', float_format='%.17g', lineterminator='\n')
        mapping.to_csv(mapping_tmp, index=False, encoding='utf-8-sig', float_format='%.17g', lineterminator='\n')
        temporary.replace(destination)
        mapping_tmp.replace(mapping_path)
    finally:
        temporary.unlink(missing_ok=True)
        mapping_tmp.unlink(missing_ok=True)
    result = {'status': 'ok' if len(output) else 'empty', 'input': str(source), 'output': str(destination.resolve()),
              'record_id': record_id, 'call': call, 'icao': icao, 'schema': schema, 'quality': quality,
              'source_rows': source_rows, 'position_candidate_rows': candidate_rows, 'output_rows': len(output),
              'duplicate_output_times': int(output['时间'].duplicated().sum()),
              'time_min': float(output['时间'].min()) if len(output) else None,
              'time_max': float(output['时间'].max()) if len(output) else None,
              'time_semantics': 'relative seconds within this record; not UTC and not cross-flight simultaneity',
              'units': {'latitude_longitude': 'degrees', 'altitude': 'm', 'barometric_altitude': 'source m; datum unchanged',
                        'heading': 'ground track degrees clockwise from north; -1 below the configured speed threshold', 'speed': 'horizontal groundspeed m/s'},
              'options': asdict(options), 'source_size_bytes': source.stat().st_size,
              'source_mtime_ns': source.stat().st_mtime_ns, 'provenance_rows': str(mapping_path.resolve()),
              'worker_pid': os.getpid(), 'seconds': round(time.monotonic() - started, 3)}
    provenance.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    log = logger.info if len(output) else logger.warning
    log('处理完成：%s；输入=%d，位置候选=%d，输出=%d，排除=%d；schema=%s，pid=%d',
        source.name, source_rows, candidate_rows, len(output), quality['rejected_rows'], schema['name'], os.getpid())
    return result


def _worker(tasks, results, logs, options):
    logger = logging.getLogger(f'{LOGGER_NAME}.worker.{os.getpid()}')
    logger.handlers.clear()
    handler = QueueHandler(logs)
    logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(logging.INFO)
    warning_logger = logging.getLogger('py.warnings')
    warning_logger.handlers[:] = [handler]
    warning_logger.propagate = False
    logging.captureWarnings(True)
    try:
        logger.info('工作进程启动：pid=%d，内部数值线程数=1', os.getpid())
        while True:
            # multiprocessing.Queue.get() already owns its interprocess lock.
            # A sentinel follows every real task. Never rely on empty()/qsize().
            task = tasks.get()
            if task is None:
                logger.info('工作进程结束：pid=%d，已领取全部可用任务', os.getpid())
                return
            ordinal, source, output, provenance = task
            try:
                logger.info('领取文件任务：index=%d，file=%s，pid=%d', ordinal, Path(source).name, os.getpid())
                result = process_file(Path(source), Path(output), Path(provenance), options, logger)
            except Exception as exc:
                logger.exception('文件处理失败，继续领取后续任务：%s', source)
                result = {'status': 'failed', 'input': source, 'worker_pid': os.getpid(), 'error': f'{type(exc).__name__}: {exc}'}
            results.put({'ordinal': ordinal, **result})
    except BaseException:
        logger.exception('工作进程异常退出：pid=%d', os.getpid())
        raise SystemExit(1)
    finally:
        logging.captureWarnings(False)
        logger.handlers.clear()
        handler.close()
        # Flush each child's feeder before exit; the parent still owns its ends.
        results.close()
        results.join_thread()
        logs.close()
        logs.join_thread()


def _merge_outputs(results, output):
    target = output / 'preprocessed_data_umn.csv'
    temporary = target.with_suffix('.csv.part')
    with temporary.open('wb') as writer:
        writer.write(b'\xef\xbb\xbf')
        writer.write((','.join(OUTPUT_COLUMNS) + '\n').encode('utf-8'))
        for result in results:
            with Path(result['output']).open('rb') as reader:
                reader.readline()  # Per-file BOM and header are not repeated.
                shutil.copyfileobj(reader, writer, length=1024 * 1024)
    temporary.replace(target)
    return str(target.resolve())


def run(input_dir: str | Path, output_dir: str | Path, options: UMNOptions | None = None,
        *, logger: logging.Logger | None = None) -> dict:
    """Queue every file once; idle workers claim another until their sentinel."""
    options = options or UMNOptions()
    owned_logger = logger is None
    if logger is None:
        logger, _ = setup_project_logging(console=False, name=LOGGER_NAME)
    processes, queues, listener = [], [], None
    output = Path(output_dir).resolve()
    created = False
    file_results = []
    try:
        options.validate()
        source = Path(input_dir).resolve()
        if not source.is_dir():
            raise ValueError(f'Input directory does not exist: {source}')
        files = sorted((p for p in source.glob(options.pattern) if p.is_file()), key=lambda p: p.name)
        if not files:
            raise ValueError(f'No files match {options.pattern!r} in {source}')
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError('Input and output directories must be disjoint')
        if output.exists():
            raise FileExistsError(f'Output directory already exists: {output}; use a new output directory')
        if options.workers is None:
            configured = Env.data_process_workers()
        else:
            configured = options.workers
        workers = min(configured, len(files))
        output.mkdir(parents=True)
        created = True
        (output / 'files').mkdir()
        (output / 'provenance').mkdir()
        for key in NATIVE_THREAD_KEYS:
            os.environ[key] = '1'
        configuration = {'input_dir': str(source), 'output_dir': str(output), 'input_files': [p.name for p in files],
                         'options': asdict(options), 'configured_workers': configured, 'actual_workers': workers,
                         'output_columns': OUTPUT_COLUMNS, 'time_semantics': 'relative_seconds',
                         'log_file': str(getattr(logger, 'project_log_path', '')),
                         'queue_policy': 'atomic multiprocessing.Queue.get; one sentinel per worker; no empty/qsize polling'}
        (output / 'configuration.json').write_text(json.dumps(configuration, ensure_ascii=False, indent=2), encoding='utf-8')
        logger.info('开始 UMN 格式转换：文件=%d，配置进程=%d，实际进程=%d，时间=相对秒', len(files), configured, workers)
        logger.warning('气压高度保留源字段及其参考面（部分为 AGL），不声称已换算为 ADS-B 标准气压高度')
        context = mp.get_context('spawn')
        tasks, results, logs = context.Queue(), context.Queue(), context.Queue()
        queues = [tasks, results, logs]
        listener = QueueListener(logs, *logger.handlers, respect_handler_level=True)
        listener.start()
        for ordinal, path in enumerate(files):
            tasks.put((ordinal, str(path), str(output / 'files' / path.name), str(output / 'provenance' / (path.stem + '.json'))))
        for _ in range(workers):
            tasks.put(None)
        for _ in range(workers):
            process = context.Process(target=_worker, args=(tasks, results, logs, options), name='umn-file-worker')
            process.start()
            processes.append(process)
        while len(file_results) < len(files):
            try:
                result = results.get(timeout=0.5)
            except Empty:
                crashed = [(p.pid, p.exitcode) for p in processes if p.exitcode not in (None, 0)]
                if crashed or all(p.exitcode is not None for p in processes):
                    raise RuntimeError(f'Workers exited before all results arrived: {crashed}; received {len(file_results)}/{len(files)}')
                continue
            file_results.append(result)
            logger.info('总进度：%d/%d；文件=%s；状态=%s', len(file_results), len(files), Path(result['input']).name, result['status'])
        for process in processes:
            process.join(timeout=10)
            if process.is_alive() or process.exitcode != 0:
                raise RuntimeError(f'Worker did not finish normally: pid={process.pid}, exitcode={process.exitcode}')
        file_results.sort(key=lambda r: r['ordinal'])
        failures = [r for r in file_results if r['status'] == 'failed']
        successful = [r for r in file_results if r['status'] != 'failed']
        merged = _merge_outputs(successful, output) if options.merge and not failures else None
        summary = {'configured_workers': configured, 'actual_workers': workers,
                   'worker_pids': [p.pid for p in processes], 'input_files': len(files),
                   'successful_files': len(successful), 'empty_files': sum(r['status'] == 'empty' for r in successful),
                   'failed_files': len(failures), 'output_rows': sum(r['output_rows'] for r in successful),
                   'merged_csv': merged, 'time_semantics': 'relative_seconds',
                   'log_file': str(getattr(logger, 'project_log_path', '')), 'files': file_results}
        (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        if failures:
            raise RuntimeError(f'{len(failures)} UMN files failed; successful per-file outputs are retained; combined CSV was not published')
        logger.info('UMN 处理完成：%d 个文件，%d 个观测点，空文件=%d，合并文件=%s', len(files), summary['output_rows'], summary['empty_files'], merged)
        return summary
    except BaseException as exc:
        logger.exception('UMN 数据处理执行失败')
        if created:
            (output / 'failure.json').write_text(json.dumps({'error': f'{type(exc).__name__}: {exc}',
                'completed_results': file_results, 'log_file': str(getattr(logger, 'project_log_path', ''))}, ensure_ascii=False, indent=2), encoding='utf-8')
        raise
    finally:
        terminate_workers(processes)
        if listener is not None:
            listener.stop()
        for queue in queues:
            # Cancel the parent feeder only after processing/termination. On a
            # crash, no worker remains to drain pending task bytes.
            queue.cancel_join_thread()
            queue.close()
        if owned_logger:
            close_project_logging(logger)
