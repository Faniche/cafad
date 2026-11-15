"""Export one faithful CSV per public UMN record using the project venv.

Direct HDF5/MAT alternatives are not appended as new observations. Vireo's
asynchronous raw sensor streams are joined on exact time plus duplicate occurrence;
there is no interpolation, filtering, coordinate conversion, or point deletion.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import struct
import sys
import time
import traceback
import zipfile

import h5py
import numpy as np
import pandas as pd
import scipy
from scipy.io import loadmat, whosmat
from scipy.io.matlab import matfile_version

from download_umn_uav import ROOT, safe_name

VERSION = 1
CSV_ROOT = ROOT / 'csv'
DETAIL_ROOT = ROOT / 'conversion_metadata'


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return {'nonfinite': str(value)}
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='replace')
    if isinstance(value, complex):
        return {'real': value.real, 'imag': value.imag}
    return value


def digest_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_hdf(source):
    values = {}
    with h5py.File(source, 'r') as handle:
        def visit(name, value):
            if isinstance(value, h5py.Dataset):
                if value.dtype.kind == 'O':
                    raise ValueError(f'Unsupported HDF5 reference/object dataset: {name}')
                values[name.replace('/', '.')] = value[...]
        handle.visititems(visit)
    return values


def read_source(candidate):
    path = ROOT / candidate['relative_path']
    if candidate.get('member'):
        with zipfile.ZipFile(path) as archive:
            stream = io.BytesIO(archive.read(candidate['member']))
            return read_hdf(stream) if candidate['extension'] in ('.h5', '.hdf5') else loadmat(stream, simplify_cells=True)
    if candidate['extension'] in ('.h5', '.hdf5'):
        return read_hdf(path)
    try:
        return loadmat(path, simplify_cells=True)
    except ValueError as exc:
        if 'Not enough bytes to read matrix' not in str(exc) or matfile_version(path)[0] != 0:
            raise
        return recover_final_mat4_array(path, str(exc))


def recover_final_mat4_array(path, original_error):
    """Recover only bytes actually present in a truncated final MAT-v4 vector.

    Declared but absent elements become NaN and get an explicit source-missing mask.
    Never rewrite the downloaded source or infer any missing measurement value.
    """
    variables = whosmat(path)
    size = path.stat().st_size
    with path.open('rb') as stream:
        header = stream.read(20)
        big = struct.unpack('>5i', header)
        little = struct.unpack('<5i', header)
        if 1000 <= big[0] < 2000:
            endian = '>'
        elif 0 <= little[0] < 1000:
            endian = '<'
        else:
            raise ValueError('Unsupported MAT-v4 machine format')
        stream.seek(0)
        while stream.tell() < size:
            header_position = stream.tell()
            header = stream.read(20)
            if len(header) != 20:
                raise ValueError('Truncated MAT-v4 variable header')
            mopt, rows, cols, imag, name_length = struct.unpack(endian + '5i', header)
            if rows < 0 or cols < 0 or not 1 <= name_length <= 10000:
                raise ValueError('Invalid MAT-v4 dimensions or variable name')
            name = stream.read(name_length).rstrip(b'\x00').decode('utf-8', errors='strict')
            precision = (mopt // 10) % 10
            kind = mopt % 10
            dtype = np.dtype(endian + {0: 'f8', 1: 'f4', 2: 'i4', 3: 'i2', 4: 'u2', 5: 'u1'}[precision])
            count = rows * cols
            payload = count * dtype.itemsize * (2 if imag else 1)
            available_bytes = size - stream.tell()
            if available_bytes >= payload:
                stream.seek(payload, 1)
                continue
            if kind != 0 or imag or name != variables[-1][0] or min(rows, cols) != 1:
                raise ValueError('Only a truncated final numeric real vector can be recovered safely')
            available_count = available_bytes // dtype.itemsize
            values = np.full(count, np.nan, dtype=np.float64)
            values[:available_count] = np.frombuffer(stream.read(available_count * dtype.itemsize), dtype=dtype)
            good_names = [entry[0] for entry in variables if entry[0] != name]
            result = loadmat(path, simplify_cells=True, variable_names=good_names)
            result[name] = values.reshape((rows, cols), order='F').squeeze()
            result['_source_missing.' + name] = np.arange(count) >= available_count
            result['__umn_recovery__'] = {'kind': 'source_truncated_final_mat4_vector', 'field': name,
                'declared_elements': count, 'recovered_elements': available_count,
                'missing_elements': count - available_count, 'trailing_incomplete_bytes': available_bytes % dtype.itemsize,
                'variable_header_offset': header_position, 'original_error': original_error,
                'policy': 'preserve available prefix exactly; absent tail is NaN with an explicit _source_missing mask; original file unchanged'}
            return result
    raise ValueError('No safely recoverable truncated final vector found')


def candidates_for(item):
    # Vireo alldata.mat contains the original asynchronous *.raw streams;
    # the standalone official dtc MAT is a filtered/interpolated alternative.
    if item['title'].startswith('Vireo '):
        for spec in item['files']:
            if spec['name'].lower().endswith('.zip'):
                with zipfile.ZipFile(ROOT / spec['relative_path']) as archive:
                    raw_member = next((n for n in archive.namelist() if PurePosixPath(n).name.lower() == 'alldata.mat'), None)
                if raw_member:
                    return [{'relative_path': spec['relative_path'], 'member': raw_member,
                             'extension': '.mat', 'logical_stem': 'alldata', 'raw_multirate': True}]
    options = []
    for spec in item['files']:
        extension = Path(spec['name']).suffix.lower()
        if extension in ('.h5', '.hdf5', '.mat'):
            options.append({'relative_path': spec['relative_path'], 'extension': extension,
                            'logical_stem': Path(spec['name']).stem.lower()})
    if not options:
        for spec in item['files']:
            if not spec['name'].lower().endswith('.zip'):
                continue
            with zipfile.ZipFile(ROOT / spec['relative_path']) as archive:
                for info in archive.infolist():
                    member = PurePosixPath(info.filename)
                    extension = member.suffix.lower()
                    if info.is_dir() or '__MACOSX' in member.parts or extension not in ('.h5', '.hdf5', '.mat'):
                        continue
                    options.append({'relative_path': spec['relative_path'], 'member': info.filename,
                                    'extension': extension, 'logical_stem': str(member.with_suffix('')).lower()})
    # One representation per logical data source; HDF5 is preferred over MAT.
    grouped = {}
    rank = {'.h5': 0, '.hdf5': 0, '.mat': 1}
    for option in options:
        grouped.setdefault(option['logical_stem'], []).append(option)
    selected = []
    for group in grouped.values():
        group = sorted(group, key=lambda c: rank[c['extension']])
        selected.append({**group[0], 'alternatives': group[1:]})
    return sorted(selected, key=lambda c: ('navrepaired' in c['logical_stem'], c['logical_stem']))


def flatten(value, prefix='', leaves=None):
    leaves = {} if leaves is None else leaves
    if isinstance(value, dict):
        for key, child in value.items():
            if not key.startswith('__'):
                flatten(child, f'{prefix}.{key}' if prefix else key, leaves)
    else:
        leaves[prefix] = value
    return leaves


def table_from_leaves(leaves, prefix=''):
    time_keys = [key for key in leaves if key.lower() == 'time']
    if not time_keys:
        time_keys = [key for key in leaves if key.lower().endswith('.time')]
    if len(time_keys) != 1:
        raise ValueError(f'Expected one native clock, found {time_keys}')
    clock_key = time_keys[0]
    clock = np.asarray(leaves[clock_key]).squeeze()
    if clock.ndim != 1 or clock.size < 2 or clock.dtype.kind not in 'biuf':
        raise ValueError(f'Invalid native clock {clock_key}: {clock.shape} {clock.dtype}')
    count = len(clock)
    columns = {}
    metadata = {}
    origins = {}
    for key, value in leaves.items():
        array = np.asarray(value)
        array = np.squeeze(array)
        if not array.dtype.isnative and array.dtype.kind in 'biufc':
            array = array.astype(array.dtype.newbyteorder('='))
        if array.ndim == 1 and len(array) == count and array.dtype.kind in 'biufcUS':
            if array.dtype.kind == 'c':
                columns[prefix + key + '.real'] = array.real
                columns[prefix + key + '.imag'] = array.imag
                origins[prefix + key + '.real'] = key
                origins[prefix + key + '.imag'] = key
            else:
                columns[prefix + key] = array
                origins[prefix + key] = key
        elif array.ndim >= 2 and count in array.shape and array.dtype.kind in 'biuf':
            axes = [axis for axis, size in enumerate(array.shape) if size == count]
            if len(axes) != 1:
                raise ValueError(f'Ambiguous sample axis: {key} {array.shape}')
            reshaped = np.moveaxis(array, axes[0], 0).reshape(count, -1)
            for component in range(reshaped.shape[1]):
                name = prefix + key + f'[{component}]'
                columns[name] = reshaped[:, component]
                origins[name] = key
        else:
            # Preserve constants, units, empty arrays and ancillary arrays in JSON.
            # They are not padded/truncated/broadcast into a fabricated time series.
            metadata[key] = {'shape': list(array.shape), 'dtype': str(array.dtype), 'value': json_safe(value)}
    frame = pd.DataFrame(columns)
    if len(frame) != count:
        raise ValueError('Table length differs from native clock')
    frame[prefix + '_sample_index'] = np.arange(count, dtype=np.int64)
    clock = np.asarray(clock, dtype=np.float64)
    if not np.isfinite(clock).all():
        raise ValueError('Clock contains nonfinite values; explicit handling required')
    stats = {'native_rows': count, 'time_min': float(clock.min()), 'time_max': float(clock.max()),
             'duplicate_times': int(pd.Series(clock).duplicated().sum()),
             'negative_steps': int((np.diff(clock) < 0).sum()),
             'clock_source': clock_key, 'column_origins': origins, 'ancillary_metadata': metadata}
    return frame, clock, stats


def exact_time_index(clock):
    occurrence = pd.Series(clock).groupby(clock, sort=False).cumcount().to_numpy()
    return pd.MultiIndex.from_arrays([clock, occurrence], names=['_time_s', '_time_occurrence'])


def merge_streams(streams):
    frames, details = [], []
    for name, leaves in streams:
        frame, clock, info = table_from_leaves(leaves, prefix=name + '.')
        frame.index = exact_time_index(clock)
        frames.append(frame)
        details.append({'stream': name, **info})
    frame = pd.concat(frames, axis=1, join='outer', sort=False).sort_index(kind='stable')
    frame = frame.reset_index()
    return frame, details


def build_table(item):
    candidates = candidates_for(item)
    if not candidates:
        raise ValueError('No numeric HDF5/MAT data source found')
    source_details, metadata, used_candidates = [], {}, []
    if candidates[0].get('raw_multirate'):
        data = read_source(candidates[0])['data']
        streams = [(name, flatten(value['raw'])) for name, value in data.items()
                   if isinstance(value, dict) and isinstance(value.get('raw'), dict) and 'time' in value['raw']]
        frame, source_details = merge_streams(streams)
        metadata = {k: json_safe(v) for k, v in data.items() if not isinstance(v, dict) or 'raw' not in v}
        selection = 'ZIP alldata.mat raw sensor streams; exact-time outer join, no interpolation; official filtered/cmntime/dtc alternatives remain in original files'
        used_candidates = candidates
    else:
        tables = []
        for index, candidate in enumerate(candidates):
            prefix = '' if index == 0 else ('NAVREPAIRED.' if 'navrepaired' in candidate['logical_stem'] else f'variant_{index + 1}.')
            failures = []
            for option in [candidate] + candidate.get('alternatives', []):
                try:
                    data = read_source(option)
                    if 'flight_data' in data and isinstance(data['flight_data'], dict):
                        leaves = flatten(data['flight_data'])
                        ancillary = {k: json_safe(v) for k, v in data.items() if k != 'flight_data' and not k.startswith('__')}
                    else:
                        leaves = flatten({k: v for k, v in data.items() if not k.startswith('__')})
                        ancillary = {}
                    table, clock, info = table_from_leaves(leaves, prefix)
                    break
                except Exception as exc:
                    failures.append({'source': option['relative_path'], 'member': option.get('member'), 'error': repr(exc)})
            else:
                raise ValueError(f'All source representations failed: {failures}')
            table.index = exact_time_index(clock)
            tables.append(table)
            source_details.append({'source': option, 'prefix': prefix, 'fallback_reasons': failures,
                                   'source_recovery': data.get('__umn_recovery__'), **info})
            used_candidates.append(option)
            metadata[f'source_{index}'] = ancillary
        if len(tables) == 1:
            # Preserve native row order, including duplicate timestamps and resets.
            frame = tables[0].reset_index(drop=True)
        else:
            frame = pd.concat(tables, axis=1, join='outer', sort=False).sort_index(kind='stable').reset_index()
        selection = 'HDF5 preferred over duplicate MAT representation; fall back to matching MAT if HDF5 lacks a valid clock; alternative variants are separate columns, not additional flight samples'
    frame.insert(0, '_row_index', np.arange(len(frame), dtype=np.int64))
    frame.insert(0, '_record_id', item['uuid'])
    assert frame.columns.is_unique
    if frame.empty:
        raise ValueError('Empty CSV table')
    return frame, {'sources': used_candidates, 'selection_policy': selection, 'streams': source_details,
                   'record_metadata': item.get('metadata', {}), 'additional_metadata': metadata}


def output_name(item):
    return safe_name(item['title']).replace(' ', '_') + '__' + item['uuid'][:8] + '.csv'


def convert_item(item, force=False):
    started = time.monotonic()
    output = CSV_ROOT / output_name(item)
    metadata_path = DETAIL_ROOT / (item['uuid'] + '.json')
    signature = [(f['uuid'], f['size_bytes'], f['checksum']['value']) for f in item['files']]
    if not force and output.exists() and metadata_path.exists():
        previous = json.loads(metadata_path.read_text(encoding='utf-8'))
        if previous.get('converter_version') == VERSION and previous.get('input_signature') == json_safe(signature):
            if output.stat().st_size == previous['csv_bytes'] and digest_file(output) == previous['csv_sha256']:
                return {**previous['summary'], 'cached': True}
    frame, detail = build_table(item)
    temporary = output.with_suffix('.csv.part')
    frame.to_csv(temporary, index=False, encoding='utf-8-sig', float_format='%.17g', chunksize=10_000)
    temporary.replace(output)
    summary = {'record_id': item['uuid'], 'title': item['title'],
               'aircraft': item['title'].split(' Flight')[0].split(' Ground')[0],
               'record_type': 'ground' if ' Ground ' in item['title'] else 'flight',
               'csv_file': output.relative_to(ROOT).as_posix(), 'rows': len(frame), 'columns': len(frame.columns),
               'streams': len(detail['streams']), 'native_stream_rows_sum': sum(s['native_rows'] for s in detail['streams']),
               'source_fallbacks': [s['fallback_reasons'] for s in detail['streams'] if s.get('fallback_reasons')],
               'source_recoveries': [s['source_recovery'] for s in detail['streams'] if s.get('source_recovery')],
               'csv_bytes': output.stat().st_size, 'seconds': round(time.monotonic() - started, 2)}
    document = {'converter_version': VERSION, 'input_signature': json_safe(signature), 'summary': summary,
                'csv_bytes': output.stat().st_size, 'csv_sha256': digest_file(output),
                'column_names': frame.columns.tolist(), 'source_item_url': item['item_url'], **detail}
    metadata_path.write_text(json.dumps(json_safe(document), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return summary


def write_summary(results, errors, expected):
    results = sorted(results, key=lambda r: (r['title'], r['record_id']))
    report = {'python_executable': sys.executable, 'converter_version': VERSION,
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'packages': {'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__, 'h5py': h5py.__version__},
              'expected_records': expected, 'converted_records': len(results), 'errors': errors,
              'total_csv_rows': sum(r['rows'] for r in results), 'total_csv_bytes': sum(r['csv_bytes'] for r in results),
              'records': results}
    (ROOT / 'conversion_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# UMN 每条记录一个 CSV', '',
             f'成功转换 {len(results)}/{expected} 条记录；错误 {len(errors)} 条；总行数 {report["total_csv_rows"]:,}；CSV 总大小 {report["total_csv_bytes"] / 1024**3:.3f} GiB。', '',
             f'转换解释器：`{sys.executable}`。', '',
             '- csv/ 每条官方记录一个 UTF-8 BOM CSV；文件名保留记录名和 UUID 前8位。12条 Ground 测试仍单独保留。',
             '- 所有原始文件保存在 raw/；未去重、删点、补零、插值、滤波、换算单位或注入异常。源数据中的零值、NaN、重复时间及时间倒退保留。',
             '- HDF5/MAT 对应同一逻辑数据源时优先 HDF5；HDF5 缺少有效时间时改用同名 MAT，并记录原因，不将两种格式拼成两倍观测。',
             '- FASER Flight 04 的 NAVREPAIRED 版本保留为 NAVREPAIRED.* 列；它与原始版是同次记录的两个版本。',
             '- Vireo 选择 ZIP 内 alldata.mat 的 *.raw 原始传感器记录。各传感器按精确时间戳及同一时间的出现序号外连接；未观测的字段留空，不前向填充。其 CSV 行数是联合时间轴长度，不等于 GPS 点数。',
             '- _record_id 为官方记录 UUID；_row_index 为导出行号；*_sample_index 保留每个原始流中的行号。其他字段保留源名称，嵌套字段用点号展开。',
             '- 非时序的单位、常量、空数组和记录说明保存在 conversion_metadata/；每个 CSV 的来源、字段、SHA-256、原始流点数也在该目录。',
             '- 原始时间按数据字典为秒，未强行解释为 UTC 日期；GPS高度、气压高度、空速、地速、姿态角等保持原始字段含义。', '',
             '复现：在项目根目录运行 `.venv\\Scripts\\python.exe -B scripts\\umn\\convert_umn_to_csv.py --workers 4`。', '',
             '| 记录 | 类型 | CSV行数 | 列数 | 原始流数 | 大小MiB |', '|---|---|---:|---:|---:|---:|']
    for r in results:
        lines.append(f'| {r["title"]} | {r["record_type"]} | {r["rows"]:,} | {r["columns"]} | {r["streams"]} | {r["csv_bytes"] / 1024**2:.2f} |')
    recovered = [r for r in results if r.get('source_recoveries')]
    fallback = [r for r in results if r.get('source_fallbacks')]
    if recovered or fallback:
        lines += ['', '## 原始文件异常及处理', '']
        for r in fallback:
            lines.append(f'- {r["title"]}：首选 HDF5 时间字段无效或缺失，已采用同名 MAT；完整原因保存在该记录的 conversion_metadata JSON。')
        for r in recovered:
            for recovery in r['source_recoveries']:
                lines.append(f'- {r["title"]}：官方 MAT 最后一个字段 `{recovery["field"]}` 的数据在源文件尾部截断。保留可读取的 {recovery["recovered_elements"]:,} 个值，缺失尾部 {recovery["missing_elements"]:,} 个值留空，并用 `_source_missing.{recovery["field"]}` 标记；其他字段完整保留。原始文件及官方 MD5 未改变。')
    if errors:
        lines += ['', '## 未完成记录', '', '```json', json.dumps(errors, ensure_ascii=False, indent=2), '```']
    (ROOT / 'CSV_README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--title', action='append', help='Convert only exact matching record titles (repeatable).')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    CSV_ROOT.mkdir(parents=True, exist_ok=True)
    DETAIL_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / 'manifest.json').read_text(encoding='utf-8'))
    items = [i for i in manifest['items'] if not args.title or i['title'] in args.title]
    if not items:
        raise ValueError('No records match requested titles')
    results, errors = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(convert_item, item, args.force): item for item in items}
        for future in as_completed(futures):
            item = futures[future]
            try:
                result = future.result()
                results.append(result)
                print('CONVERTED', len(results) + len(errors), '/', len(items), json.dumps(result, ensure_ascii=False), flush=True)
            except Exception as exc:
                error = {'record_id': item['uuid'], 'title': item['title'], 'error': repr(exc), 'traceback': traceback.format_exc()}
                errors.append(error)
                print('CONVERSION ERROR', item['title'], repr(exc), flush=True)
    report = write_summary(results, errors, len(items))
    print('FINISHED', {k: v for k, v in report.items() if k != 'records'}, flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
