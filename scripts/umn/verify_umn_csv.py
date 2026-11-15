"""Validate exported CSV coverage and representative value-preserving round trips."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import pandas as pd

from convert_umn_to_csv import ROOT, CSV_ROOT, DETAIL_ROOT, build_table, digest_file, output_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--title', action='append')
    parser.add_argument('--full-roundtrip', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'manifest.json').read_text(encoding='utf-8'))
    items = [i for i in manifest['items'] if not args.title or i['title'] in args.title]
    selected = set(args.title or [])
    if not selected:
        # Each format/family plus special archive, duplicate-clock and variant cases.
        families = set()
        for item in items:
            family = item['title'].split(' Flight')[0].split(' Ground')[0]
            if family not in families:
                selected.add(item['title'])
                families.add(family)
        selected |= {'FASER Flight 04', 'Vireo Flight 1', 'Vireo Flight 35', 'Thor Ground 09', 'Thor Flight 75',
                     'FASER Flight 02', 'Loki Flight 45', 'Thor Flight 123', 'Thor Ground 04', 'Tyr Flight 06'}
    results = []
    for item in items:
        path = CSV_ROOT / output_name(item)
        detail = json.loads((DETAIL_ROOT / (item['uuid'] + '.json')).read_text(encoding='utf-8'))
        assert path.exists() and path.stat().st_size == detail['csv_bytes'], item['title']
        assert digest_file(path) == detail['csv_sha256'], item['title']
        with path.open('rb') as stream:
            assert stream.read(3) == b'\xef\xbb\xbf'
        with path.open('r', encoding='utf-8-sig', newline='') as stream:
            header = next(csv.reader(stream))
            rows = sum(1 for _ in stream)
        assert header == detail['column_names'] and len(header) == len(set(header)), item['title']
        assert rows == detail['summary']['rows'], (item['title'], rows)
        roundtrip = args.full_roundtrip or item['title'] in selected
        if roundtrip:
            expected, _ = build_table(item)
            actual = pd.read_csv(path, float_precision='round_trip', dtype={'_record_id': str})
            assert expected.columns.tolist() == actual.columns.tolist() and expected.shape == actual.shape
            for column in expected.columns:
                if pd.api.types.is_numeric_dtype(expected[column]):
                    assert np.array_equal(expected[column].to_numpy(), actual[column].to_numpy(), equal_nan=True), (item['title'], column)
                else:
                    assert expected[column].fillna('').astype(str).equals(actual[column].fillna('').astype(str)), (item['title'], column)
            for stream in detail['streams']:
                recovery = stream.get('source_recovery')
                if not recovery:
                    continue
                source = ROOT / stream['source']['relative_path']
                spec = next(f for f in item['files'] if f['relative_path'] == stream['source']['relative_path'])
                assert hashlib.md5(source.read_bytes()).hexdigest() == spec['checksum']['value']
                # Independent direct-byte check of the three incomplete MAT-v4 tails.
                with source.open('rb') as raw:
                    raw.seek(recovery['variable_header_offset'])
                    mopt, nrows, ncols, imaginary, name_bytes = struct.unpack('>5i', raw.read(20))
                    assert mopt // 1000 == 1 and mopt % 10 == 0 and not imaginary and nrows * ncols == recovery['declared_elements']
                    name = raw.read(name_bytes).rstrip(b'\x00').decode('utf-8')
                    assert name == recovery['field']
                    count = recovery['recovered_elements']
                    dtype = np.dtype('>' + {0: 'f8', 1: 'f4', 2: 'i4', 3: 'i2', 4: 'u2', 5: 'u1'}[(mopt // 10) % 10])
                    raw_values = np.frombuffer(raw.read(count * dtype.itemsize), dtype=dtype)
                prefix = stream.get('prefix', '')
                values = actual[prefix + name].to_numpy()
                missing = actual[prefix + '_source_missing.' + name].to_numpy()
                assert np.array_equal(values[:count], raw_values, equal_nan=True)
                assert np.isnan(values[count:]).all() and not missing[:count].any() and missing[count:].all()
        results.append({'title': item['title'], 'rows': rows, 'columns': len(header), 'full_value_roundtrip': roundtrip})
        if roundtrip:
            print('VALUE ROUNDTRIP PASSED', item['title'], rows, len(header), flush=True)
    expected_paths = {output_name(i) for i in items}
    if not args.title:
        assert expected_paths == {p.name for p in CSV_ROOT.glob('*.csv')}
        assert len(items) == 342
        assert not list(CSV_ROOT.glob('*.part'))
    report = {'verified_records': len(results), 'value_roundtrip_records': sum(r['full_value_roundtrip'] for r in results),
              'all_checks_passed': True, 'records': results}
    (ROOT / ('csv_validation_subset.json' if args.title else 'csv_validation.json')).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('CSV VALIDATION PASSED', len(results), 'records;', report['value_roundtrip_records'], 'full value roundtrips', flush=True)


if __name__ == '__main__':
    main()
