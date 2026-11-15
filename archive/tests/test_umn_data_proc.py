"""UMN field semantics, causal alignment, real queue workers and CLI logging."""
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import replace
import io
import json
import logging
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.data_processing.umn_data_proc import OUTPUT_COLUMNS, UMNOptions, process_file, run
from src.utils.logging_utils import close_project_logging, setup_project_logging


ROOT = Path(__file__).resolve().parents[1]
LOG_PREFIX = re.compile(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[+-]\d{4} [\w.]+\.py:\d+ (INFO|WARNING|ERROR) ')


def native_fixture(path, record='000123'):
    # IAS and Euler yaw deliberately disagree with GPS speed/course. Duplicated
    # times and a legitimate negative altitude must not be discarded as outliers.
    frame = pd.DataFrame({'_record_id': [record] * 5, 'time': [0., 2., 1., 1., 3.],
        'lat': [0., 45., 45., 45., 45.], 'lon': [0., -93., -93., -93., -93.],
        'alt': [0., -4., 102., 103., 104.], 'h': [0., 9., 10., 11., 12.],
        'vn': [0., 0., 3., 0., -3.], 've': [0., 0., 4., -5., -4.],
        'ias': [999.] * 5, 'psi': [.25] * 5})
    frame.to_csv(path, index=False)
    return frame


class UMNFileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.logger = logging.getLogger(f'umn_fixture_{id(self)}')
        self.logger.handlers[:] = [logging.NullHandler()]
        self.logger.propagate = False

    def tearDown(self):
        self.temp.cleanup()

    def convert(self, path, options=None):
        output = self.base / 'out' / path.name
        provenance = self.base / 'provenance' / (path.stem + '.json')
        result = process_file(path, output, provenance, options or UMNOptions(workers=1, chunk_rows=2), self.logger)
        return result, pd.read_csv(output, dtype={'icao号': str, 'call号': str}, float_precision='round_trip')

    def test_physical_fields_course_wrap_relative_time_and_exact_source_rows(self):
        source = self.base / 'Flight01.csv'
        native_fixture(source)
        result, frame = self.convert(source)
        self.assertEqual(frame.columns.tolist(), OUTPUT_COLUMNS)
        np.testing.assert_array_equal(frame['时间'], [1., 1., 2., 3.])
        np.testing.assert_allclose(frame['速度'], [5., 5., 0., 5.])
        np.testing.assert_allclose(frame['航向'], [53.13010235415598, 270., -1., 233.13010235415598])
        self.assertEqual(frame['水平高度'].tolist(), [102., 103., -4., 104.])
        self.assertEqual(frame['icao号'].unique().tolist(), ['UMN_000123'])
        self.assertEqual(result['duplicate_output_times'], 1)
        self.assertEqual(result['quality']['zero_position_rows'], 1)
        mapping = pd.read_csv(result['provenance_rows'])
        self.assertEqual(mapping.source_row_index.tolist(), [2, 3, 1, 4])

    def test_vireo_causal_bounded_alignment_across_input_chunks(self):
        source = self.base / 'Vireo.csv'
        # Future 1.1-second baro must not leak into the 1.0-second GPS fix;
        # the 4-second fix has no sufficiently recent baro and remains missing.
        pd.DataFrame({'_record_id': ['v'] * 5,
            'gps.time': [np.nan, 1., np.nan, 1.2, 4.],
            'gps.latitude': [np.nan, 45., np.nan, 45., 45.],
            'gps.longitude': [np.nan, -93., np.nan, -93., -93.],
            'gps.altitude_gps': [np.nan, 100., np.nan, 101., 102.],
            'gps.Vn': [np.nan, 3., np.nan, 3., 3.], 'gps.Ve': [np.nan, 4., np.nan, 4., 4.],
            'air.time': [.8, np.nan, 1.1, np.nan, np.nan],
            'air.altitude_baro': [10., np.nan, 20., np.nan, np.nan]}).to_csv(source, index=False)
        result, frame = self.convert(source, UMNOptions(workers=1, chunk_rows=1, baro_tolerance_seconds=.5, allow_missing_baro=True))
        self.assertEqual(len(frame), 3)
        np.testing.assert_allclose(frame['气压高度'].iloc[:2], [10., 20.])
        self.assertTrue(pd.isna(frame['气压高度'].iloc[2]))
        self.assertEqual(result['quality']['non_position_packet_rows'], 2)
        self.assertEqual(result['quality']['unavailable_baro_rows'], 1)

    def test_official_ekf_schema_is_explicit_and_airspeed_is_not_substituted(self):
        source = self.base / 'Baldr.csv'
        pd.DataFrame({'dtc.time': [10., 11.], 'dtc.ekf.lat': [45., 45.], 'dtc.ekf.lon': [-93., -93.],
            'dtc.ekf.alt_ekf': [100., 101.], 'dtc.ekf.Vn': [0., 3.], 'dtc.ekf.Ve': [4., 4.],
            'dtc.air.pitot.alt_baro': [2., 3.], 'dtc.air.pitot.airspeed': [999., 999.]}).to_csv(source, index=False)
        result, frame = self.convert(source)
        self.assertEqual(result['schema']['name'], 'official_synchronized_ekf')
        self.assertEqual(frame['速度'].tolist(), [4., 5.])
        self.assertEqual(frame['气压高度'].tolist(), [2., 3.])

    def test_invalid_physical_rows_and_empty_results_have_an_honest_header(self):
        source = self.base / 'Empty.csv'
        frame = native_fixture(source)
        frame['lat'] = 100.
        frame.to_csv(source, index=False)
        result, output = self.convert(source)
        self.assertEqual(result['status'], 'empty')
        self.assertEqual(output.columns.tolist(), OUTPUT_COLUMNS)
        self.assertEqual(len(output), 0)

    def test_mixed_record_ids_and_invalid_budgets_are_rejected(self):
        source = self.base / 'Mixed.csv'
        frame = native_fixture(source)
        frame.loc[1, '_record_id'] = 'other'
        frame.to_csv(source, index=False)
        with self.assertRaisesRegex(ValueError, 'one record'):
            self.convert(source)
        for options in [UMNOptions(workers=0), UMNOptions(workers=1.5), UMNOptions(chunk_rows=0),
                        UMNOptions(baro_tolerance_seconds=-1), UMNOptions(min_track_speed=float('nan'))]:
            with self.assertRaises(ValueError):
                options.validate()


class UMNQueueTests(unittest.TestCase):
    def test_atomic_file_queue_exactly_once_merge_and_worker_log_locations(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / 'input'
            source.mkdir()
            for index in range(6):
                native_fixture(source / f'flight_{index}.csv', record=f'000{index}')
            captured = io.StringIO()
            with redirect_stdout(captured), redirect_stderr(captured):
                logger, log = setup_project_logging(console=False, project_root=base, name='umn_queue_test')
                try:
                    result = run(source, base / 'output', UMNOptions(workers=2, chunk_rows=2), logger=logger)
                finally:
                    close_project_logging(logger)
            self.assertEqual(captured.getvalue(), '')
            self.assertEqual(len(result['worker_pids']), 2)
            self.assertEqual(result['successful_files'], 6)
            self.assertEqual(result['output_rows'], 24)
            self.assertEqual(len({f['input'] for f in result['files']}), 6)
            merged = pd.read_csv(result['merged_csv'], dtype={'icao号': str})
            self.assertEqual(merged.columns.tolist(), OUTPUT_COLUMNS)
            self.assertEqual(len(merged), 24)
            self.assertEqual(merged['icao号'].nunique(), 6)
            lines = log.read_text(encoding='utf-8').splitlines()
            self.assertTrue(all(LOG_PREFIX.match(line) for line in lines))
            self.assertTrue(any('umn_data_proc.py:' in line and '领取文件任务' in line for line in lines))
            self.assertFalse(list((base / 'output').rglob('*.part')))
            with self.assertRaises(FileExistsError):
                run(source, base / 'output', UMNOptions(workers=2))

    def test_bad_file_does_not_abandon_remaining_queue_tasks_or_publish_partial_merge(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / 'input'
            source.mkdir()
            pd.DataFrame({'bad': [1]}).to_csv(source / 'a_bad.csv', index=False)
            native_fixture(source / 'b_good.csv')
            with self.assertRaisesRegex(RuntimeError, '1 UMN files failed'):
                run(source, base / 'out', UMNOptions(workers=2))
            summary = json.loads((base / 'out/summary.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['successful_files'], 1)
            self.assertEqual(summary['failed_files'], 1)
            self.assertTrue((base / 'out/files/b_good.csv').exists())
            self.assertFalse((base / 'out/preprocessed_data_umn.csv').exists())
            self.assertTrue((base / 'out/failure.json').exists())

    def test_legacy_umn_code_has_no_second_formal_entry_point(self):
        result = subprocess.run([sys.executable, '-B', str(ROOT / 'main.py'), '-m', 'preprocess', '--help'],
                                cwd=ROOT, capture_output=True, text=True, encoding='utf-8', timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('标准 ADS-B CSV', result.stdout)
        self.assertEqual(result.stderr, '')


if __name__ == '__main__':
    unittest.main()
