"""Method and real spawned-process checks on artificial, pre-injection inputs."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.data_processing.phase_pipeline import FlightTask, PipelineConfig, _parse_times, _process_flight, process_cleaned_csv
from src.data_processing.sun_phase import identify_sun_phases
from src.utils.env import Env


ROOT = Path(__file__).resolve().parents[1]


def cleaned_fixture(path: Path) -> pd.DataFrame:
    rows = []
    base = pd.Timestamp('2024-05-21 01:00:00')
    def add(icao, call, seconds, altitude, speed, rates):
        for second, rate in zip(seconds, rates):
            rows.append({'icao号': icao, 'call号': call, '时间': str(base + pd.Timedelta(seconds=second)),
                         '水平高度': altitude, '速度': speed, 'roc': rate})
    add('000123', 'A', [0, 5, 5, 10], 0., 0., [0.] * 4)
    add('000123', 'A', [200, 210], 10000., 200., [0.] * 2)
    add('0002E9', 'B', [0, 10, 20, 60, 70, 80], 10000., 200., [1000.] * 3 + [-1000.] * 3)
    add('000123', 'CHANGED', [40], 0., 0., [0.])
    frame = pd.DataFrame(rows).sample(frac=1., random_state=3).reset_index(drop=True)
    frame.to_csv(path, index=False)
    return frame


class EnvTests(unittest.TestCase):
    def setUp(self):
        self.old = Env._values.copy(), Env._path
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / '.env'
        self.clean = patch.dict(os.environ, {}, clear=True)
        self.clean.start()

    def tearDown(self):
        Env._values, Env._path = self.old
        self.clean.stop()
        self.tmp.cleanup()

    def load(self, text):
        self.path.write_text(text, encoding='utf-8')
        Env.load(self.path)

    def test_requested_physical_core_budget(self):
        self.load('nproc=56\ndata_proc_usage=80%\n')
        self.assertEqual(Env.data_process_workers(), 44)

    def test_decimal_floor_does_not_lose_a_worker_to_float_rounding(self):
        self.load('nproc=100\ndata_proc_usage=29%\n')
        self.assertEqual(Env.data_process_workers(), 29)

    def test_environment_override_and_quoted_fraction(self):
        self.load('nproc=56\ndata_proc_usage="0.8" # comment\n')
        with patch.dict(os.environ, {'NPROC': '10'}):
            self.assertEqual(Env.data_process_workers(), 8)

    def test_invalid_budgets_are_rejected(self):
        for text in ('nproc=0\ndata_proc_usage=80%', 'nproc=2\ndata_proc_usage=1%',
                     'nproc=56\ndata_proc_usage=101%', 'nproc=56\ndata_proc_usage=NaN'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.load(text)
                Env.data_process_workers()


class SunMethodTests(unittest.TestCase):
    def test_all_five_author_phase_rules_and_final_window(self):
        times = np.arange(0., 125., 5.)
        for altitude, speed, rate, expected in [(0., 0., 0., 'GND'), (10000., 200., 1000., 'CL'),
                (10000., 200., -1000., 'DE'), (35000., 600., 0., 'CR'), (10000., 200., 0., 'LVL')]:
            with self.subTest(phase=expected):
                labels = identify_sun_phases(times, np.full(len(times), altitude), np.full(len(times), speed), np.full(len(times), rate))
                self.assertTrue((labels == expected).all())
                self.assertEqual(len(labels), len(times))

    def test_single_point_and_short_flights_have_labels(self):
        for count in (1, 2, 3):
            self.assertEqual(identify_sun_phases(np.arange(count), np.zeros(count), np.zeros(count)).tolist(), ['GND'] * count)

    def test_relative_time_translation_and_inputs_are_preserved(self):
        times = np.arange(5., dtype=float)
        altitude = np.full(5, 10000.)
        original = altitude.copy()
        a = identify_sun_phases(times, altitude, np.full(5, 200.))
        b = identify_sun_phases(times + 1e9, altitude, np.full(5, 200.))
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(altitude, original)

    def test_missing_rate_is_derived_using_physical_time(self):
        times = np.arange(0., 120., 5.)
        labels = identify_sun_phases(times, 10000. + times * 1000. / 60., np.full(len(times), 200.))
        self.assertTrue((labels == 'CL').all())

    def test_duplicates_reversed_times_and_nonfinite_are_not_silently_accepted(self):
        for times in ([0., 0.], [1., 0.], [0., np.nan]):
            with self.subTest(times=times), self.assertRaises(ValueError):
                identify_sun_phases(times, [0., 0.], [0., 0.])

    def test_numeric_epoch_nanoseconds_are_exact(self):
        values = pd.Series(['1716253200000000001', '1716253200000000002'])
        np.testing.assert_array_equal(_parse_times(values, 'ns'), [1716253200000000001, 1716253200000000002])
        with self.assertRaises(ValueError):
            _parse_times(values, None)


class PipelineTests(unittest.TestCase):
    def test_sun_value_error_becomes_an_unrecognized_trajectory_report(self):
        """单条轨迹的数值失败不伪造片段；报告保留首尾位置和稳定输入行号。"""
        with tempfile.TemporaryDirectory() as directory:
            task = FlightTask(
                flight_id='flight-1', icao='ABC123', call='CAF123',
                row_indices=np.array([7, 8], dtype=np.int64),
                timestamps_ns=np.array([1_000_000_000, 3_000_000_000], dtype=np.int64),
                altitude=np.array([100., 110.]), speed=np.array([50., 51.]), vertical_rate=None,
                longitude=np.array([120., 121.]), latitude=np.array([30., 31.]),
            )
            with patch('src.data_processing.phase_pipeline.identify_sun_phases',
                       side_effect=ValueError('Spline or derived vertical rate produced nonfinite values')):
                result = _process_flight(task, PipelineConfig('m', 'm/s'), directory)
            self.assertEqual(result['status'], 'unrecognized')
            report = result['unrecognized']
            self.assertEqual(report['flight_id'], 'flight-1')
            self.assertEqual(report['points'], 2)
            self.assertEqual(report['duration_seconds'], 2.)
            self.assertEqual((report['start_longitude_deg'], report['end_latitude_deg']), (120., 31.))
            self.assertIn('Spline', report['reason'])
            points = pd.read_csv(result['unrecognized_points_path'])
            self.assertEqual(points.to_dict('records'), [
                {'source_row_index': 7, 'flight_id': 'flight-1'},
                {'source_row_index': 8, 'flight_id': 'flight-1'},
            ])

    def test_spawned_workers_unsorted_chunks_duplicate_times_and_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source, output = base / 'clean.csv', base / 'result'
            original = cleaned_fixture(source)
            before = source.read_bytes()
            config = PipelineConfig('ft', 'kt', workers=2, chunk_rows=3, buckets=3,
                                    flight_gap_seconds=90., vertical_rate_column='roc', vertical_rate_unit='ft/min')
            summary = process_cleaned_csv(source, output, config)
            self.assertEqual(summary['processed_flights'], 4)
            self.assertEqual(summary['total_segments'], 5)
            self.assertEqual(summary['input_points'], 13)
            self.assertEqual(summary['duplicate_timestamp_rows_retained'], 1)
            self.assertEqual(summary['flight_point_count'], {'count': 4, 'min': 1, 'max': 6, 'mean': 3.25, 'median': 3.})
            self.assertEqual(summary['segment_point_count'], {'count': 5, 'min': 1, 'max': 4, 'mean': 2.6, 'median': 3})
            points = pd.read_csv(output / 'points.csv', dtype={'icao': str, 'call': str}).sort_values('source_row_index')
            self.assertEqual(points.source_row_index.tolist(), list(range(13)))
            self.assertEqual(points.icao.tolist(), original['icao号'].tolist())
            self.assertEqual(points.call.tolist(), original['call号'].tolist())
            self.assertTrue(points.segment_id.notna().all())
            self.assertTrue(pd.read_csv(output / 'unrecognized_flights.csv').empty)
            self.assertTrue(pd.read_csv(output / 'unrecognized_points.csv').empty)
            self.assertEqual(source.read_bytes(), before)
            self.assertFalse(list(output.glob('_sun_work_*')))
            self.assertEqual(Path(summary['log_file']).parent, ROOT / 'logs')
            self.assertTrue(Path(summary['log_file']).is_file())
            self.assertFalse((output / 'processing.log').exists())
            with self.assertRaises(FileExistsError):
                process_cleaned_csv(source, output, config)

    def test_unit_conversion_and_worker_count_preserve_membership(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            feet_source, metre_source = base / 'ft.csv', base / 'm.csv'
            frame = cleaned_fixture(feet_source)
            frame['水平高度'] *= .3048
            frame['速度'] *= 1852. / 3600.
            frame['roc'] *= .3048 / 60.
            frame.to_csv(metre_source, index=False)
            a = process_cleaned_csv(feet_source, base / 'ft', PipelineConfig('ft', 'kt', workers=2,
                chunk_rows=3, buckets=3, flight_gap_seconds=90., vertical_rate_column='roc', vertical_rate_unit='ft/min'))
            b = process_cleaned_csv(metre_source, base / 'm', PipelineConfig('m', 'm/s', workers=1,
                chunk_rows=5, buckets=2, flight_gap_seconds=90., vertical_rate_column='roc', vertical_rate_unit='m/s'))
            left = pd.read_csv(base / 'ft/points.csv').sort_values('source_row_index')
            right = pd.read_csv(base / 'm/points.csv').sort_values('source_row_index')
            for column in ('source_row_index', 'flight_id', 'phase', 'segment_id'):
                self.assertEqual(left[column].tolist(), right[column].tolist())
            self.assertEqual(a['flight_point_count'], b['flight_point_count'])
            self.assertEqual(a['segment_point_count'], b['segment_point_count'])

    def test_invalid_cleaned_input_is_reported_as_failed(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / 'bad.csv', Path(directory) / 'result'
            pd.DataFrame({'icao': [''], 'call': ['A'], 'time': ['2024-01-01'], 'altitude': [0.], 'speed': [0.]}).to_csv(source, index=False)
            with self.assertRaises(ValueError):
                process_cleaned_csv(source, output, PipelineConfig('m', 'm/s'))
            self.assertEqual(json.loads((output / 'failure.json').read_text(encoding='utf-8'))['status'], 'failed')
            self.assertFalse((output / 'summary.json').exists())

    def test_root_entry_requires_one_module_and_logs_the_configuration_error(self):
        with tempfile.TemporaryDirectory() as directory:
            missing_env = Path(directory) / 'missing.env'
            prior = set((ROOT / 'logs').glob('run-*.log'))
            result = subprocess.run([sys.executable, '-B', str(ROOT / 'main.py'), '--env', str(missing_env)],
                                    cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=30)
            self.assertEqual(result.returncode, 1)
            self.assertEqual(result.stdout + result.stderr, '')
            created = set((ROOT / 'logs').glob('run-*.log')) - prior
            self.assertTrue(created)
            self.assertTrue(any('必须且只能指定一次' in path.read_text(encoding='utf-8') for path in created))


if __name__ == '__main__':
    unittest.main()
