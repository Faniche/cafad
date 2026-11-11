"""test_data_module_pipeline.py | 2026-11-03 | 根入口数据准备链路集成测试 | Codex."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src.app.cli import main
from src.utils.strings import PathUtils


ROOT = Path(__file__).resolve().parents[1]


def write_adsb_fixture(path: Path) -> None:
    """构造含重复时间与缺失必需字段的最小 ADS-B 输入。"""
    rows: list[dict[str, object]] = []
    start = pd.Timestamp('2024-01-01T00:00:00Z')
    for flight_number in range(14):
        icao = f'ABC{flight_number:03d}'
        call = f'CAF{flight_number:03d}'
        for offset in range(5):
            rows.append({
                'icao号': icao,
                'call号': call,
                '时间': (start + pd.Timedelta(seconds=flight_number * 120 + offset * 5)).isoformat(),
                '经度': 120.0 + flight_number / 10,
                '纬度': 30.0 + flight_number / 10,
                '水平高度': 0.0,
                '气压高度': 0.0,
                '航向': 90.0,
                '速度': 0.0,
            })
    duplicate = dict(rows[0])
    duplicate['速度'] = 12.0
    rows.append(duplicate)
    invalid = dict(rows[1])
    invalid['速度'] = ''
    invalid['时间'] = 'not-a-time'
    rows.append(invalid)
    negative_speed = dict(rows[2])
    negative_speed['时间'] = (start + pd.Timedelta(seconds=12)).isoformat()
    negative_speed['速度'] = -0.1
    rows.append(negative_speed)
    pd.DataFrame(rows).to_csv(path, index=False, encoding='utf-8')


class DataModulePipelineTests(unittest.TestCase):
    """验证每一步只依赖上一模块落盘的 manifest 和数据文件。"""

    def run_command(self, arguments: list[str], cwd: Path) -> None:
        """在临时项目根运行统一入口，不污染开发数据目录。"""
        with patch.object(PathUtils, 'get_project_root', return_value=cwd):
            self.assertEqual(main(arguments), 0)

    def test_preprocess_split_inject_and_segment_are_independent_published_steps(self) -> None:
        """规范观测、划分、单副本注入和 Sun 切分可隔进程顺序恢复。"""
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / 'preprocessed_data_fixture.csv'
            environment = base / '.env'
            write_adsb_fixture(source)
            # 让 inject 实际走两个 spawn 工作进程，验证航班级并行不会破坏
            # 主进程汇总的标签、事件和输出顺序。
            environment.write_text('nproc=2\ndata_proc_usage=100%\ninjection_minimum_trajectory_points=5\n', encoding='utf-8')
            self.run_command([
                '-m', 'preprocess', '--format', 'adsb-csv', '--input', str(source),
                '--altitude-unit', 'm', '--speed-unit', 'm/s', '--heading-unit', 'deg', '--flight-gap-seconds', '90',
            ], base)
            preprocess_dir = next((base / 'data' / '0_preprocess' / 'preprocessed_data_fixture').iterdir())
            observations = pd.read_csv(preprocess_dir / 'data' / 'observations.csv')
            rejections = pd.read_csv(preprocess_dir / 'data' / 'rejections.csv')
            self.assertEqual(len(observations), 70)
            self.assertEqual(len(rejections), 3)
            self.assertFalse(observations.duplicated(['icao', 'call', 'time_ns']).any())
            self.assertTrue(observations['speed_mps'].ge(0.0).all())
            self.assertIn('negative_speed', set(rejections['reason']))
            self.assertEqual(len(observations) + len(rejections), 73)
            self.run_command([
                '--seed', '10', '-m', 'split', '--input',
                str(preprocess_dir / 'manifest.json'),
            ], base)
            split_dir = next((base / 'data' / '1_split' / 'preprocessed_data_fixture').iterdir())
            splits = pd.read_csv(split_dir / 'data' / 'splits.csv')
            self.assertEqual(splits['split'].value_counts().to_dict(), {'train': 10, 'val': 2, 'test': 2})
            self.assertEqual(splits.groupby(['split', 'injection_plan']).size().to_dict(), {
                ('test', 'abnormal'): 1, ('test', 'normal'): 1,
                ('train', 'abnormal'): 5, ('train', 'normal'): 5,
                ('val', 'abnormal'): 1, ('val', 'normal'): 1,
            })
            protocol = base / 'test_injection_protocol.json'
            protocol.write_text(json.dumps({
                'schema_version': 2,
                'event_length_points': [3, 3],
                'context_points_each_side': 1,
                'event_type_probabilities': {
                    'gaussian_point': 0.0, 'reported_fixed_offset': 0.0,
                    'reported_progressive_drift': 0.0, 'reported_local_perturbation': 0.0,
                    'true_local_maneuver': 1.0, 'motion_transfer': 0.0,
                },
                'gaussian_point': {
                    'speed_probability': 0.5, 'speed_std_fraction': 0.01, 'speed_std_floor_mps': 0.1,
                    'speed_std_cap_mps': 1.0, 'altitude_std_fraction': 0.01, 'altitude_std_floor_m': 0.1,
                    'altitude_std_cap_m': 1.0,
                },
                'reported_fixed_offset': {
                    'field_probabilities': {'lateral_position': 1.0, 'altitude': 0.0, 'speed': 0.0},
                    'lateral_offset_m_range': [1.0, 2.0], 'altitude_offset_m_range': [1.0, 2.0],
                    'speed_offset_mps_range': [1.0, 2.0],
                },
                'reported_progressive_drift': {
                    'field_probabilities': {'altitude': 1.0, 'speed': 0.0},
                    'altitude_step_m_range': [1.0, 2.0], 'speed_step_mps_range': [1.0, 2.0],
                },
                'reported_local_perturbation': {
                    'field_probabilities': {'speed': 0.0, 'lateral_position': 1.0, 'altitude': 0.0},
                    'speed_factor_range': [0.8, 1.2], 'lateral_offset_m_range': [1.0, 2.0],
                    'altitude_amplitude_m_range': [1.0, 2.0], 'smoothing_points': 1,
                },
                'true_local_maneuver': {
                    'field_probabilities': {'lateral_position': 1.0, 'altitude': 0.0},
                    'lateral_offset_m_range': [1.0, 2.0], 'altitude_amplitude_m_range': [1.0, 2.0],
                    'speed_factor_range': [0.8, 1.2],
                    'smoothing_points': 1,
                },
                'motion_transfer': {'scale_range': [0.8, 1.2]},
            }), encoding='utf-8')
            self.run_command([
                '--seed', '10', '-m', 'inject', '--input', str(preprocess_dir / 'manifest.json'),
                '--split-manifest', str(split_dir / 'manifest.json'),
                '--protocol', str(protocol),
            ], base)
            inject_dir = next((base / 'data' / '2_inject' / 'preprocessed_data_fixture').iterdir())
            injected = pd.read_csv(inject_dir / 'data' / 'observations.csv')
            record_labels = pd.read_csv(inject_dir / 'data' / 'record_labels.csv')
            injection_events = pd.read_csv(inject_dir / 'data' / 'injection_events.csv')
            self.assertTrue(injected.columns.tolist()[-4:] == ['split', 'injection_plan', 'injection_outcome', 'injection_skip_reason'])
            self.assertTrue(record_labels['injected_label'].eq(injected['injected_label']).all())
            self.assertFalse(injection_events.empty)
            actual_flight_labels = injected.groupby('flight_id')['injected_label'].max()
            planned = splits.set_index('flight_id')['injection_plan']
            self.assertTrue(actual_flight_labels[planned.eq('normal')].eq(0).all())
            self.assertTrue(actual_flight_labels[planned.eq('abnormal')].eq(1).all())
            self.run_command([
                '-m', 'segment', '--input',
                str(inject_dir / 'manifest.json'), '--window-seconds', '60', '--buckets', '2', '--chunk-rows', '10',
            ], base)
            segment_dir = next((base / 'data' / '3_segment' / 'preprocessed_data_fixture').iterdir())
            mapping = pd.read_csv(segment_dir / 'data' / 'point_segment_map.csv')
            segments = pd.read_csv(segment_dir / 'data' / 'segments.csv')
            unrecognized = pd.read_csv(segment_dir / 'data' / 'unrecognized_trajectories.csv')
            unrecognized_points = pd.read_csv(segment_dir / 'data' / 'unrecognized_point_map.csv')
            manifest = json.loads((segment_dir / 'manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(len(mapping), len(observations))
            self.assertFalse(mapping['record_id'].duplicated().any())
            self.assertIn('split', mapping.columns)
            self.assertIn('split', segments.columns)
            self.assertGreater(segments['segment_label'].sum(), 0)
            self.assertTrue(unrecognized.empty)
            self.assertTrue(unrecognized_points.empty)
            self.assertEqual(manifest['status'], 'SUCCEEDED')
            self.assertFalse((base / '.cafad-run.lock').exists())
            counters = pd.read_csv(base / 'data' / 'record_counter.csv')
            self.assertEqual(counters.to_dict('records'), [
                {'dataset': 'preprocessed_data_fixture', 'module': 'preprocess', 'counter': 1, 'state': 'success'},
                {'dataset': 'preprocessed_data_fixture', 'module': 'split', 'counter': 1, 'state': 'success'},
                {'dataset': 'preprocessed_data_fixture', 'module': 'inject', 'counter': 1, 'state': 'success'},
                {'dataset': 'preprocessed_data_fixture', 'module': 'segment', 'counter': 1, 'state': 'success'},
            ])
            registry = json.loads((base / 'run_registry.json').read_text(encoding='utf-8'))
            self.assertTrue(all(item['state'] == 'success' for item in registry['runs']))
            logs = sorted((base / 'logs').glob('*#preprocessed_data_fixture#*.log'))
            self.assertEqual(len(logs), 4)
            for log in logs:
                lines = log.read_text(encoding='utf-8').splitlines()
                self.assertIn('运行开始时间：', lines[0])
                self.assertIn('运行结束时间：', lines[-1])

    def test_failed_module_updates_the_automatic_run_record(self) -> None:
        """模块计算失败后仍保留递增编号，并在 CSV/JSON 中标为 failure。"""
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / 'preprocessed_data_broken.csv'
            (base / '.env').write_text('nproc=1\ndata_proc_usage=100%\ninjection_minimum_trajectory_points=5\n', encoding='utf-8')
            pd.DataFrame({'not_adsb': [1]}).to_csv(source, index=False)
            with patch.object(PathUtils, 'get_project_root', return_value=base):
                status = main([
                    '-m', 'preprocess', '--format', 'adsb-csv', '--input', str(source),
                    '--altitude-unit', 'm', '--speed-unit', 'm/s', '--heading-unit', 'deg',
                ])
            self.assertEqual(status, 1)
            counters = pd.read_csv(base / 'data' / 'record_counter.csv')
            self.assertEqual(counters.to_dict('records'), [
                {'dataset': 'preprocessed_data_broken', 'module': 'preprocess', 'counter': 1, 'state': 'failure'},
            ])
            registry = json.loads((base / 'run_registry.json').read_text(encoding='utf-8'))
            self.assertEqual(registry['runs'][0]['state'], 'failure')
            log = next((base / 'logs').glob('*#preprocessed_data_broken#preprocess#1_42.log'))
            lines = log.read_text(encoding='utf-8').splitlines()
            self.assertIn('运行开始时间：', lines[0])
            self.assertIn('模块执行失败', '\n'.join(lines))
            self.assertIn('运行结束时间：', lines[-1])


if __name__ == '__main__':
    unittest.main()
