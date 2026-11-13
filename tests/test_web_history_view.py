"""test_web_history_view.py | 2026-11-09 | 历史参数、刷新约束及分组全点统计 | Codex."""
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import numpy as np
import pandas as pd
from src.app.module_catalog import artifact_module_directory
from src.web_console.web_util import ModuleHistoryUtil
from src.web_console.group_view import group_result
from src.web_console.segment_view import segment_result
from src.modules.segment.manual_merge import merge_segments
from src.app.artifacts import hash_file
from src.app.tables import read_table, write_table
from tests import test_web_console as webtests


def history_run(root: Path, module: str, parameters: dict, dependencies: list[Path] = ()) -> Path:
    """构造明确参数的成功历史，避免使用当前默认值伪造旧运行。"""
    path = root / 'data' / artifact_module_directory(module) / 'fixture' / '1#20261109_180000#42'
    (path / 'data').mkdir(parents=True)
    (path / 'resolved_config.json').write_text(json.dumps({'module_name': module, 'dataset_id': 'fixture',
        'dataset_name': 'fixture', 'run_id': '1', 'run_timestamp': '20261109_180000',
        'global': {'seed': 42}, 'module': parameters}))
    (path / 'manifest.json').write_text(json.dumps({'module': module, 'status': 'SUCCEEDED',
        'dependencies': [{'manifest': str(item / 'manifest.json')} for item in dependencies]}))
    return path


def group_fixture(root: Path) -> Path:
    """两个建组片段和一个灰色背景片段，含未被绘图抽中的高度极值。"""
    segment = history_run(root, 'segment', {})
    grouped = history_run(root, 'group', {}, [segment])
    pd.DataFrame({'group_id': ['g0','g0'], 'segment_id': ['s0','s1'], 'flight_id': ['f0','f1']}).to_csv(
        grouped / 'data/routing_segments.csv', index=False)
    records = [f'r{i}' for i in range(35)]
    ids = ['s0']*30 + ['s1']*3 + ['s2']*2
    times = np.arange(35, dtype=np.int64)*1_000_000_000 + 1700000000000000000
    mapping = pd.DataFrame({'record_id': records, 'segment_id': ids, 'time_ns': times})
    mapping.to_csv(segment / 'data/point_segment_map.csv', index=False)
    altitudes = np.arange(35, dtype=float)
    unsampled = next(i for i in range(30) if i not in np.linspace(0,29,24,dtype=int))
    altitudes[unsampled] = 999
    pd.DataFrame({'record_id':records, 'time_ns':times, 'longitude_deg':np.arange(35)/100+120,
                  'latitude_deg':np.arange(35)/100+30, 'geometric_altitude_m':altitudes}).to_csv(
        segment / 'data/input_observations.csv', index=False)
    return grouped


class WebHistoryViewTests(unittest.TestCase):
    def test_manual_segment_merge_synchronizes_segment_records(self) -> None:
        """人工合并必须同步片段表、点映射、展示 SQLite、序列审计和 manifest。"""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / 'data/2_segment/fixture/1#20261113_120000#42'
            data = run / 'data'; data.mkdir(parents=True)
            segments = pd.DataFrame([
                {'segment_id': 'f1:0', 'flight_id': 'f1', 'icao': 'ICAO001', 'call': 'CALL001', 'ordinal': 0,
                 'phase': 'LVL', 'points': 2, 'start_time': '2024-01-01T00:00:00+00:00', 'end_time': '2024-01-01T00:00:01+00:00',
                 'duration_seconds': 1., 'start_longitude_deg': 120., 'start_latitude_deg': 30., 'end_longitude_deg': 120.1,
                 'end_latitude_deg': 30.1, 'span_km': 1., 'start_altitude_m': 1000., 'end_altitude_m': 1001.,
                 'first_source_row_index': 0, 'last_source_row_index': 1, 'segment_label': 0, 'split': 'train'},
                {'segment_id': 'f1:1', 'flight_id': 'f1', 'icao': 'ICAO001', 'call': 'CALL001', 'ordinal': 1,
                 'phase': 'DE', 'points': 2, 'start_time': '2024-01-01T00:00:02+00:00', 'end_time': '2024-01-01T00:00:03+00:00',
                 'duration_seconds': 1., 'start_longitude_deg': 120.2, 'start_latitude_deg': 30.2, 'end_longitude_deg': 120.3,
                 'end_latitude_deg': 30.3, 'span_km': 1., 'start_altitude_m': 1002., 'end_altitude_m': 1003.,
                 'first_source_row_index': 2, 'last_source_row_index': 3, 'segment_label': 0, 'split': 'train'},
                {'segment_id': 'f1:2', 'flight_id': 'f1', 'icao': 'ICAO001', 'call': 'CALL001', 'ordinal': 2,
                 'phase': 'LVL', 'points': 2, 'start_time': '2024-01-01T00:00:04+00:00', 'end_time': '2024-01-01T00:00:05+00:00',
                 'duration_seconds': 1., 'start_longitude_deg': 120.4, 'start_latitude_deg': 30.4, 'end_longitude_deg': 120.5,
                 'end_latitude_deg': 30.5, 'span_km': 1., 'start_altitude_m': 1004., 'end_altitude_m': 1005.,
                 'first_source_row_index': 4, 'last_source_row_index': 5, 'segment_label': 0, 'split': 'train'},
            ])
            mapping = pd.DataFrame([
                {'record_id': f'r{index}', 'flight_id': 'f1', 'time_ns': index, 'phase': phase,
                 'segment_id': f'f1:{index // 2}', 'injected_label': 0, 'split': 'train'}
                for index, phase in enumerate(('LVL', 'LVL', 'DE', 'DE', 'LVL', 'LVL'))
            ])
            write_table(segments, data, 'segments'); write_table(mapping, data, 'point_segment_map')
            database = sqlite3.connect(data / 'segment_progress.sqlite')
            try:
                database.execute('''CREATE TABLE segments (
                    segment_id TEXT PRIMARY KEY, flight_id TEXT, icao TEXT, call TEXT, phase TEXT, points INTEGER,
                    start_time TEXT, end_time TEXT, duration_seconds REAL, start_longitude_deg REAL,
                    start_latitude_deg REAL, end_longitude_deg REAL, end_latitude_deg REAL, span_km REAL,
                    start_altitude_m REAL, end_altitude_m REAL)''')
                columns = ['segment_id', 'flight_id', 'icao', 'call', 'phase', 'points', 'start_time', 'end_time',
                           'duration_seconds', 'start_longitude_deg', 'start_latitude_deg', 'end_longitude_deg',
                           'end_latitude_deg', 'span_km', 'start_altitude_m', 'end_altitude_m']
                database.executemany('INSERT INTO segments VALUES (' + ','.join('?' for _ in columns) + ')',
                                     segments[columns].itertuples(index=False, name=None))
                database.commit()
            finally:
                database.close()
            (run / 'resolved_config.json').write_text(json.dumps({'module_name': 'segment', 'dataset_id': 'fixture',
                'dataset_name': 'fixture', 'run_id': '1', 'run_timestamp': '20261113_120000', 'global': {'seed': 42}, 'module': {}}))
            (run / 'summary.json').write_text(json.dumps({'segments': 3, 'sequence_categories': {}}))
            (run / 'validation.json').write_text(json.dumps({'passed': True, 'checks': {'segments': 3}}))
            outputs = [{'path': path.relative_to(run).as_posix(), 'bytes': path.stat().st_size, 'sha256': hash_file(path)}
                       for path in sorted(run.rglob('*')) if path.is_file()]
            (run / 'manifest.json').write_text(json.dumps({'status': 'SUCCEEDED', 'module': 'segment', 'dataset_id': 'fixture',
                'dataset_name': 'fixture', 'run_id': '1', 'dependencies': [], 'outputs': outputs, 'validation': 'validation.json'}))
            result = merge_segments(root, run.relative_to(root).as_posix(), ['f1:0', 'f1:1'], 'LVL')
            self.assertEqual(result['segments'], 2)
            merged = read_table(data, 'segments').sort_values('ordinal')
            self.assertEqual(merged[['segment_id', 'phase', 'points', 'ordinal']].values.tolist(),
                             [['f1:0', 'LVL', 4, 0], ['f1:2', 'LVL', 2, 1]])
            point_map = read_table(data, 'point_segment_map').sort_values('record_id')
            self.assertEqual(point_map['segment_id'].tolist(), ['f1:0', 'f1:0', 'f1:0', 'f1:0', 'f1:2', 'f1:2'])
            self.assertEqual(point_map['phase'].tolist(), ['LVL'] * 6)
            self.assertEqual(segment_result(root, run.relative_to(root).as_posix())['rows'][0]['points'], 4)
            review = read_table(data, 'flight_phase_sequences')
            self.assertEqual(review.iloc[0]['phase_sequence'], 'LVL>LVL')
            ledger = read_table(data, 'manual_segment_merges')
            self.assertEqual(json.loads(ledger.iloc[0]['source_segment_ids']), ['f1:0', 'f1:1'])
            from src.app.artifacts import read_manifest
            self.assertEqual(read_manifest(run / 'manifest.json', expected_module='segment')['manual_revision'], 1)

    def test_running_segment_exposes_parent_committed_progress_only(self) -> None:
        """运行中的阶段表从暂存目录 SQLite 读取，供 AJAX 安全轮询。"""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pending = root / 'data/2_segment/fixture/.1#20261112_160000#42.pending'
            data = pending / 'data'; data.mkdir(parents=True)
            (pending / 'resolved_config.json').write_text(json.dumps({
                'module_name': 'segment', 'dataset_id': 'fixture', 'dataset_name': 'fixture',
                'run_id': '1', 'run_timestamp': '20261112_160000', 'global': {'seed': 42}, 'module': {},
            }), encoding='utf-8')
            database = sqlite3.connect(data / 'segment_progress.sqlite')
            try:
                database.execute('''CREATE TABLE segments (
                    segment_id TEXT PRIMARY KEY, flight_id TEXT, icao TEXT, call TEXT, phase TEXT, points INTEGER,
                    start_time TEXT, end_time TEXT, duration_seconds REAL, start_longitude_deg REAL,
                    start_latitude_deg REAL, end_longitude_deg REAL, end_latitude_deg REAL, span_km REAL,
                    start_altitude_m REAL, end_altitude_m REAL)''')
                database.executemany('INSERT INTO segments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', [
                    ('f1:1', 'f1', 'ABC001', 'CAF001', 'CR', 8, '2024-01-01T00:01:00+00:00',
                     '2024-01-01T00:02:00+00:00', 60.0, 120.1, 30.1, 120.2, 30.2, 14.7, 1100.0, 1200.0),
                    ('f2:0', 'f2', 'ABC001', 'CAF001', 'CL', 8, '2024-01-01T00:00:00+00:00',
                     '2024-01-01T00:01:00+00:00', 60.0, 120.0, 30.0, 120.1, 30.1, 14.7, 1000.0, 1100.0),
                    ('f3:0', 'f3', 'ZZZ001', 'ZZZ001', 'DE', 8, '2024-01-01T00:00:00+00:00',
                     '2024-01-01T00:01:00+00:00', 60.0, 120.0, 30.0, 120.1, 30.1, 14.7, 1000.0, 1100.0),
                ])
                database.commit()
            finally:
                database.close()
            value = segment_result(root, pending.relative_to(root).as_posix())
            self.assertTrue(value['available'])
            self.assertEqual([(row['icao'], row['call'], row['start_time']) for row in value['rows']], [
                ('ABC001', 'CAF001', '2024-01-01T00:00:00+00:00'),
                ('ABC001', 'CAF001', '2024-01-01T00:01:00+00:00'),
                ('ZZZ001', 'ZZZ001', '2024-01-01T00:00:00+00:00'),
            ])
            self.assertEqual(value['rows'][0]['span_km'], 14.7)

    def test_saved_parameters_and_current_command_defaults_are_separate(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = history_run(root, 'train', {'d_model': 32, 'gpus': '0', 'epochs': 5})
            before = (path / 'resolved_config.json').read_bytes()
            data = ModuleHistoryUtil.query(root, 'train')
            self.assertEqual(data['runs'][0]['parameters']['d_model'], 32)
            self.assertNotIn('dropout', data['runs'][0]['parameters'])
            self.assertIn({'key': 'd_model', 'label': '--d-model'}, data['columns'])
            self.assertEqual(before, (path / 'resolved_config.json').read_bytes())
            for module in ('train','calibrate','detect'):
                entry = ModuleHistoryUtil.from_command(('python','main.py','--seed','7','-m',module,'--input','x.json'))
                self.assertEqual(entry['parameters']['gpus'], '0,1,2,3')
                self.assertEqual(entry['seed'], 7)
            entry = ModuleHistoryUtil.from_command(('python','main.py','-m','split','--input','x','--ratios','.6','.2','.2'))
            self.assertEqual(list(entry['parameters']['ratios']), [.6,.2,.2])

    def test_full_point_statistics_not_visualization_samples(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            grouped = group_fixture(root)
            value = group_result(root, grouped.relative_to(root).as_posix())
            group = value['groups'][0]
            self.assertEqual((group['segments'], group['longest'], group['shortest']), (2,30,3))
            self.assertEqual(group['max_alt'], 999)
            self.assertEqual(group['min_alt'], 0)
            self.assertEqual(group['earliest'], '2023-11-14T22:13:20+00:00')
            self.assertEqual(group['latest'], '2023-11-14T22:13:52+00:00')
            self.assertEqual(len(value['trajectories']), 3)
            self.assertTrue(all(len(item['points']) <= 24 for item in value['trajectories']))
            self.assertIsNone(next(item for item in value['trajectories'] if item['segment_id']=='s2')['group_id'])
            with self.assertRaises(ValueError):
                group_result(root, '../outside')

    def test_http_refresh_requeries_and_blocks_active_module(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = history_run(root, 'train', {'d_model': 32})
            grouped = group_fixture(root)
            with webtests.test_server(root) as server:
                base = f'http://127.0.0.1:{server.server_port}'
                cookie = webtests.WebConsoleTests._login(base, webtests.TOKEN)
                def get(url: str) -> dict:
                    with urlopen(Request(base+url, headers={'Cookie':cookie})) as response:
                        return json.load(response)
                self.assertEqual(get('/api/history?module=train')['runs'][0]['parameters']['d_model'],32)
                config = json.loads((path/'resolved_config.json').read_text()); config['module']['d_model']=96
                (path/'resolved_config.json').write_text(json.dumps(config))
                self.assertEqual(get('/api/history?module=train&refresh=1')['runs'][0]['parameters']['d_model'],96)
                with patch.object(server.cafad_manager, 'latest_snapshot', return_value={'status':'running'}):
                    with self.assertRaises(HTTPError) as error:
                        get('/api/history?module=train&refresh=1')
                    self.assertEqual(error.exception.code,409)
                    self.assertTrue(get('/api/history?module=train')['running'])
                result=get('/api/group/result?artifact='+grouped.relative_to(root).as_posix().replace('#','%23'))
                self.assertEqual(result['groups'][0]['max_alt'],999)
                with self.assertRaises(HTTPError) as error:
                    urlopen(base+'/api/group/result')
                self.assertEqual(error.exception.code,401)
