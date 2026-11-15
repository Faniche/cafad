"""test_web_history_view.py | 2026-11-09 | 历史参数、刷新约束及分组全点统计 | Codex."""
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from urllib.parse import quote
import numpy as np
import pandas as pd
from src.app.module_catalog import artifact_module_directory
from src.app.tables import write_table
from src.web_console.web_util import ModuleHistoryUtil
from src.web_console.group_view import group_result
from src.web_console.segment_plot import _altitude_axis_upper, segment_flight_html, segment_flight_svg
from src.web_console.segment_view import segment_result
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


def segment_plot_fixture(root: Path) -> Path:
    """构造一个含两种阶段的航班，验证 SVG 按最终片段标签查询。"""
    segment = history_run(root, 'segment', {})
    data = segment / 'data'
    write_table(pd.DataFrame({
        'record_id': ['p0', 'p1', 'p2', 'p3'],
        'longitude_deg': [120.0, 120.1, 120.2, 120.3],
        'latitude_deg': [30.0, 30.1, 30.1, 30.2],
        'geometric_altitude_m': [1000.0, 1200.0, 1400.0, 1600.0],
    }), data, 'input_observations')
    write_table(pd.DataFrame({
        'record_id': ['p0', 'p1', 'p2', 'p3'], 'flight_id': ['plot-flight'] * 4,
        'segment_id': ['plot-flight:0', 'plot-flight:0', 'plot-flight:1', 'plot-flight:1'],
        'cell_key': ['h0_t0_x1_y1', 'h0_t0_x1_y1', 'h2_t0_x1_y1', 'h2_t0_x1_y1'], 'time_ns': [1, 2, 3, 4],
    }), data, 'point_segment_map')
    return segment


class WebHistoryViewTests(unittest.TestCase):
    def test_flight_plot_uses_a_minimum_15000_m_altitude_axis(self) -> None:
        """低空航班使用统一高度尺度，高于其的航班不被截断。"""
        self.assertEqual(_altitude_axis_upper(np.array([0.0, 9_000.0, 12_000.0])), 15_000.0)
        self.assertEqual(_altitude_axis_upper(np.array([0.0, 15_123.0])), 15_123.0)

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
                    segment_id TEXT PRIMARY KEY, flight_id TEXT, icao TEXT, call TEXT, cell_key TEXT, height_band INTEGER,
                    time_bucket INTEGER, longitude_cell INTEGER, latitude_cell INTEGER, points INTEGER,
                    start_time TEXT, end_time TEXT, duration_seconds REAL, start_longitude_deg REAL,
                    start_latitude_deg REAL, end_longitude_deg REAL, end_latitude_deg REAL, span_km REAL,
                    start_altitude_m REAL, end_altitude_m REAL)''')
                database.executemany('INSERT INTO segments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', [
                    ('f1:1', 'f1', 'ABC001', 'CAF001', 'h2_t0_x1_y1', 2, 0, 1, 1, 8, '2024-01-01T00:01:00+00:00',
                     '2024-01-01T00:02:00+00:00', 60.0, 120.1, 30.1, 120.2, 30.2, 14.7, 1100.0, 1200.0),
                    ('f2:0', 'f2', 'ABC001', 'CAF001', 'h1_t0_x1_y1', 1, 0, 1, 1, 8, '2024-01-01T00:00:00+00:00',
                     '2024-01-01T00:01:00+00:00', 60.0, 120.0, 30.0, 120.1, 30.1, 14.7, 1000.0, 1100.0),
                    ('f3:0', 'f3', 'ZZZ001', 'ZZZ001', 'h0_t0_x1_y1', 0, 0, 1, 1, 8, '2024-01-01T00:00:00+00:00',
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

    def test_segment_flight_plot_returns_hard_cell_colored_svg(self) -> None:
        """航班 ID 图接口只查询该航班，生成包含硬单元图例的 SVG。"""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            segment = segment_plot_fixture(root)
            artifact = segment.relative_to(root).as_posix()
            image = segment_flight_svg(root, artifact, 'plot-flight')
            self.assertIn(b'<svg', image)
            self.assertIn(b'h0_t0_x1_y1', image)
            self.assertIn(b'width="720pt"', image)
            cache = segment / 'flight_trajectory_svg' / 'plot-flight.svg'
            self.assertTrue(cache.is_file())
            self.assertEqual(cache.read_bytes(), image)
            self.assertEqual(segment_flight_svg(root, artifact, 'plot-flight'), image)
            with webtests.test_server(root) as server:
                base = f'http://127.0.0.1:{server.server_port}'
                cookie = webtests.WebConsoleTests._login(base, webtests.TOKEN)
                address = f'{base}/api/segment/flight-plot?artifact={quote(artifact)}&flight_id=plot-flight'
                with urlopen(Request(address, headers={'Cookie': cookie})) as response:
                    self.assertEqual(response.headers.get_content_type(), 'image/svg+xml')
                    self.assertIn(b'<svg', response.read())

    def test_segment_flight_plot_returns_interactive_html(self) -> None:
        """交互式端点保存 Plotly HTML，并保留稳定硬单元色和起止标记。"""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            segment = segment_plot_fixture(root)
            artifact = segment.relative_to(root).as_posix()
            page = segment_flight_html(root, artifact, 'plot-flight')
            self.assertIn(b'Plotly.newPlot', page)
            self.assertIn(b'\\u2605', page)
            self.assertIn(b'\\u27a4', page)
            self.assertIn(b'h2_t0_x1_y1', page)
            self.assertIn(b'/api/segment/plotly.min.js', page)
            cache = segment / 'flight_trajectory_html' / 'plot-flight.html'
            self.assertTrue(cache.is_file())
            with webtests.test_server(root) as server:
                base = f'http://127.0.0.1:{server.server_port}'
                cookie = webtests.WebConsoleTests._login(base, webtests.TOKEN)
                address = f'{base}/api/segment/flight-plot-interactive?artifact={quote(artifact)}&flight_id=plot-flight'
                with urlopen(Request(address, headers={'Cookie': cookie})) as response:
                    self.assertEqual(response.headers.get_content_type(), 'text/html')
                    self.assertIn(b'Plotly.newPlot', response.read())
                with urlopen(Request(f'{base}/api/segment/plotly.min.js', headers={'Cookie': cookie})) as response:
                    self.assertEqual(response.headers.get_content_type(), 'text/javascript')
                    self.assertIn(b'window.Plotly', response.read())

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
