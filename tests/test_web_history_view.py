"""test_web_history_view.py | 2026-11-09 | 历史参数、刷新约束及分组全点统计 | Codex."""
import json
from pathlib import Path
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
