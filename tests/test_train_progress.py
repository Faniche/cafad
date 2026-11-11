"""test_train_progress.py | 2026-11-09 | 进度消息、隔离读取与旧日志降级测试 | Codex."""
import json
import logging
from pathlib import Path
import queue
import tempfile
import unittest
from unittest.mock import patch
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import torch
from src.modules.train.progress import ProgressListener, ProgressReporter
from src.web_console.train_monitor import training_progress, legacy_rows
from tests import test_web_console as webtests


class TrainProgressTests(unittest.TestCase):
    def test_progress_is_independent_of_log_level_and_final_flush(self) -> None:
        from logging.handlers import QueueHandler
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'progress.json'
            events = queue.Queue()
            listener = ProgressListener(events, path, [('g0', 0)], 5, [])
            logger = logging.getLogger('test.progress')
            logger.handlers = [QueueHandler(events)]; logger.setLevel(logging.ERROR); logger.propagate = False
            listener.start()
            reporter = ProgressReporter(logger, 'g0', torch.device('cpu'), 5)
            reporter.update(force=True, state='training', epoch=1)
            reporter.update(force=True, state='completed', validation_loss=float('inf'))
            listener.stop(); listener.flush()
            value = json.loads(path.read_text())['experts'][0]
            self.assertEqual(value['state'], 'completed')
            self.assertEqual(value['epoch'], 1)
            self.assertIsNone(value['validation_loss'])

    def test_legacy_log_reports_only_known_fields(self) -> None:
        text = ('2026-11-09 16:00:00+0800 parallel.py:1 INFO 领取专家任务：pid=1；device=cuda:0；group=g0\n'
                '2026-11-09 16:00:10+0800 api.py:1 DEBUG 专家 g0 epoch=2；train_loss=0.5；validation_loss=NA；checkpoint_loss=0.5；device=cuda:0\n'
                '2026-11-09 16:00:20+0800 api.py:1 INFO 专家 g0 完成：device=cuda:0；targets=2')
        row = legacy_rows(text, 5)[0]
        self.assertEqual(row['elapsed_seconds'], 20)
        self.assertEqual(row['epoch'], 2)
        self.assertNotIn('completed', row)

    def test_authenticated_snapshot_and_path_constraint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / 'data/5_train/d1/.1#20261109_160000#42.pending'
            directory.mkdir(parents=True)
            (directory / 'resolved_config.json').write_text(json.dumps({'module_name': 'train', 'dataset_id': 'd1',
                'dataset_name': 'd1', 'run_id': '1', 'run_timestamp': '20261109_160000', 'global': {'seed': 42}, 'module': {'epochs': 5}}))
            (directory / 'training_progress.json').write_text(json.dumps({'experts': [{'group_id': 'g0', 'state': 'training'}]}))
            with patch('src.web_console.train_monitor.gpu_snapshot', return_value=[]):
                self.assertEqual(training_progress(root)['experts'][0]['group_id'], 'g0')
                with self.assertRaises(ValueError):
                    training_progress(root, '../outside')
                with webtests.test_server(root) as server:
                    base = f'http://127.0.0.1:{server.server_port}'
                    with self.assertRaises(HTTPError) as error:
                        urlopen(base + '/api/train/progress')
                    self.assertEqual(error.exception.code, 401)
                    cookie = webtests.WebConsoleTests._login(base, 'fixture-token')
                    response = urlopen(Request(base + '/api/train/progress', headers={'Cookie': cookie}))
                    self.assertEqual(json.load(response)['mode'], 'events')
