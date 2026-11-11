"""test_train_queue.py | 2026-11-11 | Web 训练队列调度测试 | Codex."""
from __future__ import annotations

from dataclasses import dataclass
import time
import unittest

from src.web_console.train_util import TrainTaskQueue


@dataclass
class _Job:
    job_id: str


class _Manager:
    """只模拟 JobManager 的启动与快照接口，不创建真实子进程。"""

    def __init__(self) -> None:
        self.started: list[tuple[str, tuple[str, ...], str]] = []
        self.jobs: dict[str, dict[str, object]] = {}

    def start(self, command: tuple[str, ...], module_name: str) -> _Job:
        job_id = f'job-{len(self.started)}'
        self.started.append((module_name, command, job_id))
        self.jobs[job_id] = {'status': 'running', 'artifact_manifest': None}
        return _Job(job_id)

    def snapshot(self, job_id: str) -> dict[str, object]:
        return dict(self.jobs[job_id])

    def finish(self, job_id: str, succeeded: bool = True) -> None:
        module = self.started[int(job_id.removeprefix('job-'))][0]
        self.jobs[job_id] = {
            'status': 'succeeded' if succeeded else 'failed',
            'artifact_manifest': f'/tmp/{module}/manifest.json' if succeeded else None,
        }


class TrainTaskQueueTests(unittest.TestCase):
    """auto_go 只能在成功链路前插后续步骤，失败不得继续运行。"""

    def _wait_for_starts(self, manager: _Manager, count: int) -> None:
        deadline = time.monotonic() + 3
        while len(manager.started) < count and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(len(manager.started), count)

    @staticmethod
    def _train_command(seed: str) -> tuple[str, ...]:
        return ('python', 'main.py', '--seed', seed, '-m', 'train', '--input', 'group/manifest.json', '--gpus', '0,1')

    def test_auto_chain_runs_before_later_train(self) -> None:
        manager = _Manager()
        queue = TrainTaskQueue(manager)
        queue.enqueue_train(self._train_command('1'), auto_go=True)
        queue.enqueue_train(self._train_command('2'), auto_go=False)
        self._wait_for_starts(manager, 1)
        manager.finish('job-0')
        self._wait_for_starts(manager, 2)
        self.assertEqual(manager.started[1][0], 'calibrate')
        self.assertIn('/tmp/train/manifest.json', manager.started[1][1])
        manager.finish('job-1')
        self._wait_for_starts(manager, 3)
        self.assertEqual(manager.started[2][0], 'detect')
        self.assertIn('--split', manager.started[2][1])
        manager.finish('job-2')
        self._wait_for_starts(manager, 4)
        self.assertEqual(manager.started[3][0], 'evaluate')
        manager.finish('job-3')
        self._wait_for_starts(manager, 5)
        self.assertEqual(manager.started[4][0], 'train')

    def test_failed_train_skips_auto_steps_then_runs_next_train(self) -> None:
        manager = _Manager()
        queue = TrainTaskQueue(manager)
        queue.enqueue_train(self._train_command('1'), auto_go=True)
        queue.enqueue_train(self._train_command('2'), auto_go=False)
        self._wait_for_starts(manager, 1)
        manager.finish('job-0', succeeded=False)
        self._wait_for_starts(manager, 2)
        self.assertEqual([entry[0] for entry in manager.started], ['train', 'train'])
        states = {entry['module_name']: entry['state'] for entry in queue.snapshot()['tasks']}
        self.assertEqual(states['calibrate'], 'skipped')

