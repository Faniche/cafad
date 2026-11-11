"""test_process_stop.py | 2026-11-09 | 实际进程组及排队任务停止回归 | Codex."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
import multiprocessing as mp
from src.utils.process_pool import InterruptibleProcessPoolExecutor
from src.web_console.service import JobManager


def long_task(directory: str, index: int) -> None:
    Path(directory, f'task-{index}').write_text(str(os.getpid()))
    time.sleep(60)


def fixture(mode: str, directory: str) -> None:
    """隔离进程组中运行的长任务；只用于停止回归，不使用真实数据。"""
    if mode == 'pool':
        with InterruptibleProcessPoolExecutor(max_workers=2, mp_context=mp.get_context('spawn')) as pool:
            futures = [pool.submit(long_task, directory, index) for index in range(10)]
            for future in futures:
                future.result()
    else:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        Path(directory, mode).write_text(str(os.getpid()))
        if mode == 'tree':
            subprocess.Popen([sys.executable, '-m', 'tests.test_process_stop', '--fixture', 'child', directory])
        time.sleep(60)


@unittest.skipUnless(os.name == 'posix', '实际进程组信号在 Linux 服务器验证')
class ProcessStopTests(unittest.TestCase):
    def run_case(self, mode: str) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            manager = JobManager(root, lambda *args: None)
            job = manager.start((sys.executable, '-m', 'tests.test_process_stop', '--fixture', mode, directory), 'train')
            try:
                deadline = time.monotonic() + 15
                while len(list(Path(directory).iterdir())) < 2 and time.monotonic() < deadline:
                    time.sleep(0.1)
                files = list(Path(directory).iterdir())
                self.assertEqual(len(files), 2)
                pids = [int(path.read_text()) for path in files]
                manager.stop(job.job_id)
                deadline = time.monotonic() + 12
                while manager.snapshot(job.job_id)['status'] in ('running', 'stopping') and time.monotonic() < deadline:
                    time.sleep(0.1)
                self.assertEqual(manager.snapshot(job.job_id)['status'], 'cancelled')
                self.assertEqual(len(list(Path(directory).iterdir())), 2, '停止后不能领取新任务')
                for pid in pids:
                    state = Path(f'/proc/{pid}/stat')
                    self.assertTrue(not state.exists() or state.read_text().split(') ')[1][0] == 'Z', f'残留工作进程 {pid}')
            finally:
                if job.process is not None:
                    try:
                        os.killpg(job.process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def test_pool_cancels_queued_tasks_and_stops_workers(self) -> None:
        self.run_case('pool')

    def test_ignoring_signals_parent_and_child_are_force_killed(self) -> None:
        self.run_case('tree')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--fixture':
        fixture(sys.argv[2], sys.argv[3])
    else:
        unittest.main()
