"""process_control.py | 2026-11-09 | 统一任务进程组创建、中断及超时清理 | Codex."""
from collections.abc import Callable, Iterable
import multiprocessing as mp
import os
import signal
import subprocess
import time


def process_group_options() -> dict[str, object]:
    """每个正式任务独立进程组，避免停止信号影响网页和其他任务。"""
    return {'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == 'nt' else {'start_new_session': True}


def interrupt_process_group(process: subprocess.Popen[str]) -> None:
    """向主进程和同组子进程广播中断。"""
    try:
        if os.name == 'nt':
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError:
        return


def stop_process_group(process: subprocess.Popen[str], notify: Callable[[str], None],
                       interrupt: Callable[[subprocess.Popen[str]], None] = interrupt_process_group) -> None:
    """广播中断；2秒未退出则终止，继续等待3秒后强杀整个任务组。"""
    try:
        interrupt(process)
    except OSError as exc:
        notify(f'发送中断失败，继续尝试终止进程树：{exc}')
    for timeout, next_signal in ((2.0, signal.SIGTERM), (3.0, getattr(signal, 'SIGKILL', signal.SIGTERM))):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if os.name == 'nt':
                if process.poll() is not None:
                    return
            else:
                try:
                    os.killpg(process.pid, 0)
                except ProcessLookupError:
                    return
            time.sleep(0.1)
        notify(f'中断等待超时，升级停止进程组：{next_signal.name}')
        if os.name == 'nt':
            subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                           capture_output=True, timeout=10, check=False)
            return
        try:
            os.killpg(process.pid, next_signal)
        except ProcessLookupError:
            return


def terminate_workers(processes: Iterable[mp.Process]) -> None:
    """统一清理已持有的工作进程：先全部终止，再有界等待和强杀。"""
    workers = list(processes)
    for process in workers:
        if process.is_alive():
            process.terminate()
    deadline = time.monotonic() + 2.0
    for process in workers:
        process.join(timeout=max(0.0, deadline - time.monotonic()))
    for process in workers:
        if process.is_alive():
            process.kill()
    for process in workers:
        process.join(timeout=0.1)
