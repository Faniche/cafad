"""process_pool.py | 2026-11-09 | 可中断进程池，避免停止后继续领取排队任务 | Codex."""
from concurrent.futures import ProcessPoolExecutor as StandardProcessPoolExecutor
from collections.abc import Callable
import os
import signal
from .process_control import terminate_workers


def _exit_worker(signum: int, frame: object) -> None:
    """退出进程而非抛异常，避免 Executor 捕获 KeyboardInterrupt 后继续取任务。"""
    os._exit(128 + signum)


def _initialize_worker(initializer: Callable | None, args: tuple) -> None:
    signal.signal(signal.SIGINT, _exit_worker)
    if hasattr(signal, 'SIGBREAK'):
        signal.signal(signal.SIGBREAK, _exit_worker)
    if initializer is not None:
        initializer(*args)


class InterruptibleProcessPoolExecutor(StandardProcessPoolExecutor):
    """正常退出保持标准行为；异常退出取消排队任务并有界终止工作进程。"""

    def __init__(self, max_workers: int | None = None, mp_context: object = None,
                 initializer: Callable | None = None, initargs: tuple = (), **kwargs: object) -> None:
        super().__init__(max_workers=max_workers, mp_context=mp_context, initializer=_initialize_worker,
                         initargs=(initializer, initargs), **kwargs)

    def __exit__(self, exc_type: type | None, exc_value: BaseException | None, traceback: object) -> bool:
        if exc_type is None:
            return super().__exit__(exc_type, exc_value, traceback)
        # Python 3.13 没有公开 terminate_workers；集中兼容，不在业务模块各自访问内部字段。
        processes = list((self._processes or {}).values())
        self.shutdown(wait=False, cancel_futures=True)
        terminate_workers(processes)
        return False
