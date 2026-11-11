"""train_util.py | 2026-11-11 | Web 控制台 train 队列与自动后续实验调度 | Codex."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
import time
from typing import Protocol
from uuid import uuid4


FOLLOW_UP_MODULE = {'train': 'calibrate', 'calibrate': 'detect', 'detect': 'evaluate'}


class JobStarter(Protocol):
    """队列复用 JobManager 时所需的最小接口，避免复制进程控制逻辑。"""

    def start(self, command: tuple[str, ...], module_name: str) -> object:
        """启动一个唯一的网页后台任务。"""

    def snapshot(self, job_id: str) -> dict[str, object]:
        """读取已启动任务的状态与发布产物。"""


@dataclass
class QueuedTask:
    """一个尚未启动或已完成的队列项目；后续步骤在父任务启动时预插入。"""

    task_id: str
    module_name: str
    auto_go: bool
    command: tuple[str, ...] | None = None
    source_command: tuple[str, ...] | None = None
    parent_job_id: str | None = None
    job_id: str | None = None
    state: str = 'queued'
    message: str = ''


class TrainTaskQueue:
    """串行消费 train 任务，并在成功链路前插 calibrate、detect、evaluate。"""

    def __init__(self, manager: JobStarter) -> None:
        self._manager = manager
        self._pending: deque[QueuedTask] = deque()
        self._completed: deque[QueuedTask] = deque(maxlen=100)
        self._active: QueuedTask | None = None
        self._condition = threading.Condition()
        self._worker: threading.Thread | None = None

    def enqueue_train(self, command: tuple[str, ...], auto_go: bool) -> dict[str, object]:
        """追加用户配置的 train；空闲时立即由同一个调度线程启动。"""
        task = QueuedTask(task_id=uuid4().hex, module_name='train', command=command, source_command=command, auto_go=auto_go)
        with self._condition:
            self._pending.append(task)
            self._ensure_worker()
            self._condition.notify_all()
            return {'task': self._serialize(task), 'queue': self._snapshot_locked()}

    def has_pending(self) -> bool:
        """返回是否仍有活动或排队任务，供控制台重启保护使用。"""
        with self._condition:
            return self._active is not None or bool(self._pending)

    def active_job_id(self) -> str | None:
        """返回当前队列实际运行的网页任务，供 train 页继续展示日志与停止按钮。"""
        with self._condition:
            return self._active.job_id if self._active is not None else None

    def snapshot(self) -> dict[str, object]:
        """返回有限的队列状态，便于页面给出排队确认。"""
        with self._condition:
            return self._snapshot_locked()

    def _ensure_worker(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._worker = threading.Thread(target=self._run, name='cafad-train-queue', daemon=True)
        self._worker.start()

    def _run(self) -> None:
        """只负责队列次序和产物依赖；实际子进程生命周期完全委托 JobManager。"""
        while True:
            with self._condition:
                active = self._active
                if active is None and not self._pending:
                    self._worker = None
                    return
            if active is not None:
                self._finish_active(active)
                time.sleep(0.2)
                continue
            self._start_next()

    def _finish_active(self, task: QueuedTask) -> None:
        if task.job_id is None:
            return
        try:
            snapshot = self._manager.snapshot(task.job_id)
        except KeyError:
            snapshot = {'status': 'failed', 'return_code': None}
        status = snapshot.get('status')
        if status in {'running', 'stopping'}:
            return
        with self._condition:
            if self._active is not task:
                return
            task.state = str(status)
            task.message = '已完成' if status == 'succeeded' else '上一步未成功，后续自动步骤已跳过'
            self._active = None
            self._completed.append(task)
            self._condition.notify_all()

    def _start_next(self) -> None:
        with self._condition:
            if self._active is not None or not self._pending:
                return
            task = self._pending[0]
        if task.command is None:
            resolution = self._resolve_follow_up(task)
            if resolution == 'waiting':
                time.sleep(0.2)
                return
            if resolution == 'skipped':
                with self._condition:
                    if self._pending and self._pending[0] is task:
                        self._pending.popleft()
                        self._completed.append(task)
                return
        assert task.command is not None
        try:
            job = self._manager.start(task.command, task.module_name)
        except RuntimeError:
            # 可能有普通网页模块尚未结束；等待其自然释放 JobManager 再试。
            time.sleep(0.2)
            return
        job_id = getattr(job, 'job_id', None)
        if not isinstance(job_id, str):
            raise RuntimeError('网页任务管理器未返回有效 job_id')
        with self._condition:
            if not self._pending or self._pending[0] is not task:
                raise RuntimeError('训练队列次序异常')
            self._pending.popleft()
            task.job_id = job_id
            task.state = 'running'
            task.message = '正在运行'
            self._active = task
            next_module = FOLLOW_UP_MODULE.get(task.module_name) if task.auto_go else None
            if next_module is not None:
                self._pending.appendleft(QueuedTask(
                    task_id=uuid4().hex,
                    module_name=next_module,
                    auto_go=True,
                    source_command=task.command,
                    parent_job_id=job_id,
                    message=f'等待 {task.module_name} 成功发布产物',
                ))
            self._condition.notify_all()

    def _resolve_follow_up(self, task: QueuedTask) -> str:
        """仅在父任务成功且可唯一取得其 manifest 时，组装下一步标准命令。"""
        if task.parent_job_id is None or task.source_command is None:
            task.state = 'skipped'
            task.message = '缺少上游任务，已跳过'
            return 'skipped'
        try:
            parent = self._manager.snapshot(task.parent_job_id)
        except KeyError:
            task.state = 'skipped'
            task.message = '上游任务状态已丢失，已跳过'
            return 'skipped'
        if parent.get('status') in {'running', 'stopping'}:
            return 'waiting'
        manifest = parent.get('artifact_manifest')
        if parent.get('status') != 'succeeded' or not isinstance(manifest, str) or not manifest:
            task.state = 'skipped'
            task.message = '上游任务失败、停止或未找到唯一发布产物，已跳过'
            return 'skipped'
        task.command = self._follow_up_command(task.source_command, task.module_name, manifest)
        task.message = '等待启动'
        return 'ready'

    @staticmethod
    def _follow_up_command(source: tuple[str, ...], module_name: str, manifest: str) -> tuple[str, ...]:
        """沿用原 train 命令的解释器和全局参数，只替换模块及其确定性依赖。"""
        try:
            boundary = source.index('-m')
        except ValueError as exc:
            raise ValueError('训练命令缺少 -m，无法生成 auto_go 后续任务') from exc
        prefix = source[:boundary]
        gpus = TrainTaskQueue._option_value(source, '--gpus') or '0,1,2,3'
        command = [*prefix, '-m', module_name, '--input', manifest]
        if module_name == 'calibrate':
            command.extend(('--gpus', gpus))
        elif module_name == 'detect':
            command.extend(('--split', 'test', '--gpus', gpus))
        elif module_name != 'evaluate':
            raise ValueError(f'不支持的 auto_go 后续模块：{module_name}')
        return tuple(command)

    @staticmethod
    def _option_value(command: tuple[str, ...], option: str) -> str | None:
        try:
            index = command.index(option)
        except ValueError:
            return None
        return command[index + 1] if index + 1 < len(command) else None

    def _snapshot_locked(self) -> dict[str, object]:
        tasks = [*self._completed, *([self._active] if self._active is not None else []), *self._pending]
        return {'active_task_id': self._active.task_id if self._active is not None else None,
                'tasks': [self._serialize(task) for task in tasks]}

    @staticmethod
    def _serialize(task: QueuedTask) -> dict[str, object]:
        return {'task_id': task.task_id, 'module_name': task.module_name, 'auto_go': task.auto_go,
                'job_id': task.job_id, 'state': task.state, 'message': task.message}
