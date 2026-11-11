"""progress.py | 2026-11-09 | 限频训练事件与主进程状态快照 | Codex."""
import logging
from logging.handlers import QueueListener
from pathlib import Path
import time
import math
import os
import torch
from src.app.artifacts import write_json


class ProgressReporter:
    """复用工作进程日志队列；消息不携带模型或观测。"""
    def __init__(self, logger: logging.Logger, group: str, device: torch.device, epochs: int) -> None:
        self.logger, self.group, self.device = logger, group, device
        self.started = time.time()
        self.last_sent = 0.0
        self.values = {'group_id': group, 'device': str(device), 'pid': os.getpid(), 'epochs': epochs,
                       'started_at': self.started, 'epoch': 0, 'completed': 0, 'total': 0}

    def update(self, force: bool = False, **values: object) -> None:
        self.values.update(values)
        now = time.time()
        if not force and now - self.last_sent < 1:
            return
        self.last_sent = now
        self.values.update(updated_at=now, elapsed_seconds=now - self.started)
        if self.device.type == 'cuda':
            self.values.update(memory_allocated_mib=torch.cuda.memory_allocated(self.device) / 2 ** 20,
                               memory_reserved_mib=torch.cuda.memory_reserved(self.device) / 2 ** 20)
        payload = {key: None if isinstance(value, float) and not math.isfinite(value) else value
                   for key, value in self.values.items()}
        # 监控事件独立于用户选择的文本日志等级。
        record = self.logger.makeRecord(self.logger.name, logging.INFO, __file__, 0, '训练进度', (), None,
                                        extra={'training_progress': payload})
        self.logger.handle(record)


class ProgressListener(QueueListener):
    """同一监听线程聚合进度和日志；最终快照随训练产物发布。"""
    def __init__(self, queue: object, path: Path, tasks: list[tuple[str, int]], epochs: int,
                 handlers: list[logging.Handler]) -> None:
        super().__init__(queue, *handlers, respect_handler_level=True)
        self.path = path
        self.rows = {group: {'group_id': group, 'expert_id': expert, 'state': 'queued', 'epochs': epochs,
                             'epoch': 0, 'elapsed_seconds': 0} for group, expert in tasks}
        self.last_write = 0.0
        self.flush()

    def flush(self) -> None:
        now = time.time()
        write_json(self.path, {'schema_version': 1, 'updated_at': now, 'experts': list(self.rows.values())})
        self.last_write = now

    def handle(self, record: logging.LogRecord) -> None:
        payload = getattr(record, 'training_progress', None)
        if payload is None:
            super().handle(record)
            return
        group = payload['group_id']
        self.rows.setdefault(group, {'group_id': group}).update(payload)
        if time.time() - self.last_write >= 1 or payload.get('state') in ('completed', 'failed'):
            self.flush()
