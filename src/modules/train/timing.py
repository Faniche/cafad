"""timing.py | 2026-11-10 | 训练专家阶段耗时的低干扰统计 | Codex."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter

import torch


class StageTimer:
    """累计主机提交时间和CUDA事件时间；每轮仅同步一次GPU。"""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.host_seconds: dict[str, float] = defaultdict(float)
        self.events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = defaultdict(list)

    @contextmanager
    def stage(self, name: str, gpu: bool = False) -> Iterator[None]:
        """记录调用方墙钟时间；仅标记GPU的阶段记录不阻塞事件对。"""
        started = perf_counter()
        begin = end = None
        if gpu and self.device.type == 'cuda':
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
        try:
            yield
        finally:
            if end is not None and begin is not None:
                end.record()
                self.events[name].append((begin, end))
            self.host_seconds[name] += perf_counter() - started

    def summary_seconds(self) -> dict[str, tuple[float, float | None]]:
        """返回阶段的主机秒数和GPU秒数；CUDA只在此处同步一次。"""
        gpu_seconds: dict[str, float] = {}
        if self.device.type == 'cuda' and self.events:
            torch.cuda.synchronize(self.device)
            gpu_seconds = {name: sum(begin.elapsed_time(end) for begin, end in pairs) / 1000
                           for name, pairs in self.events.items()}
        names = set(self.host_seconds) | set(gpu_seconds)
        return {name: (self.host_seconds.get(name, 0.0), gpu_seconds.get(name)) for name in sorted(names)}


def format_timing(summary: dict[str, tuple[float, float | None]]) -> str:
    """将阶段统计压缩为可检索的DEBUG日志字段。"""
    return '；'.join(f'{name}=host:{host:.3f}s' + (f',gpu:{gpu:.3f}s' if gpu is not None else '')
                     for name, (host, gpu) in summary.items())


def dominant_timing(summary: dict[str, tuple[float, float | None]]) -> str:
    """指出主机与GPU侧各自累计最久的阶段，便于直接比较瓶颈。"""
    if not summary:
        return '无验证样本'
    host_name, (host_seconds, _) = max(summary.items(), key=lambda item: item[1][0])
    gpu_items = [(name, gpu) for name, (_, gpu) in summary.items() if gpu is not None]
    if not gpu_items:
        return f'host最长={host_name}:{host_seconds:.3f}s；GPU计时不可用'
    gpu_name, gpu_seconds = max(gpu_items, key=lambda item: item[1])
    return f'host最长={host_name}:{host_seconds:.3f}s；GPU最长={gpu_name}:{gpu_seconds:.3f}s'
