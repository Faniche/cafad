"""parallel.py | 2026-11-09 | 每设备常驻进程从公共专家池取任务 | Codex."""
from concurrent.futures import as_completed
from src.utils.process_pool import InterruptibleProcessPoolExecutor as ProcessPoolExecutor
from dataclasses import dataclass
import argparse
import logging
from logging.handlers import QueueHandler
import multiprocessing as mp
import os
from pathlib import Path
import pandas as pd
import torch
from src.app.tables import read_table
from .progress import ProgressListener


@dataclass
class WorkerState:
    """只在子进程初始化，观测每进程加载一次，专家之间无模型共享。"""
    device: torch.device
    assignments: pd.DataFrame
    validation: pd.DataFrame
    references: pd.DataFrame
    sequences: dict[str, pd.DataFrame]
    args: argparse.Namespace
    seed: int
    output: Path
    logger: logging.Logger


_state: WorkerState | None = None


def initialize_worker(devices: object, logs: object, group_manifest: Path, segment_manifest: Path, inject_manifest: Path,
                      args: argparse.Namespace, seed: int, output: Path, threads: int, level: int) -> None:
    """spawn 后绑定独占 GPU，限制 CPU 线程并把日志交回主进程。"""
    from .api import _load_sequences, _with_injected_labels
    global _state
    device = torch.device(devices.get())
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    torch.set_num_threads(threads)
    logger = logging.getLogger('cafad.train.worker')
    logger.handlers = [QueueHandler(logs)]; logger.setLevel(level); logger.propagate = False
    data = group_manifest.parent / 'data'
    assignments = read_table(data, 'train_assignments', keep_default_na=False)
    validation = read_table(data, 'validation_assignments', keep_default_na=False)
    references = read_table(data, 'references', keep_default_na=False)
    injected_segments = read_table(inject_manifest.parent / 'data', 'segments', keep_default_na=False)
    assignments = _with_injected_labels(assignments, injected_segments)
    validation = _with_injected_labels(validation, injected_segments)
    sequences, _ = _load_sequences(segment_manifest, inject_manifest)
    needed = set(assignments.segment_id) | set(validation.segment_id) | set(references.segment_id)
    sequences = {key: value for key, value in sequences.items() if key in needed}
    _state = WorkerState(device, assignments, validation, references, sequences, args, seed, output, logger)
    logger.info('训练工作进程就绪：pid=%d；device=%s；CPU线程=%d', os.getpid(), device, threads)


def train_task(group_id: str, expert_id: int) -> dict[str, object]:
    """空闲进程领取一个专家，独立初始化随机状态与模型。"""
    from .api import _train_expert, _seed
    import random
    import numpy as np
    state = _state
    if state is None:
        raise RuntimeError('训练工作进程未初始化')
    seed = _seed(state.seed, group_id) % (2 ** 32)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    state.logger.info('领取专家任务：pid=%d；device=%s；group=%s', os.getpid(), state.device, group_id)
    try:
        return _train_expert(group_id, expert_id, state.assignments, state.validation, state.references,
                             state.sequences, state.args, state.seed, state.output / group_id, state.logger, state.device)
    except Exception as exc:
        record = state.logger.makeRecord(state.logger.name, logging.ERROR, __file__, 0, '专家训练失败', (), None,
                    extra={'training_progress': {'group_id': group_id, 'state': 'failed', 'error': str(exc)}})
        state.logger.handle(record)
        raise


def train_pool(tasks: list[tuple[str, int]], devices: list[torch.device], group_manifest: Path, segment_manifest: Path,
               inject_manifest: Path,
               args: argparse.Namespace, seed: int, output: Path, logger: logging.Logger, workers: int) -> list[dict[str, object]]:
    """一个设备一个 spawn 进程，所有专家共享 executor 任务池；异常向上抛出。"""
    context = mp.get_context('spawn')
    count = min(len(devices), len(tasks), workers)
    device_queue, logs = context.Queue(), context.Queue()
    for device in devices[:count]:
        device_queue.put(str(device))
    listener = ProgressListener(logs, output.parent / 'training_progress.json', tasks, args.epochs, logger.handlers)
    listener.start()
    try:
        with ProcessPoolExecutor(max_workers=count, mp_context=context, initializer=initialize_worker,
                                 initargs=(device_queue, logs, group_manifest, segment_manifest, inject_manifest, args, seed,
                                           output, max(1, workers // count), logger.level)) as executor:
            futures = [executor.submit(train_task, group, expert) for group, expert in tasks]
            return [future.result() for future in as_completed(futures)]
    finally:
        listener.stop()
        listener.flush()
        device_queue.close(); logs.close()
