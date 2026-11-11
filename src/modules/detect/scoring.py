"""scoring.py | 2026-11-09 | 校准与检测共享的多设备动态评分池 | Codex."""
from concurrent.futures import wait, FIRST_COMPLETED
from src.utils.process_pool import InterruptibleProcessPoolExecutor as ProcessPoolExecutor
from dataclasses import dataclass
import json
import logging
import multiprocessing as mp
import os
from pathlib import Path
from typing import Iterator
import numpy as np
import pandas as pd
import torch
from src.modules.train.api import _encode
from src.modules.train.model import ExpertModel
from src.utils.precision import precision_context, validate_precision


@dataclass
class Scorer:
    """进程独占设备，仅缓存最近一个专家，限制显存占用。"""
    device: torch.device
    model: ExpertModel | None = None
    memory: torch.Tensor | None = None
    stats: dict | None = None
    folder: Path | None = None
    precision: str = 'fp32'

    def score(self, folder: Path, samples: list[tuple[int, pd.DataFrame]]) -> list[tuple[int, float]]:
        if folder != self.folder:
            self.model = None; self.memory = None
            checkpoint = torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)
            # 旧检查点没有真实精度记录，其历史计算均为FP32。
            self.precision = checkpoint.get('precision', 'fp32')
            validate_precision(self.precision, self.device)
            model = ExpertModel(**checkpoint['architecture']).to(self.device)
            model.load_state_dict(checkpoint['state_dict']); model.eval()
            self.model = model
            self.memory = torch.from_numpy(np.load(folder / 'memory.npy', allow_pickle=False)).to(self.device)
            self.stats = json.loads((folder / 'stats.json').read_text(encoding='utf-8'))
            self.folder = folder
        with torch.inference_mode(), precision_context(self.precision, self.device):
            return [(index, float(torch.sigmoid(self.model.logits(_encode(self.model, frame, self.stats, self.device),
                                                                 self.memory).float()).item())) for index, frame in samples]


_scorer: Scorer | None = None


def initialize_scorer(devices: object, threads: int) -> None:
    """spawn 进程绑定独占设备，避免 CUDA fork 与 Python GIL 争用。"""
    global _scorer
    device = torch.device(devices.get())
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    torch.set_num_threads(threads)
    _scorer = Scorer(device)


def score_task(folder: Path, samples: list[tuple[int, pd.DataFrame]]) -> tuple[list[tuple[int, float]], str, int]:
    if _scorer is None:
        raise RuntimeError('评分进程未初始化')
    return _scorer.score(folder, samples), str(_scorer.device), os.getpid()


def iter_tasks(scores: pd.DataFrame, sequences: dict[str, pd.DataFrame], trained: Path,
               task_size: int) -> Iterator[tuple[Path, list[tuple[int, pd.DataFrame]]]]:
    """惰性构建任务载荷，不向每个进程复制整个数据集。批内仍逐片段前向。"""
    for group, frame in scores.groupby('group_id', sort=True):
        for start in range(0, len(frame), task_size):
            batch = frame.iloc[start:start + task_size]
            yield trained.parent / 'models' / str(group), [(int(i), sequences[str(row.segment_id)]) for i, row in batch.iterrows()]


def score_pairs(scores: pd.DataFrame, sequences: dict[str, pd.DataFrame], trained: Path, devices: list[torch.device],
                workers: int, task_size: int, logger: logging.Logger) -> pd.DataFrame:
    """有界公共任务池；完成顺序不影响按原始 pair 索引恢复的结果。"""
    if task_size < 1 or workers < 1 or scores.empty:
        raise ValueError('评分任务大小、进程配额必须为正，评分对不能为空')
    result = scores.copy(); result['score'] = np.nan
    tasks = iter_tasks(scores, sequences, trained, task_size)
    task_count = sum((size + task_size - 1) // task_size for size in scores.groupby('group_id').size())
    count = min(len(devices), workers, task_count)
    logger.info('评分调度：pairs=%d；tasks=%d；processes=%d；devices=%s；task_size=%d',
                len(scores), task_count, count, [str(item) for item in devices[:count]], task_size)
    if count == 1:
        scorer = Scorer(devices[0])
        for folder, samples in tasks:
            for index, score in scorer.score(folder, samples):
                result.at[index, 'score'] = score
    else:
        context = mp.get_context('spawn'); queue = context.Queue()
        for device in devices[:count]:
            queue.put(str(device))
        try:
            with ProcessPoolExecutor(max_workers=count, mp_context=context, initializer=initialize_scorer,
                                     initargs=(queue, max(1, workers // count))) as executor:
                pending = set()
                exhausted = False
                while pending or not exhausted:
                    while not exhausted and len(pending) < count * 2:
                        task = next(tasks, None)
                        if task is None:
                            exhausted = True
                        else:
                            pending.add(executor.submit(score_task, *task))
                    if not pending:
                        break
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        values, device, pid = future.result()
                        for index, score in values:
                            result.at[index, 'score'] = score
                        logger.info('评分任务完成：pid=%d；device=%s；pairs=%d', pid, device, len(values))
        finally:
            queue.close(); queue.join_thread()
    if not np.isfinite(result['score']).all() or not result['score'].between(0, 1).all():
        raise ValueError('评分对缺失或异常分数超出 [0,1]')
    return result
