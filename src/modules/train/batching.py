"""batching.py | 2026-11-09 | 变长片段组批、填充与独立参考集合的批量编码 | Codex."""
from collections.abc import Iterator

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence

from .model import ExpertModel


def collate_frames(frames: list[pd.DataFrame], stats: dict, device: torch.device
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """只补到当前微批最大长度，不删除或重采样有效记录；True 表示 padding。"""
    if not frames or any(frame.empty for frame in frames):
        raise ValueError('编码微批及其中每个片段均不能为空')
    lengths = [len(frame) for frame in frames]
    shape = (len(frames), max(lengths))
    space = np.zeros((*shape, 4), dtype=np.float32)
    kinematic = np.zeros((*shape, 3), dtype=np.float32)
    times = np.zeros(shape, dtype=np.float32)
    mask = np.ones(shape, dtype=bool)
    mean = np.asarray(stats['mean'], dtype=np.float32)
    std = np.asarray(stats['std'], dtype=np.float32)
    for index, (frame, length) in enumerate(zip(frames, lengths)):
        normalized = (frame[stats['names']].to_numpy(dtype=np.float32) - mean) / std
        heading = np.deg2rad(frame['heading_deg'].to_numpy(dtype=np.float32))
        space[index, :length] = normalized[:, :4]
        kinematic[index, :length] = np.column_stack((normalized[:, 4], np.sin(heading), np.cos(heading)))
        timestamps = frame['time_ns'].to_numpy(dtype=np.int64)
        time_mode = stats.get('time_encoding', 'legacy_absolute_float32')
        if time_mode == 'relative_ns_v1':
            times[index, :length] = ((timestamps - timestamps[0]) / 1e9).astype(np.float32)
        elif time_mode == 'legacy_absolute_float32':
            times[index, :length] = (timestamps.astype(np.float64) / 1e9).astype(np.float32)
        else:
            raise ValueError(f'未知的时间编码版本：{time_mode}')
        times[index, length:] = times[index, length - 1]
        mask[index, :length] = False
    return tuple(torch.from_numpy(value).to(device) for value in (space, kinematic, times, mask))


def encode_frames(model: ExpertModel, frames: list[pd.DataFrame], stats: dict,
                  device: torch.device, micro_batch_size: int) -> torch.Tensor:
    """按长度分批编码，再恢复输入顺序；保留全部梯度，不缓存跨更新表示。"""
    if not frames or micro_batch_size < 1:
        raise ValueError('编码输入不能为空，微批大小必须为正')
    order = sorted(range(len(frames)), key=lambda index: len(frames[index]))
    outputs = []
    for start in range(0, len(order), micro_batch_size):
        selected = [frames[index] for index in order[start:start + micro_batch_size]]
        outputs.append(model.encode(*collate_frames(selected, stats, device)))
    inverse = torch.argsort(torch.tensor(order, device=device))
    return torch.cat(outputs, dim=0).index_select(0, inverse)


def length_batches(targets: pd.DataFrame, sequences: dict[str, pd.DataFrame],
                   micro_batch_size: int, seed: int | None = None) -> list[pd.DataFrame]:
    """相邻长度组成微批；训练随机打乱等长样本、微批顺序和批内顺序，每条恰好一次。"""
    if micro_batch_size < 1:
        raise ValueError('微批大小必须为正')
    rng = np.random.default_rng(seed) if seed is not None else None
    order = rng.permutation(len(targets)).tolist() if rng is not None else list(range(len(targets)))
    ids = targets['segment_id'].astype(str).tolist()
    order.sort(key=lambda index: len(sequences[ids[index]]))
    batches = [order[start:start + micro_batch_size] for start in range(0, len(order), micro_batch_size)]
    if rng is not None:
        rng.shuffle(batches)
        for batch in batches:
            rng.shuffle(batch)
    return [targets.iloc[batch] for batch in batches]


def accumulation_windows(batches: list[pd.DataFrame], batch_size: int) -> Iterator[list[pd.DataFrame]]:
    """按实际目标数打包优化器更新；必要时拆微批，不要求两种批大小整除。"""
    if batch_size < 1:
        raise ValueError('有效批大小必须为正')
    window, count = [], 0
    for batch in batches:
        start = 0
        while start < len(batch):
            take = min(len(batch) - start, batch_size - count)
            window.append(batch.iloc[start:start + take])
            count += take
            start += take
            if count == batch_size:
                yield window
                window, count = [], 0
    if window:
        yield window


def reference_memories(model: ExpertModel, selections: list[list[str]], sequences: dict[str, pd.DataFrame],
                       stats: dict, device: torch.device, micro_batch_size: int) -> torch.Tensor:
    """同源排除已在 selections 完成；参考仅在各自 batch 行内相互注意。"""
    if not selections or any(not ids for ids in selections):
        raise ValueError('每个目标必须有非空正常参考集合')
    # 保留每次参考出现，不跨目标去重，避免额外引入共享 dropout 表示的训练变化。
    frames = [sequences[item] for ids in selections for item in ids]
    encoded = encode_frames(model, frames, stats, device, micro_batch_size)
    lengths = [len(ids) for ids in selections]
    grouped = pad_sequence(encoded.split(lengths), batch_first=True)
    mask = torch.arange(grouped.shape[1], device=device)[None, :] >= torch.tensor(lengths, device=device)[:, None]
    return model.memory(grouped, mask)
