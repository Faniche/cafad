"""precision.py | 2026-11-10 | 训练与冻结评分共用的CUDA混合精度策略 | Codex."""
from collections.abc import Iterator
from contextlib import contextmanager

import torch


def validate_precision(precision: str, device: torch.device) -> None:
    """不支持指定精度时明确失败，不改变用户选择或回退设备。"""
    if precision not in {'fp32', 'fp16', 'bf16'}:
        raise ValueError(f'未知计算精度：{precision}')
    if precision == 'fp32':
        return
    if device.type != 'cuda':
        raise ValueError('FP16/BF16 AMP仅支持CUDA；显式CPU运行请指定fp32')
    if precision == 'bf16':
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported(including_emulation=False):
                raise RuntimeError(f'CUDA_DEVICE_UNAVAILABLE: {device} 不支持原生BF16，请显式选择fp16或fp32')


@contextmanager
def precision_context(precision: str, device: torch.device) -> Iterator[None]:
    """仅包围前向和损失；AMP下避开可能绕过autocast的融合Transformer推理路径。"""
    enabled = precision != 'fp32'
    dtype = torch.float16 if precision == 'fp16' else torch.bfloat16
    fastpath = torch.backends.mha.get_fastpath_enabled()
    try:
        if enabled:
            torch.backends.mha.set_fastpath_enabled(False)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled):
            yield
    finally:
        if enabled:
            torch.backends.mha.set_fastpath_enabled(fastpath)
