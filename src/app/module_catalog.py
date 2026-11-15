"""module_catalog.py | 2026-11-05 | CAFAD 模块顺序与产物目录命名 | Codex."""
from __future__ import annotations


MODULE_SEQUENCE = (
    'preprocess',
    'split',
    'segment',
    'group',
    'inject',
    'train',
    'calibrate',
    'detect',
    'evaluate',
)


def artifact_module_directory(module_name: str) -> str:
    """返回磁盘产物目录名；编号固定表达正式实验流水线顺序。"""
    try:
        return f'{MODULE_SEQUENCE.index(module_name)}_{module_name}'
    except ValueError as exc:
        raise ValueError(f'未知 CAFAD 模块：{module_name}') from exc
