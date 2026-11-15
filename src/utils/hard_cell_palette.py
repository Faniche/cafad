"""hard_cell_palette.py | 2026-11-15 | 硬时空单元的稳定可视化配色 | Codex."""
from __future__ import annotations

import hashlib

from matplotlib import colors


def cell_color(cell_key: str) -> str:
    """以稳定哈希映射硬单元；同一单元在任意航班图中保持同色。"""
    digest = hashlib.sha256(cell_key.encode('utf-8')).digest()
    hue = int.from_bytes(digest[:2], 'big') / 65535.0
    return colors.to_hex(colors.hsv_to_rgb((hue, 0.62, 0.84)))
