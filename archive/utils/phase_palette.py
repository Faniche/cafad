"""phase_palette.py | 2026-11-15 | Sun 阶段标签的统一可视化配色 | Codex."""
from __future__ import annotations


SUN_PHASE_COLORS: dict[str, str] = {
    'GND': '#6b7280',
    'CL': '#16a34a',
    'LVL': '#2563eb',
    'CR': '#7c3aed',
    'DE': '#ea580c',
    'NA': '#dc2626',
}

_ALIASES = {
    'GROUND': 'GND', 'CLIMB': 'CL', 'LEVEL': 'LVL', 'LEVEL_FLIGHT': 'LVL',
    'CRUISE': 'CR', 'DESCENT': 'DE', 'UNKNOWN': 'NA',
}
DEFAULT_PHASE_COLOR = '#475569'


def phase_color(phase: str) -> str:
    """返回 Sun 阶段或兼容历史英文阶段名的固定颜色。"""
    normalized = _ALIASES.get(str(phase).upper(), str(phase).upper())
    return SUN_PHASE_COLORS.get(normalized, DEFAULT_PHASE_COLOR)
