"""contracts.py | 2026-11-03 | 模块运行与产物数据契约 | Codex."""
from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class RunContext:
    """一次模块运行共享的只读配置和当前暂存目录。"""

    project_root: Path
    module_name: str
    dataset_id: str
    dataset_name: str
    run_id: str
    run_timestamp: str
    seed: int
    workers: int
    staging_dir: Path
    final_dir: Path
    logger: logging.Logger


@dataclass(frozen=True)
class ModuleResult:
    """模块完成后交给统一发布器的摘要和上游依赖。"""

    summary: Mapping[str, object]
    dependencies: tuple[Path, ...] = ()


@dataclass(frozen=True)
class ValidationReport:
    """模块落盘后重新检查得到的结果。"""

    checks: Mapping[str, object]

