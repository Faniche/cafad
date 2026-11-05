"""main.py | 2026-11-03 | CAFAD 项目唯一正式命令行入口 | Codex."""
from __future__ import annotations

import multiprocessing

from src.app.cli import main


if __name__ == '__main__':
    multiprocessing.freeze_support()
    raise SystemExit(main())
