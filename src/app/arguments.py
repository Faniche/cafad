"""arguments.py | 2026-11-03 | 统一命令行参数错误为可记录异常 | Codex."""
from __future__ import annotations

import argparse
from typing import NoReturn


class ProjectArgumentParser(argparse.ArgumentParser):
    """保留标准帮助输出，但把业务参数错误交由根入口记录到日志。"""

    def error(self, message: str) -> NoReturn:
        raise ValueError(f'命令行参数错误：{message}')
