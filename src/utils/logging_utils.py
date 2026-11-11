"""logging_utils.py | 2026-11-03 | 项目统一日志配置 | Codex."""
from __future__ import annotations

from datetime import datetime
import logging
import os
from pathlib import Path
import sys
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGGER_NAME = 'sun_preprocessing'


class ProjectLogFormatter(logging.Formatter):
    """Give every physical line, including traceback lines, the same prefix."""

    def __init__(self) -> None:
        super().__init__(datefmt='%Y-%m-%d %H:%M:%S%z')

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        if record.exc_info:
            message += '\n' + self.formatException(record.exc_info)
        if record.stack_info:
            message += '\n' + self.formatStack(record.stack_info)
        prefix = f'{self.formatTime(record, self.datefmt)} {record.filename}:{record.lineno} {record.levelname} '
        return '\n'.join(prefix + line for line in (message.splitlines() or ['']))


def close_project_logging(logger: logging.Logger) -> None:
    """Close only the handlers owned by this utility."""
    for handler in logger.handlers[:]:
        if getattr(handler, '_project_logging_handler', False):
            logger.removeHandler(handler)
            handler.close()


def setup_project_logging(*, console: bool = False, project_root: Path | None = None,
                          name: str = LOGGER_NAME, level: int = logging.INFO,
                          filename: str | None = None) -> tuple[logging.Logger, Path]:
    """Always create a UTF-8 file in <project>/logs; optionally mirror to stdout.

    The explicit root parameter is for isolated utility tests. Application entry
    points use the module's absolute project root, independently of their cwd.
    """
    root = PROJECT_ROOT if project_root is None else Path(project_root).resolve()
    directory = root / 'logs'
    directory.mkdir(parents=True, exist_ok=True)
    if filename is not None:
        if Path(filename).name != filename or not filename.endswith('.log'):
            raise ValueError('日志文件名必须是单个 .log 文件名，不能包含目录')
        path = directory / filename
        if path.exists():
            raise FileExistsError(f'日志文件已存在：{path}')
    else:
        stamp = datetime.now().astimezone().strftime('%Y%m%d-%H%M%S')
        path = directory / f'run-{stamp}-{os.getpid()}.log'
        if path.exists():
            path = directory / f'run-{stamp}-{os.getpid()}-{uuid4().hex[:6]}.log'
    logger = logging.getLogger(name)
    close_project_logging(logger)
    logger.setLevel(level)
    # The application owns this logger. Disable root propagation so earlier
    # basicConfig/host handlers cannot leak console output or duplicate records.
    logger.propagate = False
    formatter = ProjectLogFormatter()
    handler = logging.FileHandler(path, mode='x', encoding='utf-8')
    handler.setFormatter(formatter)
    handler._project_logging_handler = True
    logger.addHandler(handler)
    if console:
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(formatter)
        stream._project_logging_handler = True
        logger.addHandler(stream)
    logger.project_log_path = path
    return logger, path
