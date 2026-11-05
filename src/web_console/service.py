"""service.py | 2026-11-04 | Web 控制台的命令构造与非阻塞任务日志服务 | Codex."""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
from importlib import import_module
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time
from typing import Mapping, Protocol
from uuid import uuid4

from src.app.cli import IMPLEMENTED_MODULES


MAX_LOG_LINES = 10_000
MAX_STORED_JOBS = 20


class LogWriter(Protocol):
    """兼容 logging.Logger.info 的最小日志调用协议。"""

    def __call__(self, message: str, *args: object) -> None:
        """记录一条格式化日志。"""


@dataclass(frozen=True)
class OptionSpec:
    """一个模块命令行选项的浏览器可呈现描述。"""

    dest: str
    option: str
    required: bool
    is_flag: bool
    takes_multiple: bool
    choices: tuple[str, ...]
    default: str | int | float | bool | None
    help_text: str
    is_path: bool


@dataclass
class ConsoleJob:
    """一个后台 CAFAD 命令及其有界控制台输出缓存。"""

    job_id: str
    module_name: str
    command: tuple[str, ...]
    started_at: float
    status: str = 'running'
    return_code: int | None = None
    ended_at: float | None = None
    line_offset: int = 0
    lines: deque[tuple[int, str]] = field(default_factory=lambda: deque(maxlen=MAX_LOG_LINES))

    def append_line(self, line: str) -> None:
        """追加一行子进程输出，并保留单调递增的浏览器读取位置。"""
        self.lines.append((self.line_offset, line.rstrip('\r\n')))
        self.line_offset += 1


def _option_specs(module_name: str) -> tuple[OptionSpec, ...]:
    """从实际模块 api.py 的 argparse 定义提取可编辑选项，避免重复维护表单。"""
    module = import_module(IMPLEMENTED_MODULES[module_name])
    parser = module.build_parser()
    specs: list[OptionSpec] = []
    for action in parser._actions:
        if not action.option_strings or action.dest == 'help':
            continue
        option = next((item for item in action.option_strings if item.startswith('--')), action.option_strings[0])
        choices = tuple(str(value) for value in action.choices) if action.choices is not None else ()
        default = action.default if isinstance(action.default, (str, int, float, bool)) else None
        specs.append(OptionSpec(
            dest=action.dest,
            option=option,
            required=bool(action.required),
            is_flag=action.nargs == 0,
            takes_multiple=action.nargs not in (None, 0),
            choices=choices,
            default=default,
            help_text=action.help or '',
            is_path=action.type is Path,
        ))
    return tuple(specs)


def module_schemas() -> dict[str, tuple[OptionSpec, ...]]:
    """返回当前正式入口实际允许的模块和所有模块级选项。"""
    return {name: _option_specs(name) for name in sorted(IMPLEMENTED_MODULES)}


def serialize_schemas(schemas: Mapping[str, tuple[OptionSpec, ...]]) -> dict[str, list[dict[str, object]]]:
    """转换为 JSON 可编码的模块表单定义。"""
    return {
        name: [
            {
                'dest': spec.dest,
                'option': spec.option,
                'required': spec.required,
                'is_flag': spec.is_flag,
                'takes_multiple': spec.takes_multiple,
                'choices': list(spec.choices),
                'default': spec.default,
                'help_text': spec.help_text,
                'is_path': spec.is_path,
            }
            for spec in specs
        ]
        for name, specs in schemas.items()
    }


def _require_text(values: Mapping[str, object], name: str) -> str:
    """读取必填非空字符串，拒绝结构化对象和隐式转换。"""
    value = values.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'缺少必填参数：{name}')
    return value.strip()


def _optional_text(values: Mapping[str, object], name: str) -> str | None:
    """读取可选字符串；空白值不传给命令行以保留 argparse 默认值。"""
    value = values.get(name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f'参数 {name} 必须是字符串')
    stripped = value.strip()
    return stripped or None


def build_command(project_root: Path, payload: Mapping[str, object], schemas: Mapping[str, tuple[OptionSpec, ...]]) -> tuple[str, ...]:
    """从受限表单数据生成无 shell 的唯一根入口调用命令。"""
    global_values = payload.get('global')
    module_values = payload.get('module_args')
    module_name = payload.get('module')
    if not isinstance(global_values, dict) or not isinstance(module_values, dict):
        raise ValueError('请求必须包含 global 和 module_args 对象')
    if not isinstance(module_name, str) or module_name not in schemas:
        raise ValueError('模块不存在或尚未实现')
    command = [sys.executable, str(project_root / 'main.py')]
    for field, option in (('seed', '--seed'), ('log_level', '--log-level')):
        value = _optional_text(global_values, field)
        if value is not None:
            command.extend((option, value))
    console = global_values.get('console', False)
    if not isinstance(console, bool):
        raise ValueError('console 必须为布尔值')
    if console:
        command.append('--console')
    command.extend(('-m', module_name))
    for spec in schemas[module_name]:
        value = module_values.get(spec.dest)
        if spec.is_flag:
            if value is True:
                command.append(spec.option)
            elif value not in (None, False):
                raise ValueError(f'开关 {spec.dest} 必须为布尔值')
            continue
        if value is None or value == '':
            if spec.required:
                raise ValueError(f'缺少模块必填参数：{spec.option}')
            continue
        if spec.takes_multiple:
            if not isinstance(value, list) or not value or not all(isinstance(item, str) and item.strip() for item in value):
                raise ValueError(f'参数 {spec.option} 必须为非空字符串列表')
            command.append(spec.option)
            command.extend(item.strip() for item in value)
        else:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f'参数 {spec.option} 必须为非空字符串')
            command.extend((spec.option, value.strip()))
    return tuple(command)


class JobManager:
    """串行启动 CAFAD 命令，并由独立线程持续排空子进程输出管道。"""

    def __init__(self, project_root: Path, logger: LogWriter) -> None:
        self._project_root = project_root
        self._logger = logger
        self._jobs: dict[str, ConsoleJob] = {}
        self._lock = threading.Lock()

    def start(self, command: tuple[str, ...], module_name: str) -> ConsoleJob:
        """启动一个后台命令；存在运行任务时拒绝第二个网页任务。"""
        with self._lock:
            if any(job.status == 'running' for job in self._jobs.values()):
                raise RuntimeError('已有网页任务正在运行；同一时间只允许一个正式模块')
            self._prune_finished_jobs()
            job = ConsoleJob(job_id=uuid4().hex, module_name=module_name, command=command, started_at=time.time())
            job.append_line('$ ' + shlex.join(command))
            self._jobs[job.job_id] = job
        thread = threading.Thread(target=self._run, args=(job,), name=f'cafad-web-{job.job_id[:8]}', daemon=True)
        thread.start()
        return job

    def snapshot(self, job_id: str) -> dict[str, object]:
        """返回任务状态和已完成时间，不返回全部日志避免轮询重复传输。"""
        with self._lock:
            return self._snapshot(self._get_job(job_id))

    def latest_snapshot(self, module_name: str) -> dict[str, object] | None:
        """返回某模块最近一次网页任务；尚未运行时返回空值。"""
        with self._lock:
            candidates = [job for job in self._jobs.values() if job.module_name == module_name]
            if not candidates:
                return None
            return self._snapshot(max(candidates, key=lambda job: job.started_at))

    def logs_after(self, job_id: str, after: int) -> dict[str, object]:
        """只返回浏览器尚未读取的新日志行；缓存满时明确告知已截断。"""
        if after < 0:
            raise ValueError('日志位置不能小于 0')
        with self._lock:
            job = self._get_job(job_id)
            first_offset = job.lines[0][0] if job.lines else job.line_offset
            return {
                'lines': [line for offset, line in job.lines if offset >= after],
                'next_offset': job.line_offset,
                'truncated': after < first_offset,
                'status': job.status,
                'return_code': job.return_code,
            }

    def _get_job(self, job_id: str) -> ConsoleJob:
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError('任务不存在')
        return job

    @staticmethod
    def _snapshot(job: ConsoleJob) -> dict[str, object]:
        """将一个已锁定的任务状态转换为浏览器响应。"""
        return {
            'job_id': job.job_id,
            'module_name': job.module_name,
            'command': list(job.command),
            'started_at': job.started_at,
            'status': job.status,
            'return_code': job.return_code,
            'ended_at': job.ended_at,
        }

    def _prune_finished_jobs(self) -> None:
        """只在启动新任务前裁剪最早完成任务，限制长时服务的日志缓存内存。"""
        finished = sorted((job for job in self._jobs.values() if job.status != 'running'),
                          key=lambda job: job.ended_at or job.started_at)
        for job in finished[:max(0, len(self._jobs) - MAX_STORED_JOBS + 1)]:
            del self._jobs[job.job_id]

    def _run(self, job: ConsoleJob) -> None:
        """读取 stdout/stderr 直到进程结束；读取线程独立于 HTTP 请求处理线程。"""
        environment = {'PYTHONUNBUFFERED': '1'}
        try:
            with subprocess.Popen(
                job.command,
                cwd=self._project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                bufsize=1,
                env={**os.environ, **environment},
            ) as process:
                if process.stdout is None:
                    raise RuntimeError('无法建立子进程标准输出管道')
                for line in process.stdout:
                    with self._lock:
                        job.append_line(line)
                return_code = process.wait()
        except Exception as exc:
            with self._lock:
                job.append_line(f'[web-console] 启动失败：{exc}')
                job.status = 'failed'
                job.ended_at = time.time()
            self._logger('网页任务启动失败：%s', exc)
            return
        with self._lock:
            job.return_code = return_code
            job.status = 'succeeded' if return_code == 0 else 'failed'
            job.ended_at = time.time()
            job.append_line(f'[web-console] 进程结束，退出码={return_code}')
        self._logger('网页任务结束：job_id=%s，return_code=%d', job.job_id, return_code)
