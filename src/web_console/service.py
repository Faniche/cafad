"""service.py | 2026-11-04 | Web 控制台的命令构造与非阻塞任务日志服务 | Codex."""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
from importlib import import_module
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import threading
import time
from typing import Mapping, Protocol
from uuid import uuid4

from src.app.cli import IMPLEMENTED_MODULES
from src.app.module_catalog import MODULE_SEQUENCE, artifact_module_directory
from src.utils.process_control import process_group_options, interrupt_process_group, stop_process_group


MAX_LOG_LINES = 10_000
MAX_STORED_JOBS = 20
UPSTREAM_MANIFEST_INPUTS = {
    'split': {'input': 'preprocess'},
    'segment': {'input': 'split'},
    'group': {'input': 'segment'},
    'inject': {'input': 'group'},
    'train': {'input': 'inject'},
    'calibrate': {'input': 'train'},
    'detect': {'input': 'calibrate'},
    'evaluate': {'input': 'detect'},
}
AUTO_DERIVED_MANIFEST_INPUTS: dict[str, dict[str, tuple[str, str, str]]] = {}


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
    upstream_module: str | tuple[str, ...] | None
    auto_derived: bool


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
    registry_before: frozenset[str] = field(default_factory=frozenset)
    stop_requested: bool = False
    cleaned_artifacts: int = 0
    artifact_manifest: str | None = None
    process: subprocess.Popen[str] | None = field(default=None, repr=False)
    stop_finished: threading.Event = field(default_factory=threading.Event, repr=False)
    stop_started: bool = False

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
            upstream_module=UPSTREAM_MANIFEST_INPUTS.get(module_name, {}).get(action.dest),
            auto_derived=action.dest in AUTO_DERIVED_MANIFEST_INPUTS.get(module_name, {}),
        ))
    return tuple(specs)


def module_schemas() -> dict[str, tuple[OptionSpec, ...]]:
    """返回当前正式入口实际允许的模块和所有模块级选项。"""
    return {name: _option_specs(name) for name in MODULE_SEQUENCE if name in IMPLEMENTED_MODULES}


def module_catalog() -> tuple[str, ...]:
    """返回网页导航的完整实验顺序，未实现模块也需要占位显示。"""
    return MODULE_SEQUENCE


def upstream_manifest_choices(project_root: Path, module_name: str, argument_name: str) -> list[dict[str, str]]:
    """列出一个已声明上游模块的成功运行，不对大文件重新计算哈希。"""
    declared = UPSTREAM_MANIFEST_INPUTS.get(module_name, {}).get(argument_name)
    if declared is None:
        raise ValueError('该模块参数不是已声明的上游 manifest 输入')
    source_modules = (declared,) if isinstance(declared, str) else tuple(declared)
    choices: list[dict[str, str]] = []
    for source_module in source_modules:
        source_root = project_root / 'data' / artifact_module_directory(source_module)
        if not source_root.is_dir():
            continue
        for dataset_dir in sorted((item for item in source_root.iterdir() if item.is_dir() and not item.name.startswith('.')),
                                  key=lambda item: item.name):
            run_directories = sorted((item for item in dataset_dir.iterdir() if item.is_dir() and not item.name.startswith('.')),
                                     key=lambda item: item.name, reverse=True)
            for run_dir in run_directories:
                manifest_path = run_dir / 'manifest.json'
                try:
                    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
                except (OSError, json.JSONDecodeError):
                    continue
                if not isinstance(manifest, dict) or manifest.get('status') != 'SUCCEEDED' or manifest.get('module') != source_module:
                    continue
                try:
                    relative_manifest = manifest_path.relative_to(project_root).as_posix()
                except ValueError:
                    continue
                choices.append({
                    'dataset': dataset_dir.name,
                    'run_name': run_dir.name,
                    'label': (f'{source_module}: ' if len(source_modules) > 1 else '')
                             + f'{dataset_dir.name} / {run_dir.name}',
                    'manifest': relative_manifest,
                })
    return sorted(choices, key=lambda item: (item['dataset'], item['run_name'], item['label']), reverse=True)


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
                'upstream_module': spec.upstream_module,
                'auto_derived': spec.auto_derived,
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


def _project_manifest_path(project_root: Path, value: str) -> Path:
    """解析网页选择的项目内 manifest，拒绝越出项目根的依赖路径。"""
    path = Path(value)
    resolved = (path if path.is_absolute() else project_root / path).resolve()
    try:
        resolved.relative_to(project_root.resolve())
    except ValueError as exc:
        raise ValueError('自动推导的上游 manifest 必须位于项目目录内') from exc
    if not resolved.is_file():
        raise ValueError(f'找不到所选上游 manifest：{resolved}')
    return resolved


def _read_success_manifest(path: Path, expected_module: str) -> dict[str, object]:
    """读取网页依赖链所需的最小清单字段；正式入口仍会重新核验全部输出哈希。"""
    try:
        manifest = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f'无法读取所选上游 manifest：{path}') from exc
    if not isinstance(manifest, dict) or manifest.get('status') != 'SUCCEEDED':
        raise ValueError(f'所选上游产物未成功发布：{path}')
    if manifest.get('module') != expected_module:
        raise ValueError(f'所选 manifest 应为 {expected_module}，实际为 {manifest.get("module")}')
    return manifest


def _derive_manifest_arguments(project_root: Path, module_name: str,
                               module_values: Mapping[str, object]) -> dict[str, object]:
    """由最近上游清单的依赖链推导网页隐藏的较早 manifest 参数。"""
    values = dict(module_values)
    for destination, (selector, selector_module, dependency_module) in AUTO_DERIVED_MANIFEST_INPUTS.get(module_name, {}).items():
        selected_value = _require_text(values, selector)
        selected_path = _project_manifest_path(project_root, selected_value)
        selected_manifest = _read_success_manifest(selected_path, selector_module)
        dependencies = selected_manifest.get('dependencies')
        if not isinstance(dependencies, list):
            raise ValueError(f'{selector_module} manifest 缺少依赖链，无法自动推导 {destination}')
        matches: list[Path] = []
        for item in dependencies:
            if not isinstance(item, dict) or not isinstance(item.get('manifest'), str):
                continue
            dependency_path = _project_manifest_path(project_root, item['manifest'])
            dependency_manifest = _read_success_manifest(dependency_path, dependency_module)
            if dependency_manifest.get('dataset_id') == selected_manifest.get('dataset_id'):
                matches.append(dependency_path)
        if len(matches) != 1:
            raise ValueError(f'{selector_module} manifest 未能唯一推导 {dependency_module} 的 {destination}')
        values[destination] = matches[0].relative_to(project_root.resolve()).as_posix()
    return values


def build_command(project_root: Path, payload: Mapping[str, object], schemas: Mapping[str, tuple[OptionSpec, ...]]) -> tuple[str, ...]:
    """从受限表单数据生成无 shell 的唯一根入口调用命令。"""
    global_values = payload.get('global')
    module_values = payload.get('module_args')
    module_name = payload.get('module')
    if not isinstance(global_values, dict) or not isinstance(module_values, dict):
        raise ValueError('请求必须包含 global 和 module_args 对象')
    if not isinstance(module_name, str) or module_name not in schemas:
        raise ValueError('模块不存在或尚未实现')
    resolved_module_values = _derive_manifest_arguments(project_root, module_name, module_values)
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
        value = resolved_module_values.get(spec.dest)
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
            if any(job.status in {'running', 'stopping'} for job in self._jobs.values()):
                raise RuntimeError('已有网页任务正在运行；同一时间只允许一个正式模块')
            self._prune_finished_jobs()
            job = ConsoleJob(job_id=uuid4().hex, module_name=module_name, command=command, started_at=time.time(),
                             registry_before=self._run_registry_ids())
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

    def has_active_job(self) -> bool:
        """返回是否仍有网页启动的正式任务，供历史删除避免与发布冲突。"""
        with self._lock:
            return any(job.status in {'running', 'stopping'} for job in self._jobs.values())

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

    def stop(self, job_id: str) -> dict[str, object]:
        """请求中断一个网页任务；退出后仅清理该任务尚未发布的暂存产物。"""
        with self._lock:
            job = self._get_job(job_id)
            if job.status not in {'running', 'stopping'}:
                raise ValueError('该任务已经结束，不能再停止')
            job.stop_requested = True
            job.status = 'stopping'
            job.append_line('[web-console] 已请求停止；等待子进程退出并清理本次未发布产物。')
            process = job.process
        if process is not None:
            self._start_stop(job, process)
        return self.snapshot(job_id)

    def _start_stop(self, job: ConsoleJob, process: subprocess.Popen[str]) -> None:
        """同一任务只启动一次有界清理，HTTP 线程立即返回停止中。"""
        with self._lock:
            if job.stop_started:
                return
            job.stop_started = True
        threading.Thread(target=self._stop_tree, args=(job, process), daemon=True,
                         name=f'cafad-stop-{job.job_id[:8]}').start()

    def _stop_tree(self, job: ConsoleJob, process: subprocess.Popen[str]) -> None:
        """中断整个进程组，再依次终止/强杀；父进程退出也继续清理同组子进程。"""
        def notify(message: str) -> None:
            with self._lock:
                job.append_line(f'[web-console] {message}')
        try:
            stop_process_group(process, notify, self._interrupt_process)
        except (OSError, subprocess.SubprocessError) as exc:
            self._logger('任务进程树停止失败：job_id=%s；%s', job.job_id, exc)
        finally:
            job.stop_finished.set()

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
            'stop_requested': job.stop_requested,
            'cleaned_artifacts': job.cleaned_artifacts,
            'artifact_manifest': job.artifact_manifest,
        }

    def _prune_finished_jobs(self) -> None:
        """只在启动新任务前裁剪最早完成任务，限制长时服务的日志缓存内存。"""
        finished = sorted((job for job in self._jobs.values() if job.status not in {'running', 'stopping'}),
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
                **self._process_group_options(),
            ) as process:
                with self._lock:
                    job.process = process
                    stop_requested = job.stop_requested
                if stop_requested:
                    self._start_stop(job, process)
                if process.stdout is None:
                    raise RuntimeError('无法建立子进程标准输出管道')
                for line in process.stdout:
                    with self._lock:
                        job.append_line(line)
                return_code = process.wait()
        except Exception as exc:
            with self._lock:
                job.append_line(f'[web-console] 启动失败：{exc}')
                stopped = job.stop_requested
                job.status = 'cancelled' if stopped else 'failed'
                job.ended_at = time.time()
            if stopped:
                self._clean_cancelled_staging(job)
            self._logger('网页任务启动失败：%s', exc)
            return
        with self._lock:
            stopped = job.stop_requested
        if stopped and job.stop_started:
            job.stop_finished.wait()
        cleaned_artifacts = self._clean_cancelled_staging(job) if stopped else 0
        artifact_manifest = self._published_manifest(job) if not stopped and return_code == 0 else None
        with self._lock:
            job.return_code = return_code
            job.process = None
            job.cleaned_artifacts = cleaned_artifacts
            job.artifact_manifest = artifact_manifest
            job.status = 'cancelled' if stopped else ('succeeded' if return_code == 0 else 'failed')
            job.ended_at = time.time()
            job.append_line(f'[web-console] 进程结束，退出码={return_code}')
            if stopped:
                job.append_line(f'[web-console] 已清理本次未发布产物目录：{cleaned_artifacts}')
        self._logger('网页任务结束：job_id=%s，return_code=%d，stopped=%s', job.job_id, return_code, stopped)

    _process_group_options = staticmethod(process_group_options)
    _interrupt_process = staticmethod(interrupt_process_group)

    def _run_registry_ids(self) -> frozenset[str]:
        """读取当前运行台账的稳定记录键，用于精确识别本网页任务后分配的运行。"""
        registry_path = self._project_root / 'run_registry.json'
        try:
            value = json.loads(registry_path.read_text(encoding='utf-8'))
            records = value.get('runs') if isinstance(value, dict) else None
        except (OSError, json.JSONDecodeError):
            records = None
        if not isinstance(records, list):
            return frozenset()
        return frozenset(self._registry_id(record) for record in records if isinstance(record, dict))

    def _published_manifest(self, job: ConsoleJob) -> str | None:
        """从本任务新增的成功台账中定位唯一已发布 manifest，供串行后续任务溯源。"""
        registry_path = self._project_root / 'run_registry.json'
        try:
            value = json.loads(registry_path.read_text(encoding='utf-8'))
            records = value.get('runs') if isinstance(value, dict) else None
        except (OSError, json.JSONDecodeError):
            records = None
        if not isinstance(records, list):
            return None
        candidates: list[Path] = []
        for record in records:
            if not isinstance(record, dict) or record.get('module') != job.module_name:
                continue
            if record.get('state') != 'success' or self._registry_id(record) in job.registry_before:
                continue
            artifact = record.get('artifact_dir')
            if not isinstance(artifact, str):
                continue
            manifest = (self._project_root / artifact / 'manifest.json').resolve()
            try:
                manifest.relative_to(self._project_root.resolve())
                value = json.loads(manifest.read_text(encoding='utf-8'))
            except (ValueError, OSError, json.JSONDecodeError):
                continue
            if isinstance(value, dict) and value.get('status') == 'SUCCEEDED' and value.get('module') == job.module_name:
                candidates.append(manifest)
        if len(candidates) != 1:
            self._logger('网页任务未能唯一定位发布产物：job_id=%s；module=%s；count=%d',
                         job.job_id, job.module_name, len(candidates))
            return None
        return str(candidates[0])

    @staticmethod
    def _registry_id(record: Mapping[str, object]) -> str:
        """组合不会与同一模块重跑冲突的台账字段。"""
        return '|'.join(str(record.get(name, '')) for name in ('dataset', 'module', 'counter', 'started_at'))

    def _clean_cancelled_staging(self, job: ConsoleJob) -> int:
        """仅删除停止后新增台账所指向的本次 `.pending` 目录，绝不触碰已发布产物。"""
        registry_path = self._project_root / 'run_registry.json'
        try:
            value = json.loads(registry_path.read_text(encoding='utf-8'))
            records = value.get('runs') if isinstance(value, dict) else None
        except (OSError, json.JSONDecodeError):
            records = None
        if not isinstance(records, list):
            return 0
        modules = ('train', 'calibrate', 'detect', 'evaluate') if job.module_name.startswith('search_') else (job.module_name,)
        candidates: list[Path] = []
        for record in records:
            if not isinstance(record, dict) or record.get('module') not in modules:
                continue
            module_root = (self._project_root / 'data' / artifact_module_directory(record['module'])).resolve()
            if self._registry_id(record) in job.registry_before:
                continue
            artifact = record.get('artifact_dir')
            if not isinstance(artifact, str):
                continue
            final_dir = (self._project_root / artifact).resolve()
            staging_dir = final_dir.parent / f'.{final_dir.name}.pending'
            try:
                staging_dir.relative_to(module_root)
            except ValueError:
                continue
            if staging_dir.parent.parent != module_root or not staging_dir.name.endswith('.pending'):
                continue
            if staging_dir.is_dir():
                candidates.append(staging_dir)
        for directory in candidates:
            shutil.rmtree(directory)
        return len(candidates)
