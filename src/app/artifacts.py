"""artifacts.py | 2026-11-03 | 独立模块产物的暂存、校验与发布 | Codex."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket
import time
from typing import Mapping

from .contracts import ModuleResult, RunContext, ValidationReport


def write_json(path: Path, value: Mapping[str, object]) -> None:
    """原子写入 UTF-8 JSON，避免中断后留下半个配置或清单。"""
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + '\n', encoding='utf-8')
    temporary.replace(path)


def hash_file(path: Path) -> str:
    """分块计算 SHA-256，支持大 CSV 的完整性检查。"""
    digest = hashlib.sha256()
    with path.open('rb') as reader:
        while block := reader.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path: Path, expected_module: str | None = None) -> dict[str, object]:
    """读取已发布清单，并重新核验其中所有业务文件的哈希。"""
    if not path.is_file():
        raise FileNotFoundError(f'找不到上游 manifest：{path}')
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict) or value.get('status') != 'SUCCEEDED':
        raise ValueError(f'上游产物未通过验收：{path}')
    if expected_module is not None and value.get('module') != expected_module:
        raise ValueError(f'上游模块应为 {expected_module!r}，实际为 {value.get("module")!r}')
    outputs = value.get('outputs')
    if not isinstance(outputs, list):
        raise ValueError(f'上游清单缺少 outputs：{path}')
    artifact_dir = path.parent.resolve()
    for item in outputs:
        if not isinstance(item, dict) or not isinstance(item.get('path'), str) or not isinstance(item.get('sha256'), str):
            raise ValueError(f'上游清单包含无效输出项：{path}')
        output_path = (artifact_dir / item['path']).resolve()
        if artifact_dir not in output_path.parents or not output_path.is_file():
            raise ValueError(f'上游输出缺失或越出产物目录：{item["path"]}')
        if hash_file(output_path) != item['sha256']:
            raise ValueError(f'上游输出哈希不匹配：{item["path"]}')
    return value


class ProjectRunLock:
    """用原子目录创建串行化正式模块运行，绝不自动删除未知持有者。"""

    def __init__(self, project_root: Path) -> None:
        self.lock_dir = project_root / '.cafad-run.lock'
        self.owner_path = self.lock_dir / 'owner.json'
        self.owner = {'pid': os.getpid(), 'hostname': socket.gethostname(), 'started_at': time.time()}

    def __enter__(self) -> 'ProjectRunLock':
        self.lock_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.lock_dir.mkdir()
        except FileExistsError as exc:
            owner = '未知持有者'
            if self.owner_path.is_file():
                owner = self.owner_path.read_text(encoding='utf-8').strip()
            raise RuntimeError(f'已有正式模块运行锁，拒绝并发执行：{self.lock_dir}；持有者={owner}') from exc
        write_json(self.owner_path, self.owner)
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        if not self.owner_path.is_file():
            return
        current = json.loads(self.owner_path.read_text(encoding='utf-8'))
        if current != self.owner:
            raise RuntimeError(f'运行锁持有者已变化，拒绝删除：{self.lock_dir}')
        self.owner_path.unlink()
        self.lock_dir.rmdir()


class ArtifactWorkspace:
    """在同一文件系统暂存模块产物，校验通过后原子发布。"""

    def __init__(self, final_dir: Path) -> None:
        self.final_dir = final_dir
        self.staging_dir = self.final_dir.parent / f'.{self.final_dir.name}.pending'

    def create(self) -> None:
        """创建唯一暂存目录，拒绝覆盖历史运行。"""
        if self.final_dir.exists() or self.staging_dir.exists():
            raise FileExistsError(f'自动分配的运行目录已存在：{self.final_dir}')
        self.staging_dir.mkdir(parents=True)

    def publish(self) -> None:
        """用目录改名发布已校验产物。"""
        self.staging_dir.replace(self.final_dir)

    def discard(self, reason: str) -> None:
        """保留失败诊断时由调用方写入；未发布暂存目录不作为输入。"""
        if self.staging_dir.exists():
            failure = self.staging_dir / 'failure.json'
            if not failure.exists():
                write_json(failure, {'status': 'FAILED', 'reason': reason})


def write_run_metadata(context: RunContext, global_values: Mapping[str, object], module_values: Mapping[str, object]) -> None:
    """持久化解析后的参数，便于下一模块和后续复现实验读取。"""
    write_json(context.staging_dir / 'resolved_config.json', {
        'global': dict(global_values),
        'module': dict(module_values),
        'module_name': context.module_name,
        'dataset_id': context.dataset_id,
        'dataset_name': context.dataset_name,
        'run_id': context.run_id,
        'run_timestamp': context.run_timestamp,
    })


def write_manifest(context: RunContext, result: ModuleResult, report: ValidationReport) -> None:
    """枚举所有落盘文件及依赖，在发布前写入成功清单。"""
    write_json(context.staging_dir / 'validation.json', {'passed': True, 'checks': dict(report.checks)})
    write_json(context.staging_dir / 'summary.json', dict(result.summary))
    outputs = []
    for path in sorted(context.staging_dir.rglob('*')):
        if path.is_file() and path.name != 'manifest.json':
            outputs.append({'path': str(path.relative_to(context.staging_dir)).replace('\\', '/'),
                            'bytes': path.stat().st_size, 'sha256': hash_file(path)})
    dependencies = []
    for path in result.dependencies:
        dependencies.append({'manifest': str(path.resolve()), 'sha256': hash_file(path)})
    write_json(context.staging_dir / 'manifest.json', {
        'schema_version': 1,
        'status': 'SUCCEEDED',
        'module': context.module_name,
        'dataset_id': context.dataset_id,
        'dataset_name': context.dataset_name,
        'run_id': context.run_id,
        'dependencies': dependencies,
        'outputs': outputs,
        'validation': 'validation.json',
    })


def refresh_manifest_outputs(artifact_dir: Path) -> None:
    """在受控的人工修订后重算成功产物的输出清单与哈希。"""
    manifest_path = artifact_dir / 'manifest.json'
    value = json.loads(manifest_path.read_text(encoding='utf-8'))
    if not isinstance(value, dict) or value.get('status') != 'SUCCEEDED':
        raise ValueError('只能刷新已成功发布产物的清单')
    outputs = []
    for path in sorted(artifact_dir.rglob('*')):
        if path.is_file() and path.name != 'manifest.json':
            outputs.append({'path': str(path.relative_to(artifact_dir)).replace('\\', '/'),
                            'bytes': path.stat().st_size, 'sha256': hash_file(path)})
    value['outputs'] = outputs
    value['manual_revision'] = int(value.get('manual_revision', 0)) + 1
    write_json(manifest_path, value)
