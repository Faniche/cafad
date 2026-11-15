"""history.py | 2026-11-07 | Web 控制台已落盘运行的历史与依赖删除 | Codex."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

from src.app.module_catalog import MODULE_SEQUENCE, artifact_module_directory


def list_module_history(project_root: Path, module_name: str) -> list[dict[str, object]]:
    """列出一个模块的已发布和未发布运行，不读取大型业务 CSV。"""
    module_root = _module_root(project_root, module_name)
    if not module_root.is_dir():
        return []
    entries: list[dict[str, object]] = []
    for dataset_dir in module_root.iterdir():
        if not dataset_dir.is_dir() or dataset_dir.name.startswith('.'):
            continue
        for run_dir in dataset_dir.iterdir():
            if not run_dir.is_dir() or (run_dir.name.startswith('.') and not run_dir.name.endswith('.pending')):
                continue
            entry = _entry_from_directory(project_root, module_name, run_dir)
            if entry is not None:
                entries.append(entry)
    return sorted(entries, key=lambda item: (str(item['run_timestamp']), str(item['artifact'])), reverse=True)


def history_detail(project_root: Path, module_name: str, artifact: str) -> dict[str, object]:
    """读取一个历史运行的摘要、失败诊断、日志尾部及下游删除预览。"""
    run_dir = _artifact_directory(project_root, module_name, artifact)
    entry = _entry_from_directory(project_root, module_name, run_dir)
    if entry is None:
        raise ValueError('运行目录缺少有效 resolved_config.json')
    summary = _read_json_object(run_dir / 'summary.json')
    failure = _read_json_object(run_dir / 'failure.json')
    log_text, log_truncated = _read_log_tail(project_root, entry)
    return {
        'entry': entry,
        'summary': summary,
        'failure': failure,
        'logs': log_text,
        'logs_truncated': log_truncated,
        'delete_preview': dependent_history_artifacts(project_root, module_name, artifact),
    }


def dependent_history_artifacts(project_root: Path, module_name: str, artifact: str) -> list[str]:
    """返回目标运行及其显式依赖下游，供删除和受控人工修订共同使用。"""
    run_dir = _artifact_directory(project_root, module_name, artifact)
    return [item.relative_to(project_root).as_posix() for item in _dependent_directories(project_root, run_dir)]


def delete_history_run(project_root: Path, module_name: str, artifact: str) -> list[str]:
    """删除目标运行及声明依赖它的下游运行，返回实际删除的相对目录。"""
    run_dir = _artifact_directory(project_root, module_name, artifact)
    if _entry_from_directory(project_root, module_name, run_dir) is None:
        raise ValueError('运行目录缺少有效 resolved_config.json')
    targets = _dependent_directories(project_root, run_dir)
    deleted: list[str] = []
    for target in reversed(targets):
        relative = target.relative_to(project_root).as_posix()
        if target.is_dir():
            shutil.rmtree(target)
            deleted.append(relative)
    return deleted


def _module_root(project_root: Path, module_name: str) -> Path:
    """返回并约束某个正式模块的磁盘产物根。"""
    if module_name not in MODULE_SEQUENCE:
        raise ValueError('模块不存在')
    return (project_root / 'data' / artifact_module_directory(module_name)).resolve()


def _artifact_directory(project_root: Path, module_name: str, artifact: str) -> Path:
    """将 API 传来的相对目录严格限制为当前模块下的一次运行。"""
    candidate = (project_root / artifact).resolve()
    module_root = _module_root(project_root, module_name)
    try:
        relative = candidate.relative_to(module_root)
    except ValueError as exc:
        raise ValueError('运行目录不属于当前模块') from exc
    if len(relative.parts) != 2 or not candidate.is_dir():
        raise ValueError('运行目录格式无效或不存在')
    return candidate


def _entry_from_directory(project_root: Path, module_name: str, run_dir: Path) -> dict[str, object] | None:
    """从每次运行必写的解析配置取得历史表所需的稳定身份。"""
    config = _read_json_object(run_dir / 'resolved_config.json')
    if config is None or config.get('module_name') != module_name:
        return None
    required = ('dataset_id', 'dataset_name', 'run_id', 'run_timestamp')
    if not all(isinstance(config.get(name), str) and config[name] for name in required):
        return None
    manifest = _read_json_object(run_dir / 'manifest.json')
    failure = _read_json_object(run_dir / 'failure.json')
    if manifest is not None and manifest.get('status') == 'SUCCEEDED':
        state = 'success'
    elif failure is not None:
        state = 'failure'
    else:
        state = 'running'
    try:
        artifact = run_dir.relative_to(project_root).as_posix()
    except ValueError:
        return None
    global_values = config.get('global')
    seed = global_values.get('seed') if isinstance(global_values, dict) else None
    return {
        'module_name': module_name,
        'artifact': artifact,
        'dataset_id': config['dataset_id'],
        'dataset_name': config['dataset_name'],
        'run_id': config['run_id'],
        'run_timestamp': config['run_timestamp'],
        'seed': seed,
        'state': state,
        'run_name': run_dir.name.removeprefix('.').removesuffix('.pending'),
    }


def _dependent_directories(project_root: Path, root_run: Path) -> list[Path]:
    """按 manifest／解析参数的显式依赖闭包定位同数据集下的下游产物。"""
    root_entry = _entry_from_any_module(project_root, root_run)
    if root_entry is None:
        raise ValueError('运行目录不包含可识别的模块配置')
    dataset_id = str(root_entry['dataset_id'])
    root_index = MODULE_SEQUENCE.index(str(root_entry['module_name']))
    targets: set[Path] = {root_run.resolve()}
    changed = True
    while changed:
        changed = False
        for name in MODULE_SEQUENCE[root_index + 1:]:
            dataset_dir = _module_root(project_root, name) / dataset_id
            if not dataset_dir.is_dir():
                continue
            for candidate in dataset_dir.iterdir():
                if not candidate.is_dir() or candidate.resolve() in targets:
                    continue
                dependencies = _dependency_manifests(project_root, candidate)
                if any(path.parent.resolve() in targets for path in dependencies):
                    targets.add(candidate.resolve())
                    changed = True
    return sorted(targets, key=lambda item: (len(item.parts), item.as_posix()))


def _entry_from_any_module(project_root: Path, run_dir: Path) -> dict[str, object] | None:
    """在已知编号模块根中找到一个目录的配置，供依赖闭包使用。"""
    for module_name in MODULE_SEQUENCE:
        try:
            return _entry_from_directory(project_root, module_name, _artifact_directory(
                project_root, module_name, run_dir.relative_to(project_root).as_posix()))
        except ValueError:
            continue
    return None


def _dependency_manifests(project_root: Path, run_dir: Path) -> set[Path]:
    """从成功 manifest 或失败／运行中的解析参数收集直接上游 manifest。"""
    paths: set[Path] = set()
    manifest = _read_json_object(run_dir / 'manifest.json')
    if manifest is not None:
        dependencies = manifest.get('dependencies')
        if isinstance(dependencies, list):
            for item in dependencies:
                if isinstance(item, dict) and isinstance(item.get('manifest'), str):
                    path = _resolve_project_file(project_root, item['manifest'])
                    if path is not None:
                        paths.add(path)
    config = _read_json_object(run_dir / 'resolved_config.json')
    values = config.get('module') if isinstance(config, dict) else None
    if isinstance(values, dict):
        for value in values.values():
            if isinstance(value, str) and value.endswith('manifest.json'):
                path = _resolve_project_file(project_root, value)
                if path is not None:
                    paths.add(path)
    return paths


def _resolve_project_file(project_root: Path, value: str) -> Path | None:
    """解析项目内依赖清单；历史损坏或外部路径不参与删除闭包。"""
    try:
        path = (Path(value) if Path(value).is_absolute() else project_root / value).resolve()
        path.relative_to(project_root.resolve())
    except ValueError:
        return None
    return path if path.name == 'manifest.json' else None


def _read_json_object(path: Path) -> dict[str, object] | None:
    """读取小型元数据对象，损坏历史条目只影响自身展示。"""
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _read_log_tail(project_root: Path, entry: dict[str, object]) -> tuple[str, bool]:
    """读取日志末尾，避免历史页一次传输整份长日志。"""
    required = ('run_timestamp', 'dataset_name', 'run_id')
    if not all(isinstance(entry.get(name), str) for name in required):
        return '', False
    seed = entry.get('seed')
    if not isinstance(seed, int):
        return '', False
    filename = f"{entry['run_timestamp']}#{entry['dataset_name']}#{entry['module_name']}#{entry['run_id']}_{seed}.log"
    path = project_root / 'logs' / filename
    if not path.is_file():
        return '', False
    limit = 96 * 1024
    size = path.stat().st_size
    with path.open('rb') as reader:
        if size > limit:
            reader.seek(-limit, 2)
            raw = reader.read()
            text = raw.decode('utf-8', errors='replace')
            return text[text.find('\n') + 1:] if '\n' in text else text, True
        return reader.read().decode('utf-8', errors='replace'), False
