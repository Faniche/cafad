"""file_browser.py | 2026-11-04 | Web 控制台受限资源目录浏览 | Codex."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping


MAX_DIRECTORY_ENTRIES = 500


class FileBrowserError(ValueError):
    """浏览根目录、相对路径或目录状态不符合安全约束。"""


def browser_roots(project_root: Path) -> dict[str, Path]:
    """返回可浏览的当前用户目录与项目目录，均解析为绝对路径。"""
    return {'home': Path.home().resolve(), 'project': project_root.resolve()}


def list_directory(roots: Mapping[str, Path], root_name: str, relative_path: str) -> dict[str, object]:
    """在指定根内列出非隐藏子项；不返回任意文件内容。"""
    root = roots.get(root_name)
    if root is None:
        raise FileBrowserError('文件浏览根目录无效')
    directory = _resolve_under_root(root, relative_path)
    if not directory.is_dir():
        raise FileBrowserError('选择的路径不是目录')
    try:
        children = sorted(
            (item for item in directory.iterdir() if not item.name.startswith('.')),
            key=lambda item: (not item.is_dir(), item.name.casefold()),
        )
    except OSError as exc:
        raise FileBrowserError(f'无法读取目录：{exc}') from exc
    visible = children[:MAX_DIRECTORY_ENTRIES]
    entries: list[dict[str, str]] = []
    for child in visible:
        try:
            resolved = child.resolve()
            if not _is_under_root(resolved, root):
                continue
            kind = 'directory' if resolved.is_dir() else 'file'
            entries.append({
                'name': child.name,
                'kind': kind,
                'path': resolved.relative_to(root).as_posix(),
                'absolute_path': str(resolved),
            })
        except OSError:
            continue
    relative = directory.relative_to(root).as_posix()
    parent_relative = directory.parent.relative_to(root).as_posix() if directory != root else None
    return {
        'root': root_name,
        'root_path': str(root),
        'path': '' if relative == '.' else relative,
        'parent': '' if parent_relative == '.' else parent_relative,
        'entries': entries,
        'truncated': len(children) > len(visible),
    }


def _resolve_under_root(root: Path, relative_path: str) -> Path:
    """解析用户给定的相对路径，并拒绝越出浏览根的跳转和符号链接。"""
    candidate = Path(relative_path)
    if candidate.is_absolute():
        raise FileBrowserError('文件浏览路径必须相对所选根目录')
    resolved = (root / candidate).resolve()
    if not _is_under_root(resolved, root):
        raise FileBrowserError('文件浏览路径越出允许的根目录')
    return resolved


def _is_under_root(path: Path, root: Path) -> bool:
    """兼容较低 Python 版本地判断 path 是否属于 root。"""
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
