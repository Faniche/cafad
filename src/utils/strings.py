"""strings.py | 2026-11-03 | 项目根目录与项目内路径工具 | Codex."""
from __future__ import annotations

from pathlib import Path


class PathUtils:
    @staticmethod
    def get_project_root() -> Path:
        """返回项目根目录，兼容未携带 `.git` 的服务器代码副本。"""

        current_dir = Path(__file__).resolve().parent
        while current_dir != current_dir.parent:
            if ((current_dir / 'pyproject.toml').is_file()
                    or (current_dir / '.git').exists()
                    or ((current_dir / 'main.py').is_file() and (current_dir / 'src').is_dir())):
                return current_dir
            current_dir = current_dir.parent
        raise RuntimeError('无法从 src/utils/strings.py 的父目录识别 CAFAD 项目根目录')

    @staticmethod
    def get_data_dir() -> Path:
        """返回项目内数据目录路径，不依赖命令启动时的当前工作目录。"""
        return PathUtils.get_project_root() / "data"

