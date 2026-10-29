from pathlib import Path


class PathUtils:
    @staticmethod
    def get_project_root() -> Path:
        """Get the root directory of the project."""

        current_dir = Path(__file__).resolve().parent
        while current_dir != current_dir.parent:
            if (current_dir / "pyproject.toml").exists() or (
                current_dir / ".git"
            ).exists():
                return current_dir
            current_dir = current_dir.parent

        return current_dir

    @staticmethod
    def get_data_dir() -> Path:
        """Get the data directory path."""
        return PathUtils.get_project_root() / "data"

