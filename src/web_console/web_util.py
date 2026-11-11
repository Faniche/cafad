"""web_util.py | 2026-11-09 | 模块历史参数查询及启动命令预览 | Codex."""
from pathlib import Path
from src.app.cli import _global_parser, _module_api, _split_arguments
from src.app.module_catalog import MODULE_SEQUENCE
from .history import list_module_history, _read_json_object
from .service import module_schemas


class ModuleHistoryUtil:
    """历史以落盘配置为准；命令预览只用于尚未生成运行目录的任务。"""

    @staticmethod
    def _read(root: Path, module: str) -> list[dict[str, object]]:
        rows = list_module_history(root, module)
        for row in rows:
            config = _read_json_object(root / str(row['artifact']) / 'resolved_config.json') or {}
            row['parameters'] = config.get('module', {})
        return rows

    @staticmethod
    def preprocess(root: Path) -> list[dict[str, object]]:
        return ModuleHistoryUtil._read(root, 'preprocess')

    @staticmethod
    def split(root: Path) -> list[dict[str, object]]:
        return ModuleHistoryUtil._read(root, 'split')

    @staticmethod
    def inject(root: Path) -> list[dict[str, object]]:
        return ModuleHistoryUtil._read(root, 'inject')

    @staticmethod
    def segment(root: Path) -> list[dict[str, object]]:
        return ModuleHistoryUtil._read(root, 'segment')

    @staticmethod
    def group(root: Path) -> list[dict[str, object]]:
        return ModuleHistoryUtil._read(root, 'group')

    @staticmethod
    def train(root: Path) -> list[dict[str, object]]:
        return ModuleHistoryUtil._read(root, 'train')

    @staticmethod
    def calibrate(root: Path) -> list[dict[str, object]]:
        return ModuleHistoryUtil._read(root, 'calibrate')

    @staticmethod
    def detect(root: Path) -> list[dict[str, object]]:
        return ModuleHistoryUtil._read(root, 'detect')

    @staticmethod
    def evaluate(root: Path) -> list[dict[str, object]]:
        return ModuleHistoryUtil._read(root, 'evaluate')

    @staticmethod
    def query(root: Path, module: str) -> dict[str, object]:
        if module not in MODULE_SEQUENCE:
            raise ValueError('模块不存在')
        columns = [{'key': spec.dest, 'label': spec.option} for spec in module_schemas().get(module, ())
                   if not spec.auto_derived]
        runs = getattr(ModuleHistoryUtil, module)(root)
        return {'runs': runs, 'columns': columns, 'running': any(row['state'] == 'running' for row in runs)}

    @staticmethod
    def from_command(command: tuple[str, ...]) -> dict[str, object]:
        """复用实际 argparse 解析默认值和列表；不执行命令、不猜测 run ID。"""
        global_args, module, args = _split_arguments(list(command[2:]))
        parameters = vars(_module_api(module).build_parser().parse_args(args))
        parameters = {key: str(value) if isinstance(value, Path) else value for key, value in parameters.items()}
        return {'module_name': module, 'parameters': parameters, 'seed': _global_parser().parse_args(global_args).seed,
                'state': 'starting', 'dataset_name': '等待落盘', 'run_name': '正在启动', 'run_timestamp': '—'}
