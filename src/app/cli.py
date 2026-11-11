"""cli.py | 2026-11-03 | 唯一根入口的全局参数和模块调度 | Codex."""
from __future__ import annotations

import argparse
from datetime import datetime
from importlib import import_module
import logging
import sys
import time
from types import ModuleType

from .arguments import ProjectArgumentParser
from .artifacts import ArtifactWorkspace, ProjectRunLock, write_manifest, write_run_metadata
from .contracts import RunContext
from .module_catalog import MODULE_SEQUENCE, artifact_module_directory
from .run_registry import allocate_run, identify_dataset, update_run_state
from src.utils.env import Env
from src.utils.logging_utils import close_project_logging, setup_project_logging
from src.utils.strings import PathUtils


IMPLEMENTED_MODULES = {
    'preprocess': 'src.modules.preprocess.api',
    'split': 'src.modules.split.api',
    'inject': 'src.modules.inject.api',
    'segment': 'src.modules.segment.api',
    'group': 'src.modules.group.api',
    'train': 'src.modules.train.api',
    'calibrate': 'src.modules.calibrate.api',
    'detect': 'src.modules.detect.api',
    'evaluate': 'src.modules.evaluate.api',
}


def _global_parser() -> argparse.ArgumentParser:
    parser = ProjectArgumentParser(add_help=False, description='CAFAD 项目统一入口。')
    parser.add_argument('--seed', type=int, default=42, help='当前模块的随机种子')
    parser.add_argument('--console', action='store_true', help='同时向控制台镜像日志')
    parser.add_argument('--log-level', choices=('DEBUG', 'INFO', 'WARNING', 'ERROR'), default='INFO')
    return parser


def _split_arguments(argv: list[str]) -> tuple[list[str], str, list[str]]:
    """按唯一 -m 分开全局参数和模块参数。"""
    positions = [index for index, token in enumerate(argv) if token in {'-m', '--module'}]
    if len(positions) != 1:
        raise ValueError('必须且只能指定一次 -m 或 --module')
    position = positions[0]
    if position + 1 >= len(argv):
        raise ValueError('-m 后必须提供模块名')
    return argv[:position], argv[position + 1], argv[position + 2:]


def _module_api(module_name: str) -> ModuleType:
    """延迟载入已实现模块，防止未选择模块的依赖影响当前运行。"""
    if module_name not in IMPLEMENTED_MODULES:
        available = ', '.join(sorted(IMPLEMENTED_MODULES))
        raise ValueError(f'模块 {module_name!r} 尚未实现；当前可用：{available}')
    return import_module(IMPLEMENTED_MODULES[module_name])


def _root_help() -> str:
    names = ', '.join(name for name in MODULE_SEQUENCE if name in IMPLEMENTED_MODULES)
    return ('用法：python main.py [全局参数] -m 模块名 [模块参数]\n\n'
            f'当前可用模块：{names}\n'
            '全局参数必须在 -m 前；模块参数由相应模块的 api.py 定义。\n'
            '数据集身份取 --input 文件名（不含扩展名）或上游 manifest；运行 ID 自动分配。\n'
            '示例：python main.py -m preprocess --help')


def _format_elapsed(seconds: float) -> str:
    """将单调计时器的秒数格式化为紧凑、可比较的运行时长。"""
    whole_seconds = max(0, round(seconds))
    hours, remainder = divmod(whole_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def main(argv: list[str] | None = None) -> int:
    """解析一次命令，只运行一个模块并在产物校验后发布。"""
    values = list(sys.argv[1:] if argv is None else argv)
    if not values or values == ['--help'] or values == ['-h']:
        print(_root_help())
        return 0
    logger = None
    try:
        global_args, module_name, module_args = _split_arguments(values)
        api = _module_api(module_name)
        if module_args in (['--help'], ['-h']):
            api.build_parser().print_help()
            return 0
        parser = _global_parser()
        global_values = parser.parse_args(global_args)
        module_values = api.build_parser().parse_args(module_args)
        project_root = PathUtils.get_project_root()
        Env.load(project_root / '.env')
        workers = Env.data_process_workers()
        dataset = identify_dataset(project_root, module_name, module_values)
        level = getattr(logging, global_values.log_level)
        with ProjectRunLock(project_root):
            allocation = allocate_run(project_root, dataset, module_name, global_values.seed)
            workspace = ArtifactWorkspace(project_root / 'data' / artifact_module_directory(module_name) / dataset.dataset_id /
                                          allocation.directory_name)
            try:
                logger, log_path = setup_project_logging(console=global_values.console, project_root=project_root,
                                                         name=f'cafad.{module_name}', level=level,
                                                         filename=allocation.log_filename)
            except Exception:
                update_run_state(project_root, allocation, 'failure')
                raise
            state = 'failure'
            started_monotonic = time.monotonic()
            try:
                logger.info('运行开始时间：%s', allocation.started_at.isoformat())
                workspace.create()
                logger.debug('已创建暂存产物目录：%s', workspace.staging_dir)
                context = RunContext(project_root=project_root, module_name=module_name,
                                     dataset_id=dataset.dataset_id, dataset_name=dataset.dataset_name,
                                     run_id=allocation.run_id, run_timestamp=allocation.timestamp,
                                     seed=global_values.seed, workers=workers,
                                     staging_dir=workspace.staging_dir, final_dir=workspace.final_dir, logger=logger)
                logger.info('启动模块：%s；dataset=%s；dataset_name=%s；run_id=%s；workers=%d；日志=%s',
                            module_name, context.dataset_id, context.dataset_name, context.run_id,
                            context.workers, log_path)
                logger.debug('已解析全局参数：%s；模块参数：%s', vars(global_values), vars(module_values))
                write_run_metadata(context, vars(global_values), vars(module_values))
                result = api.run(context, module_values)
                logger.debug('模块逻辑执行完成，待验收：summary=%s；dependencies=%s', result.summary, result.dependencies)
                report = api.validate_outputs(context, module_values, result)
                logger.debug('产物验收通过：%s', report.checks)
                write_manifest(context, result, report)
                logger.debug('已写入 manifest，准备发布暂存产物：%s', workspace.final_dir)
                workspace.publish()
                state = 'success'
                logger.info('模块完成并发布：%s', context.final_dir)
            except Exception:
                logger.exception('模块执行失败')
                workspace.discard('模块执行或落盘校验失败；详细异常请查看本次日志')
            except (KeyboardInterrupt, SystemExit):
                logger.warning('模块被中断，运行将标记为 failure')
                workspace.discard('模块被中断；详细信息请查看本次日志')
                raise
            finally:
                try:
                    update_run_state(project_root, allocation, state)
                finally:
                    logger.info('运行结束时间：%s；state=%s；运行时长=%s', datetime.now().astimezone().isoformat(), state,
                                _format_elapsed(time.monotonic() - started_monotonic))
                    close_project_logging(logger)
                    logger = None
            return 0 if state == 'success' else 1
    except SystemExit:
        raise
    except Exception:
        if logger is not None:
            logger.exception('模块执行失败')
        else:
            bootstrap_logger, _ = setup_project_logging(console='--console' in values,
                                                        project_root=PathUtils.get_project_root(), name='cafad.bootstrap')
            try:
                bootstrap_logger.exception('启动失败；请检查命令行参数和环境配置。')
            finally:
                close_project_logging(bootstrap_logger)
        return 1
    finally:
        if logger is not None:
            close_project_logging(logger)
