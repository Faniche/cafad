"""search_pipeline.py | 2026-11-09 | 串行调用标准模块并保存参数搜索中间结果 | Codex."""
from collections.abc import Iterator
from contextlib import contextmanager
import csv
import json
import logging
import os
from pathlib import Path
import subprocess
import sys

from src.app.artifacts import write_json


class FatalDeviceError(RuntimeError):
    """设备故障应终止整个搜索，不把失效环境当成下一组超参数的正常运行环境。"""


FATAL_DEVICE_MARKERS = ('cuda error:', 'cuda initialization:', 'cuda_device_unavailable',
                        'device-side assert triggered', 'gpu has fallen off the bus')


@contextmanager
def search_lock(path: Path) -> Iterator[None]:
    """锁由操作系统持有，进程异常退出也释放；不占用正式模块自己的运行锁。"""
    with path.open('a+b') as handle:
        handle.seek(0); handle.write(b'0'); handle.flush(); handle.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == 'nt':
                handle.seek(0); msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def run_module(root: Path, module: str, source: Path, options: dict[str, object], seed: int,
               directory: Path, logger: logging.Logger) -> Path:
    """继承搜索任务进程组，标准 main.py 独立分配日志/计数/产物并验收。"""
    registry = root / 'run_registry.json'
    before = {row['artifact_dir'] for row in json.loads(registry.read_text(encoding='utf-8')).get('runs', [])} if registry.exists() else set()
    command = [sys.executable, str(root / 'main.py'), '--seed', str(seed), '--console', '--log-level', 'DEBUG',
               '-m', module, '--input', str(source)]
    for key, value in options.items():
        command.extend(['--' + key.replace('_', '-'), str(value)])
    write_json(directory / f'{module}_command.json', {'command': command, 'input': str(source)})
    logger.info('启动 %s：%s', module, subprocess.list2cmdline(command))
    device_failure = None
    with subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding='utf-8', errors='replace', env={**os.environ, 'PYTHONUNBUFFERED': '1'}) as process:
        try:
            for line in process.stdout:
                logger.info('[%s] %s', module, line.rstrip())
                if device_failure is None and any(marker in line.lower() for marker in FATAL_DEVICE_MARKERS):
                    device_failure = line.strip()
            code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
            raise
    if device_failure is not None:
        raise FatalDeviceError(f'{module} 检测到设备故障，停止搜索；{device_failure}')
    rows = json.loads(registry.read_text(encoding='utf-8')).get('runs', []) if registry.exists() else []
    candidates = [root / row['artifact_dir'] / 'manifest.json' for row in rows
                  if row['module'] == module and row['artifact_dir'] not in before and row['state'] == 'success']
    candidates = [path for path in candidates if path.is_file() and
                  (root / Path(json.loads((path.parent / 'resolved_config.json').read_text(encoding='utf-8'))['module']['input'])).resolve() == source.resolve()]
    if code != 0 or len(candidates) != 1:
        raise RuntimeError(f'{module} 未成功完成或无法唯一定位产物，exit={code}；详见模块日志')
    return candidates[0]


def pipeline(root: Path, config: dict, parameters: dict, directory: Path, logger: logging.Logger) -> float:
    """每组四步全部完成后返回验证片段 F1，中间清单实时落盘。"""
    source = root / config['group_manifest']
    links = {}
    steps = [('train', {**config['train_fixed'], **parameters, 'gpus': config['gpus']}),
             ('calibrate', {**config['calibrate'], 'gpus': config['gpus']}),
             ('detect', {**config['detect'], 'gpus': config['gpus']}), ('evaluate', {})]
    for module, options in steps:
        source = run_module(root, module, source.resolve(), options, config['training_seed'], directory, logger)
        links[module] = str(source)
        write_json(directory / 'artifacts.json', links)
    metrics = json.loads((source.parent / 'data/metrics.json').read_text(encoding='utf-8'))
    if metrics['split'] != 'val':
        raise RuntimeError('搜索只允许验证集指标，禁止使用测试集选择参数')
    score = float(metrics['segment']['f1'])
    write_json(directory / 'metrics.json', metrics)
    return score


def export_results(study: object, directory: Path) -> None:
    """每组完成或失败后输出排名，SQLite 之外保留可直接阅读的结果。"""
    trials = study.get_trials(deepcopy=False)
    keys = sorted({key for trial in trials for key in trial.params})
    temporary = directory / 'results.csv.tmp'
    with temporary.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=['trial', 'phase', 'state', 'validation_segment_f1', *keys])
        writer.writeheader()
        for trial in sorted(trials, key=lambda item: item.value if item.value is not None else -1, reverse=True):
            writer.writerow({'trial': trial.number, 'phase': trial.user_attrs.get('phase'), 'state': trial.state.name,
                             'validation_segment_f1': trial.value, **trial.params})
    temporary.replace(directory / 'results.csv')
    completed = [trial for trial in trials if trial.state.name == 'COMPLETE']
    if completed:
        best = max(completed, key=lambda trial: trial.value)
        write_json(directory / 'best.json', {'trial': best.number, 'validation_segment_f1': best.value,
                                            'parameters': best.params, 'artifacts': best.user_attrs.get('artifacts')})
