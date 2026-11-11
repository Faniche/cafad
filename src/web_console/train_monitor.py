"""train_monitor.py | 2026-11-09 | 读取训练快照，兼容旧运行日志并采集 GPU 概况 | Codex."""
import re
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from .history import list_module_history, _artifact_directory, _read_json_object, _read_log_tail

_gpu_lock = threading.Lock()
_gpu_cache: tuple[float, list[dict[str, object]]] = (0.0, [])


def gpu_snapshot() -> list[dict[str, object]]:
    """跨浏览器共享一秒缓存；无 NVIDIA 设备时仍可看训练进度。"""
    global _gpu_cache
    with _gpu_lock:
        now = time.monotonic()
        if now - _gpu_cache[0] < 1:
            return _gpu_cache[1]
        rows = []
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
                                    capture_output=True, text=True, timeout=2, check=True)
            for line in result.stdout.splitlines():
                index, memory, utilization = [item.strip() for item in line.split(',')]
                rows.append({'device': f'cuda:{index}', 'memory_used_mib': float(memory), 'utilization': float(utilization)})
        except (OSError, subprocess.SubprocessError, ValueError):
            rows = []
        _gpu_cache = (time.monotonic(), rows)
        return rows


def legacy_rows(text: str, epochs: int) -> list[dict[str, object]]:
    """旧进程没有细粒度事件，只显示日志确认的字段，不捏造批进度。"""
    rows = {}
    for line in text.splitlines():
        try:
            timestamp = datetime.strptime(line[:24], '%Y-%m-%d %H:%M:%S%z').timestamp()
        except ValueError:
            timestamp = None
        match = re.search(r'领取专家任务：.*device=(\S+)；group=(\S+)', line)
        if match:
            device, group = match.groups()
            rows.setdefault(group, {'group_id': group, 'epochs': epochs}).update(device=device, state='training',
                                                                                started_at=timestamp, updated_at=timestamp)
        match = re.search(r'专家 (\S+) epoch=(\d+)；train_loss=([^；]+)；validation_loss=([^；]+).*device=(\S+)', line)
        if match:
            group, epoch, loss, validation, device = match.groups()
            rows.setdefault(group, {'group_id': group, 'epochs': epochs}).update(
                epoch=int(epoch), train_loss=float(loss), validation_loss=None if validation == 'NA' else float(validation),
                device=device, state='training', updated_at=timestamp)
        match = re.search(r'专家 (\S+) 完成：device=(\S+)；', line)
        if match:
            group, device = match.groups()
            row = rows.setdefault(group, {'group_id': group, 'epochs': epochs})
            row.update(device=device, state='completed', updated_at=timestamp,
                       elapsed_seconds=timestamp - row['started_at'] if timestamp and row.get('started_at') else None)
    return list(rows.values())


def training_progress(root: Path, artifact: str | None = None) -> dict[str, object]:
    """每次只读取小型快照；请求路径沿用历史接口的目录约束。"""
    runs = list_module_history(root, 'train')
    if artifact is None:
        if not runs:
            return {'run': None, 'experts': [], 'gpus': []}
        artifact = str(runs[0]['artifact'])
    directory = _artifact_directory(root, 'train', artifact)
    entry = next((row for row in runs if row['artifact'] == artifact), None)
    if entry is None:
        raise ValueError('找不到训练运行')
    config = _read_json_object(directory / 'resolved_config.json')
    snapshot = _read_json_object(directory / 'training_progress.json')
    mode = 'events' if snapshot is not None else 'legacy_logs'
    if snapshot is None:
        text, _ = _read_log_tail(root, entry)
        snapshot = {'experts': legacy_rows(text, config['module'].get('epochs', 5)), 'updated_at': None}
    for row in snapshot['experts']:
        if entry['state'] == 'failure' and row.get('state') != 'completed':
            row['state'] = 'interrupted'
    return {'run': entry, 'mode': mode, **snapshot, 'gpus': gpu_snapshot(), 'server_time': time.time()}
