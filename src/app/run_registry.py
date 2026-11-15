"""run_registry.py | 2026-11-04 | 自动运行编号、数据集身份及持久化台账 | Codex."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
import sqlite3

from .artifacts import write_json
from .module_catalog import artifact_module_directory


RECORD_COLUMNS = ('dataset', 'module', 'counter', 'state')
_NAME_PATTERN = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]*\Z')


@dataclass(frozen=True)
class DatasetIdentity:
    """从模块输入确定的稳定数据集 ID 与供人阅读的文件名。"""

    dataset_id: str
    dataset_name: str


@dataclass(frozen=True)
class RunAllocation:
    """一个已经占用计数器的模块运行身份。"""

    dataset: DatasetIdentity
    module_name: str
    counter: int
    seed: int
    started_at: datetime

    @property
    def run_id(self) -> str:
        """返回模块内单调递增的运行编号。"""
        return str(self.counter)

    @property
    def timestamp(self) -> str:
        """返回用于目录与日志的本地秒级时间戳。"""
        return self.started_at.strftime('%Y%m%d_%H%M%S')

    @property
    def directory_name(self) -> str:
        """返回不可人为指定的最终产物目录名。"""
        return f'{self.run_id}#{self.timestamp}#{self.seed}'

    @property
    def log_filename(self) -> str:
        """返回统一的模块日志文件名。"""
        return f'{self.timestamp}#{self.dataset.dataset_name}#{self.module_name}#{self.run_id}_{self.seed}.log'


def identify_dataset(project_root: Path, module_name: str, args: argparse.Namespace) -> DatasetIdentity:
    """从 preprocess 源文件或上游 manifest 读取数据集身份。"""
    value = getattr(args, 'input', None)
    if not isinstance(value, Path):
        raise ValueError(f'{module_name} 模块必须通过 --input 提供输入文件')
    input_path = value if value.is_absolute() else project_root / value
    input_path = input_path.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f'找不到输入文件：{input_path}')
    if module_name == 'preprocess':
        return _identity_from_filename(input_path)
    return _identity_from_manifest(input_path)


def allocate_run(project_root: Path, dataset: DatasetIdentity, module_name: str, seed: int) -> RunAllocation:
    """在项目锁持有期间以 SQLite 事务占用计数器，并同步网页历史 JSON。"""
    data_dir = project_root / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now().astimezone()
    database = _run_database(data_dir)
    try:
        _migrate_counter_csv(data_dir, database)
        database.execute('BEGIN IMMEDIATE')
        counter = int(database.execute(
            'SELECT COALESCE(MAX(counter), 0) + 1 FROM runs WHERE dataset = ? AND module = ?',
            (dataset.dataset_id, module_name),
        ).fetchone()[0])
        database.execute(
            'INSERT INTO runs(dataset, module, counter, state, seed, started_at) VALUES (?, ?, ?, ?, ?, ?)',
            (dataset.dataset_id, module_name, counter, 'running', seed, started_at.isoformat()),
        )
        database.commit()
    finally:
        database.close()
    allocation = RunAllocation(dataset=dataset, module_name=module_name, counter=counter, seed=seed,
                               started_at=started_at)
    registry_path = project_root / 'run_registry.json'
    registry = _read_registry(registry_path)
    registry['runs'].append({
        'dataset': dataset.dataset_id,
        'dataset_name': dataset.dataset_name,
        'module': module_name,
        'counter': allocation.counter,
        'run_id': allocation.run_id,
        'seed': seed,
        'started_at': allocation.started_at.isoformat(),
        'artifact_dir': (Path('data') / artifact_module_directory(module_name) / dataset.dataset_id /
                         allocation.directory_name).as_posix(),
        'state': 'running',
    })
    write_json(registry_path, registry)
    return allocation


def update_run_state(project_root: Path, allocation: RunAllocation, state: str) -> None:
    """将已占用运行的最终状态事务写入 SQLite，并同步网页历史 JSON。"""
    if state not in {'success', 'failure'}:
        raise ValueError('运行最终状态只能是 success 或 failure')
    database = _run_database(project_root / 'data')
    try:
        result = database.execute(
            'UPDATE runs SET state = ?, ended_at = ? WHERE dataset = ? AND module = ? AND counter = ? AND started_at = ?',
            (state, datetime.now().astimezone().isoformat(), allocation.dataset.dataset_id,
             allocation.module_name, allocation.counter, allocation.started_at.isoformat()),
        )
        if result.rowcount != 1:
            raise RuntimeError('runs.sqlite 中找不到当前运行记录')
        database.commit()
    finally:
        database.close()

    registry_path = project_root / 'run_registry.json'
    registry = _read_registry(registry_path)
    for record in registry['runs']:
        if (record.get('dataset') == allocation.dataset.dataset_id
                and record.get('module') == allocation.module_name
                and record.get('counter') == allocation.counter
                and record.get('started_at') == allocation.started_at.isoformat()):
            record['state'] = state
            record['ended_at'] = datetime.now().astimezone().isoformat()
            write_json(registry_path, registry)
            return
    raise RuntimeError('run_registry.json 中找不到当前运行记录')


def _identity_from_filename(path: Path) -> DatasetIdentity:
    """以不含扩展名的完整输入文件名作为数据集身份。"""
    dataset_name = path.stem
    _validate_name(dataset_name, '输入文件名')
    return DatasetIdentity(dataset_id=dataset_name, dataset_name=dataset_name)


def _identity_from_manifest(path: Path) -> DatasetIdentity:
    """从已经落盘的上游清单读取 preprocess 写入的数据集身份。"""
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        raise ValueError(f'输入 manifest 不是有效 JSON：{path}') from exc
    if not isinstance(value, dict):
        raise ValueError(f'输入 manifest 顶层必须为对象：{path}')
    dataset_id = value.get('dataset_id')
    dataset_name = value.get('dataset_name', dataset_id)
    if not isinstance(dataset_id, str) or not isinstance(dataset_name, str):
        raise ValueError(f'输入 manifest 缺少 dataset_id 或 dataset_name：{path}')
    _validate_name(dataset_id, '数据集 ID')
    _validate_name(dataset_name, '数据集名称')
    return DatasetIdentity(dataset_id=dataset_id, dataset_name=dataset_name)


def _validate_name(value: str, label: str) -> None:
    """确保目录、日志分隔符和台账键不会因文件名歧义。"""
    if not _NAME_PATTERN.fullmatch(value) or '#' in value:
        raise ValueError(f'{label}只能包含字母、数字、点、下划线和连字符，且不能包含 #：{value!r}')


def _read_counter_rows(path: Path) -> list[dict[str, str]]:
    """读取计数台账；不存在时按空台账处理。"""
    if not path.exists():
        return []
    with path.open('r', encoding='utf-8-sig', newline='') as reader:
        table = csv.DictReader(reader)
        if tuple(table.fieldnames or ()) != RECORD_COLUMNS:
            raise ValueError(f'{path} 表头必须为 {list(RECORD_COLUMNS)}')
        rows = list(table)
    for row in rows:
        if set(row) != set(RECORD_COLUMNS) or not row['dataset'] or not row['module']:
            raise ValueError(f'{path} 包含无效记录')
        if not row['counter'].isdigit() or int(row['counter']) < 1:
            raise ValueError(f'{path} 包含无效 counter')
    return rows


def _run_database(data_dir: Path) -> sqlite3.Connection:
    """打开运行台账 SQLite；正式运行锁保证写入者只有一个。"""
    database = sqlite3.connect(data_dir / 'runs.sqlite')
    database.execute('PRAGMA journal_mode=WAL')
    database.execute('''CREATE TABLE IF NOT EXISTS runs (
        dataset TEXT NOT NULL, module TEXT NOT NULL, counter INTEGER NOT NULL,
        state TEXT NOT NULL, seed INTEGER, started_at TEXT NOT NULL, ended_at TEXT,
        PRIMARY KEY(dataset, module, counter)
    )''')
    return database


def _migrate_counter_csv(data_dir: Path, database: sqlite3.Connection) -> None:
    """首次启用 SQLite 时无损导入旧计数器；旧 CSV 不再参与后续分配。"""
    if database.execute('SELECT COUNT(*) FROM runs').fetchone()[0]:
        return
    legacy = data_dir / 'record_counter.csv'
    if not legacy.is_file():
        return
    for row in _read_counter_rows(legacy):
        database.execute(
            'INSERT OR IGNORE INTO runs(dataset, module, counter, state, seed, started_at) VALUES (?, ?, ?, ?, NULL, ?)',
            (row['dataset'], row['module'], int(row['counter']), row['state'], 'legacy'),
        )
    database.commit()


def _read_registry(path: Path) -> dict[str, object]:
    """读取根目录运行 JSON；不存在时生成当前格式的空台账。"""
    if not path.exists():
        return {'schema_version': 1, 'runs': []}
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict) or value.get('schema_version') != 1 or not isinstance(value.get('runs'), list):
        raise ValueError(f'运行台账格式无效：{path}')
    return value
