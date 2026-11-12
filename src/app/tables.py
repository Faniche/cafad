"""tables.py | 2026-11-12 | CAFAD 持久化表的 Parquet 与 DuckDB 读写 | Codex."""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import re

import duckdb
import pandas as pd


_IDENTIFIER = re.compile(r'[A-Za-z_][A-Za-z0-9_]*\Z')


def table_path(directory: Path, name: str) -> Path:
    """返回正式表的唯一 Parquet 路径，拒绝目录穿越或隐式扩展名。"""
    if not _IDENTIFIER.fullmatch(name):
        raise ValueError(f'非法表名：{name!r}')
    return directory / f'{name}.parquet'


def write_table(frame: pd.DataFrame, directory: Path, name: str) -> Path:
    """使用 Zstandard 压缩写入无索引 Parquet，并返回实际路径。"""
    directory.mkdir(parents=True, exist_ok=True)
    path = table_path(directory, name)
    frame.to_parquet(path, engine='pyarrow', compression='zstd', index=False)
    return path


def read_table(directory: Path, name: str, *, columns: Iterable[str] | None = None,
               dtype: dict[str, str] | None = None, keep_default_na: bool = True) -> pd.DataFrame:
    """经 DuckDB 读取 Parquet；仅选择请求列，并在边界恢复指定 Pandas dtype。"""
    path = table_path(directory, name)
    if not path.is_file():
        raise FileNotFoundError(f'缺少表：{path}')
    selected = '*' if columns is None else ', '.join(_column_sql(name) for name in columns)
    with duckdb.connect(':memory:') as connection:
        frame = connection.execute(f'SELECT {selected} FROM read_parquet(?)', [str(path)]).fetchdf()
    if dtype:
        for column, target in dtype.items():
            if column in frame:
                frame[column] = frame[column].astype(target)
    if not keep_default_na:
        string_columns = frame.select_dtypes(include=['object', 'string']).columns
        frame.loc[:, string_columns] = frame.loc[:, string_columns].fillna('')
    return frame


def export_csv(directory: Path, name: str, destination: Path) -> None:
    """仅为旧数值内核生成运行期 CSV；该临时文件不得作为发布产物。"""
    source = table_path(directory, name)
    if not source.is_file():
        raise FileNotFoundError(f'缺少表：{source}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    escaped_destination = str(destination.resolve()).replace("'", "''")
    with duckdb.connect(':memory:') as connection:
        connection.execute(
            f"COPY (SELECT * FROM read_parquet(?)) TO '{escaped_destination}' (FORMAT CSV, HEADER)", [str(source)]
        )


def _column_sql(name: str) -> str:
    if not _IDENTIFIER.fullmatch(name):
        raise ValueError(f'非法列名：{name!r}')
    return f'"{name}"'
