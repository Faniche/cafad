"""manual_merge.py | 2026-11-13 | 手工合并已发布阶段片段并保持所有关联表一致 | Codex."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import threading

import numpy as np
import pandas as pd

from src.app.artifacts import read_manifest, refresh_manifest_outputs, write_json
from src.app.tables import read_table, table_path, write_table
from src.data_processing.phase_pipeline import SEGMENT_COLUMNS
from src.data_processing.sun_phase import PHASES
from src.web_console.history import _artifact_directory, dependent_directories

from .api import _write_sequence_review


_MERGE_LOCK = threading.Lock()
_MERGE_COLUMNS = ('merge_id', 'created_at', 'flight_id', 'segment_id', 'source_segment_ids',
                  'source_phases', 'target_phase', 'source_points')


def _haversine_km(row: pd.Series) -> float | None:
    """计算合并片段首末位置的球面直线距离；坐标缺失时保留空值。"""
    values = [row['start_longitude_deg'], row['start_latitude_deg'],
              row['end_longitude_deg'], row['end_latitude_deg']]
    if any(pd.isna(value) for value in values):
        return None
    longitude = np.deg2rad([float(values[0]), float(values[2])])
    latitude = np.deg2rad([float(values[1]), float(values[3])])
    half = np.sin((latitude[1] - latitude[0]) / 2) ** 2
    half += np.cos(latitude[0]) * np.cos(latitude[1]) * np.sin((longitude[1] - longitude[0]) / 2) ** 2
    return float(6371.0088 * 2 * np.arcsin(np.sqrt(half)))


def _updated_segments(segments: pd.DataFrame, selected_ids: list[str], target_phase: str) -> tuple[pd.DataFrame, pd.Series]:
    """把一个航班的连续片段归并为首片段 ID，并重编该航班 ordinal。"""
    selected = segments.loc[segments['segment_id'].isin(selected_ids)].copy()
    if len(selected) != len(selected_ids):
        raise ValueError('所选片段中存在不属于当前分段运行的 ID')
    if selected['flight_id'].nunique() != 1:
        raise ValueError('只能合并同一航班的阶段片段')
    selected = selected.sort_values('ordinal', kind='stable')
    ordinals = selected['ordinal'].astype(int).tolist()
    if ordinals != list(range(ordinals[0], ordinals[0] + len(ordinals))):
        raise ValueError('只能合并同一航班中连续的阶段片段')
    source_ids = selected['segment_id'].astype(str).tolist()
    if source_ids != selected_ids:
        raise ValueError('所选片段顺序与该航班阶段顺序不一致')
    source_phases = selected['phase'].astype(str).tolist()
    if target_phase not in source_phases:
        raise ValueError('合并后的阶段标签必须从所选片段已有标签中选择')
    first, last = selected.iloc[0].copy(), selected.iloc[-1]
    first['phase'] = target_phase
    first['points'] = int(selected['points'].astype(int).sum())
    first['end_time'] = last['end_time']
    first['duration_seconds'] = (pd.Timestamp(last['end_time']) - pd.Timestamp(first['start_time'])).total_seconds()
    first['end_longitude_deg'] = last['end_longitude_deg']
    first['end_latitude_deg'] = last['end_latitude_deg']
    first['end_altitude_m'] = last['end_altitude_m']
    first['last_source_row_index'] = last['last_source_row_index']
    first['segment_label'] = int(selected['segment_label'].astype(int).max())
    for column in ('split', 'injection_plan'):
        if column in selected and selected[column].nunique(dropna=False) != 1:
            raise ValueError(f'所选片段的 {column} 不一致，拒绝合并')
    first['span_km'] = _haversine_km(first)
    anchor = source_ids[0]
    retained = segments.loc[~segments['segment_id'].isin(source_ids[1:])].copy()
    anchor_rows = retained.index[retained['segment_id'].eq(anchor)]
    if len(anchor_rows) != 1:
        raise RuntimeError('合并片段的保留 ID 不唯一')
    for column, value in first.items():
        retained.loc[anchor_rows[0], column] = value
    flight = str(first['flight_id'])
    flight_mask = retained['flight_id'].astype(str).eq(flight)
    ordered = retained.loc[flight_mask].sort_values('ordinal', kind='stable').copy()
    ordered.loc[:, 'ordinal'] = np.arange(len(ordered), dtype=np.int64)
    retained = pd.concat([retained.loc[~flight_mask], ordered], ignore_index=True)
    return retained.sort_values(['flight_id', 'ordinal'], kind='stable').reset_index(drop=True), first


def _validate_tables(segments: pd.DataFrame, mapping: pd.DataFrame) -> None:
    """校验手动修改后的片段 ID、点数、标签和阶段是否仍闭合。"""
    if segments['segment_id'].duplicated().any() or mapping['segment_id'].isna().any():
        raise RuntimeError('手动合并后出现重复或空的 segment_id')
    if set(mapping['segment_id'].astype(str)) != set(segments['segment_id'].astype(str)):
        raise RuntimeError('手动合并后点映射与片段表 ID 不一致')
    counts = mapping.groupby('segment_id', sort=False).size()
    actual = segments.set_index('segment_id')['points'].astype(int)
    if not counts.astype(int).eq(actual.reindex(counts.index)).all():
        raise RuntimeError('手动合并后片段点数与点映射不一致')
    phases = mapping.groupby('segment_id', sort=False)['phase'].nunique()
    if not phases.eq(1).all():
        raise RuntimeError('同一手动合并片段存在多个阶段标签')
    mapped_phase = mapping.groupby('segment_id', sort=False)['phase'].first().astype(str)
    if not mapped_phase.eq(segments.set_index('segment_id')['phase'].astype(str).reindex(mapped_phase.index)).all():
        raise RuntimeError('手动合并后点映射阶段与片段阶段不一致')
    labels = mapping.groupby('segment_id', sort=False)['injected_label'].max().astype(int)
    if not labels.eq(segments.set_index('segment_id')['segment_label'].astype(int).reindex(labels.index)).all():
        raise RuntimeError('手动合并后片段标签不等于逐点标签 OR')
    for _, frame in segments.groupby('flight_id', sort=False):
        if frame.sort_values('ordinal')['ordinal'].astype(int).tolist() != list(range(len(frame))):
            raise RuntimeError('手动合并后航班内 ordinal 不连续')


def _replace_tables(data_dir: Path, tables: dict[str, pd.DataFrame]) -> None:
    """先写入临时 Parquet，全部成功后才逐个替换正式表。"""
    temporary = data_dir / '.manual_merge_tables'
    if temporary.exists():
        raise RuntimeError('发现未清理的手动合并临时目录；请先检查该运行产物')
    temporary.mkdir()
    try:
        for name, frame in tables.items():
            write_table(frame, temporary, name)
        for name in tables:
            table_path(temporary, name).replace(table_path(data_dir, name))
    finally:
        if temporary.exists():
            temporary.rmdir()


def _replace_progress_database(path: Path, segments: pd.DataFrame) -> None:
    """由最终 segments 表重建展示 SQLite，避免表格展示与正式 Parquet 分叉。"""
    temporary = path.with_suffix('.sqlite.manual-merge')
    temporary.unlink(missing_ok=True)
    database = sqlite3.connect(temporary)
    try:
        database.execute('''CREATE TABLE segments (
            segment_id TEXT PRIMARY KEY, flight_id TEXT NOT NULL, icao TEXT NOT NULL, call TEXT NOT NULL,
            phase TEXT NOT NULL, points INTEGER NOT NULL, start_time TEXT NOT NULL, end_time TEXT NOT NULL,
            duration_seconds REAL NOT NULL, start_longitude_deg REAL, start_latitude_deg REAL,
            end_longitude_deg REAL, end_latitude_deg REAL, span_km REAL, start_altitude_m REAL, end_altitude_m REAL
        )''')
        columns = [name for name in SEGMENT_COLUMNS if name != 'ordinal' and name not in {'first_source_row_index', 'last_source_row_index'}]
        database.executemany(
            'INSERT INTO segments VALUES (' + ','.join('?' for _ in columns) + ')',
            [tuple(None if pd.isna(value) else value for value in row) for row in segments[columns].itertuples(index=False, name=None)],
        )
        database.commit()
    finally:
        database.close()
    temporary.replace(path)
    path.with_name(path.name + '-wal').unlink(missing_ok=True)
    path.with_name(path.name + '-shm').unlink(missing_ok=True)


def _merge_event(data_dir: Path, first: pd.Series, source_ids: list[str], source_phases: list[str]) -> pd.DataFrame:
    """追加不可变的人工合并账本，记录保留 ID 与所有来源阶段。"""
    existing_path = table_path(data_dir, 'manual_segment_merges')
    existing = read_table(data_dir, 'manual_segment_merges') if existing_path.is_file() else pd.DataFrame(columns=_MERGE_COLUMNS)
    now = datetime.now(timezone.utc).isoformat()
    event = pd.DataFrame([{
        'merge_id': f"{now}:{first['segment_id']}", 'created_at': now, 'flight_id': str(first['flight_id']),
        'segment_id': str(first['segment_id']), 'source_segment_ids': json.dumps(source_ids, ensure_ascii=False),
        'source_phases': json.dumps(source_phases, ensure_ascii=False), 'target_phase': str(first['phase']),
        'source_points': int(first['points']),
    }])
    return pd.concat([existing, event], ignore_index=True)[list(_MERGE_COLUMNS)]


def merge_segments(project_root: Path, artifact: str, segment_ids: list[str], target_phase: str) -> dict[str, object]:
    """合并一组连续片段，并同步重写正式表、SQLite、序列审计、摘要与 manifest。"""
    if len(segment_ids) < 2 or len(set(segment_ids)) != len(segment_ids) or not all(isinstance(value, str) and value for value in segment_ids):
        raise ValueError('请至少选择两个不重复的阶段片段')
    if target_phase not in PHASES:
        raise ValueError('合并后的阶段标签无效')
    with _MERGE_LOCK:
        if (project_root / '.cafad-run.lock').exists():
            raise RuntimeError('当前存在正式模块运行，不能同时修改阶段产物')
        run_dir = _artifact_directory(project_root, 'segment', artifact)
        read_manifest(run_dir / 'manifest.json', expected_module='segment')
        downstream = dependent_directories(project_root, run_dir)
        if len(downstream) > 1:
            paths = ', '.join(path.relative_to(project_root).as_posix() for path in downstream[1:])
            raise ValueError(f'该阶段运行已有下游产物，合并会使其失效；请先删除下游运行：{paths}')
        data_dir = run_dir / 'data'
        segments = read_table(data_dir, 'segments', dtype={'segment_id': 'string', 'flight_id': 'string'})
        mapping = read_table(data_dir, 'point_segment_map', dtype={'segment_id': 'string', 'flight_id': 'string'})
        updated_segments, merged = _updated_segments(segments, segment_ids, target_phase)
        ordered_ids = segments.loc[segments['segment_id'].isin(segment_ids)].sort_values('ordinal', kind='stable')['segment_id'].astype(str).tolist()
        source_phases = segments.set_index('segment_id').loc[ordered_ids, 'phase'].astype(str).tolist()
        mapping = mapping.copy()
        selected_mask = mapping['segment_id'].astype(str).isin(ordered_ids)
        mapping.loc[selected_mask, 'segment_id'] = str(merged['segment_id'])
        mapping.loc[selected_mask, 'phase'] = str(merged['phase'])
        _validate_tables(updated_segments, mapping)
        merges = _merge_event(data_dir, merged, ordered_ids, source_phases)
        _replace_tables(data_dir, {'segments': updated_segments, 'point_segment_map': mapping, 'manual_segment_merges': merges})
        _replace_progress_database(data_dir / 'segment_progress.sqlite', updated_segments)
        categories = _write_sequence_review(data_dir)
        summary_path = run_dir / 'summary.json'
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        summary['segments'] = int(len(updated_segments))
        summary['sequence_categories'] = categories
        summary['manual_segment_merges'] = int(len(merges))
        write_json(summary_path, summary)
        validation_path = run_dir / 'validation.json'
        validation = json.loads(validation_path.read_text(encoding='utf-8'))
        checks = validation.setdefault('checks', {})
        checks['segments'] = int(len(updated_segments))
        checks['manual_segment_merge_verified'] = True
        write_json(validation_path, validation)
        refresh_manifest_outputs(run_dir)
    return {'merged_segment_id': str(merged['segment_id']), 'phase': str(merged['phase']),
            'source_segment_ids': ordered_ids, 'segments': int(len(updated_segments)),
            'manual_segment_merges': int(len(merges))}
