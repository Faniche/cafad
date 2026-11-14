"""segment_view.py | 2026-11-13 | 阶段识别运行的展示与受控人工标签修订 | Codex."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

import duckdb
import pandas as pd

from src.app.artifacts import refresh_manifest_outputs, write_json
from src.app.tables import read_table, table_path, write_table
from src.modules.segment.api import write_sequence_review
from .history import _artifact_directory
from .history import dependent_history_artifacts
from .segment_plot import invalidate_segment_flight_svgs


_DISPLAY_COLUMNS = ('flight_id', 'icao', 'call', 'phase', 'points', 'start_time', 'end_time', 'duration_seconds',
                    'start_longitude_deg', 'start_latitude_deg', 'end_longitude_deg', 'end_latitude_deg',
                    'span_km', 'start_altitude_m', 'end_altitude_m')
_SUN_PHASES = ('GND', 'CL', 'DE', 'CR', 'LVL', 'NA')


def segment_result(project_root: Path, artifact: str) -> dict[str, object]:
    """读取指定 segment 运行的阶段行及可编辑状态；运行中仅返回已提交数据。"""
    directory = _artifact_directory(project_root, 'segment', artifact)
    database_path = directory / 'data' / 'segment_progress.sqlite'
    state = _edit_state(project_root, artifact)
    if not database_path.is_file():
        return {'rows': [], 'columns': list(_DISPLAY_COLUMNS), 'available': False, **state}
    database = sqlite3.connect(f'{database_path.resolve().as_uri()}?mode=ro', uri=True)
    try:
        database.execute('PRAGMA busy_timeout=1000')
        records = database.execute(
            'SELECT segment_id, flight_id, icao, call, phase, points, start_time, end_time, duration_seconds, '
            'start_longitude_deg, start_latitude_deg, end_longitude_deg, end_latitude_deg, '
            'span_km, start_altitude_m, end_altitude_m FROM segments '
            'ORDER BY icao, call, start_time, segment_id'
        ).fetchall()
    finally:
        database.close()
    columns = ('segment_id', *_DISPLAY_COLUMNS)
    rows = [dict(zip(columns, row, strict=True)) for row in records]
    for row in rows:
        row['phase_source'] = 'sun'
    return {'rows': rows,
            'columns': list(_DISPLAY_COLUMNS), 'available': True, **state}


def update_segment_phases(project_root: Path, artifact: str, segment_ids: list[str], phase: str) -> dict[str, object]:
    """同步修订片段标签、逐点标签、序列审计、进度库及 manifest，拒绝有下游依赖的运行。"""
    state = _edit_state(project_root, artifact)
    if not state['editable']:
        raise ValueError(str(state['edit_reason']))
    selected = list(dict.fromkeys(identifier for identifier in segment_ids if isinstance(identifier, str) and identifier))
    if not selected or len(selected) != len(segment_ids) or len(selected) > 1000:
        raise ValueError('请选择 1 至 1000 个不重复的阶段片段')
    if phase not in state['allowed_phases']:
        raise ValueError('目标阶段不属于本次分段器允许的标签')
    directory = _artifact_directory(project_root, 'segment', artifact)
    data_dir = directory / 'data'
    segments = read_table(data_dir, 'segments', dtype={'segment_id': 'string', 'flight_id': 'string'})
    indexed = segments.set_index('segment_id')
    if not indexed.index.is_unique:
        raise RuntimeError('segments.parquet 含重复 segment_id，拒绝人工修改')
    missing = set(selected) - set(indexed.index.astype(str))
    if missing:
        raise ValueError(f'选择中含不存在的 segment_id：{next(iter(missing))}')
    previous = indexed.loc[selected, 'phase'].astype(str)
    changed_ids = [identifier for identifier in selected if previous.loc[identifier] != phase]
    if not changed_ids:
        return {'changed_segments': 0, 'phase': phase}
    changes = pd.DataFrame({'segment_id': changed_ids, 'phase': phase})
    changed_flights = set(indexed.loc[changed_ids, 'flight_id'].astype(str))
    segments.loc[segments['segment_id'].isin(changed_ids), 'phase'] = phase
    _rewrite_point_phases(table_path(data_dir, 'point_segment_map'), changes)
    write_table(segments, data_dir, 'segments')
    _update_progress_database(data_dir / 'segment_progress.sqlite', changes)
    audit = _append_override_audit(data_dir, previous.loc[changed_ids], phase)
    write_sequence_review(data_dir)
    _update_summary(directory / 'summary.json', len(audit))
    invalidate_segment_flight_svgs(directory, changed_flights)
    refresh_manifest_outputs(directory / 'manifest.json')
    return {'changed_segments': len(changed_ids), 'phase': phase, 'override_count': int(len(audit))}


def _edit_state(project_root: Path, artifact: str) -> dict[str, object]:
    """仅允许修改已发布、无下游依赖且当前无正式运行锁的 segment 结果。"""
    directory = _artifact_directory(project_root, 'segment', artifact)
    manifest = _read_json(directory / 'manifest.json')
    values: dict[str, object] = {'allowed_phases': _allowed_phases(directory)}
    if manifest is None or manifest.get('status') != 'SUCCEEDED':
        return {**values, 'editable': False, 'edit_reason': '只有已成功发布的 segment 运行可以修改标签'}
    if _segment_method(directory) != 'sun':
        return {**values, 'editable': False, 'edit_reason': '历史 DLR 分段已废弃；请重新运行 Sun 分段'}
    if (project_root / '.cafad-run.lock').exists():
        return {**values, 'editable': False, 'edit_reason': '当前有正式模块运行，不能修改阶段标签'}
    if len(dependent_history_artifacts(project_root, 'segment', artifact)) > 1:
        return {**values, 'editable': False, 'edit_reason': '该分段结果已有下游产物；请先删除关联下游运行后再修改'}
    return {**values, 'editable': True, 'edit_reason': ''}


def _allowed_phases(directory: Path) -> list[str]:
    """纯 Sun 分段允许的人工修订标签。"""
    return list(_SUN_PHASES)


def _segment_method(directory: Path) -> str:
    config = _read_json(directory / 'resolved_config.json') or {}
    module = config.get('module')
    return str(module.get('method', 'sun')) if isinstance(module, dict) else 'sun'


def _rewrite_point_phases(path: Path, changes: pd.DataFrame) -> None:
    """以 DuckDB 流式重写大规模逐点映射，仅替换被选片段的 phase 列。"""
    temporary = path.with_suffix('.parquet.tmp')
    escaped = str(temporary.resolve()).replace("'", "''")
    with duckdb.connect(':memory:') as database:
        database.register('phase_changes', changes)
        database.execute(
            f"COPY (SELECT source.* REPLACE (COALESCE(phase_changes.phase, source.phase) AS phase) "
            f"FROM read_parquet(?) AS source LEFT JOIN phase_changes USING (segment_id)) "
            f"TO '{escaped}' (FORMAT PARQUET, COMPRESSION ZSTD)", [str(path)],
        )
    temporary.replace(path)


def _update_progress_database(path: Path, changes: pd.DataFrame) -> None:
    """让网页实时表与正式 Parquet 使用完全相同的人工修订标签。"""
    if not path.is_file():
        raise FileNotFoundError(f'缺少 segment 进度数据库：{path}')
    database = sqlite3.connect(path)
    try:
        database.executemany('UPDATE segments SET phase = ? WHERE segment_id = ?',
                             [(str(row.phase), str(row.segment_id)) for row in changes.itertuples(index=False)])
        if database.total_changes != len(changes):
            raise RuntimeError('进度数据库未能覆盖全部待修改片段')
        database.commit()
        database.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    finally:
        database.close()


def _append_override_audit(data_dir: Path, previous: pd.Series, phase: str) -> pd.DataFrame:
    """保存每个手工标签变更的旧值、新值、时间和来源，保持可追溯。"""
    path = table_path(data_dir, 'phase_label_overrides')
    existing = read_table(data_dir, 'phase_label_overrides') if path.is_file() else pd.DataFrame()
    rows = pd.DataFrame({
        'segment_id': previous.index.astype(str), 'previous_phase': previous.astype(str).to_numpy(),
        'phase': phase, 'changed_at': datetime.now(timezone.utc).isoformat(), 'source': 'web_console',
    })
    combined = pd.concat((existing, rows), ignore_index=True)
    write_table(combined, data_dir, 'phase_label_overrides')
    return combined


def _update_summary(path: Path, override_count: int) -> None:
    """在摘要中标记人工修订数量，避免历史结果被误当作纯自动分段。"""
    summary = _read_json(path)
    if summary is None:
        raise ValueError('segment 摘要不存在或损坏，拒绝修改已发布结果')
    summary['manual_phase_override_count'] = override_count
    summary['manual_phase_override_updated_at'] = datetime.now(timezone.utc).isoformat()
    write_json(path, summary)


def _read_json(path: Path) -> dict[str, object] | None:
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None
