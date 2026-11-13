"""segment_view.py | 2026-11-12 | 阶段识别运行的 SQLite 实时进度读取 | Codex."""
from __future__ import annotations

from pathlib import Path
import sqlite3

from .history import _artifact_directory


_COLUMNS = ('segment_id', 'flight_id', 'icao', 'call', 'phase', 'points', 'start_time', 'end_time', 'duration_seconds',
            'start_longitude_deg', 'start_latitude_deg', 'end_longitude_deg', 'end_latitude_deg',
            'span_km', 'start_altitude_m', 'end_altitude_m')


def segment_result(project_root: Path, artifact: str) -> dict[str, object]:
    """读取指定 segment 运行的已完成片段；运行中仅返回主进程已提交的数据。"""
    directory = _artifact_directory(project_root, 'segment', artifact)
    database_path = directory / 'data' / 'segment_progress.sqlite'
    if not database_path.is_file():
        return {'rows': [], 'columns': list(_COLUMNS), 'available': False}
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
    return {'rows': [dict(zip(_COLUMNS, row, strict=True)) for row in records],
            'columns': list(_COLUMNS), 'available': True}
