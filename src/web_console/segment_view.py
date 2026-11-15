"""segment_view.py | 2026-11-15 | 硬时空切分运行的实时表格展示 | Codex."""
from __future__ import annotations

from pathlib import Path
import sqlite3

from .history import _artifact_directory


_DISPLAY_COLUMNS = ('flight_id', 'icao', 'call', 'cell_key', 'height_band', 'time_bucket', 'longitude_cell', 'latitude_cell',
                    'points', 'start_time', 'end_time', 'duration_seconds', 'start_longitude_deg', 'start_latitude_deg',
                    'end_longitude_deg', 'end_latitude_deg', 'span_km', 'start_altitude_m', 'end_altitude_m')


def segment_result(project_root: Path, artifact: str) -> dict[str, object]:
    """读取父进程串行写入的硬分段进度库；运行中可安全每秒轮询。"""
    directory = _artifact_directory(project_root, 'segment', artifact)
    database_path = directory / 'data' / 'segment_progress.sqlite'
    if not database_path.is_file():
        return {'rows': [], 'columns': list(_DISPLAY_COLUMNS), 'available': False}
    database = sqlite3.connect(f'{database_path.resolve().as_uri()}?mode=ro', uri=True)
    try:
        database.execute('PRAGMA busy_timeout=1000')
        records = database.execute(
            'SELECT segment_id, flight_id, icao, call, cell_key, height_band, time_bucket, longitude_cell, latitude_cell, '
            'points, start_time, end_time, duration_seconds, start_longitude_deg, start_latitude_deg, '
            'end_longitude_deg, end_latitude_deg, span_km, start_altitude_m, end_altitude_m FROM segments '
            'ORDER BY icao, call, start_time, segment_id'
        ).fetchall()
    finally:
        database.close()
    columns = ('segment_id', *_DISPLAY_COLUMNS)
    return {'rows': [dict(zip(columns, row, strict=True)) for row in records],
            'columns': list(_DISPLAY_COLUMNS), 'available': True}
