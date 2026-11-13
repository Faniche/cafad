"""group_view.py | 2026-11-09 | 从已发布分组及其上游生成只读统计和三维预览 | Codex."""
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
from src.app.tables import read_table
from .history import _artifact_directory, _read_json_object


def group_result(root: Path, artifact: str) -> dict[str, object]:
    """约束依赖路径并缓存两个结果；不修改已发布产物和 manifest。"""
    directory = _artifact_directory(root, 'group', artifact)
    manifest = _read_json_object(directory / 'manifest.json') or {}
    if manifest.get('status') != 'SUCCEEDED':
        return {'groups': [], 'trajectories': [], 'note': '分组尚未成功发布。'}
    dependencies = manifest.get('dependencies', [])
    paths = [Path(item['manifest']) for item in dependencies if isinstance(item, dict) and 'manifest' in item]
    segmented = []
    for path in paths:
        path = (root / path).resolve()
        if (_read_json_object(path) or {}).get('module') == 'segment':
            candidate = _artifact_directory(root, 'segment', path.parent.relative_to(root).as_posix())
            segmented.append(candidate)
    if len(segmented) != 1:
        raise ValueError('无法唯一定位 group 的 segment 上游')
    source = segmented[0] / 'data'
    files = [_table_file(directory / 'data', 'routing_segments'), _table_file(source, 'point_segment_map'),
             _table_file(source, 'input_observations')]
    signature = tuple((file.stat().st_mtime_ns, file.stat().st_size) for file in files)
    return _build(directory, source, signature)


@lru_cache(maxsize=2)
def _build(directory: Path, source: Path, signature: tuple) -> dict[str, object]:
    """统计使用全部成员点；图每片段最多24点，观测分块读取限制额外内存。"""
    members = _read(directory / 'data', 'routing_segments').astype(str)
    mapping = _read(source, 'point_segment_map', columns=['record_id', 'segment_id', 'time_ns'],
                    dtype={'record_id': str, 'segment_id': str}).sort_values('time_ns', kind='stable')
    if mapping['record_id'].duplicated().any() or members['segment_id'].duplicated().any():
        raise ValueError('点映射或分组成员存在重复 ID')
    segment_groups = members.set_index('segment_id')['group_id'].to_dict()
    sampled = []
    for _, frame in mapping.groupby('segment_id', sort=False):
        sampled.append(frame.iloc[np.linspace(0, len(frame) - 1, min(24, len(frame)), dtype=int)])
    sample_map = pd.concat(sampled).set_index('record_id')['segment_id'] if sampled else pd.Series(dtype=str)
    point_map = mapping.set_index('record_id')['segment_id']
    partials, plotted = [], []
    fields = ['longitude_deg', 'latitude_deg', 'geometric_altitude_m']
    observations = _read(source, 'input_observations', columns=['record_id', 'time_ns', *fields], dtype={'record_id': str})
    for chunk in (observations.iloc[start:start + 100_000].copy() for start in range(0, len(observations), 100_000)):
        chunk['segment_id'] = chunk['record_id'].map(point_map)
        known = chunk.loc[chunk['segment_id'].isin(segment_groups)].copy()
        if not known.empty:
            partials.append(known.groupby('segment_id').agg(
                points=('record_id', 'size'), earliest=('time_ns', 'min'), latest=('time_ns', 'max'),
                min_lon=('longitude_deg', 'min'), max_lon=('longitude_deg', 'max'),
                min_lat=('latitude_deg', 'min'), max_lat=('latitude_deg', 'max'),
                min_alt=('geometric_altitude_m', 'min'), max_alt=('geometric_altitude_m', 'max')))
        visible = chunk.loc[chunk['record_id'].isin(sample_map.index), ['segment_id', 'time_ns', *fields]]
        plotted.append(visible)
    if not partials:
        raise ValueError('分组成员没有对应的有效观测')
    aggregate = {'points': 'sum', **{key: 'min' for key in ('earliest', 'min_lon', 'min_lat', 'min_alt')},
                 **{key: 'max' for key in ('latest', 'max_lon', 'max_lat', 'max_alt')}}
    stats = pd.concat(partials).groupby(level=0).agg(aggregate)
    if set(stats.index) != set(members['segment_id']):
        raise ValueError('分组成员与上游观测不完整')
    expected = mapping.loc[mapping['segment_id'].isin(segment_groups)].groupby('segment_id').size()
    if not stats['points'].sort_index().eq(expected.sort_index()).all():
        raise ValueError('分组成员有观测缺失，不能生成不完整范围统计')
    stats['group_id'] = stats.index.map(segment_groups)
    groups = []
    for group_id, rows in stats.groupby('group_id', sort=True):
        detail = members.loc[members['group_id'].eq(group_id)]
        groups.append({'group_id': group_id, 'segments': len(rows), 'longest': int(rows['points'].max()),
                       'shortest': int(rows['points'].min()),
                       'earliest': pd.Timestamp(int(rows['earliest'].min()), unit='ns', tz='UTC').isoformat(),
                       'latest': pd.Timestamp(int(rows['latest'].max()), unit='ns', tz='UTC').isoformat(),
                       **{name: float(rows[name].min()) for name in ('min_lon', 'min_lat', 'min_alt')},
                       **{name: float(rows[name].max()) for name in ('max_lon', 'max_lat', 'max_alt')},
                       'members': [{'segment_id': sid, 'points': int(rows.loc[sid, 'points']),
                                    'flight_id': detail.loc[detail['segment_id'].eq(sid), 'flight_id'].iloc[0]}
                                   for sid in rows.index]})
    trajectories = [{'segment_id': sid, 'group_id': segment_groups.get(sid),
                     'points': frame[fields].round(6).to_numpy().tolist()}
                    for sid, frame in pd.concat(plotted).sort_values('time_ns', kind='stable').groupby('segment_id', sort=False)]
    return {'groups': groups, 'trajectories': trajectories,
            'note': '统计为建组正常训练成员的全部观测；时间为 UTC，高度为几何高度（米）。图包含该 segment 产物中全部片段，每片段最多24个展示点；灰色为未参与建组的片段。范围框是经纬高外包框，允许重叠，不代表互斥空域；各轴独立缩放。'}


def _table_file(directory: Path, name: str) -> Path:
    """新 Parquet 优先；只读兼容旧已发布 CSV，避免历史结果无法查看。"""
    parquet = directory / f'{name}.parquet'
    return parquet if parquet.is_file() else directory / f'{name}.csv'


def _read(directory: Path, name: str, **kwargs: object) -> pd.DataFrame:
    """读取当前 Parquet 或旧历史 CSV；正式新产物只写 Parquet。"""
    if (directory / f'{name}.parquet').is_file():
        return read_table(directory, name, **kwargs)
    csv_kwargs = dict(kwargs)
    if 'columns' in csv_kwargs:
        csv_kwargs['usecols'] = csv_kwargs.pop('columns')
    return pd.read_csv(directory / f'{name}.csv', **csv_kwargs)
