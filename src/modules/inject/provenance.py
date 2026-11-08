"""provenance.py | 2026-11-06 | 事件、逐点标签与影响范围账本 | Codex."""
from __future__ import annotations

import json
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from .corruption import NUMERIC_COLUMNS


EVENT_COLUMNS = (
    'event_id', 'flight_id', 'split', 'event_type', 'semantics', 'seed',
    'support_scope',
    'index_basis', 'interval_convention',
    'start_index', 'end_index_exclusive', 'start_record_id', 'end_record_id_exclusive',
    'start_time_ns', 'end_time_ns_exclusive', 'last_affected_time_ns',
    'affected_fields_json',
    'planned_positions_json', 'planned_record_ids_json', 'n_target_points',
    'changed_positions_json', 'changed_record_ids_json', 'n_changed_points',
    'donor_flight_id', 'donor_start_index', 'donor_end_index_exclusive',
    'donor_start_record_id', 'donor_end_record_id_exclusive',
    'parameters_json',
)
LABEL_COLUMNS = (
    'record_id', 'flight_id', 'split', 'injected_label', 'is_value_changed',
    'event_ids_json', 'anomaly_types_json', 'changed_fields_json',
)
IMPACT_COLUMNS = (
    'event_id', 'flight_id', 'raw_start_index', 'raw_end_index_exclusive',
    'derived_feature_rule', 'derived_feature_start_index', 'derived_feature_end_index_exclusive',
    'detection_window_rule', 'detection_window_start_index', 'detection_window_end_index_exclusive',
)


def final_labels(before: pd.DataFrame, after: pd.DataFrame) -> np.ndarray:
    """以最终六个物理字段的逐记录差异生成二值标签，绝不使用计划位置代替。"""
    original = before.loc[:, NUMERIC_COLUMNS].to_numpy(dtype=np.float64)
    changed = after.loc[:, NUMERIC_COLUMNS].to_numpy(dtype=np.float64)
    if original.shape != changed.shape or not np.isfinite(changed).all():
        raise ValueError('注入前后物理字段形状不一致或包含非有限值')
    return np.any(original != changed, axis=1).astype(np.int8)


def changed_fields(before: pd.DataFrame, after: pd.DataFrame) -> dict[int, tuple[str, ...]]:
    """计算最终逐点改变字段，供事件与记录标签共享同一事实来源。"""
    original = before.loc[:, NUMERIC_COLUMNS].to_numpy(dtype=np.float64)
    changed = after.loc[:, NUMERIC_COLUMNS].to_numpy(dtype=np.float64)
    matrix = original != changed
    return {int(position): tuple(np.asarray(NUMERIC_COLUMNS)[matrix[position]].tolist())
            for position in np.flatnonzero(np.any(matrix, axis=1))}


def event_rows(events: Iterable[Mapping[str, object]], after: pd.DataFrame, *, flight_id: str,
               split: str, seed: int) -> list[dict[str, object]]:
    """把一次实际事件写成精确的计划区间、实际改动区间及供体谱系。"""
    rows: list[dict[str, object]] = []
    for ordinal, event in enumerate(events):
        planned = np.unique(np.asarray(event['planned_positions'], dtype=np.int64))
        changed = np.unique(np.asarray(event['changed_positions'], dtype=np.int64))
        if not len(planned) or not len(changed):
            raise ValueError('事件必须同时具有计划支持集与实际改变记录')
        event_id = f'{flight_id}:event_{ordinal:04d}'
        if isinstance(event, dict):
            event['event_id'] = event_id
        start, end = int(planned[0]), int(planned[-1] + 1)
        donor = event.get('donor', {})
        if not isinstance(donor, Mapping):
            raise ValueError('供体信息必须是对象')
        rows.append({
            'event_id': event_id, 'flight_id': flight_id, 'split': split,
            'event_type': str(event['event_type']), 'semantics': str(event['semantics']), 'seed': str(seed),
            'support_scope': str(event['support_scope']),
            'index_basis': 'zero_based_positions_within_source_flight', 'interval_convention': '[start,end)',
            'start_index': start, 'end_index_exclusive': end,
            'start_record_id': str(after.iloc[start]['record_id']),
            'end_record_id_exclusive': str(after.iloc[end]['record_id']) if end < len(after) else None,
            'start_time_ns': int(after.iloc[start]['time_ns']),
            'end_time_ns_exclusive': int(after.iloc[end]['time_ns']) if end < len(after) else None,
            'last_affected_time_ns': int(after.iloc[int(changed[-1])]['time_ns']),
            'affected_fields_json': json.dumps(list(event['affected_fields']), ensure_ascii=False),
            'planned_positions_json': json.dumps(planned.tolist()),
            'planned_record_ids_json': json.dumps(after.iloc[planned]['record_id'].astype(str).tolist(), ensure_ascii=False),
            'n_target_points': int(len(planned)),
            'changed_positions_json': json.dumps(changed.tolist()),
            'changed_record_ids_json': json.dumps(after.iloc[changed]['record_id'].astype(str).tolist(), ensure_ascii=False),
            'n_changed_points': int(len(changed)),
            'donor_flight_id': donor.get('flight_id'), 'donor_start_index': donor.get('start_index'),
            'donor_end_index_exclusive': donor.get('end_index_exclusive'),
            'donor_start_record_id': donor.get('start_record_id'), 'donor_end_record_id_exclusive': donor.get('end_record_id_exclusive'),
            'parameters_json': json.dumps(event['parameters'], ensure_ascii=False, sort_keys=True),
        })
    return rows


def record_labels(before: pd.DataFrame, after: pd.DataFrame, events: Iterable[Mapping[str, object]], *,
                  labels: np.ndarray | None = None) -> pd.DataFrame:
    """构造逐点标签；正常记录向量化处理，仅为实际改变点写事件账本字段。"""
    fields = changed_fields(before, after)
    event_list = list(events)
    event_at_position: dict[int, tuple[str, str]] = {}
    for event in event_list:
        event_id = event.get('event_id')
        if not isinstance(event_id, str):
            raise ValueError('构造记录标签前必须为每个事件分配 event_id')
        for position in np.asarray(event['changed_positions'], dtype=np.int64):
            index = int(position)
            if index not in fields:
                raise ValueError('事件账本包含未实际改变的记录')
            if index in event_at_position:
                raise ValueError('默认协议禁止不同事件的实际改动记录重叠')
            event_at_position[index] = event_id, str(event['event_type'])
    if set(event_at_position) != set(fields):
        raise ValueError('逐点实际改变记录必须恰有一个事件归属')
    expected_labels = final_labels(before, after) if labels is None else np.asarray(labels, dtype=np.int8)
    if len(expected_labels) != len(after) or not np.isin(expected_labels, (0, 1)).all():
        raise ValueError('传入的逐点标签长度或取值无效')
    length = len(after)
    event_ids_json = np.full(length, '[]', dtype=object)
    anomaly_types_json = np.full(length, '[]', dtype=object)
    changed_fields_json = np.full(length, '[]', dtype=object)
    for position, (event_id, event_type) in event_at_position.items():
        event_ids_json[position] = json.dumps([event_id])
        anomaly_types_json[position] = json.dumps([event_type])
        changed_fields_json[position] = json.dumps(list(fields[position]), ensure_ascii=False)
    return pd.DataFrame({
        'record_id': after['record_id'].astype(str).to_numpy(),
        'flight_id': after['flight_id'].astype(str).to_numpy(),
        'split': after['split'].astype(str).to_numpy(),
        'injected_label': expected_labels,
        'is_value_changed': expected_labels,
        'event_ids_json': event_ids_json,
        'anomaly_types_json': anomaly_types_json,
        'changed_fields_json': changed_fields_json,
    }, columns=LABEL_COLUMNS)


def normal_record_labels(after: pd.DataFrame) -> pd.DataFrame:
    """为未修改的正常航班直接生成全零标签账本；发布前全局核验仍会比较物理字段。"""
    length = len(after)
    empty_json = np.full(length, '[]', dtype=object)
    zero_labels = np.zeros(length, dtype=np.int8)
    return pd.DataFrame({
        'record_id': after['record_id'].astype(str).to_numpy(),
        'flight_id': after['flight_id'].astype(str).to_numpy(),
        'split': after['split'].astype(str).to_numpy(),
        'injected_label': zero_labels,
        'is_value_changed': zero_labels,
        'event_ids_json': empty_json,
        'anomaly_types_json': empty_json.copy(),
        'changed_fields_json': empty_json.copy(),
    }, columns=LABEL_COLUMNS)


def impact_rows(events: Iterable[Mapping[str, object]], *, flight_id: str, flight_points: int) -> list[dict[str, object]]:
    """独立保存原始事件支持集、保守一阶派生影响域与尚未实现的检测窗口域。"""
    rows: list[dict[str, object]] = []
    for event in events:
        planned = np.asarray(event['planned_positions'], dtype=np.int64)
        changed = np.asarray(event['changed_positions'], dtype=np.int64)
        rows.append({
            'event_id': str(event['event_id']), 'flight_id': flight_id,
            'raw_start_index': int(planned.min()), 'raw_end_index_exclusive': int(planned.max() + 1),
            'derived_feature_rule': 'first_order_feature_may_use_current_and_previous_record',
            'derived_feature_start_index': int(changed.min()),
            'derived_feature_end_index_exclusive': min(flight_points, int(changed.max() + 2)),
            'detection_window_rule': 'not_materialized_until_detect_module',
            'detection_window_start_index': None, 'detection_window_end_index_exclusive': None,
        })
    return rows


def validate_provenance(before: pd.DataFrame, after: pd.DataFrame, labels: pd.DataFrame,
                        events: pd.DataFrame, impacts: pd.DataFrame) -> None:
    """核验身份、最终差异、事件支持集、标签和影响范围之间的守恒关系。"""
    before_ids = before['record_id'].astype(str).to_numpy()
    after_ids = after['record_id'].astype(str).to_numpy()
    identical_order = np.array_equal(before_ids, after_ids)
    if before['record_id'].duplicated().any() or after['record_id'].duplicated().any() or not identical_order:
        mismatch = np.flatnonzero(before_ids != after_ids) if len(before_ids) == len(after_ids) else np.asarray([], dtype=np.int64)
        location = int(mismatch[0]) if len(mismatch) else None
        detail = (f'；first_mismatch={location}；before={before_ids[location]!r}；after={after_ids[location]!r}'
                  if location is not None else f'；before_rows={len(before_ids)}；after_rows={len(after_ids)}')
        raise ValueError(f'注入不得新增、删除、重排或替换稳定 record_id{detail}')
    expected = final_labels(before, after)
    actual = labels['injected_label'].to_numpy(dtype=np.int8)
    if not np.array_equal(expected, actual) or not np.array_equal(actual, labels['is_value_changed'].to_numpy(dtype=np.int8)):
        raise ValueError('记录标签不等于最终前后物理字段差异')
    if (labels['record_id'].duplicated().any()
            or not np.array_equal(labels['record_id'].astype(str).to_numpy(), after['record_id'].astype(str).to_numpy())):
        raise ValueError('记录标签未与最终观测一一对应')
    if events['event_id'].duplicated().any() or (events['n_changed_points'] <= 0).any():
        raise ValueError('注入事件必须唯一且包含实际改动记录')
    changed_ids: set[str] = set()
    for event in events.itertuples(index=False):
        planned = json.loads(event.planned_positions_json)
        changed = json.loads(event.changed_positions_json)
        planned_ids = json.loads(event.planned_record_ids_json)
        actual_ids = json.loads(event.changed_record_ids_json)
        if (not planned or not changed or len(planned) != int(event.n_target_points)
                or len(changed) != int(event.n_changed_points) or planned != sorted(set(planned))
                or changed != sorted(set(changed) or []) or int(event.end_index_exclusive) - int(event.start_index) != len(planned)):
            raise ValueError('事件的半开区间或支持集长度不一致')
        flight_rows = after.loc[after['flight_id'].eq(event.flight_id)].sort_values('time_ns', kind='stable').reset_index(drop=True)
        if max(planned) >= len(flight_rows) or max(changed) >= len(flight_rows):
            raise ValueError('事件位置超出所属来源航班范围')
        if (flight_rows.iloc[planned]['record_id'].astype(str).tolist() != planned_ids
                or flight_rows.iloc[changed]['record_id'].astype(str).tolist() != actual_ids):
            raise ValueError('事件位置与稳定记录 ID 不一致')
        if set(actual_ids) & changed_ids:
            raise ValueError('默认协议不允许事件实际改动记录重叠')
        changed_ids.update(actual_ids)
    expected_ids = set(after.loc[expected.astype(bool), 'record_id'].astype(str))
    if changed_ids != expected_ids:
        raise ValueError('事件实际改变记录与逐点标签不一致')
    for row in labels.itertuples(index=False):
        event_ids = json.loads(row.event_ids_json)
        fields = json.loads(row.changed_fields_json)
        if bool(row.injected_label) != bool(event_ids) or bool(row.injected_label) != bool(fields):
            raise ValueError('逐点事件、字段和二值标签不一致')
    if impacts['event_id'].duplicated().any() or set(impacts['event_id']) != set(events['event_id']):
        raise ValueError('派生影响账本未与事件表一一对应')
    if (impacts['derived_feature_end_index_exclusive'] <= impacts['derived_feature_start_index']).any():
        raise ValueError('派生特征影响区间无效')
