"""corruption.py | 2026-11-06 | 事件化 ADS-B 异常注入与协议校验 | Codex.

事件在整条合格航班内生成；注入后再对最终观测执行 Sun 阶段划分。报送失真
不会伪造运动学一致性，真实运动事件则同步重算速度和航向。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import pandas as pd
import yaml


EARTH_RADIUS_M = 6_371_000.0
NUMERIC_COLUMNS = ('longitude_deg', 'latitude_deg', 'geometric_altitude_m', 'barometric_altitude_m',
                   'heading_deg', 'speed_mps')
EVENT_TYPES = (
    'gaussian_point', 'reported_fixed_offset', 'reported_progressive_drift',
    'reported_local_perturbation', 'true_local_maneuver', 'motion_transfer',
)


@dataclass(frozen=True)
class CorruptionPlan:
    """主进程确定的事件类型、位置与可选供体窗口，可安全传给单航班工作进程。"""

    event_type: str
    positions: np.ndarray
    donor: dict[str, object] | None


def load_protocol(path: Path) -> dict[str, object]:
    """读取 schema_version=2 的显式事件协议，拒绝猜测强度或事件类型。"""
    try:
        text = path.read_text(encoding='utf-8')
    except OSError as exc:
        raise FileNotFoundError(f'找不到注入协议：{path}') from exc
    try:
        raw = json.loads(text) if path.suffix.lower() == '.json' else yaml.safe_load(text)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f'注入协议不是有效 JSON/YAML：{path}') from exc
    if not isinstance(raw, Mapping) or raw.get('schema_version') != 2:
        raise ValueError('注入协议必须是 schema_version=2 的对象')
    event_lengths = _integer_range(raw.get('event_length_points'), 'event_length_points', minimum=2)
    context_points = _integer(raw.get('context_points_each_side'), 'context_points_each_side', minimum=0)
    event_probabilities = _probabilities(raw.get('event_type_probabilities'), 'event_type_probabilities', EVENT_TYPES)
    gaussian = _mapping(raw.get('gaussian_point'), 'gaussian_point')
    fixed = _mapping(raw.get('reported_fixed_offset'), 'reported_fixed_offset')
    drift = _mapping(raw.get('reported_progressive_drift'), 'reported_progressive_drift')
    local = _mapping(raw.get('reported_local_perturbation'), 'reported_local_perturbation')
    maneuver = _mapping(raw.get('true_local_maneuver'), 'true_local_maneuver')
    transfer = _mapping(raw.get('motion_transfer'), 'motion_transfer')
    return {
        'schema_version': 2,
        'event_length_points': event_lengths,
        'context_points_each_side': context_points,
        'event_type_probabilities': event_probabilities,
        'gaussian_point': {
            'speed_probability': _fraction(gaussian.get('speed_probability'), 'gaussian_point.speed_probability'),
            'speed_std_fraction': _finite(gaussian.get('speed_std_fraction'), 'gaussian_point.speed_std_fraction', minimum=0),
            'speed_std_floor_mps': _finite(gaussian.get('speed_std_floor_mps'), 'gaussian_point.speed_std_floor_mps', minimum=0,
                                            positive=True),
            'speed_std_cap_mps': _finite(gaussian.get('speed_std_cap_mps'), 'gaussian_point.speed_std_cap_mps', minimum=0,
                                          positive=True),
            'altitude_std_fraction': _finite(gaussian.get('altitude_std_fraction'), 'gaussian_point.altitude_std_fraction', minimum=0),
            'altitude_std_floor_m': _finite(gaussian.get('altitude_std_floor_m'), 'gaussian_point.altitude_std_floor_m', minimum=0,
                                            positive=True),
            'altitude_std_cap_m': _finite(gaussian.get('altitude_std_cap_m'), 'gaussian_point.altitude_std_cap_m', minimum=0,
                                          positive=True),
        },
        'reported_fixed_offset': {
            'field_probabilities': _probabilities(fixed.get('field_probabilities'), 'reported_fixed_offset.field_probabilities',
                                                   ('lateral_position', 'altitude', 'speed')),
            'lateral_offset_m_range': _float_range(fixed.get('lateral_offset_m_range'), 'reported_fixed_offset.lateral_offset_m_range',
                                                    minimum=0, positive=True),
            'altitude_offset_m_range': _float_range(fixed.get('altitude_offset_m_range'), 'reported_fixed_offset.altitude_offset_m_range',
                                                     minimum=0, positive=True),
            'speed_offset_mps_range': _float_range(fixed.get('speed_offset_mps_range'), 'reported_fixed_offset.speed_offset_mps_range',
                                                    minimum=0, positive=True),
        },
        'reported_progressive_drift': {
            'field_probabilities': _probabilities(drift.get('field_probabilities'), 'reported_progressive_drift.field_probabilities',
                                                   ('altitude', 'speed')),
            'altitude_step_m_range': _float_range(drift.get('altitude_step_m_range'), 'reported_progressive_drift.altitude_step_m_range',
                                                   minimum=0, positive=True),
            'speed_step_mps_range': _float_range(drift.get('speed_step_mps_range'), 'reported_progressive_drift.speed_step_mps_range',
                                                  minimum=0, positive=True),
        },
        'reported_local_perturbation': _local_config(local, 'reported_local_perturbation', ('speed', 'lateral_position', 'altitude')),
        'true_local_maneuver': _local_config(maneuver, 'true_local_maneuver', ('lateral_position', 'altitude')),
        'motion_transfer': {
            'scale_range': _float_range(transfer.get('scale_range'), 'motion_transfer.scale_range', minimum=0, positive=True),
        },
    }


def minimum_required_points(protocol: Mapping[str, object]) -> int:
    """返回能容纳最长事件及两侧正常上下文的最低航班点数。"""
    lengths = protocol['event_length_points']
    assert isinstance(lengths, tuple)
    return int(lengths[1]) + 2 * int(protocol['context_points_each_side'])


def flight_seed(global_seed: int, flight_id: str) -> int:
    """从主种子和稳定来源航班 ID 派生独立、跨进程可复现的随机种子。"""
    digest = hashlib.sha256(f'{global_seed}|inject|{flight_id}'.encode('utf-8')).digest()
    return int.from_bytes(digest[:8], byteorder='big', signed=False)


def corrupt_flight(frame: pd.DataFrame, donor_pool: pd.DataFrame, protocol: Mapping[str, object],
                  rng: np.random.Generator) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """在一条合格航班内生成一个互不重叠、可追溯的主异常事件。"""
    target_flight_id = str(frame['flight_id'].iloc[0])
    target_split = str(frame['split'].iloc[0])
    plan = prepare_corruption_plan(
        frame, protocol, rng,
        has_donor=lambda length: _has_donor(donor_pool, target_flight_id=target_flight_id,
                                             target_split=target_split, length=length),
        select_donor=lambda length: _select_donor(donor_pool, target_flight_id=target_flight_id,
                                                   target_split=target_split, length=length, rng=rng),
    )
    return apply_corruption_plan(frame, plan, protocol, rng)


def prepare_corruption_plan(frame: pd.DataFrame, protocol: Mapping[str, object], rng: np.random.Generator, *,
                            has_donor: Callable[[int], bool],
                            select_donor: Callable[[int], dict[str, object]]) -> CorruptionPlan:
    """在主进程确定一个事件计划，避免工作进程复制完整候选供体库。"""
    _validate_flight_frame(frame)
    required = minimum_required_points(protocol)
    if len(frame) < required:
        raise ValueError(f'航班少于所需的 {required} 个观测点')
    lengths = protocol['event_length_points']
    assert isinstance(lengths, tuple)
    length = int(rng.integers(int(lengths[0]), int(lengths[1]) + 1))
    positions = _event_positions(len(frame), length, int(protocol['context_points_each_side']), rng)
    event_type = _select_event_type(length, protocol, rng, has_donor=has_donor(length))
    donor = select_donor(len(positions)) if event_type == 'motion_transfer' else None
    return CorruptionPlan(event_type=event_type, positions=positions, donor=donor)


def apply_corruption_plan(frame: pd.DataFrame, plan: CorruptionPlan, protocol: Mapping[str, object],
                          rng: np.random.Generator) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """只使用目标航班、已选事件与最多一个供体窗口完成实际字段修改。"""
    _validate_flight_frame(frame)
    result = frame.copy()
    event_type = plan.event_type
    positions = plan.positions.copy()
    if event_type not in EVENT_TYPES:
        raise ValueError(f'未知的事件类型：{event_type}')
    before = result.loc[:, NUMERIC_COLUMNS].to_numpy(dtype=np.float64, copy=True)
    donor = plan.donor
    if event_type == 'gaussian_point':
        positions = positions[:1]
        parameters, semantics = _apply_gaussian(result, positions, protocol['gaussian_point'], rng)
    elif event_type == 'reported_fixed_offset':
        parameters, semantics = _apply_fixed_offset(result, positions, protocol['reported_fixed_offset'], rng)
    elif event_type == 'reported_progressive_drift':
        parameters, semantics = _apply_progressive_drift(result, positions, protocol['reported_progressive_drift'], rng)
    elif event_type == 'reported_local_perturbation':
        parameters, semantics = _apply_local_perturbation(result, positions, protocol['reported_local_perturbation'], rng,
                                                           recompute_motion=False)
    elif event_type == 'true_local_maneuver':
        parameters, semantics = _apply_local_perturbation(result, positions, protocol['true_local_maneuver'], rng,
                                                           recompute_motion=True)
    else:
        if donor is None:
            raise ValueError('真实运动片段迁移缺少已选供体窗口')
        parameters, semantics = _apply_motion_transfer(result, positions, donor['rows'], protocol['motion_transfer'], rng)
    changed, changed_fields = _changed_positions(before, result)
    _validate_flight_frame(result)
    if not len(changed):
        raise ValueError(f'{event_type} 没有产生实际字段变化')
    event: dict[str, object] = {
        'event_type': event_type,
        'semantics': semantics,
        'support_scope': 'whole_flight',
        'planned_positions': positions,
        'changed_positions': changed,
        'changed_fields_by_position': changed_fields,
        'affected_fields': tuple(sorted({field for fields in changed_fields.values() for field in fields})),
        'parameters': parameters,
    }
    if donor is not None:
        event['donor'] = {key: value for key, value in donor.items() if key != 'rows'}
    return result, [event]


def _event_positions(flight_points: int, length: int, context: int, rng: np.random.Generator) -> np.ndarray:
    """从航班内部采样左闭右开的支持区间，保留两侧未修改上下文。"""
    first = context
    last = flight_points - context - length
    if last < first:
        raise ValueError('航班无法同时容纳事件与两侧正常上下文')
    start = int(rng.integers(first, last + 1))
    return np.arange(start, start + length, dtype=np.int64)


def _select_event_type(length: int, protocol: Mapping[str, object], rng: np.random.Generator, *, has_donor: bool) -> str:
    """按协议概率选择当前整条航班真正可实现的事件类型。"""
    probabilities = protocol['event_type_probabilities']
    assert isinstance(probabilities, Mapping)
    names = [name for name in EVENT_TYPES if float(probabilities[name]) > 0]
    if 'motion_transfer' in names and not has_donor:
        names.remove('motion_transfer')
    if not names:
        raise ValueError('没有可用于当前片段的异常事件类型')
    weights = np.asarray([float(probabilities[name]) for name in names], dtype=np.float64)
    return str(rng.choice(np.asarray(names, dtype=object), p=weights / weights.sum()))


def _has_donor(donor_pool: pd.DataFrame, *, target_flight_id: str, target_split: str, length: int) -> bool:
    """检查同分区、非同源的供体航班是否具有足够长的连续观测。"""
    candidates = donor_pool.loc[(donor_pool['split'].eq(target_split))
                                & (~donor_pool['flight_id'].eq(target_flight_id))].copy()
    return bool((candidates.groupby('flight_id', sort=False).size() >= length).any())


def _select_donor(donor_pool: pd.DataFrame, *, target_flight_id: str, target_split: str, length: int,
                  rng: np.random.Generator) -> dict[str, object]:
    """从同分区异源航班抽取真实连续运动窗口，并保存精确供体区间。"""
    candidates = donor_pool.loc[(donor_pool['split'].eq(target_split))
                                & (~donor_pool['flight_id'].eq(target_flight_id))].copy()
    grouped = [(str(flight_id), group.sort_values('time_ns', kind='stable').reset_index(drop=True))
               for flight_id, group in candidates.groupby('flight_id', sort=False, observed=True)
               if len(group) >= length]
    if not grouped:
        raise ValueError('不存在同分区且足够长的异源真实运动供体')
    donor_flight_id, source = grouped[int(rng.integers(len(grouped)))]
    start = int(rng.integers(0, len(source) - length + 1))
    rows = source.iloc[start:start + length].copy().reset_index(drop=True)
    return {
        'flight_id': donor_flight_id, 'start_index': start, 'end_index_exclusive': start + length,
        'start_record_id': str(rows['record_id'].iloc[0]), 'end_record_id_exclusive': (
            str(source['record_id'].iloc[start + length]) if start + length < len(source) else None
        ),
        'rows': rows,
    }


def _apply_gaussian(frame: pd.DataFrame, positions: np.ndarray, config: object,
                    rng: np.random.Generator) -> tuple[dict[str, object], str]:
    """生成单点报送高斯失真；非零标准差下确保裁剪后确有改动。"""
    assert isinstance(config, Mapping)
    index = int(positions[0])
    if rng.random() < float(config['speed_probability']):
        value = float(frame.at[index, 'speed_mps'])
        scale = min(max(abs(value) * float(config['speed_std_fraction']), float(config['speed_std_floor_mps'])),
                    float(config['speed_std_cap_mps']))
        delta = _nonzero_normal(scale, rng)
        changed = max(0.0, value + delta)
        if changed == value:
            changed = value + abs(delta)
        frame.at[index, 'speed_mps'] = changed
        return {'field': 'speed_mps', 'std_mps': scale, 'sampled_delta_mps': changed - value}, 'reported_data_distortion'
    value = float(frame.at[index, 'geometric_altitude_m'])
    scale = min(max(abs(value) * float(config['altitude_std_fraction']), float(config['altitude_std_floor_m'])),
                float(config['altitude_std_cap_m']))
    delta = _nonzero_normal(scale, rng)
    frame.at[index, 'geometric_altitude_m'] = value + delta
    frame.at[index, 'barometric_altitude_m'] = float(frame.at[index, 'barometric_altitude_m']) + delta
    return {'field': 'altitude', 'std_m': scale, 'sampled_delta_m': delta}, 'reported_data_distortion'


def _apply_fixed_offset(frame: pd.DataFrame, positions: np.ndarray, config: object,
                        rng: np.random.Generator) -> tuple[dict[str, object], str]:
    """施加固定报送偏移；位置报送失真故意不改写速度／航向。"""
    assert isinstance(config, Mapping)
    kind = _choose_probability(config['field_probabilities'], rng)
    if kind == 'lateral_position':
        amplitude = _signed_uniform(config['lateral_offset_m_range'], rng)
        direction = float(rng.uniform(0.0, 2.0 * np.pi))
        _offset_coordinates(frame, positions, np.full(len(positions), abs(amplitude)), direction + (np.pi if amplitude < 0 else 0.0))
        return {'field': kind, 'lateral_offset_m': amplitude, 'direction_radians': direction}, 'reported_data_distortion'
    if kind == 'altitude':
        delta = _signed_uniform(config['altitude_offset_m_range'], rng)
        frame.loc[positions, 'geometric_altitude_m'] += delta
        frame.loc[positions, 'barometric_altitude_m'] += delta
        return {'field': kind, 'offset_m': delta}, 'reported_data_distortion'
    delta = _signed_uniform(config['speed_offset_mps_range'], rng)
    values = frame.loc[positions, 'speed_mps'].to_numpy(dtype=np.float64)
    changed = np.maximum(0.0, values + delta)
    if np.array_equal(changed, values):
        changed = values + abs(delta)
        delta = abs(delta)
    frame.loc[positions, 'speed_mps'] = changed
    return {'field': kind, 'offset_mps': delta}, 'reported_data_distortion'


def _apply_progressive_drift(frame: pd.DataFrame, positions: np.ndarray, config: object,
                             rng: np.random.Generator) -> tuple[dict[str, object], str]:
    """按点序累积报送漂移，复现 DAE 的逐步增减机制而非固定 25 ft 参数。"""
    assert isinstance(config, Mapping)
    kind = _choose_probability(config['field_probabilities'], rng)
    sign = float(rng.choice(np.asarray((-1.0, 1.0))))
    if kind == 'altitude':
        step = _uniform(config['altitude_step_m_range'], rng) * sign
        delta = step * np.arange(1, len(positions) + 1, dtype=np.float64)
        frame.loc[positions, 'geometric_altitude_m'] += delta
        frame.loc[positions, 'barometric_altitude_m'] += delta
        return {'field': kind, 'step_m': step, 'final_delta_m': float(delta[-1])}, 'reported_data_distortion'
    step = _uniform(config['speed_step_mps_range'], rng) * sign
    delta = step * np.arange(1, len(positions) + 1, dtype=np.float64)
    values = frame.loc[positions, 'speed_mps'].to_numpy(dtype=np.float64)
    changed = np.maximum(0.0, values + delta)
    if np.array_equal(changed, values):
        step = abs(step)
        delta = step * np.arange(1, len(positions) + 1, dtype=np.float64)
        changed = values + delta
    frame.loc[positions, 'speed_mps'] = changed
    return {'field': kind, 'step_mps': step, 'final_delta_mps': float(delta[-1])}, 'reported_data_distortion'


def _apply_local_perturbation(frame: pd.DataFrame, positions: np.ndarray, config: object,
                              rng: np.random.Generator, *, recompute_motion: bool) -> tuple[dict[str, object], str]:
    """施加平滑局部波形；仅真实运动语义同步重算速度与航向。"""
    assert isinstance(config, Mapping)
    kind = _choose_local_field(frame, positions, config['field_probabilities'], rng)
    wave = _smooth_wave(np.sin(np.linspace(0.0, np.pi, len(positions), dtype=np.float64)) ** 2,
                        int(config['smoothing_points']))
    semantics = 'true_motion_anomaly' if recompute_motion else 'reported_data_distortion'
    if kind == 'speed':
        factor = _uniform(config['speed_factor_range'], rng)
        values = frame.loc[positions, 'speed_mps'].to_numpy(dtype=np.float64)
        frame.loc[positions, 'speed_mps'] = np.maximum(0.0, values * (1.0 + (factor - 1.0) * wave))
        return {'field': kind, 'speed_factor': factor, 'smoothing_points': int(config['smoothing_points'])}, semantics
    if kind == 'lateral_position':
        amplitude = _uniform(config['lateral_offset_m_range'], rng)
        direction = float(rng.uniform(0.0, 2.0 * np.pi))
        _offset_coordinates(frame, positions, amplitude * wave, direction)
        if recompute_motion:
            _recompute_motion_near_positions(frame, positions)
        return {'field': kind, 'lateral_offset_m': amplitude, 'direction_radians': direction,
                'smoothing_points': int(config['smoothing_points']), 'motion_recomputed': recompute_motion}, semantics
    amplitude = _signed_uniform(config['altitude_amplitude_m_range'], rng)
    delta = amplitude * wave
    frame.loc[positions, 'geometric_altitude_m'] += delta
    frame.loc[positions, 'barometric_altitude_m'] += delta
    return {'field': kind, 'altitude_amplitude_m': amplitude, 'smoothing_points': int(config['smoothing_points'])}, semantics


def _apply_motion_transfer(frame: pd.DataFrame, positions: np.ndarray, donor_rows: object,
                           config: object, rng: np.random.Generator) -> tuple[dict[str, object], str]:
    """将异源真实片段的相对平移和高度变化锚定到目标起点，并重算运动学字段。"""
    assert isinstance(donor_rows, pd.DataFrame)
    assert isinstance(config, Mapping)
    scale = _uniform(config['scale_range'], rng)
    donor_latitude = np.deg2rad(donor_rows['latitude_deg'].to_numpy(dtype=np.float64))
    donor_longitude = np.unwrap(np.deg2rad(donor_rows['longitude_deg'].to_numpy(dtype=np.float64)))
    target_latitude = np.deg2rad(float(frame.at[int(positions[0]), 'latitude_deg']))
    target_longitude = np.deg2rad(float(frame.at[int(positions[0]), 'longitude_deg']))
    donor_reference = donor_latitude[0]
    north = (donor_latitude - donor_reference) * EARTH_RADIUS_M * scale
    east = (donor_longitude - donor_longitude[0]) * EARTH_RADIUS_M * np.cos(donor_reference) * scale
    cosine = np.cos(target_latitude)
    if abs(cosine) < 1e-8:
        raise ValueError('真实运动迁移在接近极点的位置不稳定')
    latitude = target_latitude + north / EARTH_RADIUS_M
    longitude = target_longitude + east / (EARTH_RADIUS_M * cosine)
    if np.any(np.abs(np.rad2deg(latitude)) > 90.0):
        raise ValueError('真实运动迁移产生无效纬度')
    frame.loc[positions, 'latitude_deg'] = np.rad2deg(latitude)
    frame.loc[positions, 'longitude_deg'] = (np.rad2deg(longitude) + 180.0) % 360.0 - 180.0
    geometric = donor_rows['geometric_altitude_m'].to_numpy(dtype=np.float64)
    barometric = donor_rows['barometric_altitude_m'].to_numpy(dtype=np.float64)
    frame.loc[positions, 'geometric_altitude_m'] = float(frame.at[int(positions[0]), 'geometric_altitude_m']) + scale * (geometric - geometric[0])
    frame.loc[positions, 'barometric_altitude_m'] = float(frame.at[int(positions[0]), 'barometric_altitude_m']) + scale * (barometric - barometric[0])
    _recompute_motion_near_positions(frame, positions)
    return {'scale': scale, 'motion_recomputed': True, 'transfer_rule': 'donor_relative_displacement_anchored_to_target_start'}, 'true_motion_anomaly'


def _offset_coordinates(frame: pd.DataFrame, positions: np.ndarray, distance_m: np.ndarray, direction: float) -> None:
    """按球面近似移动位置字段；该函数本身不决定是否同步更新运动学字段。"""
    latitude = frame.loc[positions, 'latitude_deg'].to_numpy(dtype=np.float64)
    cosine = np.cos(np.deg2rad(latitude))
    if np.any(np.abs(cosine) < 1e-8):
        raise ValueError('横向注入在接近极点的位置不稳定')
    latitude = latitude + np.rad2deg(np.cos(direction) * distance_m / EARTH_RADIUS_M)
    longitude = frame.loc[positions, 'longitude_deg'].to_numpy(dtype=np.float64)
    longitude = longitude + np.rad2deg(np.sin(direction) * distance_m / (EARTH_RADIUS_M * cosine))
    if np.any(np.abs(latitude) > 90.0) or not np.isfinite(longitude).all():
        raise ValueError('横向注入产生无效地理坐标')
    frame.loc[positions, 'latitude_deg'] = latitude
    frame.loc[positions, 'longitude_deg'] = (longitude + 180.0) % 360.0 - 180.0


def _recompute_motion_near_positions(frame: pd.DataFrame, positions: np.ndarray) -> None:
    """重算位置改变段及其后一条记录的速度／航向，避免真实运动事件字段自相矛盾。"""
    affected = np.unique(np.concatenate((positions[positions > 0], positions[positions + 1 < len(frame)] + 1)))
    times = frame['time_ns'].to_numpy(dtype=np.int64)
    latitude = np.deg2rad(frame['latitude_deg'].to_numpy(dtype=np.float64))
    longitude = np.deg2rad(frame['longitude_deg'].to_numpy(dtype=np.float64))
    for index in affected:
        delta_time = (times[index] - times[index - 1]) / 1_000_000_000.0
        if not np.isfinite(delta_time) or delta_time <= 0:
            raise ValueError('横向注入需要严格递增的时间戳')
        delta_latitude = latitude[index] - latitude[index - 1]
        delta_longitude = longitude[index] - longitude[index - 1]
        haversine = (np.sin(delta_latitude / 2.0) ** 2
                     + np.cos(latitude[index - 1]) * np.cos(latitude[index]) * np.sin(delta_longitude / 2.0) ** 2)
        distance = 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(haversine, 0.0, 1.0)))
        bearing = np.arctan2(np.sin(delta_longitude) * np.cos(latitude[index]),
                             np.cos(latitude[index - 1]) * np.sin(latitude[index])
                             - np.sin(latitude[index - 1]) * np.cos(latitude[index]) * np.cos(delta_longitude))
        frame.at[int(index), 'speed_mps'] = distance / delta_time
        frame.at[int(index), 'heading_deg'] = np.degrees(bearing) % 360.0


def _changed_positions(before: np.ndarray, frame: pd.DataFrame) -> tuple[np.ndarray, dict[int, tuple[str, ...]]]:
    """返回实际改变记录及逐记录字段，绝不把计划支持集直接当成标签。"""
    after = frame.loc[:, NUMERIC_COLUMNS].to_numpy(dtype=np.float64)
    matrix = after != before
    positions = np.flatnonzero(np.any(matrix, axis=1)).astype(np.int64)
    fields = {int(position): tuple(np.asarray(NUMERIC_COLUMNS)[matrix[position]].tolist()) for position in positions}
    return positions, fields


def _validate_flight_frame(frame: pd.DataFrame) -> None:
    """防止注入产生非有限值、无效坐标、负速度或非单调时间。"""
    values = frame.loc[:, NUMERIC_COLUMNS].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (frame['speed_mps'].to_numpy(dtype=np.float64) < 0).any():
        raise ValueError('注入前后物理字段必须有限，速度不得为负')
    latitude = frame['latitude_deg'].to_numpy(dtype=np.float64)
    longitude = frame['longitude_deg'].to_numpy(dtype=np.float64)
    if (np.abs(latitude) > 90.0).any() or (np.abs(longitude) > 180.0).any():
        raise ValueError('注入前后经纬度必须处于合法范围')
    if np.any(np.diff(frame['time_ns'].to_numpy(dtype=np.int64)) <= 0):
        raise ValueError('注入航班必须按时间严格递增')


def _local_config(raw: Mapping[str, object], name: str, fields: tuple[str, ...]) -> dict[str, object]:
    return {
        'field_probabilities': _probabilities(raw.get('field_probabilities'), f'{name}.field_probabilities', fields),
        'speed_factor_range': _float_range(raw.get('speed_factor_range', [1.0, 1.0]), f'{name}.speed_factor_range',
                                            minimum=0, positive=True),
        'lateral_offset_m_range': _float_range(raw.get('lateral_offset_m_range', [1.0, 1.0]), f'{name}.lateral_offset_m_range',
                                                minimum=0, positive=True),
        'altitude_amplitude_m_range': _float_range(raw.get('altitude_amplitude_m_range', [1.0, 1.0]),
                                                    f'{name}.altitude_amplitude_m_range', minimum=0, positive=True),
        'smoothing_points': _integer(raw.get('smoothing_points'), f'{name}.smoothing_points', minimum=1),
    }


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f'{name} 必须是对象')
    return value


def _finite(value: object, name: str, *, minimum: float, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f'{name} 必须是数值')
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} 必须是数值') from exc
    if not np.isfinite(result) or result < minimum or (positive and result <= minimum):
        raise ValueError(f'{name} 超出有效范围')
    return result


def _fraction(value: object, name: str, *, positive: bool = False) -> float:
    result = _finite(value, name, minimum=0.0, positive=positive)
    if result > 1.0:
        raise ValueError(f'{name} 必须在 0 到 1 之间')
    return result


def _integer(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < minimum:
        raise ValueError(f'{name} 必须是不小于 {minimum} 的整数')
    return int(value)


def _integer_range(value: object, name: str, *, minimum: int) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f'{name} 必须是两个整数的列表')
    lower, upper = (_integer(item, name, minimum=minimum) for item in value)
    if lower > upper:
        raise ValueError(f'{name} 下限不能大于上限')
    return lower, upper


def _float_range(value: object, name: str, *, minimum: float, positive: bool) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f'{name} 必须是两个数值的列表')
    lower, upper = (_finite(item, name, minimum=minimum, positive=positive) for item in value)
    if lower > upper:
        raise ValueError(f'{name} 下限不能大于上限')
    return lower, upper


def _probabilities(value: object, name: str, keys: tuple[str, ...]) -> dict[str, float]:
    mapping = _mapping(value, name)
    if set(mapping) != set(keys):
        raise ValueError(f'{name} 必须且只能包含 {"、".join(keys)}')
    values = {key: _fraction(mapping[key], f'{name}.{key}') for key in keys}
    if not np.isclose(sum(values.values()), 1.0):
        raise ValueError(f'{name} 的概率必须相加为 1')
    return values


def _choose_probability(probabilities: object, rng: np.random.Generator) -> str:
    assert isinstance(probabilities, Mapping)
    names = tuple(str(name) for name, value in probabilities.items() if float(value) > 0)
    values = np.asarray([float(probabilities[name]) for name in names], dtype=np.float64)
    return str(rng.choice(np.asarray(names, dtype=object), p=values / values.sum()))


def _choose_local_field(frame: pd.DataFrame, positions: np.ndarray, probabilities: object,
                        rng: np.random.Generator) -> str:
    """在零速度窗口排除乘性速度扰动，避免地面等待轨迹产生零改变量。"""
    assert isinstance(probabilities, Mapping)
    candidates = {str(name): float(weight) for name, weight in probabilities.items() if float(weight) > 0}
    speeds = frame.loc[positions, 'speed_mps'].to_numpy(dtype=np.float64)
    if 'speed' in candidates and not np.any(speeds != 0.0):
        del candidates['speed']
    if not candidates:
        raise ValueError('当前事件窗口没有可产生实际改动的局部扰动字段')
    names = tuple(candidates)
    weights = np.asarray([candidates[name] for name in names], dtype=np.float64)
    return str(rng.choice(np.asarray(names, dtype=object), p=weights / weights.sum()))


def _uniform(bounds: object, rng: np.random.Generator) -> float:
    assert isinstance(bounds, tuple)
    return float(rng.uniform(float(bounds[0]), float(bounds[1])))


def _signed_uniform(bounds: object, rng: np.random.Generator) -> float:
    return _uniform(bounds, rng) * float(rng.choice(np.asarray((-1.0, 1.0))))


def _nonzero_normal(scale: float, rng: np.random.Generator) -> float:
    delta = float(rng.normal(0.0, scale))
    return delta if delta != 0.0 else scale


def _smooth_wave(wave: np.ndarray, width: int) -> np.ndarray:
    if width == 1 or len(wave) == 1:
        return wave
    width = min(width, len(wave))
    return np.convolve(wave, np.full(width, 1.0 / width, dtype=np.float64), mode='same')
