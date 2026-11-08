"""distance.py | 2026-11-08 | 片段锚点采样与 Haversine 片段距离 | Codex."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import yaml

from src.data_processing.sampling import sample_observed_times


@dataclass(frozen=True)
class DistanceConfig:
    """已冻结的片段距离定义；不以代码默认值替代实验选择。"""

    time_mode: str
    sigma_time_s: float
    sigma_space_m: float
    sigma_heading_rad: float
    aggregation: str
    earth_radius_m: float


def load_distance_config(path: Path) -> DistanceConfig:
    """读取并严格校验距离配置。"""
    try:
        raw = json.loads(path.read_text(encoding='utf-8')) if path.suffix.lower() == '.json' else yaml.safe_load(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f'无法读取片段距离配置：{path}') from exc
    if not isinstance(raw, dict) or raw.get('schema_version') != 1:
        raise ValueError('距离配置必须为 schema_version=1 的 JSON/YAML 对象')
    mode, aggregation = raw.get('time_mode'), raw.get('aggregation')
    if mode not in {'absolute', 'time_of_day'} or aggregation != 'mean':
        raise ValueError('当前仅支持 time_mode=absolute/time_of_day 且 aggregation=mean')
    values = {name: raw.get(name) for name in ('sigma_time_s', 'sigma_space_m', 'sigma_heading_rad', 'earth_radius_m')}
    try:
        numbers = {name: float(value) for name, value in values.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError('距离尺度和 earth_radius_m 必须为有限正数') from exc
    if any(not np.isfinite(value) or value <= 0 for value in numbers.values()):
        raise ValueError('距离尺度和 earth_radius_m 必须为有限正数')
    return DistanceConfig(time_mode=mode, aggregation=aggregation, **numbers)


def sample_anchors(times_ns: np.ndarray, longitude_deg: np.ndarray, latitude_deg: np.ndarray,
                   heading_deg: np.ndarray, maximum: int) -> np.ndarray:
    """从一个严格递增片段抽取实际锚点 [time_s, lon_rad, lat_rad, heading_rad]。"""
    if maximum < 1 or not len(times_ns) or len(times_ns) != len(longitude_deg) or len(times_ns) != len(latitude_deg):
        raise ValueError('锚点输入为空或字段长度不一致')
    seconds = np.asarray(times_ns, dtype=np.float64) / 1_000_000_000.0
    if not np.isfinite(seconds).all() or np.any(np.diff(seconds) <= 0):
        raise ValueError('片段时间必须有限且严格递增')
    positions = sample_observed_times(seconds, maximum)
    return np.column_stack((seconds[positions], np.deg2rad(np.asarray(longitude_deg)[positions]),
                            np.deg2rad(np.asarray(latitude_deg)[positions]),
                            np.deg2rad(np.mod(np.asarray(heading_deg)[positions], 360.0))))


def segment_distance(left: np.ndarray, right: np.ndarray, config: DistanceConfig) -> float:
    """按等相对位置对应锚点，返回对称的组合平均距离。"""
    if not len(left) or not len(right):
        raise ValueError('片段锚点不能为空')
    count = min(len(left), len(right))
    left_index = np.linspace(0, len(left) - 1, count, dtype=np.int64)
    right_index = np.linspace(0, len(right) - 1, count, dtype=np.int64)
    a, b = left[left_index], right[right_index]
    if config.time_mode == 'absolute':
        time_delta = np.abs(a[:, 0] - b[:, 0])
    else:
        time_delta = np.abs(np.mod(a[:, 0], 86_400.0) - np.mod(b[:, 0], 86_400.0))
        time_delta = np.minimum(time_delta, 86_400.0 - time_delta)
    lat_a, lon_a, lat_b, lon_b = a[:, 2], a[:, 1], b[:, 2], b[:, 1]
    hav = np.sin((lat_a - lat_b) / 2.0) ** 2 + np.cos(lat_a) * np.cos(lat_b) * np.sin((lon_a - lon_b) / 2.0) ** 2
    space_delta = 2.0 * config.earth_radius_m * np.arcsin(np.sqrt(np.clip(hav, 0.0, 1.0)))
    heading_delta = np.abs(a[:, 3] - b[:, 3])
    heading_delta = np.minimum(heading_delta, 2.0 * np.pi - heading_delta)
    distances = np.sqrt((time_delta / config.sigma_time_s) ** 2 + (space_delta / config.sigma_space_m) ** 2
                        + (heading_delta / config.sigma_heading_rad) ** 2)
    value = float(np.mean(distances))
    if not np.isfinite(value) or value < 0:
        raise ValueError('片段距离计算结果无效')
    return value
