"""features.py | 2026-11-12 | 阶段片段统计特征、标准化与加权距离 | Codex."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES = (
    'mean_longitude_deg', 'mean_latitude_deg',
    'mean_geometric_altitude_m', 'altitude_range_m',
    'mean_speed_mps', 'std_speed_mps',
    'mean_vertical_rate_mps', 'mean_acceleration_mps2',
    'mean_heading_sin', 'mean_heading_cos', 'mean_abs_turn_rate_radps',
    'center_time_s', 'duration_s',
)

FEATURE_WEIGHT_ARGUMENTS = (
    ('mean_longitude_deg', 'weight_mean_longitude', '片段平均经度特征权重'),
    ('mean_latitude_deg', 'weight_mean_latitude', '片段平均纬度特征权重'),
    ('mean_geometric_altitude_m', 'weight_mean_altitude', '片段平均几何高度特征权重'),
    ('altitude_range_m', 'weight_altitude_range', '片段几何高度极差特征权重'),
    ('mean_speed_mps', 'weight_mean_speed', '片段平均速度特征权重'),
    ('std_speed_mps', 'weight_speed_std', '片段速度标准差特征权重'),
    ('mean_vertical_rate_mps', 'weight_vertical_rate', '片段平均垂直速率特征权重'),
    ('mean_acceleration_mps2', 'weight_acceleration', '片段平均加速度特征权重'),
    ('mean_heading_sin', 'weight_heading_sin', '片段平均航向正弦特征权重'),
    ('mean_heading_cos', 'weight_heading_cos', '片段平均航向余弦特征权重'),
    ('mean_abs_turn_rate_radps', 'weight_turn_rate', '片段平均绝对转弯率特征权重'),
    ('center_time_s', 'weight_center_time', '相对数据集起点的片段中心时间特征权重'),
    ('duration_s', 'weight_duration', '片段持续时间特征权重'),
)


@dataclass(frozen=True)
class FeatureScaler:
    """仅由训练正常片段拟合的标准化统计量及各维非负权重。"""

    means: np.ndarray
    scales: np.ndarray
    weights: np.ndarray
    time_origin_ns: int

    def as_dict(self) -> dict[str, object]:
        """返回可复现实验的 JSON 记录。"""
        return {
            'feature_names': list(FEATURE_NAMES),
            'means': self.means.tolist(),
            'scales': self.scales.tolist(),
            'weights': self.weights.tolist(),
            'distance': 'weighted_euclidean_on_standardized_features',
            'altitude_source': 'geometric_altitude_m',
            'barometric_altitude_policy': 'excluded_as_highly_correlated_with_geometric_altitude',
            'center_time_origin': 'minimum_observed_time_ns_in_segment_input',
            'center_time_origin_ns': self.time_origin_ns,
        }


def build_segment_features(segments: pd.DataFrame, values: pd.DataFrame, time_origin_ns: int | None = None) -> pd.DataFrame:
    """将每个片段的原始观测压缩为固定 13 维物理—时空统计向量。"""
    required = {'segment_id'}
    required_values = {
        'segment_id', 'time_ns', 'longitude_deg', 'latitude_deg', 'geometric_altitude_m', 'heading_deg', 'speed_mps',
    }
    if not required.issubset(segments.columns) or not required_values.issubset(values.columns):
        raise ValueError('片段或观测缺少统计特征所需字段')
    if segments['segment_id'].duplicated().any() or values['segment_id'].isna().any():
        raise ValueError('片段 ID 必须唯一，观测必须具有片段 ID')
    origin_ns = int(values['time_ns'].min()) if time_origin_ns is None else int(time_origin_ns)
    rows: list[dict[str, float | str]] = []
    for segment_id, frame in values.groupby('segment_id', sort=False, observed=True):
        ordered = frame.sort_values('time_ns', kind='stable')
        times = ordered['time_ns'].to_numpy(dtype=np.int64)
        longitude = ordered['longitude_deg'].to_numpy(dtype=np.float64)
        latitude = ordered['latitude_deg'].to_numpy(dtype=np.float64)
        altitude = ordered['geometric_altitude_m'].to_numpy(dtype=np.float64)
        heading_rad = np.deg2rad(np.mod(ordered['heading_deg'].to_numpy(dtype=np.float64), 360.0))
        speed = ordered['speed_mps'].to_numpy(dtype=np.float64)
        if (not len(times) or np.any(np.diff(times) <= 0) or
                not np.isfinite(np.column_stack((longitude, latitude, altitude, heading_rad, speed))).all()):
            raise ValueError(f'片段 {segment_id} 的观测字段非有限或时间不严格递增')
        elapsed_s = np.diff(times).astype(np.float64) / 1_000_000_000.0
        vertical_rate = np.diff(altitude) / elapsed_s if len(elapsed_s) else np.zeros(1, dtype=np.float64)
        acceleration = np.diff(speed) / elapsed_s if len(elapsed_s) else np.zeros(1, dtype=np.float64)
        heading_delta = np.arctan2(np.sin(np.diff(heading_rad)), np.cos(np.diff(heading_rad)))
        turn_rate = np.abs(heading_delta / elapsed_s) if len(elapsed_s) else np.zeros(1, dtype=np.float64)
        row: dict[str, float | str] = {
            'segment_id': str(segment_id),
            'mean_longitude_deg': float(longitude.mean()),
            'mean_latitude_deg': float(latitude.mean()),
            'mean_geometric_altitude_m': float(altitude.mean()),
            'altitude_range_m': float(altitude.max() - altitude.min()),
            'mean_speed_mps': float(speed.mean()),
            'std_speed_mps': float(speed.std(ddof=0)),
            'mean_vertical_rate_mps': float(vertical_rate.mean()),
            'mean_acceleration_mps2': float(acceleration.mean()),
            'mean_heading_sin': float(np.sin(heading_rad).mean()),
            'mean_heading_cos': float(np.cos(heading_rad).mean()),
            'mean_abs_turn_rate_radps': float(turn_rate.mean()),
            'center_time_s': float(((times[0] - origin_ns) + (times[-1] - times[0]) / 2.0) / 1_000_000_000.0),
            'duration_s': float((times[-1] - times[0]) / 1_000_000_000.0),
        }
        rows.append(row)
    result = pd.DataFrame(rows, columns=('segment_id', *FEATURE_NAMES))
    if set(result['segment_id']) != set(segments['segment_id'].astype(str)):
        raise ValueError('存在没有完整统计特征的阶段片段')
    if not np.isfinite(result.loc[:, FEATURE_NAMES].to_numpy(dtype=np.float64)).all():
        raise ValueError('片段统计特征包含非有限数值')
    return result


def weights_from_arguments(arguments: object) -> np.ndarray:
    """按确定字段顺序读取网页和命令行共同暴露的各维权重。"""
    weights = np.asarray([getattr(arguments, destination) for _, destination, _ in FEATURE_WEIGHT_ARGUMENTS], dtype=np.float64)
    if len(weights) != len(FEATURE_NAMES) or not np.isfinite(weights).all() or np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError('片段特征权重必须是有限非负数，且至少一个维度大于零')
    return weights


def fit_feature_scaler(features: pd.DataFrame, weights: np.ndarray, time_origin_ns: int) -> FeatureScaler:
    """只以训练正常片段拟合 StandardScaler，并固定后续所有分区的同一变换。"""
    values = features.loc[:, FEATURE_NAMES].to_numpy(dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('训练正常片段特征为空或无效')
    fitted = StandardScaler().fit(values)
    return FeatureScaler(np.asarray(fitted.mean_, dtype=np.float64), np.asarray(fitted.scale_, dtype=np.float64), weights,
                         int(time_origin_ns))


def feature_scaler_from_dict(value: object) -> FeatureScaler:
    """读取 group 已冻结的特征统计量，拒绝缺维或与当前特征顺序不一致的产物。"""
    if not isinstance(value, dict) or value.get('feature_names') != list(FEATURE_NAMES):
        raise ValueError('group 特征统计量版本或维度不兼容')
    try:
        means = np.asarray(value['means'], dtype=np.float64)
        scales = np.asarray(value['scales'], dtype=np.float64)
        weights = np.asarray(value['weights'], dtype=np.float64)
        time_origin_ns = int(value['center_time_origin_ns'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('group 特征统计量格式无效') from exc
    if (means.shape != (len(FEATURE_NAMES),) or scales.shape != means.shape or weights.shape != means.shape or
            not np.isfinite(means).all() or not np.isfinite(scales).all() or not np.isfinite(weights).all() or
            np.any(scales <= 0) or np.any(weights < 0) or not np.any(weights > 0)):
        raise ValueError('group 特征统计量数值无效')
    return FeatureScaler(means, scales, weights, time_origin_ns)


def transform_features(features: pd.DataFrame, scaler: FeatureScaler) -> pd.DataFrame:
    """应用冻结 StandardScaler，再以权重平方根实现加权欧氏距离。"""
    values = features.loc[:, FEATURE_NAMES].to_numpy(dtype=np.float64)
    transformed = (values - scaler.means) / scaler.scales
    transformed *= np.sqrt(scaler.weights)
    if not np.isfinite(transformed).all():
        raise ValueError('标准化后的片段特征无效')
    return pd.DataFrame(transformed, columns=FEATURE_NAMES, index=features.index)


def weighted_distance_matrix(vectors: np.ndarray, block_size: int) -> np.ndarray:
    """分块计算完整加权欧氏预计算矩阵，避免构造三维差值张量。"""
    if block_size < 1 or vectors.ndim != 2 or not len(vectors) or not np.isfinite(vectors).all():
        raise ValueError('距离矩阵输入或分块大小无效')
    squared_norms = np.einsum('ij,ij->i', vectors, vectors)
    matrix = np.empty((len(vectors), len(vectors)), dtype=np.float64)
    for start in range(0, len(vectors), block_size):
        stop = min(start + block_size, len(vectors))
        squared = squared_norms[start:stop, None] + squared_norms[None, :] - 2.0 * vectors[start:stop] @ vectors.T
        matrix[start:stop] = np.sqrt(np.maximum(squared, 0.0))
    np.fill_diagonal(matrix, 0.0)
    return matrix


def centroids_by_group(core: pd.DataFrame, vectors_by_segment: pd.DataFrame) -> dict[str, np.ndarray]:
    """从以 segment_id 为索引的冻结特征表计算各组中心。"""
    result: dict[str, np.ndarray] = {}
    for group_id, members in core.groupby('group_id', sort=False, observed=True):
        ids = members['segment_id'].astype(str).tolist()
        result[str(group_id)] = vectors_by_segment.loc[ids, list(FEATURE_NAMES)].to_numpy(dtype=np.float64).mean(axis=0)
    return result
