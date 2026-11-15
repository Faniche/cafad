"""metrics.py | 2026-11-09 | 二分类计数、指标与严格 ID 对齐 | Codex."""
import numpy as np
import pandas as pd


def binary_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, object]:
    """未定义分母有样本时记零并标记；无样本指标记 null。"""
    if len(labels) != len(predictions) or not set(labels).issubset({0, 1}) or not set(predictions).issubset({0, 1}):
        raise ValueError('标签和预测必须一一对应且为 0/1')
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    ratios = {'precision': (tp, tp + fp), 'recall': (tp, tp + fn), 'f1': (2 * tp, 2 * tp + fp + fn),
              'false_positive_rate': (fp, fp + tn), 'accuracy': (tp + tn, len(labels))}
    return {'samples': len(labels), 'normal': tn + fp, 'abnormal': tp + fn, 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            **{key: numerator / denominator if denominator else (0.0 if len(labels) else None)
               for key, (numerator, denominator) in ratios.items()},
            'undefined_metrics': [key for key, (_, denominator) in ratios.items() if denominator == 0]}


def align_segments(predictions: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """严格匹配完整分区，不通过 inner join 静默丢弃未预测样本。"""
    if predictions.empty or predictions.segment_id.duplicated().any() or truth.segment_id.duplicated().any():
        raise ValueError('预测为空或片段 ID 重复')
    if set(predictions.segment_id) != set(truth.segment_id):
        raise ValueError('预测与真实分区的片段 ID 不完全一致')
    result = predictions.merge(truth, on='segment_id', validate='one_to_one', suffixes=('', '_truth'))
    for field in ('flight_id', 'cell_key', 'split'):
        if not result[field].equals(result[field + '_truth']):
            raise ValueError(f'预测与真实片段的 {field} 不一致')
    if not np.isfinite(result.score).all() or not result.score.between(0, 1).all():
        raise ValueError('预测分数无效')
    return result.drop(columns=['flight_id_truth', 'cell_key_truth', 'split_truth'])
