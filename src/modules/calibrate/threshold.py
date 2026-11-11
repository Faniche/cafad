"""threshold.py | 2026-11-09 | 完整验证片段 F1 阈值扫描 | Codex."""
import numpy as np
import pandas as pd


def select_segment_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, pd.DataFrame]:
    """score>=threshold；F1 并列时取较大阈值，两类标签必须齐全。"""
    if len(scores) != len(labels) or not np.isfinite(scores).all() or set(labels) != {0, 1}:
        raise ValueError('完整验证集必须包含正常和异常标签，分数必须有限')
    rows = []
    for threshold in np.unique(np.r_[0.0, scores, 1.0]):
        predicted = scores >= threshold
        tp = int(np.sum(predicted & (labels == 1)))
        fp = int(np.sum(predicted & (labels == 0)))
        fn = int(np.sum(~predicted & (labels == 1)))
        rows.append({'threshold': float(threshold), 'tp': tp, 'fp': fp, 'fn': fn, 'f1': 2 * tp / (2 * tp + fp + fn)})
    scan = pd.DataFrame(rows)
    best = scan.sort_values(['f1', 'threshold']).iloc[-1]
    return float(best['threshold']), scan
