"""fusion.py | 2026-11-09 | 固定规则融合与片段/航班判别 | Codex."""
import numpy as np
import pandas as pd

def fallback_score(scores: np.ndarray) -> float:
    """严格多数最短窗口；并列比较与全体中位数距离、起始索引。"""
    values = np.sort(scores, kind='stable')
    size = len(values) // 2 + 1
    median = np.median(values)
    start = min(range(len(values) - size + 1), key=lambda i: (
        values[i + size - 1] - values[i], abs(np.median(values[i:i + size]) - median), i))
    return float(np.median(values[start:start + size]))


def make_predictions(scores: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """固定阈值逐片段判别，再按来源航班取 OR。"""
    result = scores.copy()
    result['prediction'] = (result['score'] >= threshold).astype(int)
    flights = result.groupby('flight_id', sort=True).agg(score=('score', 'max'), prediction=('prediction', 'max'),
                                                        segments=('segment_id', 'size')).reset_index()
    return result, flights
