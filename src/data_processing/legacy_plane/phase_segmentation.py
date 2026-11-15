"""Historical vertical-rate threshold approximation; NOT the Sun method.

Altitude remains in native input units. This offline helper uses a centered
median window and merges short runs; it is not a streaming phase detector.
Phase codes: 0 level, 1 climb, 2 descent. Ranges use exclusive end indices.
"""
from __future__ import annotations

from typing import List, Tuple
import numpy as np
import pandas as pd

def _runs(values: np.ndarray) -> List[Tuple[int, int]]:
    boundaries = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1, len(values)]
    return list(zip(boundaries[:-1].tolist(), boundaries[1:].tolist()))


def phase_ranges_vertical_rate(time_seconds: np.ndarray, altitude: np.ndarray, threshold: float) -> List[Tuple[int, int, int]]:
    dt = np.diff(time_seconds)
    if len(dt) == 0 or np.any(dt <= 0):
        raise ValueError('Phase segmentation requires strictly increasing observed timestamps')
    vertical = np.r_[0.0, np.diff(altitude.astype(np.float64)) / dt]
    vertical = pd.Series(vertical).rolling(5, min_periods=1, center=True).median().to_numpy()
    phase = np.where(vertical > threshold, 1, np.where(vertical < -threshold, 2, 0)).astype(np.int8)
    while True:
        runs = _runs(phase)
        if len(runs) == 1:
            break
        short = next(((j, start, end) for j, (start, end) in enumerate(runs) if end - start < 3 or time_seconds[end - 1] - time_seconds[start] < 5.0), None)
        if short is None:
            break
        j, start, end = short
        if j == 0:
            neighbor = phase[runs[j + 1][0]]
        elif j == len(runs) - 1:
            neighbor = phase[runs[j - 1][0]]
        else:
            left_duration = time_seconds[runs[j - 1][1] - 1] - time_seconds[runs[j - 1][0]]
            right_duration = time_seconds[runs[j + 1][1] - 1] - time_seconds[runs[j + 1][0]]
            neighbor = phase[runs[j - 1][0]] if left_duration >= right_duration else phase[runs[j + 1][0]]
        phase[start:end] = neighbor
    return [(start, end, int(phase[start])) for start, end in _runs(phase)]
