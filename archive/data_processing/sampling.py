"""Physical-time sampling extracted from the existing preparation code.

Pass nonempty strictly increasing times and a positive integer count.
Nearest observed records are selected; no synthetic observations are created.
"""
from __future__ import annotations

import numpy as np

def sample_observed_times(times: np.ndarray, count: int) -> np.ndarray:
    """Choose actual observations nearest equally spaced physical times."""
    if count >= len(times):
        return np.arange(len(times), dtype=np.int64)
    targets = np.linspace(times[0], times[-1], count)
    right = np.minimum(np.searchsorted(times, targets), len(times) - 1)
    left = np.maximum(right - 1, 0)
    selected = np.where(np.abs(times[right] - targets) < np.abs(times[left] - targets), right, left)
    return np.unique(selected).astype(np.int64)
