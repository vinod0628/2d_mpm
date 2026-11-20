"""Algorithm of the method — GUI-free, unit-testable. (Template example: weighted mean of min–max-normalised evidence.)"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def weighted_mean_score(arrays: Sequence[np.ma.MaskedArray], weights: Sequence[float]) -> np.ma.MaskedArray:
    """Prospectivity score in [0, 1]: weighted mean of evidence layers rescaled to [0, 1] (all on one grid)."""
    if len(arrays) != len(weights) or not arrays:
        raise ValueError("Need one weight per evidence layer.")
    total = np.zeros(np.ma.asarray(arrays[0]).shape)
    mask = np.zeros(total.shape, bool)
    for a, w in zip(arrays, weights):
        a = np.ma.asarray(a, dtype=float)
        lo, hi = float(a.min()), float(a.max())
        norm = (a - lo) / (hi - lo) if hi > lo else a * 0.0
        total += w * np.ma.filled(norm, 0.0)
        mask |= np.ma.getmaskarray(a)
    return np.ma.array(total / float(np.sum(weights)), mask=mask)
