"""Descriptive statistics and histogram intervals for the Statistics tab of the layer properties dialog.

Moments are population moments (as the ArcGIS histogram tools report them): standard deviation √m₂,
skewness m₃ / m₂^1.5 and kurtosis m₄ / m₂² (Pearson; 3 for a normal distribution). Quartiles use linear
interpolation between order statistics (numpy default).
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np

MAX_BINS = 1000


def describe(values, nulls: int = 0) -> Dict[str, Optional[float]]:
    """Statistics of the finite values of ``values`` (NaN / ±inf are counted as nulls)."""
    v = np.asarray(values, dtype=float).ravel()
    finite = np.isfinite(v)
    nulls = int(nulls) + int((~finite).sum())
    v = v[finite]
    out: Dict[str, Optional[float]] = dict(count=int(v.size), nulls=nulls)
    keys = ("min", "max", "range", "sum", "mean", "std", "cv", "skewness", "kurtosis", "q1", "median", "q3")
    if v.size == 0:
        out.update({k: None for k in keys})
        return out
    mean = float(v.mean())
    d = v - mean
    m2 = float(np.mean(d ** 2))
    std = math.sqrt(m2)
    if m2 > 0:
        skew = float(np.mean(d ** 3)) / m2 ** 1.5
        kurt = float(np.mean(d ** 4)) / m2 ** 2
    else:
        skew = kurt = None                                   # constant values: undefined
    q1, med, q3 = (float(x) for x in np.percentile(v, [25, 50, 75]))
    out.update(min=float(v.min()), max=float(v.max()), range=float(v.max() - v.min()), sum=float(v.sum()),
               mean=mean, std=std, cv=(std / abs(mean) if mean != 0 else None), skewness=skew, kurtosis=kurt,
               q1=q1, median=med, q3=q3)
    return out


def sturges_bins(n: int) -> int:
    """Default number of intervals: Sturges' rule ⌈log₂ n⌉ + 1, kept between 5 and 50."""
    return int(min(50, max(5, math.ceil(math.log2(max(n, 1))) + 1)))


def nice_width(span: float, n: int) -> float:
    """A round interval width (1, 2, 2.5 or 5 × 10^k) close to span / n."""
    if not (span > 0) or n < 1:
        return 1.0
    raw = span / n
    p = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if m * p >= raw * 0.999:
            return m * p
    return 10 * p


def histogram_edges(vmin: float, vmax: float, count: Optional[int] = None, width: Optional[float] = None,
                    max_bins: int = MAX_BINS) -> np.ndarray:
    """Interval edges covering [vmin, vmax].

    ``count``: that many equal intervals from vmin to vmax. ``width``: intervals of that size, aligned to whole
    multiples of the width (e.g. 0, 0.5, 1.0 … for a width of 0.5) so the class limits are round values.
    """
    if vmax < vmin:
        raise ValueError("vmax < vmin")
    if vmax == vmin:                                          # a single value: one interval around it
        h = 0.5 * (width if width and width > 0 else (abs(vmin) * 0.01 or 1.0))
        return np.array([vmin - h, vmin + h])
    if width is not None:
        if not (width > 0):
            raise ValueError("the interval width must be positive")
        start = math.floor(vmin / width) * width
        k = max(1, math.ceil((vmax - start) / width - 1e-9))
        if start + k * width < vmax:
            k += 1
        if k > max_bins:
            raise ValueError(f"{k} intervals — more than {max_bins}; use a larger width")
        return start + width * np.arange(k + 1)
    n = int(count or 10)
    if not 1 <= n <= max_bins:
        raise ValueError(f"the number of intervals must be 1 … {max_bins}")
    return np.linspace(vmin, vmax, n + 1)


def ecdf(values, max_points: int = 2000):
    """(x, F(x)) of the empirical cumulative distribution, at most ``max_points`` points (sorted values)."""
    v = np.sort(np.asarray(values, dtype=float).ravel())
    v = v[np.isfinite(v)]
    n = v.size
    if n == 0:
        return v, v
    idx = np.unique(np.linspace(0, n - 1, min(n, max_points)).round().astype(int))
    return v[idx], (idx + 1) / n
