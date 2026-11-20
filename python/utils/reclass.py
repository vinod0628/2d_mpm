"""Raster reclassification: continuous -> classes (equal interval, quantile, natural breaks,
standard deviation, manual breaks) and categorical value remapping.

Class codes are 1..n (uint8/uint16); 0 is nodata. A ``.classes.csv`` sidecar records
``value, label, lower, upper``.

Command line (from the ``python`` folder)::

    python -m utils.reclass <src.tif> <dst.tif> --method quantile --classes 10
    python -m utils.reclass <src.tif> <dst.tif> --method manual --breaks 500 1000 2000 4000
"""
from __future__ import annotations

import argparse
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .raster_tools import Grid, read_raster, write_raster

from i18n import N_

METHODS = ("equal_interval", "quantile", "natural_breaks", "std_dev", "manual")
METHOD_LABELS = {
    "equal_interval": N_("Equal interval"),
    "quantile": N_("Quantile (equal count)"),
    "natural_breaks": N_("Natural breaks (Fisher-Jenks)"),
    "std_dev": N_("Standard deviation"),
    "manual": N_("Manual breaks"),
}


def compute_breaks(values: np.ndarray, method: str = "quantile", n_classes: int = 5,
                   breaks: Optional[Sequence[float]] = None, max_sample: int = 20000,
                   seed: int = 0) -> List[float]:
    """Return interior class boundaries (length n_classes-1, ascending)."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise ValueError("No valid values to classify.")
    if method == "manual":
        if not breaks:
            raise ValueError("Manual method needs break values.")
        return sorted(float(b) for b in breaks)
    if method == "equal_interval":
        lo, hi = v.min(), v.max()
        return list(np.linspace(lo, hi, n_classes + 1)[1:-1])
    if method == "quantile":
        b = np.quantile(v, np.linspace(0, 1, n_classes + 1)[1:-1])
        return sorted(set(float(x) for x in b))      # ties collapse classes
    if method == "std_dev":
        # n-1 interior breaks, 1 standard deviation apart, centred on the mean
        m, s = v.mean(), v.std()
        b = [m + s * (i - (n_classes - 2) / 2.0) for i in range(n_classes - 1)]
        return sorted(x for x in b if v.min() < x < v.max())
    if method == "natural_breaks":
        return fisher_jenks_breaks(v, n_classes, max_sample=min(max_sample, 2500), seed=seed)
    raise ValueError(f"Unknown method {method!r}; use one of {METHODS}")


def fisher_jenks_breaks(values, n_classes: int, max_sample: int = 2500, seed: int = 0) -> List[float]:
    """Exact Fisher–Jenks natural breaks (minimum within-class sum of squares), vectorised with numpy.

    Large inputs are reduced to a random sample of ``max_sample`` values (min and max always kept).
    Dynamic programming over the sorted sample: O(k·n²) array operations, ~0.1 s for n = 2500, k = 10.
    Returns the interior class boundaries (upper bound of each class except the last).
    """
    v = np.sort(np.asarray(values, float)[np.isfinite(values)])
    if v.size > max_sample:
        rng = np.random.default_rng(seed)
        v = np.sort(np.r_[v[0], v[-1], rng.choice(v[1:-1], max_sample - 2, replace=False)])
    n = v.size
    k = int(min(n_classes, np.unique(v).size))
    if k < 2:
        return []
    s1 = np.r_[0.0, np.cumsum(v)]
    s2 = np.r_[0.0, np.cumsum(v * v)]
    i = np.arange(n)[:, None]                 # segment start
    j = np.arange(n)[None, :]                 # segment end (inclusive)
    cnt = (j - i + 1).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        ssd = (s2[j + 1] - s2[i]) - (s1[j + 1] - s1[i]) ** 2 / cnt
    ssd = np.where(j >= i, np.maximum(ssd, 0.0), np.inf)
    cost = ssd[0].copy()                      # 1 class covering v[0..j]
    back = []
    for _ in range(1, k):
        # new class starts at i (>= 1): previous classes cover v[0..i-1]
        total = np.full((n, n), np.inf)
        total[1:, :] = cost[:-1, None] + ssd[1:, :]
        arg = np.argmin(total, axis=0)
        cost = total[arg, np.arange(n)]
        back.append(arg)
    starts, end = [], n - 1
    for arg in reversed(back):
        s = int(arg[end])
        starts.append(s)
        end = s - 1
    starts = sorted(starts)
    return [float(v[s - 1]) for s in starts]


def reclass_by_breaks(array, breaks: Sequence[float], right: bool = True
                      ) -> Tuple[np.ma.MaskedArray, pd.DataFrame]:
    """Classify with interior breaks -> codes 1..len(breaks)+1. Returns (classes, class table).

    right=True: a value equal to a break goes to the lower class (upper bound inclusive).
    """
    arr = np.ma.asarray(array).astype("float64")
    b = np.asarray(sorted(breaks), float)
    data = np.ma.filled(arr, np.nan)
    codes = np.digitize(data, b, right=right) + 1
    n = len(b) + 1
    dtype = "uint8" if n < 255 else "uint16"
    out = np.ma.array(codes.astype(dtype), mask=np.ma.getmaskarray(arr) | ~np.isfinite(data))
    vals = np.ma.compressed(arr)
    lo_all, hi_all = (float(vals.min()), float(vals.max())) if vals.size else (np.nan, np.nan)
    edges = [lo_all] + list(b) + [hi_all]
    rows = []
    for i in range(n):
        lo, hi = edges[i], edges[i + 1]
        rows.append(dict(value=i + 1, label=f"{_fmt(lo)} - {_fmt(hi)}", lower=lo, upper=hi,
                         count=int(np.ma.sum(out == i + 1))))
    return out, pd.DataFrame(rows)


def reclass_by_table(array, mapping: Dict[float, float], default=None) -> np.ma.MaskedArray:
    """Remap discrete values; values not in ``mapping`` become ``default`` (None = nodata)."""
    arr = np.ma.asarray(array)
    data = np.ma.filled(arr.astype("float64"), np.nan)
    out = np.full(data.shape, np.nan)
    for k, v in mapping.items():
        out[data == k] = v
    if default is not None:
        out[np.isnan(out) & np.isfinite(data)] = default
    return np.ma.masked_invalid(out)


def reclass_by_ranges(array, ranges: Sequence[Tuple[float, float, float]]) -> np.ma.MaskedArray:
    """Remap value ranges ``(lower, upper, new)``; lower < x <= upper. Unmatched -> nodata."""
    arr = np.ma.asarray(array)
    data = np.ma.filled(arr.astype("float64"), np.nan)
    out = np.full(data.shape, np.nan)
    for lo, hi, new in ranges:
        out[(data > lo) & (data <= hi)] = new
    return np.ma.masked_invalid(out)


def reclass_raster_file(src: str, dst: str, method: str = "quantile", n_classes: int = 5,
                        breaks: Optional[Sequence[float]] = None) -> Tuple[str, pd.DataFrame]:
    arr, grid, _ = read_raster(src, as_float=True)
    b = compute_breaks(np.ma.compressed(arr), method, n_classes, breaks)
    classes, table = reclass_by_breaks(arr, b)
    write_raster(dst, classes, grid, nodata=0, class_table=table)
    return dst, table


def _fmt(x: float) -> str:
    if not np.isfinite(x):
        return "nan"
    return f"{x:.0f}" if abs(x) >= 1000 else f"{x:.4g}"


def _main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--method", choices=METHODS, default="quantile")
    ap.add_argument("--classes", type=int, default=5)
    ap.add_argument("--breaks", type=float, nargs="*")
    a = ap.parse_args(argv)
    dst, table = reclass_raster_file(a.src, a.dst, a.method, a.classes, a.breaks)
    print(dst); print(table.to_string(index=False))


if __name__ == "__main__":
    _main()
