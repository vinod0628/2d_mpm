"""Target delineation: threshold any raster (prospectivity map or evidence layer) into a binary target map.

Two data-driven ways to pick the threshold:

* **Prediction–area (P-A) plot** — Yousefi & Carranza (2015, Comput. Geosci. 79, 69–81). For every threshold t:
  prediction rate P(t) = % of known deposits in cells with value ≥ t, occupied area A(t) = % of the study area with
  value ≥ t. The intersection of P(t) with 100 − A(t) is the threshold where the share of deposits predicted equals the
  share of the area left out (the best trade-off). Needs known deposits.
* **Concentration–area (C-A) fractal** — Cheng, Agterberg & Ballantyne (1994, J. Geochem. Explor. 51, 109–130).
  log10 A(≥v) is plotted against log10 v; populations appear as straight segments and the breakpoints between them
  are thresholds. The break before the highest-valued segment separates the anomaly / target. Needs no deposits.
  Segments are found by an exact least-squares segmentation (dynamic programming).

The binary map has 1 = target (value ≥ threshold), 0 = non-target, masked = outside the study area / nodata.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from i18n import tr

TARGET_COLOR = "#e53935"          # red
TRANSPARENT = "#00000000"


def valid_values(array) -> np.ndarray:
    a = np.ma.asarray(array)
    v = np.ma.getdata(a)[~np.ma.getmaskarray(a)].astype(float)
    return v[np.isfinite(v)]


def _thresholds(values: np.ndarray, n: int, extra: Optional[np.ndarray] = None) -> np.ndarray:
    qs = np.quantile(values, np.linspace(0, 1, n))
    t = np.unique(np.r_[qs, extra] if extra is not None and len(extra) else qs)
    return t


# ============================================================================== P-A
@dataclass
class PACurve:
    thresholds: np.ndarray            # ascending raster values
    prediction: np.ndarray            # % deposits with value >= t
    area: np.ndarray                  # % study area with value >= t
    intersection: Dict[str, float]    # threshold / prediction / area at P = 100 - A ({} if none)
    n_deposits: int


def pa_curve(array, dep_rows, dep_cols, n: int = 400) -> PACurve:
    a = np.ma.asarray(array)
    valid = ~np.ma.getmaskarray(a) & np.isfinite(np.ma.getdata(a).astype(float))
    rows, cols = np.asarray(dep_rows, int), np.asarray(dep_cols, int)
    keep = valid[rows, cols] if len(rows) else np.zeros(0, bool)
    dep = np.ma.getdata(a)[rows[keep], cols[keep]].astype(float)
    if dep.size == 0:
        raise ValueError(tr("No deposits fall on valid cells of this raster."))
    values = np.sort(np.ma.getdata(a)[valid].astype(float))
    dep_sorted = np.sort(dep)
    t = _thresholds(values, n, dep)
    area = 100.0 * (1.0 - np.searchsorted(values, t, side="left") / values.size)
    pred = 100.0 * (1.0 - np.searchsorted(dep_sorted, t, side="left") / dep_sorted.size)
    inter = {}
    diff = pred - (100.0 - area)                   # decreasing in t: + below the intersection, - above
    idx = np.where((diff[:-1] >= 0) & (diff[1:] < 0))[0]
    if idx.size:
        i = int(idx[0])
        f = diff[i] / (diff[i] - diff[i + 1]) if diff[i] != diff[i + 1] else 0.0
        inter = dict(threshold=float(t[i] + f * (t[i + 1] - t[i])),
                     prediction=float(pred[i] + f * (pred[i + 1] - pred[i])),
                     area=float(area[i] + f * (area[i + 1] - area[i])))
    return PACurve(t, pred, area, inter, int(dep.size))


# ============================================================================== C-A
@dataclass
class CACurve:
    values: np.ndarray                # thresholds v (ascending, original units)
    area: np.ndarray                  # A(>= v) in area units (km² if cell area given, else cells)
    log_v: np.ndarray
    log_a: np.ndarray
    shift: float                      # added to v before log10 (0 unless the raster has values <= 0)
    segments: List[Tuple[int, int, float, float]] = field(default_factory=list)  # (i0, i1, slope, intercept)
    breakpoints: List[float] = field(default_factory=list)                       # thresholds (original units)

    @property
    def target_threshold(self) -> Optional[float]:
        """Break before the highest-valued segment."""
        return self.breakpoints[-1] if self.breakpoints else None


def _segment_sse(x, y):
    """Prefix sums -> function sse(i, j) of a least-squares line through points i..j (inclusive)."""
    c = lambda z: np.r_[0.0, np.cumsum(z)]           # noqa: E731
    S1, Sx, Sy, Sxx, Sxy, Syy = c(np.ones_like(x)), c(x), c(y), c(x * x), c(x * y), c(y * y)

    def sse(i, j):
        n = S1[j + 1] - S1[i]
        sx, sy = Sx[j + 1] - Sx[i], Sy[j + 1] - Sy[i]
        sxx, sxy, syy = Sxx[j + 1] - Sxx[i], Sxy[j + 1] - Sxy[i], Syy[j + 1] - Syy[i]
        vx = sxx - sx * sx / n
        r = syy - sy * sy / n - ((sxy - sx * sy / n) ** 2 / vx if vx > 1e-15 else 0.0)
        return max(r, 0.0)
    return sse


def fit_segments(x: np.ndarray, y: np.ndarray, k: int, min_pts: int = 3) -> List[Tuple[int, int, float, float]]:
    """Optimal piecewise-linear segmentation of (x, y) into ``k`` least-squares lines (dynamic programming)."""
    n = len(x)
    k = max(1, min(k, n // min_pts))
    sse = _segment_sse(x, y)
    INF = np.inf
    cost = np.full((k + 1, n + 1), INF)            # cost[s][j]: best SSE of points 0..j-1 with s segments
    arg = np.zeros((k + 1, n + 1), int)
    cost[0][0] = 0.0
    for s in range(1, k + 1):
        for j in range(s * min_pts, n + 1):
            best, bi = INF, -1
            for i in range((s - 1) * min_pts, j - min_pts + 1):
                if cost[s - 1][i] < INF:
                    c = cost[s - 1][i] + sse(i, j - 1)
                    if c < best:
                        best, bi = c, i
            cost[s][j], arg[s][j] = best, bi
    bounds, j = [], n
    for s in range(k, 0, -1):
        i = arg[s][j]
        bounds.append((i, j - 1))
        j = i
    segs = []
    for i0, i1 in reversed(bounds):
        slope, icpt = np.polyfit(x[i0:i1 + 1], y[i0:i1 + 1], 1) if i1 > i0 else (0.0, float(y[i0]))
        segs.append((int(i0), int(i1), float(slope), float(icpt)))
    return segs


def ca_curve(array, cell_area: float = 1.0, n_points: int = 120, n_segments: int = 3) -> CACurve:
    values = np.sort(valid_values(array))
    if values.size < 10:
        raise ValueError(tr("Too few valid cells for a C-A plot."))
    shift = 0.0
    if values[0] <= 0:                              # log10 needs positive values
        rng = values[-1] - values[0]
        shift = -values[0] + (rng * 1e-3 if rng > 0 else 1.0)
    t = _thresholds(values, n_points)
    t = t[t < values[-1]] if (t < values[-1]).sum() >= 10 else t      # drop the single top value (A = 1 cell)
    count = values.size - np.searchsorted(values, t, side="left")
    area = count * cell_area
    log_v, log_a = np.log10(t + shift), np.log10(area)
    segs = fit_segments(log_v, log_a, n_segments)
    breaks = [float(t[s[0]]) for s in segs[1:]]
    return CACurve(t, area, log_v, log_a, shift, segs, breaks)


# ============================================================================== binary map + statistics
def binary_target(array, threshold: float) -> np.ma.MaskedArray:
    """1 where value >= threshold, 0 elsewhere on valid cells; invalid cells masked (written as nodata 255)."""
    a = np.ma.asarray(array)
    data = np.ma.getdata(a).astype(float)
    mask = np.ma.getmaskarray(a) | ~np.isfinite(data)
    out = np.where(data >= threshold, 1, 0).astype("uint8")
    return np.ma.array(out, mask=mask)


def target_stats(array, threshold: float, cell_area_km2: Optional[float] = None,
                 dep_rows: Optional[Sequence[int]] = None, dep_cols: Optional[Sequence[int]] = None) -> Dict[str, float]:
    a = np.ma.asarray(array)
    data = np.ma.getdata(a).astype(float)
    valid = ~np.ma.getmaskarray(a) & np.isfinite(data)
    target = valid & (data >= threshold)
    n_valid, n_target = int(valid.sum()), int(target.sum())
    st = dict(threshold=float(threshold), target_cells=n_target, valid_cells=n_valid,
              target_area_pct=100.0 * n_target / max(1, n_valid))
    if cell_area_km2:
        st["target_area_km2"] = n_target * cell_area_km2
    if dep_rows is not None and len(dep_rows):
        r, c = np.asarray(dep_rows, int), np.asarray(dep_cols, int)
        ok = valid[r, c]
        hit = int(target[r[ok], c[ok]].sum())
        st.update(deposits=int(ok.sum()), deposits_in_target=hit, deposits_in_target_pct=100.0 * hit / max(1, ok.sum()))
    return st


def target_class_table() -> pd.DataFrame:
    """Class table for the binary map: 1 = red, 0 = transparent (sidecar .classes.csv keeps the colours)."""
    return pd.DataFrame(dict(value=[0, 1], label=["non-target", "target"], color=[TRANSPARENT, TARGET_COLOR]))
