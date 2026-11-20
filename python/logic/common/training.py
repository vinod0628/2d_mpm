"""Training-data helpers shared by the data-driven (and hybrid) methods.

Point selection follows the ArcSDM site-reduction / sampling tools (github.com/gtkfi/ArcSDM, master):

* :func:`split_deposits` — train/test split. ``"random"`` = ArcSDM *Splitting* (shuffle, first x % train);
  ``"spatial"`` (default) = spatially stratified: deposits are ordered along a Hilbert curve and one test deposit is
  drawn at random from each run of ~1/f consecutive deposits, so test (and training) deposits cover the study area
  evenly (spatially balanced sampling; Stevens & Olsen 2004).
* :func:`thin_points` — ArcSDM *Thinning*: keep a point only if it is ≥ ``min_distance`` from every kept point
  (input order); ArcSDM derives the distance from a unit area as √(A/π).
* :func:`background_candidates` / :func:`sample_background` — ArcSDM *Create Random Points*: non-deposit samples
  inside the study area where all evidence has data, outside a buffer around the known deposits
  (Carranza et al. 2008), at least ``min_spacing`` apart (ArcSDM ``minimum_allowed_distance``; dart throwing).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

SPLIT_METHODS = ("spatial", "random")


# ============================================================================== spatial ordering
def hilbert_index(rows, cols, order: int = 16) -> np.ndarray:
    """Position of cells (rows, cols < 2**order) along a Hilbert curve (Skilling's / Wikipedia xy2d)."""
    x = np.asarray(cols, np.int64).copy()
    y = np.asarray(rows, np.int64).copy()
    d = np.zeros(x.shape, np.int64)
    full = 1 << order
    s = full >> 1
    while s > 0:
        rx = ((x & s) > 0).astype(np.int64)
        ry = ((y & s) > 0).astype(np.int64)
        d += s * s * ((3 * rx) ^ ry)
        flip = ry == 0                                  # rotate the quadrant
        swap_x = flip & (rx == 1)
        x = np.where(swap_x, full - 1 - x, x)
        y = np.where(swap_x, full - 1 - y, y)
        x, y = np.where(flip, y, x), np.where(flip, x, y)
        s >>= 1
    return d


# ============================================================================== deposits
def thin_points(rows, cols, min_distance: float, res: Tuple[float, float] = (1.0, 1.0)) -> np.ndarray:
    """ArcSDM *Thinning*: boolean keep-mask so that no two kept points are closer than ``min_distance`` (map units,
    distance between cell centres; ``res`` = (x size, y size)). Points are visited in input order."""
    rows, cols = np.asarray(rows, float), np.asarray(cols, float)
    keep = np.zeros(rows.size, bool)
    if min_distance <= 0:
        keep[:] = True
        return keep
    xs, ys = cols * res[0], rows * res[1]
    kx, ky = [], []
    for i in range(rows.size):
        if not kx or np.min(np.hypot(np.array(kx) - xs[i], np.array(ky) - ys[i])) >= min_distance:
            keep[i] = True
            kx.append(xs[i])
            ky.append(ys[i])
    return keep


def thin_deposits(rows, cols, min_distance: float, res: Tuple[float, float]):
    """:func:`thin_points` applied to deposit cells; returns (rows, cols, messages)."""
    rows, cols = np.asarray(rows, int), np.asarray(cols, int)
    if not min_distance or min_distance <= 0:
        return rows, cols, []
    keep = thin_points(rows, cols, min_distance, res)
    msg = [f"thinning (≥ {min_distance:g} map units apart) kept {int(keep.sum())} of {keep.size} deposits"]
    return rows[keep], cols[keep], msg


def split_deposits(rows, cols, test_fraction: float = 0.0, seed: int = 0, method: str = "spatial"
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Train/test split of deposit cells. Returns (train_rows, train_cols, test_rows, test_cols).

    method "random": ArcSDM Splitting (random permutation). "spatial": spatially stratified — along a Hilbert curve
    the deposits are cut into ``n_test`` consecutive runs and one random deposit of each run is held out.
    """
    if method not in SPLIT_METHODS:
        raise ValueError(f"method must be one of {SPLIT_METHODS}")
    rows, cols = np.asarray(rows, int), np.asarray(cols, int)
    n = len(rows)
    n_test = int(round(n * test_fraction))
    if n_test <= 0:
        return rows, cols, rows[:0], cols[:0]
    if n - n_test < 2:
        raise ValueError("Too few deposits left for training after the test split.")
    rng = np.random.default_rng(seed)
    if method == "random":
        perm = rng.permutation(n)
        te = perm[:n_test]
    else:
        order = np.lexsort((rng.random(n), hilbert_index(rows, cols)))   # ties (same cell) in random order
        bounds = np.linspace(0, n, n_test + 1).round().astype(int)
        te = np.array([order[rng.integers(a, b)] for a, b in zip(bounds[:-1], bounds[1:])], int)
    is_test = np.zeros(n, bool)
    is_test[te] = True
    return rows[~is_test], cols[~is_test], rows[is_test], cols[is_test]


def split_point_layers(gdf, grid, train_rows, train_cols, test_rows=None, test_cols=None):
    """The original deposit points (geometry + attributes) that were used for training and for testing.

    Each point goes to the set of its cell (several points in one cell share it); points whose cell is in neither
    set (outside the study area, removed by thinning) are left out. Returns (training_gdf, test_gdf) with a
    ``split`` column, in the CRS of ``gdf``.
    """
    import geopandas as gpd
    pts = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    xy = pts.to_crs(grid.crs) if (pts.crs is not None and grid.crs is not None and pts.crs != grid.crs) else pts
    if len(pts):
        r, c = grid.xy_to_rowcol([g.centroid.x for g in xy.geometry], [g.centroid.y for g in xy.geometry])
        key = np.asarray(r, np.int64) * grid.width + np.asarray(c, np.int64)
    else:
        key = np.zeros(0, np.int64)

    def pick(rows, cols, label):
        if rows is None or len(rows) == 0:
            return gpd.GeoDataFrame(pts.iloc[:0].assign(split=[]), geometry=pts.geometry.name, crs=pts.crs)
        cells = np.asarray(rows, np.int64) * grid.width + np.asarray(cols, np.int64)
        sel = pts[np.isin(key, cells)].copy()
        sel["split"] = label
        return gpd.GeoDataFrame(sel, geometry=pts.geometry.name, crs=pts.crs).reset_index(drop=True)

    return pick(train_rows, train_cols, "training"), pick(test_rows, test_cols, "test")


TRAIN_STYLE = dict(color="#43a047", edgecolor="#ffffff", markersize=34, marker="o")
TEST_STYLE = dict(color="#29b6f6", edgecolor="#102027", markersize=44, marker="^")


def deposit_point_outputs(gdf, grid, train_rows, train_cols, test_rows=None, test_cols=None) -> list:
    """Method outputs ``training_points`` (+ ``test_points`` when deposits were held out): the deposit points of
    each set, written to the run folder and added to the map."""
    from core.methods import OutputVector
    from i18n import tr
    train, test = split_point_layers(gdf, grid, train_rows, train_cols, test_rows, test_cols)
    out = [OutputVector("training_points", train, tr("Deposits used for training ({n} points)", n=len(train)),
                        dict(TRAIN_STYLE))]
    if len(test):
        out.append(OutputVector("test_points", test, tr("Deposits held out for testing ({n} points)", n=len(test)),
                                dict(TEST_STYLE)))
    return out


# ============================================================================== non-deposit samples
def background_candidates(study: np.ndarray, dep_rows, dep_cols, min_distance: float = 0.0,
                          res: Tuple[float, float] = (1.0, 1.0)) -> np.ndarray:
    """Cells that may serve as non-deposit (negative) training samples.

    Study-area cells without a known deposit and at least ``min_distance`` map units (cell-centre distance)
    from every deposit cell — Carranza et al. (2008): negatives "sufficiently far" from known deposits;
    ArcSDM Create Random Points: outside a buffer around the deposits. ``res`` = (x size, y size) of a cell.
    """
    study = np.asarray(study, bool)
    dep = np.zeros(study.shape, bool)
    if len(dep_rows):
        dep[np.asarray(dep_rows, int), np.asarray(dep_cols, int)] = True
    cand = study & ~dep
    if min_distance > 0 and dep.any():
        from scipy import ndimage
        dist = ndimage.distance_transform_edt(~dep, sampling=(float(res[1]), float(res[0])))
        cand &= dist >= min_distance
    return cand


def auto_spacing(n_candidates: int, n: int, res: Tuple[float, float] = (1.0, 1.0)) -> float:
    """Default spacing ½·√(candidate area / n): even coverage that still leaves room for all n points."""
    if n <= 0:
        return 0.0
    return 0.5 * float(np.sqrt(n_candidates * res[0] * res[1] / n))


def sample_positions(cand_pos: np.ndarray, rows: np.ndarray, cols: np.ndarray, n: int, rng,
                     min_spacing: float = 0.0, res: Tuple[float, float] = (1.0, 1.0)) -> np.ndarray:
    """``n`` random entries of ``cand_pos`` (positions whose cell is rows[i], cols[i]) at least ``min_spacing`` map
    units apart — dart throwing in random order with a background grid (ArcSDM ``minimum_allowed_distance``)."""
    cand_pos = np.asarray(cand_pos)
    if cand_pos.size < n:
        raise ValueError(f"Only {cand_pos.size} candidate non-deposit cells for {n} samples — lower the number of "
                         "samples or the minimum distance from deposits.")
    if min_spacing <= 0:
        return np.sort(rng.choice(cand_pos, size=int(n), replace=False))
    xs, ys = cols * float(res[0]), rows * float(res[1])
    cell = min_spacing / np.sqrt(2.0)                    # at most one accepted point per background-grid cell
    grid = {}
    picked = []
    r2 = min_spacing * min_spacing
    for i in rng.permutation(cand_pos.size):
        x, y = xs[i], ys[i]
        gx, gy = int(x // cell), int(y // cell)
        ok = True
        for dx in (-2, -1, 0, 1, 2):
            for dy in (-2, -1, 0, 1, 2):
                j = grid.get((gx + dx, gy + dy))
                if j is not None and (xs[j] - x) ** 2 + (ys[j] - y) ** 2 < r2:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            grid[(gx, gy)] = i
            picked.append(i)
            if len(picked) == n:
                break
    if len(picked) < n:
        raise ValueError(f"Only {len(picked)} of {n} non-deposit samples fit with a spacing of {min_spacing:g} "
                         "map units — lower the spacing or the number of samples.")
    return np.sort(cand_pos[np.array(picked)])


def sample_background(candidates: np.ndarray, n: int, seed: int = 0, min_spacing: float = 0.0,
                      res: Tuple[float, float] = (1.0, 1.0)) -> Tuple[np.ndarray, np.ndarray]:
    """``n`` random candidate cells (see :func:`background_candidates`), at least ``min_spacing`` apart."""
    candidates = np.asarray(candidates, bool)
    r, c = np.nonzero(candidates)
    pick = sample_positions(np.arange(r.size), r, c, n, np.random.default_rng(seed), min_spacing, res)
    return r[pick], c[pick]
