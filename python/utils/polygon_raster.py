"""Polygon to categorical raster (cf. ArcGIS *Polygon to Raster*, GDAL *rasterize*) for class-based MPM evidence.

Every distinct value of a label attribute becomes one class code; the class table (value, label, count and the
attributes that are constant within each class) is written next to the GeoTIFF. Cells are assigned by

* ``center``   — the polygon containing the cell centre (ArcGIS CELL_CENTER; with overlaps the last feature wins);
* ``max_area`` — the polygon covering the largest part of the cell (ArcGIS MAXIMUM_AREA), estimated on
  ``subcells`` × ``subcells`` sub-cell centres; a cell touched by no polygon stays a gap.

Cells of the study area that no polygon covers (gaps) become one extra class with its own label, or NoData.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
from affine import Affine
from rasterio.features import rasterize

from utils.raster_tools import Grid

CODE_MODES = ("sequential", "field")
ASSIGNMENTS = ("center", "max_area")


@dataclass
class PolygonRasterResult:
    data: np.ma.MaskedArray            # class codes; masked = NoData (outside the study area / grid gaps)
    table: pd.DataFrame                # value, label, count (+ constant attributes)
    nodata: int
    gap_code: Optional[int]            # class code of the gaps (None: gaps are NoData)
    skipped: int = 0                   # features without a label or geometry
    empty: List[str] = dc_field(default_factory=list)    # labels without any cell (polygons smaller than a cell)


def _sorted_labels(values: pd.Series) -> list:
    u = pd.unique(values)
    try:
        return sorted(u)
    except TypeError:
        return sorted(u, key=lambda x: str(x))


def class_codes(labels: pd.Series, mode: str = "sequential") -> dict:
    """{label value: class code}. ``sequential``: 1 … K in sorted label order; ``field``: the label values
    themselves (whole numbers)."""
    if mode not in CODE_MODES:
        raise ValueError(f"unknown code mode {mode!r}")
    uniq = _sorted_labels(labels.dropna())
    if mode == "sequential":
        return {v: i + 1 for i, v in enumerate(uniq)}
    try:
        vals = np.asarray(uniq, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("Class codes from the field need numeric values.") from None
    if not np.all(np.isfinite(vals)) or not np.all(vals == np.round(vals)):
        raise ValueError("Class codes from the field need whole numbers.")
    return {v: int(round(float(v))) for v in uniq}


def _label_text(v) -> str:
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    return str(v)


def _burn_center(shapes, grid: Grid) -> np.ndarray:
    return rasterize(shapes, out_shape=grid.shape, transform=grid.transform, fill=0, dtype="int32")


def _burn_max_area(shapes, grid: Grid, subcells: int, block_rows: int,
                   progress: Optional[Callable] = None) -> np.ndarray:
    """Code of the polygon covering most sub-cell centres of every cell (block by block: bounded memory)."""
    k = int(subcells)
    H, W = grid.shape
    out = np.zeros((H, W), dtype="int32")
    for r0 in range(0, H, block_rows):
        bh = min(block_rows, H - r0)
        t = grid.transform * Affine.translation(0, r0) * Affine.scale(1.0 / k)
        fine = rasterize(shapes, out_shape=(bh * k, W * k), transform=t, fill=0, dtype="int32")
        sub = fine.reshape(bh, k, W, k).transpose(0, 2, 1, 3).reshape(bh, W, k * k)
        best = np.zeros((bh, W), dtype="int32")
        best_n = np.zeros((bh, W), dtype="int32")
        for c in np.unique(sub):
            if c == 0:
                continue
            n = (sub == c).sum(-1, dtype="int32")
            upd = n > best_n                                   # ties: the lower code (first in sort order)
            best[upd], best_n[upd] = c, n[upd]
        out[r0:r0 + bh] = best
        if progress:
            progress(min(1.0, (r0 + bh) / H))
    return out


def polygons_to_raster(gdf, grid: Grid, field: Optional[str] = None, codes: str = "sequential",
                       assignment: str = "center", study: Optional[np.ndarray] = None,
                       gap_label: Optional[str] = "Other", single_label: str = "polygon",
                       copy_attributes: bool = True, subcells: int = 4, block_rows: int = 256,
                       nodata: int = -9999, progress: Optional[Callable] = None) -> PolygonRasterResult:
    """Burn polygons into ``grid`` as class codes.

    field          : label attribute (one class per distinct value); None = one class ``single_label``.
    codes          : "sequential" (1 … K, labels sorted) or "field" (the whole-number values of ``field``).
    assignment     : "center" or "max_area" (see the module docstring).
    study          : boolean array (grid shape), True inside the study area; None = the whole grid.
    gap_label      : label of the class given to study-area cells outside every polygon; None = NoData.
    copy_attributes: add the attributes that have one value within every class to the class table.
    nodata         : NoData when a class code is ≤ 0 (otherwise 0 is used, codes start at 1).
    """
    if assignment not in ASSIGNMENTS:
        raise ValueError(f"unknown assignment {assignment!r}")
    if study is not None and study.shape != grid.shape:
        raise ValueError("The study-area mask does not match the grid.")
    if gdf.crs is not None and grid.crs is not None and gdf.crs != grid.crs:
        gdf = gdf.to_crs(grid.crs)
    geom = gdf.geometry
    ok = geom.notna() & ~geom.is_empty & geom.geom_type.isin(["Polygon", "MultiPolygon"])
    if field is None:
        labels = pd.Series(single_label, index=gdf.index, dtype=object)
    else:
        labels = gdf[field]
        ok &= labels.notna()
    if not ok.any():
        raise ValueError("The layer has no polygons with a label.")
    skipped = int((~ok).sum())
    g, labels = gdf[ok], labels[ok]
    cmap = class_codes(labels, codes if field is not None else "sequential")
    order = list(cmap)                                       # labels in sort order
    internal = {v: i + 1 for i, v in enumerate(order)}       # burn 1 … K (0 = no polygon), map to codes after
    shapes = [(geo, internal[v]) for geo, v in zip(g.geometry, labels)]
    if assignment == "center":
        burned = _burn_center(shapes, grid)
        if progress:
            progress(1.0)
    else:
        burned = _burn_max_area(shapes, grid, subcells, block_rows, progress)

    inside = np.ones(grid.shape, bool) if study is None else np.asarray(study, bool)
    final = [cmap[v] for v in order]
    gap_code = None
    if gap_label is not None:
        gap_code = max(final) + 1
        burned = np.where((burned == 0) & inside, len(order) + 1, burned)
    valid = inside & (burned != 0)
    if not valid.any():
        raise ValueError("No polygon falls inside the output grid / study area.")
    all_codes = final + ([gap_code] if gap_code is not None else [])
    lo, hi = min(all_codes), max(all_codes)
    if lo >= 1:
        nd = 0
        dtype = "uint8" if hi <= 255 else "uint16" if hi <= 65535 else "int32"
    else:
        nd = int(nodata)
        while nd in all_codes:
            nd -= 1
        dtype = "int32"
    lut = np.array([nd] + all_codes, dtype="int64")
    data = np.ma.array(lut[np.where(valid, burned, 0)].astype(dtype), mask=~valid)

    n_int = np.bincount(burned[valid].ravel(), minlength=len(all_codes) + 1)
    rows, empty = [], []
    for i, v in enumerate(order, start=1):
        if n_int[i] == 0:
            empty.append(_label_text(v))
            continue
        rows.append(dict(value=cmap[v], label=_label_text(v), count=int(n_int[i])))
    table = pd.DataFrame(rows, columns=["value", "label", "count"])
    if copy_attributes and field is not None and len(table):
        label_of = {cmap[v]: v for v in order}
        grouped = g.groupby(labels, sort=False)
        for col in [c for c in g.columns if c not in (field, g.geometry.name)]:
            nun = grouped[col].nunique(dropna=False)
            if len(nun) and (nun <= 1).all():
                first = grouped[col].first()
                table[str(col)] = [first.get(label_of[c]) for c in table["value"]]
    n_gap = int(n_int[len(order) + 1]) if gap_code is not None else 0
    if n_gap:
        gap_row = {k: None for k in table.columns}
        gap_row.update(value=gap_code, label=str(gap_label), count=n_gap)
        table = pd.concat([table, pd.DataFrame([gap_row])], ignore_index=True) if len(table) else             pd.DataFrame([gap_row])
    table = table.sort_values("value", kind="stable").reset_index(drop=True)
    return PolygonRasterResult(data, table, nd, gap_code if n_gap else None, skipped, empty)


def snapped_grid(bounds, cell: float, crs=None, origin=None) -> Grid:
    """Grid covering ``bounds`` (left, bottom, right, top) with cells of ``cell``; the cell lines go through
    ``origin`` (x, y) — e.g. the project grid corner — else through multiples of ``cell``."""
    from rasterio.crs import CRS
    from rasterio.transform import from_origin
    left, bottom, right, top = bounds
    ox, oy = origin if origin is not None else (0.0, 0.0)
    eps = 1e-9
    l = ox + np.floor((left - ox) / cell + eps) * cell
    r = ox + np.ceil((right - ox) / cell - eps) * cell
    b = oy + np.floor((bottom - oy) / cell + eps) * cell
    t = oy + np.ceil((top - oy) / cell - eps) * cell
    w, h = max(1, int(round((r - l) / cell))), max(1, int(round((t - b) / cell)))
    c = CRS.from_user_input(crs) if crs is not None and not isinstance(crs, CRS) else crs
    return Grid(from_origin(l, t, cell, cell), c, w, h)
