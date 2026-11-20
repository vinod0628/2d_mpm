"""Raster helpers shared by the GUI, the data-preparation scripts and the MPM methods.

Conventions
-----------
* Arrays are numpy ``MaskedArray`` objects; the mask marks nodata / outside-study-area cells.
* A :class:`Grid` (transform + crs + shape) describes where an array lives. Methods that combine
  several layers require them to share one grid (see :func:`align_to_grid`).
* Class labels of categorical rasters live in a sidecar CSV ``<raster>.classes.csv`` with at least the
  columns ``value`` and ``label`` (see :func:`read_class_table` / :func:`write_class_table`).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import Affine, from_origin, rowcol
from rasterio.warp import reproject
from scipy import ndimage
from scipy.spatial import cKDTree


# --------------------------------------------------------------------------------------------- grid
@dataclass(frozen=True)
class Grid:
    """Raster grid definition (north-up)."""

    transform: Affine
    crs: Optional[CRS]
    width: int
    height: int

    @property
    def shape(self) -> Tuple[int, int]:
        return self.height, self.width

    @property
    def res(self) -> Tuple[float, float]:
        return abs(self.transform.a), abs(self.transform.e)

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """(left, bottom, right, top)."""
        left, top = self.transform.c, self.transform.f
        right = left + self.transform.a * self.width
        bottom = top + self.transform.e * self.height
        return left, bottom, right, top

    @property
    def extent(self) -> Tuple[float, float, float, float]:
        """matplotlib imshow extent (left, right, bottom, top)."""
        left, bottom, right, top = self.bounds
        return left, right, bottom, top

    @property
    def cell_area(self) -> float:
        return abs(self.transform.a * self.transform.e)

    @property
    def is_geographic(self) -> bool:
        return bool(self.crs is not None and self.crs.is_geographic)

    @classmethod
    def from_dataset(cls, ds) -> "Grid":
        return cls(ds.transform, ds.crs, ds.width, ds.height)

    @classmethod
    def from_file(cls, path: str) -> "Grid":
        with rasterio.open(path) as ds:
            return cls.from_dataset(ds)

    @classmethod
    def from_bounds(cls, left, bottom, right, top, res: float, crs=None, snap: bool = True) -> "Grid":
        if snap:
            left, bottom = np.floor(left / res) * res, np.floor(bottom / res) * res
            right, top = np.ceil(right / res) * res, np.ceil(top / res) * res
        width, height = int(round((right - left) / res)), int(round((top - bottom) / res))
        return cls(from_origin(left, top, res, res), CRS.from_user_input(crs) if crs else None, width, height)

    def same_as(self, other: "Grid", tol: float = 1e-6) -> bool:
        if self.shape != other.shape:
            return False
        if not all(abs(a - b) <= tol * max(1.0, abs(a)) for a, b in zip(self.transform[:6], other.transform[:6])):
            return False
        if self.crs is None or other.crs is None:
            return self.crs is None and other.crs is None
        return self.crs == other.crs

    def xy_to_rowcol(self, xs, ys) -> Tuple[np.ndarray, np.ndarray]:
        rows, cols = rowcol(self.transform, np.asarray(xs, float), np.asarray(ys, float))
        return np.asarray(rows, int), np.asarray(cols, int)

    def cell_centers(self) -> Tuple[np.ndarray, np.ndarray]:
        cols = np.arange(self.width) + 0.5
        rows = np.arange(self.height) + 0.5
        xs = self.transform.c + cols * self.transform.a
        ys = self.transform.f + rows * self.transform.e
        return np.meshgrid(xs, ys)


# --------------------------------------------------------------------------------------------- I/O
def read_raster(path: str, band: int = 1, as_float: bool = False) -> Tuple[np.ma.MaskedArray, Grid, dict]:
    """Read one band as a masked array. Returns (array, grid, profile)."""
    with rasterio.open(path) as ds:
        arr = ds.read(band, masked=True)
        if ds.nodata is not None and np.issubdtype(arr.dtype, np.floating):
            arr = np.ma.masked_invalid(arr)
        profile = dict(ds.profile)
        grid = Grid.from_dataset(ds)
    if as_float:
        arr = arr.astype("float64")
    return arr, grid, profile


def nodata_dtype(dtype, value) -> np.dtype:
    """Data type that can hold both the raster values (``dtype``) and the NoData ``value`` (e.g. 8-bit classes
    with NoData -9999 -> int16)."""
    dt = np.dtype(dtype)
    if dt.kind == "f" or value is None:
        return dt
    if dt.kind == "b":
        dt = np.dtype("uint8")
    value = float(value)
    if value.is_integer():
        for cand in (dt, np.dtype("int16"), np.dtype("int32"), np.dtype("int64")):
            info = np.iinfo(cand)
            if np.can_cast(dt, cand) and info.min <= value <= info.max:
                return cand
    return np.dtype("float32") if dt.itemsize <= 2 else np.dtype("float64")


def default_nodata(dtype) -> float:
    dt = np.dtype(dtype)
    if dt == np.uint8:
        return 255
    if dt == np.uint16:
        return 65535
    if dt.kind in "iu":
        return int(np.iinfo(dt).min)
    return -9999.0


def write_raster(path: str, array, grid: Grid, nodata=None, dtype=None, compress: str = "LZW",
                 class_table: Optional[pd.DataFrame] = None) -> str:
    """Write a (masked) 2-D array to GeoTIFF. Masked cells are written as ``nodata``."""
    arr = np.ma.asarray(array)
    dtype = np.dtype(dtype or arr.dtype)
    if dtype == np.float64:
        dtype = np.dtype("float32")
    if dtype == np.bool_:
        dtype = np.dtype("uint8")
    if nodata is None:
        nodata = default_nodata(dtype)
    dtype = nodata_dtype(dtype, nodata)              # e.g. 8-bit classes with NoData -9999: written as int16
    data = np.ma.filled(arr.astype(dtype), nodata)
    profile = dict(driver="GTiff", width=grid.width, height=grid.height, count=1, dtype=dtype.name,
                   crs=grid.crs, transform=grid.transform, nodata=nodata)
    if compress:
        profile["compress"] = compress
    if grid.width >= 256 and grid.height >= 256:
        profile.update(tiled=True, blockxsize=256, blockysize=256)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)
    if class_table is not None:
        write_class_table(path, class_table)
    return path


def class_table_path(raster_path: str) -> str:
    return os.path.splitext(raster_path)[0] + ".classes.csv"


def read_class_table(raster_path: str) -> Optional[pd.DataFrame]:
    p = class_table_path(raster_path)
    if not os.path.isfile(p):
        return None
    df = pd.read_csv(p)
    if "value" not in df.columns:
        return None
    if "label" not in df.columns:
        df["label"] = df["value"].astype(str)
    return df


def write_class_table(raster_path: str, table: pd.DataFrame) -> str:
    p = class_table_path(raster_path)
    table.to_csv(p, index=False)
    return p


# --------------------------------------------------------------------------------------------- gridding
def align_to_grid(array, src_grid: Grid, dst_grid: Grid, resampling: str = "nearest",
                  src_nodata=None) -> np.ma.MaskedArray:
    """Reproject/resample a (masked) array onto ``dst_grid``."""
    src = np.ma.asarray(array)
    src_data = np.ma.filled(src.astype("float64"), np.nan)
    if src_nodata is not None:
        src_data[src_data == src_nodata] = np.nan
    dst = np.full(dst_grid.shape, np.nan, dtype="float64")
    reproject(source=src_data, destination=dst,
              src_transform=src_grid.transform, src_crs=src_grid.crs or dst_grid.crs, src_nodata=np.nan,
              dst_transform=dst_grid.transform, dst_crs=dst_grid.crs or src_grid.crs, dst_nodata=np.nan,
              resampling=getattr(Resampling, resampling))
    out = np.ma.masked_invalid(dst)
    if resampling == "nearest" and not np.issubdtype(src.dtype, np.floating):
        out = np.ma.array(np.ma.filled(out, 0).astype(src.dtype), mask=np.ma.getmaskarray(out))
    return out


def rasterize_vector(gdf, grid: Grid, field: Optional[str] = None, fill=0, all_touched: bool = False,
                     dtype="float32") -> np.ndarray:
    """Burn vector features into ``grid``. ``field`` = attribute to burn, else 1."""
    if gdf.crs is not None and grid.crs is not None and gdf.crs != grid.crs:
        gdf = gdf.to_crs(grid.crs)
    geoms = [g for g in gdf.geometry if g is not None and not g.is_empty]
    if field is None:
        shapes = ((g, 1) for g in geoms)
    else:
        shapes = ((g, v) for g, v in zip(gdf.geometry, gdf[field]) if g is not None and not g.is_empty)
    return rasterize(shapes, out_shape=grid.shape, transform=grid.transform, fill=fill,
                     all_touched=all_touched, dtype=dtype)


def clip_by_mask(array, grid: Grid, gdf, crop: bool = True, invert: bool = False, all_touched: bool = False
                 ) -> Tuple[np.ma.MaskedArray, Grid]:
    """Clip a raster with polygon features (like GDAL -cutline / ArcGIS Extract by Mask).

    crop        : shrink the output grid to the mask extent (cells stay on the input grid, no resampling).
                  Ignored when ``invert`` is True.
    invert      : keep the cells outside the polygons instead of inside.
    all_touched : a cell counts as inside when the polygon touches it, not only when it covers the cell centre.
    Cells outside the kept area become masked (nodata). The mask is reprojected to the raster CRS if needed.
    """
    if gdf.crs is not None and grid.crs is not None and gdf.crs != grid.crs:
        gdf = gdf.to_crs(grid.crs)
    geoms = [g for g in gdf.geometry
             if g is not None and not g.is_empty and g.geom_type in ("Polygon", "MultiPolygon")]
    if not geoms:
        raise ValueError("The mask layer has no polygon features.")
    arr = np.ma.asarray(array)
    out_grid = grid
    if crop and not invert:
        minx = min(g.bounds[0] for g in geoms)
        miny = min(g.bounds[1] for g in geoms)
        maxx = max(g.bounds[2] for g in geoms)
        maxy = max(g.bounds[3] for g in geoms)
        t = grid.transform
        rx, ry = abs(t.a), abs(t.e)
        eps = 1e-9
        c0 = max(0, int(np.floor((minx - t.c) / rx + eps)))
        c1 = min(grid.width, int(np.ceil((maxx - t.c) / rx - eps)))
        r0 = max(0, int(np.floor((t.f - maxy) / ry + eps)))
        r1 = min(grid.height, int(np.ceil((t.f - miny) / ry - eps)))
        if c1 <= c0 or r1 <= r0:
            raise ValueError("The mask does not overlap the raster.")
        arr = arr[r0:r1, c0:c1]
        out_grid = Grid(t @ Affine.translation(c0, r0), grid.crs, c1 - c0, r1 - r0)
    inside = rasterize(((g, 1) for g in geoms), out_shape=out_grid.shape, transform=out_grid.transform, fill=0,
                       all_touched=all_touched, dtype="uint8").astype(bool)
    keep = ~inside if invert else inside
    out = np.ma.array(np.ma.getdata(arr).copy(), mask=np.ma.getmaskarray(arr) | ~keep)
    if out.mask.all():
        raise ValueError("The clipped raster has no valid cells (the mask does not overlap the raster data).")
    return out, out_grid


def distance_to_features(gdf, grid: Grid, mode: str = "feature", max_distance: Optional[float] = None
                         ) -> np.ndarray:
    """Euclidean distance (map units) from every cell centre to the nearest feature.

    mode = "feature": polygons count as 0 inside (distance to the polygon area);
           "boundary": distance to polygon outlines / lines.
    """
    if gdf.crs is not None and grid.crs is not None and gdf.crs != grid.crs:
        gdf = gdf.to_crs(grid.crs)
    geoms = gdf.geometry
    if mode == "boundary":
        geoms = geoms.boundary
    burned = rasterize(((g, 1) for g in geoms if g is not None and not g.is_empty), out_shape=grid.shape,
                       transform=grid.transform, fill=0, all_touched=True, dtype="uint8")
    if not burned.any():
        raise ValueError("No features fall inside the grid.")
    rx, ry = grid.res
    dist = ndimage.distance_transform_edt(burned == 0, sampling=(ry, rx)).astype("float32")
    if max_distance is not None:
        dist = np.minimum(dist, max_distance)
    return dist


def idw_interpolate(xs, ys, values, grid: Grid, power: float = 2.0, k: int = 12,
                    max_distance: Optional[float] = None, mask: Optional[np.ndarray] = None
                    ) -> np.ma.MaskedArray:
    """Inverse-distance-weighted interpolation of point values onto ``grid`` (k nearest samples)."""
    xs, ys, values = (np.asarray(a, float) for a in (xs, ys, values))
    ok = np.isfinite(values) & np.isfinite(xs) & np.isfinite(ys)
    xs, ys, values = xs[ok], ys[ok], values[ok]
    tree = cKDTree(np.column_stack([xs, ys]))
    gx, gy = grid.cell_centers()
    sel = np.ones(grid.shape, bool) if mask is None else np.asarray(mask, bool)
    q = np.column_stack([gx[sel], gy[sel]])
    k = min(k, len(values))
    d, idx = tree.query(q, k=k, distance_upper_bound=np.inf if max_distance is None else max_distance)
    if k == 1:
        d, idx = d[:, None], idx[:, None]
    valid = np.isfinite(d)
    d = np.where(valid, np.maximum(d, 1e-9), np.inf)
    w = np.where(valid, 1.0 / d ** power, 0.0)
    vals = np.where(valid, values[np.minimum(idx, len(values) - 1)], 0.0)
    wsum = w.sum(axis=1)
    est = np.where(wsum > 0, (w * vals).sum(axis=1) / np.where(wsum > 0, wsum, 1), np.nan)
    out = np.full(grid.shape, np.nan)
    out[sel] = est
    return np.ma.masked_invalid(out)


def points_to_cells(gdf, grid: Grid, valid_mask: Optional[np.ndarray] = None, unique: bool = True
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Row/col indices of point features inside ``grid`` (and inside ``valid_mask`` if given)."""
    if gdf.crs is not None and grid.crs is not None and gdf.crs != grid.crs:
        gdf = gdf.to_crs(grid.crs)
    pts = gdf.geometry[gdf.geometry.notna()]
    pts = pts[~pts.is_empty]
    if len(pts) == 0:
        return np.array([], int), np.array([], int)
    xs = np.array([p.centroid.x for p in pts])
    ys = np.array([p.centroid.y for p in pts])
    rows, cols = grid.xy_to_rowcol(xs, ys)
    inside = (rows >= 0) & (rows < grid.height) & (cols >= 0) & (cols < grid.width)
    rows, cols = rows[inside], cols[inside]
    if valid_mask is not None:
        keep = np.asarray(valid_mask, bool)[rows, cols]
        rows, cols = rows[keep], cols[keep]
    if unique and len(rows):
        flat = np.unique(rows * grid.width + cols)
        rows, cols = flat // grid.width, flat % grid.width
    return rows, cols


def apply_mask(array, mask: np.ndarray) -> np.ma.MaskedArray:
    """Mask cells where ``mask`` is False."""
    arr = np.ma.asarray(array)
    return np.ma.array(arr, mask=np.ma.getmaskarray(arr) | ~np.asarray(mask, bool))


def valid_mask(arrays: Iterable[np.ma.MaskedArray]) -> np.ndarray:
    """True where every array has data."""
    m = None
    for a in arrays:
        v = ~np.ma.getmaskarray(a)
        m = v if m is None else (m & v)
    return m


def unique_classes(array: np.ma.MaskedArray) -> np.ndarray:
    return np.unique(np.ma.compressed(array))


def percentile_stretch(array: np.ma.MaskedArray, lo: float = 2, hi: float = 98) -> Tuple[float, float]:
    vals = np.ma.compressed(array)
    if vals.size == 0:
        return 0.0, 1.0
    if vals.size > 2_000_000:
        vals = vals[:: vals.size // 1_000_000]
    a, b = np.percentile(vals, [lo, hi])
    if a == b:
        a, b = float(vals.min()), float(vals.max()) or 1.0
    return float(a), float(b)


def sample_to_cells(array, rows: Sequence[int], cols: Sequence[int]) -> np.ndarray:
    a = np.ma.asarray(array)
    return np.ma.filled(a[np.asarray(rows, int), np.asarray(cols, int)].astype(float), np.nan)
