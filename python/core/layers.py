"""Layer model shared by the TOC, the map canvas, the dialogs and the methods."""
from __future__ import annotations

import itertools
import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from PyQt5.QtCore import QObject, pyqtSignal
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject

from i18n import tr
from utils.geo_env import unrecognized

from utils.vector_io import read_vector
from utils.raster_tools import (Grid, align_to_grid, percentile_stretch, read_class_table, read_raster,
                                write_class_table, write_raster)

_ids = itertools.count(1)
MAX_CATEGORIES = 64
DEFAULT_RASTER_COLOR = "#4f8fd6"
DEFAULT_NODATA = -9999.0             # project NoData value for raster outputs (Project ▸ Project properties)
DEFAULT_LABELS = dict(enabled=False, field=None, size=8.0, color="#202020", family="")


def _jsonable(obj):
    """dicts / lists with numpy scalars -> plain JSON types."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


class Layer:
    kind = "layer"

    def __init__(self, name: str, source: Optional[str] = None):
        self.id = next(_ids)
        self.name = name
        self.source = source
        self.visible = True
        self.opacity = 1.0
        self.description = ""

    @property
    def crs(self):
        raise NotImplementedError

    def bounds(self):
        raise NotImplementedError


log = logging.getLogger("mpm")


def _warn_assumed(name, src, dst, exc=None):
    what = tr("cannot be transformed ({err})", err=str(exc).splitlines()[0][:120]) if exc is not None else         tr("is not recognised")
    log.warning(tr("{name}: coordinate system {src} {what} — treated as the current coordinate system {dst}",
                   name=name, src=(src.to_wkt()[:60] + "…") if src is not None else tr("(none)"), what=what,
                   dst=describe_crs(CRS.from_user_input(dst))[0]))


def conform_vector_crs(gdf, target_crs, name):
    """(gdf in ``target_crs``, reprojected?): reprojected if its CRS is known and transformable, else (missing,
    unrecognised or not transformable) taken to be ``target_crs`` with a warning."""
    tgt = CRS.from_user_input(target_crs)
    if unrecognized(gdf.crs):
        _warn_assumed(name, None if gdf.crs is None else CRS.from_user_input(gdf.crs), tgt)
        return gdf.set_crs(tgt.to_wkt(), allow_override=True), False
    if CRS.from_user_input(gdf.crs) == tgt:
        return gdf, False
    try:
        return gdf.to_crs(tgt.to_wkt()), True
    except Exception as exc:  # noqa: BLE001
        _warn_assumed(name, CRS.from_user_input(gdf.crs), tgt, exc)
        return gdf.set_crs(tgt.to_wkt(), allow_override=True), False


class RasterLayer(Layer):
    kind = "raster"

    def __init__(self, name: str, data: Optional[np.ma.MaskedArray], grid: Grid, source: Optional[str] = None,
                 categorical: Optional[bool] = None, class_table: Optional[pd.DataFrame] = None,
                 cmap: Optional[str] = None, nodata=None, loader: Optional[Callable] = None):
        """``data`` = the cells, or None with ``loader`` (a function returning them): lazy layer, read on first use
        of :attr:`data` — in a background thread via :meth:`LayerStore.request_data`, or on demand."""
        super().__init__(name, source)
        self._lock = threading.RLock()
        self._data = None if data is None else np.ma.asarray(data)
        self._loader = loader
        self._cache: dict = {}                  # classes / stats of the current data
        self.grid = grid
        self.nodata = nodata
        self.class_table = class_table
        if categorical is None:
            categorical = class_table is not None or self._looks_categorical()
        self.categorical = bool(categorical)
        self.cmap = cmap or ("tab20" if self.categorical else "viridis")
        self._vmin = self._vmax = None          # stretch: 2-98 % of the data, computed when first needed
        self.stretch = "percent_2_98"
        self._legend_cache = None
        # symbology (core/symbology.py): renderer None = unique values for classes, stretched otherwise
        self._renderer: Optional[str] = None
        self.single_color = DEFAULT_RASTER_COLOR
        self.unique: dict = {}              # cmap, colors {value: hex}, labels {value: text}
        self.categorized: dict = {}         # field, method, n, cmap, classes [...]

    # ------------------------------------------------------------ lazy data
    @property
    def data(self) -> np.ma.MaskedArray:
        if self._data is None:
            with self._lock:                    # one read even when a background thread is reading it
                if self._data is None:
                    self._data = np.ma.asarray(self._loader())
                    self._loader = None
        return self._data

    @data.setter
    def data(self, value):
        with self._lock:
            self._data = np.ma.asarray(value)
            self._loader = None
            self._cache = {}

    @property
    def loaded(self) -> bool:
        return self._data is not None

    def load(self) -> "RasterLayer":
        """Read the cells (and the default stretch) — called in a worker thread by the layer store."""
        _ = self.data
        if self._vmin is None:
            _ = self.vmin
        return self

    @property
    def vmin(self) -> float:
        if self._vmin is None:
            self._vmin, self._vmax = percentile_stretch(self.data)
        return self._vmin

    @vmin.setter
    def vmin(self, v):
        self._vmin = float(v)
        if self._vmax is None:
            self._vmax = self._vmin

    @property
    def vmax(self) -> float:
        if self._vmax is None:
            self._vmin, self._vmax = percentile_stretch(self.data)
        return self._vmax

    @vmax.setter
    def vmax(self, v):
        self._vmax = float(v)

    @property
    def renderer(self) -> str:
        return self._renderer or ("unique" if self.categorical else "stretch")

    @renderer.setter
    def renderer(self, value: Optional[str]):
        self._renderer = value

    def symbology_dict(self) -> dict:
        return dict(renderer=self._renderer, single_color=self.single_color,
                    unique=dict(cmap=self.unique.get("cmap"),
                                colors={str(k): v for k, v in (self.unique.get("colors") or {}).items()},
                                labels={str(k): v for k, v in (self.unique.get("labels") or {}).items()}),
                    categorized=_jsonable(self.categorized))

    def apply_symbology(self, d: Optional[dict]):
        if not d:
            return
        self._renderer = d.get("renderer")
        self.single_color = d.get("single_color") or self.single_color
        u = d.get("unique") or {}
        self.unique = {k: v for k, v in dict(cmap=u.get("cmap"), colors=u.get("colors") or {},
                                              labels=u.get("labels") or {}).items() if v}
        self.categorized = d.get("categorized") or {}

    def _looks_categorical(self) -> bool:
        if np.issubdtype(self.data.dtype, np.floating):
            return False
        vals = np.unique(np.ma.compressed(self.data[:: max(1, self.data.shape[0] // 300),
                                                   :: max(1, self.data.shape[1] // 300)]))
        return vals.size <= 32

    @staticmethod
    def _looks_categorical_file(ds) -> bool:
        """Class raster? From the data type and a reduced read (at most 300 x 300 cells) — no full read."""
        if np.issubdtype(np.dtype(ds.dtypes[0]), np.floating):
            return False
        small = ds.read(1, masked=True, out_shape=(min(ds.height, 300), min(ds.width, 300)))
        return np.unique(np.ma.compressed(small)).size <= 32

    # ------------------------------------------------------------ factory
    @classmethod
    def from_file(cls, path: str, name: Optional[str] = None, target_crs=None,
                  categorical: Optional[bool] = None, assume_crs=None) -> "RasterLayer":
        """Lazy layer: only the header (grid, CRS, NoData, class table) is read here; the cells are read on first
        use. With ``target_crs`` the cells are reprojected when they are read. A missing / unrecognised CRS
        (:func:`utils.geo_env.unrecognized`), or one that cannot be transformed to ``target_crs``, is taken to be
        ``target_crs`` (or ``assume_crs``) — the layer is loaded as stored, with a warning."""
        table = read_class_table(path)
        lyr_name = name or os.path.splitext(os.path.basename(path.rstrip("\\/")))[0]
        with rasterio.open(path) as ds:
            grid = Grid.from_dataset(ds)
            nodata = ds.nodata
            if categorical is None and table is None:
                categorical = cls._looks_categorical_file(ds)
        dst = None
        lyr_label = name or os.path.basename(path)
        current = target_crs if target_crs is not None else assume_crs
        if current is not None and unrecognized(grid.crs):
            _warn_assumed(lyr_label, grid.crs, current)
            grid = Grid(grid.transform, CRS.from_user_input(current), grid.width, grid.height)
        elif target_crs is not None and CRS.from_user_input(target_crs) != grid.crs:
            tgt = CRS.from_user_input(target_crs)
            left, bottom, right, top = grid.bounds
            try:
                transform, width, height = calculate_default_transform(grid.crs, tgt, grid.width, grid.height,
                                                                       left, bottom, right, top)
            except Exception as exc:  # noqa: BLE001 — no transformation available: treat as the current CRS
                _warn_assumed(lyr_label, grid.crs, tgt, exc)
                grid = Grid(grid.transform, tgt, grid.width, grid.height)
            else:
                dst = tgt
                grid = Grid(transform, dst, width, height)
        cat = bool(categorical) if categorical is not None else table is not None

        def loader():
            data, g, _prof = read_raster(path)
            if dst is not None:
                data, _g = reproject_masked(data, g, dst, "nearest" if cat else "bilinear")
            return data
        return cls(lyr_name, None, grid, source=path, categorical=categorical if categorical is not None else cat,
                   class_table=table, nodata=nodata, loader=loader)

    # ------------------------------------------------------------ info
    @property
    def crs(self):
        return self.grid.crs

    def bounds(self):
        return self.grid.bounds

    def value_counts(self):
        """(distinct cell values, number of cells of each) — one pass, computed once per data and reused by the
        legend, the drawing and the symbology dialog (never recounted on redraws)."""
        hit = self._cache.get("counts")
        if hit is None or self._data is None or hit[0] != id(self._data):
            vals, counts = np.unique(np.ma.compressed(self.data), return_counts=True)
            self._cache["counts"] = hit = (id(self._data), vals, counts)
        return hit[1], hit[2]

    def classes(self) -> np.ndarray:
        """Distinct cell values (cached, see :meth:`value_counts`)."""
        return self.value_counts()[0]

    def labels(self) -> Dict[float, str]:
        if self.class_table is None:
            return {}
        return {float(v): str(l) for v, l in zip(self.class_table["value"], self.class_table["label"])}

    def label_for(self, value) -> str:
        return self.labels().get(float(value), f"{value:g}")

    def value_at(self, x: float, y: float):
        r, c = self.grid.xy_to_rowcol([x], [y])
        r, c = int(r[0]), int(c[0])
        if not (0 <= r < self.grid.height and 0 <= c < self.grid.width):
            return None
        v = self.data[r, c]
        return None if v is np.ma.masked else v.item() if hasattr(v, "item") else v

    def stats(self) -> Dict[str, float]:
        """Valid cells, min, max, mean, std — computed once per data."""
        hit = self._cache.get("stats")
        if hit is not None and hit[0] == id(self._data) and self._data is not None:
            return hit[1]
        v = np.ma.compressed(self.data).astype(float)
        st = dict(valid=0) if v.size == 0 else dict(valid=int(v.size), min=float(v.min()), max=float(v.max()),
                                                     mean=float(v.mean()), std=float(v.std()))
        self._cache["stats"] = (id(self._data), st)
        return st

    def save(self, path: str) -> str:
        write_raster(path, self.data, self.grid, nodata=self.nodata,
                     class_table=self.class_table if self.categorical else None)
        self.source = path
        return path

    def to_dict(self, base: str) -> dict:
        return dict(type="raster", name=self.name, path=_rel(self.source, base), visible=self.visible,
                    categorical=self.categorical,
                    style=dict(cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, opacity=self.opacity),
                    symbology=self.symbology_dict())


class VectorLayer(Layer):
    kind = "vector"

    def __init__(self, name: str, gdf: gpd.GeoDataFrame, source: Optional[str] = None, style: Optional[dict] = None):
        super().__init__(name, source)
        self.gdf = gdf
        gtypes = set(t.replace("Multi", "") for t in gdf.geom_type.dropna().unique())
        self.geom_type = "point" if gtypes <= {"Point"} else ("line" if gtypes <= {"LineString"} else "polygon")
        default = {"point": dict(color="#ffd54f", edgecolor="#202020", markersize=24, marker="o"),
                   "line": dict(color="#ff8a65", linewidth=0.8),
                   "polygon": dict(facecolor="none", edgecolor="#b0bec5", linewidth=0.8)}[self.geom_type]
        self.style = {**default, **(style or {})}
        # symbology (core/symbology.py): style = the single symbol
        self.renderer = "single"
        self.unique: dict = {}              # cmap, field (row labels), styles {row: symbol}
        self.categorized: dict = {}         # field, method, n, cmap, classes [...]
        self.labels: dict = dict(DEFAULT_LABELS)

    def symbology_dict(self) -> dict:
        return dict(renderer=self.renderer, unique=_jsonable(self.unique), categorized=_jsonable(self.categorized),
                    labels=_jsonable(self.labels))

    def apply_symbology(self, d: Optional[dict]):
        if not d:
            return
        self.renderer = d.get("renderer") or "single"
        self.unique = d.get("unique") or {}
        self.categorized = d.get("categorized") or {}
        self.labels = {**DEFAULT_LABELS, **{k: v for k, v in (d.get("labels") or {}).items() if k in DEFAULT_LABELS}}

    @classmethod
    def from_file(cls, path: str, name: Optional[str] = None, target_crs=None, assign_crs=None,
                  style: Optional[dict] = None) -> "VectorLayer":
        gdf = read_vector(path)                     # pyogrio: ~10x faster than geopandas.read_file
        if gdf.crs is None and assign_crs is not None:
            gdf = gdf.set_crs(assign_crs)
        if target_crs is not None:
            gdf, _ = conform_vector_crs(gdf, target_crs, name or os.path.basename(path))
        return cls(name or os.path.splitext(os.path.basename(path))[0], gdf, source=path, style=style)

    @property
    def crs(self):
        return CRS.from_user_input(self.gdf.crs) if self.gdf.crs is not None else None

    def bounds(self):
        key = (id(self.gdf), len(self.gdf))             # per-feature bounds are slow with Shapely 1.8: cached
        if getattr(self, "_bounds", (None,))[0] != key:
            b = self.gdf.total_bounds
            self._bounds = (key, (float(b[0]), float(b[1]), float(b[2]), float(b[3])))
        return self._bounds[1]

    def save(self, path: str) -> str:
        """Write the layer (.gpkg, .shp or .geojson by extension) and make the file its source."""
        from utils.vector_io import write_vector
        write_vector(self.gdf, path)
        self.source = path
        return path

    def to_dict(self, base: str) -> dict:
        st = {k: v for k, v in self.style.items()}
        st["opacity"] = self.opacity
        return dict(type="vector", name=self.name, path=_rel(self.source, base), visible=self.visible, style=st,
                    symbology=self.symbology_dict())


def reproject_masked(data: np.ma.MaskedArray, grid: Grid, dst_crs: CRS, resampling: str = "nearest"):
    left, bottom, right, top = grid.bounds
    transform, width, height = calculate_default_transform(grid.crs, dst_crs, grid.width, grid.height,
                                                           left, bottom, right, top)
    src = np.ma.filled(data.astype("float64"), np.nan)
    dst = np.full((height, width), np.nan)
    reproject(src, dst, src_transform=grid.transform, src_crs=grid.crs, src_nodata=np.nan,
              dst_transform=transform, dst_crs=dst_crs, dst_nodata=np.nan, resampling=getattr(Resampling, resampling))
    out = np.ma.masked_invalid(dst)
    if resampling == "nearest" and not np.issubdtype(data.dtype, np.floating):
        out = np.ma.array(np.ma.filled(out, 0).astype(data.dtype), mask=np.ma.getmaskarray(out))
    return out, Grid(transform, dst_crs, width, height)


def conform_method(layer: "RasterLayer", grid: Grid) -> str:
    """Resampling for :func:`conform_raster`: nearest for classes, average when the cells get bigger, else bilinear."""
    if layer.categorical:
        return "nearest"
    same_units = layer.grid.crs is None or grid.crs is None or layer.grid.crs == grid.crs
    if same_units and grid.res[0] > max(layer.grid.res) * 1.0001:
        return "average"
    return "bilinear"


def conform_raster(layer: "RasterLayer", grid: Grid, method: Optional[str] = None) -> "RasterLayer":
    """The raster on ``grid`` (CRS, cell size, extent, alignment) in ONE resampling step from the original data.

    Cells of ``grid`` outside the raster become NoData. The result is an in-memory layer (no source file), so
    *Save project* writes it to the project's data folder.
    """
    method = method or conform_method(layer, grid)
    data = align_to_grid(layer.data, layer.grid, grid, method)
    out = RasterLayer(layer.name, data, grid, categorical=layer.categorical, class_table=layer.class_table,
                      cmap=layer.cmap, nodata=layer.nodata)
    out.description = f"{layer.source or layer.name} on the project grid ({method})"
    return out


def describe_crs(crs):
    """(short, long) human-readable CRS description; custom CRSs -> 'method / ellipsoid'."""
    if crs is None:
        return "—", "no CRS"
    try:
        from pyproj import CRS as PCRS
        p = PCRS.from_user_input(crs.to_wkt())
        epsg = p.to_epsg()
        name = p.name
        if not name or name.lower() == "unnamed":
            method = p.coordinate_operation.method_name if p.coordinate_operation else ("Geographic" if p.is_geographic else "Projected")
            name = f"{method} / {p.ellipsoid.name if p.ellipsoid else '?'}"
        return (f"EPSG:{epsg}" if epsg else name), (f"EPSG:{epsg} — {name}" if epsg else f"custom: {name}")
    except Exception:  # noqa: BLE001
        return "custom", "custom CRS"


_UNIT_ABBR = {"metre": "m", "meter": "m", "degree": "°", "foot": "ft", "us survey foot": "ftUS", "kilometre": "km"}


def crs_unit(crs) -> str:
    """Short unit of the CRS axes ('m', '°', 'ft' …); '' when unknown or no CRS."""
    if crs is None:
        return ""
    try:
        from pyproj import CRS as PCRS
        p = PCRS.from_user_input(crs.to_wkt() if hasattr(crs, "to_wkt") else crs)
        name = (p.axis_info[0].unit_name if p.axis_info else "").lower()
        return _UNIT_ABBR.get(name, name)
    except Exception:  # noqa: BLE001
        return ""


def format_length(value: Optional[float], crs) -> str:
    """Cell size / length with the CRS unit, e.g. '250 m', '0.0083333°'; '—' for None."""
    if not value:
        return "—"
    u = crs_unit(crs)
    v = f"{value:.8g}" if u == "°" or value < 1 else f"{value:,.6g}"
    return f"{v}°" if u == "°" else (f"{v} {u}" if u else v)


def _rel(path: Optional[str], base: str) -> Optional[str]:
    if not path:
        return None
    try:
        return os.path.relpath(path, base).replace("\\", "/")
    except ValueError:          # different drive
        return path


class LayerStore(QObject):
    """Ordered layer list (index 0 = top of the TOC = drawn last)."""

    changed = pyqtSignal()                 # any structural change (add/remove/reorder)
    layer_changed = pyqtSignal(object)     # style / visibility of one layer
    active_changed = pyqtSignal(object)
    data_loaded = pyqtSignal(object)       # a lazy raster was read in the background (not a change: not dirty)
    data_failed = pyqtSignal(object, str)

    def __init__(self):
        super().__init__()
        self.layers: List[Layer] = []
        # project settings = the analysis grid every raster of the project shares (MPM methods combine rasters cell
        # by cell). Editable while the project has no layers (Project ▸ Project properties), otherwise fixed.
        # Unset values are taken from the first layer (CRS) / the first raster layer (cell size, extent).
        self.crs: Optional[CRS] = None
        self.cell_size: Optional[float] = None   # raster unit cell size in map units
        self.extent: Optional[tuple] = None      # (left, bottom, right, top) of the project grid
        # NoData value written into the rasters the MPM methods and tools create (editable at any time)
        self.nodata: float = DEFAULT_NODATA
        self.active: Optional[Layer] = None
        self.project_path: Optional[str] = None
        self.dirty = False                      # unsaved changes (layers added/removed/re-styled)
        self.changed.connect(self._mark_dirty)
        self.layer_changed.connect(self._mark_dirty)
        self._pool: Optional[ThreadPoolExecutor] = None
        self._pending = set()                  # ids of rasters being read in the background

    # ------------------------------------------------------------ background reading
    def request_data(self, layer) -> bool:
        """Read a lazy raster in a worker thread (rasterio releases the GIL: several files are read in parallel);
        ``data_loaded`` is emitted in the GUI thread when it is ready. False = already loaded."""
        if not isinstance(layer, RasterLayer) or layer.loaded or layer.id in self._pending:
            return not isinstance(layer, RasterLayer) or not layer.loaded
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=max(2, min(4, (os.cpu_count() or 2))),
                                            thread_name_prefix="raster-read")
        self._pending.add(layer.id)

        def done(fut):
            self._pending.discard(layer.id)
            exc = fut.exception()
            if exc is None:
                self.data_loaded.emit(layer)             # queued to the GUI thread
            else:
                self.data_failed.emit(layer, str(exc))
        self._pool.submit(layer.load).add_done_callback(done)
        return True

    def prefetch(self, layers=None):
        """Read the visible lazy rasters in the background (top of the TOC first)."""
        for l in (layers if layers is not None else self.layers):
            if isinstance(l, RasterLayer) and l.visible and not l.loaded:
                self.request_data(l)

    def wait_loaded(self, timeout: float = 60.0) -> bool:
        """Block until background reads are finished (tests, saving)."""
        import time
        end = time.monotonic() + timeout
        while self._pending and time.monotonic() < end:
            time.sleep(0.02)
        return not self._pending

    def shutdown(self):
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None
        self._pending.clear()

    def _mark_dirty(self, *_):
        self.dirty = True

    # ------------------------------------------------------------ access
    def rasters(self) -> List[RasterLayer]:
        return [l for l in self.layers if isinstance(l, RasterLayer)]

    def vectors(self, geom_type: Optional[str] = None) -> List[VectorLayer]:
        return [l for l in self.layers if isinstance(l, VectorLayer) and (geom_type is None or l.geom_type == geom_type)]

    def by_id(self, lid: int) -> Optional[Layer]:
        return next((l for l in self.layers if l.id == lid), None)

    def by_name(self, name: str) -> Optional[Layer]:
        return next((l for l in self.layers if l.name == name), None)

    def unique_name(self, name: str) -> str:
        names = {l.name for l in self.layers}
        if name not in names:
            return name
        for i in itertools.count(2):
            if f"{name}_{i}" not in names:
                return f"{name}_{i}"

    # ------------------------------------------------------------ edit
    # ------------------------------------------------------------ project settings
    @property
    def settings_locked(self) -> bool:
        """CRS, cell size and extent can only be changed while the project has no layers."""
        return bool(self.layers)

    def set_project_settings(self, crs, cell_size: Optional[float], extent: Optional[tuple] = None):
        """Set the project CRS / cell size / extent (None = unset). Only allowed while there are no layers."""
        if self.settings_locked:
            raise RuntimeError("The coordinate system, cell size and extent are fixed once the project has layers.")
        self.crs = CRS.from_user_input(crs) if crs is not None and not isinstance(crs, CRS) else crs
        self.cell_size = float(cell_size) if cell_size else None
        self.extent = tuple(float(v) for v in extent) if extent else None
        self.changed.emit()

    def adopt_settings(self, layer: Layer):
        """Unset project settings are taken from the first layer (CRS) / first raster layer (cell size, extent).

        A cell size set beforehand keeps its value; the extent is then the raster's extent snapped outwards to
        multiples of the cell size.
        """
        if self.crs is None and layer.crs is not None:
            self.crs = layer.crs
        if not isinstance(layer, RasterLayer):
            return
        if self.cell_size is None:
            self.cell_size = float(layer.grid.res[0])
            if self.extent is None:
                self.extent = tuple(float(v) for v in layer.grid.bounds)     # exactly the first raster
        if self.extent is None:
            g = Grid.from_bounds(*layer.grid.bounds, self.cell_size, snap=True)
            self.extent = tuple(float(v) for v in g.bounds)

    _adopt_settings = adopt_settings

    def project_grid(self) -> Optional[Grid]:
        """The grid every project raster shares (None while the cell size or extent is not known yet)."""
        if not self.cell_size or not self.extent:
            return None
        left, bottom, right, top = self.extent
        c = self.cell_size
        width = max(1, int(np.ceil((right - left) / c - 1e-9)))
        height = max(1, int(np.ceil((top - bottom) / c - 1e-9)))
        from rasterio.transform import from_origin
        return Grid(from_origin(left, top, c, c), self.crs, width, height)

    def grid_mismatch(self, layer: Layer) -> bool:
        """True for a raster that is not on the project grid (other extent, cell size or alignment)."""
        g = self.project_grid()
        return bool(isinstance(layer, RasterLayer) and g is not None and not layer.grid.same_as(g))

    def cell_size_mismatch(self, layer: Layer) -> bool:
        if not isinstance(layer, RasterLayer) or not self.cell_size:
            return False
        return any(abs(r - self.cell_size) > 1e-6 * self.cell_size for r in layer.grid.res)

    def add(self, layer: Layer, top: bool = True) -> Layer:
        self._adopt_settings(layer)
        layer.name = self.unique_name(layer.name)
        if top:
            self.layers.insert(0, layer)
        else:
            self.layers.append(layer)
        self.changed.emit()
        self.set_active(layer)
        return layer

    def add_raster_file(self, path: str, **kw) -> RasterLayer:
        return self.add(RasterLayer.from_file(path, target_crs=self.crs, **kw))

    def add_vector_file(self, path: str, **kw) -> VectorLayer:
        return self.add(VectorLayer.from_file(path, target_crs=self.crs, **kw))

    def remove(self, layer: Layer):
        if layer in self.layers:
            self.layers.remove(layer)
            if self.active is layer:
                self.active = self.layers[0] if self.layers else None
            # the project CRS / cell size stay as they are (editable again now that there are no layers)
            self.changed.emit()
            self.active_changed.emit(self.active)

    def reorder(self, ids: List[int]):
        lookup = {l.id: l for l in self.layers}
        new = [lookup[i] for i in ids if i in lookup]
        new += [l for l in self.layers if l not in new]
        if new != self.layers:
            self.layers = new
            self.changed.emit()

    def move(self, layer: Layer, delta: int):
        i = self.layers.index(layer)
        j = max(0, min(len(self.layers) - 1, i + delta))
        if i != j:
            self.layers.insert(j, self.layers.pop(i))
            self.changed.emit()

    def set_visible(self, layer: Layer, visible: bool):
        if layer.visible != visible:
            layer.visible = visible
            self.layer_changed.emit(layer)

    def set_active(self, layer: Optional[Layer]):
        self.active = layer
        self.active_changed.emit(layer)

    def notify(self, layer: Layer):
        self.layer_changed.emit(layer)

    def clear(self):
        self.layers.clear()
        self.crs = None
        self.cell_size = None
        self.extent = None
        self.nodata = DEFAULT_NODATA
        self.active = None
        self.project_path = None
        self.changed.emit()
        self.active_changed.emit(None)

    # ------------------------------------------------------------ project files
    def save_project(self, path: str, name: Optional[str] = None):
        base = os.path.dirname(os.path.abspath(path))
        unsaved = [l.name for l in self.layers if not l.source]
        proj = dict(format="mpm-project", version=1, name=name or os.path.splitext(os.path.basename(path))[0],
                    crs=self.crs.to_wkt() if self.crs else None, cell_size=self.cell_size,
                    extent=list(self.extent) if self.extent else None, nodata=self.nodata,
                    layers=[l.to_dict(base) for l in self.layers if l.source])
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(proj, fh, indent=2)
        self.project_path = path
        return unsaved

    def load_project(self, path: str, progress=None) -> List[str]:
        with open(path, encoding="utf-8") as fh:
            proj = json.load(fh)
        if proj.get("format") != "mpm-project":
            raise ValueError("Not an MPM project file.")
        base = os.path.dirname(os.path.abspath(path))
        self.blockSignals(True)
        errors = []
        try:
            self.layers.clear()
            self.crs = CRS.from_wkt(proj["crs"]) if proj.get("crs") else None
            self.cell_size = float(proj["cell_size"]) if proj.get("cell_size") else None
            self.extent = tuple(float(v) for v in proj["extent"]) if proj.get("extent") else None
            self.nodata = float(proj["nodata"]) if proj.get("nodata") is not None else DEFAULT_NODATA
            entries = proj.get("layers", [])
            for i, e in enumerate(entries):
                p = e["path"] if os.path.isabs(e["path"]) else os.path.normpath(os.path.join(base, e["path"]))
                if progress:
                    progress(int(100 * i / max(1, len(entries))), f"loading {e.get('name')}")
                try:
                    st = dict(e.get("style") or {})
                    if e["type"] == "raster":
                        lyr = RasterLayer.from_file(p, name=e.get("name"), target_crs=self.crs,
                                                    categorical=e.get("categorical"))
                        if "cmap" in st:
                            lyr.cmap = st["cmap"]
                        if st.get("vmin") is not None and st.get("vmax") is not None:
                            lyr.vmin, lyr.vmax = float(st["vmin"]), float(st["vmax"])
                    else:
                        opacity = st.pop("opacity", 1.0)
                        lyr = VectorLayer.from_file(p, name=e.get("name"), target_crs=self.crs, style=st)
                        lyr.opacity = opacity
                    lyr.opacity = float(st.get("opacity", lyr.opacity))
                    lyr.apply_symbology(e.get("symbology"))
                    lyr.visible = bool(e.get("visible", True))
                    self.layers.append(lyr)
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{e.get('name')}: {exc}")
        finally:
            self.blockSignals(False)
        for lyr in reversed(self.layers):           # older project files: settings from the first (bottom) layer
            self._adopt_settings(lyr)
        self.project_path = path
        self.active = self.layers[0] if self.layers else None
        self.changed.emit()
        self.active_changed.emit(self.active)
        return errors
