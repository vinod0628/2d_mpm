"""Vector file reading and writing.

``geopandas.read_file`` (fiona engine, geopandas 0.11) builds the table feature by feature: a GeoPackage with 3,472
points and 47 fields takes ~9 s. pyogrio's raw reader returns whole columns from GDAL (~0.6 s). The pyogrio
GeoDataFrame reader needs Shapely 2, so the table is assembled here (geometries from WKB with Shapely 1.8).
Falls back to ``geopandas.read_file`` for anything the raw reader cannot open.

:func:`write_vector` chooses the format from the extension: ``.gpkg`` GeoPackage, ``.shp`` ESRI Shapefile (for
ArcMap: UTF-8 attributes with a ``.cpg`` file, ``.prj`` from the CRS, field names cut to 10 characters by the
driver), ``.geojson`` / ``.json`` GeoJSON.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import geopandas as gpd
import pandas as pd

log = logging.getLogger("mpm")

DRIVERS = {".gpkg": "GPKG", ".shp": "ESRI Shapefile", ".geojson": "GeoJSON", ".json": "GeoJSON"}
SHAPEFILE_PARTS = (".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix", ".sbn", ".sbx")


def write_vector(gdf: gpd.GeoDataFrame, path: str) -> str:
    """Write ``gdf`` in the format given by the extension of ``path`` (see module doc). Returns ``path``."""
    ext = os.path.splitext(path)[1].lower()
    driver = DRIVERS.get(ext)
    if driver is None:
        raise ValueError(f"Unsupported vector format '{ext}' — use .gpkg, .shp or .geojson")
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    out = gdf
    kw = {}
    if driver == "ESRI Shapefile":
        out = gdf.copy()
        for col in out.columns:
            if col == out.geometry.name:
                continue
            s = out[col]
            if pd.api.types.is_bool_dtype(s):
                out[col] = s.astype(int)                       # no boolean type in .dbf
            elif pd.api.types.is_datetime64_any_dtype(s):
                out[col] = s.dt.strftime("%Y-%m-%d %H:%M:%S")
            elif s.dtype == object:
                out[col] = s.map(lambda v: None if v is None or (isinstance(v, float) and pd.isna(v)) else str(v))
        long_names = [c for c in out.columns if c != out.geometry.name and len(str(c)) > 10]
        if long_names:
            log.info("%s: field names cut to 10 characters for the Shapefile: %s", os.path.basename(path),
                     ", ".join(f"{c} → {str(c)[:10]}" for c in long_names))
        base = os.path.splitext(path)[0]
        for part in SHAPEFILE_PARTS:                           # overwrite cleanly (stale .prj/.cpg/.qix)
            if os.path.exists(base + part):
                os.remove(base + part)
        kw["encoding"] = "utf-8"
    elif os.path.exists(path):
        os.remove(path)
    import warnings
    with warnings.catch_warnings():                            # truncation is logged above
        warnings.filterwarnings("ignore", message="Column names longer than 10")
        out.to_file(path, driver=driver, **kw)
    return path


def read_vector(path: str, layer: Optional[str] = None) -> gpd.GeoDataFrame:
    try:
        from pyogrio import raw
        from shapely import wkb
        meta, _fids, geom, fields = raw.read(path, layer=layer)
        data = {name: values for name, values in zip(meta["fields"], fields)}
        geoms = [wkb.loads(bytes(b)) if b is not None else None for b in geom]
        return gpd.GeoDataFrame(data, geometry=geoms, crs=meta.get("crs"))
    except Exception as exc:  # noqa: BLE001 — e.g. formats or layers pyogrio cannot read
        log.debug("pyogrio could not read %s (%s) — using geopandas.read_file", path, exc)
        return gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
