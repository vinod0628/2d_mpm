"""File-format conversion: any GDAL-readable raster (incl. ArcInfo binary grids) -> GeoTIFF,
and any OGR-readable vector -> GeoPackage / Shapefile, with optional CRS assignment.

Command line (from the ``python`` folder)::

    python -m utils.convert raster  <src> <dst.tif> [--crs EPSG:28354] [--nodata -9999]
    python -m utils.convert vector  <src> <dst.gpkg> [--crs ...] [--to-crs ...]
    python -m utils.convert folder  <src_dir> <dst_dir> [--crs ...]     # every raster + vector inside
"""
from __future__ import annotations

import argparse
import glob
import os
from typing import List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS

from .vector_io import read_vector
from .raster_tools import Grid, default_nodata, write_class_table, write_raster

RASTER_EXT = (".tif", ".tiff", ".img", ".asc", ".grd", ".nc", ".vrt", ".jp2", ".bil", ".ers", ".sdat")
VECTOR_EXT = (".shp", ".gpkg", ".geojson", ".json", ".kml", ".gml", ".tab", ".mif", ".csv")


def is_arcinfo_grid(path: str) -> bool:
    """True for an ArcInfo binary grid folder (contains hdr.adf)."""
    return os.path.isdir(path) and any(os.path.isfile(os.path.join(path, f)) for f in ("hdr.adf", "HDR.ADF"))


def list_rasters(folder: str) -> List[str]:
    out = []
    for root, dirs, files in os.walk(folder):
        if is_arcinfo_grid(root):
            out.append(root)
            dirs[:] = []
            continue
        out += [os.path.join(root, f) for f in files if f.lower().endswith(RASTER_EXT)]
    return sorted(out)


def list_vectors(folder: str) -> List[str]:
    return sorted(p for p in glob.glob(os.path.join(folder, "**", "*"), recursive=True)
                  if p.lower().endswith((".shp", ".gpkg", ".geojson")))


def read_attribute_table(path: str) -> Optional[pd.DataFrame]:
    """Read a raster attribute table (e.g. ArcInfo grid VAT) through rasterio's own GDAL library."""
    from .gdal_rat import read_rat
    try:
        df = read_rat(path)
    except Exception:  # noqa: BLE001
        return None
    if df is None:
        return None
    df.columns = [c.lower() if c.upper() in ("VALUE", "COUNT") else c for c in df.columns]
    return df


def raster_to_geotiff(src: str, dst: str, crs=None, override_crs: bool = False, nodata=None,
                      dtype: Optional[str] = None, label_field: Optional[str] = None,
                      write_rat: bool = True) -> str:
    """Convert a raster to compressed GeoTIFF.

    crs: CRS to assign when the source has none (or always, with ``override_crs``).
    nodata: new nodata value (default: keep the source value, else a dtype default).
    label_field: attribute-table column used as class ``label`` in the sidecar ``.classes.csv``.
    """
    with rasterio.open(src) as ds:
        arr = ds.read(1, masked=True)
        grid = Grid.from_dataset(ds)
        src_nodata = ds.nodata
    if crs is not None and (grid.crs is None or override_crs):
        grid = Grid(grid.transform, CRS.from_user_input(crs), grid.width, grid.height)
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.ma.masked_invalid(arr)
        if src_nodata is not None and abs(src_nodata) > 1e30:   # ArcInfo float grids use -FLT_MAX
            nodata = -9999.0 if nodata is None else nodata
    out_dtype = np.dtype(dtype) if dtype else arr.dtype
    if nodata is None:
        nodata = src_nodata if src_nodata is not None else default_nodata(out_dtype)
    table = read_attribute_table(src) if write_rat else None
    if table is not None and "value" in table.columns:
        if label_field and label_field in table.columns:
            table.insert(1, "label", table[label_field].astype(str))
        elif "label" not in table.columns:
            str_cols = [c for c in table.columns if table[c].dtype == object]
            table.insert(1, "label", table[str_cols[0]].astype(str) if str_cols else table["value"].astype(str))
    else:
        table = None
    write_raster(dst, arr, grid, nodata=nodata, dtype=out_dtype, class_table=table)
    return dst


def vector_convert(src: str, dst: str, crs=None, to_crs=None, layer: Optional[str] = None) -> str:
    """Convert a vector file; ``crs`` is assigned if missing, ``to_crs`` reprojects."""
    gdf = read_vector(src, layer=layer)
    if crs is not None and gdf.crs is None:
        gdf = gdf.set_crs(crs)
    if to_crs is not None:
        gdf = gdf.to_crs(to_crs)
    from .vector_io import write_vector
    return write_vector(gdf, dst)


def points_from_table(csv_path: str, x: str, y: str, crs, to_crs=None, dst: Optional[str] = None):
    """Create a point GeoDataFrame from a CSV with coordinate columns."""
    df = pd.read_csv(csv_path)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[x], df[y]), crs=crs)
    if to_crs is not None:
        gdf = gdf.to_crs(to_crs)
    if dst:
        gdf.to_file(dst, driver="GPKG")
    return gdf


def convert_folder(src_dir: str, dst_dir: str, crs=None) -> List[str]:
    """Convert every raster (to .tif) and vector (to .gpkg) below ``src_dir``."""
    done = []
    for r in list_rasters(src_dir):
        name = os.path.splitext(os.path.basename(r.rstrip("\\/")))[0]
        done.append(raster_to_geotiff(r, os.path.join(dst_dir, name + ".tif"), crs=crs))
    for v in list_vectors(src_dir):
        name = os.path.splitext(os.path.basename(v))[0]
        done.append(vector_convert(v, os.path.join(dst_dir, name + ".gpkg"), crs=crs))
    return done


def _main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("raster"); r.add_argument("src"); r.add_argument("dst")
    r.add_argument("--crs"); r.add_argument("--override-crs", action="store_true")
    r.add_argument("--nodata", type=float); r.add_argument("--label-field")
    v = sub.add_parser("vector"); v.add_argument("src"); v.add_argument("dst")
    v.add_argument("--crs"); v.add_argument("--to-crs")
    f = sub.add_parser("folder"); f.add_argument("src"); f.add_argument("dst"); f.add_argument("--crs")
    a = ap.parse_args(argv)
    if a.cmd == "raster":
        print(raster_to_geotiff(a.src, a.dst, crs=a.crs, override_crs=a.override_crs, nodata=a.nodata,
                                label_field=a.label_field))
    elif a.cmd == "vector":
        print(vector_convert(a.src, a.dst, crs=a.crs, to_crs=a.to_crs))
    else:
        print("\n".join(convert_folder(a.src, a.dst, crs=a.crs)))


if __name__ == "__main__":
    _main()
