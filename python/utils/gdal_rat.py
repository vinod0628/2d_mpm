"""Read raster attribute tables (RAT) with the GDAL C library that ships inside the rasterio wheel.

rasterio has no RAT API, and the GDAL Python bindings (``osgeo``) are not pip-installable on Windows.
rasterio's wheel already contains the GDAL DLL (``rasterio.libs/gdal-<hash>.dll``), so we call the few
RAT functions of the GDAL C API directly through ``ctypes``. Loading the DLL by its full path returns the
module rasterio has already loaded — no second GDAL copy, no extra install.

Used for ArcInfo binary grids (VAT tables) and any other format whose driver exposes a RAT
(GeoTIFF + .aux.xml, ERDAS IMAGINE .img, …).
"""
from __future__ import annotations

import ctypes
import glob
import os
import sys
from typing import Optional

import pandas as pd

_LIB = None


def _candidate_dirs():
    import rasterio
    pkg = os.path.dirname(os.path.abspath(rasterio.__file__))
    site = os.path.dirname(pkg)
    dirs = [os.path.join(site, "rasterio.libs"), pkg, os.path.join(pkg, ".libs"), site]
    meipass = getattr(sys, "_MEIPASS", None)                      # PyInstaller build
    if meipass:
        dirs = [os.path.join(meipass, "rasterio.libs"), meipass] + dirs
    return dirs


def gdal_library():
    """ctypes handle of rasterio's GDAL library (None if it cannot be found)."""
    global _LIB
    if _LIB is not None:
        return _LIB
    import rasterio  # noqa: F401  (make sure rasterio has loaded GDAL and registered the drivers)
    patterns = ["gdal*.dll", "libgdal*.dll"] if os.name == "nt" else ["libgdal*.so*", "libgdal*.dylib"]
    for d in _candidate_dirs():
        for pat in patterns:
            for path in sorted(glob.glob(os.path.join(d, pat))):
                try:
                    lib = ctypes.CDLL(path)
                    lib.GDALVersionInfo  # noqa: B018  (raises AttributeError if not GDAL)
                except (OSError, AttributeError):
                    continue
                _bind(lib)
                _LIB = lib
                return lib
    return None


def _bind(lib):
    vp, cp, ci = ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int
    lib.GDALVersionInfo.argtypes, lib.GDALVersionInfo.restype = [cp], cp
    lib.GDALAllRegister.argtypes, lib.GDALAllRegister.restype = [], None
    lib.GDALOpen.argtypes, lib.GDALOpen.restype = [cp, ci], vp
    lib.GDALClose.argtypes, lib.GDALClose.restype = [vp], None
    lib.GDALGetRasterBand.argtypes, lib.GDALGetRasterBand.restype = [vp, ci], vp
    lib.GDALGetDefaultRAT.argtypes, lib.GDALGetDefaultRAT.restype = [vp], vp
    lib.GDALRATGetColumnCount.argtypes, lib.GDALRATGetColumnCount.restype = [vp], ci
    lib.GDALRATGetRowCount.argtypes, lib.GDALRATGetRowCount.restype = [vp], ci
    lib.GDALRATGetNameOfCol.argtypes, lib.GDALRATGetNameOfCol.restype = [vp, ci], cp
    lib.GDALRATGetValueAsString.argtypes, lib.GDALRATGetValueAsString.restype = [vp, ci, ci], cp


def gdal_version() -> Optional[str]:
    lib = gdal_library()
    return lib.GDALVersionInfo(b"RELEASE_NAME").decode() if lib else None


def read_rat(path: str, band: int = 1) -> Optional[pd.DataFrame]:
    """Raster attribute table of ``path`` as a DataFrame (numeric columns converted), or None."""
    lib = gdal_library()
    if lib is None:
        return None
    lib.GDALAllRegister()
    ds = lib.GDALOpen(os.fsdecode(path).encode("utf-8"), 0)      # GA_ReadOnly; GDAL expects UTF-8 paths
    if not ds:
        return None
    try:
        b = lib.GDALGetRasterBand(ds, band)
        rat = lib.GDALGetDefaultRAT(b) if b else None
        if not rat:
            return None
        ncol, nrow = lib.GDALRATGetColumnCount(rat), lib.GDALRATGetRowCount(rat)
        if nrow == 0 or ncol == 0:
            return None
        cols = [lib.GDALRATGetNameOfCol(rat, i).decode("utf-8", "replace") for i in range(ncol)]
        data = {c: [(lib.GDALRATGetValueAsString(rat, r, i) or b"").decode("utf-8", "replace") for r in range(nrow)]
                for i, c in enumerate(cols)}
    finally:
        lib.GDALClose(ds)
    df = pd.DataFrame(data)
    for c in df.columns:
        conv = pd.to_numeric(df[c], errors="coerce")
        if conv.notna().all():
            df[c] = conv
    return df
