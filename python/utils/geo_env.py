"""Offline coordinate-system set-up: make GDAL / PROJ use the EPSG database shipped with the application.

rasterio, fiona and pyproj each bundle their own ``proj.db`` (EPSG database) and GDAL data. rasterio and fiona use
the directory in the environment variables ``PROJ_DATA`` / ``PROJ_LIB`` (and ``GDAL_DATA``) *before* their own copy.
Other GIS software (PostgreSQL/PostGIS, QGIS, ArcGIS, GRASS, OSGeo4W, Anaconda …) often sets these variables for the
whole computer, pointing at its own, older or incompatible database. GDAL then cannot read the EPSG code of a
GeoTIFF and returns a datum-less "engineering" CRS with the right name (e.g. ``LOCAL_CS["GDA94 / MGA zone 54"]``),
which cannot be transformed to anything ("Cannot find coordinate operations …"). Nothing is downloaded — the
lookup simply fails locally.

:func:`isolate` (called first thing in ``main.py`` and the PyInstaller runtime hook, before any geo library is
imported) removes such foreign variables, so every library uses its bundled database; ``PROJ_NETWORK`` stays OFF.
:func:`crs_database_problem` checks after start-up that EPSG codes resolve; :func:`unrecognized` flags layer CRSs
that cannot be used (such layers are loaded as being in the current project CRS).
"""
from __future__ import annotations

import os
import sys
from typing import List, Optional

VARS = ("PROJ_DATA", "PROJ_LIB", "GDAL_DATA")


def _app_root() -> str:
    """Folder of the installation (frozen exe folder or the Python environment)."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.abspath(sys.prefix)


def _inside(path: str, root: str) -> bool:
    try:
        return os.path.commonpath([os.path.abspath(path).lower(), root.lower()]) == root.lower()
    except ValueError:                     # different drives
        return False


def isolate(environ=os.environ) -> List[str]:
    """Remove PROJ / GDAL data variables that point outside this installation; returns notes for the log.
    ``MPM_KEEP_PROJ_ENV=1`` keeps them (expert override)."""
    environ["PROJ_NETWORK"] = "OFF"                  # never download transformation grids
    if environ.get("MPM_KEEP_PROJ_ENV") == "1":
        return []
    root = _app_root()
    notes = []
    for v in VARS:
        val = environ.get(v)
        if val and not all(_inside(p, root) for p in val.split(os.pathsep) if p):
            notes.append(f"{v}={val} (set by other software) ignored — the application's own EPSG / GDAL data "
                         "are used")
            del environ[v]
    return notes


def crs_database_problem() -> Optional[str]:
    """None if GDAL (rasterio) and pyproj resolve EPSG codes from their local databases, else an explanation."""
    problems = []
    try:
        from rasterio.crs import CRS
        if CRS.from_epsg(28354).to_epsg() != 28354 or not CRS.from_epsg(4326).is_geographic:
            problems.append("GDAL/rasterio")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"GDAL/rasterio ({exc})")
    try:
        from pyproj import CRS as PCRS
        if PCRS.from_epsg(28354).to_epsg() != 28354:
            problems.append("pyproj")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"pyproj ({exc})")
    if not problems:
        return None
    env = ", ".join(f"{v}={os.environ[v]}" for v in VARS if v in os.environ) or "no PROJ/GDAL variables"
    return ("The local coordinate-system database (proj.db) cannot be used by " + " and ".join(problems)
            + f" ({env}). Rasters may get an unknown CRS. Reinstall the application or remove PROJ_LIB / PROJ_DATA "
            "set by other GIS software.")


def unrecognized(crs) -> bool:
    """True if a layer's CRS cannot be used for transformations: missing, unparsable, an engineering / LOCAL_CS
    CRS without a datum (what GDAL returns when it cannot identify the stored coordinate system) or GDAL's
    "Undefined … SRS" placeholder. Such layers are loaded as being in the current project CRS."""
    if crs is None:
        return True
    try:
        from pyproj import CRS as PCRS
        c = PCRS.from_user_input(crs.to_wkt() if hasattr(crs, "to_wkt") else crs)
    except Exception:  # noqa: BLE001
        return True
    if c.type_name.startswith("Engineering") or c.datum is None:
        return True
    # placeholder written by GDAL / GeoPackage for "no CRS": "Undefined geographic / cartesian SRS", datum "unknown"
    return c.name.strip().lower().startswith("undefined") or c.datum.name.strip().lower() in ("unknown", "")
