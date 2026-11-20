"""Popular coordinate systems for the project CRS picker: datums (geographic CRS), UTM zones, projected CRS.

* ``DATUMS``     — the most used geodetic datums with their geographic CRS (GCS) EPSG code.
* ``utm_crs()``  — UTM zone on any of these datums: the official EPSG code when there is one (e.g. WGS 84 / UTM zone
                   50N = EPSG:32650, GDA94 / MGA zone 54 = EPSG:28354), otherwise a UTM definition on that datum.
* ``PROJECTED``  — popular non-UTM projected CRS (web, world, continental and national grids).
* ``classify()`` — which of these a CRS is (to show an existing project CRS in the picker).

Offline: everything is read from the local EPSG database ``proj.db`` installed with pyproj (and bundled in the
PyInstaller release); no internet access is used (main.py also sets PROJ_NETWORK=OFF).
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict, Optional, Tuple

from rasterio.crs import CRS

# (label, GCS EPSG code) — datums of the geographic coordinate systems, WGS 84 first
DATUMS = [
    ("WGS 84", 4326), ("WGS 72", 4322), ("NAD83", 4269), ("NAD27", 4267), ("ETRS89", 4258), ("ED50", 4230),
    ("OSGB36", 4277), ("RGF93", 4171), ("GDA2020", 7844), ("GDA94", 4283), ("NZGD2000", 4167),
    ("CGCS2000", 4490), ("Beijing 1954", 4214), ("Xian 1980", 4610), ("Pulkovo 1942", 4284), ("JGD2011", 6668),
    ("Tokyo", 4301), ("Korea 2000", 4737), ("Indian 1975", 4240), ("SIRGAS 2000", 4674), ("SAD69", 4618),
    ("Hartebeesthoek94", 4148), ("Arc 1960", 4210), ("Adindan", 4201),
]

# popular projected CRS other than UTM
PROJECTED = [
    "EPSG:3857", "EPSG:3395", "EPSG:4087", "ESRI:54009", "ESRI:54030", "EPSG:6933",          # world
    "EPSG:3035", "EPSG:3034", "EPSG:2154", "EPSG:27700",                                     # Europe
    "EPSG:5070", "EPSG:3978",                                                                # North America
    "EPSG:5880",                                                                             # South America
    "EPSG:9473", "EPSG:3577", "EPSG:3112", "EPSG:2193",                                      # Australia, NZ
    "ESRI:102025", "EPSG:4547",                                                              # Asia, China
    "EPSG:3031", "EPSG:3413",                                                                # polar
]

_UTM_NAME = re.compile(r"^(.*) / (?:UTM zone (\d+)([NS])|MGA zone (\d+))$")


@lru_cache(maxsize=1)
def _utm_codes() -> Tuple[Dict[Tuple[str, int, str], str], Dict[str, Tuple[str, int, str]]]:
    """EPSG UTM / MGA projected CRS by (datum name, zone, hemisphere) and the reverse map (≈0.4 s, cached)."""
    from pyproj.database import query_crs_info
    from pyproj.enums import PJType
    fwd, rev = {}, {}
    for info in query_crs_info(auth_name="EPSG", pj_types=PJType.PROJECTED_CRS, allow_deprecated=False):
        m = _UTM_NAME.match(info.name)
        if not m:
            continue
        datum = m.group(1)
        zone, hemi = (int(m.group(2)), m.group(3)) if m.group(2) else (int(m.group(4)), "S")
        key = (datum, zone, hemi)
        if key not in fwd or "UTM" in info.name:        # prefer "UTM zone" over "MGA zone" for the same datum
            fwd[key] = f"EPSG:{info.code}"
        rev[f"EPSG:{info.code}"] = key
    return fwd, rev


@lru_cache(maxsize=None)
def gcs_name(epsg: int) -> str:
    from pyproj import CRS as PCRS
    return PCRS.from_epsg(epsg).name


@lru_cache(maxsize=None)
def crs_name(code: str) -> str:
    from pyproj import CRS as PCRS
    try:
        return PCRS.from_user_input(code).name
    except Exception:  # noqa: BLE001
        return code


def datum_label(label: str, epsg: int) -> str:
    return f"{label}  (EPSG:{epsg})"


def projected_label(code: str) -> str:
    return f"{code} — {crs_name(code)}"


def utm_zones():
    """All 120 UTM zones as signed numbers −60 … −1, +1 … +60 (−50 = 50S, +50 = 50N): (zone, hemisphere, label).

    The label gives the signed zone, the zone with its hemisphere letter and the 6° longitude band.
    """
    def lon(v):
        return f"{abs(v)}°{'E' if v > 0 else 'W' if v < 0 else ''}"
    out = []
    for signed in [-z for z in range(60, 0, -1)] + list(range(1, 61)):
        z, hemi = abs(signed), ("N" if signed > 0 else "S")
        west = -180 + (z - 1) * 6
        out.append((z, hemi, f"{'+' if signed > 0 else '−'}{z}  ({z}{hemi})    {lon(west)} – {lon(west + 6)}"))
    return out


def utm_crs(gcs_epsg: int, zone: int, hemisphere: str) -> CRS:
    """UTM zone on the datum of ``gcs_epsg``; hemisphere 'N' or 'S'."""
    if not 1 <= int(zone) <= 60 or hemisphere not in ("N", "S"):
        raise ValueError(f"Invalid UTM zone {zone}{hemisphere}")
    code = _utm_codes()[0].get((gcs_name(gcs_epsg), int(zone), hemisphere))
    if code:
        return CRS.from_user_input(code)
    from pyproj import CRS as PCRS
    from pyproj.crs import ProjectedCRS
    from pyproj.crs.coordinate_operation import UTMConversion
    p = ProjectedCRS(name=f"{gcs_name(gcs_epsg)} / UTM zone {int(zone)}{hemisphere}",
                     conversion=UTMConversion(int(zone), hemisphere), geodetic_crs=PCRS.from_epsg(gcs_epsg))
    return CRS.from_wkt(p.to_wkt())


def classify(crs) -> dict:
    """Describe ``crs`` for the picker.

    kind = 'none' | 'geographic' (gcs) | 'utm' (gcs, zone, hemisphere) | 'projected' (code) | 'custom'.
    """
    if crs is None:
        return dict(kind="none")
    from pyproj import CRS as PCRS
    p = PCRS.from_user_input(crs.to_wkt() if hasattr(crs, "to_wkt") else crs)
    auth = p.to_authority(min_confidence=100)
    code = f"{auth[0]}:{auth[1]}" if auth else None
    gcs_codes = {e for _l, e in DATUMS}
    if p.is_geographic and code and code.startswith("EPSG:") and int(code[5:]) in gcs_codes:
        return dict(kind="geographic", gcs=int(code[5:]))
    if code in PROJECTED:
        return dict(kind="projected", code=code)
    by_name = {gcs_name(e): e for e in gcs_codes}
    if code and code in _utm_codes()[1]:
        datum, zone, hemi = _utm_codes()[1][code]
        if datum in by_name:
            return dict(kind="utm", gcs=by_name[datum], zone=zone, hemisphere=hemi)
    if p.is_projected and p.utm_zone and p.geodetic_crs is not None and p.geodetic_crs.name in by_name:
        z = p.utm_zone
        return dict(kind="utm", gcs=by_name[p.geodetic_crs.name], zone=int(z[:-1]), hemisphere=z[-1])
    return dict(kind="custom", code=code)
