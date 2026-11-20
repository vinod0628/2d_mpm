"""Distance, azimuth and area measurement on the map (GUI-free).

Projected CRS -> planar geometry in the CRS units converted to metres; geographic CRS (degrees) -> geodesic
(ellipsoidal) lengths, azimuths and areas with pyproj.Geod; no CRS -> plain map units.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from i18n import tr

Point = Tuple[float, float]


class Measurer:
    def __init__(self, crs=None):
        self.crs = crs
        self.geod = None                  # set for geographic CRSs (lengths / areas on the ellipsoid)
        self.az_geod = None               # ellipsoid used for geodetic azimuths (any CRS with a datum)
        self.to_lonlat = None             # projected -> geographic transformer (geodetic azimuth)
        self.unit_factor = 1.0            # metres per map unit (projected)
        self.has_units = crs is not None
        if crs is not None:
            try:
                from pyproj import CRS as PCRS, Transformer
                p = PCRS.from_user_input(crs.to_wkt() if hasattr(crs, "to_wkt") else crs)
                if p.is_geographic:
                    self.geod = self.az_geod = p.get_geod()
                else:
                    self.unit_factor = float(p.axis_info[0].unit_conversion_factor or 1.0)
                    self.az_geod = p.get_geod()
                    self.to_lonlat = Transformer.from_crs(p, p.geodetic_crs, always_xy=True)
            except Exception:  # noqa: BLE001
                self.has_units = self.has_units and self.geod is not None

    # ------------------------------------------------------------------ lengths / angles
    def segment(self, p: Point, q: Point) -> float:
        """Length of p->q in metres (map units if the CRS is unknown)."""
        if self.geod is not None:
            _az, _baz, d = self.geod.inv(p[0], p[1], q[0], q[1])
            return float(d)
        return math.hypot(q[0] - p[0], q[1] - p[1]) * self.unit_factor

    def azimuth(self, p: Point, q: Point) -> float:
        """Geodetic azimuth p->q: degrees clockwise from TRUE north (0 = N, 90 = E), 0-360, on the CRS ellipsoid.

        Projected CRS: both points are converted to longitude/latitude of the CRS datum first, so the value
        includes the meridian convergence (equal to the grid azimuth on the projection's central meridian).
        Without a CRS the grid azimuth is returned.
        """
        if self.az_geod is not None:
            if self.to_lonlat is not None:
                (lon1, lon2), (lat1, lat2) = self.to_lonlat.transform([p[0], q[0]], [p[1], q[1]])
            else:
                lon1, lat1, lon2, lat2 = p[0], p[1], q[0], q[1]
            az, _baz, _d = self.az_geod.inv(lon1, lat1, lon2, lat2)
            return float(az % 360.0)
        return self.grid_azimuth(p, q)

    @staticmethod
    def grid_azimuth(p: Point, q: Point) -> float:
        """Direction p->q clockwise from grid north (the map's up direction), 0-360."""
        return math.degrees(math.atan2(q[0] - p[0], q[1] - p[1])) % 360.0

    def path_length(self, pts: Sequence[Point]) -> float:
        return float(sum(self.segment(a, b) for a, b in zip(pts[:-1], pts[1:])))

    # ------------------------------------------------------------------ area
    def area_perimeter(self, pts: Sequence[Point]) -> Tuple[float, float]:
        """(area m², perimeter m) of the closed polygon through ``pts``."""
        if len(pts) < 3:
            return 0.0, self.path_length(list(pts) + list(pts[:1])) if len(pts) == 2 else 0.0
        if self.geod is not None:
            lons, lats = zip(*pts)
            a, per = self.geod.polygon_area_perimeter(lons, lats)
            return abs(float(a)), float(per)
        s = 0.0
        for (x1, y1), (x2, y2) in zip(pts, list(pts[1:]) + [pts[0]]):
            s += x1 * y2 - x2 * y1
        area = abs(s) / 2.0 * self.unit_factor ** 2
        return area, self.path_length(list(pts) + [pts[0]])

    # ------------------------------------------------------------------ formatting
    def fmt_length(self, m: float) -> str:
        if not self.has_units:
            return tr("{v:,.3f} map units", v=m)
        return f"{m:,.1f} m" if m < 1000 else f"{m / 1000:,.3f} km"

    def fmt_area(self, m2: float) -> List[str]:
        if not self.has_units:
            return [tr("{v:,.3f} map units²", v=m2)]
        return [f"{m2 / 1e6:,.4f} km²", f"{m2 / 1e4:,.2f} ha", f"{m2:,.0f} m²"]

    @staticmethod
    def fmt_angle(deg: Optional[float]) -> str:
        if deg is None:
            return "—"
        dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        return f"{deg:.2f}°  ({dirs[int((deg + 22.5) // 45) % 8]})"
