import pytest
from rasterio.crs import CRS

from utils.measure import Measurer


def test_projected_lengths_azimuth_area():
    m = Measurer(CRS.from_epsg(28354))                      # metres
    assert m.segment((0, 0), (3000, 4000)) == pytest.approx(5000)
    # geodetic azimuth on the central meridian of MGA54 (easting 500 000) equals the grid azimuth
    o = (500000.0, 6650000.0)
    assert m.azimuth(o, (o[0], o[1] + 100)) == pytest.approx(0, abs=1e-6)          # north
    assert m.azimuth(o, (o[0] + 100, o[1])) == pytest.approx(90, abs=0.01)         # east
    assert m.azimuth(o, (o[0] - 100, o[1] - 100)) == pytest.approx(225, abs=0.01)  # south-west
    assert m.grid_azimuth((0, 0), (0, 100)) == 0 and m.grid_azimuth((0, 0), (100, 0)) == 90
    # 200 km east of the central meridian true north differs from grid north (meridian convergence ~1.2 deg)
    e = (700000.0, 6650000.0)
    conv = m.azimuth(e, (e[0], e[1] + 100))
    assert 0.5 < min(conv, 360 - conv) < 2.0
    sq = [(0, 0), (1000, 0), (1000, 1000), (0, 1000)]
    area, per = m.area_perimeter(sq)
    assert area == pytest.approx(1e6) and per == pytest.approx(4000)
    assert m.fmt_length(5000) == "5.000 km" and m.fmt_area(1e6)[0] == "1.0000 km²"


def test_feet_crs_unit_factor():
    m = Measurer(CRS.from_epsg(2272))                       # US survey feet
    assert m.unit_factor == pytest.approx(0.3048006, rel=1e-5)
    assert m.segment((0, 0), (1000, 0)) == pytest.approx(304.8006, rel=1e-5)


def test_geographic_uses_geodesics():
    m = Measurer(CRS.from_epsg(4326))
    d = m.segment((0, 0), (1, 0))                           # 1 degree of longitude at the equator
    assert d == pytest.approx(111319.49, rel=1e-4)
    assert m.azimuth((0, 0), (1, 0)) == pytest.approx(90)
    area, _ = m.area_perimeter([(0, 0), (1, 0), (1, 1), (0, 1)])
    assert area == pytest.approx(1.2309e10, rel=1e-3)       # ~12 309 km² (1° x 1° cell at the equator)


def test_no_crs_map_units():
    m = Measurer(None)
    assert m.segment((0, 0), (0, 2)) == 2 and "map units" in m.fmt_length(2)
