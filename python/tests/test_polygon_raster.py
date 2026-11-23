"""Polygon to categorical raster (utils/polygon_raster.py)."""
import os
import sys

import geopandas as gpd
import numpy as np
import pytest
from rasterio.transform import from_origin
from shapely.geometry import box

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.polygon_raster import class_codes, polygons_to_raster, snapped_grid  # noqa: E402
from utils.raster_tools import Grid  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CURNAMONA = os.path.join(ROOT, "demo_data", "curnamona_ree", "processed")

GRID = Grid(from_origin(0, 10, 1, 1), None, 10, 10)          # 10 × 10 cells of 1 unit, x 0 … 10, y 0 … 10


def _units():
    """Granite (left half), shale (top right), a second granite polygon; bottom right is a gap."""
    return gpd.GeoDataFrame({"unit": ["granite", "shale", "granite"], "code": [7, 3, 7],
                             "host": ["no", "yes", "no"], "age": [1.0, 2.0, 1.5]},
                            geometry=[box(0, 0, 5, 10), box(5, 5, 10, 10), box(8, 0, 9, 1)])


def test_classes_codes_and_table():
    r = polygons_to_raster(_units(), GRID, "unit")
    d = np.ma.filled(r.data, 0)
    assert d[0, 0] == 1 and d[0, 9] == 2 and d[9, 8] == 1        # granite = 1, shale = 2 (sorted labels)
    assert d[9, 6] == 3 and r.gap_code == 3                      # gap inside the study area -> class 3 "Other"
    assert r.data.dtype == np.uint8 and r.nodata == 0 and not r.data.mask.any()
    t = r.table.set_index("label")
    assert t.loc["granite", "count"] == 51 and t.loc["shale", "count"] == 25 and t.loc["Other", "count"] == 24
    assert list(t.index) == ["granite", "shale", "Other"]
    # attributes constant within every class are copied, varying ones (age) are not
    assert t.loc["granite", "host"] == "no" and t.loc["granite", "code"] == 7 and "age" not in t.columns


def test_field_codes_gap_nodata_and_study_mask():
    study = np.zeros(GRID.shape, bool)
    study[:, 3:] = True                                          # columns 3 … 9 only
    r = polygons_to_raster(_units(), GRID, "code", codes="field", study=study, gap_label=None)
    d = r.data
    assert d[0, 3] == 7 and d[0, 9] == 3 and r.gap_code is None
    assert d.mask[0, 0] and d.mask[9, 6] and not d.mask[9, 8]    # outside study and gaps -> NoData
    assert set(r.table["value"]) == {3, 7}
    with pytest.raises(ValueError):
        class_codes(_units()["unit"], "field")                   # text cannot be a code
    with pytest.raises(ValueError):
        class_codes(_units()["age"], "field")                    # 1.5 is not a whole number
    z = _units().assign(code=[0, -2, 0])                         # codes ≤ 0: project NoData, int32
    rz = polygons_to_raster(z, GRID, "code", codes="field")
    assert rz.data.dtype == np.int32 and rz.nodata == -9999 and rz.data[0, 0] == 0 and rz.gap_code == 1


def test_cell_centre_vs_maximum_area():
    # cell x 2 … 3 of the bottom row: c covers 48 %, b 38 %, the narrow a 14 % but contains the centre 2.5
    g = gpd.GeoDataFrame({"u": ["c", "a", "b"]},
                         geometry=[box(0, 0, 2.48, 1), box(2.48, 0, 2.62, 1), box(2.62, 0, 10, 1)])
    c = polygons_to_raster(g, GRID, "u", gap_label=None)
    m = polygons_to_raster(g, GRID, "u", assignment="max_area", gap_label=None, subcells=10, block_rows=3)
    assert c.data[9, 2] == 1 and m.data[9, 2] == 3               # centre: a (code 1); maximum area: c (code 3)
    assert (c.data[9, 3:] == 2).all() and (m.data[9, 3:] == 2).all() and m.data[9, 0] == 3
    sliver = gpd.GeoDataFrame({"u": ["s"]}, geometry=[box(0.05, 0.05, 0.3, 0.3)])
    rs = polygons_to_raster(sliver, GRID, "u", gap_label="none")
    assert rs.empty == ["s"] and list(rs.table["label"]) == ["none"]   # polygon smaller than a cell reported
    rm = polygons_to_raster(sliver, GRID, "u", assignment="max_area", gap_label=None, subcells=8)
    assert rm.data[9, 0] == 1 and rm.data.count() == 1           # maximum area keeps it


def test_single_class_skipped_and_crs():
    g = _units()
    g.loc[1, "unit"] = None
    r = polygons_to_raster(g, GRID, "unit", gap_label="Other")
    assert r.skipped == 1 and set(r.table.label) == {"granite", "Other"}
    one = polygons_to_raster(_units(), GRID, None, single_label="intrusive", gap_label="outside")
    assert list(one.table.label) == ["intrusive", "outside"] and set(np.unique(one.data)) == {1, 2}


def test_snapped_grid():
    s = snapped_grid((103, 207, 451, 389), 50, origin=(10, 1000))   # cell lines through x = 10 + 50k
    left, bottom, right, top = s.bounds
    assert (left, right) == (60, 460) and (bottom, top) == (200, 400) and s.shape == (4, 8)
    assert snapped_grid((0, 0, 100, 100), 25).shape == (4, 4)


@pytest.mark.skipif(not os.path.isdir(CURNAMONA), reason="demo data missing")
def test_curnamona_geology():
    from utils.raster_tools import read_raster
    mask, grid, _ = read_raster(os.path.join(CURNAMONA, "rasters", "study_mask.tif"))
    meso = gpd.read_file(os.path.join(CURNAMONA, "vectors", "mesoproterozoic.gpkg"))
    study = ~np.ma.getmaskarray(mask) & (np.ma.filled(mask, 0) > 0)
    r = polygons_to_raster(meso, grid, "MAPUNIT", study=study, gap_label="Other")
    assert r.data.count() == study.sum()                         # every study cell has a class
    assert (r.data.mask == ~study).all()
    assert "STRATNAME" in r.table.columns and r.table["count"].sum() == study.sum()
    assert r.table.label.iloc[-1] == "Other" and len(r.table) == meso.MAPUNIT.nunique() + 1 - len(r.empty)
