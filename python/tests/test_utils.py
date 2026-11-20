import numpy as np
import pytest

from utils.reclass import compute_breaks, fisher_jenks_breaks, reclass_by_breaks


def _ssd(v, b):
    c = np.digitize(v, b, right=True)
    return sum(((v[c == i] - v[c == i].mean()) ** 2).sum() for i in np.unique(c))


def test_fisher_jenks_finds_obvious_groups():
    v = np.r_[np.full(50, 1.0), np.full(50, 5.0), np.full(50, 9.0)]
    assert fisher_jenks_breaks(v, 3) == [1.0, 5.0]


def test_fisher_jenks_is_optimal_vs_mapclassify():
    mapclassify = pytest.importorskip("mapclassify")
    rng = np.random.default_rng(1)
    v = np.r_[rng.normal(0, 1, 700), rng.normal(6, 1, 800), rng.normal(12, 0.5, 500)]
    for k in (3, 5, 8):
        mine = fisher_jenks_breaks(v, k, max_sample=5000)
        ref = mapclassify.FisherJenks(v, k=k).bins[:-1]
        assert _ssd(v, mine) <= _ssd(v, ref) + 1e-9


def test_natural_breaks_large_input_is_fast():
    import time
    v = np.random.default_rng(0).lognormal(0, 1, 500_000)
    t = time.time()
    b = compute_breaks(v, "natural_breaks", 8)
    assert len(b) == 7 and np.all(np.diff(b) > 0) and time.time() - t < 15


def test_reclass_by_breaks_codes_and_table():
    arr = np.ma.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], mask=[[0, 0, 0], [0, 0, 1]])
    cls, table = reclass_by_breaks(arr, [1.5, 3.5])
    assert cls.tolist() == [[1, 1, 2], [2, 3, None]]
    assert table["count"].tolist() == [2, 2, 1]


def test_clip_by_mask_crop_invert_touched():
    import geopandas as gpd
    from rasterio.transform import from_origin
    from shapely.geometry import box

    from utils.raster_tools import Grid, clip_by_mask

    grid = Grid(from_origin(0, 100, 10, 10), None, 10, 10)          # 10 x 10 cells of 10 m
    data = np.ma.array(np.arange(100, dtype="int16").reshape(10, 10), mask=np.zeros((10, 10), bool))
    data.mask[2, 1] = True                                           # existing nodata stays nodata
    mask = gpd.GeoDataFrame(geometry=[box(15, 58, 52, 82)])        # covers cell centres x 25..45, y 65..75

    out, g = clip_by_mask(data, grid, mask)
    assert g.shape == (4, 5) and g.bounds == (10, 50, 60, 90)       # extent snapped outwards to the input grid
    assert (~out.mask).sum() == 6 and out[1, 1] == 22               # 3 x 2 centres inside; values unchanged
    assert out.mask[1, 0]                                            # input nodata kept masked
    out_t, _ = clip_by_mask(data, grid, mask, all_touched=True)
    assert (~out_t.mask).sum() == 4 * 5 - 1                         # every touched cell (minus the nodata one)
    full, g_full = clip_by_mask(data, grid, mask, crop=False)
    assert g_full == grid and (~full.mask).sum() == 6
    outside, g_out = clip_by_mask(data, grid, mask, invert=True)    # invert keeps the full extent
    assert g_out == grid and (~outside.mask).sum() == 100 - 1 - 6
    with pytest.raises(ValueError):
        clip_by_mask(data, grid, gpd.GeoDataFrame(geometry=[box(500, 500, 600, 600)]))


def test_symbology_classification():
    from core import symbology as sym
    rng = np.random.default_rng(0)
    v = np.r_[rng.normal(0, 1, 300), rng.normal(10, 1, 300)]
    b = sym.class_breaks(v, "jenks", 2)                       # natural break between the two groups
    assert len(b) == 3 and 2 < b[1] < 8
    assert sym.class_breaks(v, "equal", 4) == list(np.linspace(v.min(), v.max(), 5))
    q = sym.class_breaks(v, "quantile", 2)
    assert abs(q[1] - np.median(v)) < 1e-9
    cats = sym.build_categories(v, "equal", 3, "viridis")
    assert [c["count"] for c in cats] and sum(c["count"] for c in cats) == v.size
    idx = sym.assign_categories(np.array([v.min(), v.max(), 1e9]), cats)
    assert list(idx) == [0, 2, -1]                            # outside every range: no category
    u = sym.build_categories(["b", "a", "b", None], "unique")
    assert [(c["value"], c["count"]) for c in u] == [("a", 1), ("b", 2)]
    w = sym.build_categories([1, 2, 3], "unique", weights=[10, 5, 1], labels={1.0: "Q"})   # raster value counts
    assert [(c["label"], c["count"]) for c in w] == [("Q", 10), ("2", 5), ("3", 1)]
    try:
        sym.build_categories(np.arange(sym.MAX_UNIQUE + 1.0), "unique")
        raise AssertionError("too many unique values accepted")
    except ValueError:
        pass
    assert sym.ramp_colors("tab10", 12)[10] == sym.ramp_colors("tab10", 12)[0]   # qualitative ramps cycle


def test_read_vector_matches_geopandas(tmp_path):
    import geopandas as gpd
    from shapely.geometry import LineString, Point, Polygon
    from utils.vector_io import read_vector
    gdf = gpd.GeoDataFrame({"name": ["a", "b", "c"], "v": [1.5, 2.0, None], "n": [1, 2, 3]},
                           geometry=[Point(0, 0), Polygon([(0, 0), (1, 0), (1, 1)]), LineString([(0, 0), (2, 2)])],
                           crs="EPSG:32611")
    for name in ("t.gpkg", "t.shp"):
        p = str(tmp_path / name)
        (gdf.iloc[:1] if name.endswith(".shp") else gdf).to_file(p)
        a, b = read_vector(p), gpd.read_file(p)
        assert list(a.columns) == list(b.columns) and a.crs == b.crs and len(a) == len(b)
        assert all(a.geometry.geom_equals(b.geometry))
        for c in a.columns:
            if c != "geometry":
                assert a[c].astype(str).tolist() == b[c].astype(str).tolist()


def test_write_vector_shapefile_for_arcmap(tmp_path):
    import os
    import pandas as pd
    import geopandas as gpd
    from shapely.geometry import Point
    from utils.vector_io import read_vector, write_vector
    gdf = gpd.GeoDataFrame({"deposit_name_long": ["金矿 A", "Mine B"], "ok": [True, False],
                            "found": pd.to_datetime(["2001-02-03", "1999-12-31"]), "grade": [1.25, None],
                            "split": ["training", "test"]},
                           geometry=[Point(560000, 4516000), Point(561000, 4517000)], crs="EPSG:32611")
    p = str(tmp_path / "pts.shp")
    write_vector(gdf, p)
    for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
        assert os.path.exists(str(tmp_path / ("pts" + ext))), ext              # ArcMap needs .prj; .cpg = UTF-8
    back = read_vector(p)
    assert back.crs == gdf.crs and len(back) == 2 and all(back.geometry.geom_equals(gdf.geometry))
    assert "deposit_na" in back.columns                                        # .dbf names: 10 characters
    assert back["deposit_na"].tolist() == ["金矿 A", "Mine B"]                  # Unicode kept
    assert back["ok"].tolist() == [1, 0] and back["found"].tolist()[0].startswith("2001-02-03")
    assert back["split"].tolist() == ["training", "test"]
    write_vector(gdf.iloc[:1], p)                                              # overwrite cleanly
    assert len(read_vector(p)) == 1
    g = str(tmp_path / "pts.gpkg")
    assert len(read_vector(write_vector(gdf, g))) == 2


def test_write_raster_widens_type_for_nodata(tmp_path):
    import rasterio
    from rasterio.transform import from_origin
    from utils.raster_tools import Grid, nodata_dtype, write_raster
    assert nodata_dtype("uint8", -9999) == np.dtype("int16") and nodata_dtype("float32", -9999) == np.float32
    g = Grid(from_origin(0, 10, 1, 1), None, 3, 3)
    a = np.ma.array(np.arange(9, dtype="uint8").reshape(3, 3), mask=np.eye(3, dtype=bool))
    p = write_raster(str(tmp_path / "c.tif"), a, g, nodata=-9999)          # 8-bit classes, project NoData
    with rasterio.open(p) as ds:
        assert ds.nodata == -9999 and ds.dtypes[0] == "int16"
        m = ds.read(1, masked=True)
    assert (m.mask == a.mask).all() and (m.compressed() == a.compressed()).all()

