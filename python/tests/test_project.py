"""Folder-based projects: new / open / save / save as / results run folders (no GUI dialogs)."""
import json
import os
import warnings

import numpy as np
import pytest

from conftest import CARLIN

warnings.filterwarnings("ignore")
need_data = pytest.mark.skipif(not os.path.isdir(CARLIN), reason="run utils.prepare_demo_data first")


@pytest.fixture(scope="module")
def qapp():
    from PyQt5.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_folder_helpers(tmp_path):
    from core import project
    assert project.is_empty_folder(str(tmp_path))
    (tmp_path / "desktop.ini").write_text("x")
    assert project.is_empty_folder(str(tmp_path))            # system files are ignored
    (tmp_path / "a.txt").write_text("x")
    assert not project.is_empty_folder(str(tmp_path))
    assert project.safe_name("WofE posterior (v2)!") == "WofE_posterior_v2"
    assert len(project.timestamp()) == len("1118_153012")
    p = tmp_path / "a.txt"
    assert project.unique_path(str(p)).endswith("a_2.txt")


def test_new_save_save_as(tmp_path, qapp):
    from core import project
    from core.layers import LayerStore, RasterLayer
    from utils.raster_tools import Grid

    store = LayerStore()
    path = project.create_project(store, str(tmp_path / "p1"))
    assert os.path.isfile(path) and os.path.basename(path) == project.PROJECT_FILE
    assert (tmp_path / "p1" / "data").is_dir() and (tmp_path / "p1" / "results").is_dir()
    assert not store.dirty

    grid = Grid.from_bounds(0, 0, 1000, 1000, res=100, crs="EPSG:28354")
    store.add(RasterLayer("in memory layer", np.ma.array(np.arange(100.0).reshape(10, 10)), grid))
    assert store.dirty and project.unsaved_layers(store)
    written = project.save_project(store)                     # no dialog: in-memory -> data/
    assert len(written) == 1 and written[0].endswith(os.path.join("data", "in_memory_layer.tif"))
    assert not store.dirty and not project.unsaved_layers(store)
    proj = json.loads((tmp_path / "p1" / project.PROJECT_FILE).read_text())
    assert proj["layers"][0]["path"] == "data/in_memory_layer.tif"     # relative inside the project

    run_id, run_dir = project.new_run(store, "wofe")
    assert run_id.startswith("wofe_") and os.path.isdir(run_dir) and "results" in run_dir

    project.save_project_as(store, str(tmp_path / "p2"))      # copy whole folder, continue in the copy
    assert store.project_path == str(tmp_path / "p2" / project.PROJECT_FILE)
    assert store.layers[0].source.startswith(str(tmp_path / "p2"))
    assert (tmp_path / "p2" / "results" / run_id).is_dir()
    assert (tmp_path / "p1" / "data" / "in_memory_layer.tif").is_file()   # original untouched

    store2 = LayerStore()
    assert project.open_project(store2, str(tmp_path / "p2")) == []
    assert store2.layers[0].name == "in memory layer"
    with pytest.raises(ValueError):
        project.save_project_as(store, str(tmp_path / "p2" / "inside"))


@need_data
def test_copy_demo_project(tmp_path, qapp):
    from core import project
    from core.layers import LayerStore

    path, dropped = project.copy_project(CARLIN, str(tmp_path / "demo"))
    assert os.path.basename(path) == project.PROJECT_FILE
    store = LayerStore()
    assert project.open_project(store, str(tmp_path / "demo")) == []
    assert store.by_name("deposits") is not None
    inside = [l for l in store.layers if l.source and l.source.startswith(str(tmp_path))]
    assert len(inside) >= 21                                   # demo layers point to the copies


def test_open_requires_exactly_one_project_file(tmp_path, qapp):
    from core import project
    from core.layers import LayerStore

    store = LayerStore()
    empty = tmp_path / "none"
    empty.mkdir()
    (empty / "settings.json").write_text("{}")               # other json files do not count
    with pytest.raises(project.ProjectFileError, match="no project file"):
        project.open_project(store, str(empty))

    two = tmp_path / "two"
    project.create_project(store, str(two))
    (two / "other.mpm.json").write_text((two / project.PROJECT_FILE).read_text())
    with pytest.raises(project.ProjectFileError, match="more than one project file"):
        project.open_project(store, str(two))

    # creating / copying a project into that folder replaces the other project files -> exactly one again
    project.create_project(store, str(two))
    assert [os.path.basename(f) for f in project.project_files(str(two))] == [project.PROJECT_FILE]
    project.open_project(store, str(two))


def test_open_project_alerts_in_gui(tmp_path, qapp, monkeypatch):
    from PyQt5.QtWidgets import QMessageBox
    from gui.main_window import MainWindow

    warned = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a[2]))
    win = MainWindow()
    win.open_project(str(tmp_path))                            # folder without project file
    assert warned and "no project file" in warned[0] and win.store.project_path is None
    assert not win.a_add_r.isEnabled()
    win.close()


def test_project_crs_and_cell_size(tmp_path, qapp):
    """CRS / cell size: editable without layers, taken from the first layer / raster when unset, saved, then fixed."""
    import geopandas as gpd
    from rasterio.crs import CRS
    from rasterio.transform import from_origin
    from shapely.geometry import box

    from core import project
    from core.layers import LayerStore, RasterLayer, VectorLayer, format_length
    from utils.raster_tools import Grid

    def raster(name, res, crs="EPSG:32611"):
        g = Grid(from_origin(500000, 4500000, res, res), CRS.from_user_input(crs), 20, 10)
        return RasterLayer(name, np.ma.ones((10, 20)), g)

    # unset -> first layer gives the CRS, first RASTER layer the cell size
    store = LayerStore()
    path = project.create_project(store, str(tmp_path / "p"))
    assert store.crs is None and store.cell_size is None and not store.settings_locked
    v = VectorLayer("aoi", gpd.GeoDataFrame(geometry=[box(500000, 4499000, 501000, 4500000)], crs="EPSG:32611"))
    store.add(v)
    assert store.crs.to_epsg() == 32611 and store.cell_size is None and store.settings_locked
    with pytest.raises(RuntimeError):
        store.set_project_settings(CRS.from_epsg(4326), 10)
    store.add(raster("a", 50))
    assert store.cell_size == 50 and format_length(store.cell_size, store.crs) == "50 m"
    assert store.extent == (500000, 4499500, 501000, 4500000)              # exactly the first raster
    g = store.project_grid()
    assert (g.width, g.height, g.res) == (20, 10, (50.0, 50.0)) and not store.grid_mismatch(store.by_name("a"))
    b = store.add(raster("b", 25))
    assert store.cell_size == 50 and store.cell_size_mismatch(b) and store.grid_mismatch(b)
    assert not store.cell_size_mismatch(store.by_name("a"))
    for l in list(store.layers):
        l.save(str(tmp_path / "p" / "data" / f"{l.name}{'.tif' if l.kind == 'raster' else '.gpkg'}"))
    project.save_project(store)
    with open(path, encoding="utf-8") as fh:
        saved = json.load(fh)
    assert saved["cell_size"] == 50 and saved["extent"] == [500000, 4499500, 501000, 4500000]

    # reopen: kept; removing every layer keeps the settings but makes them editable again
    s2 = LayerStore()
    project.open_project(s2, path)
    assert s2.cell_size == 50 and s2.crs.to_epsg() == 32611 and s2.settings_locked and s2.extent == store.extent
    for l in list(s2.layers):
        s2.remove(l)
    assert s2.cell_size == 50 and s2.crs.to_epsg() == 32611 and not s2.settings_locked

    # set before any data: kept when layers are added (the raster is reprojected to the project CRS)
    s3 = LayerStore()
    project.create_project(s3, str(tmp_path / "p3"))
    s3.set_project_settings(CRS.from_epsg(32612), 100.0)
    tif = raster("c", 30).save(str(tmp_path / "c.tif"))
    lyr = s3.add_raster_file(tif)
    assert s3.crs.to_epsg() == 32612 and lyr.grid.crs.to_epsg() == 32612 and s3.cell_size == 100
    assert s3.cell_size_mismatch(lyr)
    l, b_, r, t = s3.extent                                 # first raster extent snapped to the 100 m cell size
    assert all(v % 100 == 0 for v in s3.extent) and l <= lyr.grid.bounds[0] and t >= lyr.grid.bounds[3]

    # conform_raster: one resampling step onto the project grid (any CRS / cell size / extent)
    from core.layers import conform_raster
    s5 = LayerStore()
    s5.set_project_settings(CRS.from_epsg(32611), 100.0, (500000, 4499000, 502000, 4500000))
    src = raster("d", 30)
    out = conform_raster(src, s5.project_grid())
    assert out.grid.same_as(s5.project_grid()) and out.source is None
    assert (out.grid.width, out.grid.height) == (20, 10)
    assert np.ma.count(out.data) > 0 and out.data.mask[:, 10:].all()       # outside the source raster: NoData
    geo = RasterLayer("e", np.ma.ones((50, 50)), Grid(from_origin(-117.0, 40.65, 0.001, 0.001), CRS.from_epsg(4326), 50, 50))
    out2 = conform_raster(geo, s5.project_grid())
    assert out2.grid.same_as(s5.project_grid())

    # older project files without "cell_size": taken from the bottom raster when opened
    with open(path, encoding="utf-8") as fh:
        proj = json.load(fh)
    proj.pop("cell_size")
    proj.pop("extent")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(proj, fh)
    s4 = LayerStore()
    project.open_project(s4, path)
    assert s4.cell_size == s4.rasters()[-1].grid.res[0] and s4.extent == s4.rasters()[-1].grid.bounds


def test_crs_catalog_offline():
    from rasterio.crs import CRS
    from pyproj import network
    from utils import crs_catalog as cat

    assert not network.is_network_enabled()                       # local proj.db only
    assert cat.DATUMS[0] == ("WGS 84", 4326)
    assert cat.utm_crs(4326, 50, "N").to_epsg() == 32650
    assert cat.utm_crs(4326, 33, "S").to_epsg() == 32733
    assert cat.utm_crs(4283, 54, "S").to_epsg() == 28354           # GDA94 / MGA zone 54
    assert cat.utm_crs(4269, 11, "N").to_epsg() == 26911
    cg = cat.utm_crs(4490, 50, "N")                                # no EPSG UTM for CGCS2000: defined on the datum
    assert cg.to_epsg() is None and cat.classify(cg) == dict(kind="utm", gcs=4490, zone=50, hemisphere="N")
    assert cat.classify(None) == dict(kind="none")
    assert cat.classify(CRS.from_epsg(4326)) == dict(kind="geographic", gcs=4326)
    assert cat.classify(CRS.from_epsg(28354)) == dict(kind="utm", gcs=4283, zone=54, hemisphere="S")
    assert cat.classify(CRS.from_epsg(3857)) == dict(kind="projected", code="EPSG:3857")
    assert cat.classify(CRS.from_epsg(2056))["kind"] == "custom"
    for code in cat.PROJECTED:
        assert CRS.from_user_input(code) is not None
    with pytest.raises(ValueError):
        cat.utm_crs(4326, 61, "N")
    zones = cat.utm_zones()                                        # all 120 zones, -60 ... -1, +1 ... +60
    assert len(zones) == 120 and zones[0][:2] == (60, "S") and zones[59][:2] == (1, "S")
    assert zones[60][:2] == (1, "N") and zones[-1][:2] == (60, "N")
    z50s = next(l for z, h, l in zones if (z, h) == (50, "S"))
    assert z50s.startswith("\u221250  (50S)") and "114\u00b0E \u2013 120\u00b0E" in z50s
