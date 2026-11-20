"""Offline CRS set-up and unrecognised coordinate systems (loaded as the current project CRS)."""
import logging
import os
import sys

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import CRS
from rasterio.transform import from_origin
from shapely.geometry import Point

from core.layers import RasterLayer, VectorLayer
from utils import geo_env

LOCAL = 'LOCAL_CS["GDA94 / MGA zone 54",UNIT["metre",1],AXIS["Easting",EAST],AXIS["Northing",NORTH]]'
PROJECT = CRS.from_epsg(28354)


def test_isolate_removes_foreign_proj_variables_only():
    inside = os.path.join(sys.prefix, "Lib", "site-packages", "rasterio", "proj_data")
    env = {"PROJ_LIB": r"C:\Program Files\PostgreSQL\16\share\contrib\postgis-3.4\proj", "PROJ_DATA": inside,
           "GDAL_DATA": r"D:\OSGeo4W\share\gdal", "PATH": "x"}
    notes = geo_env.isolate(env)
    assert "PROJ_LIB" not in env and "GDAL_DATA" not in env and env["PROJ_DATA"] == inside
    assert env["PROJ_NETWORK"] == "OFF" and len(notes) == 2 and env["PATH"] == "x"
    keep = {"PROJ_LIB": r"C:\other\proj", "MPM_KEEP_PROJ_ENV": "1"}
    assert geo_env.isolate(keep) == [] and keep["PROJ_LIB"] == r"C:\other\proj"
    assert geo_env.crs_database_problem() is None                                          # this environment


def test_unrecognized():
    assert geo_env.unrecognized(None) and geo_env.unrecognized(LOCAL) and geo_env.unrecognized("not a crs")
    assert geo_env.unrecognized('GEOGCS["Undefined geographic SRS",DATUM["unknown",SPHEROID["unknown",6378137,'
                                '298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]')
    assert not geo_env.unrecognized(PROJECT) and not geo_env.unrecognized(CRS.from_epsg(4326))
    assert not geo_env.unrecognized(CRS.from_epsg(4283)) and not geo_env.unrecognized(CRS.from_epsg(26911))


def _raster(path, crs):
    with rasterio.open(path, "w", driver="GTiff", width=4, height=3, count=1, dtype="float32", crs=crs,
                       transform=from_origin(400000, 6600000, 200, 200)) as ds:
        ds.write(np.arange(12, dtype="float32").reshape(1, 3, 4))


def test_unrecognised_raster_crs_is_taken_as_project_crs(tmp_path, caplog):
    caplog.set_level(logging.WARNING, logger="mpm")
    for crs in (LOCAL, None):
        p = str(tmp_path / f"r_{crs is None}.tif")
        _raster(p, crs)
        lyr = RasterLayer.from_file(p, target_crs=PROJECT.to_wkt())
        assert CRS.from_wkt(lyr.grid.crs.to_wkt()) == PROJECT and (lyr.grid.width, lyr.grid.height) == (4, 3)
        assert lyr.grid.transform.c == 400000 and lyr.data[2, 3] == 11                 # not moved, not resampled
        add = RasterLayer.from_file(p, assume_crs=PROJECT.to_wkt())                     # Add Raster Data path
        assert CRS.from_wkt(add.grid.crs.to_wkt()) == PROJECT
    assert sum("treated as the current coordinate system" in r.getMessage() for r in caplog.records) == 4
    known = str(tmp_path / "zone55.tif")
    _raster(known, "EPSG:28355")
    moved = RasterLayer.from_file(known, target_crs=PROJECT.to_wkt())                  # known CRS: reprojected
    assert moved.grid.transform.c != 400000


def test_unrecognised_vector_crs_is_taken_as_project_crs(tmp_path, caplog):
    caplog.set_level(logging.WARNING, logger="mpm")
    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[Point(400100, 6599900)])
    p = str(tmp_path / "pts.gpkg")
    gdf.to_file(p, driver="GPKG")                                                      # no CRS
    lyr = VectorLayer.from_file(p, target_crs=PROJECT.to_wkt())
    assert CRS.from_user_input(lyr.gdf.crs) == PROJECT and lyr.gdf.geometry.iloc[0].x == 400100
    assert any("treated as the current coordinate system" in r.getMessage() for r in caplog.records)
    p2 = str(tmp_path / "pts55.gpkg")
    gdf.set_crs(28355).to_file(p2, driver="GPKG")
    assert VectorLayer.from_file(p2, target_crs=PROJECT.to_wkt()).gdf.geometry.iloc[0].x != 400100   # reprojected
