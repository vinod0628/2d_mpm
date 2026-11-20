"""Build analysis-ready layers from the raw demo datasets.

For each dataset in ``demo_data/<dataset>/raw`` this writes ``processed/``:

* ``converted/``  native-grid GeoTIFFs of formats the app cannot use directly (e.g. ArcInfo grids)
* ``rasters/``    every evidence layer on ONE common metric grid, masked to the study area
* ``vectors/``    GeoPackages with a CRS (deposits, faults, geochemical samples, study area)
* ``layers.csv``  manifest (name, group, kind, description)
* ``<dataset>.mpm.json``  project file that opens in the MPM application

Run from the ``python`` folder::

    python -m utils.prepare_demo_data            # both datasets
    python -m utils.prepare_demo_data carlin     # or: curnamona
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS

if __package__ in (None, ""):  # allow `python utils/prepare_demo_data.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils.convert import raster_to_geotiff, read_attribute_table  # type: ignore
    from utils.raster_tools import (Grid, align_to_grid, apply_mask, distance_to_features,  # type: ignore
                                    idw_interpolate, read_raster, rasterize_vector, write_raster)
    from utils.reclass import compute_breaks, reclass_by_breaks, reclass_by_table  # type: ignore
else:
    from .convert import raster_to_geotiff, read_attribute_table
    from .raster_tools import (Grid, align_to_grid, apply_mask, distance_to_features, idw_interpolate,
                               read_raster, rasterize_vector, write_raster)
    from .reclass import compute_breaks, reclass_by_breaks, reclass_by_table

DEMO = Path(__file__).resolve().parents[2] / "demo_data"


class _Builder:
    """Collects outputs + manifest rows for one dataset."""

    def __init__(self, ds_dir: Path):
        self.raw = ds_dir / "raw"
        self.out = ds_dir / "processed"
        if self.out.exists():
            shutil.rmtree(self.out)
        self.rdir, self.vdir, self.cdir = self.out / "rasters", self.out / "vectors", self.out / "converted"
        for d in (self.rdir, self.vdir, self.cdir):
            d.mkdir(parents=True, exist_ok=True)
        self.rows = []

    def raster(self, name, arr, grid, group, kind, desc, nodata=None, dtype=None, table=None):
        path = self.rdir / f"{name}.tif"
        write_raster(str(path), arr, grid, nodata=nodata, dtype=dtype, class_table=table)
        self.rows.append(dict(name=name, file=f"rasters/{name}.tif", group=group, kind=kind, description=desc))
        return path

    def vector(self, name, gdf, group, desc):
        path = self.vdir / f"{name}.gpkg"
        gdf.to_file(path, driver="GPKG")
        self.rows.append(dict(name=name, file=f"vectors/{name}.gpkg", group=group, kind="vector", description=desc))
        return path

    def finish(self, project_name: str, crs: CRS, visible_layers, styles=None):
        man = pd.DataFrame(self.rows)
        man.to_csv(self.out / "layers.csv", index=False)
        styles = styles or {}
        order = [r for r in self.rows if r["kind"] == "vector"] + [r for r in self.rows if r["kind"] != "vector"]
        layers = []
        for r in order:
            layers.append(dict(name=r["name"], path=r["file"], type="vector" if r["kind"] == "vector" else "raster",
                               visible=r["name"] in visible_layers, categorical=r["kind"] == "categorical",
                               style=styles.get(r["name"], {})))
        proj = dict(format="mpm-project", version=1, name=project_name, crs=crs.to_wkt(), layers=layers)
        p = self.out / f"{project_name}.mpm.json"
        p.write_text(json.dumps(proj, indent=2))
        return p


# ================================================================================================ Carlin
def prepare_carlin(root: Path = DEMO) -> Path:
    t0 = time.time()
    b = _Builder(root / "nevada_carlin_au")
    raw = b.raw
    with rasterio.open(raw / "aeromag") as ds:      # the only grids with a projection file
        crs = ds.crs                                 # LCC, Clarke 1866, std par. 33/45, CM -117

    # ---- 1. ArcInfo grids -> GeoTIFF on their native grid, CRS assigned
    native = {"KBGEOL": ("geology", "S_VALUE"), "EXPERT2": ("expert_map", "CARLIN"),
              "RCLSSB2": ("sb_class_1km", "value"), "SBFACE1": ("sb_surface_1km", None),
              "aeromag": ("aeromag_2400m", "value"), "bouguer": ("bouguer_4800m", "value"),
              "potassium": ("potassium_2400m", "value"), "thorium": ("thorium_2400m", "value"),
              "uranium": ("uranium_2400m", "value"), "studygrd3": ("study_grid_1180m", "value")}
    conv = {}
    for src, (name, label) in native.items():
        conv[name] = raster_to_geotiff(str(raw / src), str(b.cdir / f"{name}.tif"), crs=crs, label_field=label)

    # ---- 2. common grid = geology grid (250 m); mask = mapped geology inside study polygon
    geol, grid, _ = read_raster(conv["geology"])
    study = gpd.read_file(raw / "studyarea.shp").set_crs(crs)
    mask = (rasterize_vector(study, grid, dtype="uint8") == 1) & ~np.ma.getmaskarray(geol)
    b.raster("study_mask", np.ma.array(np.ones(grid.shape, "uint8"), mask=~mask), grid, "mask", "categorical",
             "Study area (mapped geology inside study polygon)", nodata=0,
             table=pd.DataFrame(dict(value=[1], label=["study area"])))

    # geology + favourable host (ArcSDM VALUE2: 2 = Inside, 1 = Outside, -99 = Missing)
    rat = read_attribute_table(str(raw / "KBGEOL"))
    gtable = pd.DataFrame(dict(value=rat["value"], label=rat["S_VALUE"], host=rat["S_VALUE2"],
                               fuzzy_membership=rat["FMEMSHP1"]))
    b.raster("geology", apply_mask(geol, mask), grid, "geology", "categorical",
             "Lithology (25 units, Kbgeol), labels = unit codes", nodata=255, table=gtable)
    host_map = {int(v): (2 if v2 == 2 else 1) for v, v2 in zip(rat["value"], rat["VALUE2"]) if v2 != -99}
    h = reclass_by_table(geol, host_map)
    host = np.ma.array(np.ma.filled(h, 0).astype("uint8"), mask=np.ma.getmaskarray(h))
    b.raster("geology_host", apply_mask(host, mask), grid, "geology", "categorical",
             "Favourable host rocks for Carlin-type Au (Inside/Outside; 'Missing' units = nodata)",
             nodata=0, table=pd.DataFrame(dict(value=[1, 2], label=["other", "favourable host"])))

    exp, egrid, _ = read_raster(conv["expert_map"])
    b.raster("expert_map", apply_mask(exp, mask), grid, "reference", "categorical",
             "Expert (knowledge-driven) permissive map from the course: 0 non-permissive, 1 permissive, 2 favourable",
             nodata=255, table=pd.DataFrame(dict(value=[0, 1, 2], label=["non-permissive", "permissive", "favourable"])))

    # ---- 3. geophysics / geochem grids -> common grid
    for name, out, group, kind, res, desc in [
        ("aeromag_2400m", "aeromag", "geophysics", "categorical", "nearest", "Aeromagnetic anomaly, classified (nT)"),
        ("bouguer_4800m", "bouguer", "geophysics", "categorical", "nearest", "Bouguer gravity, classified (mGal)"),
        ("potassium_2400m", "potassium", "geophysics", "categorical", "nearest", "Radiometric K, classified"),
        ("thorium_2400m", "thorium", "geophysics", "categorical", "nearest", "Radiometric Th, classified"),
        ("uranium_2400m", "uranium", "geophysics", "categorical", "nearest", "Radiometric U, classified"),
        ("sb_class_1km", "sb_class", "geochemistry", "categorical", "nearest",
         "Stream-sediment Sb surface, 15 classes (course data)"),
        ("sb_surface_1km", "sb_surface", "geochemistry", "continuous", "bilinear",
         "Stream-sediment Sb interpolated surface (course data)"),
    ]:
        a, g, prof = read_raster(conv[name])
        al = apply_mask(align_to_grid(a, g, grid, res), mask)
        table = None
        if kind == "categorical":
            vals = np.unique(np.ma.compressed(al))
            table = pd.DataFrame(dict(value=vals, label=[f"{v:g}" for v in vals]))
        b.raster(out, al, grid, group, kind, desc,
                 nodata=prof.get("nodata"), dtype=a.dtype if kind == "categorical" else "float32", table=table)

    # ---- 4. faults -> distance (m) + classified distance
    faults = gpd.read_file(raw / "gbfaults3.shp").set_crs(crs)
    dist = apply_mask(distance_to_features(faults, grid), mask)
    b.raster("dist_faults", dist, grid, "structure", "continuous", "Distance to nearest fault (m)", nodata=-9999.0)
    fb = [500, 1000, 1500, 2000, 3000, 4000, 6000, 8000, 12000]
    cls, tab = reclass_by_breaks(dist, fb)
    b.raster("dist_faults_class", cls, grid, "structure", "categorical",
             "Distance to faults, 10 classes (use cumulative ASCENDING in WofE)", nodata=0, table=tab)

    # ---- 5. NURE stream-sediment geochemistry -> IDW of log10 As, Sb
    naa = gpd.read_file(raw / "Naa.shp").set_crs(crs)
    for el in ("AS", "SB"):
        v = naa[f"NAA_{el}"].astype(float).where(lambda s: s > 0)
        est = idw_interpolate(naa.geometry.x, naa.geometry.y, np.log10(v), grid, power=2, k=12,
                              max_distance=15000, mask=mask)
        nm = el.lower()
        b.raster(f"{nm}_log_idw", est, grid, "geochemistry", "continuous",
                 f"log10 {el.title()} (ppm), NURE stream sediments, IDW (p=2, k=12, r<=15 km)", nodata=-9999.0)
        q = compute_breaks(np.ma.compressed(est), "quantile", 8)
        cls, tab = reclass_by_breaks(est, q)
        b.raster(f"{nm}_q8", cls, grid, "geochemistry", "categorical",
                 f"log10 {el.title()} IDW, 8 quantile classes (use cumulative DESCENDING in WofE)", nodata=0, table=tab)

    # ---- 6. vectors
    dep = gpd.read_file(raw / "Train2.shp").set_crs(crs)
    keep = ["SITE", "DISTRICT", "DEP_TYPE", "MODEL_NAME", "STATUS", "HR_TYPE", "RANDOM"]
    b.vector("deposits", dep[keep + ["geometry"]], "deposits", "35 Carlin-type Au deposits (training points)")
    b.vector("faults", faults[["LENGTH", "AZIMUTH", "geometry"]], "structure", "Faults (1 674)")
    b.vector("study_area", study[["geometry"]], "mask", "Study area polygon")
    el_cols = [c for c in naa.columns if c.startswith("NAA_") or c == "DNC_U"]
    b.vector("nure_geochem", naa[["SAMPLEID"] + el_cols + ["geometry"]], "geochemistry",
             "NURE stream-sediment samples (3 472, 46 elements; 0 = below detection/not analysed)")

    styles = {"deposits": dict(color="#ffd54f", edgecolor="#000000", markersize=36, marker="*"),
              "faults": dict(color="#ff7043", linewidth=0.6), "study_area": dict(facecolor="none", edgecolor="#9e9e9e"),
              "nure_geochem": dict(color="#80cbc4", markersize=4), "geology": dict(cmap="tab20"),
              "dist_faults": dict(cmap="magma_r")}
    p = b.finish("carlin_au", crs, visible_layers={"deposits", "faults", "geology"}, styles=styles)
    print(f"Carlin: {len(b.rows)} layers -> {b.out}  ({time.time() - t0:.1f}s)")
    return p


# ================================================================================================ Curnamona
CURN_CRS = "EPSG:28354"   # GDA94 / MGA zone 54
CURN_RES = 200.0


def _curn_name(group: str, stem: str) -> str:
    s = stem.lower().replace("_onshore", "")
    if group == "DEM":
        return "dem"
    if group == "Magnetics":                    # SA_TMI -> mag_tmi, SA_TMI_VRTP_1VD -> mag_tmi_vrtp_1vd
        return "mag_" + s[len("sa_"):]
    if group == "Gravity":                      # SA_GRAV -> grav, SA_GRAV_1VD -> grav_1vd
        rest = s[len("sa_grav"):].lstrip("_")
        return "grav" + ("_" + rest if rest else "")
    if group == "Radiometrics":                 # SA_RAD_K -> rad_k
        return "rad_" + s[len("sa_rad_"):]
    return "aster_" + s


def prepare_curnamona(root: Path = DEMO) -> Path:
    t0 = time.time()
    b = _Builder(root / "curnamona_ree")
    raw = b.raw / "Input_Data_Layers"
    crs = CRS.from_user_input(CURN_CRS)
    frame = gpd.read_file(raw / "GIS" / "Clip_frame.shp").to_crs(crs)
    grid = Grid.from_bounds(*frame.total_bounds, res=CURN_RES, crs=crs)
    mask = rasterize_vector(frame, grid, dtype="uint8") == 1
    b.raster("study_mask", np.ma.array(np.ones(grid.shape, "uint8"), mask=~mask), grid, "mask", "categorical",
             "Study area (Clip_frame)", nodata=0, table=pd.DataFrame(dict(value=[1], label=["study area"])))

    groups = {"Gravity": "geophysics", "Magnetics": "geophysics", "Radiometrics": "geophysics",
              "Remote Sensing": "remote sensing", "DEM": "topography"}
    for folder, group in groups.items():
        for tif in sorted((raw / folder).glob("*.tif")):
            a, g, _ = read_raster(str(tif), as_float=True)
            al = apply_mask(align_to_grid(a, g, grid, "bilinear"), mask)
            name = _curn_name(folder, tif.stem)
            b.raster(name, al, grid, group, "continuous", f"{tif.stem} (reprojected to MGA54, {CURN_RES:g} m, bilinear)",
                     nodata=-9999.0, dtype="float32")

    gf = raw / "Geology factors" / "original"
    faults = gpd.read_file(gf / "Archean_Early_Mesoproterozoic_Faults.shp").to_crs(crs)
    granite = gpd.read_file(gf / "Felsic_granite.shp").to_crs(crs)
    meso = gpd.read_file(gf / "Mesoproterozoic.shp").to_crs(crs)
    d = apply_mask(distance_to_features(faults, grid), mask)
    b.raster("dist_faults", d, grid, "structure", "continuous", "Distance to Archean-Early Mesoproterozoic faults (m)",
             nodata=-9999.0)
    cls, tab = reclass_by_breaks(d, [250, 500, 1000, 1500, 2000, 3000, 5000])
    b.raster("dist_faults_class", cls, grid, "structure", "categorical",
             "Distance to faults, 8 classes (use cumulative ASCENDING in WofE)", nodata=0, table=tab)
    b.raster("dist_granite", apply_mask(distance_to_features(granite, grid), mask), grid, "geology", "continuous",
             "Distance to felsic granite (m, 0 inside)", nodata=-9999.0)
    b.raster("dist_mesoproterozoic", apply_mask(distance_to_features(meso, grid), mask), grid, "geology",
             "continuous", "Distance to Mesoproterozoic units (m, 0 inside)", nodata=-9999.0)
    inside = rasterize_vector(granite, grid, dtype="uint8") + 2 * rasterize_vector(meso, grid, dtype="uint8")
    lith = np.ma.array(np.where(inside >= 2, 3, np.where(inside == 1, 2, 1)).astype("uint8"), mask=~mask)
    b.raster("geology_units", lith, grid, "geology", "categorical", "Simplified units from the vector geology",
             nodata=0, table=pd.DataFrame(dict(value=[1, 2, 3], label=["other / cover", "felsic granite",
                                                                       "Mesoproterozoic"])))

    geo = pd.read_csv(raw / "Geochemical" / "ILRRPCA.csv")
    pts = gpd.GeoDataFrame(geo, geometry=gpd.points_from_xy(geo.lon_gda94, geo.lat_gda94), crs="EPSG:4283").to_crs(crs)
    for el in ("La", "Y", "Th", "U3O8", "Nb"):
        v = pts[el].astype(float).where(lambda s: s > 0)
        est = idw_interpolate(pts.geometry.x, pts.geometry.y, np.log10(v), grid, power=2, k=12,
                              max_distance=6000, mask=mask)
        b.raster(f"geochem_{el.lower()}_log_idw", est, grid, "geochemistry", "continuous",
                 f"log10 {el} (ppm), surface samples, IDW (p=2, k=12, r<=6 km)", nodata=-9999.0)
    q = compute_breaks(np.ma.compressed(read_raster(str(b.rdir / "geochem_la_log_idw.tif"))[0]), "quantile", 8)
    la = read_raster(str(b.rdir / "geochem_la_log_idw.tif"))[0]
    cls, tab = reclass_by_breaks(la, q)
    b.raster("geochem_la_q8", cls, grid, "geochemistry", "categorical",
             "log10 La IDW, 8 quantile classes (use cumulative DESCENDING in WofE)", nodata=0, table=tab)

    dep = gpd.read_file(raw / "GIS" / "ree.shp").to_crs(crs)
    b.vector("deposits", dep[["DEP_NAME", "COMMODS", "STATUS_VAL", "SIZE_VAL", "geometry"]], "deposits",
             "7 REE occurrences")
    b.vector("faults", faults[["DESCRIPTIO", "AGE", "geometry"]], "structure", "Archean-Early Mesoproterozoic faults")
    b.vector("felsic_granite", granite[["STRATNAME", "geometry"]] if "STRATNAME" in granite else granite[["geometry"]],
             "geology", "Felsic granite polygons")
    b.vector("mesoproterozoic", meso[["STRATNAME", "MAPUNIT", "geometry"]], "geology", "Mesoproterozoic units")
    b.vector("geochem_samples", pts, "geochemistry", "Surface geochemical samples (615 x 27 elements)")
    b.vector("study_area", frame[["geometry"]], "mask", "Clip frame")

    styles = {"deposits": dict(color="#ffd54f", edgecolor="#000000", markersize=60, marker="*"),
              "faults": dict(color="#ff7043", linewidth=0.8),
              "felsic_granite": dict(facecolor="none", edgecolor="#f48fb1", linewidth=0.8),
              "mesoproterozoic": dict(facecolor="none", edgecolor="#90caf9", linewidth=0.6),
              "study_area": dict(facecolor="none", edgecolor="#9e9e9e"),
              "geochem_samples": dict(color="#80cbc4", markersize=6), "mag_tmi_vrtp": dict(cmap="turbo"),
              "rad_th": dict(cmap="viridis")}
    p = b.finish("curnamona_ree", crs, visible_layers={"deposits", "faults", "mag_tmi_vrtp"}, styles=styles)
    print(f"Curnamona: {len(b.rows)} layers -> {b.out}  ({time.time() - t0:.1f}s)")
    return p


def _main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", nargs="?", choices=["carlin", "curnamona", "all"], default="all")
    ap.add_argument("--root", default=str(DEMO))
    a = ap.parse_args(argv)
    root = Path(a.root)
    if a.dataset in ("carlin", "all"):
        prepare_carlin(root)
    if a.dataset in ("curnamona", "all"):
        prepare_curnamona(root)


if __name__ == "__main__":
    _main()
