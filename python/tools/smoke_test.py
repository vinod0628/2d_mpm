"""Smoke test for the 2D MPM Python environment.

Run from the project root:
    .venv\\Scripts\\python.exe python\\tools\\smoke_test.py

Exits 0 when every check passes, 1 otherwise. Data-dependent checks are skipped (not failed)
when demo_data is not present.
"""
import os
import sys
import time
import traceback
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))
os.environ.setdefault("USE_PYGEOS", "0")
DATA = os.path.join(ROOT, "demo_data")
results = []


def check(name):
    def deco(fn):
        t = time.time()
        try:
            msg = fn()
            status = "SKIP" if isinstance(msg, str) and msg.startswith("SKIP") else "PASS"
            results.append((status, name, (msg or "") + f" ({time.time() - t:.1f}s)"))
        except Exception as exc:  # noqa: BLE001
            results.append(("FAIL", name, f"{type(exc).__name__}: {exc}"))
            traceback.print_exc()
        return fn
    return deco


@check("python version is 3.10")
def _():
    assert sys.version_info[:2] == (3, 10), sys.version
    return sys.version.split()[0]


@check("GDAL (inside the rasterio wheel): PROJ lookups + attribute-table reader")
def _():
    import geopandas  # noqa: F401
    import sklearn  # noqa: F401
    import pyproj
    from rasterio.crs import CRS
    from utils.gdal_rat import gdal_version, read_rat

    assert CRS.from_epsg(28354).to_epsg() == 28354
    assert "MGA zone 54" in pyproj.CRS.from_epsg(28354).name
    ver = gdal_version()
    assert ver, "rasterio's GDAL library not found"
    msg = f"GDAL {ver} via rasterio, EPSG lookups ok"
    grid = os.path.join(DATA, "nevada_carlin_au", "raw", "KBGEOL")
    if os.path.isdir(grid):
        rat = read_rat(grid)
        assert rat is not None and len(rat) == 25 and "S_VALUE" in rat.columns
        msg += ", ArcInfo VAT read (25 classes)"
    return msg


@check("core scientific / geospatial imports")
def _():
    import fiona, numpy, pandas, pyproj, rasterio, scipy, shapely, skimage  # noqa: F401,E401
    return f"numpy {numpy.__version__}, rasterio {rasterio.__version__} (GDAL {rasterio.__gdal_version__}), shapely {shapely.__version__}"


@check("all eis_toolkit modules import")
def _():
    import importlib
    import pkgutil

    import eis_toolkit

    mods = list(pkgutil.walk_packages(eis_toolkit.__path__, "eis_toolkit."))
    bad = []
    for m in mods:
        try:
            importlib.import_module(m.name)
        except Exception as exc:  # noqa: BLE001
            bad.append(f"{m.name}: {exc}")
    bad = [b for b in bad if not b.startswith("eis_toolkit.utilities.gdal:")]   # osgeo helper, unused
    assert not bad, bad
    return f"{len(mods)} modules (utilities.gdal skipped: needs osgeo)"


@check("ML libraries fit (RF, SVM, Logit, XGB, LGBM, SMOTE, Keras)")
def _():
    import lightgbm
    import numpy as np
    import statsmodels.api as sm
    import tensorflow as tf
    import xgboost
    from imblearn.over_sampling import SMOTE
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC

    rng = np.random.default_rng(0)
    X = rng.random((300, 4))
    y = (X[:, 0] + 0.2 * rng.random(300) > 0.8).astype(int)
    Xb, yb = SMOTE(random_state=0).fit_resample(X, y)
    RandomForestClassifier(20, random_state=0).fit(Xb, yb)
    SVC(probability=True).fit(Xb, yb)
    sm.Logit(y, sm.add_constant(X)).fit(disp=0)
    xgboost.XGBClassifier(n_estimators=10).fit(X, y)
    lightgbm.LGBMClassifier(n_estimators=10, verbose=-1).fit(X, y)
    model = tf.keras.Sequential([tf.keras.layers.Dense(8, activation="relu", input_shape=(4,)),
                                 tf.keras.layers.Dense(1, activation="sigmoid")])
    model.compile("adam", "binary_crossentropy")
    model.fit(X, y, epochs=1, verbose=0)
    return f"tf {tf.__version__}, xgb {xgboost.__version__}, lgbm {lightgbm.__version__}"


@check("eis WofE on Carlin aeromag + 35 deposits")
def _():
    path = os.path.join(DATA, "nevada_carlin_au", "raw")
    if not os.path.isdir(path):
        return "SKIP: demo data not found"
    import geopandas as gpd
    import rasterio
    from eis_toolkit.prediction.weights_of_evidence import weights_of_evidence_calculate_weights

    dep = gpd.read_file(os.path.join(path, "Train2.shp"))
    with rasterio.open(os.path.join(path, "aeromag")) as src:
        dep = dep.set_crs(src.crs)
        df = weights_of_evidence_calculate_weights(src, dep, raster_nodata=src.nodata, weights_type="unique")[0]
    row = df[df["Class"] == -200.0].iloc[0]
    assert int(row["Deposit count"]) == 21 and abs(row["Contrast"] - 1.023) < 0.01, row.to_dict()
    return f"class -200 nT: 21 deposits, C = {row['Contrast']:.3f}"


@check("eis distance_to_anomaly (numba fallback, no osgeo) on Curnamona Th")
def _():
    path = os.path.join(DATA, "curnamona_ree", "processed", "rasters", "rad_th.tif")
    if not os.path.isfile(path):
        return "SKIP: demo data not found"
    import numpy as np
    import rasterio
    from eis_toolkit.raster_processing.distance_to_anomaly import distance_to_anomaly

    with rasterio.open(path) as src:
        prof = dict(src.profile)
        a = src.read(1).astype("float32")
    d, _ = distance_to_anomaly(prof, a, 100.0, "higher")
    assert np.nanmax(d) > 0
    return f"max distance {float(np.nanmax(d)):.0f} m"


width = max(len(n) for _, n, _ in results)
print("\n" + "=" * (width + 30))
for status, name, msg in results:
    print(f"[{status}] {name:<{width}}  {msg}")
n_fail = sum(s == "FAIL" for s, _, _ in results)
print("=" * (width + 30))
print("SMOKE TEST " + ("FAILED" if n_fail else "PASSED") + f": {len(results) - n_fail}/{len(results)} ok")
sys.exit(1 if n_fail else 0)
