# Python packages for 2D MPM

Environment: **Python 3.10.11** (python.org build, installed per-user) in the virtualenv `.venv/` (repository root).
Python 3.10 is required because `eis_toolkit` 1.1.6 declares `requires-python >=3.9,<3.11`.

- Direct dependencies: [`python/requirements.txt`](../../python/requirements.txt)
- Full resolved set: [`python/requirements-lock.txt`](../../python/requirements-lock.txt) (= `requirements-offline.txt`)
- Offline package set: [`python/whl/`](../../python/whl) (all wheels + Python/VC++ installers) installed by `python/install_env.bat`, see [offline_install.md](offline_install.md)
- Checks: `python/tools/pip_check.py` (pip check, tolerating the one known eis_toolkit metadata line) and
  `python/tools/smoke_test.py`

```powershell
.\.venv\Scripts\Activate.ps1                    # activate
python\install_env.bat /clean                   # rebuild .venv offline from python\whl\
```

## GDAL: provided by the rasterio / fiona wheels (no separate install)

GDAL is the C library behind all raster/vector I/O. **No separate GDAL installation, runtime zip or self-built
wheel is used.** The PyPI wheels of rasterio, fiona and pyogrio each contain their own GDAL DLL
(`site-packages/rasterio.libs/gdal-<hash>.dll`, GDAL 3.10.3):

| Need | Provided by |
|---|---|
| read/write/reproject rasters | rasterio (its bundled GDAL) |
| read/write vectors | fiona / geopandas (their bundled GDAL) |
| CRS / EPSG database | pyproj + rasterio (bundled PROJ data) |
| raster attribute tables (ArcInfo grid VAT → class labels in *Data ▸ Convert*) | `python/utils/gdal_rat.py`: calls `GDALGetDefaultRAT` & co. in **rasterio's** GDAL DLL via `ctypes` (verified identical to the GDAL Python bindings on all Carlin grids) |

The GDAL **Python bindings** (`osgeo`: `from osgeo import gdal`) are *not* installed: PyPI has no Windows wheel for
them. Nothing in MPM needs them:

- MPM Studio uses rasterio/fiona/pyproj + `utils/gdal_rat.py`.
- `eis_toolkit` uses `osgeo` only in `utilities/gdal.py` (a helper module, not used by any MPM method) and as an
  optional fast path in `raster_processing.distance_to_anomaly`, which automatically falls back to its numba
  implementation. All other 106 eis_toolkit modules (WofE, fuzzy overlay, PCA, logistic regression, ML wrappers,
  training-data tools, validation …) import and run without it.
- eis_toolkit's package metadata still lists `gdal` as a requirement → it is installed with `--no-deps`, and
  `pip check` reports `eis-toolkit 1.1.6 requires gdal, which is not installed` — expected and harmless
  (`python/tools/pip_check.py` ignores exactly this line).

> History: an earlier setup compiled the GDAL 3.12.1 bindings from source (VS Build Tools + GISInternals SDK) and
> needed a start-up hook against an MSVC-runtime crash. It was removed on 2026-11-18 in favour of the above.

## Packages and what they are used for

| Package | Version | Role in 2D MPM |
|---|---|---|
| **eis_toolkit** | 1.1.6 | EIS Horizon MPM toolkit (GTK/Gispo): WofE, fuzzy overlay, PCA, distance/density rasters, training-data tools, ML wrappers, validation |
| numpy | 1.26.4 | arrays (pinned <2 for TF 2.15 / numba 0.60) |
| scipy | 1.11.4 | stats, optimisation (RVM), interpolation, distance transforms — pinned by eis_toolkit |
| pandas | 2.3.3 | tables (geochemistry, WofE tables, results) |
| numba | 0.60.0 | JIT for heavy loops (eis_toolkit, e.g. distance_to_anomaly) |
| rasterio | 1.4.4 | raster read/write, reprojection, resampling, rasterize, masking — **bundles GDAL 3.10.3** |
| geopandas | 0.11.1 | vector data (geology polygons, faults, deposits, geochem samples) — pinned <0.12 by eis_toolkit |
| Shapely | 1.8.5.post1 | geometry ops (buffers, distances) — 1.x to match geopandas 0.11 |
| fiona | 1.9.6 | shapefile / GeoPackage I/O (bundles its own GDAL) |
| pyproj | 3.7.1 | CRS definitions and transforms (bundles PROJ data) |
| rtree | 1.4.1 | spatial index for vector joins |
| scikit-image | 0.25.2 | image filters, morphology, texture (GLCM) features |
| pysal 23.7 / esda 2.5.1 / libpysal 4.7.0 | | spatial autocorrelation (Moran's I, LISA); sub-packages resolved to versions compatible with geopandas 0.11 / Shapely 1.8 |
| mapclassify | 2.8.1 | class-break schemes (MPM Studio uses its own numpy Fisher–Jenks: `utils.reclass.fisher_jenks_breaks`) |
| PyKrige | 1.7.3 | ordinary/universal kriging of geochemical point data |
| scikit-learn | 1.3.2 | RF, SVM, LR, MLP, IsolationForest, OC-SVM, PCA, CV, ROC/AUC — pinned 1.3.x because imbalanced-learn 0.11 needs `sklearn.utils.parse_version` |
| imbalanced-learn | 0.11.0 | SMOTE / under-sampling of deposit vs non-deposit cells |
| statsmodels | 0.14.6 | logistic regression with standard errors, significance tests |
| xgboost | 3.2.0 | gradient-boosted trees |
| lightgbm | 4.5.0 | gradient-boosted trees (4.7.0 crashed with an access violation on `fit`) |
| tensorflow / keras | 2.15.1 / 2.15.0 | MLP / CNN / autoencoder models (eis_toolkit uses Keras 2) |
| shap | 0.49.1 | model explanation (SHAP values) |
| beartype / typer | 0.13.1 / 0.9.4 | eis_toolkit runtime type checks / CLI |
| matplotlib / seaborn | 3.10.9 / 0.13.2 | map canvas, ROC, success-rate and P-A plots |
| **PyQt5** | 5.15.11 | MPM Studio GUI |
| QtAwesome | 1.4.2 | toolbar / menu icons |
| jupyterlab / ipykernel | 4.6.6 / 7.3.0 | notebooks |
| openpyxl | 3.1.5 | Excel export of result tables |
| tqdm | 4.70.1 | progress bars |
| pytest | 9.2.0 | test suite (`python/tests`) |
| pyinstaller (+ hooks-contrib, altgraph, pefile, pywin32-ctypes) | 6.22.3 | Windows release build, see [how_to_build_python_app.md](how_to_build_python_app.md) |

**RVM**: no maintained scikit-learn-compatible RVM exists on PyPI, so RVM will be implemented in this project
(Tipping 2001 sparse Bayesian classifier on numpy/scipy).

## Compatibility notes

1. `pip check` → only `eis-toolkit 1.1.6 requires gdal, which is not installed` (expected, see above).
2. `pysal` 23.7 is a meta-package that pulls the newest sub-packages (geopandas ≥ 0.12, Shapely 2); they were pinned to
   older versions compatible with eis_toolkit's geopandas < 0.12.
3. `pygeos` is installed by a PySAL sub-package; `USE_PYGEOS=0` is set by the app/tests so geopandas uses Shapely.
4. Verified (smoke test): EPSG lookups, ArcInfo attribute tables via rasterio's GDAL, all eis_toolkit modules,
   eis WofE on the Carlin data (class −200 nT: 21 deposits, C = 1.023), eis `distance_to_anomaly` (numba fallback),
   RF/SVM/Logit/XGBoost/LightGBM/SMOTE/Keras fits.

## Clean-up of the former self-built GDAL (already done on this PC)

- removed: `osgeo` from `.venv`, the `00_mpm_gdal.pth` start-up hook, the GDAL wheel and runtime zip from `whl\`,
  `%USERPROFILE%\.local\gdal\`
- optional: Visual Studio 2022 Build Tools are no longer needed by this project —
  `winget uninstall Microsoft.VisualStudio.2022.BuildTools` frees ~1.7 GB (keep them if other projects compile C/C++).
