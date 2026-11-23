<img src="python/gui/resources/mpm_studio.png" alt="2D MPM icon" width="96" align="right">

# 2D MPM

**2D MPM** is a Windows desktop application for **2D mineral prospectivity mapping (MPM)**. In one program you can:

- prepare evidence layers;
- run knowledge-driven, data-driven and hybrid prospectivity models;
- validate the resulting maps;
- delineate exploration targets.

It is written in Python 3.10 with PyQt5. It works **fully offline**, which matters for mine sites and field offices
without internet: no downloads, licence servers or online coordinate services. It ships with two ready-made demo
projects.

Version 1.0.0 · Windows 10/11, 64-bit · interface in English and 简体中文 (Simplified Chinese) · dark and light themes.

---

## Contents

1. [Functionality](#1-functionality)
2. [Quick start: release zip (no Python)](#2-quick-start-release-zip-no-python)
3. [Installation from source](#3-installation-from-source)
4. [Running and testing from source](#4-running-and-testing-from-source)
5. [Build and release](#5-build-and-release)
6. [Demo data and sources](#6-demo-data-and-sources)
7. [Documentation](#7-documentation)
8. [Folder layout](#8-folder-layout)

---

## 1. Functionality

### Projects and map

- **Folder-based projects:**
  - a project has one coordinate system, one cell size and a study area;
  - results of every run go to their own `results/<run>/` folder, with GeoTIFFs, CSV tables, run information and the training / test points as Shapefiles.
- **Map view:**
  - layer list (TOC) with drawing order, identify, measure, pan and zoom, and a coordinate bar;
  - layer properties: symbology, labels, a statistics tab with histogram, cumulative curve and PNG export, and legend export.
- **Class tables:** categorical rasters such as geological maps carry a class table (`<raster>.classes.csv`), so class names appear in the legend, in identify and in every results table.
- **Offline coordinate systems:**
  - EPSG lookups use the PROJ database bundled with rasterio / pyproj, and nothing is downloaded;
  - PROJ / GDAL environment variables set by other GIS software are isolated at start-up.

### Data preparation (Data menu)

| Tool | What it does |
|---|---|
| Convert to GeoTIFF / GeoPackage | Converts any GDAL raster (including ArcInfo grids, with their attribute table → class table) or OGR vector, and assigns a CRS if one is missing |
| Reclassify raster | Equal interval, quantile, natural breaks (Fisher–Jenks), standard deviation or manual breaks, with histogram preview |
| Align raster to grid | Reprojects / resamples onto the project grid (nearest, bilinear, cubic, average, mode) |
| Clip raster by mask | Keeps cells inside or outside mask polygons, optionally filtered by attribute |
| Polygon to raster | Geological map → categorical raster + class table (cell-centre or maximum-area rule, class attributes copied) |
| Distance to features | Euclidean distance to faults, contacts, intrusions, … |

### Prospectivity methods (25 tools)

| Family | Tools |
|---|---|
| **Knowledge-driven** (expert weights; deposits only validate) | Boolean logic overlay · Index overlay (AHP / fuzzy AHP weights) · Fuzzy logic overlay (AND, OR, product, sum, γ) · Logistic fuzzification + geometric average · TOPSIS / VIKOR (AHP, fuzzy AHP or entropy weights) |
| **Data-driven: statistical** | Weights of evidence · Logistic regression · Evidential belief functions (Dempster–Shafer) · Bayesian network |
| **Data-driven: machine learning** | Random forest · Support vector machine · XGBoost · Relevance vector machine · Neural network (MLP) · Extreme learning machine · Convolutional neural network (CNN) |
| **Hybrid** (expert mineral-system model + training data) | Fuzzy weights of evidence · Data-calibrated fuzzy memberships (WofE / LR) · Bayesian-shrinkage fuzzy WofE (BS-WofE) · Neuro-fuzzy inference (ANFIS-type) · Knowledge-prior residual boosting (MS-KPRB) · Knowledge-pooled random forest (KP-RF) · Knowledge-weighted SVM (KW-SVM) · Permissive-tract hurdle model (PT-Hurdle) · Knowledge-trend regression-kriging (KT-RK, grade / continuous targets) |

What the data-driven and hybrid tools share:

- **Training points:** random or spatial (block) train / test split and thinning of clustered deposits. Non-deposit samples can be random or knowledge-guided.
- **Missing evidence:** handled per tool; for example, Dempster–Shafer treats it as "no evidence".
- **Results window:** maps, tables and charts, such as weights and contrasts, feature importance, uncertainty maps, rule tables and independence tests.
- **GPU:** where the library supports it (for example XGBoost with an NVIDIA GPU, device *auto*). Without a GPU everything runs on the CPU.
- **Adding methods:** new methods are plug-ins (`logic/<category>/<method>/`) and appear in the menus automatically.

### Validation and targets (Validation menu)

- **ROC / AUC validation** works on any raster:
  - ROC curve with DeLong standard error and 95 % CI, success-rate curve, prediction–area (P-A) plot and summary table;
  - training and test deposits shown separately, with optional absence points.
- **Target delineation** turns any map into a binary target map. The threshold comes from the **P-A intersection** (Yousefi & Carranza 2015) or a **concentration–area (C-A) fractal** breakpoint (Cheng et al. 1994). You can drag it on the graph, with live target area and deposit counts.

---

## 2. Quick start: release zip (no Python)

1. Unzip `MPM_Studio_1.0.0_win64.zip` anywhere (about 1.5 GB unpacked). It needs no installation, no admin rights and no Python.
2. Double-click `2D MPM.exe`.
3. Choose **Project ▸ Open demo project** and pick Carlin Au or Curnamona REE. Then run, for example, *Data-driven ▸ Statistical ▸ Weights of Evidence…*, and check the result with *Validation ▸ ROC / AUC validation…*.

The log file is `%LOCALAPPDATA%\2D MPM\mpm_studio.log`; look there first if something goes wrong.
Unsigned exe files can trigger Windows SmartScreen: choose *More info ▸ Run anyway*.

---

## 3. Installation from source

The development environment is a virtual environment `.venv` **in the repository root** (`2d_mpm\.venv`), with
**Python 3.10** (64-bit). Python must stay at 3.10, because `eis_toolkit` 1.1.6, used for cross-checks and tests,
requires < 3.11. You need about 3.5 GB of free disk space. GDAL needs no separate installation: it comes inside the
rasterio / fiona wheels.

### 3a. Offline (no internet), the supported route

Everything needed is inside the project folder:

- `python\whl\`: about 220 wheels, ≈ 1.7 GB together with the runtime installers, including PyInstaller;
- `python\whl\runtime\`: the Python 3.10.11 installer and the VC++ runtime.

```bat
python\install_env.bat            :: install, or repair an existing .venv
python\install_env.bat /clean     :: delete .venv first, then a fresh install (≈ 15 min)
python\install_env.bat /test      :: only run the checks on the existing .venv
```

The script does four things:

1. Installs Python 3.10.11 for the current user if it is missing.
2. Installs the VC++ 2015–2022 runtime if it is missing.
3. Creates `.venv` and installs `requirements-offline.txt` with `pip --no-index` (pip never goes online).
4. Runs `pip check` and a smoke test, which should end with `SMOKE TEST PASSED: 7/7 ok`.

Copy the whole project folder to the target PC, without `.venv\` and `python\distribute\`. Details and
troubleshooting are in [docs/python/offline_install.md](../docs/python/offline_install.md).

### 3b. Online

1. Install **Python 3.10.11 (64-bit)** from <https://www.python.org/downloads/release/python-31011/>.
2. From the repository root (`2d_mpm\`), create the environment and install the exact, tested package set:

   ```bat
   py -3.10 -m venv .venv
   .venv\Scripts\python.exe -m pip install --upgrade pip wheel
   .venv\Scripts\python.exe -m pip install --no-deps -r python\requirements-lock.txt
   ```

   `requirements-lock.txt` is the full pinned set, including PyInstaller. `--no-deps` is needed because
   `eis_toolkit`'s metadata asks for the GDAL Python bindings (`osgeo`). Those have no Windows wheel on PyPI and are
   not used.

   Alternatively, install only the direct dependencies and let pip resolve the rest:

   ```bat
   .venv\Scripts\python.exe -m pip install -r python\requirements.txt
   .venv\Scripts\python.exe -m pip install --no-deps eis_toolkit==1.1.6
   ```

3. Check the environment: `python\install_env.bat /test`.

If a package fails to install from PyPI, install it from the shipped wheels instead:
`.venv\Scripts\python.exe -m pip install --no-index --find-links python\whl <package>`.

Package choices and versions are explained in [docs/python/python_packages.md](../docs/python/python_packages.md).

---

## 4. Running and testing from source

```bat
python\run_mpm.bat                          :: start 2D MPM
python\run_mpm.bat D:\mpm\my_project        :: open a project folder (or its *.mpm.json)
```

The processed demo layers ship in `demo_data\*\processed`. To rebuild them from the raw data (≈ 40 s), use
*Project ▸ Open demo project ▸ Rebuild demo data…*, or run:

```bat
cd python
..\.venv\Scripts\python.exe -m utils.prepare_demo_data
```

Tests use pytest; the GUI tests run off-screen:

```bat
cd python
..\.venv\Scripts\python.exe -m pytest tests\test_wofe.py -q        :: one module
..\.venv\Scripts\python.exe -m pytest tests -q                     :: everything (GUI tests ≈ 3 min)
```

What each test module covers is listed in [docs/python/how_to_test.md](../docs/python/how_to_test.md).

---

## 5. Build and release

The release is a **one-folder PyInstaller build**: `2D MPM.exe` plus `_internal\`. It runs on PCs without Python.

```bat
python\build_release.bat                    :: options: --skip-build  --no-zip  --keep-work
```

| Step | Result |
|---|---|
| 1. PyInstaller (`python\packaging\mpm_studio.spec`) | `python\distribute\dist\2D MPM\` |
| 2. Adds the processed demo data, docs and `README.txt` | in the same folder |
| 3. Headless self-test of the built exe (`--selftest`) | must print `SELFTEST OK` |
| 4. Zip | `python\distribute\release\MPM_Studio_<version>_win64.zip` |

A full build takes about 10 minutes plus about 1 minute for the self-test. It needs about 3 GB of free disk space
during the build. v1.0.0 is ≈ 1.5 GB unpacked and the zip ≈ 520 MB; TensorFlow for the CNN tool is most of it.

**Making a new release:**

1. Set `VERSION` in `python\version.py`. It is used by the window title, the About box, the exe properties and the zip name.
2. Run the tests.
3. If the app now imports a new package, add its wheel to `python\whl\` and refresh the lock files.
4. Run `python\build_release.bat`.
5. Unzip the release on a clean PC, open a demo project, run a method and check ROC / AUC.

New method plug-ins are bundled automatically.

You can also run the self-test of a built exe yourself with `"2D MPM.exe" --selftest report.txt`;
`"2D MPM.exe" --version` prints the version.

Details, including the spec file, the PyInstaller hooks, windowed-mode logging and troubleshooting, are in
[docs/python/how_to_build_python_app.md](../docs/python/how_to_build_python_app.md).

---

## 6. Demo data and sources

Two open case studies cover every evidence type and method family. The original files are kept untouched in
`demo_data\<case>\raw\`. The analysis-ready, co-registered layers in `processed\` (the demo projects) are made by
`python\utils\prepare_demo_data.py`.

| | Carlin-type Au, Nevada | REE (+U, Cu, Co), Curnamona Province (South Australia) |
|---|---|---|
| Folder | `demo_data\nevada_carlin_au\` | `demo_data\curnamona_ree\` |
| Source | USGS Open-File Report 01-221 v1.1, Raines (2001, rev. 2006), the ArcSDM course data, <https://pubs.usgs.gov/of/2001/of01-221/> | DEEP-SEAM v1.0 (EarthByte), Zenodo [10.5281/zenodo.17098677](https://doi.org/10.5281/zenodo.17098677); underlying data from the Geological Survey of South Australia (SARIG) |
| Licence | USGS public domain | GPL-3.0 (Zenodo package); the SARIG data are under SA government terms, so check them before redistributing |
| Deposits | 35 Carlin-type Au deposits | 7 REE occurrences |
| Evidence | 25-class geology with a favourable-host flag, 1 674 faults, 3 472 NURE stream-sediment samples × 46 elements, Sb surface, aeromagnetics, Bouguer gravity, radiometrics (pre-classified), expert map | faults, granites, Mesoproterozoic units, 615 geochemical samples × 27 elements, gravity, 12 magnetic derivatives, radiometrics, 15 ASTER mineral products, DEM |
| Demo project grid | Lambert conformal conic (Clarke 1866), 250 m cells, 1008 × 1112 | GDA94 / MGA zone 54 (EPSG:28354), 200 m cells, 227 × 252 |

Raw-data details, processing notes and the suitability of each dataset for each method family are in
[demo_data/README.md](../demo_data/README.md).

---

## 7. Documentation

| Document | Content |
|---|---|
| [docs/python/mpm_studio.md](../docs/python/mpm_studio.md) | User and developer guide: window, data tools, every method, validation, settings, translations, code structure, how to add a method |
| [docs/python/tools/](../docs/python/tools/) | One Word document per method: theory, equations, algorithm, dialog settings, ideal parameters on both demos with geological reasoning, references |
| [docs/python/offline_install.md](../docs/python/offline_install.md) | Offline installation with `install_env.bat`; updating the wheel set |
| [docs/python/python_packages.md](../docs/python/python_packages.md) | Packages and versions |
| [docs/python/how_to_build_python_app.md](../docs/python/how_to_build_python_app.md) | Release build in detail |
| [docs/python/how_to_test.md](../docs/python/how_to_test.md) | Test suite |
| [docs/methods_survey.md](../docs/methods_survey.md) | Survey of MPM methods (shared with the MATLAB tools) |

---

## 8. Folder layout

```
2d_mpm\
├─ .venv\                 Python 3.10 environment (created by install_env.bat, not distributed)
├─ demo_data\             Carlin Au and Curnamona REE: raw\ (original) and processed\ (demo projects)
├─ docs\python\           documentation of 2D MPM (Markdown + per-method Word documents)
├─ matlab\                separate MATLAB tools (independent of 2D MPM)
└─ python\                2D MPM
   ├─ main.py             entry point
   ├─ run_mpm.bat         start from source
   ├─ install_env.bat     offline environment installer
   ├─ build_release.bat   Windows release build
   ├─ version.py          application name and version
   ├─ core\               projects, layers, method registry
   ├─ gui\                main window, map canvas, dialogs, splash screen, resources
   ├─ logic\              prospectivity methods: data_driven\, knowledge_driven\, hybrid\ (one plug-in folder each)
   ├─ evaluation\         ROC / AUC, success rate, P-A, target delineation
   ├─ utils\              data preparation (convert, reclassify, align, clip, polygon to raster, distances)
   ├─ i18n\               translations (zh_CN.json)
   ├─ tests\              pytest suite (unit and offscreen GUI tests)
   ├─ tools\              developer scripts (icon / splash images, translations, smoke test, benchmarks)
   ├─ packaging\          PyInstaller spec, hooks, build script
   ├─ whl\                offline wheels + Python / VC++ installers
   └─ distribute\         build outputs (dist\, release\)
```
