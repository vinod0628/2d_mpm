# How to build the MPM Studio application (Windows release)

This explains how the Python source in `python/` is turned into a stand-alone Windows program
(`MPM Studio.exe`) that runs on PCs **without Python**, and how to make a new release.

## 1. Quick start

```bat
pythonuild_release.bat
```

Result:

| Output | Content |
|---|---|
| `python\distribute\dist\MPM Studio\` | the application folder: `MPM Studio.exe`, `_internal\` (Python runtime + libraries), `demo_data\`, `docs\`, `README.txt` |
| `python\distribute\release\MPM_Studio_<version>_win64.zip` | the same folder zipped, for distribution |

Users unzip it anywhere and double-click `MPM Studio.exe`. No installation, no admin rights, no Python.

Options (passed to `python\packaging\build_release.py`):

| Option | Effect |
|---|---|
| `--skip-build` | reuse the existing `python\distribute\dist\MPM Studio\` (only re-copy demo data/docs, self-test, zip) |
| `--no-zip` | do not create the zip |
| `--keep-work` | keep PyInstaller's `python\distribute\build\` work folder (for debugging) |

## 2. Prerequisites

- The development environment: `.venv` in the repository root (Python 3.10). Create it from `python\whl\` with `python\install_env.bat` (see [offline_install.md](offline_install.md)).
- **PyInstaller** — its wheels are in `python\whl\`, and `python\build_release.bat` installs it automatically from there if missing:
  `pyinstaller 6.22.3`, `pyinstaller-hooks-contrib 2026.10`, `altgraph`, `pefile`, `pywin32-ctypes`.
- About **3 GB free disk** during the build (work folder + application folder + zip). The work folder is deleted afterwards.
- The processed demo data (`demo_data\*\processed`). If missing: `cd python && ..\.venv\Scripts\python.exe -m utils.prepare_demo_data`.

## 3. Why PyInstaller, one-folder mode

| Tool | Verdict |
|---|---|
| **PyInstaller** (chosen) | Mature, best support for PyQt5 + the scientific stack via `pyinstaller-hooks-contrib` (rasterio, fiona, pyproj, shapely, geopandas, matplotlib, scikit-learn, qtawesome hooks exist). Pure pip install, works offline from wheels. |
| cx_Freeze | Similar result, fewer ready-made hooks for the geospatial libraries. |
| Nuitka | Compiles to C — needs a C compiler and long build times; little gain for a GUI that spends its time in numpy/GDAL. |
| conda-pack / embedded Python | Ships the whole environment (several GB incl. TensorFlow); no single exe. |

**One-folder** (`exe + _internal\`) instead of **one-file**: a one-file exe would unpack ~600 MB to a temp folder at
every start (slow start, antivirus false alarms). The one-folder build starts immediately.

v1.0.0 sizes: application folder **562 MB** (exe 15 MB + `_internal`), zip **226 MB**; build time ≈ 5 min, self-test ≈ 1 min.

## 4. How the build works

```
python\build_release.bat
└─ python\packaging\build_release.py            (all outputs in python\distribute\)
   ├─ 1. PyInstaller  python\packaging\mpm_studio.spec -> python\distribute\dist\MPM Studio\   (work dir python\distribute\build\)
   ├─ 2. payload: demo_data\*\processed, docs, README.txt
   │      (project files are cleaned: layers stored outside the demo folder are removed)
   ├─ 3. self-test:  "MPM Studio.exe" --selftest report.txt   (headless, must print SELFTEST OK)
   └─ 4. zip         -> python\distribute\release\MPM_Studio_<version>_win64.zip
```

### `python\packaging\mpm_studio.spec` (the PyInstaller recipe)

| Part | Purpose |
|---|---|
| `Analysis([python\main.py], pathex=[python])` | entry point; PyInstaller follows imports from here |
| `datas = gui\resources` | icon (`mpm_studio.ico/png`), splash image (`splash.png`) and checkbox SVGs used by the stylesheet |
| `HIDDEN` | modules imported indirectly: **every module under `logic/`** (method plug-ins are discovered at run time; `collect_submodules("logic")`, template excluded — new methods are bundled automatically) and the Qt5Agg backend |
| `collect_all(rasterio, fiona, pyproj, shapely, geopandas, qtawesome)` | these packages import submodules from compiled Cython code that PyInstaller cannot see (e.g. `rasterio._base` → `rasterio.serde`) and ship data files (GDAL/PROJ databases, icon fonts) — so all their submodules, data and DLLs are collected (their test modules are filtered out) |
| `EXCLUDES` | packages in the dev venv that the app does not use: LightGBM, SHAP, numba, statsmodels, PySAL family, mapclassify, eis_toolkit, GDAL Python bindings (`osgeo`), Jupyter/IPython, pytest, tkinter, unused Qt modules. Without this list the build would be several GB. TensorFlow / Keras 2.15 (CNN tool, ≈ 1.2 GB) and XGBoost (≈ 140 MB) **are bundled**; TensorFlow is also named in `HIDDEN` because the CNN imports it lazily. |
| `runtime_hooks = rthook_mpm.py` | sets `USE_PYGEOS=0` before the app starts |
| `hookspath = packaging\hooks` | `hook-pyproj.py` replaces the pyinstaller-hooks-contrib hook. That hook looks for `pyproj/proj_dir` among the collected paths, but on Windows they read `pyproj\proj_dir`, so with the pip wheel the check always fails and prints the false warning "Datas for pyproj not found at: …\.venv\Library\share\proj" (a conda folder). The fixed hook collects the same data (proj.db included) and DLLs and warns only if proj.db is really missing |
| `EXE(... icon=mpm_studio.ico, version=VSVersionInfo, console=False)` | Windows icon, file properties (version, description) and no console window |
| `COLLECT(...)` | one-folder output |

### Windowed mode (no console)

The exe is built with `console=False`, so it has no stdout/stderr. `main._setup_streams_and_crash_log()` routes
them to `%LOCALAPPDATA%\MPM Studio\mpm_studio.log`, enables `faulthandler` there and installs an exception hook
that logs the traceback and shows a message box. **Look in this log first when the release misbehaves.**

### Paths inside the frozen app

- `main.resource_path()` uses `sys._MEIPASS` (the `_internal` folder) for bundled resources.
- `gui.main_window.ROOT` is the folder of `MPM Studio.exe` when frozen (`sys.frozen`), so `demo_data\` and `docs\`
  are found beside the exe; from source it is the repository root.
- *Rebuild demo data* is disabled in the release when the raw demo files are absent (they are not shipped).

### Application icon

- Generated by `python\tools\make_icon.py` → `python\gui\resources\mpm_studio.ico` (16–256 px) and `mpm_studio.png`:
  a mineral crystal on a banded prospectivity hotspot (inferno palette, as the posterior maps) with a target ring
  and map grid. Re-run the script after editing it.
- Used for the exe file (spec `icon=`), the window/taskbar (`QApplication.setWindowIcon`) and the About box.
- `main._set_windows_app_id()` sets a Windows AppUserModelID, so the taskbar shows the MPM icon — also when the app
  is started from source with `python.exe`.

## 5. Making a new release

1. Update `python\version.py` → `VERSION = "1.1.0"` (used by the window title, About box, exe properties and zip name).
2. Run the tests: `cd python && ..\.venv\Scripts\python.exe -m pytest tests -q`.
3. If you added a package that the app imports: add its wheel to `python\whl\` (and the lock files); check whether it is in
   `EXCLUDES` of the spec.
4. New methods under `python\logic\<category>\<method>\` are bundled automatically (no spec change).
5. `python\build_release.bat` → check the self-test output and `python\distribute\release\MPM_Studio_<version>_win64.zip`.
6. Smoke-test on a clean PC: unzip, start, *Project ▸ Open demo project*, run WofE, open ROC/AUC.

## 6. Self-test

`"MPM Studio.exe" --selftest report.txt` (also works from source: `python python\main.py --selftest report.txt`)
opens the main window, loads the Carlin demo, runs WofE (3 evidence layers, 20 % test deposits), adds the results
to the map, opens the ROC/AUC dialog, runs a natural-breaks reclassification, checks EPSG lookups in the bundled
PROJ database (rasterio and pyproj) and loads the Curnamona demo. The report ends with `SELFTEST OK` and the
exit code is 0. `build_release.py` runs it with `QT_QPA_PLATFORM=offscreen`, so no window appears.

`"MPM Studio.exe" --version` prints the version (no window).

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` in the self-test | a module is imported dynamically → add it to `HIDDEN` in the spec; or it was excluded by mistake → remove it from `EXCLUDES` |
| A data file is missing (icons, PROJ database, fonts) | add it to `datas` in the spec (`collect_data_files("package")`) |
| The exe starts and closes without a message | build with `console=True` in the spec temporarily to see the traceback, or run `--selftest report.txt` |
| Build is several GB | a heavy package slipped in → check `python\distribute\build\mpm_studio\warn-mpm_studio.txt` / `xref-mpm_studio.html` (`--keep-work`) and extend `EXCLUDES` |
| A function is fast from source but extremely slow in the exe | an optional accelerator present in the dev venv is excluded from the build. Example found while building v1.0.0: mapclassify's Fisher–Jenks falls back to pure Python without numba (the self-test hung > 15 min) → natural breaks were re-implemented in numpy (`utils.reclass.fisher_jenks_breaks`, exact and ~1 s) |
| Start-up error in the self-test report | the report contains the traceback (`START-UP ERROR`); typically a missing module → `HIDDEN` / `collect_all` |
| Antivirus quarantines the exe | common for unsigned PyInstaller builds; sign the exe (`signtool`) or whitelist the folder |
| Windows SmartScreen warning on first start | unsigned exe downloaded from the internet → "More info ▸ Run anyway", or sign the exe |

## 8. GDAL in the release

GDAL is inside the rasterio / fiona wheels and therefore inside the release (`_internal\rasterio.libs\gdal-*.dll`).
No separate GDAL runtime and no GDAL Python bindings (`osgeo`) are needed or bundled. ArcInfo-grid class labels in
*Data ▸ Convert* are read through rasterio's GDAL with `ctypes` (`python\utils\gdal_rat.py`); the self-test
reports `attribute-table reader: GDAL 3.10.3 (rasterio wheel)`.

## 9. Not included in v1.0.0

- An installer (MSI / Inno Setup) and code signing — the zip is portable.
