# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller recipe for MPM Studio (one-folder Windows build).

Run through build_release.bat / python/packaging/build_release.py, or directly from the python folder:
    ..\\.venv\\Scripts\\pyinstaller.exe packaging\\mpm_studio.spec --noconfirm --distpath dist --workpath build
"""
import os
import sys

from PyInstaller.utils.hooks import collect_all, collect_submodules
from PyInstaller.utils.win32.versioninfo import (FixedFileInfo, StringFileInfo, StringStruct, StringTable,
                                                 VarFileInfo, VarStruct, VSVersionInfo)

SRC = os.path.abspath(os.path.join(SPECPATH, ".."))        # python/
sys.path.insert(0, SRC)
from version import APP_NAME, VERSION  # noqa: E402

ICON = os.path.join(SRC, "gui", "resources", "mpm_studio.ico")
_v = tuple(int(x) for x in VERSION.split(".")) + (0,) * (4 - len(VERSION.split(".")))

version_info = VSVersionInfo(
    ffi=FixedFileInfo(filevers=_v, prodvers=_v, mask=0x3F, flags=0x0, OS=0x40004, fileType=0x1, subtype=0x0),
    kids=[
        StringFileInfo([StringTable("040904B0", [
            StringStruct("CompanyName", "MPM Studio"),
            StringStruct("FileDescription", f"{APP_NAME} — 2D mineral prospectivity mapping"),
            StringStruct("FileVersion", VERSION),
            StringStruct("InternalName", "MPMStudio"),
            StringStruct("OriginalFilename", f"{APP_NAME}.exe"),
            StringStruct("ProductName", APP_NAME),
            StringStruct("ProductVersion", VERSION),
        ])]),
        VarFileInfo([VarStruct("Translation", [0x0409, 1200])]),
    ],
)

# Packages present in the development venv that the application does NOT use. Excluding them keeps the
# build small and avoids pulling in their PyInstaller hooks. Bundled although large: TensorFlow / Keras 2.15
# (CNN tool, ~1.2 GB, CPU on native Windows) and xgboost (XGBoost tool, ~140 MB incl. CUDA).
EXCLUDES = [
    "torch", "jax", "lightgbm", "shap", "numba", "llvmlite",
    "statsmodels", "pykrige", "pysal", "esda", "libpysal", "giddy", "inequality", "mgwr", "momepy", "pointpats",
    "segregation", "spaghetti", "spglm", "spint", "splot", "spopt", "spreg", "spvcm", "tobler", "access",
    "eis_toolkit", "osgeo", "osgeo_utils",   # GDAL comes from the rasterio/fiona wheels
    "pygeos", "imblearn", "seaborn", "skimage",   # networkx: Bayesian network (TAN) and scikit-fuzzy
    "IPython", "ipykernel", "jupyter", "jupyterlab", "jupyter_server", "notebook", "nbformat", "nbconvert",
    "pytest", "_pytest", "tkinter", "PyQt5.QtWebEngineWidgets", "PyQt5.QtWebEngineCore", "PyQt5.QtQml",
    "PyQt5.QtQuick", "PyQt5.Qt3DCore", "PyQt5.QtMultimedia", "PyQt5.QtBluetooth", "PyQt5.QtSql",
    "sphinx", "docutils", "mapclassify",   # natural breaks: utils.reclass.fisher_jenks_breaks
]

# Method plug-ins are imported dynamically (core.methods.load_builtin_methods discovers
# logic/<category>/<method>/plugin.py), so PyInstaller cannot see them: collect the whole logic package.
# New methods are bundled automatically; logic.method_template is only a copy-me skeleton.
os.environ.setdefault("USE_PYGEOS", "0")
HIDDEN = [m for m in collect_submodules("logic") if not m.startswith("logic.method_template")]
HIDDEN += ["matplotlib.backends.backend_qt5agg"]
# CNN tool: TensorFlow is imported lazily inside logic.data_driven.cnn.model._tf(); name it explicitly
# (pyinstaller-hooks-contrib's hook-tensorflow collects its submodules, data files and DLLs).
HIDDEN += ["tensorflow", "keras"]

# Geospatial packages import submodules from compiled (Cython) code, which PyInstaller cannot see
# (e.g. rasterio._base -> rasterio.serde), and ship GDAL/PROJ data files: collect them completely.
DATAS = [(os.path.join(SRC, "gui", "resources"), os.path.join("gui", "resources")),
         (os.path.join(SRC, "i18n", "*.json"), "i18n")]          # translation catalogs
BINARIES = []
for pkg in ("rasterio", "fiona", "pyproj", "shapely", "geopandas", "qtawesome"):
    d, b, h = collect_all(pkg)
    DATAS += d
    BINARIES += b
    HIDDEN += [m for m in h if not any(t in m for t in (".tests", "conftest", ".testing"))]

a = Analysis(
    [os.path.join(SRC, "main.py")],
    pathex=[SRC],
    binaries=BINARIES,
    datas=DATAS,
    hiddenimports=HIDDEN,
    hookspath=[os.path.join(SPECPATH, "hooks")],   # hook-pyproj.py: fixed copy of the contrib hook (Windows paths)
    runtime_hooks=[os.path.join(SPECPATH, "rthook_mpm.py")],
    excludes=EXCLUDES,
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    icon=ICON,
    version=version_info,
    console=False,              # GUI application (no console window)
    upx=False,
    debug=False,
    strip=False,
)

coll = COLLECT(exe, a.binaries, a.datas, name=APP_NAME, upx=False, strip=False)
