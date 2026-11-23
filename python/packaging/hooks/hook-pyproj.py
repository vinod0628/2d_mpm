"""PyInstaller hook for pyproj — replaces the pyinstaller-hooks-contrib one (hookspath takes precedence).

The contrib hook checks whether pyproj's PROJ data was collected with
``dest.startswith("pyproj/proj_dir")``, but on Windows the destinations use backslashes
(``pyproj\\proj_dir\\share\\proj``). The check therefore always fails with the pip wheel: the hook looks for a
conda-style ``<venv>\\Library\\share\\proj`` instead and warns "Datas for pyproj not found", although the
data (including proj.db, needed for every EPSG lookup offline) has been collected. This copy compares
normalised paths and warns only when proj.db is really missing.
"""
import os

from PyInstaller.utils.hooks import collect_data_files, collect_delvewheel_libs_directory, logger

hiddenimports = ["pyproj.datadir", "pyproj._compat"]
binaries = []
datas = collect_data_files("pyproj")

if not any(os.path.basename(src) == "proj.db" for src, _dest in datas):
    logger.warning("pyproj: proj.db (PROJ coordinate-system database) not found in the pyproj package — "
                   "the frozen application could not look up EPSG codes offline.")

# pyproj >= 3.4 Windows wheels keep their DLLs in pyproj.libs (delvewheel)
datas, binaries = collect_delvewheel_libs_directory("pyproj", datas=datas, binaries=binaries)
