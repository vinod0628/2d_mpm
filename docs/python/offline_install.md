# Offline installation with `install_env.bat`

`python\install_env.bat` builds the Python development environment (`.venv` in the repository root) on a Windows
PC **without internet**, using only files shipped inside the project folder. Use it after copying the project to
another PC, or whenever `.venv` is missing or broken. `.venv` itself (≈2.5 GB) does not need to be copied.

(To only *use* MPM Studio on another PC, the release zip is enough — no Python, see
[how_to_build_python_app.md](how_to_build_python_app.md).)

## 1. What to copy

Copy the whole project folder. The parts the installer needs:

```
2d_mpm\
├─ .venv\                             ← created by the installer (do not copy)
├─ demo_data\                         ← optional; three smoke tests use it and are skipped without it
└─ python\
   ├─ install_env.bat                 ← the installer
   ├─ run_mpm.bat                     ← start MPM Studio from source
   ├─ build_release.bat               ← build the Windows release
   ├─ requirements-offline.txt        ← exact package list the installer uses
   ├─ whl\                            ← 220 wheels (≈806 MB): every package in the lock file (incl. PyInstaller) + pip + wheel + setuptools
   │  └─ runtime\
   │     ├─ python-3.10.11-amd64.exe  ← Python installer (used only if Python 3.10 is not installed)
   │     └─ vc_redist.x64.exe         ← Microsoft VC++ 2015-2022 runtime (used only if missing)
   └─ tools\
      ├─ pip_check.py                 ← pip check (ignores the known eis_toolkit→gdal metadata line)
      └─ smoke_test.py                ← post-install checks
```

You can leave out `.venv\` and `python\distribute\` (build outputs). No GDAL installation and no Build Tools are
needed: GDAL comes inside the rasterio / fiona wheels (see [python_packages.md](python_packages.md)).

## 2. Requirements on the target PC

- Windows 10 or 11, **64-bit**.
- About **3.5 GB free disk space** (`.venv` ≈2.5 GB, Python ≈100 MB).
- No admin rights needed, unless the VC++ runtime is missing. Then `vc_redist.x64.exe` asks for elevation (a UAC prompt).

## 3. Usage

Run from any folder (the script finds its own location):

| Command | What it does |
|---|---|
| `python\install_env.bat` | Install, or repair an existing `.venv` (already-installed packages are kept) |
| `python\install_env.bat /clean` | Delete `.venv` first, then do a full fresh install |
| `python\install_env.bat /test` | Only run the checks on the existing `.venv` |

From PowerShell prefix with `.\` (e.g. `.\python\install_env.bat`). Exit code: `0` = success, `1` = failure (the last
messages show which step failed). A full `/clean` install takes about **15 minutes**; unpacking tensorflow is most of it.

## 4. What the installer does

| Step | Action | Skipped when |
|---|---|---|
| 1/4 Python 3.10 | Uses `%LOCALAPPDATA%\Programs\Python\Python310\python.exe` or `%ProgramFiles%\Python310\python.exe`; otherwise silently installs `python-3.10.11-amd64.exe` **for the current user** (no PATH change, no py launcher) | Python 3.10 already installed |
| 2/4 VC++ runtime | Runs `vc_redist.x64.exe /install /quiet /norestart` | `C:\Windows\System32\msvcp140_1.dll` exists |
| 3/4 venv + packages | `python -m venv <repo>\.venv`, upgrade `pip` + install `wheel` (both left out of `pip freeze`; `wheel` is needed by tensorflow's `astunparse`), then `pip install --no-index --find-links python\whl --no-deps -r python\requirements-offline.txt` (pip **never** goes online) | — |
| 4/4 Checks | `python\tools\pip_check.py` + `python\tools\smoke_test.py` | — |

The smoke test checks:
1. Python is 3.10.
2. GDAL inside the rasterio wheel: EPSG lookups and the ArcInfo attribute-table reader (`utils/gdal_rat.py`).
3. Core geospatial imports work.
4. All eis_toolkit modules import (except its `utilities.gdal` helper, which needs the unused `osgeo` bindings).
5. RF, SVM, Logit, XGBoost, LightGBM, SMOTE and Keras all fit a small test dataset.
6. `eis_toolkit` WofE runs on the Carlin data.
7. `eis_toolkit` `distance_to_anomaly` runs on the Curnamona data (numba implementation).

Expected end of output:

```
SMOKE TEST PASSED: 7/7 ok

DONE. Activate with:   C:\...\2d_mpm\.venv\Scripts\activate.bat     (cmd)
                   or:  C:\...\2d_mpm\.venv\Scripts\Activate.ps1     (PowerShell)
Start MPM Studio:       run_mpm.bat
```

`pip check` prints `(ignored …) eis-toolkit 1.1.6 requires gdal, which is not installed.` — this is expected.

## 5. After installation

```powershell
.\.venv\Scripts\Activate.ps1          # PowerShell, from the repository root (if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned)
.venv\Scripts\activate.bat            # cmd
python\run_mpm.bat                    # start MPM Studio from source
python python\tools\smoke_test.py     # re-run checks any time
```

In VS Code select the interpreter `.venv\Scripts\python.exe`.

## 6. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `whl\ folder is missing or incomplete` | Copy the whole `python\whl\` folder, including `whl\runtime\` |
| `Python 3.10 installation failed` | Run `python\whl\runtime\python-3.10.11-amd64.exe` manually (choose "Install for current user"), then re-run the bat |
| `ERROR: Could not find a version that satisfies ...` during step 3 | A wheel is missing from `python\whl\` — re-copy it from the source PC |
| `tensorflow` DLL load failed | VC++ runtime missing: run `python\whl\runtime\vc_redist.x64.exe` as admin |
| Anything else | `python\install_env.bat /clean` for a fresh start |

## 7. Updating the wheel set (on the online PC)

After adding or upgrading packages in `.venv` (commands run from the repository root):

```powershell
.venv\Scripts\python.exe -m pip freeze > python\requirements-lock.txt
Copy-Item python\requirements-lock.txt python\requirements-offline.txt
# wheels: downloads binaries, builds wheels for source-only packages
.venv\Scripts\python.exe -m pip wheel --no-deps -w python\whl -r python\requirements-offline.txt
# pip and wheel are not in pip freeze output - fetch them separately
.venv\Scripts\python.exe -m pip download --no-deps -d python\whl pip wheel
```

Then delete wheels in `python\whl\` whose versions are no longer in the lock file, and run
`python\install_env.bat /clean` to confirm.
