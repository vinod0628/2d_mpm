@echo off
rem ===========================================================================
rem  2D MPM - offline environment installer (Windows x64)
rem
rem  This file lives in the python folder. It creates ..\.venv (repository root, Python 3.10)
rem  from the local wheels in python\whl\ with NO internet:
rem    1. Python 3.10.11   - uses an existing install, else installs whl\runtime\python-3.10.11-amd64.exe (per-user)
rem    2. VC++ runtime     - installs whl\runtime\vc_redist.x64.exe only if msvcp140_1.dll is missing (needs admin)
rem    3. ..\.venv         - python -m venv, then pip install --no-index from whl\
rem    4. pip check + smoke test (tools\pip_check.py, tools\smoke_test.py)
rem
rem  GDAL is provided by the rasterio / fiona wheels - no separate GDAL install is needed.
rem
rem  Usage:   install_env.bat            install / repair
rem           install_env.bat /clean     delete ..\.venv first, then install
rem           install_env.bat /test      only run the checks
rem ===========================================================================
setlocal EnableExtensions
cd /d "%~dp0"
set "ROOT=%~dp0"
rem the virtual environment stays in the repository root (one level up from this folder)
for %%I in ("%~dp0..") do set "REPO=%%~fI"
set "WHL=%ROOT%whl"
set "RT=%WHL%\runtime"
set "VENV=%REPO%\.venv"
set "VPY=%VENV%\Scripts\python.exe"
set "REQ=%ROOT%requirements-offline.txt"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"

if /i "%~1"=="/test" goto :checks

if not exist "%WHL%\rasterio-*.whl" (
  echo [ERROR] whl\ folder is missing or incomplete: %WHL%
  goto :fail
)
if not exist "%REQ%" (
  echo [ERROR] missing %REQ%
  goto :fail
)

rem ---------------------------------------------------------------- 1. Python 3.10
echo.
echo [1/4] Python 3.10
set "PY="
if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
if not defined PY if exist "%ProgramFiles%\Python310\python.exe" set "PY=%ProgramFiles%\Python310\python.exe"
if not defined PY (
  echo       not found - installing Python 3.10.11 for the current user ...
  "%RT%\python-3.10.11-amd64.exe" /quiet InstallAllUsers=0 PrependPath=0 Include_launcher=0 Include_test=0 Shortcuts=0
  if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
)
if not defined PY (
  echo [ERROR] Python 3.10 installation failed.
  goto :fail
)
echo       using %PY%

rem ---------------------------------------------------------------- 2. VC++ runtime
echo.
echo [2/4] Microsoft Visual C++ 2015-2022 runtime
if exist "%SystemRoot%\System32\msvcp140_1.dll" (
  echo       already present
) else (
  echo       missing - running vc_redist.x64.exe ^(accept the UAC prompt^) ...
  "%RT%\vc_redist.x64.exe" /install /quiet /norestart
)

rem ---------------------------------------------------------------- 3. venv + wheels
echo.
echo [3/4] virtual environment %VENV% + packages ^(offline, from whl\^)
if /i "%~1"=="/clean" if exist "%VENV%" (
  echo       /clean: removing existing .venv ...
  rmdir /s /q "%VENV%"
)
if not exist "%VPY%" (
  "%PY%" -m venv "%VENV%"
  if errorlevel 1 goto :fail
)
rem pip/wheel are not in the lock file (pip freeze omits them); astunparse (tensorflow) needs wheel
"%VPY%" -m pip install --no-index --find-links "%WHL%" --upgrade pip wheel --quiet
if errorlevel 1 goto :fail
"%VPY%" -m pip install --no-index --find-links "%WHL%" --no-deps -r "%REQ%"
if errorlevel 1 goto :fail

rem ---------------------------------------------------------------- 4. checks
:checks
echo.
echo [4/4] pip check + smoke test
if not exist "%VPY%" (
  echo [ERROR] %VPY% not found - run install_env.bat without /test first.
  goto :fail
)
"%VPY%" "%ROOT%tools\pip_check.py"
if errorlevel 1 goto :fail
"%VPY%" "%ROOT%tools\smoke_test.py"
if errorlevel 1 goto :fail

echo.
echo DONE. Activate with:   %VENV%\Scripts\activate.bat     ^(cmd^)
echo                    or:  %VENV%\Scripts\Activate.ps1     ^(PowerShell^)
echo Start MPM Studio:       run_mpm.bat
endlocal
exit /b 0

:fail
echo.
echo *** INSTALL FAILED - see messages above ***
endlocal
exit /b 1
