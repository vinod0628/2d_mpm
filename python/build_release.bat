@echo off
rem Build the MPM Studio Windows release (PyInstaller).
rem Outputs (python\distribute\):  dist\MPM Studio\ (application)  release\MPM_Studio_<version>_win64.zip  build\ (work, temporary)
rem Options are passed through: --skip-build  --no-zip  --keep-work
setlocal
rem shared offline wheel store (common to all projects); override with the PYTHON_PACKAGES variable
if not defined PYTHON_PACKAGES set "PYTHON_PACKAGES=E:\python_packages"
cd /d "%~dp0"
set "VPY=%~dp0..\.venv\Scripts\python.exe"
if not exist "%VPY%" (
  echo ..\.venv not found - run install_env.bat first.
  exit /b 1
)
"%VPY%" -m PyInstaller --version >nul 2>&1
if errorlevel 1 (
  echo Installing PyInstaller from %PYTHON_PACKAGES% ...
  "%VPY%" -m pip install --no-index --find-links "%PYTHON_PACKAGES%" pyinstaller
  if errorlevel 1 exit /b 1
)
"%VPY%" "%~dp0packaging\build_release.py" %*
exit /b %errorlevel%
