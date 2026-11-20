@echo off
rem Start MPM Studio (2D mineral prospectivity mapping) from source. Optional argument: a project file (*.mpm.json)
rem The virtual environment lives in the repository root: ..\.venv  (create it with install_env.bat)
setlocal
cd /d "%~dp0"
set "VPY=%~dp0..\.venv\Scripts\python.exe"
if not exist "%VPY%" (
  echo ..\.venv not found - run install_env.bat first.
  exit /b 1
)
"%VPY%" "%~dp0main.py" %*
endlocal
