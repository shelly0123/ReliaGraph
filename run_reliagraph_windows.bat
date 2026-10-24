@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo.
echo  RELIAGRAPH - LOCAL KNOWLEDGE WORKSPACE
echo  This prototype binds to 127.0.0.1 only.
echo.

REM Prefer Python 3.14; fall back to 3.13, 3.12 or 3.11 if needed.
set "PY_CMD="
where py >nul 2>nul
if not errorlevel 1 (
  py -3.14 -c "import sys" >nul 2>nul
  if not errorlevel 1 set "PY_CMD=py -3.14"
  if defined PY_CMD goto :python_found
  py -3.13 -c "import sys" >nul 2>nul
  if not errorlevel 1 set "PY_CMD=py -3.13"
  if defined PY_CMD goto :python_found
  py -3.12 -c "import sys" >nul 2>nul
  if not errorlevel 1 set "PY_CMD=py -3.12"
  if defined PY_CMD goto :python_found
  py -3.11 -c "import sys" >nul 2>nul
  if not errorlevel 1 set "PY_CMD=py -3.11"
  if defined PY_CMD goto :python_found
  py -c "import sys; raise SystemExit(0 if (3,11) <= sys.version_info[:2] < (3,15) else 1)" >nul 2>nul
  if not errorlevel 1 set "PY_CMD=py"
  if defined PY_CMD goto :python_found
)
where python >nul 2>nul
if not errorlevel 1 (
  python -c "import sys; raise SystemExit(0 if (3,11) <= sys.version_info[:2] < (3,15) else 1)" >nul 2>nul
  if not errorlevel 1 set "PY_CMD=python"
)
if not defined PY_CMD (
  echo ERROR: Python 3.11, 3.12, 3.13 or 3.14 was not found.
  echo Install 64-bit Python from https://www.python.org/downloads/windows/
  echo During installation, enable the option to add Python to PATH.
  pause
  exit /b 1
)

:python_found
echo Using: %PY_CMD%
echo Python 3.14 is preferred when installed; supported older versions are accepted as fallback.
if not exist ".venv\Scripts\python.exe" (
  if exist ".venv" (
    echo An incomplete .venv folder exists. Renaming it before recreating the environment...
    if exist ".venv_incomplete" rmdir /s /q ".venv_incomplete"
    ren ".venv" ".venv_incomplete" >nul 2>nul
    if errorlevel 1 (
      echo ERROR: Could not rename the incomplete .venv folder.
      echo Close Python/VS Code windows using this folder, then delete .venv and run this file again.
      pause
      exit /b 1
    )
  )
  echo Creating the isolated Python environment...
  %PY_CMD% -m venv .venv
  if errorlevel 1 goto :venv_error
)
if not exist ".venv\Scripts\python.exe" goto :venv_error

echo Installing dashboard dependencies. The first run requires internet access...
".venv\Scripts\python.exe" -m ensurepip --upgrade >nul 2>nul
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :install_error
".venv\Scripts\python.exe" -m pip install -r requirements-dashboard.txt
if errorlevel 1 goto :install_error

echo.
echo Installing optional multilingual retrieval support (model weights download on first use)...
".venv\Scripts\python.exe" -m pip install -r requirements-language.txt
if errorlevel 1 echo WARNING: multilingual packages did not install; English/local baseline will still run.
echo Installing Neo4j driver and ChromaDB...
".venv\Scripts\python.exe" -m pip install -r requirements-stores.txt
if errorlevel 1 echo WARNING: store packages did not install; SQLite/NetworkX fallback will be used.
if exist "reliagraph.env" (
  for /f "usebackq eol=# tokens=1,* delims==" %%a in ("reliagraph.env") do set "%%a=%%b"
  echo Loaded settings from reliagraph.env
)
set "TW_ENABLE_CROSSLINGUAL=1"
echo Starting ReliaGraph at http://127.0.0.1:8765
echo Leave this window open while using the dashboard. Press Ctrl+C to stop it.
start "ReliaGraph Browser" /b cmd /c "ping 127.0.0.1 -n 4 >nul ^& start http://127.0.0.1:8765"
".venv\Scripts\python.exe" -m uvicorn webapp.main:app --host 127.0.0.1 --port 8765
exit /b 0

:venv_error
echo.
echo ERROR: Python could not create .venv.
echo Check that Python is installed correctly and that this folder is writable.
echo If .venv already exists and is broken, close this window, delete the .venv folder, and retry.
echo See docs\USER_GUIDE.md for troubleshooting.
pause
exit /b 1

:install_error
echo.
echo ERROR: Dependency installation failed.
echo Check internet access, available disk space, and the pip error printed above.
echo You can retry by running run_reliagraph_windows.bat again.
pause
exit /b 1
