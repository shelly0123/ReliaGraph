@echo off
setlocal EnableExtensions
cd /d "%~dp0"
echo.
echo  RELIAGRAPH - OPTIONAL MULTILINGUAL MODE
echo This installs the optional multilingual embedding dependencies and starts a separate local server.
echo Model weights may download on first use and require additional disk space.
echo.
if not exist ".venv\Scripts\python.exe" (
  echo The base Python environment does not exist yet.
  echo First run run_reliagraph_windows.bat and wait until the dashboard starts.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m pip install -r requirements-language.txt
if errorlevel 1 (
  echo Could not install multilingual dependencies. Check internet access, disk space, and pip's error above.
  pause
  exit /b 1
)
set "TW_ENABLE_CROSSLINGUAL=1"
echo Starting ReliaGraph with multilingual retrieval enabled at http://127.0.0.1:8765
echo The model must load successfully before Hindi-English quality can be claimed.
echo Keep this window open. Press Ctrl+C to stop the server.
start "ReliaGraph Browser" /b cmd /c "ping 127.0.0.1 -n 4 >nul ^& start http://127.0.0.1:8765"
".venv\Scripts\python.exe" -m uvicorn webapp.main:app --host 127.0.0.1 --port 8765
