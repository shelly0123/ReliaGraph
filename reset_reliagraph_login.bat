@echo off
setlocal EnableExtensions
cd /d "%~dp0"
echo.
echo  RELIAGRAPH - RESET LOCAL LOGIN
 echo This clears only the account and session rows, not documents or the graph.
echo.
echo IMPORTANT: Close the ReliaGraph server window before continuing.
pause
set "PY_CMD="
where py >nul 2>nul
if not errorlevel 1 (
  py -3.14 -c "import sqlite3" >nul 2>nul
  if not errorlevel 1 set "PY_CMD=py -3.14"
  if defined PY_CMD goto :run
  py -3.13 -c "import sqlite3" >nul 2>nul
  if not errorlevel 1 set "PY_CMD=py -3.13"
  if defined PY_CMD goto :run
  py -3.12 -c "import sqlite3" >nul 2>nul
  if not errorlevel 1 set "PY_CMD=py -3.12"
  if defined PY_CMD goto :run
  py -3.11 -c "import sqlite3" >nul 2>nul
  if not errorlevel 1 set "PY_CMD=py -3.11"
  if defined PY_CMD goto :run
)
where python >nul 2>nul
if not errorlevel 1 (
  python -c "import sqlite3" >nul 2>nul
  if not errorlevel 1 set "PY_CMD=python"
)
if not defined PY_CMD (
  echo ERROR: Python was not found. Install Python 3.11-3.14 first.
  pause
  exit /b 1
)
:run
%PY_CMD% "%~dp0reset_reliagraph_login.py"
pause
