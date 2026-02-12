@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"
set "ROOT=%cd%"

REM =========================================================
REM  Local Subtitle Translator - START (offline launch only)
REM  Run this to launch the app. No downloads, no network.
REM  Run install.bat first (once, with network) if not yet installed.
REM =========================================================

set "VENV_DIR=%ROOT%\.venv"
set "RUNTIME_DIR=%ROOT%\runtime"
set "FFMPEG_BIN=%RUNTIME_DIR%\ffmpeg\bin"

REM ---- Isolate caches & temp inside this folder (same as install) ----
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONNOUSERSITE=1"
set "APP_CACHE=%RUNTIME_DIR%\cache"
set "HF_HOME=%APP_CACHE%\hf"
set "HUGGINGFACE_HUB_CACHE=%HF_HOME%\hub"
set "HF_HUB_DISABLE_TELEMETRY=1"
set "PIP_CACHE_DIR=%APP_CACHE%\pip"
set "TEMP=%RUNTIME_DIR%\temp"
set "TMP=%RUNTIME_DIR%\temp"
set "GRADIO_TEMP_DIR=%RUNTIME_DIR%\temp\gradio"

REM ---- PATH: include portable ffmpeg if present ----
if exist "%FFMPEG_BIN%\ffmpeg.exe" set "PATH=%FFMPEG_BIN%;%PATH%"

REM ------------------------------
REM 1) Require .venv (run install.bat first)
REM ------------------------------
if not exist "%VENV_DIR%\Scripts\python.exe" (
  echo [ERROR] .venv not found. Run install.bat first ^(one-time, with network^).
  echo         Then use start.bat for offline launch.
  pause
  exit /b 1
)

call "%VENV_DIR%\Scripts\activate.bat"

REM ------------------------------
REM 2) Check local model files (offline check; exit if missing)
REM ------------------------------
echo [INFO] Checking local model files...
python scripts\check_models.py
if errorlevel 1 (
  echo.
  echo [ACTION REQUIRED] Download the models manually, then put them into:
  echo   %ROOT%\models
  echo.
  echo See README.md for download links. Then run start.bat again.
  echo Opening README.md...
  start "" "%ROOT%\README.md"
  pause
  exit /b 1
)

REM ------------------------------
REM 3) Ensure model_prompts.csv has UTF-8 BOM (local file only)
REM ------------------------------
python scripts\ensure_csv_bom.py

REM ------------------------------
REM 4) Launch app (offline)
REM ------------------------------
echo [INFO] Launching app...
python app.py

pause
