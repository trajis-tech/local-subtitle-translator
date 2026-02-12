@echo off
setlocal
cd /d "%~dp0\.."

REM Install audio model dependencies (HF Wav2Vec2 speech emotion recognition).
REM Run this from project root, or after: .venv\Scripts\activate
REM Requires: pip, and optionally .venv already activated by start.bat

set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"

echo [INFO] Installing audio model dependencies (HF Wav2Vec2 Run A)...
"%PY%" -m pip install --upgrade pip
"%PY%" -m pip install -r requirements_base.txt

if errorlevel 1 (
  echo [ERROR] pip install failed. Try: pip install -r requirements_base.txt
  exit /b 1
)

echo [INFO] Audio dependencies installed. Ensure ffmpeg is in PATH for Run A.
exit /b 0
