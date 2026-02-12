@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

REM =========================================================
REM  Local Subtitle Translator - Uninstall (Windows)
REM  Counterpart of uninstall.sh (Linux/macOS).
REM  Removes only project-local files inside this folder.
REM =========================================================

echo ==========================================
echo  Local Subtitle Translator - Uninstall
echo ==========================================
echo This will ONLY remove files inside this folder:
echo   - runtime\          (portable Python, ffmpeg, caches, temp)
echo   - .venv\            (local Python environment)
echo   - models\           (GGUF + Run A audio model)
echo   - work\             (intermediate translation results, JSONL)
echo   - data\             (glossary)
echo   - gradio_cached_examples
echo   - config.json, *.log
echo.
choice /M "Continue?"
if errorlevel 2 exit /b 0

echo.
echo [1/1] Removing project-local files...

if exist "runtime" rmdir /s /q "runtime"
if exist ".venv" rmdir /s /q ".venv"
if exist "models" rmdir /s /q "models"
if exist "work" rmdir /s /q "work"
if exist "data" rmdir /s /q "data"
if exist "gradio_cached_examples" rmdir /s /q "gradio_cached_examples"

if exist "config.json" del /q "config.json"
del /q "*.log" >nul 2>nul

echo.
echo Uninstall completed.
echo You can now delete this folder manually if you want.
pause
