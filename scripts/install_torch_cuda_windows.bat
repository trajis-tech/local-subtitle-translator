@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0\.."
if errorlevel 1 (
    echo [ERROR] Cannot cd to project dir: %~dp0..
    pause
    exit /b 1
)

REM ============================================================
REM Install CUDA PyTorch (Windows) - stable cu128/cu126/cu118
REM Optional: nightly cu129 for CUDA 12.9
REM ============================================================

echo.
echo ============================================================
echo  Install CUDA PyTorch (Windows)
echo ============================================================
echo.

set "PY="
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if "!PY!"=="" (
    py -3 -c "pass" 2>nul
    if not errorlevel 1 set "PY=py -3"
)
if "!PY!"=="" (
    py -c "pass" 2>nul
    if not errorlevel 1 set "PY=py"
)
if "!PY!"=="" (
    python -c "pass" 2>nul
    if not errorlevel 1 set "PY=python"
)
if "!PY!"=="" (
    echo [ERROR] Python not found. Create .venv or add Python to PATH.
    pause
    exit /b 1
)

echo [INFO] Python: !PY!
"!PY!" --version 2>nul
if errorlevel 1 (
    echo [ERROR] Python not runnable.
    pause
    exit /b 1
)
echo.

echo Choose build:
echo   1^) Stable CUDA 12.8 ^(recommended^)
echo   2^) Stable CUDA 12.6 ^(fallback^)
echo   3^) Stable CUDA 11.8 ^(wider compatibility^)
echo   4^) Nightly CUDA 12.9 ^(only if you need cu129^)
echo   5^) CPU only ^(no GPU^)
echo   6^) Cancel
echo.
set /p CHOICE="Enter 1/2/3/4/5/6: "

if "!CHOICE!"=="1" (
    set "INDEX_URL=https://download.pytorch.org/whl/cu128"
    set "CUDA_VER=cu128"
) else if "!CHOICE!"=="2" (
    set "INDEX_URL=https://download.pytorch.org/whl/cu126"
    set "CUDA_VER=cu126"
) else if "!CHOICE!"=="3" (
    set "INDEX_URL=https://download.pytorch.org/whl/cu118"
    set "CUDA_VER=cu118"
) else if "!CHOICE!"=="4" (
    set "INDEX_URL=https://download.pytorch.org/whl/nightly/cu129"
    set "CUDA_VER=cu129-nightly"
    set "EXTRA_FLAGS=--pre"
) else if "!CHOICE!"=="5" (
    set "INDEX_URL=https://download.pytorch.org/whl/cpu"
    set "CUDA_VER=cpu"
    set "EXTRA_FLAGS="
) else if "!CHOICE!"=="6" (
    echo [INFO] Cancelled.
    pause
    exit /b 0
) else (
    echo [INFO] Invalid. Cancelled.
    pause
    exit /b 0
)
if not defined EXTRA_FLAGS set "EXTRA_FLAGS="

echo.
echo [INFO] Installing !CUDA_VER! ...
echo.

echo [STEP 1/3] Uninstalling existing torch...
"!PY!" -m pip uninstall -y torch 2>nul
echo.

echo [STEP 2/3] Installing torch ^(!CUDA_VER!^)...
if "!EXTRA_FLAGS!"=="" (
    "!PY!" -m pip install torch --index-url !INDEX_URL!
) else (
    "!PY!" -m pip install !EXTRA_FLAGS! torch --index-url !INDEX_URL!
)
if errorlevel 1 (
    echo [ERROR] Install failed. Try: !PY! -m pip install --upgrade pip
    pause
    exit /b 1
)

echo.
echo [STEP 3/3] Verifying...
echo.
"!PY!" -c "import torch; print(torch.__version__); print('torch.version.cuda=', torch.version.cuda); print('cuda_available=', torch.cuda.is_available()); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

echo.
echo ============================================================
if "!CUDA_VER!"=="cpu" (
    echo  Done ^(CPU version^)
) else (
    echo  Done. Check above: cuda_available should be True
    echo  If False: update NVIDIA driver or try another CUDA option
)
echo ============================================================
echo.
echo Press any key to close...
pause >nul
exit /b 0
