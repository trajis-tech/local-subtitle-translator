@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"
set "ROOT=%cd%"

REM =========================================================
REM  Local Subtitle Translator - INSTALL (online / one-time)
REM  Run this ONCE with network: downloads and installs everything.
REM  After this, use start.bat for OFFLINE launch only.
REM
REM  Install does:
REM  - Download portable Python into .\runtime\python
REM  - Create .\.venv (virtualenv)
REM  - Install deps: base + audio (HF Wav2Vec2/torch/transformers) + llama-cpp-python
REM  - Download ffmpeg to .\runtime\ffmpeg if not in PATH
REM  - Download Run A audio model to models\audio
REM  - Create config if missing; ensure model_prompts.csv BOM
REM  - (Manual) You still download GGUF models into .\models
REM =========================================================

set "PY_VER=3.12.10"
set "PY_EMBED_URL=https://www.python.org/ftp/python/%PY_VER%/python-%PY_VER%-embed-amd64.zip"
set "GETPIP_URL=https://bootstrap.pypa.io/get-pip.py"

set "RUNTIME_DIR=%ROOT%\runtime"
set "PY_DIR=%RUNTIME_DIR%\python"
set "PY_EXE=%PY_DIR%\python.exe"
set "VENV_DIR=%ROOT%\.venv"
set "FFMPEG_DIR=%RUNTIME_DIR%\ffmpeg"
set "FFMPEG_BIN=%FFMPEG_DIR%\bin"
set "FFMPEG_URL=https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl-shared.zip"

REM ---- Isolate caches & temp inside this folder ----
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

if not exist "%RUNTIME_DIR%" mkdir "%RUNTIME_DIR%" >nul 2>nul
if not exist "%APP_CACHE%" mkdir "%APP_CACHE%" >nul 2>nul
if not exist "%TEMP%" mkdir "%TEMP%" >nul 2>nul

REM ------------------------------
REM 1) Download portable Python
REM ------------------------------
if not exist "%PY_EXE%" (
  echo [INFO] Downloading portable Python %PY_VER%...
  powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$ErrorActionPreference='Stop';" ^
    "New-Item -ItemType Directory -Force '%RUNTIME_DIR%' | Out-Null;" ^
    "$zip='%RUNTIME_DIR%\\python-embed.zip';" ^
    "Invoke-WebRequest -Uri '%PY_EMBED_URL%' -OutFile $zip;" ^
    "if(Test-Path '%PY_DIR%'){Remove-Item -Recurse -Force '%PY_DIR%'};" ^
    "New-Item -ItemType Directory -Force '%PY_DIR%' | Out-Null;" ^
    "Expand-Archive -Path $zip -DestinationPath '%PY_DIR%' -Force;" ^
    "Remove-Item $zip -Force;"
  if errorlevel 1 (
    echo [ERROR] Failed to download/extract portable Python.
    echo         If PowerShell downloads are blocked, download the embeddable zip manually:
    echo         %PY_EMBED_URL%
    pause
    exit /b 1
  )
)

if not exist "%PY_EXE%" (
  echo [ERROR] Portable Python not found: %PY_EXE%
  pause
  exit /b 1
)

REM --------------------------------------------
REM 2) Enable "import site" for embeddable Python
REM --------------------------------------------
for %%F in ("%PY_DIR%\python*._pth") do (
  set "PTH_FILE=%%~fF"
)
if defined PTH_FILE (
  powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$p='%PTH_FILE%';" ^
    "$t=Get-Content -Raw $p;" ^
    "if($t -match '#\s*import\s+site'){ $t=$t -replace '#\s*import\s+site','import site'; Set-Content -NoNewline -Encoding ASCII $p $t }"
)

REM ------------------------------
REM 3) Bootstrap pip (get-pip.py)
REM ------------------------------
"%PY_EXE%" -m pip --version >nul 2>nul
if errorlevel 1 (
  echo [INFO] Bootstrapping pip...
  powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$ErrorActionPreference='Stop';" ^
    "$gp='%RUNTIME_DIR%\\get-pip.py';" ^
    "Invoke-WebRequest -Uri '%GETPIP_URL%' -OutFile $gp;"
  if errorlevel 1 (
    echo [ERROR] Failed to download get-pip.py
    pause
    exit /b 1
  )
  "%PY_EXE%" "%RUNTIME_DIR%\get-pip.py" --no-warn-script-location
  if errorlevel 1 (
    echo [ERROR] pip bootstrap failed.
    pause
    exit /b 1
  )
)

REM ---------------------------------
REM 4) Create venv with virtualenv
REM ---------------------------------
echo [INFO] Preparing virtualenv...
"%PY_EXE%" -m pip install -U pip virtualenv --no-warn-script-location
if errorlevel 1 (
  echo [ERROR] Failed to install virtualenv using portable Python.
  pause
  exit /b 1
)

if not exist "%VENV_DIR%\Scripts\python.exe" (
  echo [INFO] Creating local environment .venv ...
  "%PY_EXE%" -m virtualenv "%VENV_DIR%" --copies --clear
  if errorlevel 1 (
    echo [ERROR] Failed to create .venv.
    pause
    exit /b 1
  )
)

call "%VENV_DIR%\Scripts\activate.bat"

echo [INFO] Upgrading pip inside .venv ...
python -m pip install -U pip setuptools wheel --no-warn-script-location
if errorlevel 1 (
  echo [ERROR] Failed to upgrade pip in .venv.
  pause
  exit /b 1
)

REM ------------------------------
REM 5) Install base requirements (Run A: HF Wav2Vec2 emotion model)
REM ------------------------------
echo [INFO] Installing Python requirements (base + audio Run A + video)...
python -m pip install -r requirements_base.txt --no-warn-script-location
if errorlevel 1 (
  echo [ERROR] Failed to install base requirements.
  echo         Retry: python -m pip install -r requirements_base.txt
  pause
  exit /b 1
)
python -c "import torch; import transformers; import soundfile; import scipy; print('[INFO] Run A (audio) deps OK')" 2>nul
if errorlevel 1 (
  echo [WARN] Run A audio deps import check failed. Run A may fail. Retry: python -m pip install -r requirements_base.txt
)

REM ------------------------------
REM 5a) Ensure CUDA PyTorch if NVIDIA GPU present
REM ------------------------------
python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>nul
if errorlevel 1 (
  where nvidia-smi >nul 2>nul
  if not errorlevel 1 (
    echo [INFO] NVIDIA GPU detected but PyTorch is CPU-only. Installing CUDA 12.8 torch...
    python -m pip uninstall -y torch 2>nul
    python -m pip install torch --index-url https://download.pytorch.org/whl/cu128 --no-warn-script-location
    if errorlevel 1 (
      echo [WARN] CUDA torch install failed. Run A will use CPU. You can retry later: scripts\install_torch_cuda_windows.bat
    ) else (
      echo [INFO] CUDA PyTorch installed. Verifying...
      python -c "import torch; print('cuda_available=', torch.cuda.is_available(), 'gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
    )
  )
)

REM ------------------------------
REM 5b) Ensure ffmpeg (download to runtime\ffmpeg if not in PATH)
REM ------------------------------
where ffmpeg >nul 2>nul
if errorlevel 1 (
  if not exist "%FFMPEG_BIN%\ffmpeg.exe" (
    echo [INFO] ffmpeg not found. Downloading portable ffmpeg to %FFMPEG_DIR%...
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
      "$ErrorActionPreference='Stop';" ^
      "New-Item -ItemType Directory -Force '%RUNTIME_DIR%' | Out-Null;" ^
      "$zip='%RUNTIME_DIR%\\ffmpeg.zip';" ^
      "Invoke-WebRequest -Uri '%FFMPEG_URL%' -OutFile $zip -UseBasicParsing;" ^
      "$extract='%RUNTIME_DIR%\\ffmpeg_extract';" ^
      "if(Test-Path $extract){Remove-Item -Recurse -Force $extract};" ^
      "Expand-Archive -Path $zip -DestinationPath $extract -Force;" ^
      "$top=Get-ChildItem -Path $extract -Directory | Select-Object -First 1;" ^
      "if(-not $top){ throw 'No folder in zip' };" ^
      "if(Test-Path '%FFMPEG_DIR%'){Remove-Item -Recurse -Force '%FFMPEG_DIR%'};" ^
      "Move-Item -Path $top.FullName -Destination '%FFMPEG_DIR%';" ^
      "Remove-Item $zip -Force -ErrorAction SilentlyContinue;" ^
      "Remove-Item $extract -Recurse -Force -ErrorAction SilentlyContinue;"
    if errorlevel 1 (
      echo [WARN] ffmpeg auto-download failed. See FFMPEG_INSTALL.md for manual install.
      echo        Run A ^(audio^) and video features need ffmpeg in PATH.
    ) else (
      echo [INFO] ffmpeg installed to %FFMPEG_DIR%
    )
  )
)

REM ------------------------------
REM 5c) Download Run A audio model to models\audio (~1.27 GB)
REM ------------------------------
echo [INFO] Downloading Run A audio model to models\audio (~1.27 GB)...
python scripts\download_audio_model.py
if errorlevel 1 (
  echo [WARN] Audio model download failed or skipped. Run A will download on first translation if needed.
)

REM ------------------------------
REM 6a) Download llama-cpp-python wheel to build_wheel\dist
REM ------------------------------
set "WHEEL_DIR=%ROOT%\build_wheel\dist"
set "LLAMA_WHEEL_URL=https://huggingface.co/trajis-tech/llama-cpp-python-trajis-tech-nonavx512-cuda/resolve/main/llama_cpp_python-0.3.16-cp312-cp312-win_amd64.whl"
set "LLAMA_WHEEL_FILE=%WHEEL_DIR%\llama_cpp_python-0.3.16-cp312-cp312-win_amd64.whl"

if not exist "%WHEEL_DIR%" mkdir "%WHEEL_DIR%" >nul 2>nul
if not exist "%LLAMA_WHEEL_FILE%" (
  echo [INFO] Downloading llama-cpp-python wheel to %WHEEL_DIR%...
  powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$ErrorActionPreference='Stop';" ^
    "Invoke-WebRequest -Uri '%LLAMA_WHEEL_URL%' -OutFile '%LLAMA_WHEEL_FILE%' -UseBasicParsing;"
  if errorlevel 1 (
    echo [WARN] Failed to download llama-cpp-python wheel.
  ) else (
    echo [INFO] Wheel downloaded: %LLAMA_WHEEL_FILE%
  )
) else (
  echo [INFO] Wheel already exists: %LLAMA_WHEEL_FILE%
)

REM ------------------------------
REM 6) Install llama-cpp-python
REM ------------------------------
echo [INFO] Installing llama-cpp-python (GPU if available)...
python scripts\install_llama_cpp.py
if errorlevel 1 (
  echo [ERROR] llama-cpp-python install failed.
  echo         You can still try CPU-only by rerunning:
  echo         .venv\Scripts\python.exe -m pip install --only-binary=:all: llama-cpp-python
  pause
  exit /b 1
)

REM ------------------------------
REM 7) Create recommended config (if missing)
REM ------------------------------
echo [INFO] Creating recommended config (if missing)...
python scripts\plan_models.py --write-config-if-missing

REM ------------------------------
REM 8) Check local model files (manual download; warn only so install completes)
REM ------------------------------
echo [INFO] Checking local model files...
python scripts\check_models.py
if errorlevel 1 (
  echo.
  echo [WARN] GGUF models not yet in %ROOT%\models
  echo        Download them manually ^(see README^), then run start.bat to launch.
  echo.
)

REM ------------------------------
REM 9) Ensure model_prompts.csv has UTF-8 BOM
REM ------------------------------
echo [INFO] Ensuring model_prompts.csv has UTF-8 BOM...
python scripts\ensure_csv_bom.py

echo.
echo =========================================================
echo  Installation complete.
echo  Run start.bat to launch the app ^(offline^).
echo =========================================================
echo.
pause
