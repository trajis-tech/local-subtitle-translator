@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0\.."

REM ============================================================
REM Verify PyTorch CUDA installation
REM ============================================================

echo.
echo ============================================================
echo  PyTorch CUDA Verify
echo ============================================================
echo.

set "PY="
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if "!PY!"=="" ( py -3 -c "pass" 2>nul && set "PY=py -3" )
if "!PY!"=="" ( py -c "pass" 2>nul && set "PY=py" )
if "!PY!"=="" set "PY=python"

echo [INFO] Python: !PY!
"!PY!" --version
echo.

echo [INFO] PyTorch diagnostics:
echo.

"!PY!" -c "
import sys
print('Python:', sys.executable)
print()

try:
    import torch
    print('torch version:', torch.__version__)
    print('torch.version.cuda:', torch.version.cuda)
    print('torch.cuda.is_available():', torch.cuda.is_available())
    print('torch.backends.cudnn.enabled:', torch.backends.cudnn.enabled if hasattr(torch.backends, 'cudnn') else 'N/A')
    print()
    if torch.cuda.is_available():
        print('CUDA device count:', torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
        print()
        print('GPU memory:')
        for i in range(torch.cuda.device_count()):
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f'  GPU {i}: {total:.1f} GB')
    else:
        print('No CUDA devices available')
        if torch.version.cuda is None:
            print()
            print('>>> This is a CPU-only PyTorch build!')
            print('>>> Run scripts\\install_torch_cuda_windows.bat to install CUDA version.')
except ImportError as e:
    print(f'torch import failed: {e}')
    print('Run: scripts\\install_torch_cuda_windows.bat or pip install torch --index-url https://download.pytorch.org/whl/cu128')

print()

try:
    import transformers
    print('transformers version:', getattr(transformers, '__version__', 'unknown'))
except ImportError as e:
    print(f'transformers import failed: {e}')
"

echo.
echo ============================================================
echo  Done
echo ============================================================
echo.
echo Press any key to close...
pause >nul
exit /b 0
