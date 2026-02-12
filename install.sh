#!/usr/bin/env bash
# =========================================================
#  Local Subtitle Translator - INSTALL (online / one-time)
#  Run this ONCE with network: downloads and installs everything.
#  After this, use start.sh for OFFLINE launch only.
#
#  Install does:
#  - Create .venv (system python3 + venv)
#  - Install deps: base + audio (HF Wav2Vec2/torch/transformers) + llama-cpp-python
#  - Optional: CUDA PyTorch if NVIDIA GPU present
#  - Download Run A audio model to models/audio
#  - Create config if missing; ensure model_prompts.csv BOM
#  - FFmpeg: Linux/macOS checks PATH only (see FFMPEG_INSTALL.md if missing)
#  - (Manual) You still download GGUF models into ./models
# =========================================================
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"

# ---- Isolate caches & temp inside this folder ----
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export PYTHONNOUSERSITE=1
RUNTIME_DIR="${ROOT}/runtime"
APP_CACHE="${RUNTIME_DIR}/cache"
export HF_HOME="${APP_CACHE}/hf"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_DISABLE_TELEMETRY=1
export PIP_CACHE_DIR="${APP_CACHE}/pip"
export TEMP="${RUNTIME_DIR}/temp"
export TMP="${RUNTIME_DIR}/temp"
export GRADIO_TEMP_DIR="${RUNTIME_DIR}/temp/gradio"
mkdir -p "${RUNTIME_DIR}" "${APP_CACHE}" "${TEMP}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[ERROR] python3 not found. Install Python 3.10+."
  exit 1
fi

# ------------------------------
# 1) Create venv if missing
# ------------------------------
if [ ! -d ".venv" ]; then
  echo "[INFO] Creating virtual environment..."
  python3 -m venv .venv
fi

source .venv/bin/activate

# ------------------------------
# 2) Upgrade pip inside .venv
# ------------------------------
echo "[INFO] Upgrading pip inside .venv..."
python -m pip install -U pip setuptools wheel -q

# ------------------------------
# 3) Install base requirements (Run A: HF Wav2Vec2 emotion model)
# ------------------------------
echo "[INFO] Installing Python requirements (base + audio Run A + video)..."
python -m pip install -r requirements_base.txt
if ! python -c "import torch; import transformers; import soundfile; import scipy; print('[INFO] Run A (audio) deps OK')" 2>/dev/null; then
  echo "[WARN] Run A audio deps import check failed. Run A may fail. Retry: python -m pip install -r requirements_base.txt"
fi

# ------------------------------
# 3a) Ensure CUDA PyTorch if NVIDIA GPU present
# ------------------------------
if ! python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[INFO] NVIDIA GPU detected but PyTorch is CPU-only. Installing CUDA 12.4 torch..."
    python -m pip uninstall -y torch 2>/dev/null || true
    if python -m pip install torch --index-url https://download.pytorch.org/whl/cu124 -q 2>/dev/null; then
      echo "[INFO] CUDA PyTorch installed. Verifying..."
      python -c "import torch; print('cuda_available=', torch.cuda.is_available(), 'gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
    else
      echo "[WARN] CUDA torch install failed. Run A will use CPU. You can retry later: pip install torch --index-url https://download.pytorch.org/whl/cu124"
    fi
  fi
fi

# ------------------------------
# 4) FFmpeg: warn if missing (Linux/macOS: no auto-download)
# ------------------------------
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "[WARN] ffmpeg not found in PATH. Run A (audio) and video features need ffmpeg."
  echo "       See FFMPEG_INSTALL.md for manual install."
fi

# ------------------------------
# 4a) Download Run A audio model to models/audio (~1.27 GB)
# ------------------------------
echo "[INFO] Downloading Run A audio model to models/audio (~1.27 GB)..."
if ! python scripts/download_audio_model.py; then
  echo "[WARN] Audio model download failed or skipped. Run A will download on first translation if needed."
fi

# ------------------------------
# 5) Install llama-cpp-python (GPU if available, else CPU wheel)
# ------------------------------
echo "[INFO] Installing llama-cpp-python (GPU if available)..."
if ! python scripts/install_llama_cpp.py; then
  echo "[ERROR] llama-cpp-python install failed."
  echo "        You can try CPU-only: python -m pip install --only-binary=:all: llama-cpp-python"
  exit 1
fi

# ------------------------------
# 6) Create recommended config (if missing)
# ------------------------------
echo "[INFO] Creating recommended config (if missing)..."
python scripts/plan_models.py --write-config-if-missing || true

# ------------------------------
# 7) Check local model files (warn only so install completes)
# ------------------------------
echo "[INFO] Checking local model files..."
if ! python scripts/check_models.py; then
  echo
  echo "[WARN] GGUF models not yet in ${ROOT}/models"
  echo "       Download them manually (see README), then run ./start.sh to launch."
  echo
fi

# ------------------------------
# 8) Ensure model_prompts.csv has UTF-8 BOM
# ------------------------------
echo "[INFO] Ensuring model_prompts.csv has UTF-8 BOM..."
python scripts/ensure_csv_bom.py

echo
echo "========================================================="
echo " Installation complete."
echo " Run ./start.sh to launch the app (offline)."
echo "========================================================="
echo
