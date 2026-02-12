#!/usr/bin/env bash
# =========================================================
#  Local Subtitle Translator - START (offline launch only)
#  Run this to launch the app. No downloads, no network.
#  Run install.sh first (once, with network) if not yet installed.
# =========================================================
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"

VENV_DIR="${ROOT}/.venv"
RUNTIME_DIR="${ROOT}/runtime"

# ---- Isolate caches & temp inside this folder (same as install) ----
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export PYTHONNOUSERSITE=1
APP_CACHE="${RUNTIME_DIR}/cache"
export HF_HOME="${APP_CACHE}/hf"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_DISABLE_TELEMETRY=1
export PIP_CACHE_DIR="${APP_CACHE}/pip"
export TEMP="${RUNTIME_DIR}/temp"
export TMP="${RUNTIME_DIR}/temp"
export GRADIO_TEMP_DIR="${RUNTIME_DIR}/temp/gradio"

# ------------------------------
# 1) Require .venv (run install.sh first)
# ------------------------------
if [ ! -x "${VENV_DIR}/bin/python" ]; then
  echo "[ERROR] .venv not found. Run install.sh first (one-time, with network)."
  echo "        Then use start.sh for offline launch."
  exit 1
fi

source "${VENV_DIR}/bin/activate"

# ------------------------------
# 2) Check local model files (offline check; exit if missing)
# ------------------------------
echo "[INFO] Checking local model files..."
if ! python scripts/check_models.py; then
  echo
  echo "[ACTION REQUIRED] Download the models manually, then put them into:"
  echo "  ${ROOT}/models"
  echo
  echo "See README.md for download links. Then run start.sh again."
  exit 1
fi

# ------------------------------
# 3) Ensure model_prompts.csv has UTF-8 BOM (local file only)
# ------------------------------
python scripts/ensure_csv_bom.py

# ------------------------------
# 4) Launch app (offline)
# ------------------------------
echo "[INFO] Launching app..."
python app.py
