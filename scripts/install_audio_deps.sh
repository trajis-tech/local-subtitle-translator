#!/usr/bin/env bash
# =========================================================
# Install audio model dependencies (HF Wav2Vec2 speech emotion recognition).
# Counterpart of scripts/install_audio_deps.bat (Windows).
# Run from project root, or after: source .venv/bin/activate
# =========================================================
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

PY="python3"
if [ -x ".venv/bin/python" ]; then
  PY=".venv/bin/python"
fi

echo "[INFO] Installing audio model dependencies (HF Wav2Vec2 Run A)..."
if ! ( "$PY" -m pip install --upgrade pip \
       && "$PY" -m pip install -r requirements_base.txt ); then
  echo "[ERROR] pip install failed. Try: pip install -r requirements_base.txt"
  exit 1
fi

echo "[INFO] Audio dependencies installed. Ensure ffmpeg is in PATH for Run A."

