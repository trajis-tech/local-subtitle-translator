"""Create/print a best-effort model plan based on detected hardware.

This script **does not download any model files**.
It only writes `config.json` (if missing) and prints what to download.

Rationale: automated downloads from model hubs can fail (auth / rate-limit / 404 / file renames).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Hardware detection (best-effort)
# ---------------------------------------------------------------------------

def _run(cmd: list[str]) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception:
        return ""


def detect_vram_gb() -> float:
    # NVIDIA (preferred)
    out = _run([
        "nvidia-smi",
        "--query-gpu=memory.total",
        "--format=csv,noheader,nounits",
    ])
    if out:
        # If multiple GPUs, take the max
        vals = []
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                mib = float(line)
                vals.append(mib / 1024.0)
            except Exception:
                continue
        if vals:
            return max(vals)

    # AMD/Intel: no reliable cli in base Windows; return 0
    return 0.0


def detect_ram_gb() -> float:
    # Try PowerShell (works on modern Windows)
    ps = _run([
        "powershell",
        "-NoProfile",
        "-Command",
        "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory",
    ])
    if ps:
        try:
            b = float(ps)
            return b / (1024.0 ** 3)
        except Exception:
            pass

    # Fallback: wmic (deprecated on some Windows builds)
    w = _run([
        "wmic",
        "computersystem",
        "get",
        "TotalPhysicalMemory",
        "/value",
    ])
    if w:
        m = re.search(r"TotalPhysicalMemory=(\d+)", w)
        if m:
            try:
                b = float(m.group(1))
                return b / (1024.0 ** 3)
            except Exception:
                pass

    return 0.0


# ---------------------------------------------------------------------------
# Model plan (filenames in ./models)
# ---------------------------------------------------------------------------

QWEN_FILES = {
    "q4_k_m": "qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf",
    "q5_k_m": "qwen2.5-14b-instruct-q5_k_m-00001-of-00003.gguf",
    "q6_k": "qwen2.5-14b-instruct-q6_k-00001-of-00004.gguf",
    "q8_0": "qwen2.5-14b-instruct-q8_0-00001-of-00004.gguf",
}

BREEZE_FILES = {
    "Q4_K_M": "Breeze-7B-Instruct-v1_0.Q4_K_M.gguf",
    "Q5_K_M": "Breeze-7B-Instruct-v1_0.Q5_K_M.gguf",
    "Q6_K": "Breeze-7B-Instruct-v1_0.Q6_K.gguf",
    "Q8_0": "Breeze-7B-Instruct-v1_0.Q8_0.gguf",
}

VISION_FILES = {
    "text": "moondream2-text-model-f16.gguf",
    "mmproj": "moondream2-mmproj-f16.gguf",
}


def pick_quant(vram_gb: float, ram_gb: float) -> tuple[str, str]:
    """Return (qwen_quant_key, breeze_quant_key)."""

    # We run models sequentially, but VRAM still matters due to GPU offload.
    # These are conservative picks that aim for 'good quality without leaving huge slack'.
    if vram_gb >= 24:
        return "q8_0", "Q8_0"
    if vram_gb >= 16:
        # 14B Q6_K can be a bit heavier; prefer Q5_K_M unless RAM is abundant.
        if ram_gb >= 48:
            return "q6_k", "Q6_K"
        return "q5_k_m", "Q5_K_M"
    if vram_gb >= 12:
        return "q5_k_m", "Q5_K_M"
    if vram_gb >= 8:
        return "q4_k_m", "Q4_K_M"

    # CPU-only
    return "q4_k_m", "Q4_K_M"


def build_default_config(models_dir: str, qwen_file: str, breeze_file: str) -> dict:
    """
    Build a default config.json that matches the **directory-based layout**
    used by the new pipeline:
      - ./models/main   → reasoning model (Stage 2)
      - ./models/local  → localization model (Stage 3)
      - ./models/vision → vision models (text + mmproj)

    We still keep qwen_model / breeze_model / vision.text_model for legacy
    compatibility, but the app will primarily look at the *_dir fields.
    """
    return {
        "models_dir": models_dir,
        "qwen_model": qwen_file,
        "breeze_model": breeze_file,
        "qwen_ctx": 8192,
        "breeze_ctx": 8192,
        "qwen_threads": 8,
        "breeze_threads": 8,
        # If you don't have an NVIDIA GPU or you want pure CPU inference, set these to 0.
        "qwen_gpu_layers": 60,
        "breeze_gpu_layers": 60,
        # New directory-based layout (matches src.config.AppConfig defaults)
        "reason_dir": str(Path(models_dir) / "main"),
        "translate_dir": str(Path(models_dir) / "local"),
        "vision_text_dir": str(Path(models_dir) / "vision"),
        "vision_mmproj_dir": str(Path(models_dir) / "vision"),
        "vision": {
            "enabled": False,
            "text_model": VISION_FILES["text"],
            "mmproj_model": VISION_FILES["mmproj"],
            "max_frames_per_sub": 1,
            "frame_offsets": [0.5],
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-config-if-missing", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    config_path = root / "config.json"
    models_dir = root / "models"
    models_dir.mkdir(exist_ok=True)

    vram = detect_vram_gb()
    ram = detect_ram_gb()

    qwen_q, breeze_q = pick_quant(vram, ram)
    qwen_file = QWEN_FILES[qwen_q]
    breeze_file = BREEZE_FILES[breeze_q]

    print(f"[HW] OS={os.name} RAM={ram:.2f} GB VRAM={vram:.2f} GB")
    print(f"[PLAN] Stage2 (Reason): Qwen2.5-14B {qwen_q} -> {qwen_file}")
    print(f"[PLAN] Stage3 (Translate): Breeze-7B {breeze_q} -> {breeze_file}")
    print(f"[PLAN] Stage1 (Optional Vision): {VISION_FILES['text']} + {VISION_FILES['mmproj']}")

    if args.write_config_if_missing:
        if config_path.exists() and not args.force:
            print(f"[INFO] config.json already exists: {config_path} (keep)")
        else:
            cfg = build_default_config("models", qwen_file, breeze_file)
            if vram <= 0:
                cfg["qwen_gpu_layers"] = 0
                cfg["breeze_gpu_layers"] = 0
            config_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"[OK] Wrote config: {config_path}")

    # Print what user must do next (directory-based layout)
    print("\n[NEXT] Download the model files manually and place them into:")
    print(f"       {models_dir / 'main'}   (reasoning model, e.g. {qwen_file} + shards if any)")
    print(f"       {models_dir / 'local'}  (localization model, e.g. {breeze_file})")
    print(f"       {models_dir / 'vision'} (optional vision model: {VISION_FILES['text']} + {VISION_FILES['mmproj']})")
    print("       (See README.md -> Model download)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
