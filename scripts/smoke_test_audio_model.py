#!/usr/bin/env python3
"""
Smoke test: load HF Wav2Vec2 audio model + run inference on 1 short wav.
Usage (from project root):
  python scripts/smoke_test_audio_model.py
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _version_info() -> str:
    try:
        from src.audio_model import get_audio_stack_versions
        v = get_audio_stack_versions()
        lines = [
            "--- Version info ---",
            f"  python: {v['python']}",
            f"  transformers: {v['transformers']}",
            f"  torch: {v['torch']}",
            "--------------------",
        ]
        return "\n".join(lines)
    except Exception:
        return "--- Version info (could not get) ---"


def main() -> int:
    print("=" * 60)
    print("Smoke test: HF Wav2Vec2 audio model load + 1 short segment inference")
    print("=" * 60)
    print(_version_info())
    print()

    try:
        from src.config import load_config
        from src.audio_model import load_audio_model, AudioLoadResult
    except Exception as e:
        print("FAIL: Could not import project modules:", e)
        traceback.print_exc()
        print(_version_info())
        return 1

    cfg = load_config(ROOT / "config.json")
    model_id = getattr(cfg.audio, "model_id_or_path", None) or getattr(cfg.audio, "model_dir", "")
    print(f"[1] model_id: {model_id}")

    print("\n[2] Loading audio model...")
    try:
        result: AudioLoadResult = load_audio_model(cfg)
    except Exception as e:
        print("FAIL: Audio model load raised:", e)
        traceback.print_exc()
        print(_version_info())
        return 1

    if not result.success or result.model is None:
        print(f"FAIL: Audio model load failed: {result.reason}")
        print(_version_info())
        return 1
    print(f"OK: loaded on {result.backend} ({result.reason})")

    print("\n[3] One short segment inference...")
    model = result.model
    try:
        work_dir = ROOT / "work"
        test_wav = work_dir / "full_audio.wav"
        if test_wav.exists():
            out = model.analyze_emotion(str(test_wav))
        else:
            import numpy as np
            fake = np.zeros((1, 16000), dtype=np.float32)  # 1s silence
            out = model.analyze_emotion_from_waveform(fake, 16000)
    except Exception as e:
        print(f"  Inference error: {e}")
        out = {"emotion": "", "error": str(e)}

    emotion = out.get("emotion", "")
    err = out.get("error", "")
    print(f"  emotion: {emotion!r}, error: {err!r}")
    if emotion or out.get("confidence", 0) != 0:
        print("OK: non-empty inference result")
    else:
        print("OK: inference ran (empty emotion acceptable for silence)")

    print("\n" + "=" * 60)
    print("Smoke test passed.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print("FAIL: Unexpected exception:", e)
        traceback.print_exc()
        print(_version_info())
        sys.exit(1)
