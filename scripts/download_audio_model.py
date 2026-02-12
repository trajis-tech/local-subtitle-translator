"""Download Run A audio model (Wav2Vec2) to models/audio (~1.27 GB).

Called by start.bat so the model is available before first translation.
Uses huggingface_hub.snapshot_download; requires network.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    try:
        from src.config import load_config
    except Exception as e:
        print(f"[WARN] Could not load config: {e}")
        # Use defaults
        model_id = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
        cache_dir = "models/audio"
    else:
        cfg = load_config(ROOT / "config.json")
        model_id = (
            getattr(cfg.audio, "model_id_or_path", None)
            or getattr(cfg.audio, "model_dir", None)
            or "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
        )
        cache_dir = getattr(cfg.audio, "cache_dir", "models/audio") or "models/audio"

    local_dir = ROOT / cache_dir if not Path(cache_dir).is_absolute() else Path(cache_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    # Already present if config.json exists (snapshot_download puts files in local_dir)
    if (local_dir / "config.json").exists():
        print(f"[INFO] Run A audio model already present: {local_dir}")
        return 0

    print(f"[INFO] Downloading Run A audio model to {local_dir} (~1.27 GB)...")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[WARN] huggingface_hub not found. Install with: pip install -r requirements_base.txt")
        return 1

    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )
        print(f"[INFO] Run A audio model downloaded to {local_dir}")
        return 0
    except Exception as e:
        print(f"[WARN] Audio model download failed: {e}")
        print("        Run A will download on first translation (requires network).")
        return 1


if __name__ == "__main__":
    sys.exit(main())
