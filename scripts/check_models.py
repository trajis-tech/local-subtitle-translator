from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    """
    Check local model files using the **same resolution logic as the main app**.

    This script now respects the directory-based layout:
      - ./models/main   → reasoning model (Stage 2)
      - ./models/local  → localization model (Stage 3)
      - ./models/vision → vision text + mmproj models

    It delegates all path resolution to `src.config` + `src.model_path_utils`
    so that CLI checks and `app.py` stay perfectly in sync.
    """
    root = Path(__file__).resolve().parents[1]
    cfg_path = root / "config.json"
    if not cfg_path.exists():
        print("[CHECK] config.json missing.")
        return 1

    # Ensure src/ is importable
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    try:
        from src.config import load_config
        from src.model_path_utils import (
            resolve_reason_model_path,
            resolve_translate_model_path,
            resolve_vision_paths,
        )
    except Exception as e:
        print(f"[CHECK] Failed to import config/model_path_utils: {e}")
        return 1

    cfg = load_config(cfg_path)
    missing: list[str] = []

    # Reasoning model (Stage 2)
    try:
        reason_p = resolve_reason_model_path(cfg)
        if not reason_p.exists():
            missing.append(f"Reason model missing: {reason_p}")
    except Exception as e:
        missing.append(f"Reason model not found ({e})")

    # Localization model (Stage 3)
    try:
        translate_p = resolve_translate_model_path(cfg)
        if not translate_p.exists():
            missing.append(f"Translate model missing: {translate_p}")
    except Exception as e:
        missing.append(f"Translate model not found ({e})")

    # Vision models (optional)
    if getattr(cfg, "vision", None) and getattr(cfg.vision, "enabled", False):
        try:
            text_p, mmproj_p, _ = resolve_vision_paths(cfg)
            if text_p is None or not text_p.exists():
                missing.append(
                    f"Vision text model missing. Searched in: "
                    f"{getattr(cfg, 'vision_text_dir', None) or cfg.models_dir}"
                )
            if mmproj_p is None or not mmproj_p.exists():
                missing.append(
                    f"Vision mmproj model missing. Searched in: "
                    f"{getattr(cfg, 'vision_mmproj_dir', None) or cfg.models_dir}"
                )
        except Exception as e:
            missing.append(f"Vision models not found ({e})")

    if missing:
        print("[CHECK] Missing:", *missing, sep="\n - ")
        return 1

    print("[CHECK] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
