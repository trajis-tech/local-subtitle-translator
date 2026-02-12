"""
共享的模型路徑解析工具，避免循環導入。

提供與 app.py 中相同的路徑解析邏輯，供 pipeline_runs.py 使用。
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Any
import re


def pick_gguf_from_dir(dir_path: Path) -> Optional[Path]:
    """
    Smart GGUF picker for a directory.
    
    Rules:
      - If there is exactly one .gguf file, use it.
      - If there are multiple .gguf files, assume they are shards of the SAME model.
        In that case:
          * Prefer the shard file matching '*-00001-of-*.gguf'
          * Otherwise, fall back to the largest .gguf file.
    
    Args:
        dir_path: 目錄路徑
    
    Returns:
        GGUF 檔案路徑，如果找不到則返回 None
    """
    if not dir_path.exists() or not dir_path.is_dir():
        return None

    ggufs = sorted(dir_path.glob("*.gguf"))
    if not ggufs:
        return None
    if len(ggufs) == 1:
        return ggufs[0]

    # Prefer the first-shard pattern: *-00001-of-00003.gguf
    shard1 = [
        p for p in ggufs
        if re.search(r"-00001-of-\d+\.gguf$", p.name)
    ]
    if shard1:
        return shard1[0]

    # Fallback: pick the largest file (often the main model file)
    return max(ggufs, key=lambda p: p.stat().st_size)


def ensure_gguf_shards_present(first_shard_path: Path) -> None:
    """
    若路徑為多分片 GGUF 的第一片（*-00001-of-N.gguf），檢查同目錄下其餘分片是否存在。
    若缺少任一分片會導致載入時 GGML_ASSERT(tensor buffer not set) 崩潰，故在此提前拋出明確錯誤。
    """
    name = first_shard_path.name
    m = re.search(r"-(00001)-of-(\d+)\.gguf$", name, re.IGNORECASE)
    if not m:
        return
    total = int(m.group(2))
    if total <= 1:
        return
    stem = name[: m.start(0)]  # 例如 qwen2.5-14b-instruct-q5_k_m
    dir_path = first_shard_path.parent
    missing = []
    for i in range(2, total + 1):
        shard_name = f"{stem}-{i:05d}-of-{total:05d}.gguf"
        if not (dir_path / shard_name).exists():
            missing.append(shard_name)
    if missing:
        raise FileNotFoundError(
            f"多分片 GGUF 模型缺少分片檔。請將「所有」分片放在同一目錄：\n"
            f"  目錄: {dir_path}\n"
            f"  已選第一片: {first_shard_path.name}\n"
            f"  缺少: {', '.join(missing)}\n"
            f"請下載並放置完整分片後再啟動。"
        )


def resolve_reason_model_path(cfg: Any) -> Path:
    """
    Resolve the GGUF path for the 'reason' model (Stage 2).
    
    Priority:
      1) If cfg.reason_dir is set and has GGUFs, pick from there.
      2) Otherwise, fall back to models_dir + qwen_model (legacy).
    
    Args:
        cfg: AppConfig 實例
    
    Returns:
        模型檔案路徑
    
    Raises:
        FileNotFoundError: 如果找不到模型檔案
    """
    from pathlib import Path as PathLib
    
    # New directory-based layout
    if hasattr(cfg, 'reason_dir') and cfg.reason_dir:
        d = PathLib(cfg.reason_dir)
        p = pick_gguf_from_dir(d)
        if p is not None:
            ensure_gguf_shards_present(p)
            return p

    # Legacy filename-based layout
    if hasattr(cfg, 'qwen_model') and cfg.qwen_model:
        legacy_path = PathLib(cfg.models_dir) / cfg.qwen_model
        if legacy_path.exists():
            ensure_gguf_shards_present(legacy_path)
            return legacy_path

    # Fallback: try models_dir directly
    models_dir = PathLib(cfg.models_dir)
    p = pick_gguf_from_dir(models_dir)
    if p is not None:
        ensure_gguf_shards_present(p)
        return p

    raise FileNotFoundError(
        f"Reason model not found. Searched in: "
        f"{cfg.reason_dir if hasattr(cfg, 'reason_dir') and cfg.reason_dir else cfg.models_dir}"
    )


def resolve_translate_model_path(cfg: Any) -> Path:
    """
    Resolve the GGUF path for the 'translate' model (Stage 3).
    
    Priority:
      1) If cfg.translate_dir is set and has GGUFs, pick from there.
      2) Otherwise, fall back to models_dir + breeze_model (legacy).
    
    Args:
        cfg: AppConfig 實例
    
    Returns:
        模型檔案路徑
    
    Raises:
        FileNotFoundError: 如果找不到模型檔案
    """
    from pathlib import Path as PathLib
    
    if hasattr(cfg, 'translate_dir') and cfg.translate_dir:
        d = PathLib(cfg.translate_dir)
        p = pick_gguf_from_dir(d)
        if p is not None:
            ensure_gguf_shards_present(p)
            return p

    # Legacy filename-based layout
    if hasattr(cfg, 'breeze_model') and cfg.breeze_model:
        legacy_path = PathLib(cfg.models_dir) / cfg.breeze_model
        if legacy_path.exists():
            ensure_gguf_shards_present(legacy_path)
            return legacy_path

    # Fallback: try models_dir directly
    models_dir = PathLib(cfg.models_dir)
    p = pick_gguf_from_dir(models_dir)
    if p is not None:
        ensure_gguf_shards_present(p)
        return p

    raise FileNotFoundError(
        f"Translate model not found. Searched in: "
        f"{cfg.translate_dir if hasattr(cfg, 'translate_dir') and cfg.translate_dir else cfg.models_dir}"
    )


def resolve_vision_paths(cfg: Any) -> tuple[Optional[Path], Optional[Path], Optional[str]]:
    """
    Resolve GGUF paths for vision text + mmproj models and detect model type.
    
    Directory-based layout:
      - cfg.vision_text_dir: directory containing vision text model GGUFs
      - cfg.vision_mmproj_dir: directory containing vision mmproj GGUFs
    
    Fallback:
      - Use models_dir + cfg.vision.text_model / cfg.vision.mmproj_model
      - Auto-detect model type based on filename patterns
      - Supports all llama-cpp-python vision models (Moondream2, LLaVA, BakLLaVA, etc.)
    
    Args:
        cfg: AppConfig 實例
    
    Returns:
        (text_model_path, mmproj_path, model_type)
        model_type: detected model type string or None (will be auto-detected by LocalVisionModel)
    """
    from pathlib import Path as PathLib
    
    text_p: Optional[Path] = None
    mmproj_p: Optional[Path] = None
    model_type: Optional[str] = None

    # New layout (directories)
    text_dir: Optional[Path] = PathLib(cfg.vision_text_dir).resolve() if getattr(cfg, "vision_text_dir", None) else None
    mmproj_dir: Optional[Path] = PathLib(cfg.vision_mmproj_dir).resolve() if getattr(cfg, "vision_mmproj_dir", None) else None

    if text_dir and text_dir.exists():
        if mmproj_dir and text_dir == mmproj_dir:
            # 同一目錄：必須依檔名區分主模型與 mmproj，否則會選到同一個檔（CLIP 會從主模型載入失敗）
            all_gguf = sorted(text_dir.glob("*.gguf"), key=lambda p: p.stat().st_size if p.exists() else 0, reverse=True)
            mmproj_candidates = [p for p in all_gguf if "mmproj" in p.name.lower()]
            text_candidates = [p for p in all_gguf if "mmproj" not in p.name.lower()]
            if mmproj_candidates and text_candidates:
                text_p = text_candidates[0]
                mmproj_p = mmproj_candidates[0]
            else:
                text_p = pick_gguf_from_dir(text_dir)
                mmproj_p = mmproj_candidates[0] if mmproj_candidates else None
        else:
            text_p = pick_gguf_from_dir(text_dir)
            if mmproj_dir and mmproj_dir.exists():
                mmproj_p = pick_gguf_from_dir(mmproj_dir)
            else:
                mmproj_p = None

    # Fallback to legacy filenames if needed
    if text_p is None and hasattr(cfg, 'vision') and hasattr(cfg.vision, 'text_model'):
        text_p = PathLib(cfg.models_dir) / cfg.vision.text_model
    if mmproj_p is None and hasattr(cfg, 'vision') and hasattr(cfg.vision, 'mmproj_model'):
        mmproj_p = PathLib(cfg.models_dir) / cfg.vision.mmproj_model

    # Auto-detect model type based on filename (optional, LocalVisionModel will also auto-detect)
    if text_p and text_p.exists():
        text_name_lower = text_p.name.lower()
        # 檢測常見的模型類型關鍵字
        if "llava" in text_name_lower:
            model_type = "llava"
        elif "moondream" in text_name_lower:
            model_type = "moondream"
        elif "bakllava" in text_name_lower:
            model_type = "bakllava"
        elif "minicpm" in text_name_lower:
            model_type = "minicpm-v"
        elif "qwen-vl" in text_name_lower or "qwenvl" in text_name_lower:
            model_type = "qwen-vl"
        elif "cogvlm" in text_name_lower:
            model_type = "cogvlm"
        elif "yi-vl" in text_name_lower or "yivl" in text_name_lower:
            model_type = "yi-vl"
    
    # If still not found, try globbing for any vision model files
    if (text_p is None or not text_p.exists()) or (mmproj_p is None or not mmproj_p.exists()):
        vision_dir = PathLib(
            getattr(cfg, "vision_text_dir", None) or cfg.models_dir
        )
        
        # 嘗試查找任何視覺模型檔案（按優先順序）
        # 1. LLaVA
        llava_text = sorted(
            vision_dir.glob("*llava*.gguf"),
            key=lambda p: p.stat().st_size if p.exists() else 0,
            reverse=True
        )
        llava_mmproj = sorted(
            vision_dir.glob("*mmproj*.gguf"),
            key=lambda p: p.stat().st_size if p.exists() else 0,
            reverse=True
        )
        
        if llava_text and llava_mmproj:
            text_p = llava_text[0]
            mmproj_p = llava_mmproj[0]
            model_type = "llava"
        else:
            # 2. Moondream2
            moondream_text = sorted(
                vision_dir.glob("moondream2-text-model*.gguf"),
                key=lambda p: p.stat().st_size if p.exists() else 0,
                reverse=True
            )
            moondream_mmproj = sorted(
                vision_dir.glob("moondream2-mmproj*.gguf"),
                key=lambda p: p.stat().st_size if p.exists() else 0,
                reverse=True
            )
            
            if moondream_text and moondream_mmproj:
                text_p = moondream_text[0]
                mmproj_p = moondream_mmproj[0]
                model_type = "moondream"
            else:
                # 3. 嘗試查找任何包含 mmproj 的檔案（通用檢測）
                all_mmproj = sorted(
                    vision_dir.glob("*mmproj*.gguf"),
                    key=lambda p: p.stat().st_size if p.exists() else 0,
                    reverse=True
                )
                all_text = sorted(
                    vision_dir.glob("*.gguf"),
                    key=lambda p: p.stat().st_size if p.exists() else 0,
                    reverse=True
                )
                
                if all_mmproj and all_text:
                    # 排除 mmproj 檔案，選擇主模型
                    text_candidates = [p for p in all_text if "mmproj" not in p.name.lower()]
                    if text_candidates:
                        text_p = text_candidates[0]
                        mmproj_p = all_mmproj[0]
                        # model_type 保持 None，讓 LocalVisionModel 自動檢測

    return text_p, mmproj_p, model_type
