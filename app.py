from __future__ import annotations

import asyncio
import csv
import json
import re
import shutil
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import gradio as gr
import pysrt

from src.config import load_config, AppConfig
from src.glossary import (
    GlossaryEntry, load_glossary, save_glossary,
    import_csv_two_cols, import_subtitleedit_multiple_replace_template
)
from src.srt_utils import save_srt, sub_midpoints_ms, clean_srt_text
from src.video import open_video, get_frame_at_ms, encode_jpg_bytes
from src.models import TextModel, VisionModel
from src.local_vision_model import LocalVisionModel
from src.audio_model import AudioModel
from src.subtitle_item import create_subtitle_items_from_srt, SubtitleItem
from src.pipeline_runs import (
    run_audio,
    run_brief_text,
    run_vision_single,
    run_vision_multi,
    run_context_expansion,
    run_final_translate,
)
from src.jsonl_compat import (
    load_audio_results_compat,
    load_brief_results_compat,
    load_vision_results_compat,
    load_final_translations_compat,
)
from src.pipeline import parse_pack_from_reasons

# Paths relative to app directory so they work regardless of CWD
_APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = _APP_DIR / "config.json"
PO_DIR = _APP_DIR / "po"
PROMPTS_CSV_PATH = _APP_DIR / "model_prompts.csv"
RUN_E_PROMPTS_CSV_PATH = _APP_DIR / "model_prompts_run_f.csv"  # Run F 翻譯用：main_group_translate, main_assemble, local_polish, localization
LANG_CONFIG_PATH = _APP_DIR / "language_config.json"

# Async batching configuration
BATCH_SIZE_STAGE2 = 32
BATCH_SIZE_STAGE3 = 32
MAX_CONCURRENT_BATCHES = 2  # Control GPU memory pressure


def _cache_pack_ok(loaded_items: dict, version: str, min_ratio: float = 0.90) -> bool:
    """
    Check if cached brief_{version} has PACK in reasons for enough items.
    Old cache may have reasons without PACK, causing pack=None in Run F.
    Returns True only if at least min_ratio (default 90%) of items with brief_vX have parseable PACK.
    """
    if not loaded_items or version not in ("v1", "v2", "v3"):
        return False
    attr = "brief_v1" if version == "v1" else "brief_v2" if version == "v2" else "brief_v3"
    total = 0
    ok = 0
    for item in loaded_items.values():
        brief = getattr(item, attr, None)
        if brief is None:
            continue
        total += 1
        if parse_pack_from_reasons(brief.reasons or "") is not None:
            ok += 1
    if total == 0:
        return False
    return (ok / total) >= min_ratio


def _copy_brief_snapshot(work_dir: Path, snapshot_version: str) -> None:
    """單一 brief：在階段更新前將當前 brief_work.jsonl 複製為 snapshot（brief_v1/v2/v3/v4.jsonl）。"""
    # 新版主檔名為 brief_work.jsonl；為相容舊版，若不存在則回退至 brief.jsonl。
    src = work_dir / "brief_work.jsonl"
    if not src.exists():
        src = work_dir / "brief.jsonl"
    dst = work_dir / f"brief_{snapshot_version}.jsonl"
    if src.exists():
        shutil.copy2(src, dst)


def _load_available_locales_from_csv(csv_path: Path) -> list[str]:
    """
    （保留給未來擴充用，目前不再從 CSV 讀語言碼）
    為了相容舊程式碼，現在永遠回傳空列表，實際語言設定改由 language_config.json 管理。
    """
    return []


@dataclass
class ModelPromptConfig:
    """從 CSV 載入的模型 prompt 設定"""
    model_name: str
    role: str  # "main", "localization", "vision"
    chat_format: str
    system_prompt_template: str
    user_prompt_template: str
    batch_user_prompt_template: str = ""  # optional; for role=main, batch brief (multi-item in one request)


def _load_prompt_from_csv_by_model_name(
    csv_path: Path,
    model_filename: str,
    role: str,  # "main", "main_assemble", "main_group_translate", "localization", "local_polish"
) -> ModelPromptConfig | None:
    """
    從 CSV 依 model_name 匹配載入 prompt 設定。
    
    匹配規則：只要 model_filename（不分大小寫）包含 CSV 中的 model_name（不分大小寫），
    就視為匹配成功。返回第一個匹配的 row。
    
    Args:
        csv_path: CSV 檔案路徑
        model_filename: 實際模型檔名（例如 "Breeze-7B-Instruct-v1_0.Q5_K_M.gguf"）
        role: 要匹配的角色（"main"、"main_assemble"、"main_group_translate" 綁主模型；"localization"、"local_polish" 綁在地化模型）
    
    Returns:
        ModelPromptConfig 或 None（找不到匹配時）
    """
    if not csv_path.exists():
        return None
    
    model_lower = model_filename.lower()
    
    try:
        with csv_path.open("r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                csv_role = (row.get("role") or "").strip()
                if csv_role != role:
                    continue
                
                csv_model_name = (row.get("model_name") or "").strip()
                if not csv_model_name:
                    continue
                
                # 匹配：model_filename（不分大小寫）包含 CSV 的 model_name（不分大小寫）
                if csv_model_name.lower() in model_lower:
                    return ModelPromptConfig(
                        model_name=csv_model_name,
                        role=csv_role,
                        chat_format=(row.get("chat_format") or "chatml").strip(),
                        system_prompt_template=(row.get("system_prompt_template") or "").strip(),
                        user_prompt_template=(row.get("user_prompt_template") or "").strip(),
                        batch_user_prompt_template=(row.get("batch_user_prompt_template") or "").strip(),
                    )
    except Exception:
        return None
    
    return None


def _load_run_e_prompt_from_csv(model_filename: str, role: str) -> ModelPromptConfig | None:
    """
    Run F 翻譯用：優先從 model_prompts_run_e.csv 載入 prompt；若無檔案或無匹配則回退到 model_prompts.csv。
    role 應為 main_group_translate, main_assemble, local_polish, localization 之一。
    """
    if RUN_E_PROMPTS_CSV_PATH.exists():
        cfg = _load_prompt_from_csv_by_model_name(RUN_E_PROMPTS_CSV_PATH, model_filename, role)
        if cfg is not None:
            return cfg
    return _load_prompt_from_csv_by_model_name(PROMPTS_CSV_PATH, model_filename, role)


def _load_language_prefs_from_csv(csv_path: Path) -> tuple[str, str, list[str]]:
    """
    從 model_prompts.csv 推斷語言偏好（舊邏輯，用於 fallback，現階段通常不會生效）：
      - 過去會讀取 role == \"localization\" 的 `target_language` 欄位決定語言；
      - 目前已移除該欄位，此函式多半回傳預設值，實際語言設定由 language_config.json 管理。
    """
    ui_locale = "zh-TW"
    target_locale = "zh-TW"
    available_locales: list[str] = []
    try:
        if not csv_path.exists():
            return ui_locale, target_locale, available_locales
        with csv_path.open("r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                role = (row.get("role") or "").strip()
                if role != "localization":
                    continue
                target = (row.get("target_language") or "").strip()
                if target:
                    if not available_locales:
                        ui_locale = target
                        target_locale = target
                    if target not in available_locales:
                        available_locales.append(target)
    except Exception:
        return ui_locale, target_locale, available_locales
    return ui_locale, target_locale, available_locales


def _load_language_prefs(lang_cfg_path: Path, csv_path: Path) -> tuple[str, str, list[str]]:
    """
    從獨立的語言設定檔載入：
      - ui_locale: UI 介面語言（對應 .po）
      - target_locale_default: 預設翻譯目標語言
      - available_target_locales: 目標語言下拉選單可選清單

    介面語言與目標語言以 language_config.json 為準；僅在 JSON 不存在或讀取失敗時，
    才用 CSV 推斷。這樣修改 language_config.json 的 ui_locale 即可切換介面語言。
    """
    # 先給一組合理的預設（繁體中文）
    ui_locale = "zh-TW"
    target_locale = "zh-TW"
    available_locales: list[str] = ["zh-TW", "zh-CN", "ja-JP", "es-ES"]
    json_loaded = False

    # 優先從 language_config.json 載入（介面語言與目標語言以此為準）
    try:
        if lang_cfg_path.exists():
            data = json.loads(lang_cfg_path.read_text(encoding="utf-8"))
            if data is not None:
                ui_locale = (data.get("ui_locale") or ui_locale).strip()
                target_locale = (data.get("target_locale_default") or target_locale).strip()
                cfg_locales = data.get("available_target_locales")
                if isinstance(cfg_locales, list) and cfg_locales:
                    available_locales = [str(x).strip() for x in cfg_locales if str(x).strip()]
                json_loaded = True
    except Exception:
        pass

    # 僅當未成功載入 JSON 時，才用 CSV 推斷（避免 CSV 覆寫 JSON 的介面語言）
    csv_ui, csv_target, csv_locales = _load_language_prefs_from_csv(csv_path)
    if not json_loaded:
        if csv_locales:
            available_locales = csv_locales
        if csv_ui:
            ui_locale = csv_ui
        if csv_target:
            target_locale = csv_target

    return ui_locale, target_locale, available_locales


_UI_LOCALE, _TARGET_LANG_LOCALE, _AVAILABLE_TARGET_LOCALES = _load_language_prefs(
    LANG_CONFIG_PATH, PROMPTS_CSV_PATH
)


def _parse_po_string(s: str) -> str:
    """Unescape .po string (strip outer quotes, handle \\ \\n \\")."""
    s = s.strip()
    if len(s) >= 2 and s.startswith('"') and s.endswith('"'):
        s = s[1:-1]
    return s.replace("\\n", "\n").replace("\\\"", '"').replace("\\\\", "\\")


def _load_po_catalog(po_dir: Path, locale: str) -> dict[str, str]:
    """
    Load translations from po/{locale}.po (flat .po files).
    gettext expects po/{locale}/LC_MESSAGES/messages.mo; we use flat .po instead
    so interface language follows language_config.json ui_locale without .mo build.
    Returns dict msgid -> msgstr. Empty msgid (header) is skipped.
    """
    catalog: dict[str, str] = {}
    po_path = po_dir / f"{locale}.po"
    if not po_path.exists():
        return catalog
    try:
        text = po_path.read_text(encoding="utf-8")
    except Exception:
        return catalog
    current_msgid: list[str] = []
    current_msgstr: list[str] = []
    in_msgid = False
    in_msgstr = False

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("msgid "):
            if in_msgstr and current_msgid:
                key = "".join(current_msgid)
                val = "".join(current_msgstr)
                if key != "":
                    catalog[key] = val
            current_msgid = [_parse_po_string(stripped[6:].strip())]
            current_msgstr = []
            in_msgid = True
            in_msgstr = False
        elif stripped.startswith("msgstr "):
            current_msgstr = [_parse_po_string(stripped[7:].strip())]
            in_msgid = False
            in_msgstr = True
        elif stripped.startswith('"') and (in_msgid or in_msgstr):
            if in_msgid:
                current_msgid.append(_parse_po_string(stripped))
            else:
                current_msgstr.append(_parse_po_string(stripped))
    if current_msgid and in_msgstr:
        key = "".join(current_msgid)
        val = "".join(current_msgstr)
        if key != "":
            catalog[key] = val
    return catalog


_PO_CATALOG = _load_po_catalog(PO_DIR, _UI_LOCALE)


def _(s: str) -> str:
    """Translate string using language_config.json ui_locale and po/{locale}.po."""
    if not s:
        return s
    return _PO_CATALOG.get(s, s)


def _as_path(x: Any) -> Optional[str]:
    """Best-effort to extract a local file path from Gradio inputs."""
    if x is None:
        return None
    if isinstance(x, str):
        return x
    # common Gradio objects
    if hasattr(x, "path") and isinstance(getattr(x, "path"), str):
        return getattr(x, "path")
    if hasattr(x, "name") and isinstance(getattr(x, "name"), str):
        return getattr(x, "name")
    if isinstance(x, dict):
        p = x.get("path") or x.get("name")
        if isinstance(p, str):
            return p
    try:
        return str(x)
    except Exception:
        return None


# -----------------------------
# Helpers
# -----------------------------
def _ensure_config() -> AppConfig:
    if not CONFIG_PATH.exists():
        # create minimal config with placeholders
        CONFIG_PATH.write_text(json.dumps({"models_dir": "./models"}, indent=2), encoding="utf-8")
    return load_config(CONFIG_PATH)


def _model_path(models_dir: str, filename: str) -> str:
    """Legacy helper: join models_dir + filename."""
    p = Path(models_dir) / filename
    return str(p.resolve())


def _pick_gguf_from_dir(dir_path: Path) -> Optional[Path]:
    """
    Smart GGUF picker for a directory.

    Rules:
      - If there is exactly one .gguf file, use it.
      - If there are multiple .gguf files, assume they are shards of the SAME model.
        In that case:
          * Prefer the shard file matching '*-00001-of-*.gguf'
          * Otherwise, fall back to the largest .gguf file.
    
    Note: 實際實作在 src.model_path_utils.pick_gguf_from_dir，這裡是相容性包裝。
    """
    from src.model_path_utils import pick_gguf_from_dir
    return pick_gguf_from_dir(dir_path)


def _resolve_reason_model_path(cfg: AppConfig) -> Path:
    """
    Resolve the GGUF path for the 'reason' model (Stage 2).

    Priority:
      1) If cfg.reason_dir is set and has GGUFs, pick from there.
      2) Otherwise, fall back to models_dir + qwen_model (legacy).
    
    Note: 實際實作在 src.model_path_utils.resolve_reason_model_path
    """
    from src.model_path_utils import resolve_reason_model_path
    return resolve_reason_model_path(cfg)


def _resolve_translate_model_path(cfg: AppConfig) -> Path:
    """
    Resolve the GGUF path for the 'translate' model (Stage 3).

    Priority:
      1) If cfg.translate_dir is set and has GGUFs, pick from there.
      2) Otherwise, fall back to models_dir + breeze_model (legacy).
    
    Note: 實際實作在 src.model_path_utils.resolve_translate_model_path
    """
    from src.model_path_utils import resolve_translate_model_path
    return resolve_translate_model_path(cfg)


def _check_vision_assets(cfg: AppConfig) -> tuple[bool, list[str]]:
    """
    檢查 vision 模型檔案是否存在。
    
    Args:
        cfg: AppConfig 實例
    
    Returns:
        (ok: bool, missing: list[str])
        - ok: True 表示所有檔案都存在
        - missing: 缺少的檔案路徑列表（用於錯誤提示）
    """
    text_p, mmproj_p, _ = _resolve_vision_paths(cfg)
    missing = []
    
    if text_p is None or not text_p.exists():
        expected_path = Path(cfg.models_dir) / cfg.vision.text_model
        if getattr(cfg, "vision_text_dir", None):
            expected_path = Path(cfg.vision_text_dir) / "*.gguf"
        missing.append(f"Vision text model: {expected_path}")
    
    if mmproj_p is None or not mmproj_p.exists():
        expected_path = Path(cfg.models_dir) / cfg.vision.mmproj_model
        if getattr(cfg, "vision_mmproj_dir", None):
            expected_path = Path(cfg.vision_mmproj_dir) / "*.gguf"
        missing.append(f"Vision mmproj model: {expected_path}")
    
    return len(missing) == 0, missing


def _resolve_vision_paths(cfg: AppConfig) -> tuple[Optional[Path], Optional[Path], Optional[str]]:
    """
    Resolve GGUF paths for vision text + mmproj models and detect model type.

    Directory-based layout:
      - cfg.vision_text_dir: directory containing vision text model GGUFs
      - cfg.vision_mmproj_dir: directory containing vision mmproj GGUFs

    Fallback:
      - Use models_dir + cfg.vision.text_model / cfg.vision.mmproj_model
      - Auto-detect model type based on filename patterns
      - Supports all llama-cpp-python vision models (Moondream2, LLaVA, BakLLaVA, etc.)

    Returns:
        (text_model_path, mmproj_path, model_type)
        model_type: detected model type string or None (will be auto-detected by LocalVisionModel)
    
    Note: 實際實作在 src.model_path_utils.resolve_vision_paths
    """
    from src.model_path_utils import resolve_vision_paths
    return resolve_vision_paths(cfg)


def _check_models(cfg: AppConfig):
    missing = []
    # Reason model
    reason_p = _resolve_reason_model_path(cfg)
    if not reason_p.exists():
        if cfg.reason_dir:
            missing.append(f"Reason model missing in directory: {cfg.reason_dir} (no *.gguf found)")
        else:
            missing.append(f"Reason model missing: {reason_p}")

    # Translate model
    translate_p = _resolve_translate_model_path(cfg)
    if not translate_p.exists():
        if cfg.translate_dir:
            missing.append(f"Translate model missing in directory: {cfg.translate_dir} (no *.gguf found)")
        else:
            missing.append(f"Translate model missing: {translate_p}")

    if cfg.vision.enabled:
        v_text, v_mmproj, v_type = _resolve_vision_paths(cfg)

        if not v_text.exists():
            missing.append(f"Vision text model missing: {v_text}")
        if not v_mmproj.exists():
            missing.append(f"Vision mmproj missing: {v_mmproj}")
    return missing


def _gradio_video_path(video_value) -> str | None:
    """
    Gradio Video input returns a filepath string by default.
    But we defensively handle dict-like or object-like values too.
    """
    if video_value is None:
        return None
    if isinstance(video_value, str):
        return video_value
    # Some gradio versions/components may return dict-like
    if isinstance(video_value, dict):
        return video_value.get("name") or video_value.get("path") or video_value.get("data")
    # Fallback: try .name
    return getattr(video_value, "name", None)


def _gradio_file_path(file_obj) -> str | None:
    """
    Gradio File(type="file") usually returns a temporary file object with `.name` path.
    """
    if file_obj is None:
        return None
    if isinstance(file_obj, str):
        return file_obj
    if isinstance(file_obj, dict):
        return file_obj.get("name") or file_obj.get("path")
    return getattr(file_obj, "name", None)


def _read_uploaded_file_bytes(file_obj) -> bytes:
    """
    Robustly read bytes from Gradio uploaded file object.
    """
    if file_obj is None:
        return b""
    # If it's already bytes:
    if isinstance(file_obj, (bytes, bytearray)):
        return bytes(file_obj)
    # If dict with path:
    if isinstance(file_obj, dict):
        p = file_obj.get("name") or file_obj.get("path")
        if p:
            return Path(p).read_bytes()
        return b""
    # file-like
    try:
        file_obj.seek(0)
    except Exception:
        pass
    try:
        return file_obj.read()
    except Exception:
        # last resort by path
        p = getattr(file_obj, "name", None)
        if p:
            return Path(p).read_bytes()
        return b""


def _save_preview_bytes_to_file(img_bytes: bytes, work_dir: Path) -> str:
    """Save preview image bytes to work_dir and return path string for Gradio Image component."""
    if not img_bytes:
        return None
    try:
        preview_path = work_dir / "preview_frame.jpg"
        preview_path.write_bytes(img_bytes)
        return str(preview_path)
    except Exception:
        return None


# -----------------------------
# Async Dynamic Batched Inference
# -----------------------------

@dataclass
class LineTask:
    """單行字幕的完整任務上下文，用於批次處理"""
    index: int  # 在 subs 中的索引
    sub: pysrt.SubRipItem  # SRT 項目
    line_en_raw: str  # 清理後的英文原文
    prev_ctx: list[str] = field(default_factory=list)  # Round 1 上下文（前一句）
    next_ctx: list[str] = field(default_factory=list)  # Round 1 上下文（後一句）
    more_prev: list[str] = field(default_factory=list)  # Round 2 擴展上下文（前4句）
    more_next: list[str] = field(default_factory=list)  # Round 2 擴展上下文（後4句）
    
    # Round 1 結果
    s2_round1: Optional[Stage2Result] = None
    
    # Round 2 結果（如果執行）
    s2_round2: Optional[Stage2Result] = None
    
    # Vision 結果（如果執行）
    visual_hint: Optional[str] = None
    s2_vision: Optional[Stage2Result] = None
    
    # 最終使用的結果
    final_s2: Optional[Stage2Result] = None
    
    # Stage3 結果
    zh_output: Optional[str] = None


async def run_stage2_batch(
    reason_model: TextModel,
    tasks: list[LineTask],
    semaphore: asyncio.Semaphore,
    use_more_context: bool = False,
    visual_hint_map: dict[int, str] | None = None,
    prompt_config: ModelPromptConfig | None = None,
) -> list[Stage2Result]:
    """
    批次執行 Stage2 推理。
    
    Args:
        reason_model: Stage2 模型
        tasks: 要處理的 LineTask 列表
        semaphore: 控制併發數
        use_more_context: 是否使用擴展上下文（Round 2）
        visual_hint_map: {index: visual_hint} 對應表（Round 4）
    
    Returns:
        對應順序的 Stage2Result 列表
    """
    async def process_one(task: LineTask) -> Stage2Result:
        async with semaphore:
            # 選擇上下文
            if use_more_context:
                prev_lines = task.more_prev
                next_lines = task.more_next
            else:
                prev_lines = task.prev_ctx
                next_lines = task.next_ctx
            
            # 選擇 visual_hint
            visual_hint = None
            if visual_hint_map and task.index in visual_hint_map:
                visual_hint = visual_hint_map[task.index]
            
            # 在執行緒池中執行同步的 stage2_reason_and_score
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                stage2_reason_and_score,
                reason_model,
                task.line_en_raw,
                prev_lines,
                next_lines,
                visual_hint,
                prompt_config,  # 傳入 CSV 載入的 prompt 設定
            )
            return result
    
    # 並發執行所有任務
    results = await asyncio.gather(*[process_one(task) for task in tasks])
    return list(results)


async def run_stage3_batch(
    translate_model: TextModel,
    tasks: list[LineTask],
    prev_zh_map: dict[int, list[str]],  # {index: [prev_zh_lines]}
    glossary: list[GlossaryEntry],
    target_language: str,
    semaphore: asyncio.Semaphore,
    prompt_config: ModelPromptConfig | None = None,
) -> list[str]:
    """
    批次執行 Stage3 翻譯。
    
    Args:
        translate_model: Stage3 模型
        tasks: 要處理的 LineTask 列表（必須已有 final_s2）
        prev_zh_map: {index: [prev_zh_lines]} 前文中文對應
        glossary: 詞彙表
        target_language: 目標語言標籤
        semaphore: 控制併發數
    
    Returns:
        對應順序的中文翻譯列表
    """
    async def process_one(task: LineTask) -> str:
        async with semaphore:
            if task.final_s2 is None:
                raise ValueError(f"Task {task.index} missing final_s2")
            
            prev_zh = prev_zh_map.get(task.index, [])
            
            # 在執行緒池中執行同步的 stage3_translate
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                stage3_translate,
                translate_model,
                task.line_en_raw,
                task.final_s2.meaning_en,
                prev_zh,
                glossary,
                target_language,
                prompt_config,  # 傳入 CSV 載入的 prompt 設定
            )
            return result
    
    # 並發執行所有任務
    results = await asyncio.gather(*[process_one(task) for task in tasks])
    return list(results)


# -----------------------------
# Model Manager (確保任一時間只載入一個模型)
# -----------------------------
class ModelManager:
    """管理模型載入/卸載，確保任一時間只存在一個模型實例"""
    
    def __init__(self, log_lines_ref: list[str]):
        """初始化，需要 log_lines 的引用"""
        self.current_model: Any | None = None
        self.current_model_type: str | None = None  # "audio", "reason", "vision", "translate"
        self.log_lines = log_lines_ref
    
    def _assert_no_model_loaded(self, operation: str):
        """內部檢查：確保沒有模型已載入（用於調試）"""
        if self.current_model is not None:
            self.log_lines.append(
                f"[ModelManager] ⚠️ 警告：在 {operation} 時發現已載入的模型 "
                f"({self.current_model_type})，將先卸載"
            )
            self.unload_all()
    
    def load_reason_model(self, cfg: AppConfig, prompt_config: Any | None) -> TextModel:
        """載入主推理模型（卸載其他模型），優先使用 GPU 加速"""
        # 嚴格檢查：確保沒有其他模型已載入
        self._assert_no_model_loaded("load_reason_model")
        self.unload_all()  # 雙重保險
        
        self.log_lines.append("[ModelManager] 🔄 Unloading all models, preparing to load main reasoning model...")
        reason_path = _resolve_reason_model_path(cfg)
        reason_chat_format = prompt_config.chat_format if prompt_config else "chatml"
        
        try:
            model = TextModel(
                model_path=str(reason_path),
                chat_format=reason_chat_format,
                n_ctx=cfg.llama_cpp.n_ctx_reason,
                n_gpu_layers=cfg.llama_cpp.n_gpu_layers_reason,  # 會被 TextModel 強制設為 -1（GPU）
                n_threads=cfg.llama_cpp.n_threads,
            )
            self.current_model = model
            self.current_model_type = "reason"
            self.log_lines.append("[ModelManager] ✓ Main reasoning model loaded successfully (GPU acceleration: enabled)")
            return model
        except Exception as e:
            # 如果載入失敗，確保狀態清理
            self.current_model = None
            self.current_model_type = None
            raise
    
    def load_translate_model(self, cfg: AppConfig, prompt_config: Any | None) -> TextModel:
        """載入翻譯模型（卸載其他模型），優先使用 GPU 加速"""
        # 嚴格檢查：確保沒有其他模型已載入
        self._assert_no_model_loaded("load_translate_model")
        self.unload_all()  # 雙重保險
        
        self.log_lines.append("[ModelManager] 🔄 Unloading all models, preparing to load translation model...")
        translate_path = _resolve_translate_model_path(cfg)
        translate_chat_format = prompt_config.chat_format if prompt_config else "chatml"
        
        try:
            model = TextModel(
                model_path=str(translate_path),
                chat_format=translate_chat_format,
                n_ctx=cfg.llama_cpp.n_ctx_translate,
                n_gpu_layers=cfg.llama_cpp.n_gpu_layers_translate,  # 會被 TextModel 強制設為 -1（GPU）
                n_threads=cfg.llama_cpp.n_threads,
            )
            self.current_model = model
            self.current_model_type = "translate"
            self.log_lines.append("[ModelManager] ✓ Translation model loaded successfully (GPU acceleration: enabled)")
            return model
        except Exception as e:
            # 如果載入失敗，確保狀態清理
            self.current_model = None
            self.current_model_type = None
            raise
    
    def load_vision_model(self, cfg: AppConfig) -> LocalVisionModel:
        """載入視覺模型（卸載其他模型），自動檢測模型類型（支援所有 llama-cpp-python 視覺模型），優先使用 GPU 加速"""
        # 嚴格檢查：確保沒有其他模型已載入
        self._assert_no_model_loaded("load_vision_model")
        self.unload_all()  # 雙重保險
        
        self.log_lines.append("[ModelManager] 🔄 Unloading all models, preparing to load vision model...")
        vision_text_p, vision_mmproj_p, model_type = _resolve_vision_paths(cfg)
        
        if not vision_text_p or not vision_text_p.exists():
            raise FileNotFoundError(f"Vision text model not found. Searched in: {cfg.models_dir}")
        if not vision_mmproj_p or not vision_mmproj_p.exists():
            raise FileNotFoundError(f"Vision mmproj model not found. Searched in: {cfg.models_dir}")
        
        try:
            # LocalVisionModel 會自動檢測模型類型並選擇合適的 ChatHandler
            # 這裡可以選擇性地傳入 model_type 以加速檢測，或讓它自動檢測
            # LocalVisionModel 內部會設置 n_gpu_layers=-1 以優先使用 GPU
            model = LocalVisionModel(
                model_path=str(vision_text_p.resolve()),
                clip_model_path=str(vision_mmproj_p.resolve()),
                model_type=model_type,  # 如果為 None，會自動檢測
                n_ctx=None,  # 讓 LocalVisionModel 根據模型類型自動設定
                n_threads=cfg.llama_cpp.n_threads,
            )
            self.current_model = model
            self.current_model_type = "vision"
            detected_type = getattr(model, 'model_type', 'auto')
            self.log_lines.append(f"[ModelManager] ✓ Vision model loaded successfully (type: {detected_type}, GPU acceleration: enabled)")
            return model
        except Exception as e:
            # 如果載入失敗，確保狀態清理
            self.current_model = None
            self.current_model_type = None
            raise
    
    def load_audio_model(self, cfg: AppConfig) -> AudioModel:
        """載入音訊模型（卸載其他模型）"""
        # 嚴格檢查：確保沒有其他模型已載入
        self._assert_no_model_loaded("load_audio_model")
        self.unload_all()  # 雙重保險
        
        self.log_lines.append("[ModelManager] 🔄 Unloading all models, preparing to load audio model...")
        audio_model_dir = Path(cfg.audio.model_dir)
        if not audio_model_dir.exists():
            raise FileNotFoundError(f"Audio model directory not found: {audio_model_dir}")
        
        try:
            model = AudioModel(audio_model_dir)
            self.current_model = model
            self.current_model_type = "audio"
            self.log_lines.append("[ModelManager] ✓ Audio model loaded successfully")
            return model
        except Exception as e:
            # 如果載入失敗，確保狀態清理
            self.current_model = None
            self.current_model_type = None
            raise
    
    def unload_all(self):
        """卸載當前模型（強制釋放資源）"""
        if self.current_model is not None:
            model_type_str = self.current_model_type or "unknown"
            self.log_lines.append(f"[ModelManager] 🔄 Unloading {model_type_str} model...")
            
            try:
                # 對於 llama-cpp-python 模型，嘗試手動釋放
                if hasattr(self.current_model, 'llm'):
                    # TextModel 或 LocalVisionModel
                    llm = self.current_model.llm
                    if hasattr(llm, 'free'):
                        try:
                            llm.free()  # 如果 llama-cpp-python 提供 free 方法
                        except Exception:
                            pass
                
                # 音訊模型（HF pipeline）隨 del 釋放，無需額外清理

                # 強制刪除引用
                del self.current_model
                self.current_model = None
                self.current_model_type = None
                
                # 強制垃圾回收（可選，但可能有助於立即釋放 VRAM）
                import gc
                gc.collect()
                
                self.log_lines.append(f"[ModelManager] ✓ {model_type_str} model unloaded (resources released)")
            except Exception as e:
                self.log_lines.append(f"[ModelManager] ⚠️ Error occurred while unloading model (but references cleared): {e}")
                # 即使出錯，也要清除引用
                self.current_model = None
                self.current_model_type = None


# log_lines 將在 translate_ui 中作為局部變數初始化


# -----------------------------
# Main Translate UI (New Pipeline: Run A→E)
# -----------------------------
def translate_ui(
    video_file,
    srt_file,
    srt_encoding,
    enable_vision,
    enable_context_expansion,
    max_frames,
    offsets_csv,
    run_mode: str = "all",  # "all", "A", "B", "C", "D", "E", "F"
    run_e_scheme: str = "full",  # "full" | "main_led" | "local_led" | "draft_first"
    progress=gr.Progress(),
):
    """
    New Pipeline: Run A→B→(C/D 可選)→E 上下文擴充→F 翻譯 with SubtitleItem and sub_id alignment.
    Run A: Audio; B: Brief (brief.jsonl); C/D: vision 更新 brief；E: 上下文擴充；F: 最終翻譯。
    """
    global log_lines
    log_lines = []  # 重置 log
    
    # Guard EVERYTHING so Gradio doesn't just show a red "Error" without details.
    try:
        cfg = _ensure_config()
        cfg.vision.enabled = bool(enable_vision)
        cfg.pipeline.enable_context_expansion = bool(enable_context_expansion)
        cfg.vision.max_frames_per_sub = int(max_frames)
        cfg.pipeline.run_e_scheme = str(run_e_scheme or "full").strip() or "full"

        # 目標語言（Locale Code），完全由 language_config.json 控制
        target_locale = _TARGET_LANG_LOCALE.strip() if _TARGET_LANG_LOCALE else "zh-TW"

        # Parse frame offsets (comma-separated 0..1).
        try:
            offsets = [float(x.strip()) for x in str(offsets_csv).split(",") if x.strip()]
            if offsets:
                cfg.vision.frame_offsets = offsets
        except Exception:
            pass
    except Exception as e:
        tb = traceback.format_exc()
        errmsg = _("Error: {error}").format(error=str(e))
        yield None, None, tb[-12000:], None, errmsg, (tb.strip().split("\n")[-1].strip() if tb.strip() else errmsg)
        return

    try:
        # Validate uploads
        srt_path = _gradio_file_path(srt_file)
        video_path = _gradio_video_path(video_file)

        if not srt_path:
            msg = _("Please upload an SRT file.")
            yield None, None, "", None, msg, msg
            return
        
        # Video is required for Run A (audio) and Run C/D (vision)
        if not video_path:
            msg = _("Video is required for audio and vision analysis. Upload a video (MKV/MP4).")
            yield None, None, "", None, msg, msg
            return

        # Validate models exist
        missing = _check_models(cfg)
        if missing:
            msg = (
                _("Missing model files. Please follow the README to download GGUF models into ./models\n\n")
                + "\n".join(missing)
            )
            msg2 = _("Missing models.")
            yield None, None, msg, None, msg2, msg2
            return

        # Show initial status so user看到程式有開始動作
        log_lines.append("[Init] Loading configuration and validating inputs...")
        progress(0.01, desc=_("Initializing..."))
        yield None, None, "\n".join(log_lines), None, _("Validating inputs..."), (log_lines[-1] if log_lines else "")

        # Load glossary
        glossary = load_glossary(cfg.glossary.json_path)

        # Load SRT
        subs = pysrt.open(srt_path, encoding=srt_encoding)
        total = len(subs)
        if total == 0:
            msg = _("SRT has 0 lines.")
            yield None, None, "", None, msg, msg
            return

        # 創建 SubtitleItem 字典（使用 sub_id 對齊）
        items = create_subtitle_items_from_srt(subs)
        log_lines.append(f"[Init] Created {len(items)} subtitle items with sub_id alignment")
        progress(0.03, desc=_("Parsing subtitles..."))
        yield None, None, "\n".join(log_lines), None, _("Parsing subtitles... ({total} lines)").format(total=total), (log_lines[-1] if log_lines else "")

        # Work directory
        work_dir = Path(cfg.pipeline.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        log_lines.append(f"[Init] Work directory: {work_dir}")

        # Copy video to work_dir so Run A/C/D read a stable path (Gradio temp can cause "Permission denied" in background thread/ffmpeg)
        try:
            import shutil
            src_video = Path(video_path)
            ext = src_video.suffix or ".mkv"
            stable_video_path = work_dir / f"source_video{ext}"
            if str(stable_video_path) != str(Path(video_path).resolve()):
                log_lines.append(f"[Init] Copying video to work dir for stable access: {stable_video_path.name}")
                shutil.copy2(video_path, stable_video_path)
                video_path = str(stable_video_path)
        except Exception as e:
            log_lines.append(f"[Init] ⚠️ Could not copy video to work dir: {e}; using original path (may cause Permission denied in Run A)")

        # Output path（依 target_locale 命名，避免寫死 zh-TW）
        safe_locale = target_locale.replace("/", "_").replace("\\", "_")
        out_path = Path(tempfile.gettempdir()) / f"{Path(srt_path).stem}.translated.{safe_locale}.srt"
        preview = None
        preview_holder = [None]  # 供 Run C/D 寫入最後擷取的影格，供 UI 預覽

        # 資源檢測和動態調整（優先檢測 GPU）
        log_lines.append("[Init] Detecting system resources (GPU/CPU/RAM)...")
        progress(0.05, desc=_("Detecting resources..."))
        yield None, None, "\n".join(log_lines), None, _("Detecting GPU and system resources..."), (log_lines[-1] if log_lines else "")
        
        from src.resource_utils import (
            get_resource_info,
            detect_gpu,
            calculate_batch_size,
            calculate_parallel_workers,
        )
        
        log_lines.append("[Init] Detecting GPU...")
        gpu_info = detect_gpu()
        log_lines.append("[Init] Detecting CPU and memory...")
        resource_info = get_resource_info()
        
        # 記錄資源資訊（優先顯示 GPU）
        if gpu_info["available"]:
            gpu_msg = f"[GPU] ✓ Available - {gpu_info['device_name'] or 'NVIDIA GPU'}"
            if gpu_info["vram_mb"]:
                gpu_msg += f" ({gpu_info['vram_mb']} MB VRAM)"
            log_lines.append(gpu_msg)
            if gpu_info["driver_version"]:
                log_lines.append(f"[GPU] Driver version: {gpu_info['driver_version']}")
        else:
            log_lines.append("[GPU] ✗ Not available - Using CPU mode")
        
        cpu_msg = f"[CPU] Cores: {resource_info['cpu_count']}"
        if resource_info['available_memory_mb']:
            cpu_msg += f", Available RAM: {resource_info['available_memory_mb']} MB"
        log_lines.append(cpu_msg)
        
        # 計算批次大小（如果未在配置中指定）
        # 優先使用 GPU VRAM，如果 GPU 不可用則使用 CPU + RAM
        # 激進模式：使用更大的批次大小以提高 GPU 利用率
        if cfg.pipeline.batch_size is None:
            # 根據 GPU 可用性和總項目數動態調整基礎批次大小
            if gpu_info["available"]:
                # GPU 模式：使用更大的批次（充分利用 VRAM）
                if total >= 1000:
                    base_batch_size = 128  # 大檔案：大批次
                elif total >= 500:
                    base_batch_size = 64   # 中檔案：中批次
                else:
                    base_batch_size = 48   # 小檔案：中批次
            else:
                # CPU 模式：使用較小的批次（避免 OOM）
                base_batch_size = 32
            
            batch_size = calculate_batch_size(
                total_items=total,
                available_memory_mb=resource_info['available_memory_mb'],
                cpu_count=resource_info['cpu_count'],
                gpu_info=gpu_info,
                base_batch_size=base_batch_size,
                min_batch_size=cfg.pipeline.min_batch_size,
                max_batch_size=cfg.pipeline.max_batch_size,
            )
            acceleration_mode = "GPU" if gpu_info["available"] else "CPU"
            log_lines.append(f"[Optimization] Auto-detected batch size: {batch_size} (mode: {acceleration_mode}, base: {base_batch_size})")
        else:
            batch_size = cfg.pipeline.batch_size
            acceleration_mode = "GPU" if gpu_info["available"] else "CPU"
            log_lines.append(f"[Optimization] Using configured batch size: {batch_size} (mode: {acceleration_mode})")
        
        # 計算並行工作數（如果未在配置中指定）
        if cfg.pipeline.max_workers is None:
            max_workers = calculate_parallel_workers(
                cpu_count=resource_info['cpu_count'],
                max_workers=None,
            )
            log_lines.append(f"[Optimization] Auto-detected max workers: {max_workers}")
        else:
            max_workers = cfg.pipeline.max_workers
        
        # 記錄 GPU 加速狀態
        if gpu_info["available"]:
            log_lines.append(f"[GPU] Models will be loaded with GPU acceleration (n_gpu_layers=-1)")
        else:
            log_lines.append(f"[GPU] GPU not available - Models will use CPU (slower but still functional)")
        
        progress(0.08, desc=_("Loading prompt configurations..."))
        yield None, None, "\n".join(log_lines), None, _("Loading prompt configurations..."), (log_lines[-1] if log_lines else "")

        # 載入 prompt 設定
        log_lines.append("[Init] Loading prompt configurations from CSV...")
        reason_path = _resolve_reason_model_path(cfg)
        translate_path = _resolve_translate_model_path(cfg)
        log_lines.append(f"[Init] Reason model: {reason_path.name}")
        log_lines.append(f"[Init] Translate model: {translate_path.name}")
        reason_prompt_config = _load_prompt_from_csv_by_model_name(
            PROMPTS_CSV_PATH, reason_path.name, role="main"
        )
        # Run F 相關 prompt 優先從 model_prompts_run_e.csv 載入
        reason_assemble_prompt_config = _load_run_e_prompt_from_csv(reason_path.name, "main_assemble")
        translate_prompt_config = _load_run_e_prompt_from_csv(translate_path.name, "localization")
        reason_group_translate_prompt_config = _load_run_e_prompt_from_csv(reason_path.name, "main_group_translate")
        translate_polish_prompt_config = _load_run_e_prompt_from_csv(translate_path.name, "local_polish")
        if RUN_E_PROMPTS_CSV_PATH.exists():
            log_lines.append("[Init] Run F prompts from model_prompts_run_e.csv (fallback: model_prompts.csv)")
        log_lines.append("[Init] Prompt configurations loaded")
        
        progress(0.10, desc=_("Preparing audio analysis (Run A)..."))
        yield None, None, "\n".join(log_lines), None, _("Preparing audio analysis (Run A)..."), (log_lines[-1] if log_lines else "")

        # Progress weights: Audio 20% / Brief 35% / Vision 25% / Translate 20%
        total_progress = 100.0
        progress_audio = 20.0
        progress_brief = 35.0
        progress_vision = 25.0
        progress_expansion = 5.0   # Run E 上下文擴充
        progress_translate = 15.0  # Run F 最終翻譯
        current_progress = 0.0

        def update_progress(step_progress: float, desc: str):
            """更新進度條"""
            nonlocal current_progress
            current_progress = step_progress
            progress(current_progress / total_progress, desc=desc)

        # ========== Run A: Audio Analysis ==========
        if run_mode in ("all", "A"):
            log_lines.append("[Run A] ====== Starting audio analysis ======")
            yield None, None, "\n".join(log_lines), preview, _("Run A: Audio analysis..."), (log_lines[-1] if log_lines else "")
            update_progress(0.0, _("Run A: Audio analysis"))
            
            # 嘗試載入已存在的結果（相容舊格式）
            try:
                log_lines.append("[Run A] Checking for existing audio analysis results...")
                loaded_items = load_audio_results_compat(work_dir, items)
                if loaded_items and len(loaded_items) == len(items):
                    log_lines.append(f"[Run A] ✓ Loaded existing audio analysis results: {len(loaded_items)} items")
                    items = loaded_items
                    yield None, None, "\n".join(log_lines), preview, _("Run A: Loaded existing results ({count} items)").format(count=len(items)), (log_lines[-1] if log_lines else "")
                else:
                    log_lines.append("[Run A] No existing results found, starting analysis...")
                    log_lines.append(f"[Run A] Expected to process {len(items)} subtitle items, this may take some time (4K video)...")
                    log_lines.append("[Run A] Note: Each item requires audio segment extraction and analysis, please wait patiently...")
                    yield None, None, "\n".join(log_lines), preview, _("Run A: Starting audio analysis..."), (log_lines[-1] if log_lines else "")
                    
                    # 在後台執行 run_audio，並定期 yield 以更新 UI
                    # 使用執行緒執行 run_audio
                    result_container = {"items": None, "done": False, "error": None}
                    
                    def run_audio_thread():
                        try:
                            result_container["items"] = run_audio(
                                items,
                                str(video_path),
                                work_dir,
                                cfg,
                                log_lines=log_lines,
                                progress_callback=lambda p, d: update_progress(
                                    (p * progress_audio) / total_progress, d
                                ),
                            )
                            result_container["done"] = True
                        except Exception as e:
                            result_container["error"] = e
                            result_container["done"] = True
                    
                    thread = threading.Thread(target=run_audio_thread, daemon=True)
                    thread.start()
                    
                    # 定期檢查進度並 yield UI 更新（同步版）
                    while not result_container["done"]:
                        time.sleep(1.0)  # 每 1 秒檢查一次
                        yield None, None, "\n".join(log_lines), preview, _("Run A: Processing... (check log for detailed progress)"), (log_lines[-1] if log_lines else "")
                    
                    # 等待執行緒完成
                    thread.join(timeout=1.0)
                    
                    if result_container["error"]:
                        raise result_container["error"]
                    
                    items = result_container["items"]
                    
                    # 執行完成後立即 yield 一次以顯示最終狀態
                    yield None, None, "\n".join(log_lines), preview, _("Run A: Completed ({count} items)").format(count=len(items)), (log_lines[-1] if log_lines else "")
            except Exception as e:
                log_lines.append(f"[Run A] Error loading existing results: {e}, re-analyzing...")
                yield None, None, "\n".join(log_lines), preview, _("Run A: Error occurred, re-analyzing..."), (log_lines[-1] if log_lines else "")
                
                # 同樣使用執行緒執行
                result_container = {"items": None, "done": False, "error": None}
                
                def run_audio_thread():
                    try:
                        result_container["items"] = run_audio(
                            items,
                            str(video_path),
                            work_dir,
                            cfg,
                            log_lines=log_lines,
                            progress_callback=lambda p, d: update_progress(
                                (p * progress_audio) / total_progress, d
                            ),
                        )
                        result_container["done"] = True
                    except Exception as e:
                        result_container["error"] = e
                        result_container["done"] = True
                
                thread = threading.Thread(target=run_audio_thread, daemon=True)
                thread.start()
                
                # 定期檢查進度並 yield UI 更新（同步版）
                while not result_container["done"]:
                    time.sleep(1.0)
                    yield None, None, "\n".join(log_lines), preview, _("Run A: Processing... (check log for detailed progress)"), (log_lines[-1] if log_lines else "")
                
                thread.join(timeout=1.0)
                
                if result_container["error"]:
                    raise result_container["error"]
                
                items = result_container["items"]
            
            update_progress(progress_audio, _("Run A: Completed"))
            yield None, None, "\n".join(log_lines), preview, _("Run A: Completed ({count} items)").format(count=len(items)), (log_lines[-1] if log_lines else "")
        
        # ========== Run B: Brief Generation v1 ==========
        if run_mode in ("all", "B"):
            log_lines.append("[Run B] Preparing to load main reasoning model (Stage 2) and generate brief_v1...")
            yield None, None, "\n".join(log_lines), preview, _("Run B: Loading main reasoning model (Stage 2)..."), (log_lines[-1] if log_lines else "")
            update_progress(progress_audio, _("Run B: Loading main reasoning model (Stage 2)"))
            
            # 嘗試載入已存在的結果（相容舊格式）
            try:
                loaded_items = load_brief_results_compat(work_dir, "v1", items)
                if loaded_items and len(loaded_items) == len(items) and _cache_pack_ok(loaded_items, "v1"):
                    log_lines.append(f"[Run B] Loaded existing brief v1: {len(loaded_items)} items")
                    items = loaded_items
                else:
                    if loaded_items and len(loaded_items) == len(items) and not _cache_pack_ok(loaded_items, "v1"):
                        log_lines.append("[Run B] Cache missing PACK, forcing regenerate")
                    else:
                        log_lines.append("[Run B] No existing brief v1 or count mismatch, loading main model and generating...")
                    yield None, None, "\n".join(log_lines), preview, _("Run B: Loading main reasoning model (Stage 2)..."), (log_lines[-1] if log_lines else "")
                    result_container = {"items": None, "done": False, "error": None}
                    def run_brief_b():
                        try:
                            result_container["items"] = run_brief_text(
                                items,
                                work_dir,
                                cfg,
                                reason_prompt_config,
                                version="v1",
                                vision_hint_map=None,
                                log_lines=log_lines,
                                progress_callback=lambda p, d: update_progress(
                                    (progress_audio + p * progress_brief) / total_progress, d
                                ),
                                batch_size=batch_size if batch_size < total else None,
                                target_language=target_locale,
                            )
                        except Exception as e:
                            result_container["error"] = e
                        result_container["done"] = True
                    thread = threading.Thread(target=run_brief_b, daemon=True)
                    thread.start()
                    while not result_container["done"]:
                        time.sleep(1.0)
                        yield None, None, "\n".join(log_lines), preview, _("Run B: Loading model / processing... (see log)"), (log_lines[-1] if log_lines else "")
                    thread.join(timeout=1.0)
                    if result_container["error"]:
                        raise result_container["error"]
                    items = result_container["items"]
                    yield None, None, "\n".join(log_lines), preview, _("Run B: Model loaded, processing with parallel batch inference..."), (log_lines[-1] if log_lines else "")
            except Exception as e:
                log_lines.append(f"[Run B] Error loading existing results: {e}, reloading main model and regenerating...")
                yield None, None, "\n".join(log_lines), preview, _("Run B: Reloading main reasoning model (Stage 2)..."), (log_lines[-1] if log_lines else "")
                result_container = {"items": None, "done": False, "error": None}
                def run_brief_b():
                    try:
                        result_container["items"] = run_brief_text(
                            items,
                            work_dir,
                            cfg,
                            reason_prompt_config,
                            version="v1",
                            vision_hint_map=None,
                            log_lines=log_lines,
                            progress_callback=lambda p, d: update_progress(
                                (progress_audio + p * progress_brief) / total_progress, d
                            ),
                            batch_size=batch_size if batch_size < total else None,
                            target_language=target_locale,
                        )
                    except Exception as ex:
                        result_container["error"] = ex
                    result_container["done"] = True
                thread = threading.Thread(target=run_brief_b, daemon=True)
                thread.start()
                while not result_container["done"]:
                    time.sleep(1.0)
                    yield None, None, "\n".join(log_lines), preview, _("Run B: Reloading / processing... (see log)"), (log_lines[-1] if log_lines else "")
                thread.join(timeout=1.0)
                if result_container["error"]:
                    raise result_container["error"]
                items = result_container["items"]
                yield None, None, "\n".join(log_lines), preview, _("Run B: Model reloaded, processing with parallel batch inference..."), (log_lines[-1] if log_lines else "")
            
            # Debug log：顯示前 3 句的 audio_hint（用於驗收）
            sorted_items = sorted(items.items(), key=lambda x: x[1].start_ms)
            for debug_idx in range(min(3, len(sorted_items))):
                sub_id, item = sorted_items[debug_idx]
                audio_hint = item.get_audio_hint()
                log_lines.append(f"[Debug] Item {debug_idx+1} (sub_id={sub_id[:8]}...) audio_hint: {audio_hint}")
            
            update_progress(progress_audio + progress_brief, _("Run B: Completed"))
            yield None, None, "\n".join(log_lines), preview, _("Run B: Completed ({count} items)").format(count=len(items)), (log_lines[-1] if log_lines else "")
        
        # ========== Run C: Single-frame Vision Fallback ==========
        # 檢查 vision 模型檔案是否存在
        if cfg.vision.enabled:
            vision_ok, vision_missing = _check_vision_assets(cfg)
            if not vision_ok:
                # 自動停用 vision（僅 runtime，不修改 config.json）
                log_lines.append(f"[Vision] ⚠️ Vision disabled: missing files:")
                for missing_file in vision_missing:
                    log_lines.append(f"[Vision]   - {missing_file}")
                log_lines.append(f"[Vision] Please place vision model files in: {cfg.models_dir}")
                cfg.vision.enabled = False  # Runtime disable
        
        if run_mode in ("all", "C") and cfg.vision.enabled:
            # 確保有 brief_v1（C-only 時從 brief.jsonl / brief_v1.jsonl 載入）
            loaded_brief = load_brief_results_compat(work_dir, "v1", items)
            if loaded_brief and len(loaded_brief) == len(items):
                items = loaded_brief
            # 找出需要 vision 的 sub_id（brief_v1.need_vision === true）
            sorted_items = sorted(items.items(), key=lambda x: x[1].start_ms)
            target_sub_ids = [
                sub_id for sub_id, item in sorted_items
                if item.brief_v1 and item.brief_v1.need_vision is True
            ]

            if target_sub_ids:
                log_lines.append(f"[Run C] Preparing to load vision model (Stage 3, 1-frame) and analyze {len(target_sub_ids)} items...")
                yield None, None, "\n".join(log_lines), preview, _("Run C: Loading vision model (1-frame) for {count} items...").format(count=len(target_sub_ids)), (log_lines[-1] if log_lines else "")
                update_progress(progress_audio + progress_brief, _("Run C: Loading vision model (1-frame)"))
                
                # 嘗試載入已存在的視覺結果（相容舊格式）
                try:
                    loaded_items = load_vision_results_compat(work_dir, items, single_frame=True)
                    if loaded_items and len(loaded_items) == len(items):
                        log_lines.append(f"[Run C] Loaded existing vision (1-frame) results: {len(loaded_items)} items")
                        items = loaded_items
                    else:
                        log_lines.append("[Run C] No existing vision (1-frame) results or count mismatch, loading vision model...")
                        yield None, None, "\n".join(log_lines), preview, _("Run C: Loading vision model (1-frame)..."), (log_lines[-1] if log_lines else "")
                        items = run_vision_single(
                            items,
                            str(video_path),
                            work_dir,
                            cfg,
                            target_sub_ids,
                            log_lines=log_lines,
                            progress_callback=lambda p, d: update_progress(
                                (progress_audio + progress_brief + p * (progress_vision * 0.5)) / total_progress, d
                            ),
                            preview_callback=lambda b: preview_holder.__setitem__(0, _save_preview_bytes_to_file(b, work_dir)),
                        )
                        if preview_holder[0] is not None:
                            preview = preview_holder[0]
                            if isinstance(preview, bytes):
                                preview = _save_preview_bytes_to_file(preview, work_dir)
                        yield None, None, "\n".join(log_lines), preview, _("Run C: Vision model loaded, analyzing frames..."), (log_lines[-1] if log_lines else "")
                except Exception as e:
                    log_lines.append(f"[Run C] Error loading existing results: {e}, reloading vision model...")
                    yield None, None, "\n".join(log_lines), preview, _("Run C: Reloading vision model (1-frame)..."), (log_lines[-1] if log_lines else "")
                    items = run_vision_single(
                        items,
                        str(video_path),
                        work_dir,
                        cfg,
                        target_sub_ids,
                        log_lines=log_lines,
                        progress_callback=lambda p, d: update_progress(
                            (progress_audio + progress_brief + p * (progress_vision * 0.5)) / total_progress, d
                        ),
                        preview_callback=lambda b: preview_holder.__setitem__(0, _save_preview_bytes_to_file(b, work_dir)),
                    )
                    if preview_holder[0] is not None:
                        preview = preview_holder[0]
                        if isinstance(preview, bytes):
                            preview = _save_preview_bytes_to_file(preview, work_dir)
                    yield None, None, "\n".join(log_lines), preview, _("Run C: Vision model reloaded, analyzing frames..."), (log_lines[-1] if log_lines else "")
                
                # 重新生成 brief v2（帶視覺提示）；更新前先留 snapshot
                _copy_brief_snapshot(work_dir, "v1")
                yield None, None, "\n".join(log_lines), preview, _("Run C: Regenerating brief v2 with vision..."), (log_lines[-1] if log_lines else "")

                # 建立 vision_hint_map
                vision_hint_map = {
                    sub_id: item.vision_desc_1
                    for sub_id, item in items.items()
                    if item.vision_desc_1
                }
                
                # 嘗試載入已存在的 brief_v2
                try:
                    loaded_items = load_brief_results_compat(work_dir, "v2", items)
                    if loaded_items and len(loaded_items) == len(items) and _cache_pack_ok(loaded_items, "v2"):
                        log_lines.append(f"[Run C] Loaded existing brief v2: {len(loaded_items)} items")
                        items = loaded_items
                    else:
                        if loaded_items and len(loaded_items) == len(items) and not _cache_pack_ok(loaded_items, "v2"):
                            log_lines.append("[Run C] Cache missing PACK, forcing regenerate")
                        else:
                            log_lines.append("[Run C] No existing brief v2 or count mismatch, reloading main model and regenerating...")
                        yield None, None, "\n".join(log_lines), preview, _("Run C: Reloading main reasoning model for brief v2..."), (log_lines[-1] if log_lines else "")
                        result_container = {"items": None, "done": False, "error": None}
                        def run_brief_c2():
                            try:
                                result_container["items"] = run_brief_text(
                                    items,
                                    work_dir,
                                    cfg,
                                    reason_prompt_config,
                                    version="v2",
                                    vision_hint_map=vision_hint_map,
                                    log_lines=log_lines,
                                    progress_callback=lambda p, d: update_progress(
                                        (progress_audio + progress_brief + p * (progress_vision * 0.5)) / total_progress, d
                                    ),
                                    batch_size=batch_size if batch_size < total else None,
                                    target_language=target_locale,
                                )
                            except Exception as ex:
                                result_container["error"] = ex
                            result_container["done"] = True
                        thread = threading.Thread(target=run_brief_c2, daemon=True)
                        thread.start()
                        while not result_container["done"]:
                            time.sleep(1.0)
                            yield None, None, "\n".join(log_lines), preview, _("Run C: Loading model / brief v2... (see log)"), (log_lines[-1] if log_lines else "")
                        thread.join(timeout=1.0)
                        if result_container["error"]:
                            raise result_container["error"]
                        items = result_container["items"]
                        yield None, None, "\n".join(log_lines), preview, _("Run C: Main model reloaded, regenerating brief v2 with parallel batch inference..."), (log_lines[-1] if log_lines else "")
                except Exception as e:
                    log_lines.append(f"[Run C] Error loading existing brief v2: {e}, reloading main model and regenerating...")
                    yield None, None, "\n".join(log_lines), preview, _("Run C: Reloading main reasoning model for brief v2..."), (log_lines[-1] if log_lines else "")
                    result_container = {"items": None, "done": False, "error": None}
                    def run_brief_c2():
                        try:
                            result_container["items"] = run_brief_text(
                                items,
                                work_dir,
                                cfg,
                                reason_prompt_config,
                                version="v2",
                                vision_hint_map=vision_hint_map,
                                log_lines=log_lines,
                                progress_callback=lambda p, d: update_progress(
                                    (progress_audio + progress_brief + p * (progress_vision * 0.5)) / total_progress, d
                                ),
                                batch_size=batch_size if batch_size < total else None,
                                target_language=target_locale,
                            )
                        except Exception as ex:
                            result_container["error"] = ex
                        result_container["done"] = True
                    thread = threading.Thread(target=run_brief_c2, daemon=True)
                    thread.start()
                    while not result_container["done"]:
                        time.sleep(1.0)
                        yield None, None, "\n".join(log_lines), preview, _("Run C: Reloading / brief v2... (see log)"), (log_lines[-1] if log_lines else "")
                    thread.join(timeout=1.0)
                    if result_container["error"]:
                        raise result_container["error"]
                    items = result_container["items"]
                    yield None, None, "\n".join(log_lines), preview, _("Run C: Main model reloaded, regenerating brief v2 with parallel batch inference..."), (log_lines[-1] if log_lines else "")
            else:
                log_lines.append("[Run C] No items need vision (1-frame) fallback")
            
            update_progress(progress_audio + progress_brief + progress_vision * 0.5, _("Run C: Completed"))
            yield None, None, "\n".join(log_lines), preview, _("Run C: Completed ({count} items)").format(count=len(items)), (log_lines[-1] if log_lines else "")
            # 無論是否執行 Run D，Run C 結束後都建立 brief_v2 快照（代表「Run C 後的 brief 狀態」）
            _copy_brief_snapshot(work_dir, "v2")
        
        # ========== Run D: Multi-frame Vision Fallback ==========
        # 檢查 vision 模型檔案是否存在（如果之前未檢查過）
        if cfg.vision.enabled:
            vision_ok, vision_missing = _check_vision_assets(cfg)
            if not vision_ok:
                # 自動停用 vision（僅 runtime，不修改 config.json）
                if "[Vision] ⚠️ Vision disabled" not in "\n".join(log_lines[-50:]):  # 避免重複提示
                    log_lines.append(f"[Vision] ⚠️ Vision disabled: missing files:")
                    for missing_file in vision_missing:
                        log_lines.append(f"[Vision]   - {missing_file}")
                    log_lines.append(f"[Vision] Please place vision model files in: {cfg.models_dir}")
                cfg.vision.enabled = False  # Runtime disable
        
        if run_mode in ("all", "D") and cfg.vision.enabled:
            # 找出需要多張影像的 sub_id（brief_v2.need_multi_frame_vision === true，若無則用 brief_v1）
            sorted_items = sorted(items.items(), key=lambda x: x[1].start_ms)
            target_sub_ids = []
            for sub_id, item in sorted_items:
                brief_to_check = item.brief_v2 if item.brief_v2 else item.brief_v1
                if brief_to_check and brief_to_check.need_multi_frame_vision is True:
                    target_sub_ids.append(sub_id)
            
            if target_sub_ids:
                log_lines.append(f"[Run D] Preparing to load vision model (Stage 3, multi-frame) and analyze {len(target_sub_ids)} items...")
                yield None, None, "\n".join(log_lines), preview, _("Run D: Loading vision model (multi-frame) for {count} items...").format(count=len(target_sub_ids)), (log_lines[-1] if log_lines else "")
                update_progress(progress_audio + progress_brief + progress_vision * 0.5, _("Run D: Loading vision model (multi-frame)"))
                
                # 嘗試載入已存在的視覺結果（相容舊格式）
                try:
                    loaded_items = load_vision_results_compat(work_dir, items, single_frame=False)
                    if loaded_items and len(loaded_items) == len(items):
                        log_lines.append(f"[Run D] Loaded existing vision (multi-frame) results: {len(loaded_items)} items")
                        items = loaded_items
                    else:
                        log_lines.append("[Run D] No existing vision (multi-frame) results or count mismatch, loading vision model...")
                        yield None, None, "\n".join(log_lines), preview, _("Run D: Loading vision model (multi-frame)..."), (log_lines[-1] if log_lines else "")
                        items = run_vision_multi(
                            items,
                            str(video_path),
                            work_dir,
                            cfg,
                            target_sub_ids,
                            cfg.vision.max_frames_per_sub,
                            log_lines=log_lines,
                            progress_callback=lambda p, d: update_progress(
                                (progress_audio + progress_brief + progress_vision * 0.5 + p * (progress_vision * 0.5)) / total_progress, d
                            ),
                            preview_callback=lambda b: preview_holder.__setitem__(0, _save_preview_bytes_to_file(b, work_dir)),
                        )
                        if preview_holder[0] is not None:
                            preview = preview_holder[0]
                            if isinstance(preview, bytes):
                                preview = _save_preview_bytes_to_file(preview, work_dir)
                        yield None, None, "\n".join(log_lines), preview, _("Run D: Vision model loaded, analyzing multiple frames..."), (log_lines[-1] if log_lines else "")
                except Exception as e:
                    log_lines.append(f"[Run D] Error loading existing results: {e}, reloading vision model...")
                    yield None, None, "\n".join(log_lines), preview, _("Run D: Reloading vision model (multi-frame)..."), (log_lines[-1] if log_lines else "")
                    items = run_vision_multi(
                        items,
                        str(video_path),
                        work_dir,
                        cfg,
                        target_sub_ids,
                        cfg.vision.max_frames_per_sub,
                        log_lines=log_lines,
                        progress_callback=lambda p, d: update_progress(
                            (progress_audio + progress_brief + progress_vision * 0.5 + p * (progress_vision * 0.5)) / total_progress, d
                        ),
                        preview_callback=lambda b: preview_holder.__setitem__(0, _save_preview_bytes_to_file(b, work_dir)),
                    )
                    if preview_holder[0] is not None:
                        preview = preview_holder[0]
                        if isinstance(preview, bytes):
                            preview = _save_preview_bytes_to_file(preview, work_dir)
                    yield None, None, "\n".join(log_lines), preview, _("Run D: Vision model reloaded, analyzing multiple frames..."), (log_lines[-1] if log_lines else "")
                
                # 重新生成 brief v3（帶多張影像提示）；更新前先留 snapshot
                _copy_brief_snapshot(work_dir, "v2")
                yield None, None, "\n".join(log_lines), preview, _("Run D: Regenerating brief v3 with multi-frame vision..."), (log_lines[-1] if log_lines else "")

                # 建立 vision_hint_map
                vision_hint_map = {
                    sub_id: item.vision_desc_n
                    for sub_id, item in items.items()
                    if item.vision_desc_n
                }
                
                # 嘗試載入已存在的 brief_v3
                try:
                    loaded_items = load_brief_results_compat(work_dir, "v3", items)
                    if loaded_items and len(loaded_items) == len(items) and _cache_pack_ok(loaded_items, "v3"):
                        log_lines.append(f"[Run D] Loaded existing brief v3: {len(loaded_items)} items")
                        items = loaded_items
                    else:
                        if loaded_items and len(loaded_items) == len(items) and not _cache_pack_ok(loaded_items, "v3"):
                            log_lines.append("[Run D] Cache missing PACK, forcing regenerate")
                        else:
                            log_lines.append("[Run D] No existing brief v3 or count mismatch, reloading main model and regenerating...")
                        yield None, None, "\n".join(log_lines), preview, _("Run D: Reloading main reasoning model for brief v3..."), (log_lines[-1] if log_lines else "")
                        result_container = {"items": None, "done": False, "error": None}
                        def run_brief_d3():
                            try:
                                result_container["items"] = run_brief_text(
                                    items,
                                    work_dir,
                                    cfg,
                                    reason_prompt_config,
                                    version="v3",
                                    vision_hint_map=vision_hint_map,
                                    log_lines=log_lines,
                                    progress_callback=lambda p, d: update_progress(
                                        (progress_audio + progress_brief + progress_vision * 0.5 + p * (progress_vision * 0.5)) / total_progress, d
                                    ),
                                    batch_size=batch_size if batch_size < total else None,
                                    target_language=target_locale,
                                )
                            except Exception as ex:
                                result_container["error"] = ex
                            result_container["done"] = True
                        thread = threading.Thread(target=run_brief_d3, daemon=True)
                        thread.start()
                        while not result_container["done"]:
                            time.sleep(1.0)
                            yield None, None, "\n".join(log_lines), preview, _("Run D: Loading model / brief v3... (see log)"), (log_lines[-1] if log_lines else "")
                        thread.join(timeout=1.0)
                        if result_container["error"]:
                            raise result_container["error"]
                        items = result_container["items"]
                        yield None, None, "\n".join(log_lines), preview, _("Run D: Main model reloaded, regenerating brief v3 with parallel batch inference..."), (log_lines[-1] if log_lines else "")
                except Exception as e:
                    log_lines.append(f"[Run D] Error loading existing brief v3: {e}, reloading main model and regenerating...")
                    yield None, None, "\n".join(log_lines), preview, _("Run D: Reloading main reasoning model for brief v3..."), (log_lines[-1] if log_lines else "")
                    result_container = {"items": None, "done": False, "error": None}
                    def run_brief_d3():
                        try:
                            result_container["items"] = run_brief_text(
                                items,
                                work_dir,
                                cfg,
                                reason_prompt_config,
                                version="v3",
                                vision_hint_map=vision_hint_map,
                                log_lines=log_lines,
                                progress_callback=lambda p, d: update_progress(
                                    (progress_audio + progress_brief + progress_vision * 0.5 + p * (progress_vision * 0.5)) / total_progress, d
                                ),
                                batch_size=batch_size if batch_size < total else None,
                                target_language=target_locale,
                            )
                        except Exception as ex:
                            result_container["error"] = ex
                        result_container["done"] = True
                    thread = threading.Thread(target=run_brief_d3, daemon=True)
                    thread.start()
                    while not result_container["done"]:
                        time.sleep(1.0)
                        yield None, None, "\n".join(log_lines), preview, _("Run D: Reloading / brief v3... (see log)"), (log_lines[-1] if log_lines else "")
                    thread.join(timeout=1.0)
                    if result_container["error"]:
                        raise result_container["error"]
                    items = result_container["items"]
                    yield None, None, "\n".join(log_lines), preview, _("Run D: Main model reloaded, regenerating brief v3 with parallel batch inference..."), (log_lines[-1] if log_lines else "")
            else:
                log_lines.append("[Run D] No items need vision (multi-frame) fallback")
            
            update_progress(progress_audio + progress_brief + progress_vision, _("Run D: Completed"))
            yield None, None, "\n".join(log_lines), preview, _("Run D: Completed ({count} items)").format(count=len(items)), (log_lines[-1] if log_lines else "")

        # ========== Run E: Context Expansion（need_more_context 用 prev-3/next-3 更新 brief）==========
        # 僅在 run_mode 為 E 或（all 且 啟用更多上下文備援）時執行
        if run_mode in ("all", "E") and (run_mode == "E" or getattr(cfg.pipeline, "enable_context_expansion", True)):
            # 確保有當前 brief（E-only 時從 brief_work.jsonl / 舊名 brief.jsonl 載入）
            loaded_brief_e = load_brief_results_compat(work_dir, "v3", items)
            if loaded_brief_e and len(loaded_brief_e) == len(items):
                items = loaded_brief_e
            _copy_brief_snapshot(work_dir, "v3")
            log_lines.append("[Run E] Context expansion (need_more_context → prev-3/next-3 stage2)...")
            yield None, None, "\n".join(log_lines), preview, _("Run E: Context expansion..."), (log_lines[-1] if log_lines else "")
            update_progress(progress_audio + progress_brief + progress_vision, _("Run E: Context expansion"))
            items = run_context_expansion(
                items,
                work_dir,
                cfg,
                reason_prompt_config,
                target_language=target_locale,
                log_lines=log_lines,
                progress_callback=lambda p, d: update_progress(
                    (progress_audio + progress_brief + progress_vision + p * progress_expansion) / total_progress, d
                ),
            )
            update_progress(progress_audio + progress_brief + progress_vision + progress_expansion, _("Run E: Completed"))
            yield None, None, "\n".join(log_lines), preview, _("Run E: Completed"), (log_lines[-1] if log_lines else "")

        # ========== Run F: Final Translation ==========
        name_mappings: list = []
        if run_mode in ("all", "F"):
            # Run F 前先留一份 snapshot（brief_v4.jsonl），並確保 brief 為 E 更新後（從 brief_work.jsonl / 舊名 brief.jsonl 載入）
            _copy_brief_snapshot(work_dir, "v4")
            loaded_brief_f = load_brief_results_compat(work_dir, "v3", items)
            if loaded_brief_f and len(loaded_brief_f) == len(items):
                items = loaded_brief_f
            log_lines.append("[Run F] Preparing to load translation model (localization) and generate final subtitles...")
            yield None, None, "\n".join(log_lines), preview, _("Run F: Loading translation model (localization)..."), (log_lines[-1] if log_lines else "")
            update_progress(progress_audio + progress_brief + progress_vision + progress_expansion, _("Run F: Loading translation model (localization)"))

            # 嘗試載入已存在的最終翻譯結果（相容舊格式）
            try:
                loaded_items = load_final_translations_compat(work_dir, items)
                if loaded_items and len(loaded_items) == len(items):
                    log_lines.append(f"[Run F] Loaded existing final translations: {len(loaded_items)} items")
                    items = loaded_items
                else:
                    log_lines.append("[Run F] No existing final translations or count mismatch, loading translation model...")
                    yield None, None, "\n".join(log_lines), preview, _("Run F: Loading translation model (localization)..."), (log_lines[-1] if log_lines else "")
                    result_container = {"items": None, "done": False, "error": None}
                    def run_translate_f():
                        try:
                            result_container["items"] = run_final_translate(
                                items,
                                work_dir,
                                cfg,
                                glossary,
                                target_locale,
                                translate_prompt_config,
                                reason_prompt_config=reason_prompt_config,
                                reason_assemble_prompt_config=reason_assemble_prompt_config,
                                reason_group_translate_prompt_config=reason_group_translate_prompt_config,
                                translate_polish_prompt_config=translate_polish_prompt_config,
                                log_lines=log_lines,
                                progress_callback=lambda p, d: update_progress(
                                    (progress_audio + progress_brief + progress_vision + progress_expansion + p * progress_translate) / total_progress, d
                                ),
                                batch_size=batch_size if batch_size < total else None,
                            )
                        except Exception as ex:
                            result_container["error"] = ex
                        result_container["done"] = True
                    thread = threading.Thread(target=run_translate_f, daemon=True)
                    thread.start()
                    while not result_container["done"]:
                        time.sleep(1.0)
                        yield None, None, "\n".join(log_lines), preview, _("Run F: Loading translation model / processing... (see log)"), (log_lines[-1] if log_lines else "")
                    thread.join(timeout=1.0)
                    if result_container["error"]:
                        raise result_container["error"]
                    result = result_container["items"]
                    items = result[0] if isinstance(result, (list, tuple)) and result else result
                    name_mappings = result[1] if isinstance(result, (list, tuple)) and len(result) > 1 else []
                    yield None, None, "\n".join(log_lines), preview, _("Run F: Translation model loaded, processing with parallel batch inference..."), (log_lines[-1] if log_lines else "")
            except Exception as e:
                log_lines.append(f"[Run F] Error loading existing translations: {e}, reloading translation model...")
                yield None, None, "\n".join(log_lines), preview, _("Run F: Reloading translation model (localization)..."), (log_lines[-1] if log_lines else "")
                result_container = {"items": None, "done": False, "error": None}
                def run_translate_f():
                    try:
                        result_container["items"] = run_final_translate(
                            items,
                            work_dir,
                            cfg,
                            glossary,
                            target_locale,
                            translate_prompt_config,
                            reason_prompt_config=reason_prompt_config,
                            reason_assemble_prompt_config=reason_assemble_prompt_config,
                            reason_group_translate_prompt_config=reason_group_translate_prompt_config,
                            translate_polish_prompt_config=translate_polish_prompt_config,
                            log_lines=log_lines,
                            progress_callback=lambda p, d: update_progress(
                                (progress_audio + progress_brief + progress_vision + progress_expansion + p * progress_translate) / total_progress, d
                            ),
                            batch_size=batch_size if batch_size < total else None,
                        )
                    except Exception as ex:
                        result_container["error"] = ex
                    result_container["done"] = True
                thread = threading.Thread(target=run_translate_f, daemon=True)
                thread.start()
                while not result_container["done"]:
                    time.sleep(1.0)
                    yield None, None, "\n".join(log_lines), preview, _("Run F: Reloading / processing... (see log)"), (log_lines[-1] if log_lines else "")
                thread.join(timeout=1.0)
                if result_container["error"]:
                    raise result_container["error"]
                result = result_container["items"]
                items = result[0] if isinstance(result, (list, tuple)) and result else result
                name_mappings = result[1] if isinstance(result, (list, tuple)) and len(result) > 1 else []
                yield None, None, "\n".join(log_lines), preview, _("Run F: Translation model reloaded, processing with parallel batch inference..."), (log_lines[-1] if log_lines else "")
            
            # 將翻譯結果寫回 SRT：依 (start_ms, end_ms) 對齊，同一時間鍵可對應多筆字幕，依原始順序消耗
            time_to_item: dict[tuple[int, int], list] = {}
            for sub_id in sorted(items.keys()):
                item = items[sub_id]
                key = (round(item.start_ms), round(item.end_ms))
                time_to_item.setdefault(key, []).append(item)
            for sub in subs:
                start_ms, end_ms, _mid = sub_midpoints_ms(sub)
                key = (round(start_ms), round(end_ms))
                queue = time_to_item.get(key)
                if not queue:
                    continue
                item = queue.pop(0)
                text = (item.translated_text or "").strip()
                if not text:
                    continue
                # 保留原文的 <i> 標籤：若原文有 <i> 則用 <i> 包住譯文
                orig_raw = (sub.text or "").strip()
                if "<i>" in orig_raw or orig_raw.startswith("<i"):
                    text = f"<i>{text}</i>" if "<i>" not in text else text
                sub.text = text

            # 保存最終 SRT
                save_srt(subs, str(out_path), encoding="utf-8")

            # 人名/專有名詞 CSV（格式同術語表：English, Target, Note, Enabled）
            names_csv_path: Optional[Path] = None
            if name_mappings:
                names_csv_path = out_path.parent / f"{out_path.stem}.names.csv"
                target_col = target_locale or "Target"
                with open(names_csv_path, "w", encoding="utf-8-sig", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["English", target_col, "Note", "Enabled"])
                    for en, target in name_mappings:
                        w.writerow([en, target or "", "", True])
            
            update_progress(total_progress, _("Run F: Completed"))
            yield str(out_path), str(names_csv_path) if names_csv_path else None, "\n".join(log_lines), preview, _("All runs completed!"), (log_lines[-1] if log_lines else "")
        else:
            # Run F 未執行，但可能有部分結果
            name_mappings = []
            yield None, None, "\n".join(log_lines), preview, _("Translation not executed (Run F not selected)."), (log_lines[-1] if log_lines else "")

    except Exception as e:
        tb = traceback.format_exc()
        # Keep UI usable: show traceback in Log and a short message in Status.
        # 確保卸載模型（新 pipeline 使用 ModelMutex，會在 finally 中自動處理）
        errmsg = _("Error: {error}").format(error=str(e))
        yield None, None, tb[-12000:], None, errmsg, (tb.strip().split("\n")[-1].strip() if tb.strip() else errmsg)
        return


# -----------------------------
# Glossary UI
# -----------------------------
def glossary_load():
    cfg = _ensure_config()
    entries = load_glossary(cfg.glossary.json_path)
    return [[e.en, e.zh, e.note, e.enabled] for e in entries], _("Loaded {count} entries from {path}").format(count=len(entries), path=cfg.glossary.json_path)


def glossary_save(table):
    cfg = _ensure_config()
    entries = []
    
    # Handle Gradio DataFrame - it might be a pandas DataFrame or a list
    if table is None:
        table = []
    elif hasattr(table, 'values'):
        # pandas DataFrame
        table = table.values.tolist()
    elif not isinstance(table, list):
        # Try to convert to list
        try:
            table = list(table)
        except Exception:
            table = []
    
    for row in table:
        if not row:
            continue
        # Handle row as list or tuple
        if not isinstance(row, (list, tuple)):
            continue
        if len(row) < 2:
            continue
        en = (row[0] or "").strip() if row[0] is not None else ""
        zh = (row[1] or "").strip() if row[1] is not None else ""
        if not en or not zh:
            continue
        note = (row[2] or "").strip() if len(row) > 2 and row[2] is not None else ""
        enabled = bool(row[3]) if len(row) > 3 and row[3] is not None else True
        entries.append(GlossaryEntry(en=en, zh=zh, note=note, enabled=enabled))
    save_glossary(cfg.glossary.json_path, entries)
    return _("Saved {count} entries to {path}").format(count=len(entries), path=cfg.glossary.json_path)


def glossary_import_csv(file_obj):
    cfg = _ensure_config()
    if not file_obj:
        return gr.update(), _("No file.")
    data = _read_uploaded_file_bytes(file_obj)
    new_entries = import_csv_two_cols(data)
    entries = load_glossary(cfg.glossary.json_path) + new_entries
    save_glossary(cfg.glossary.json_path, entries)
    return [[e.en, e.zh, e.note, e.enabled] for e in entries], _("Imported {new_count} from CSV. Total {total_count}.").format(new_count=len(new_entries), total_count=len(entries))


def glossary_import_template(file_obj):
    cfg = _ensure_config()
    if not file_obj:
        return gr.update(), _("No file.")
    data = _read_uploaded_file_bytes(file_obj)
    new_entries = import_subtitleedit_multiple_replace_template(data)
    entries = load_glossary(cfg.glossary.json_path) + new_entries
    save_glossary(cfg.glossary.json_path, entries)
    return [[e.en, e.zh, e.note, e.enabled] for e in entries], _("Imported {new_count} from .template (best-effort). Total {total_count}.").format(new_count=len(new_entries), total_count=len(entries))


# -----------------------------
# UI
# -----------------------------
# 從 language_config.json 獲取確切的語言代碼用於標題
_TITLE_LANG = _TARGET_LANG_LOCALE if _TARGET_LANG_LOCALE else "zh-TW"

with gr.Blocks(title=_("Local Subtitle Translator (Vision + Reason)")) as demo:
    gr.Markdown(
        f"# 🎬 Local Subtitle Translator (EN → {_TITLE_LANG})\n"
        f"{_('No Ollama needed. Runs with llama-cpp-python (GGUF) + optional local vision (GGUF vision model).')}\n"
    )

    with gr.Tab(_("Translate")):
        with gr.Row():
            with gr.Column(scale=1):
                # NOTE: Gradio's Video component requires an external ffmpeg executable.
                # To keep this app fully self-contained, we use File input for videos.
                video = gr.File(label=_("Video (MKV/MP4) — required for audio and vision analysis"), file_types=[".mp4", ".mkv", ".mov", ".avi"])
                srt = gr.File(label=_("SRT (English)"), file_types=[".srt"])
                srt_encoding = gr.Dropdown(
                    ["utf-8", "utf-8-sig", "cp1252"],
                    value="utf-8",
                    label=_("SRT encoding"),
                )
                # 語言設定已由 language_config.json 控制，不再需要 UI 選擇器
                gr.Markdown(
                    f"**{_('Target language')}:** `{_TARGET_LANG_LOCALE}` "
                    f"({_('configured in language_config.json')})"
                )
                # Vision is optional. Leave it off by default so the app works even
                # if you haven't downloaded a vision GGUF + mmproj yet.
                enable_vision = gr.Checkbox(value=False, label=_("Enable vision fallback (Run C/D, local GGUF vision model)"))
                enable_context_expansion = gr.Checkbox(value=True, label=_("Enable context expansion fallback (Run E, need_more_context)"))
                max_frames = gr.Slider(
                    1, 4, value=1, step=1,
                    label=_("Max frames per subtitle (Run D)"),
                )
                offsets = gr.Textbox(
                    value="0.5",
                    label=_("Frame offsets within subtitle span (comma-separated, 0..1)"),
                    placeholder=_("e.g. 0.3,0.7"),
                )
                run_mode = gr.Dropdown(
                    choices=["all", "A", "B", "C", "D", "E", "F"],
                    value="all",
                    label=_("Run mode"),
                    info=_("all = A→B→(C/D)→E→F; E = context expansion; F = Translate"),
                )
                run_e_scheme = gr.Dropdown(
                    choices=["full", "main_led", "local_led", "draft_first"],
                    value="full",
                    label=_("Run F scheme (when to use main vs. localization model)"),
                    info=_("Full=both strong | MAIN-led=main strong | LOCAL-led=local strong | Draft-first=both weak. See README Run F."),
                )
                with gr.Row():
                    run = gr.Button(_("🚀 Translate"), variant="primary")
                    btn_reset = gr.Button(_("Reset"), variant="secondary")
                log = gr.Textbox(
                    label=_("Log (last 200 lines)"),
                    lines=14,
                    max_lines=14,
                    interactive=False,
                )

            with gr.Column(scale=1):
                out_file = gr.File(label=_("Download translated SRT"))
                out_names_file = gr.File(label=_("Download names CSV (glossary format)"))
                status = gr.Textbox(label=_("Status (EN)"), value=_("Idle."), interactive=False)
                latest_log = gr.Textbox(
                    label=_("Latest log"),
                    value="",
                    lines=1,
                    max_lines=1,
                    interactive=False,
                )
                img = gr.Image(label=_("Preview frame (when vision runs)"), height=300)

        def reset_translate_tab():
            """一鍵重設：清空輸入/輸出與日誌，恢復預設值，方便開始新翻譯。"""
            return (
                gr.update(value=None),   # video（File 需 update 才會清空）
                gr.update(value=None),   # srt
                "utf-8",  # srt_encoding
                False,  # enable_vision
                True,    # enable_context_expansion
                1,      # max_frames
                "0.5",  # offsets
                "all",  # run_mode
                "full", # run_e_scheme
                gr.update(value=None),   # out_file
                gr.update(value=None),   # out_names_file
                "",     # log
                gr.update(value=None),   # img
                _("Idle. Ready for new translation."),
                "",     # latest_log
                )

        run.click(
            translate_ui,
            inputs=[video, srt, srt_encoding, enable_vision, enable_context_expansion, max_frames, offsets, run_mode, run_e_scheme],
            outputs=[out_file, out_names_file, log, img, status, latest_log],
        )
        btn_reset.click(
            reset_translate_tab,
            inputs=[],
            outputs=[video, srt, srt_encoding, enable_vision, enable_context_expansion, max_frames, offsets, run_mode, run_e_scheme, out_file, out_names_file, log, img, status, latest_log],
        )

    with gr.Tab(_("Glossary")):
        # 根據目標語言顯示對應的語言名稱
        target_lang_name = {
            "zh-TW": "繁體中文",
            "zh-CN": "簡體中文",
            "ja-JP": "日本語",
            "es-ES": "Español",
        }.get(_TARGET_LANG_LOCALE, _TARGET_LANG_LOCALE)
        
        gr.Markdown(
            _("""### Glossary / Replace Library
- Edit in-table
- Import CSV (2 columns: English, {target_lang})
- Import Subtitle Edit `.template` (best-effort XML extraction)
""").format(target_lang=_TARGET_LANG_LOCALE)
        )
        table = gr.Dataframe(
            headers=[_("English"), target_lang_name, _("Note"), _("Enabled")],
            datatype=["str", "str", "str", "bool"],
            interactive=True,
            row_count=(0, "dynamic"),
            column_count=(4, "fixed"),
        )

        with gr.Row():
            btn_load = gr.Button(_("Load"))
            btn_save = gr.Button(_("Save"))

        with gr.Row():
            csv_file = gr.File(label=_("Import CSV"), file_types=[".csv"])
            tpl_file = gr.File(label=_("Import Subtitle Edit .template"), file_types=[".template"])

        with gr.Row():
            btn_imp_csv = gr.Button(_("Import CSV"))
            btn_imp_tpl = gr.Button(_("Import Subtitle Edit .template"))

        g_status = gr.Textbox(label=_("Glossary status"), interactive=False)

        btn_load.click(glossary_load, inputs=[], outputs=[table, g_status])
        btn_save.click(glossary_save, inputs=[table], outputs=[g_status])
        btn_imp_csv.click(glossary_import_csv, inputs=[csv_file], outputs=[table, g_status])
        btn_imp_tpl.click(glossary_import_template, inputs=[tpl_file], outputs=[table, g_status])

    with gr.Tab(_("Setup / Model Paths")):
        # Ensure config.json exists (do not shadow gettext `_`)
        _ensure_config()
        gr.Markdown(
            _("""### config.json
Edit `config.json` to point to your GGUF model files under `./models/`.

If missing, the Translate tab will show a clear error.
""")
        )

        cfg_box = gr.Code(value=CONFIG_PATH.read_text(encoding="utf-8"), language="json")
        btn_write = gr.Button(_("Write to config.json"))
        cfg_write_status = gr.Textbox(interactive=False)

        def write_cfg(txt: str):
            try:
                CONFIG_PATH.write_text(txt, encoding="utf-8")
                return _("Saved config.json")
            except Exception as e:
                return _("Failed: {error}").format(error=str(e))

        btn_write.click(write_cfg, inputs=[cfg_box], outputs=[cfg_write_status])

if __name__ == "__main__":
    # show_error=True: show backend tracebacks in the browser when something goes wrong.
    demo.launch(inbrowser=True, show_error=True)
