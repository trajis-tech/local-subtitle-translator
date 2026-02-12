"""
JSONL 格式相容性工具。

提供舊格式（idx-based）與新格式（sub_id-based）之間的轉換。
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Optional
import json

from .subtitle_item import SubtitleItem, AudioMeta, BriefMeta, generate_sub_id


def load_audio_results_compat(
    work_dir: Path,
    items: Optional[dict[str, SubtitleItem]] = None,
) -> Optional[dict[str, SubtitleItem]]:
    """
    載入音訊結果（相容舊格式和新格式）。
    
    舊格式：{"idx": 0, "start_ms": ..., "audio_tags": {...}}
    新格式：{"sub_id": "...", "start_ms": ..., "audio_tags": {...}}
    
    Args:
        work_dir: 工作目錄
        items: 可選的 SubtitleItem 字典（用於對齊）
    
    Returns:
        更新後的 items 字典，或 None（如果檔案不存在）
    """
    jsonl_path = work_dir / "audio_tags.jsonl"
    if not jsonl_path.exists():
        return None
    
    if items is None:
        items = {}
    
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            # 略過 Run A 寫入的 _run_meta 第一行（方便驗收載入成敗原因）
            if "_run_meta" in data:
                continue
            # 判斷格式：新格式有 sub_id，舊格式有 idx
            if "sub_id" in data:
                # 新格式
                sub_id = data["sub_id"]
                item = items.get(sub_id)
                if item:
                    audio_tags_data = data.get("audio_tags", {})
                    item.audio_meta = AudioMeta(
                        emotion=audio_tags_data.get("emotion", ""),
                        tone=audio_tags_data.get("tone", ""),
                        intensity=audio_tags_data.get("intensity", ""),
                        speaking_style=audio_tags_data.get("speaking_style", ""),
                        audio_reason=data.get("audio_reason", ""),
                    )
            elif "idx" in data:
                # 舊格式：需要通過 start_ms/end_ms/text_raw 找到對應的 sub_id
                start_ms = data.get("start_ms", 0)
                end_ms = data.get("end_ms", 0)
                # 嘗試從 items 中找到匹配的項目
                for sub_id, item in items.items():
                    if abs(item.start_ms - start_ms) < 1.0 and abs(item.end_ms - end_ms) < 1.0:
                        audio_tags_data = data.get("audio_tags", {})
                        item.audio_meta = AudioMeta(
                            emotion=audio_tags_data.get("emotion", ""),
                            tone=audio_tags_data.get("tone", ""),
                            intensity=audio_tags_data.get("intensity", ""),
                            speaking_style=audio_tags_data.get("speaking_style", ""),
                            audio_reason=data.get("audio_reason", ""),
                        )
                        break
    
    return items if items else None


def load_brief_results_compat(
    work_dir: Path,
    version: str,
    items: Optional[dict[str, SubtitleItem]] = None,
) -> Optional[dict[str, SubtitleItem]]:
    """
    載入 brief 結果（相容舊格式和新格式）。
    
    舊格式：{"idx": 0, "translation_brief": ..., ...}
    新格式：{"sub_id": "...", "translation_brief": ..., ...}
    
    Args:
        work_dir: 工作目錄
        version: 版本標記（v1, v2, v3）
        items: 可選的 SubtitleItem 字典（用於對齊）
    
    Returns:
        更新後的 items 字典，或 None（如果檔案不存在）
    """
    # 單一 brief：優先讀取 brief_work.jsonl（當前），否則舊名 brief.jsonl，再否則 brief_{version}.jsonl（snapshot）
    jsonl_path = work_dir / "brief_work.jsonl"
    if not jsonl_path.exists():
        legacy = work_dir / "brief.jsonl"
        if legacy.exists():
            jsonl_path = legacy
        else:
            jsonl_path = work_dir / f"brief_{version}.jsonl"
    if not jsonl_path.exists():
        return None

    if items is None:
        items = {}
    
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            
            def _brief_bool(key: str):
                v = data.get(key)
                if v is None:
                    return None
                if isinstance(v, bool):
                    return v
                if isinstance(v, str):
                    return v.lower() in ("true", "1", "yes")
                try:
                    return bool(int(v))
                except (TypeError, ValueError):
                    return None

            brief_meta = BriefMeta(
                translation_brief=data.get("translation_brief", ""),
                reasons=data.get("reasons", ""),
                version=data.get("version", version),
                need_vision=_brief_bool("need_vision"),
                need_multi_frame_vision=_brief_bool("need_multi_frame_vision"),
                need_more_context=_brief_bool("need_more_context"),
            )
            
            # 判斷格式
            if "sub_id" in data:
                # 新格式
                sub_id = data["sub_id"]
                item = items.get(sub_id)
                if item:
                    if version == "v1":
                        item.brief_v1 = brief_meta
                    elif version == "v2":
                        item.brief_v2 = brief_meta
                    elif version == "v3":
                        item.brief_v3 = brief_meta
            elif "idx" in data:
                # 舊格式：通過 start_ms/end_ms 匹配
                start_ms = data.get("start_ms", 0)
                end_ms = data.get("end_ms", 0)
                for sub_id, item in items.items():
                    if abs(item.start_ms - start_ms) < 1.0 and abs(item.end_ms - end_ms) < 1.0:
                        if version == "v1":
                            item.brief_v1 = brief_meta
                        elif version == "v2":
                            item.brief_v2 = brief_meta
                        elif version == "v3":
                            item.brief_v3 = brief_meta
                        break
    
    return items if items else None


def load_vision_results_compat(
    work_dir: Path,
    single_frame: bool = True,
    items: Optional[dict[str, SubtitleItem]] = None,
) -> Optional[dict[str, SubtitleItem]]:
    """
    載入視覺結果（相容舊格式和新格式）。
    
    舊格式：{"idx": 0, "vision_desc": ...}
    新格式：{"sub_id": "...", "vision_desc": ...}
    
    Args:
        work_dir: 工作目錄
        single_frame: True = 單張影像，False = 多張影像
        items: 可選的 SubtitleItem 字典（用於對齊）
    
    Returns:
        更新後的 items 字典，或 None（如果檔案不存在）
    """
    jsonl_path = work_dir / ("vision_1frame.jsonl" if single_frame else "vision_multiframe.jsonl")
    if not jsonl_path.exists():
        return None
    
    if items is None:
        items = {}
    
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            
            vision_desc = data.get("vision_desc", "")
            
            # 判斷格式
            if "sub_id" in data:
                # 新格式
                sub_id = data["sub_id"]
                item = items.get(sub_id)
                if item:
                    if single_frame:
                        item.vision_desc_1 = vision_desc
                    else:
                        item.vision_desc_n = vision_desc
            elif "idx" in data:
                # 舊格式：通過 start_ms/end_ms 匹配
                start_ms = data.get("start_ms", 0)
                end_ms = data.get("end_ms", 0)
                for sub_id, item in items.items():
                    if abs(item.start_ms - start_ms) < 1.0 and abs(item.end_ms - end_ms) < 1.0:
                        if single_frame:
                            item.vision_desc_1 = vision_desc
                        else:
                            item.vision_desc_n = vision_desc
                        break
    
    return items if items else None


def load_final_translations_compat(
    work_dir: Path,
    items: Optional[dict[str, SubtitleItem]] = None,
) -> Optional[dict[str, SubtitleItem]]:
    """
    載入最終翻譯結果（相容舊格式和新格式）。
    
    新格式：{"sub_id": "...", "translated_text": ...}
    注意：舊格式沒有單獨的 final_translations.jsonl，翻譯結果直接寫入 SRT
    
    Args:
        work_dir: 工作目錄
        items: 可選的 SubtitleItem 字典（用於對齊）
    
    Returns:
        更新後的 items 字典，或 None（如果檔案不存在）
    """
    jsonl_path = work_dir / "final_translations.jsonl"
    if not jsonl_path.exists():
        return None
    
    if items is None:
        items = {}
    
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            
            if "sub_id" in data:
                sub_id = data["sub_id"]
                item = items.get(sub_id)
                if item:
                    item.translated_text = data.get("translated_text", "")
    
    return items if items else None
