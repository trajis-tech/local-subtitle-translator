"""
統一的字幕項目資料結構，確保所有 run 之間資料不錯位。

使用 sub_id（固定且不可變）來對齊結果，而不是 list index。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import hashlib


@dataclass
class AudioMeta:
    """音訊分析結果"""
    emotion: str = ""
    tone: str = ""
    intensity: str = ""
    speaking_style: str = ""
    audio_reason: str = ""
    
    def to_hint(self) -> str:
        """轉換為 audio_hint 文字"""
        parts = []
        if self.emotion and self.emotion.strip():
            parts.append(f"Emotion: {self.emotion.strip()}")
        if self.tone and self.tone.strip():
            parts.append(f"Tone: {self.tone.strip()}")
        if self.intensity and self.intensity.strip():
            parts.append(f"Intensity: {self.intensity.strip()}")
        if self.speaking_style and self.speaking_style.strip():
            parts.append(f"Style: {self.speaking_style.strip()}")
        if self.audio_reason and self.audio_reason.strip():
            reason = self.audio_reason.strip()
            if reason.lower() not in ["audio model not loaded", "audio analysis failed"]:
                parts.append(f"Note: {reason}")
        return ", ".join(parts) if parts else "(none)"


@dataclass
class BriefMeta:
    """Brief 生成結果（含指代/語氣/場景/專名/改寫指引；PACK 存於 reasons）"""
    translation_brief: str = ""
    reasons: str = ""
    version: str = "v1"  # v1, v2, v3
    # v1: 是否需視覺（單張）才能精準翻譯
    need_vision: Optional[bool] = None
    # v2: 單張視覺是否不足，需多張視覺
    need_multi_frame_vision: Optional[bool] = None
    # 是否需更多前後文才能消歧（Run E 方案 C：先擴充上下文，不滿意再補 brief 重翻）
    need_more_context: Optional[bool] = None


@dataclass
class SubtitleItem:
    """
    統一的字幕項目資料結構。
    
    所有 run 都必須使用 sub_id 來對齊結果，不可使用 list index。
    """
    # 固定識別符（不可變）
    sub_id: str  # hash(start_ms, end_ms, text_raw) 或原始序號字串
    
    # 基本資訊（從 SRT 讀取）
    start_ms: float
    end_ms: float
    text_raw: str  # 原始字幕文字（未清理）
    text_clean: str = ""  # 清理後的文字
    
    # Run A: 音訊分析結果
    audio_meta: Optional[AudioMeta] = None
    
    # Run B/C/D: Brief 生成結果
    brief_v1: Optional[BriefMeta] = None
    brief_v2: Optional[BriefMeta] = None
    brief_v3: Optional[BriefMeta] = None
    
    # Run C/D: Vision 分析結果
    vision_desc_1: str = ""  # 單張影像描述
    vision_desc_n: str = ""  # 多張影像描述

    # Run E: 最終翻譯結果
    translated_text: str = ""  # 最終翻譯後的文字
    
    def get_best_brief(self) -> Optional[BriefMeta]:
        """取得最佳 brief：v3 > v2 > v1"""
        if self.brief_v3:
            return self.brief_v3
        if self.brief_v2:
            return self.brief_v2
        if self.brief_v1:
            return self.brief_v1
        return None
    
    def get_audio_hint(self) -> str:
        """取得 audio_hint 文字"""
        if self.audio_meta:
            return self.audio_meta.to_hint()
        return "(none)"


def generate_sub_id(start_ms: float, end_ms: float, text_raw: str, index: Optional[int] = None) -> str:
    """
    生成 sub_id。
    
    優先使用 hash(start_ms, end_ms, text_raw) 確保唯一性。
    如果 hash 衝突風險高，可以加上 index。
    
    Args:
        start_ms: 開始時間（毫秒）
        end_ms: 結束時間（毫秒）
        text_raw: 原始文字
        index: 可選的原始序號（用於 fallback）
    
    Returns:
        sub_id 字串
    """
    # 使用 hash 確保唯一性
    content = f"{start_ms:.3f}|{end_ms:.3f}|{text_raw}"
    hash_obj = hashlib.md5(content.encode("utf-8"))
    hash_hex = hash_obj.hexdigest()[:16]  # 前 16 位
    
    # 如果提供了 index，加上它作為後綴（用於可讀性和除錯）
    if index is not None:
        return f"{hash_hex}_{index}"
    return hash_hex


def create_subtitle_items_from_srt(subs: Any) -> dict[str, SubtitleItem]:
    """
    從 pysrt.SubRipFile 創建 SubtitleItem 字典。
    
    Args:
        subs: pysrt.SubRipFile
    
    Returns:
        dict[sub_id, SubtitleItem]
    """
    from .srt_utils import sub_midpoints_ms, clean_srt_text
    
    items: dict[str, SubtitleItem] = {}
    
    for i, sub in enumerate(subs):
        start_ms, end_ms, _mid = sub_midpoints_ms(sub)
        text_raw = sub.text
        text_clean = clean_srt_text(text_raw)
        
        sub_id = generate_sub_id(start_ms, end_ms, text_raw, index=i)
        
        item = SubtitleItem(
            sub_id=sub_id,
            start_ms=start_ms,
            end_ms=end_ms,
            text_raw=text_raw,
            text_clean=text_clean,
        )
        
        items[sub_id] = item
    
    return items


def verify_subtitle_items_consistency(
    items: dict[str, SubtitleItem],
    expected_sub_ids: Optional[set[str]] = None,
    run_name: str = "unknown",
) -> None:
    """
    驗證 SubtitleItem 字典的一致性。
    
    Args:
        items: SubtitleItem 字典
        expected_sub_ids: 預期的 sub_id 集合（如果為 None，則不檢查）
        run_name: Run 名稱（用於錯誤訊息）
    
    Raises:
        ValueError: 如果發現不一致
    """
    actual_sub_ids = set(items.keys())
    
    if expected_sub_ids is None:
        # 只檢查是否有重複的 sub_id
        if len(actual_sub_ids) != len(items):
            duplicates = []
            seen = set()
            for sub_id in items.keys():
                if sub_id in seen:
                    duplicates.append(sub_id)
                seen.add(sub_id)
            raise ValueError(
                f"[{run_name}] 發現重複的 sub_id: {duplicates}"
            )
        return
    
    # 檢查是否缺少 sub_id
    missing = expected_sub_ids - actual_sub_ids
    if missing:
        raise ValueError(
            f"[{run_name}] 缺少以下 sub_id（共 {len(missing)} 個）:\n"
            f"  {sorted(list(missing))[:10]}{'...' if len(missing) > 10 else ''}"
        )
    
    # 檢查是否多出 sub_id
    extra = actual_sub_ids - expected_sub_ids
    if extra:
        raise ValueError(
            f"[{run_name}] 多出以下 sub_id（共 {len(extra)} 個）:\n"
            f"  {sorted(list(extra))[:10]}{'...' if len(extra) > 10 else ''}"
        )
