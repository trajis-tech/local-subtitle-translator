"""
Run B/C/D: Main model generates translation brief.

This module handles:
- Run B: Initial brief generation (EN + prev/next + audio_tags)
- Run C: Re-generate brief with single-frame vision hint
- Run D: Re-generate brief with multi-frame vision hint
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any, Optional

from .pipeline_audio import AudioResult, AudioTags
from .srt_utils import sub_midpoints_ms


def _format_audio_hint(audio_result: AudioResult | None) -> str:
    """
    將 AudioResult 轉換為乾淨的文字提示（audio_hint）。
    
    Args:
        audio_result: AudioResult 實例（可為 None）
    
    Returns:
        格式化的音訊提示文字，例如：
        "Emotion: excited, Tone: sarcastic, Intensity: high, Style: fast"
        如果 audio_result 為 None 或所有欄位為空，返回 "(none)"
    """
    if audio_result is None or audio_result.audio_tags is None:
        return "(none)"
    
    tags = audio_result.audio_tags
    parts = []
    
    # 情緒
    if tags.emotion and tags.emotion.strip():
        parts.append(f"Emotion: {tags.emotion.strip()}")
    
    # 語氣
    if tags.tone and tags.tone.strip():
        parts.append(f"Tone: {tags.tone.strip()}")
    
    # 強度
    if tags.intensity and tags.intensity.strip():
        parts.append(f"Intensity: {tags.intensity.strip()}")
    
    # 說話方式
    if tags.speaking_style and tags.speaking_style.strip():
        parts.append(f"Style: {tags.speaking_style.strip()}")
    
    # 如果有 audio_reason，也加入（可能包含額外資訊）
    if audio_result.audio_reason and audio_result.audio_reason.strip():
        reason = audio_result.audio_reason.strip()
        # 避免重複資訊
        if reason.lower() not in ["audio model not loaded", "audio analysis failed"]:
            parts.append(f"Note: {reason}")
    
    if not parts:
        return "(none)"
    
    return ", ".join(parts)


@dataclass
class BriefResult:
    """翻譯描述結果"""
    idx: int
    start_ms: float
    end_ms: float
    translation_brief: str  # 指導翻譯方向的描述
    reasons: str = ""  # 為什麼這個分數（含 PACK JSON）
    version: str = "v1"  # v1, v2, v3
    need_vision: Optional[bool] = None  # v1: 是否需視覺
    need_multi_frame_vision: Optional[bool] = None  # v2: 是否需多張視覺
    need_more_context: Optional[bool] = None  # 是否需更多前後文（Run E 方案 C）


def run_brief_generation(
    reason_model: Any,  # TextModel
    subs: Any,  # pysrt.SubRipFile
    en_lines: list[str],
    audio_results: list[AudioResult] | None,
    vision_hint: str | None = None,
    version: str = "v1",
    work_dir: Path | None = None,
    prompt_config: Any | None = None,
    progress_callback: Any | None = None,
    batch_size: int | None = None,
    enable_parallel: bool = True,
    max_workers: int | None = None,
) -> list[BriefResult]:
    """
    生成翻譯描述（translation_brief）。
    
    Args:
        reason_model: 主推理模型（已載入）
        subs: pysrt.SubRipFile
        en_lines: 清理後的英文字幕列表
        audio_results: 音訊結果（可選）
        vision_hint: 視覺提示（可選，用於 v2/v3）
        version: 版本標記（v1, v2, v3）
        work_dir: 工作目錄（用於寫入 JSONL）
        prompt_config: CSV 載入的 prompt 設定
        progress_callback: 進度回調
        batch_size: 批次大小（None = 不使用批次處理）
        enable_parallel: 是否啟用並行處理（僅用於 I/O 操作）
        max_workers: 最大並行工作數（None = 自動檢測）
    
    Returns:
        BriefResult 列表
    """
    from .pipeline import stage2_reason_and_score
    
    results: list[BriefResult] = []
    total = len(subs)
    
    # 計算批次大小（如果未指定）
    if batch_size is None:
        batch_size = total  # 預設：不使用批次處理（一次性處理所有）
    
    # 注意：enable_parallel 和 max_workers 參數保留以備將來使用
    # 目前模型推理必須順序進行（模型不是線程安全的），所以不使用並行處理
    
    # 批次處理
    processed = 0
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_indices = list(range(batch_start, batch_end))
        
        # 處理批次（模型推理必須順序進行，因為模型不是線程安全的）
        batch_results: list[BriefResult] = []
        
        for i in batch_indices:
            sub = subs[i]
            start_ms, end_ms, _mid = sub_midpoints_ms(sub)
            line_en = en_lines[i]
            
            # 上下文：前後各一句
            prev_ctx = [en_lines[i - 1]] if i > 0 else []
            next_ctx = [en_lines[i + 1]] if i < total - 1 else []
            
            # 提取音訊提示（從 audio_results）
            audio_hint: str | None = None
            if audio_results and i < len(audio_results):
                audio_result = audio_results[i]
                audio_hint = _format_audio_hint(audio_result)
                # 如果 audio_hint 是 "(none)"，設為 None 以便在 prompt 中明確標示
                if audio_hint == "(none)":
                    audio_hint = None
            
            # Debug log：在前 3 句顯示 audio_hint（用於驗收）
            if i < 3:
                audio_hint_display = audio_hint if audio_hint else "(none)"
                # 這個 log 會在調用端通過 progress_callback 或直接輸出顯示
                # 實際的 log 輸出會在 app.py 的 translate_ui 中處理
            
            # 如果有視覺提示，加入視覺上下文
            visual_context = vision_hint if vision_hint else None
            
            # 調用 Stage2（使用 CSV prompt，包含 audio_hint；v1 輸出 need_vision，v2 輸出 need_multi_frame_vision）
            s2_result = stage2_reason_and_score(
                reason_model,
                line_en,
                prev_ctx,
                next_ctx,
                visual_hint=visual_context,
                audio_hint=audio_hint,
                prompt_config=prompt_config,
                brief_version=version,
            )
            
            # 建立 BriefResult
            brief_result = BriefResult(
                idx=i,
                start_ms=start_ms,
                end_ms=end_ms,
                translation_brief=s2_result.meaning_en,
                reasons=s2_result.notes,
                version=version,
                need_vision=getattr(s2_result, "need_vision", None),
                need_multi_frame_vision=getattr(s2_result, "need_multi_frame_vision", None),
                need_more_context=getattr(s2_result, "need_more_context", None),
            )
            batch_results.append(brief_result)
            processed += 1
            if progress_callback:
                progress_callback(processed / total, f"Brief {version}: {processed}/{total}")
        
        results.extend(batch_results)
        
        # 批次寫入 JSONL（增量寫入）
        if work_dir:
            work_dir.mkdir(parents=True, exist_ok=True)
            jsonl_path = work_dir / f"brief_{version}.jsonl"
            mode = "a" if batch_start > 0 else "w"  # 第一個批次用寫入模式，後續用追加模式
            with jsonl_path.open(mode, encoding="utf-8") as f:
                for result in batch_results:
                    line = {
                        "idx": result.idx,
                        "start_ms": result.start_ms,
                        "end_ms": result.end_ms,
                        "translation_brief": result.translation_brief,
                        "reasons": result.reasons,
                        "version": result.version,
                        "need_vision": getattr(result, "need_vision", None),
                        "need_multi_frame_vision": getattr(result, "need_multi_frame_vision", None),
                        "need_more_context": getattr(result, "need_more_context", None),
                    }
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")
    
    return results




def load_brief_results(work_dir: Path, version: str) -> list[BriefResult] | None:
    """
    從 JSONL 載入 brief 結果（用於續跑）。
    """
    jsonl_path = work_dir / f"brief_{version}.jsonl"
    if not jsonl_path.exists():
        return None
    
    results: list[BriefResult] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            def _load_bool(key: str):
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

            results.append(BriefResult(
                idx=data["idx"],
                start_ms=data["start_ms"],
                end_ms=data["end_ms"],
                translation_brief=data["translation_brief"],
                reasons=data.get("reasons", ""),
                version=data.get("version", version),
                need_vision=_load_bool("need_vision"),
                need_multi_frame_vision=_load_bool("need_multi_frame_vision"),
                need_more_context=_load_bool("need_more_context"),
            ))
    
    return results if results else None


def select_best_brief(
    brief_v1: list[BriefResult] | None,
    brief_v2: list[BriefResult] | None,
    brief_v3: list[BriefResult] | None,
    idx: int,
) -> BriefResult | None:
    """
    選擇最佳 brief：v3 > v2 > v1
    """
    if brief_v3 and idx < len(brief_v3):
        return brief_v3[idx]
    if brief_v2 and idx < len(brief_v2):
        return brief_v2[idx]
    if brief_v1 and idx < len(brief_v1):
        return brief_v1[idx]
    return None
