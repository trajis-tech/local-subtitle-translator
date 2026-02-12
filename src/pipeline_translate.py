"""
Run F: Final translation using localization model.

This module loads the best brief (v3 > v2 > v1) and translates to target language.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Optional

from .pipeline_brief import BriefResult, select_best_brief
from .glossary import GlossaryEntry


def run_final_translation(
    translate_model: Any,  # TextModel
    subs: Any,  # pysrt.SubRipFile
    en_lines: list[str],
    brief_v1: list[BriefResult] | None,
    brief_v2: list[BriefResult] | None,
    brief_v3: list[BriefResult] | None,
    glossary: list[GlossaryEntry],
    target_language: str,
    prompt_config: Any | None = None,
    progress_callback: Any | None = None,
    batch_size: int | None = None,
    log_lines: Optional[list[str]] = None,
) -> list[str]:
    """
    Run F: 最終翻譯（使用最佳 brief）。
    
    Args:
        translate_model: 翻譯模型（已載入）
        subs: pysrt.SubRipFile
        en_lines: 英文字幕列表
        brief_v1: Brief v1 結果
        brief_v2: Brief v2 結果
        brief_v3: Brief v3 結果
        glossary: 詞彙表
        target_language: 目標語言（Locale Code）
        prompt_config: CSV 載入的 prompt 設定
        progress_callback: 進度回調
        batch_size: 批次大小（None = 不使用批次處理）
    
    Returns:
        翻譯結果列表（對應順序）
    """
    from .pipeline import stage3_translate, parse_pack_from_reasons

    results: list[str] = []
    total = len(subs)
    prev_zh: list[str] = []
    
    # 計算批次大小（如果未指定）
    if batch_size is None:
        batch_size = total  # 預設：不使用批次處理
    
    # 批次處理
    processed = 0
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        
        # 處理批次
        for i in range(batch_start, batch_end):
            sub = subs[i]
            line_en = en_lines[i]
            
            # 選擇最佳 brief
            best_brief = select_best_brief(brief_v1, brief_v2, brief_v3, i)
            if not best_brief:
                meaning_en = ""
                pack = None
            else:
                meaning_en = best_brief.translation_brief
                pack = parse_pack_from_reasons(best_brief.reasons)
            sub_id = f"item_{i}"
            log_label = f"item {i + 1}/{total}" if log_lines else ""
            zh = stage3_translate(
                translate_model,
                line_en,
                meaning_en,
                prev_zh,
                glossary,
                target_language,
                prompt_config=prompt_config,
                log_lines=log_lines,
                log_label=log_label,
                sub_id=sub_id,
                pack=pack,
                use_out_template=(pack is not None),
            )
            # Fallback when postprocess returns "" (e.g. "completed declaration" junk)
            if not (zh and zh.strip()):
                zh = line_en
            results.append(zh)
            prev_zh.append(zh)
            sub.text = zh  # 更新 SRT
            
            processed += 1
            if progress_callback:
                progress_callback(processed / total, f"Translate: {processed}/{total}")
        
        # 批次完成後，可以選擇性地清理 prev_zh（但需要保留最後幾條用於上下文）
        # 為了保持上下文連貫性，我們保留所有 prev_zh
    
    return results
