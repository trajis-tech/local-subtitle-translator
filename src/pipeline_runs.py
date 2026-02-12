"""
重構後的 Run 函式，使用 SubtitleItem 和 sub_id 對齊。

每個 run 函式：
1. 接受 dict[sub_id, SubtitleItem] 作為輸入
2. 返回更新後的 dict[sub_id, SubtitleItem]
3. 內部載入和釋放模型（使用互斥鎖）
4. 執行一致性檢查
5. 使用激進的批次推理和並行處理以提高 GPU 利用率
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional
import gc
import json
import re
import threading
import time

from .subtitle_item import (
    SubtitleItem, AudioMeta, BriefMeta,
    verify_subtitle_items_consistency,
    generate_sub_id,
)
from .model_lock import get_model_mutex
from .srt_utils import sub_midpoints_ms
from .model_path_utils import (
    resolve_reason_model_path,
    resolve_translate_model_path,
    resolve_vision_paths,
)
from .resource_utils import detect_gpu


# ----- Group 建立器（Run F group translate / fallback 用） -----

@dataclass
class Group:
    """一組相鄰字幕，可一併翻譯以維持語序。"""
    group_id: str  # 通常為本 group 第一段 sub_id
    segments: list[dict]  # 每項: {id, start_ms, end_ms, duration_ms, en_text}
    group_span_ms: Optional[float] = None  # 可選：第一段 start 到最後一段 end 的時長


# 下一句像「延續」：小寫開頭 / 常見連接詞或主詞 / to|that
_CONTINUATION_FIRST_WORD = re.compile(
    r"^(and|but|so|or|because|he|she|it|they|we|to|that)\b",
    re.IGNORECASE,
)


def _segment_from_item(sub_id: str, item: SubtitleItem) -> dict:
    duration = max(0.0, (item.end_ms or 0) - (item.start_ms or 0))
    return {
        "id": sub_id,
        "start_ms": float(item.start_ms),
        "end_ms": float(item.end_ms),
        "duration_ms": duration,
        "en_text": (item.text_clean or item.text_raw or "").strip(),
    }


def _should_merge_adjacent(
    prev_seg: dict,
    next_seg: dict,
    gap_ms: float,
) -> bool:
    """只考慮相鄰、gap<250、前句非句尾、下一句像延續或任一段為短殘片時合併。"""
    if gap_ms >= 250:
        return False
    prev_text = (prev_seg.get("en_text") or "").strip()
    next_text = (next_seg.get("en_text") or "").strip()
    if not prev_text or not next_text:
        return False
    # 前句以句號/問號/驚嘆號結尾 → 不併
    last_char = prev_text[-1]
    if last_char in ".?!":
        return False
    # 下一句開頭像延續：小寫字母 或 常見連接詞/主詞/to/that
    next_stripped = next_text.lstrip()
    if next_stripped and next_stripped[0].islower():
        return True
    first_word = next_stripped.split(None, 1)[0] if next_stripped else ""
    if _CONTINUATION_FIRST_WORD.match(first_word):
        return True
    # 任一段太短且像殘片（< 20 字且非句尾）
    def looks_fragment(t: str) -> bool:
        if len(t) >= 20:
            return False
        return t.rstrip()[-1] not in ".?!" if t else False
    if looks_fragment(prev_text) or looks_fragment(next_text):
        return True
    return False


def build_sentence_groups(items: dict[str, SubtitleItem]) -> list[Group]:
    """
    將 items 依相鄰關係合併成 Group 列表；每組至少 1 段，保留原本 sub_id 順序。
    合併規則：相鄰、gap<250ms、前句非句尾、下一句像延續或任一段為短殘片。
    """
    if not items:
        return []
    sorted_pairs = sorted(items.items(), key=lambda x: (x[1].start_ms, x[1].end_ms))
    segments_list: list[dict] = []
    for sub_id, item in sorted_pairs:
        segments_list.append(_segment_from_item(sub_id, item))
    if not segments_list:
        return []
    # 合併相鄰：從左到右，能併就併（以當前 group 最後一段與下一段比較）
    groups: list[list[dict]] = []
    current = [segments_list[0]]
    for i in range(1, len(segments_list)):
        prev_seg = current[-1]
        next_seg = segments_list[i]
        gap_ms = next_seg["start_ms"] - prev_seg["end_ms"]
        if _should_merge_adjacent(prev_seg, next_seg, gap_ms):
            current.append(next_seg)
        else:
            groups.append(current)
            current = [next_seg]
    groups.append(current)
    # 轉成 Group
    result: list[Group] = []
    for segs in groups:
        if not segs:
            continue
        first_id = segs[0]["id"]
        start_ms = segs[0]["start_ms"]
        end_ms = segs[-1]["end_ms"]
        span = end_ms - start_ms if end_ms > start_ms else None
        result.append(Group(group_id=first_id, segments=segs, group_span_ms=span))
    return result


def run_audio(
    items: dict[str, SubtitleItem],
    video_path: str,
    work_dir: Path,
    cfg: Any,  # AppConfig
    log_lines: Optional[list[str]] = None,
    progress_callback: Optional[Any] = None,
) -> dict[str, SubtitleItem]:
    """
    Run A: 音訊分析。
    
    內部載入和釋放音訊模型，使用互斥鎖保護。
    
    Args:
        items: dict[sub_id, SubtitleItem]
        video_path: 影片路徑
        work_dir: 工作目錄
        cfg: AppConfig
        log_lines: 可選的 log 列表
        progress_callback: 進度回調
    
    Returns:
        更新後的 dict[sub_id, SubtitleItem]（已填充 audio_meta）
    """
    from .audio_model import (
        load_audio_model,
        run_audio_env_check,
        analyze_audio_for_item,
        raw_emotion_to_meta,
        AudioLoadResult,
    )
    import shutil

    expected_sub_ids = set(items.keys())
    model_mutex = get_model_mutex()
    total = len(items)
    extract_mode = getattr(cfg.audio, "extract_mode", "full_then_slice") or "full_then_slice"
    if log_lines:
        log_lines.append(f"[Run A] Preparing to process {total} subtitle items...")
    if progress_callback:
        progress_callback(0.0, f"Loading audio model... ({total} items)")

    # 載入失敗時寫入每筆與 JSONL 的具體原因（禁止靜默 fallback）
    load_failure_reason: str = ""

    with model_mutex.hold_model("audio"):
        audio_model = None
        try:
            audio_model_id = getattr(cfg.audio, "model_id_or_path", None) or getattr(cfg.audio, "model_dir", "")
            if log_lines:
                run_audio_env_check(audio_model_id, log_lines)
            if log_lines:
                log_lines.append("[Run A] Loading audio model (Wav2Vec2 / Hugging Face)...")
            if cfg.audio.enabled:
                load_result: AudioLoadResult = load_audio_model(cfg)
                if load_result.success and load_result.model is not None:
                    # 檢查 ffmpeg：無 ffmpeg 則無法提取音訊，視為不可用
                    if not shutil.which("ffmpeg"):
                        load_failure_reason = "FFmpeg missing: cannot extract audio"
                        if log_lines:
                            log_lines.append(f"[Run A] Audio model load failed: {load_failure_reason}")
                        try:
                            del load_result.model
                        except Exception:
                            pass
                        audio_model = None
                    else:
                        audio_model = load_result.model
                        if log_lines:
                            log_lines.append(f"[Run A] ✓ Audio model loaded successfully ({load_result.backend})")
                else:
                    load_failure_reason = load_result.reason
                    if log_lines:
                        log_lines.append(f"[Run A] Audio model load failed: {load_failure_reason}")
                    audio_model = None
            else:
                load_failure_reason = "Audio analysis disabled (cfg.audio.enabled=False)"
                if log_lines:
                    log_lines.append(f"[Run A] ⚠️ {load_failure_reason}")
                audio_model = None

            # Fail-fast：enabled 且音訊模型未載入時，預設中止；僅 allow_fail=True 才降級輸出空 tags
            if cfg.audio.enabled and audio_model is None and load_failure_reason:
                if not getattr(cfg.audio, "allow_fail", False):
                    if log_lines:
                        log_lines.append(f"[Run A] Full failure reason: {load_failure_reason}")
                    raise RuntimeError(
                        f"Run A audio model load failed, pipeline aborted. Reason: {load_failure_reason}\n"
                        "To continue on load failure (output empty audio_tags), set audio.allow_fail=true in config."
                    )
                if log_lines:
                    for line in [
                        "",
                        "=" * 60,
                        "AUDIO DISABLED DUE TO LOAD FAILURE",
                        load_failure_reason,
                        "Output will have empty audio_tags.",
                        "=" * 60,
                    ]:
                        log_lines.append(f"[Run A] {line}")

            sorted_items = sorted(items.items(), key=lambda x: x[1].start_ms)

            full_audio_path = work_dir / "full_audio.wav"
            need_extract_full = extract_mode == "full_then_slice" and audio_model is not None

            if full_audio_path.exists() and need_extract_full:
                if log_lines:
                    log_lines.append(f"[Run A] Found existing extracted audio file: {full_audio_path}")
                need_extract_full = False

            if audio_model is not None and need_extract_full:
                if log_lines:
                    log_lines.append("[Run A] Starting one-time full audio extraction (this may take a few minutes, but only needs to run once)...")
                if progress_callback:
                    progress_callback(0.05, "Extracting full audio (one-time operation, please wait)...")
                try:
                    audio_model.extract_full_audio(
                        video_path,
                        str(full_audio_path),
                        progress_callback=lambda p, d: progress_callback(
                            0.05 + p * 0.15, f"Extracting full audio: {d}"
                        ) if progress_callback else None,
                    )
                    if log_lines:
                        log_lines.append(f"[Run A] ✓ Full audio extraction completed: {full_audio_path}")
                except Exception as e:
                    if log_lines:
                        log_lines.append(f"[Run A] ⚠️ Full audio extraction failed: {e}, falling back to per-item extraction mode")
                    try:
                        if full_audio_path.exists():
                            full_audio_path.unlink()
                    except OSError:
                        pass
                    full_audio_path = None  # 回退到舊模式

            # 處理每個字幕項目
            processed = 0
            use_fast_mode = (
                extract_mode == "full_then_slice"
                and full_audio_path is not None
                and full_audio_path.exists()
            )
            if log_lines:
                if use_fast_mode:
                    log_lines.append(f"[Run A] Starting to extract segments from pre-extracted audio (fast mode)...")
                else:
                    log_lines.append(f"[Run A] Starting to extract audio segments from video (slow mode)...")
            progress_start = 0.20 if use_fast_mode else 0.0
            if progress_callback:
                progress_callback(progress_start, f"Processing audio segments... (0/{total})")

            batch_size = max(1, min(int(getattr(cfg.audio, "batch_size", 16) or 16), 1024))
            if log_lines and batch_size > 1:
                log_lines.append(f"[Run A] Using batch inference (batch_size={batch_size}) for better GPU utilization")

            for chunk_start in range(0, total, batch_size):
                chunk = sorted_items[chunk_start : chunk_start + batch_size]
                if audio_model is None:
                    for _sub_id, item in chunk:
                        item.audio_meta = AudioMeta(
                            emotion="",
                            tone="",
                            intensity="",
                            speaking_style="",
                            audio_reason=load_failure_reason or "Audio model not loaded",
                        )
                else:
                    paths: list[str | None] = []
                    for idx_in_chunk, (sub_id, item) in enumerate(chunk):
                        idx = chunk_start + idx_in_chunk
                        if log_lines and idx % max(1, total // 20) == 0:
                            log_lines.append(f"[Run A] Processing audio segment {idx+1}/{total}: {item.start_ms//1000:.1f}s-{item.end_ms//1000:.1f}s")
                        try:
                            if use_fast_mode:
                                path = audio_model.extract_segment_from_audio(
                                    str(full_audio_path),
                                    item.start_ms,
                                    item.end_ms,
                                )
                            else:
                                path = audio_model.extract_audio_segment(
                                    video_path,
                                    item.start_ms,
                                    item.end_ms,
                                )
                            paths.append(path)
                        except Exception as e:
                            if log_lines:
                                log_lines.append(f"[Run A] ⚠️ Item {idx+1} audio extract failed: {e}")
                            paths.append(None)
                            item.audio_meta = AudioMeta(
                                emotion="",
                                tone="",
                                intensity="",
                                speaking_style="",
                                audio_reason=f"Audio extraction failed: {e}",
                            )
                    valid_paths = [p for p in paths if p is not None]
                    if valid_paths:
                        if log_lines and (chunk_start == 0 or (chunk_start + len(chunk)) % max(1, total // 20) == 0):
                            log_lines.append(f"[Run A] Analyzing emotion batch {chunk_start+1}-{chunk_start+len(chunk)}/{total}...")
                        raw_results = audio_model.analyze_emotion_batch(valid_paths)
                        valid_idx = 0
                        for i, (sub_id, item) in enumerate(chunk):
                            if paths[i] is not None:
                                meta = raw_emotion_to_meta(raw_results[valid_idx])
                                item.audio_meta = AudioMeta(
                                    emotion=meta.get("emotion", ""),
                                    tone=meta.get("tone", ""),
                                    intensity=meta.get("intensity", ""),
                                    speaking_style=meta.get("speaking_style", ""),
                                    audio_reason=meta.get("audio_reason", ""),
                                )
                                try:
                                    Path(paths[i]).unlink()
                                except OSError:
                                    pass
                                valid_idx += 1
                processed += len(chunk)
                if progress_callback:
                    if use_fast_mode:
                        progress = progress_start + (processed / total) * (1.0 - progress_start)
                    else:
                        progress = processed / total
                    progress_callback(
                        progress,
                        f"Audio analysis: {processed}/{total} (time: {chunk[-1][1].start_ms//1000:.1f}s-{chunk[-1][1].end_ms//1000:.1f}s)"
                    )
                if log_lines and (processed % max(1, total // 20) == 0 or processed % 5 == 0):
                    elapsed_estimate = ""
                    if processed > 0:
                        remaining_items = total - processed
                        remaining_seconds = remaining_items * 1.0
                        if remaining_seconds > 60:
                            elapsed_estimate = f" (estimated remaining: {remaining_seconds//60:.0f} minutes)"
                        else:
                            elapsed_estimate = f" (estimated remaining: {remaining_seconds:.0f} seconds)"
                    log_lines.append(f"[Run A] Progress: {processed}/{total} ({processed*100//total}%){elapsed_estimate}")

            # 寫入 JSONL（使用 sub_id 作為 key）；第一行寫入 _run_meta 方便驗收
            jsonl_path = work_dir / "audio_tags.jsonl"
            work_dir.mkdir(parents=True, exist_ok=True)
            if log_lines:
                log_lines.append(f"[Run A] Writing results to {jsonl_path}...")
            with jsonl_path.open("w", encoding="utf-8") as f:
                meta_line = {
                    "_run_meta": {
                        "run": "Run A",
                        "audio_load_success": audio_model is not None,
                        "audio_load_reason": load_failure_reason if audio_model is None else "",
                    }
                }
                f.write(json.dumps(meta_line, ensure_ascii=False) + "\n")
                for sub_id, item in sorted_items:
                    if item.audio_meta:
                        line = {
                            "sub_id": sub_id,
                            "start_ms": item.start_ms,
                            "end_ms": item.end_ms,
                            "audio_tags": {
                                "emotion": item.audio_meta.emotion,
                                "tone": item.audio_meta.tone,
                                "intensity": item.audio_meta.intensity,
                                "speaking_style": item.audio_meta.speaking_style,
                            },
                            "audio_reason": item.audio_meta.audio_reason,
                        }
                        f.write(json.dumps(line, ensure_ascii=False) + "\n")
            verify_subtitle_items_consistency(items, expected_sub_ids, "Run A")
            if log_lines:
                log_lines.append(f"[Run A] Completed: {len(items)} subtitle items")
        finally:
            if audio_model is not None:
                del audio_model
                gc.collect()
            if log_lines:
                log_lines.append("[Run A] Audio model unloaded")

    return items


def _infer_tone_from_emotion(emotion: str) -> str:
    """根據情緒推斷語氣"""
    emotion_lower = emotion.lower()
    if "angry" in emotion_lower or "frustrated" in emotion_lower:
        return "aggressive"
    elif "sad" in emotion_lower or "sorrowful" in emotion_lower:
        return "melancholic"
    elif "happy" in emotion_lower or "joyful" in emotion_lower:
        return "cheerful"
    elif "fear" in emotion_lower or "anxious" in emotion_lower:
        return "nervous"
    elif "surprised" in emotion_lower:
        return "excited"
    else:
        return "neutral"


def _infer_intensity_from_confidence(confidence: float) -> str:
    """根據信心分數推斷強度"""
    if confidence >= 0.8:
        return "high"
    elif confidence >= 0.5:
        return "medium"
    else:
        return "low"


def _infer_speaking_style_from_emotion(emotion: str) -> str:
    """根據情緒推斷說話方式"""
    emotion_lower = emotion.lower()
    if "angry" in emotion_lower or "excited" in emotion_lower:
        return "fast"
    elif "sad" in emotion_lower or "calm" in emotion_lower:
        return "slow"
    elif "whisper" in emotion_lower or "quiet" in emotion_lower:
        return "whispering"
    else:
        return "normal"


def run_brief_text(
    items: dict[str, SubtitleItem],
    work_dir: Path,
    cfg: Any,  # AppConfig
    prompt_config: Any,  # ModelPromptConfig
    version: str = "v1",  # v1, v2, v3
    vision_hint_map: Optional[dict[str, str]] = None,  # {sub_id: vision_desc}
    log_lines: Optional[list[str]] = None,
    progress_callback: Optional[Any] = None,
    batch_size: Optional[int] = None,
    target_language: str = "zh-TW",
) -> dict[str, SubtitleItem]:
    """
    Run B/C/D: Brief 生成（文字推理模型）。
    
    內部載入和釋放主推理模型，使用互斥鎖保護。
    
    Args:
        items: dict[sub_id, SubtitleItem]
        work_dir: 工作目錄
        cfg: AppConfig
        prompt_config: ModelPromptConfig from CSV
        version: 版本標記（v1, v2, v3）
        vision_hint_map: 可選的視覺提示映射 {sub_id: vision_desc}
        log_lines: 可選的 log 列表
        progress_callback: 進度回調
        batch_size: 批次大小
        target_language: 目標語言 locale（如 zh-TW, ja-JP），供 Stage2 輸出 meaning_tl/draft_tl 等
    
    Returns:
        更新後的 dict[sub_id, SubtitleItem]（已填充 brief_v1/v2/v3）
    """
    from .models import TextModel
    from .pipeline import stage2_reason_and_score, stage2_reason_and_score_batch, _is_likely_target_language
    from .safe_infer import NeedReload

    expected_sub_ids = set(items.keys())
    request_batch_size = max(1, min(32, getattr(cfg.pipeline, "request_batch_size", 24) or 24))
    model_mutex = get_model_mutex()

    with model_mutex.hold_model("reason"):
        reason_model = None
        try:
            if log_lines:
                log_lines.append(f"[Run Brief {version}] Loading main reasoning model...")
            gpu_info = detect_gpu()
            n_ctx_reason = cfg.llama_cpp.n_ctx_reason
            n_gpu_layers_reason = cfg.llama_cpp.n_gpu_layers_reason
            if log_lines and gpu_info.get("vram_mb"):
                log_lines.append(f"[Run Brief {version}] VRAM {gpu_info['vram_mb']} MB; using config n_ctx={n_ctx_reason}, n_gpu_layers={n_gpu_layers_reason} (OOM → safe_chat fallback)")
            reason_path = resolve_reason_model_path(cfg)
            reason_chat_format = prompt_config.chat_format if prompt_config else "chatml"
            reason_model = TextModel(
                model_path=str(reason_path),
                chat_format=reason_chat_format,
                n_ctx=n_ctx_reason,
                n_gpu_layers=n_gpu_layers_reason,
                n_threads=cfg.llama_cpp.n_threads,
            )
            if log_lines:
                log_lines.append(f"[Run Brief {version}] Main reasoning model loaded successfully (n_ctx={n_ctx_reason}, n_gpu_layers={n_gpu_layers_reason})")
                if n_gpu_layers_reason > 0:
                    log_lines.append(f"[Run Brief {version}] GPU acceleration enabled (first {n_gpu_layers_reason} layers on GPU)")
                elif n_gpu_layers_reason == -1:
                    log_lines.append(f"[Run Brief {version}] GPU acceleration enabled (all layers on GPU)")
                else:
                    log_lines.append(f"[Run Brief {version}] ⚠️ GPU acceleration disabled (n_gpu_layers={n_gpu_layers_reason}), will use CPU (slower)")

            sorted_items = sorted(items.items(), key=lambda x: x[1].start_ms)
            total = len(sorted_items)

            def apply_brief_results(chunk: list, results_map: dict) -> None:
                for sub_id, item in chunk:
                    if sub_id not in results_map:
                        continue
                    s2_result = results_map[sub_id]
                    meaning_en = s2_result.meaning_en or ""
                    if meaning_en and _is_likely_target_language(meaning_en, target_language):
                        meaning_en = (item.text_clean or "").strip() or meaning_en
                    brief_meta = BriefMeta(
                        translation_brief=meaning_en,
                        reasons=s2_result.notes,
                        version=version,
                        need_vision=getattr(s2_result, "need_vision", None),
                        need_multi_frame_vision=getattr(s2_result, "need_multi_frame_vision", None),
                        need_more_context=getattr(s2_result, "need_more_context", None),
                    )
                    if version == "v1":
                        item.brief_v1 = brief_meta
                    elif version == "v2":
                        item.brief_v2 = brief_meta
                    elif version == "v3":
                        item.brief_v3 = brief_meta

            def process_one_batch(chunk: list, reason_m, reason_path, reason_fmt, n_ctx, n_gpu) -> dict:
                if not chunk:
                    return {}
                try:
                    batch_result = stage2_reason_and_score_batch(
                        reason_m, chunk, sorted_items,
                        target_language=target_language, prompt_config=prompt_config,
                        vision_hint_map=vision_hint_map, brief_version=version,
                        log_lines=log_lines, log_label=f"Run Brief {version}",
                    )
                except NeedReload:
                    raise
                except Exception as e:
                    if log_lines:
                        log_lines.append(f"[Run Brief {version}] Batch exception (will retry with smaller batch or single): {e}")
                    batch_result = None
                if batch_result is not None:
                    return batch_result
                if len(chunk) == 1:
                    sub_id, item = chunk[0]
                    prev_items = [x[1] for x in sorted_items if x[1].start_ms < item.start_ms]
                    next_items = [x[1] for x in sorted_items if x[1].start_ms > item.start_ms]
                    prev_lines = [prev_items[-1].text_clean] if prev_items else []
                    next_lines = [next_items[0].text_clean] if next_items else []
                    audio_hint = item.get_audio_hint()
                    if audio_hint == "(none)":
                        audio_hint = None
                    visual_hint = vision_hint_map.get(sub_id) if vision_hint_map else None
                    try:
                        s2 = stage2_reason_and_score(
                            reason_m, item.text_clean, prev_lines, next_lines,
                            visual_hint=visual_hint, audio_hint=audio_hint,
                            prompt_config=prompt_config, target_language=target_language,
                            brief_version=version,
                            log_lines=log_lines, log_label=f"Run Brief {version}",
                        )
                        return {sub_id: s2}
                    except Exception as e:
                        if log_lines:
                            log_lines.append(f"[Run Brief {version}] ⚠️ Item {sub_id[:8]}... fallback failed: {e}")
                        fake = type("Stage2Result", (), {"meaning_en": "", "notes": f"Error: {e}", "need_vision": None, "need_multi_frame_vision": None, "need_more_context": False})()
                        return {sub_id: fake}
                mid = len(chunk) // 2
                left = process_one_batch(chunk[:mid], reason_m, reason_path, reason_fmt, n_ctx, n_gpu)
                right = process_one_batch(chunk[mid:], reason_m, reason_path, reason_fmt, n_ctx, n_gpu)
                return {**left, **right}

            if log_lines:
                log_lines.append(f"[Run Brief {version}] request_batch_size={request_batch_size}, total items={total}")
                log_lines.append(f"[Run Brief {version}] Multi-item in one request (batch brief); binary split on parse failure")
            processed = 0
            # 單一 brief：一律寫入 brief_work.jsonl（每階段更新同一份；snapshot 由 app 在階段前 copy 至 brief_v1/v2/v3/v4.jsonl）
            jsonl_path = work_dir / "brief_work.jsonl"
            work_dir.mkdir(parents=True, exist_ok=True)
            with jsonl_path.open("w", encoding="utf-8") as f:
                for batch_start in range(0, total, request_batch_size):
                    batch_end = min(batch_start + request_batch_size, total)
                    chunk = sorted_items[batch_start:batch_end]
                    if log_lines:
                        log_lines.append(f"[Run Brief {version}] Request batch {batch_start//request_batch_size + 1}/{(total-1)//request_batch_size + 1} ({batch_start+1}-{batch_end}/{total}) items in one request...")
                    while True:
                        try:
                            results_map = process_one_batch(
                                chunk, reason_model, reason_path, reason_chat_format, n_ctx_reason, n_gpu_layers_reason,
                            )
                            break
                        except NeedReload as e:
                            if log_lines:
                                log_lines.append(f"[Run Brief {version}] NeedReload {e.level}: {e.params}, reloading reason model...")
                            del reason_model
                            gc.collect()
                            n_ctx_reason = e.params.get("n_ctx", n_ctx_reason)
                            n_gpu_layers_reason = e.params.get("n_gpu_layers", n_gpu_layers_reason)
                            reason_model = TextModel(
                                model_path=str(reason_path),
                                chat_format=reason_chat_format,
                                n_ctx=n_ctx_reason,
                                n_gpu_layers=n_gpu_layers_reason,
                                n_threads=cfg.llama_cpp.n_threads,
                            )
                            continue
                    apply_brief_results(chunk, results_map)
                    processed += len(chunk)
                    if progress_callback:
                        progress_callback(processed / total, f"Brief {version}: {processed}/{total}")
                    for sub_id, item in chunk:
                        brief = item.get_best_brief()
                        if brief:
                            line = {
                                "sub_id": sub_id,
                                "start_ms": item.start_ms,
                                "end_ms": item.end_ms,
                                "translation_brief": brief.translation_brief,
                                "reasons": brief.reasons,
                                "version": brief.version,
                                "need_vision": getattr(brief, "need_vision", None),
                                "need_multi_frame_vision": getattr(brief, "need_multi_frame_vision", None),
                                "need_more_context": getattr(brief, "need_more_context", None),
                            }
                            f.write(json.dumps(line, ensure_ascii=False) + "\n")
            verify_subtitle_items_consistency(items, expected_sub_ids, f"Run Brief {version}")
            if log_lines:
                log_lines.append(f"[Run Brief {version}] Completed: {len(items)} subtitle items")
        finally:
            if reason_model is not None:
                del reason_model
                gc.collect()
            if log_lines:
                log_lines.append(f"[Run Brief {version}] Main reasoning model unloaded")

    return items


def _write_current_brief(work_dir: Path, items: dict[str, SubtitleItem]) -> None:
    """將當前每個 item 的 get_best_brief() 寫入 work_dir/brief_work.jsonl。"""
    work_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = work_dir / "brief_work.jsonl"
    sorted_items = sorted(items.items(), key=lambda x: x[1].start_ms)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for sub_id, item in sorted_items:
            brief = item.get_best_brief()
            if brief:
                line = {
                    "sub_id": sub_id,
                    "start_ms": item.start_ms,
                    "end_ms": item.end_ms,
                    "translation_brief": brief.translation_brief,
                    "reasons": brief.reasons,
                    "version": brief.version,
                    "need_vision": getattr(brief, "need_vision", None),
                    "need_multi_frame_vision": getattr(brief, "need_multi_frame_vision", None),
                    "need_more_context": getattr(brief, "need_more_context", None),
                }
                f.write(json.dumps(line, ensure_ascii=False) + "\n")


def run_context_expansion(
    items: dict[str, SubtitleItem],
    work_dir: Path,
    cfg: Any,
    prompt_config: Any,
    target_language: str = "zh-TW",
    log_lines: Optional[list[str]] = None,
    progress_callback: Optional[Any] = None,
) -> dict[str, SubtitleItem]:
    """
    Run E：上下文擴充。僅對 need_more_context 為 True 的 item 用 prev-3/next-3 再跑 stage2 更新 brief，寫回 brief_work.jsonl。
    """
    from .models import TextModel
    from .pipeline import stage2_reason_and_score

    model_mutex = get_model_mutex()
    sorted_items = sorted(items.items(), key=lambda x: x[1].start_ms)
    id_to_index = {sid: i for i, (sid, _) in enumerate(sorted_items)}
    need_more = [
        (sid, it) for sid, it in sorted_items
        if (it.get_best_brief() and getattr(it.get_best_brief(), "need_more_context", None) is True)
    ]
    if not need_more:
        if log_lines:
            log_lines.append("[Run E] No items with need_more_context, skipping context expansion.")
        return items

    if log_lines:
        log_lines.append(f"[Run E] Context expansion: {len(need_more)} items with need_more_context (prev-3/next-3 stage2).")
    with model_mutex.hold_model("reason"):
        reason_model = None
        try:
            reason_path = resolve_reason_model_path(cfg)
            reason_model = TextModel(
                model_path=str(reason_path),
                chat_format=prompt_config.chat_format if prompt_config else "chatml",
                n_ctx=cfg.llama_cpp.n_ctx_reason,
                n_gpu_layers=cfg.llama_cpp.n_gpu_layers_reason,
                n_threads=cfg.llama_cpp.n_threads,
            )
            total = len(need_more)
            for idx, (sid, it) in enumerate(need_more):
                i = id_to_index.get(sid, 0)
                line_en = (it.text_clean or it.text_raw or "").strip()
                prev_3 = [
                    (sorted_items[j][1].text_clean or sorted_items[j][1].text_raw or "").strip()
                    for j in range(max(0, i - 3), i)
                    if (sorted_items[j][1].text_clean or sorted_items[j][1].text_raw or "").strip()
                ]
                next_3 = [
                    (sorted_items[j][1].text_clean or sorted_items[j][1].text_raw or "").strip()
                    for j in range(i + 1, min(len(sorted_items), i + 4))
                    if (sorted_items[j][1].text_clean or sorted_items[j][1].text_raw or "").strip()
                ]
                audio_hint = it.get_audio_hint() if it else "(none)"
                s2 = stage2_reason_and_score(
                    reason_model,
                    line_en,
                    prev_3,
                    next_3,
                    visual_hint=None,
                    audio_hint=audio_hint or None,
                    prompt_config=prompt_config,
                    target_language=target_language,
                    brief_version="v3",
                    log_lines=log_lines,
                    log_label="Run E context expansion",
                )
                brief_meta = BriefMeta(
                    translation_brief=getattr(s2, "meaning_en", "") or "",
                    reasons=getattr(s2, "notes", "") or "",
                    version="v3",
                    need_vision=getattr(s2, "need_vision", None),
                    need_multi_frame_vision=getattr(s2, "need_multi_frame_vision", None),
                    need_more_context=getattr(s2, "need_more_context", None),
                )
                it.brief_v3 = brief_meta
                if progress_callback:
                    progress_callback((idx + 1) / total, f"Run E expansion: {idx + 1}/{total}")
            _write_current_brief(work_dir, items)
            if log_lines:
                log_lines.append(f"[Run E] Context expansion done, brief.jsonl updated.")
        finally:
            if reason_model is not None:
                del reason_model
                gc.collect()
    return items


def run_vision_single(
    items: dict[str, SubtitleItem],
    video_path: str,
    work_dir: Path,
    cfg: Any,  # AppConfig
    target_sub_ids: list[str],
    log_lines: Optional[list[str]] = None,
    progress_callback: Optional[Any] = None,
    preview_callback: Optional[Callable[[bytes], None]] = None,
) -> dict[str, SubtitleItem]:
    """
    Run C: 單張影像分析（single-frame）。

    僅對 target_sub_ids 中的字幕項目進行分析，結果寫入 vision_desc_1。
    """
    from .local_vision_model import LocalVisionModel
    from .video import open_video, get_frame_at_ms, encode_jpg_bytes

    expected_sub_ids = set(items.keys())
    model_mutex = get_model_mutex()

    with model_mutex.hold_model("vision"):
        vision_model: Optional[LocalVisionModel] = None
        try:
            if log_lines:
                log_lines.append("[Run C] Loading vision model...")
            text_model_path, mmproj_path, model_type = resolve_vision_paths(cfg)
            if not text_model_path or not text_model_path.exists() or not mmproj_path or not mmproj_path.exists():
                if log_lines:
                    log_lines.append(f"[Run C] ⚠️ Vision model or mmproj not found, skipping Run C")
                return items
            vision_model = LocalVisionModel(
                model_path=str(text_model_path),
                clip_model_path=str(mmproj_path),
                model_type=None,
                n_ctx=None,
                n_threads=cfg.llama_cpp.n_threads,
            )
            if log_lines:
                log_lines.append("[Run C] Vision model loaded successfully")
            work_dir.mkdir(parents=True, exist_ok=True)
            cap = open_video(video_path)
            try:
                total = len(target_sub_ids)
                for idx_in_target, sub_id in enumerate(target_sub_ids):
                    item = items.get(sub_id)
                    if not item:
                        continue
                    start_ms = item.start_ms
                    end_ms = item.end_ms
                    mid_ms = (start_ms + end_ms) / 2.0
                    if end_ms <= start_ms:
                        item.vision_desc_1 = "[VisionError] Invalid time range"
                    else:
                        frame = get_frame_at_ms(cap, mid_ms)
                        if frame is None:
                            item.vision_desc_1 = "[VisionError] Failed to extract frame"
                        else:
                            try:
                                img_bytes = encode_jpg_bytes(frame)
                                if preview_callback:
                                    preview_callback(img_bytes)
                                desc = vision_model.describe_with_grounding(
                                    img_bytes, item.text_clean or item.text_raw
                                )
                                item.vision_desc_1 = desc
                            except Exception as e:
                                item.vision_desc_1 = f"[VisionError] {e}"
                    if progress_callback:
                        progress_callback(
                            (idx_in_target + 1) / max(total, 1),
                            f"Vision (1-frame): {idx_in_target+1}/{total}",
                        )
            finally:
                cap.release()
            jsonl_path = work_dir / "vision_1frame.jsonl"
            with jsonl_path.open("w", encoding="utf-8") as f:
                for sub_id, item in items.items():
                    if item.vision_desc_1:
                        line = {
                            "sub_id": sub_id,
                            "start_ms": item.start_ms,
                            "end_ms": item.end_ms,
                            "vision_desc": item.vision_desc_1,
                            "frame_count": 1,
                        }
                        f.write(json.dumps(line, ensure_ascii=False) + "\n")
            verify_subtitle_items_consistency(items, expected_sub_ids, "Run C (vision_1frame)")
            if log_lines:
                log_lines.append(f"[Run C] Completed: {len(target_sub_ids)} subtitle items")
        finally:
            if vision_model is not None:
                del vision_model
                gc.collect()
            if log_lines:
                log_lines.append("[Run C] Vision model unloaded")

    return items


def run_vision_multi(
    items: dict[str, SubtitleItem],
    video_path: str,
    work_dir: Path,
    cfg: Any,  # AppConfig
    target_sub_ids: list[str],
    n_frames: int,
    log_lines: Optional[list[str]] = None,
    progress_callback: Optional[Any] = None,
    preview_callback: Optional[Callable[[bytes], None]] = None,
) -> dict[str, SubtitleItem]:
    """
    Run D: 多張影像分析（multi-frame）。

    僅對 target_sub_ids 中的字幕項目進行分析，結果寫入 vision_desc_n。
    """
    from .local_vision_model import LocalVisionModel
    from .video import open_video, get_frame_at_ms, encode_jpg_bytes

    expected_sub_ids = set(items.keys())
    model_mutex = get_model_mutex()

    with model_mutex.hold_model("vision"):
        vision_model: Optional[LocalVisionModel] = None
        try:
            if log_lines:
                log_lines.append("[Run D] Loading vision model...")
            text_model_path, mmproj_path, model_type = resolve_vision_paths(cfg)
            if not text_model_path or not text_model_path.exists() or not mmproj_path or not mmproj_path.exists():
                if log_lines:
                    log_lines.append(f"[Run D] ⚠️ Vision model or mmproj not found, skipping Run D")
                return items
            vision_model = LocalVisionModel(
                model_path=str(text_model_path),
                clip_model_path=str(mmproj_path),
                model_type=model_type,
                n_ctx=None,
                n_threads=cfg.llama_cpp.n_threads,
            )
            if log_lines:
                log_lines.append("[Run D] Vision model loaded successfully")
            work_dir.mkdir(parents=True, exist_ok=True)
            cap = open_video(video_path)
            try:
                total = len(target_sub_ids)
                for idx_in_target, sub_id in enumerate(target_sub_ids):
                    item = items.get(sub_id)
                    if not item:
                        continue
                    start_ms = item.start_ms
                    end_ms = item.end_ms
                    duration = end_ms - start_ms
                    if duration <= 0:
                        item.vision_desc_n = "[VisionError] Invalid time range"
                    else:
                        frame_descs: list[str] = []
                        for frame_idx in range(max(n_frames, 1)):
                            t = start_ms + (duration * (frame_idx + 0.5) / max(n_frames, 1))
                            frame = get_frame_at_ms(cap, t)
                            if frame is None:
                                continue
                            try:
                                img_bytes = encode_jpg_bytes(frame)
                                if preview_callback:
                                    preview_callback(img_bytes)
                                desc = vision_model.describe_with_grounding(
                                    img_bytes, item.text_clean or item.text_raw
                                )
                                frame_descs.append(f"[Frame {frame_idx+1}/{n_frames}] {desc}")
                            except Exception as e:
                                frame_descs.append(f"[Frame {frame_idx+1}/{n_frames}] Error: {e}")
                        item.vision_desc_n = (
                            "\n".join(frame_descs)
                            if frame_descs
                            else "[VisionError] No frames extracted"
                        )
                    if progress_callback:
                        progress_callback(
                            (idx_in_target + 1) / max(total, 1),
                            f"Vision (multi-frame): {idx_in_target+1}/{total}",
                        )
            finally:
                cap.release()
            jsonl_path = work_dir / "vision_multiframe.jsonl"
            with jsonl_path.open("w", encoding="utf-8") as f:
                for sub_id, item in items.items():
                    if item.vision_desc_n:
                        line = {
                            "sub_id": sub_id,
                            "start_ms": item.start_ms,
                            "end_ms": item.end_ms,
                            "vision_desc": item.vision_desc_n,
                            "frame_count": n_frames,
                        }
                        f.write(json.dumps(line, ensure_ascii=False) + "\n")
            verify_subtitle_items_consistency(items, expected_sub_ids, "Run D (vision_multiframe)")
            if log_lines:
                log_lines.append(f"[Run D] Completed: {len(target_sub_ids)} subtitle items")
        finally:
            if vision_model is not None:
                del vision_model
                gc.collect()
            if log_lines:
                log_lines.append("[Run D] Vision model unloaded")

    return items


def _build_protected_terms(
    items: dict[str, SubtitleItem],
    glossary: list[Any],
    target_language: str,
    limit: int = 200,
) -> list[str]:
    """從 glossary 目標語言詞、placeholder、拉丁/數字抽取，上限 limit 個。"""
    from .glossary import get_glossary_target_term

    protected: list[str] = []
    seen: set[str] = set()
    # 目標語言詞（依 target_language 選：targets[lang] > to > zh）
    for e in (glossary or []):
        term = get_glossary_target_term(e, target_language)
        if term and term not in seen:
            seen.add(term)
            protected.append(term)
    # Placeholder 與拉丁/數字（從所有字幕與 brief 抽取）
    placeholder_re = re.compile(r"<I\d+>|__P\d+__", re.IGNORECASE)
    latin_re = re.compile(r"[A-Za-z0-9]+")
    for item in items.values():
        for raw in (item.text_clean or "", item.text_raw or ""):
            if not raw:
                continue
            for m in placeholder_re.finditer(raw):
                t = m.group(0)
                if t not in seen:
                    seen.add(t)
                    protected.append(t)
            for m in latin_re.finditer(raw):
                t = m.group(0)
                if len(t) > 1 and t not in seen:
                    seen.add(t)
                    protected.append(t)
        brief = item.get_best_brief()
        pack = None
        if brief and brief.reasons:
            try:
                from .pipeline import parse_pack_from_reasons
                pack = parse_pack_from_reasons(brief.reasons)
            except Exception:
                pass
            if pack:
                draft = (pack.get("draft_tl") or "").strip()
                for m in placeholder_re.finditer(draft):
                    t = m.group(0)
                    if t not in seen:
                        seen.add(t)
                        protected.append(t)
    return protected[:limit]


def _has_idiom_requests(pack: Any) -> bool:
    """True if pack has non-empty idiom_requests list."""
    if pack is None:
        return False
    ir = pack.get("idiom_requests")
    return isinstance(ir, list) and len(ir) > 0


def _build_draft_map_from_pack(
    items: dict[str, SubtitleItem],
    sorted_items: list[tuple[str, SubtitleItem]],
    parse_pack_from_reasons: Any,
) -> dict[str, str]:
    """Build draft_map from PACK draft_tl only (no MAIN group translate)."""
    draft_map: dict[str, str] = {}
    for sub_id, item in sorted_items:
        best = item.get_best_brief()
        pack = parse_pack_from_reasons(best.reasons) if best else None
        draft = (pack.get("draft_tl") or "").strip() if pack else ""
        if not draft and best:
            draft = (best.translation_brief or "").strip()
        if not draft:
            draft = item.text_clean or ""
        draft_map[sub_id] = draft
    return draft_map


def run_final_translate(
    items: dict[str, SubtitleItem],
    work_dir: Path,
    cfg: Any,  # AppConfig
    glossary: list[Any],
    target_language: str,
    prompt_config: Any,
    reason_prompt_config: Optional[Any] = None,
    reason_assemble_prompt_config: Optional[Any] = None,
    reason_group_translate_prompt_config: Optional[Any] = None,
    translate_polish_prompt_config: Optional[Any] = None,
    log_lines: Optional[list[str]] = None,
    progress_callback: Optional[Any] = None,
    batch_size: Optional[int] = None,
) -> tuple[dict[str, SubtitleItem], list[tuple[str, str]]]:
    """
    Run F: 最終翻譯。依 cfg.pipeline.run_e_scheme 分支：
    - full: MAIN 群組翻譯 → LOCAL 潤飾
    - main_led: MAIN 群組翻譯，不潤飾
    - local_led: 以 PACK draft_tl 為初稿 → LOCAL 填 slot + LOCAL 潤飾
    - draft_first: 以 PACK draft_tl 為主，僅必要時 LOCAL 填 slot，不潤飾
    """
    from .models import TextModel
    from .pipeline import (
        parse_pack_from_reasons,
        stage_main_group_translate,
        stage2_reason_and_score,
        stage3_polish_local_lines,
        stage3_suggest_local_phrases,
        _fill_draft_with_suggestions,
        _default_tl_instruction,
    )
    from .glossary import apply_glossary_post, force_glossary_replace_output, apply_glossary_pre_anchor, apply_glossary_post_anchor
    from .fallback_split_utils import safe_fallback_split, strip_punctuation as _strip_punctuation
    from .safe_infer import NeedReload, _release
    from .heavy_request import chunk_list

    _raw_scheme = (getattr(cfg.pipeline, "run_e_scheme", "full") or "full")
    scheme = str(_raw_scheme).strip().lower()
    if scheme not in ("full", "main_led", "local_led", "draft_first"):
        scheme = "full"
    max_segments = getattr(cfg.pipeline, "group_translate_max_segments", 4)
    polish_chunk_size = getattr(cfg.pipeline, "local_polish_chunk_size", 60)

    def _is_prompt_polluted(text: str) -> bool:
        """避免寫入 prompt 汙染內容（lines=、{tl_instruction}、JSON 等）。"""
        if not text or not isinstance(text, str):
            return False
        t = text.strip()
        return ("lines=" in t or "{tl_instruction}" in t or "JSON" in t)

    def _tl_instruction_for_run_e(pack_tl_instruction: str, target_lang: str) -> str:
        """Run F 使用：若 PACK 的 tl_instruction 與目標語言不符（如 brief 誤為 Target: en），改為 config 的預設指令，確保輸出正確目標語。"""
        candidate = (pack_tl_instruction or "").strip()
        if not candidate:
            return _default_tl_instruction(target_lang)
        norm = (target_lang or "").strip().lower()
        if norm and norm in candidate.lower():
            return candidate
        return _default_tl_instruction(target_lang)

    def _is_unsatisfactory(translated: str, source_en: str, target_lang: str) -> bool:
        """方案 C：判斷翻譯是否不滿意，需觸發補 brief 再翻。空/過短、或 CJK 目標卻輸出像英文（高 ASCII+高與原文重疊）→ True。"""
        t = (translated or "").strip()
        if not t or len(t) < 2:
            return True
        tl = (target_lang or "").strip().lower()
        if tl.startswith("zh") or tl.startswith("ja"):
            total = sum(1 for c in t if not c.isspace())
            if total == 0:
                return True
            ascii_count = sum(1 for c in t if not c.isspace() and ord(c) < 128)
            if ascii_count / total >= 0.6:
                tw = set(re.findall(r"\b\w+\b", t.lower()))
                sw = set(re.findall(r"\b\w+\b", (source_en or "").lower()))
                if sw and len(tw & sw) / len(sw) >= 0.5:
                    return True
        return False

    def _chunk_needs_retry_for_format(
        sub_segments: list[dict],
        id_to_text: dict[str, str],
        anchors_per_id: dict[str, set[str]],
        target_lang: str,
    ) -> bool:
        """
        檢查一個 chunk 內是否有任一 segment 格式不符，需要重試：
        - 整句語言不符（沿用 _is_unsatisfactory）
        - 有輸入 anchor <PN__...> 卻沒在輸出中保留
        - 出現不允許的 tag（僅允許 <i>...</i>/<b>...</b>/<u>...</u> 與 <PN__...>）
        """
        for s in sub_segments:
            sid = s["id"]
            text = (id_to_text.get(sid) or "").strip()
            if not text:
                # 缺文字也視為不合格，交由重試/後續 fallback 處理
                return True
            source_en = s.get("en_text", "") or ""
            expected_anchors = anchors_per_id.get(sid) or set()

            # 語言不符：目標為 CJK 卻過於英文、且與原文高度重疊
            if _is_unsatisfactory(text, source_en, target_lang):
                return True

            # Anchor 必須完整保留
            for a in expected_anchors:
                if a not in text:
                    return True

            # Tag 檢查：僅允許 <i>/<b>/<u> 及 <PN__...>
            tags = re.findall(r"<[^>]+>", text)
            for t in tags:
                if re.fullmatch(r"</?(i|b|u)>", t):
                    continue
                if re.fullmatch(r"<PN__[^>]+>", t):
                    continue
                return True

        return False

    expected_sub_ids = set(items.keys())
    model_mutex = get_model_mutex()
    sorted_items = sorted(items.items(), key=lambda x: x[1].start_ms)
    id_to_index = {sid: i for i, (sid, _) in enumerate(sorted_items)}
    total = len(sorted_items)

    if log_lines:
        log_lines.append(f"[Run F] scheme={scheme} (full=dual-strong, main_led=main-strong, local_led=local-strong, draft_first=dual-weak)")

    groups = build_sentence_groups(items)
    protected_terms = _build_protected_terms(items, glossary, target_language, limit=200)
    draft_map: dict[str, str] = {}
    all_name_translations: list[dict] = []

    if scheme in ("full", "main_led"):
        # Phase1: MAIN group translate → draft_map
        with model_mutex.hold_model("reason"):
            reason_model: Optional[TextModel] = None
            try:
                if log_lines:
                    log_lines.append("[Run F] Phase1: Loading MAIN (reason) model for group translate...")
                gpu_info = detect_gpu()
                n_ctx_reason = cfg.llama_cpp.n_ctx_reason
                n_gpu_layers_reason = cfg.llama_cpp.n_gpu_layers_reason
                if log_lines and gpu_info.get("vram_mb"):
                    log_lines.append(f"[Run F] Phase1 VRAM {gpu_info['vram_mb']} MB; config n_ctx={n_ctx_reason}, n_gpu_layers={n_gpu_layers_reason} (OOM → safe_chat fallback)")
                reason_path = resolve_reason_model_path(cfg)
                reason_chat_format = (reason_prompt_config.chat_format if reason_prompt_config else "chatml")
                reason_model = TextModel(
                    model_path=str(reason_path),
                    chat_format=reason_chat_format,
                    n_ctx=n_ctx_reason,
                    n_gpu_layers=n_gpu_layers_reason,
                    n_threads=cfg.llama_cpp.n_threads,
                )
                if log_lines:
                    log_lines.append("[Run F] Phase1: MAIN model loaded.")
                num_groups = len(groups)
                phase1_fallback_count = 0
                phase1_anchor_to_target: dict[str, str] = {}
                for g_idx, group in enumerate(groups):
                    if progress_callback:
                        progress_callback((g_idx + 1) / max(num_groups, 1), f"Run F group translate: {g_idx+1}/{num_groups}")
                    # 依 group_translate_max_segments 切子 group，避免單次請求過大
                    sub_segment_lists = chunk_list(group.segments, max_segments)
                    for chunk_idx, sub_segments in enumerate(sub_segment_lists):
                        first_id = sub_segments[0]["id"]
                        last_id = sub_segments[-1]["id"]
                        first_item = items.get(first_id)
                        pack0 = parse_pack_from_reasons(first_item.get_best_brief().reasons) if (first_item and first_item.get_best_brief()) else None
                        tl_instruction = _tl_instruction_for_run_e(
                            (pack0.get("tl_instruction") or "").strip() if pack0 else "",
                            target_language,
                        )
                        # Glossary Anchor：翻譯前將術語替換為 <PN__Key>，並累積 anchor -> 目標語
                        anchored_list = []
                        anchors_per_id: dict[str, set[str]] = {}
                        for s in sub_segments:
                            en_raw = s.get("en_text", "")
                            en_anchored, chunk_map = apply_glossary_pre_anchor(en_raw, glossary or [], target_language)
                            phase1_anchor_to_target.update(chunk_map)
                            found = set(re.findall(r"<PN__[^>]+>", en_anchored))
                            if found:
                                anchors_per_id[s["id"]] = found
                            anchored_list.append({"id": s["id"], "en": en_anchored, "ms": s.get("duration_ms", 0)})
                        segments_json = json.dumps(anchored_list, ensure_ascii=False)
                        # Run F 僅翻譯：上下文擴充已在 Run E 完成，不再在此傳 expanded_context
                        group_id = first_id
                        load_params_reason = {
                            "chat_format": reason_chat_format,
                            "n_ctx": n_ctx_reason,
                            "n_gpu_layers": n_gpu_layers_reason,
                            "n_threads": cfg.llama_cpp.n_threads,
                        }
                        # 一次主翻譯 + 最多一次重試（格式不符則補充硬規則重試）
                        def _call_main_group_once(effective_tl_instruction: str):
                            nonlocal reason_model, n_ctx_reason, n_gpu_layers_reason
                            while True:
                                try:
                                    return stage_main_group_translate(
                                        reason_model,
                                        target_language,
                                        segments_json,
                                        effective_tl_instruction,
                                        reason_group_translate_prompt_config,
                                        expanded_context=None,
                                        model_path=str(reason_path),
                                        load_params=load_params_reason,
                                        cfg=cfg,
                                        group_segments_count=len(sub_segments),
                                        log_lines=log_lines,
                                        log_label=f"main_group_translate group={group.group_id} chunk={chunk_idx}",
                                    )
                                except NeedReload as e:
                                    if log_lines:
                                        log_lines.append(f"[Run F] Phase1 group {group_id} OOM → {e.level} (params {e.params}), reloading MAIN (CPU if L5)...")
                                    del reason_model
                                    gc.collect()
                                    n_ctx_reason = e.params.get("n_ctx", n_ctx_reason)
                                    n_gpu_layers_reason = e.params.get("n_gpu_layers", n_gpu_layers_reason)
                                    reason_model = TextModel(
                                        model_path=str(reason_path),
                                        chat_format=reason_chat_format,
                                        n_ctx=n_ctx_reason,
                                        n_gpu_layers=n_gpu_layers_reason,
                                        n_threads=cfg.llama_cpp.n_threads,
                                    )
                                    load_params_reason["n_ctx"] = n_ctx_reason
                                    load_params_reason["n_gpu_layers"] = n_gpu_layers_reason

                        # 初次呼叫
                        segment_texts = _call_main_group_once(tl_instruction)

                        # 轉成 id -> text
                        _release(reason_model)
                        gc.collect()
                        expected_ids_chunk = {str(s["id"]).strip() for s in sub_segments}
                        id_to_text: dict[str, str] = {}
                        if segment_texts is not None and isinstance(segment_texts, list):
                            for x in segment_texts:
                                sid = x.get("id")
                                if sid is None:
                                    continue
                                sid = str(sid).strip()
                                if sid not in expected_ids_chunk:
                                    continue
                                text = x.get("text")
                                if isinstance(text, str) and text.strip():
                                    id_to_text[sid] = text.strip()

                        # 檢查格式；若不符合，最多重試一次（補充硬規則）
                        if id_to_text and _chunk_needs_retry_for_format(sub_segments, id_to_text, anchors_per_id, target_language):
                            if log_lines:
                                log_lines.append(
                                    f"[Run F] Phase1 group {group_id} chunk={chunk_idx} format invalid, retrying once with stricter prompt..."
                                )
                            hard_tl = (tl_instruction or "") + (
                                " HARD FORMAT RULES: "
                                "1) Keep every <PN__...> anchor exactly as in input; do NOT translate/modify/split/remove it. "
                                "2) Preserve subtitle tags <i>...</i>, <b>...</b>, <u>...</u> with correct pairing. "
                                f"3) Entire line must be in target language {target_language}, not English."
                            )
                            segment_texts_retry = _call_main_group_once(hard_tl)
                            # 重新解析 retry 結果；若成功則覆蓋 id_to_text
                            expected_ids_chunk_retry = expected_ids_chunk
                            id_to_text_retry: dict[str, str] = {}
                            if segment_texts_retry is not None and isinstance(segment_texts_retry, list):
                                for x in segment_texts_retry:
                                    sid = x.get("id")
                                    if sid is None:
                                        continue
                                    sid = str(sid).strip()
                                    if sid not in expected_ids_chunk_retry:
                                        continue
                                    text = x.get("text")
                                    if isinstance(text, str) and text.strip():
                                        id_to_text_retry[sid] = text.strip()
                            if id_to_text_retry:
                                id_to_text = id_to_text_retry

                        if id_to_text:
                            for s in sub_segments:
                                sid = s["id"]
                                draft = id_to_text.get(sid)
                                if not draft or _is_prompt_polluted(draft):
                                    it = items.get(sid)
                                    p = parse_pack_from_reasons(it.get_best_brief().reasons) if (it and it.get_best_brief()) else None
                                    draft = (p.get("draft_tl") or "").strip() if p else ""
                                if not draft or _is_prompt_polluted(draft):
                                    draft = s.get("en_text", "")
                                draft = apply_glossary_post_anchor(draft or "", phase1_anchor_to_target)
                                draft_map[sid] = draft
                            if len(id_to_text) < len(sub_segments) and log_lines:
                                log_lines.append(f"[Run F] Phase1 group {group_id}: using MAIN for {len(id_to_text)}/{len(sub_segments)} segments, rest from PACK")
                        else:
                            phase1_fallback_count += 1
                            parts = []
                            for seg in sub_segments:
                                it = items.get(seg["id"])
                                best = it.get_best_brief() if it else None
                                p = parse_pack_from_reasons(best.reasons) if best else None
                                parts.append((p.get("draft_tl") or "").strip() if p else seg.get("en_text", ""))
                            group_text = " ".join(p for p in parts if p)
                            if not group_text:
                                group_text = " ".join(seg.get("en_text", "") for seg in sub_segments)
                            fallback_map = safe_fallback_split(group_text, sub_segments, protected_terms)
                            for sid, line in fallback_map.items():
                                draft_map[sid] = line
                if phase1_fallback_count and log_lines:
                    log_lines.append(f"[Run F] Phase1: {phase1_fallback_count} chunk(s) used safe_fallback_split (MAIN JSON failed or empty).")
                # 補齊未出現在任何 group 的 sub_id（理論上不會發生，因 groups 覆蓋所有 items）
                for sub_id in expected_sub_ids:
                    if sub_id not in draft_map:
                        it = items.get(sub_id)
                        best = it.get_best_brief() if it else None
                        pack = parse_pack_from_reasons(best.reasons) if best else None
                        raw_draft = (pack.get("draft_tl") if pack else "") or (best.translation_brief if best else "") or (it.text_clean if it else "") or ""
                        draft_map[sub_id] = str(raw_draft).strip() if raw_draft else ""
            finally:
                if reason_model is not None:
                    del reason_model
                    gc.collect()
                if log_lines:
                    log_lines.append("[Run F] Phase1: MAIN model unloaded.")

    elif scheme in ("local_led", "draft_first"):
        # draft_map from PACK only (no MAIN group translate)
        draft_map = _build_draft_map_from_pack(items, sorted_items, parse_pack_from_reasons)
        need_assembly = any(
            _has_idiom_requests(
                parse_pack_from_reasons(item.get_best_brief().reasons) if item.get_best_brief() else None
            )
            for item in items.values()
        )
        if need_assembly:
            with model_mutex.hold_model("translate"):
                translate_model_ia: Optional[TextModel] = None
                try:
                    if log_lines:
                        log_lines.append("[Run F] (local_led/draft_first) Loading LOCAL for idiom suggestions...")
                    translate_path_ia = resolve_translate_model_path(cfg)
                    translate_chat_format_ia = prompt_config.chat_format if prompt_config else "chatml"
                    translate_model_ia = TextModel(
                        model_path=str(translate_path_ia),
                        chat_format=translate_chat_format_ia,
                        n_ctx=cfg.llama_cpp.n_ctx_translate,
                        n_gpu_layers=cfg.llama_cpp.n_gpu_layers_translate,
                        n_threads=cfg.llama_cpp.n_threads,
                    )
                    for idx, (sub_id, item) in enumerate(sorted_items):
                        if progress_callback and idx % 20 == 0:
                            progress_callback(idx / max(total, 1), f"Run F idiom suggestions: {idx}/{total}")
                        best = item.get_best_brief()
                        pack = parse_pack_from_reasons(best.reasons) if best else None
                        if not _has_idiom_requests(pack):
                            continue
                        tl_instruction = _tl_instruction_for_run_e(
                            (pack.get("tl_instruction") or "").strip() if pack else "",
                            target_language,
                        )
                        requests_json = json.dumps(pack.get("idiom_requests", []), ensure_ascii=False)
                        suggestions = stage3_suggest_local_phrases(
                            translate_model_ia,
                            requests_json,
                            tl_instruction,
                            target_language,
                            prompt_config,
                            log_lines=log_lines,
                            log_label="Run F idiom",
                        )
                        draft_tl = (pack.get("draft_tl") or "").strip()
                        filled = _fill_draft_with_suggestions(draft_tl, suggestions)
                        if filled:
                            draft_map[sub_id] = filled
                finally:
                    if translate_model_ia is not None:
                        del translate_model_ia
                        gc.collect()
                    if log_lines:
                        log_lines.append("[Run F] LOCAL (idiom suggestions) unloaded.")

    # Apply omit_sfx: main model marks CC-only sound effects/onomatopoeia; output empty for those lines
    for sub_id in expected_sub_ids:
        it = items.get(sub_id)
        if not it:
            continue
        best = it.get_best_brief() if it else None
        pack = parse_pack_from_reasons(best.reasons) if best else None
        if pack and pack.get("omit_sfx") is True:
            draft_map[sub_id] = ""

    if scheme == "full" or scheme == "local_led":
        # Phase2: LOCAL polish
        with model_mutex.hold_model("translate"):
            translate_model: Optional[TextModel] = None
            try:
                if log_lines:
                    log_lines.append("[Run F] Phase2: Loading LOCAL (translate) model for polish...")
                gpu_info = detect_gpu()
                n_ctx_translate = cfg.llama_cpp.n_ctx_translate
                n_gpu_layers_translate = cfg.llama_cpp.n_gpu_layers_translate
                if log_lines and gpu_info.get("vram_mb"):
                    log_lines.append(f"[Run F] Phase2 VRAM {gpu_info['vram_mb']} MB; config n_ctx={n_ctx_translate}, n_gpu_layers={n_gpu_layers_translate} (OOM → safe_chat fallback)")
                translate_path = resolve_translate_model_path(cfg)
                translate_chat_format = prompt_config.chat_format if prompt_config else "chatml"
                translate_model = TextModel(
                    model_path=str(translate_path),
                    chat_format=translate_chat_format,
                    n_ctx=n_ctx_translate,
                    n_gpu_layers=n_gpu_layers_translate,
                    n_threads=cfg.llama_cpp.n_threads,
                )
                if log_lines:
                    log_lines.append("[Run F] Phase2: LOCAL model loaded.")
                local_polish_mode = getattr(cfg.pipeline, "local_polish_mode", "strong") or "strong"
                local_polish_mode = str(local_polish_mode).strip().lower()
                if local_polish_mode not in ("weak", "strong"):
                    local_polish_mode = "strong"
                if log_lines:
                    log_lines.append(f"[Run F] Phase2: local_polish_mode={local_polish_mode} (weak=strip punctuation before input; strong=keep punctuation, first char not preceded by punctuation)")
                lines_for_polish = [
                    {"id": sub_id, "text": draft_map.get(sub_id, "")}
                    for sub_id, _ in sorted_items
                ]
                if local_polish_mode == "weak":
                    lines_for_polish = [
                        {"id": line["id"], "text": _strip_punctuation((line.get("text") or "").strip(), keep_decimal=True, keep_acronym=True)}
                        for line in lines_for_polish
                    ]
                first_item = sorted_items[0][1] if sorted_items else None
                pack0 = parse_pack_from_reasons(first_item.get_best_brief().reasons) if (first_item and first_item.get_best_brief()) else None
                tl_instruction = _tl_instruction_for_run_e(
                    (pack0.get("tl_instruction") or "").strip() if pack0 else "",
                    target_language,
                )
                polished_map: dict[str, str] = {}
                all_name_translations: list[dict] = []
                phase2_reload_count = 0
                max_phase2_reloads = 3
                # 二分拆半最小單位：len==1 仍失敗則跳過該句 polish，保留 MAIN draft
                min_polish_chunk_lines = 1

                def _polish_one_chunk(
                    model: Any,
                    chunk_lines: list[dict],
                    model_path: str,
                    load_params: dict,
                    cfg: Any,
                    items_map: dict,
                    parse_pack_fn: Any,
                    polish_mode: str = "strong",
                ) -> tuple[dict[str, str], list[dict]]:
                    """單一 chunk 呼叫 stage3_polish_local_lines；回傳 (polished_dict, name_translations)。若回傳空且 len>1 則二分拆半重試。"""
                    transliteration_terms: list[str] = []
                    for line in chunk_lines:
                        sid = line.get("id")
                        it = items_map.get(sid) if sid else None
                        best = it.get_best_brief() if it else None
                        pack = parse_pack_fn(best.reasons) if best else None
                        req = pack.get("transliteration_requests", []) if pack and isinstance(pack.get("transliteration_requests"), list) else []
                        for t in req:
                            if isinstance(t, str) and t.strip() and t.strip() not in transliteration_terms:
                                transliteration_terms.append(t.strip())
                    result_dict, result_names = stage3_polish_local_lines(
                        model,
                        target_language,
                        tl_instruction,
                        chunk_lines,
                        translate_polish_prompt_config,
                        log_lines=log_lines,
                        log_label="Run F",
                        model_path=model_path,
                        load_params=load_params,
                        cfg=cfg,
                        lines_count=len(chunk_lines),
                        transliteration_terms=transliteration_terms if transliteration_terms else None,
                        local_polish_mode=polish_mode,
                    )
                    if result_dict:
                        return (result_dict, result_names or [])
                    if len(chunk_lines) <= min_polish_chunk_lines:
                        return ({}, [])
                    mid = len(chunk_lines) // 2
                    left = chunk_lines[:mid]
                    right = chunk_lines[mid:]
                    if log_lines:
                        log_lines.append(f"[Run F] Phase2: chunk {len(chunk_lines)} lines empty, binary split → {len(left)} + {len(right)}")
                    left_out, left_names = _polish_one_chunk(model, left, model_path, load_params, cfg, items, parse_pack_from_reasons, polish_mode=local_polish_mode)
                    right_out, right_names = _polish_one_chunk(model, right, model_path, load_params, cfg, items, parse_pack_from_reasons, polish_mode=local_polish_mode)
                    return ({**left_out, **right_out}, (left_names or []) + (right_names or []))

                # local_polish 永遠分 chunk（例如 60 行），逐 chunk 呼叫 stage3_polish_local_lines；避免整集一次丟造成 heavy/isolated 雙載
                polish_chunks = chunk_list(lines_for_polish, polish_chunk_size)
                total_items = len(lines_for_polish)
                chunks_to_do: list[list[dict]] = list(polish_chunks)
                if log_lines:
                    log_lines.append(f"[Run F] Phase2: local_polish chunk_size={polish_chunk_size}, {len(polish_chunks)} chunks ({total_items} items)")

                while chunks_to_do:
                    chunk = chunks_to_do.pop(0)
                    if progress_callback:
                        progress_callback(len(polished_map) / max(total_items, 1), f"Run F local polish: {len(polished_map)}/{total_items}")
                    load_params_translate = {
                        "chat_format": translate_chat_format,
                        "n_ctx": n_ctx_translate,
                        "n_gpu_layers": n_gpu_layers_translate,
                        "n_threads": cfg.llama_cpp.n_threads,
                    }
                    chunk_result: dict[str, str] = {}
                    chunk_names: list[dict] = []
                    while True:
                        try:
                            chunk_result, chunk_names = _polish_one_chunk(
                                translate_model,
                                chunk,
                                str(translate_path),
                                load_params_translate,
                                cfg,
                                items,
                                parse_pack_from_reasons,
                                polish_mode=local_polish_mode,
                            )
                            break
                        except NeedReload as e:
                            phase2_reload_count += 1
                            if log_lines:
                                log_lines.append(f"[Run F] Phase2 OOM → {e.level} (params {e.params}), reload #{phase2_reload_count} LOCAL...")
                            if phase2_reload_count > max_phase2_reloads:
                                if log_lines:
                                    log_lines.append("[Run F] Phase2: skipping polish after OOM/reload limit, keeping MAIN draft; output will apply glossary + strip_punctuation")
                                chunk_result = {}
                                chunk_names = []
                                break
                            del translate_model
                            gc.collect()
                            n_ctx_translate = e.params.get("n_ctx", n_ctx_translate)
                            n_gpu_layers_translate = e.params.get("n_gpu_layers", n_gpu_layers_translate)
                            translate_model = TextModel(
                                model_path=str(translate_path),
                                chat_format=translate_chat_format,
                                n_ctx=n_ctx_translate,
                                n_gpu_layers=n_gpu_layers_translate,
                                n_threads=cfg.llama_cpp.n_threads,
                            )
                            load_params_translate["n_ctx"] = n_ctx_translate
                            load_params_translate["n_gpu_layers"] = n_gpu_layers_translate
                            # OOM → binary split：將當前 chunk 拆半放回 queue，不重試同 chunk 避免再 OOM
                            if len(chunk) > 1:
                                mid = len(chunk) // 2
                                left_half = chunk[:mid]
                                right_half = chunk[mid:]
                                chunks_to_do.insert(0, right_half)
                                chunks_to_do.insert(0, left_half)
                                if log_lines:
                                    log_lines.append(f"[Run F] Phase2: OOM binary split → {len(left_half)} + {len(right_half)}")
                                chunk_result = {}
                                chunk_names = []
                                break
                            # len==1：跳過該句 polish，保留 MAIN draft
                            chunk_result = {}
                            chunk_names = []
                            break
                    polished_map.update(chunk_result)
                    all_name_translations.extend(chunk_names or [])
                    _release(translate_model)
                    gc.collect()
                    if phase2_reload_count > max_phase2_reloads:
                        break  # 已達 reload 上限，跳過後續 chunk，保留 MAIN draft
                for sid, text in polished_map.items():
                    if sid not in expected_sub_ids or sid not in draft_map or not text:
                        continue
                    draft_map[sid] = text
                if log_lines:
                    num_polished = len(polished_map)
                    num_fallback = total_items - num_polished
                    log_lines.append(f"[Run F] Phase2 summary: {total_items} items, polish ok {num_polished}, fallback MAIN draft {num_fallback}")
            finally:
                if translate_model is not None:
                    del translate_model
                    gc.collect()
                if log_lines:
                    log_lines.append("[Run F] Phase2: LOCAL model unloaded.")

            # Run F 僅翻譯：Run E 已做上下文擴充，不再 Phase 2.5 補 brief 重翻

    # 6) 最終輸出：每句 glossary + strip_punctuation（可配置）+ 寫入 translated_text
    do_strip = getattr(cfg.pipeline, "strip_punctuation", True)
    keep_decimal = getattr(cfg.pipeline, "strip_punctuation_keep_decimal", True)
    keep_acronym = getattr(cfg.pipeline, "strip_punctuation_keep_acronym", True)
    for sub_id, item in sorted_items:
        zh = draft_map.get(sub_id, "")
        zh = apply_glossary_post(zh or "")
        zh = force_glossary_replace_output(zh or "", glossary or [], target_language)
        zh_before_strip = (zh or "").replace("\n", " ").strip()
        if do_strip and zh_before_strip:
            zh_stripped = _strip_punctuation(zh_before_strip, keep_decimal=keep_decimal, keep_acronym=keep_acronym)
            # Avoid clearing to empty: if strip would remove all content, keep pre-strip
            zh = (zh_stripped or "").strip() or zh_before_strip
        else:
            zh = zh_before_strip
        # Fallback: if still empty, keep MAIN draft (already in draft_map) without further strip
        if not zh and draft_map.get(sub_id):
            zh = (draft_map.get(sub_id) or "").replace("\n", " ").strip()
        item.translated_text = zh

    verify_subtitle_items_consistency(items, expected_sub_ids, "Run F")
    work_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = work_dir / "final_translations.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for sub_id, item in sorted_items:
            f.write(json.dumps({
                "sub_id": sub_id,
                "start_ms": item.start_ms,
                "end_ms": item.end_ms,
                "translated_text": item.translated_text,
            }, ensure_ascii=False) + "\n")
    if log_lines:
        log_lines.append(f"[Run F] Completed: {len(items)} subtitle items")

    # Build name_mappings for CSV (glossary format): (original, translated); include transliteration_requests without translation as (term, "")
    all_transliteration_requests: set[str] = set()
    for it in items.values():
        best = it.get_best_brief() if it else None
        pack = parse_pack_from_reasons(best.reasons) if best else None
        reqs = pack.get("transliteration_requests", []) if pack and isinstance(pack.get("transliteration_requests"), list) else []
        for t in reqs:
            if isinstance(t, str) and t.strip():
                all_transliteration_requests.add(t.strip())
    seen_original: set[str] = set()
    name_mappings: list[tuple[str, str]] = []
    for x in all_name_translations:
        orig = (x.get("original") or "").strip()
        trans = (x.get("translated") or "").strip()
        if orig and orig not in seen_original:
            name_mappings.append((orig, trans))
            seen_original.add(orig)
    for req in sorted(all_transliteration_requests):
        if req not in seen_original:
            name_mappings.append((req, ""))
            seen_original.add(req)

    return (items, name_mappings)
