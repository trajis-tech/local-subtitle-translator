"""
Run A: Audio emotion/tone analysis for all subtitle lines.

This module extracts audio segments from video and analyzes emotion/tone
using Hugging Face Wav2Vec2 speech emotion recognition (audio-classification).
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any

from .srt_utils import sub_midpoints_ms
from .audio_model import AudioModel


@dataclass
class AudioTags:
    """音訊標籤結構"""
    emotion: str = ""  # e.g., "excited", "sad", "neutral"
    tone: str = ""  # e.g., "sarcastic", "questioning", "whispering"
    intensity: str = ""  # e.g., "high", "medium", "low"
    speaking_style: str = ""  # e.g., "fast", "slow", "paused"


@dataclass
class AudioResult:
    """音訊處理結果"""
    idx: int
    start_ms: float
    end_ms: float
    audio_tags: AudioTags
    audio_reason: str = ""  # 簡短說明


def run_audio_analysis(
    video_path: str,
    subs: Any,  # pysrt.SubRipFile
    work_dir: Path,
    audio_model: AudioModel | None = None,
    progress_callback: Any | None = None,
) -> list[AudioResult]:
    """
    Run A: 對所有字幕行進行音訊分析。
    
    Args:
        video_path: 影片路徑
        subs: pysrt.SubRipFile
        work_dir: 工作目錄（./work/）
        audio_model: AudioModel 實例（已載入）
        progress_callback: 進度回調函數
    
    Returns:
        AudioResult 列表
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    results: list[AudioResult] = []
    total = len(subs)
    
    # 如果沒有提供音訊模型，產生空結果
    if audio_model is None:
        for i, sub in enumerate(subs):
            start_ms, end_ms, _mid = sub_midpoints_ms(sub)
            audio_tags = AudioTags(
                emotion="",
                tone="",
                intensity="",
                speaking_style="",
            )
            result = AudioResult(
                idx=i,
                start_ms=start_ms,
                end_ms=end_ms,
                audio_tags=audio_tags,
                audio_reason="Audio model not loaded",
            )
            results.append(result)
            if progress_callback:
                progress_callback((i + 1) / total, f"Audio: {i+1}/{total} (skipped)")
    else:
        # 使用音訊模型進行分析
        for i, sub in enumerate(subs):
            start_ms, end_ms, _mid = sub_midpoints_ms(sub)
            
            try:
                # 提取音訊片段
                audio_segment_path = audio_model.extract_audio_segment(
                    video_path,
                    start_ms,
                    end_ms,
                )
                
                # 分析情緒
                emotion_result = audio_model.analyze_emotion(audio_segment_path)
                
                # 清理臨時文件
                try:
                    Path(audio_segment_path).unlink()
                except OSError:
                    pass
                
                # 若推論回傳 error，使用空標籤並記錄原因
                if emotion_result.get("error"):
                    audio_tags = AudioTags(
                        emotion="",
                        tone="",
                        intensity="",
                        speaking_style="",
                    )
                    reason = f"Audio analysis failed: {emotion_result['error']}"
                else:
                    # 映射情緒到 AudioTags
                    # HF Wav2Vec2 模型輸出情緒標籤（如 "happy", "sad", "angry", "neutral"）
                    emotion = emotion_result.get("emotion", "neutral")
                    confidence = emotion_result.get("confidence", 0.0)
                    
                    # 根據情緒推斷語氣和強度
                    tone = _infer_tone_from_emotion(emotion)
                    intensity = _infer_intensity_from_confidence(confidence)
                    speaking_style = _infer_speaking_style_from_emotion(emotion)
                    
                    audio_tags = AudioTags(
                        emotion=emotion,
                        tone=tone,
                        intensity=intensity,
                        speaking_style=speaking_style,
                    )
                    reason = f"Emotion: {emotion} (confidence: {confidence:.2f})"
                
            except Exception as e:
                # 如果分析失敗，使用空結果
                audio_tags = AudioTags(
                    emotion="",
                    tone="",
                    intensity="",
                    speaking_style="",
                )
                reason = f"Audio analysis failed: {e}"
            
            result = AudioResult(
                idx=i,
                start_ms=start_ms,
                end_ms=end_ms,
                audio_tags=audio_tags,
                audio_reason=reason,
            )
            results.append(result)
            
            if progress_callback:
                progress_callback((i + 1) / total, f"Audio: {i+1}/{total}")
    
    # 寫入 JSONL
    jsonl_path = work_dir / "audio_tags.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for result in results:
            line = {
                "idx": result.idx,
                "start_ms": result.start_ms,
                "end_ms": result.end_ms,
                "audio_tags": {
                    "emotion": result.audio_tags.emotion,
                    "tone": result.audio_tags.tone,
                    "intensity": result.audio_tags.intensity,
                    "speaking_style": result.audio_tags.speaking_style,
                },
                "audio_reason": result.audio_reason,
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    
    return results


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


def load_audio_results(work_dir: Path) -> list[AudioResult] | None:
    """
    從 JSONL 載入音訊結果（用於續跑）。
    """
    jsonl_path = work_dir / "audio_tags.jsonl"
    if not jsonl_path.exists():
        return None
    
    results: list[AudioResult] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            if "_run_meta" in data:
                continue
            audio_tags_data = data.get("audio_tags", {})
            audio_tags = AudioTags(
                emotion=audio_tags_data.get("emotion", ""),
                tone=audio_tags_data.get("tone", ""),
                intensity=audio_tags_data.get("intensity", ""),
                speaking_style=audio_tags_data.get("speaking_style", ""),
            )
            results.append(AudioResult(
                idx=data["idx"],
                start_ms=data["start_ms"],
                end_ms=data["end_ms"],
                audio_tags=audio_tags,
                audio_reason=data.get("audio_reason", ""),
            ))
    
    return results if results else None
