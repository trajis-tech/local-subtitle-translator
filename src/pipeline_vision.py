"""
Run C/D: Vision model for lines that need visual context (need_vision / need_multi_frame_vision).

This module handles:
- Run C: Single-frame vision analysis
- Run D: Multi-frame vision analysis
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any

from .video import open_video, get_frame_at_ms, encode_jpg_bytes
from .srt_utils import sub_midpoints_ms


@dataclass
class VisionResult:
    """視覺分析結果"""
    idx: int
    start_ms: float
    end_ms: float
    vision_desc: str  # 場景/人物/動作描述
    frame_count: int = 1  # 使用的幀數（1 = 單張，>1 = 多張）


def run_vision_single_frame(
    vision_model: Any,  # LocalVisionModel or VisionModel (backward compatible)
    video_path: str,
    subs: Any,  # pysrt.SubRipFile
    en_lines: list[str],
    target_indices: list[int],  # 需要處理的行索引
    work_dir: Path,
    progress_callback: Any | None = None,
) -> list[VisionResult]:
    """
    Run C: 單張影像 fallback（取時間區間正中間）。
    
    Args:
        vision_model: 視覺模型（已載入）
        video_path: 影片路徑
        subs: pysrt.SubRipFile
        en_lines: 英文字幕列表
        target_indices: 需要處理的行索引列表
        work_dir: 工作目錄
        progress_callback: 進度回調
    
    Returns:
        VisionResult 列表
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    results: list[VisionResult] = []
    cap = open_video(video_path)
    
    try:
        for idx_in_target, sub_idx in enumerate(target_indices):
            sub = subs[sub_idx]
            start_ms, end_ms, mid_ms = sub_midpoints_ms(sub)
            line_en = en_lines[sub_idx]
            
            # 檢查時間範圍是否有效
            if end_ms <= start_ms:
                vision_desc = "[VisionError] Invalid time range"
                result = VisionResult(
                    idx=sub_idx,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    vision_desc=vision_desc,
                    frame_count=0,
                )
                results.append(result)
                if progress_callback:
                    progress_callback((idx_in_target + 1) / len(target_indices), 
                                    f"Vision (1-frame): {idx_in_target+1}/{len(target_indices)}")
                continue
            
            # 取正中間的 frame
            frame = get_frame_at_ms(cap, mid_ms)
            if frame is None:
                vision_desc = "[VisionError] Failed to extract frame"
            else:
                try:
                    img_bytes = encode_jpg_bytes(frame)
                    vision_desc = vision_model.describe_with_grounding(img_bytes, line_en)
                except Exception as e:
                    vision_desc = f"[VisionError] {e}"
            
            result = VisionResult(
                idx=sub_idx,
                start_ms=start_ms,
                end_ms=end_ms,
                vision_desc=vision_desc,
                frame_count=1,
            )
            results.append(result)
            
            if progress_callback:
                progress_callback((idx_in_target + 1) / len(target_indices), 
                                f"Vision (1-frame): {idx_in_target+1}/{len(target_indices)}")
    finally:
        cap.release()
    
    # 寫入 JSONL
    jsonl_path = work_dir / "vision_1frame.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for result in results:
            line = {
                "idx": result.idx,
                "start_ms": result.start_ms,
                "end_ms": result.end_ms,
                "vision_desc": result.vision_desc,
                "frame_count": result.frame_count,
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    
    return results


def run_vision_multiframe(
    vision_model: Any,  # LocalVisionModel or VisionModel (backward compatible)
    video_path: str,
    subs: Any,  # pysrt.SubRipFile
    en_lines: list[str],
    target_indices: list[int],  # 需要處理的行索引
    n_frames: int,  # 要抽取的幀數
    work_dir: Path,
    progress_callback: Any | None = None,
) -> list[VisionResult]:
    """
    Run D: 多張影像 fallback（等分切割取樣）。
    
    Args:
        vision_model: 視覺模型（已載入）
        video_path: 影片路徑
        subs: pysrt.SubRipFile
        en_lines: 英文字幕列表
        target_indices: 需要處理的行索引列表
        n_frames: 要抽取的幀數
        work_dir: 工作目錄
        progress_callback: 進度回調
    
    Returns:
        VisionResult 列表
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    results: list[VisionResult] = []
    cap = open_video(video_path)
    
    try:
        for idx_in_target, sub_idx in enumerate(target_indices):
            sub = subs[sub_idx]
            start_ms, end_ms, _mid = sub_midpoints_ms(sub)
            line_en = en_lines[sub_idx]
            
            # 等分切割取樣
            duration = end_ms - start_ms
            if duration <= 0:
                # 如果時間範圍無效，跳過這一行
                vision_desc = "[VisionError] Invalid time range"
                result = VisionResult(
                    idx=sub_idx,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    vision_desc=vision_desc,
                    frame_count=0,
                )
                results.append(result)
                if progress_callback:
                    progress_callback((idx_in_target + 1) / len(target_indices),
                                    f"Vision (multi-frame): {idx_in_target+1}/{len(target_indices)}")
                continue
            
            frame_descs = []
            
            for frame_idx in range(n_frames):
                # 計算取樣時間點（等分）
                t = start_ms + (duration * (frame_idx + 0.5) / n_frames)
                frame = get_frame_at_ms(cap, t)
                if frame is None:
                    continue
                
                try:
                    img_bytes = encode_jpg_bytes(frame)
                    desc = vision_model.describe_with_grounding(img_bytes, line_en)
                    frame_descs.append(f"[Frame {frame_idx+1}/{n_frames}] {desc}")
                except Exception as e:
                    frame_descs.append(f"[Frame {frame_idx+1}/{n_frames}] Error: {e}")
            
            # 合併描述
            vision_desc = "\n".join(frame_descs) if frame_descs else "[VisionError] No frames extracted"
            
            result = VisionResult(
                idx=sub_idx,
                start_ms=start_ms,
                end_ms=end_ms,
                vision_desc=vision_desc,
                frame_count=len(frame_descs),
            )
            results.append(result)
            
            if progress_callback:
                progress_callback((idx_in_target + 1) / len(target_indices),
                                f"Vision (multi-frame): {idx_in_target+1}/{len(target_indices)}")
    finally:
        cap.release()
    
    # 寫入 JSONL
    jsonl_path = work_dir / "vision_multiframe.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for result in results:
            line = {
                "idx": result.idx,
                "start_ms": result.start_ms,
                "end_ms": result.end_ms,
                "vision_desc": result.vision_desc,
                "frame_count": result.frame_count,
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    
    return results


def load_vision_results(work_dir: Path, single_frame: bool = True) -> dict[int, VisionResult] | None:
    """
    從 JSONL 載入視覺結果（用於續跑）。
    
    Returns:
        {idx: VisionResult} 字典
    """
    jsonl_path = work_dir / ("vision_1frame.jsonl" if single_frame else "vision_multiframe.jsonl")
    if not jsonl_path.exists():
        return None
    
    results: dict[int, VisionResult] = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            result = VisionResult(
                idx=data["idx"],
                start_ms=data["start_ms"],
                end_ms=data["end_ms"],
                vision_desc=data["vision_desc"],
                frame_count=data.get("frame_count", 1),
            )
            results[result.idx] = result
    
    return results if results else None
