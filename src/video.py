from __future__ import annotations
import cv2
import numpy as np

def open_video(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    return cap

def get_frame_at_ms(cap: cv2.VideoCapture, time_ms: float, max_width: int = 640) -> np.ndarray | None:
    """
    從影片中提取指定時間點的幀。
    
    Args:
        cap: OpenCV VideoCapture 物件
        time_ms: 時間點（毫秒）
        max_width: 最大寬度（用於縮放）
        
    Returns:
        RGB 格式的幀（numpy array），如果提取失敗則返回 None
    """
    if time_ms < 0:
        return None
    
    cap.set(cv2.CAP_PROP_POS_MSEC, float(time_ms))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    h, w = frame.shape[:2]
    if w > max_width:
        new_w = max_width
        new_h = int(h * (new_w / w))
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

def encode_jpg_bytes(rgb: np.ndarray, quality: int = 85) -> bytes:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("Failed to encode jpg")
    return buf.tobytes()
