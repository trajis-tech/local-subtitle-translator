"""
Heavy request 隔離：判定與切塊（語言無關），避免大請求把 VRAM/上下文吃爆。

提供：estimate_tokens_rough、is_heavy、chunk_list。
"""

from __future__ import annotations
import math
from typing import Any, List, Sequence, TypeVar

# CJK 字元範圍（粗略：中文、日文假名/漢字、韓文）
_CJK_RANGES = (
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs
    (0x3040, 0x309F),   # Hiragana
    (0x30A0, 0x30FF),   # Katakana
    (0xAC00, 0xD7AF),   # Hangul Syllables
)


def _cjk_ratio(text: str) -> float:
    """回傳 text 中 CJK 字元佔比 0..1。"""
    if not text or not isinstance(text, str):
        return 0.0
    n = len(text)
    if n == 0:
        return 0.0
    cjk = 0
    for c in text:
        cp = ord(c)
        for lo, hi in _CJK_RANGES:
            if lo <= cp <= hi:
                cjk += 1
                break
    return cjk / n


def estimate_tokens_rough(text: str) -> int:
    """
    粗估 token 數（語言無關）。
    - CJK 比例 > 0.25：tokens ≈ ceil(len/1.8)
    - 否則（偏英文）：tokens ≈ ceil(len/3.2)
    """
    if not text or not isinstance(text, str):
        return 0
    n = len(text)
    if n == 0:
        return 0
    if _cjk_ratio(text) > 0.25:
        return max(1, math.ceil(n / 1.8))
    return max(1, math.ceil(n / 3.2))


def _get_content(msg: Any) -> str:
    """從 message 取出 content 字串（支援 dict 或 object）。"""
    if isinstance(msg, dict):
        c = msg.get("content")
    else:
        c = getattr(msg, "content", None)
    if isinstance(c, str):
        return c
    if c is not None and not isinstance(c, str):
        return str(c)
    return ""


def is_heavy(
    messages: Sequence[Any],
    max_tokens: int,
    thresholds: Any,
    *,
    lines_count: int | None = None,
    group_segments_count: int | None = None,
) -> bool:
    """
    判定是否為 heavy request（任一條件成立即為 heavy）。

    - total 估計 tokens（input + max_tokens）>= heavy_token_threshold
    - 任一 message content 長度 >= heavy_msg_char_threshold
    - lines_count >= heavy_lines_threshold（例如 JSON batch 行數）
    - group_segments_count >= heavy_group_segments_threshold（例如 group 內 segment 數）

    thresholds：具備屬性 heavy_token_threshold, heavy_msg_char_threshold,
    heavy_lines_threshold, heavy_group_segments_threshold 的物件或 dict。
    若為 dict 則用 thresholds["key"]。
    """
    def get(name: str, default: int) -> int:
        if isinstance(thresholds, dict):
            return int(thresholds.get(name, default))
        return int(getattr(thresholds, name, default))

    token_th = get("heavy_token_threshold", 2800)
    char_th = get("heavy_msg_char_threshold", 6000)
    lines_th = get("heavy_lines_threshold", 120)
    seg_th = get("heavy_group_segments_threshold", 4)

    tokens_in = 0
    for m in messages:
        c = _get_content(m)
        tokens_in += estimate_tokens_rough(c)
        if len(c) >= char_th:
            return True
    total = tokens_in + max_tokens
    if total >= token_th:
        return True
    if lines_count is not None and lines_count >= lines_th:
        return True
    if group_segments_count is not None and group_segments_count >= seg_th:
        return True
    return False


def is_heavy_with_reason(
    messages: Sequence[Any],
    max_tokens: int,
    thresholds: Any,
    *,
    lines_count: int | None = None,
    group_segments_count: int | None = None,
) -> tuple[bool, str]:
    """
    同 is_heavy，但回傳 (heavy, reason)。
    reason 為觸發條件： "msg_char" | "token" | "lines" | "segments"；未觸發為 ""。
    供 log 記錄 heavy 判定原因（效能瓶頸／崩潰來源定位）。
    """
    def get(name: str, default: int) -> int:
        if isinstance(thresholds, dict):
            return int(thresholds.get(name, default))
        return int(getattr(thresholds, name, default))

    token_th = get("heavy_token_threshold", 2800)
    char_th = get("heavy_msg_char_threshold", 6000)
    lines_th = get("heavy_lines_threshold", 120)
    seg_th = get("heavy_group_segments_threshold", 4)

    tokens_in = 0
    for m in messages:
        c = _get_content(m)
        tokens_in += estimate_tokens_rough(c)
        if len(c) >= char_th:
            return True, "msg_char"
    total = tokens_in + max_tokens
    if total >= token_th:
        return True, "token"
    if lines_count is not None and lines_count >= lines_th:
        return True, "lines"
    if group_segments_count is not None and group_segments_count >= seg_th:
        return True, "segments"
    return False, ""


T = TypeVar("T")


def chunk_list(seq: Sequence[T], chunk_size: int) -> List[List[T]]:
    """
    將大批次拆成固定大小的小批次。
    例如 local_polish 每 60 行一批：chunk_list(lines, 60)。
    """
    if chunk_size <= 0:
        chunk_size = 1
    out: List[List[T]] = []
    for i in range(0, len(seq), chunk_size):
        out.append(list(seq[i : i + chunk_size]))
    return out
