"""
Run E fallback 拆分工具：僅在 main_group_translate 回傳格式錯或長度不符時使用。

提供：strip_punctuation、tokenize_preserving_terms、split_tokens_by_duration、safe_fallback_split。
"""

from __future__ import annotations
import re
from typing import List


# 全半形標點與常見符號（移除後壓縮空白、trim、單行）
_PUNCT_STR = (
    ".,!?;:\"'()[]{}<>"
    "\u2026\u2014\u3001\uff0c\u3002\uff01\uff1f\uff1b\uff1a"
    "\u300e\u300f\u300c\u300d\uff08\uff09\u300a\u300b\u3010\u3011"
    "\u301c\u00b7\u2022 \t\n\r"
)
PUNCTUATION_PATTERN = re.compile(
    "[" + re.escape(_PUNCT_STR) + "]+",
    re.UNICODE,
)

# Placeholder 樣式：<I1>, <I2>, __P0__, __P1__ 等
PLACEHOLDER_PATTERN = re.compile(r"<I\d+>|__P\d+__", re.IGNORECASE)

# 連續拉丁字母/數字
LATIN_DIGITS_PATTERN = re.compile(r"[A-Za-z0-9]+")

# CJK 連續區段（中文、平假名、片假名、韓文）
CJK_RUN_PATTERN = re.compile(
    r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]+",
    re.UNICODE,
)


# 保護小數與縮寫時使用的佔位符前綴（還原時依序替回）
_DEC_PLACEHOLDER_PREFIX = "__DEC"
_ACR_PLACEHOLDER_PREFIX = "__ACR"


def strip_punctuation(
    text: str,
    *,
    keep_decimal: bool = True,
    keep_acronym: bool = True,
) -> str:
    """
    移除常見全半形標點，壓縮連續空白為單空白，trim，回傳單行。
    - keep_decimal: 為 True 時保護小數（如 3.14），避免變成 3 14
    - keep_acronym: 為 True 時保護縮寫（如 U.S.），避免變成 U S
    """
    if not text or not isinstance(text, str):
        return ""
    s = text
    saved_dec: List[str] = []
    saved_acr: List[str] = []

    if keep_decimal:
        # 保護 \d+\.\d+（小數）
        def _save_dec(m: re.Match) -> str:
            saved_dec.append(m.group(0))
            return f"{_DEC_PLACEHOLDER_PREFIX}{len(saved_dec)-1}__"
        s = re.sub(r"\d+\.\d+", _save_dec, s)

    if keep_acronym:
        # 保護 A.B. 或 U.S. 等（大寫字母+點，至少兩段；結尾可選一點以保留 "U.S."）
        def _save_acr(m: re.Match) -> str:
            saved_acr.append(m.group(0))
            return f"{_ACR_PLACEHOLDER_PREFIX}{len(saved_acr)-1}__"
        s = re.sub(r"\b(?:[A-Z]\.)+[A-Z]?\.?", _save_acr, s)

    # 移除標點與多餘空白
    s = PUNCTUATION_PATTERN.sub(" ", s)
    s = " ".join(s.split())
    s = s.strip()

    # 還原縮寫（先還原後匹配的，避免索引錯位）
    for i in range(len(saved_acr) - 1, -1, -1):
        s = s.replace(f"{_ACR_PLACEHOLDER_PREFIX}{i}__", saved_acr[i])
    for i in range(len(saved_dec) - 1, -1, -1):
        s = s.replace(f"{_DEC_PLACEHOLDER_PREFIX}{i}__", saved_dec[i])

    return s


def tokenize_preserving_terms(text: str, protected: List[str]) -> List[str]:
    """
    將文字切成 token，以下視為不可拆：
    - protected terms（如 glossary 目標詞、lock terms）
    - 連續拉丁字母/數字
    - placeholder（<I1>、__P0__ 等）
    - 連續 CJK 一段（不單字切）
    其餘以空白或標點為界，產出較粗 token。
    回傳順序與原文一致。
    """
    if not text or not isinstance(text, str):
        return []
    protected = [p for p in protected if p]
    # 依長度降序，長詞先匹配避免短詞吃掉長詞
    protected_sorted = sorted(set(protected), key=len, reverse=True)
    tokens: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        # 1) 受保護詞（最長匹配）
        matched = False
        for term in protected_sorted:
            if text[i : i + len(term)] == term:
                tokens.append(term)
                i += len(term)
                matched = True
                break
        if matched:
            continue
        # 2) Placeholder
        m = PLACEHOLDER_PATTERN.match(text[i:])
        if m:
            tokens.append(m.group(0))
            i += m.end()
            continue
        # 3) 拉丁/數字
        m = LATIN_DIGITS_PATTERN.match(text[i:])
        if m:
            tokens.append(m.group(0))
            i += m.end()
            continue
        # 4) CJK 一段
        m = CJK_RUN_PATTERN.match(text[i:])
        if m:
            tokens.append(m.group(0))
            i += m.end()
            continue
        # 5) 空白/標點：跳過（當分隔）
        if text[i].isspace() or _is_punct(text[i]):
            i += 1
            continue
        # 6) 其他（單字或符號）：當一個 token 避免死循環
        tokens.append(text[i])
        i += 1
    return tokens


def _is_punct(c: str) -> bool:
    if not c:
        return False
    cp = ord(c)
    return (
        cp <= 0x7F and c in ".,!?;:\"\'()[]{}<>"
        or 0x3000 <= cp <= 0x303F  # CJK 標點
        or cp in (0x00B7, 0x2026, 0x2014, 0x2212)
    )


def split_tokens_by_duration(
    tokens: List[str],
    durations_ms: List[float],
) -> List[str]:
    """
    依 duration 比例把 token 分到每段；至少每段一個 token，最後段吃剩下。
    絕不拆開 token。回傳每段拼成的一行字串（不自動 strip_punctuation）。
    """
    n_seg = len(durations_ms)
    if n_seg == 0:
        return []
    if not tokens:
        return [""] * n_seg
    if len(tokens) < n_seg:
        # 不夠分：前 len(tokens) 段各 1 個，其餘段空字串
        lines: List[str] = []
        t = 0
        for i in range(n_seg):
            if t < len(tokens):
                lines.append(tokens[t])
                t += 1
            else:
                lines.append("")
        return lines
    total_ms = sum(durations_ms)
    if total_ms <= 0:
        # 均分
        base = len(tokens) // n_seg
        remainder = len(tokens) % n_seg
        counts = [base + (1 if i < remainder else 0) for i in range(n_seg)]
    else:
        # 先每段至少 1，剩餘依比例分配，最後段吃餘數
        counts = [1] * n_seg
        remainder = len(tokens) - n_seg
        if remainder > 0:
            extra = [int(remainder * d / total_ms) for d in durations_ms]
            for i in range(n_seg):
                counts[i] += extra[i]
            counts[-1] += remainder - sum(extra)
    idx = 0
    lines = []
    for c in counts:
        seg_tokens = tokens[idx : idx + c]
        idx += c
        lines.append(" ".join(seg_tokens) if seg_tokens else "")
    return lines


def safe_fallback_split(
    group_text: str,
    segments: List[dict],
    protected_terms: List[str],
) -> dict[str, str]:
    """
    先 tokenize_preserving_terms，再 split_tokens_by_duration，回傳 sub_id -> 單行字串。
    segments 每項需有 "id"（sub_id）與 "ms" 或 "duration_ms"（毫秒）。
    """
    if not segments:
        return {}
    ids = [s.get("id") or s.get("sub_id") or "" for s in segments]
    durations = [
        float(s.get("ms", s.get("duration_ms", 0)) or 0)
        for s in segments
    ]
    tokens = tokenize_preserving_terms(group_text or "", protected_terms or [])
    lines = split_tokens_by_duration(tokens, durations)
    out: dict[str, str] = {}
    for i, sub_id in enumerate(ids):
        if not sub_id:
            continue
        line = lines[i] if i < len(lines) else ""
        out[sub_id] = strip_punctuation(line) if line else ""
    return out
