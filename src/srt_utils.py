from __future__ import annotations
import pysrt

def load_srt(path: str, encoding="utf-8"):
    return pysrt.open(path, encoding=encoding)

def save_srt(subs: pysrt.SubRipFile, path: str, encoding="utf-8"):
    subs.save(path, encoding=encoding)

def sub_midpoints_ms(sub):
    start_ms = (sub.start.hours * 3600 + sub.start.minutes * 60 + sub.start.seconds) * 1000 + sub.start.milliseconds
    end_ms = (sub.end.hours * 3600 + sub.end.minutes * 60 + sub.end.seconds) * 1000 + sub.end.milliseconds
    return start_ms, end_ms, (start_ms + end_ms) / 2.0

def clean_srt_text(t: str) -> str:
    # Keep it simple for CC subtitles; preserve punctuation; collapse whitespace
    return " ".join(t.replace("\n", " ").split())
