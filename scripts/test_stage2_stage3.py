"""
Minimal test for Stage2 JSON parse, PACK extraction, OUT validator, and Stage3 fallback.
Run from project root: python scripts/test_stage2_stage3.py
No model required; uses fixed JSON and mock.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import (
    Stage2Result,
    parse_pack_from_reasons,
    validate_out_block,
    _postprocess_stage3_output,
    _extract_json_from_raw,
    stage4_assemble_by_main,
    _fill_draft_with_suggestions,
)
from src.subtitle_item import SubtitleItem, BriefMeta


def test_stage2_json_parse():
    """Fixed input: curr with idiom; assert idiom_flag=yes, plain_en present."""
    raw = json.dumps({
        "idiom_flag": "yes",
        "idiom_span": "put all its chips on",
        "plain_en": "They invested everything in the War College.",
        "src_lit": "The Federation invested everything in the War College.",
        "ctx_brief": "Consequence of failure implied.",
        "referents": "",
        "scene_brief": "",
        "register": "R2",
        "metaphor": "M1",
        "style_lock": "one sentence",
        "terms": "",
        "main_conf": 0.85,
        "notes": "",
    }, ensure_ascii=False)
    # Reuse parsing logic: strip, load, extract plain_en, PACK in notes
    obj = json.loads(raw)
    meaning_en = str(obj.get("plain_en", obj.get("meaning_en", ""))).strip()
    pack_obj = {k: v for k, v in obj.items() if k not in ("notes",)}
    notes = "PACK:" + json.dumps(pack_obj, ensure_ascii=False)
    result = Stage2Result(meaning_en=meaning_en, notes=notes)
    assert result.meaning_en == "They invested everything in the War College.", result.meaning_en
    assert "idiom_flag" in result.notes and "yes" in result.notes
    print("[OK] Stage2 JSON parse: idiom_flag=yes, plain_en present")
    return result


def test_stage2_new_schema_parse():
    """translation_brief must be draft_tl (one-line subtitle); meaning_tl stays in PACK for LOCAL."""
    # Example: "Yeah. You could never be too careful." -> zh-TW
    raw = json.dumps({
        "target_language": "zh-TW",
        "tl_instruction": "請依語意產出一行台灣繁體中文字幕，可保留<I1> slot 由在地化模型填入片語。",
        "meaning_tl": "越小心越好",
        "draft_tl": "嗯，<I1>。",
        "idiom_requests": [
            {"slot": "I1", "meaning_tl": "越小心越好", "register": "R2", "max_len": 10}
        ],
        "idiom_flag": "yes",
        "idiom_span": "never be too careful",
        "ctx_brief": "說話者認同「再小心也不為過」。",
        "main_conf": 0.85,
    }, ensure_ascii=False)
    obj = json.loads(raw)
    # Same priority as stage2_reason_and_score: draft_tl first (subtitle line), then meaning_tl
    draft_tl = str(obj.get("draft_tl", "")).strip()
    meaning_tl = str(obj.get("meaning_tl", "")).strip()
    translation_brief = draft_tl if draft_tl else meaning_tl
    if not translation_brief:
        translation_brief = str(obj.get("plain_en", obj.get("meaning_en", ""))).strip()
    assert translation_brief == "嗯，<I1>。", f"translation_brief must be draft_tl, got {translation_brief!r}"
    assert meaning_tl == "越小心越好", "meaning_tl stays for PACK/LOCAL"
    assert len(obj.get("idiom_requests", [])) >= 1 and obj["idiom_requests"][0].get("meaning_tl") == "越小心越好"
    assert "zh-TW" in obj.get("tl_instruction", "") or "請" in obj.get("tl_instruction", "")
    print("[OK] Stage2 new schema: translation_brief=draft_tl, meaning_tl in PACK (zh-TW)")


def test_extract_json_and_pack_on_failure():
    """Truncated or wrapped JSON still yields PACK in reasons so Run E gets pack."""
    # Valid JSON with prefix -> _extract_json_from_raw returns dict
    raw_with_prefix = '  \n{"draft_tl":"嗯，<I1>。","meaning_tl":"越小心越好","main_conf":0.85}'
    obj = _extract_json_from_raw(raw_with_prefix)
    assert obj is not None and obj.get("draft_tl") == "嗯，<I1>。", obj
    assert obj.get("meaning_tl") == "越小心越好"
    # Truncated JSON (no closing }) -> raw_decode fails -> returns None; minimal PACK still used in stage2
    truncated = '{"draft_tl":"嗯，","meaning_tl":"越小'
    obj_trunc = _extract_json_from_raw(truncated)
    assert obj_trunc is None, "truncated JSON should not parse"
    # Simulate what stage2 does when no JSON: minimal PACK must be parseable by Run E
    minimal_pack = {
        "draft_tl": "fallback line",
        "meaning_tl": "fallback line",
        "main_conf": 0.5,
        "idiom_requests": [],
        "tl_instruction": "",
    }
    reasons = "PACK:" + json.dumps(minimal_pack, ensure_ascii=False)
    pack = parse_pack_from_reasons(reasons)
    assert pack is not None, "Run E must get pack from minimal PACK"
    assert pack.get("draft_tl") == "fallback line"
    assert pack.get("meaning_tl") == "fallback line"
    assert pack.get("idiom_requests") == []
    print("[OK] _extract_json_from_raw + minimal PACK → parse_pack_from_reasons")


def test_cache_pack_ok_behavior():
    """Old cache without PACK in reasons should force regenerate (cache_pack_ok False)."""
    # Same logic as app._cache_pack_ok: ok/total >= 0.90 only if reasons contain PACK
    def cache_pack_ok(loaded_items, version, min_ratio=0.90):
        if not loaded_items or version not in ("v1", "v2", "v3"):
            return False
        attr = "brief_v1" if version == "v1" else "brief_v2" if version == "v2" else "brief_v3"
        total = ok = 0
        for item in loaded_items.values():
            brief = getattr(item, attr, None)
            if brief is None:
                continue
            total += 1
            if parse_pack_from_reasons(brief.reasons or "") is not None:
                ok += 1
        return total > 0 and (ok / total) >= min_ratio

    # All items with brief_v1 but reasons without PACK -> False
    items_no_pack = {
        "s1": SubtitleItem(sub_id="s1", start_ms=0, end_ms=1000, text_raw="x", text_clean="x",
            brief_v1=BriefMeta(translation_brief="ab", reasons="(no PACK here)", version="v1")),
        "s2": SubtitleItem(sub_id="s2", start_ms=1000, end_ms=2000, text_raw="y", text_clean="y",
            brief_v1=BriefMeta(translation_brief="cd", reasons="legacy text", version="v1")),
    }
    assert cache_pack_ok(items_no_pack, "v1") is False, "cache without PACK must force regenerate"

    # All items with brief_v1 and reasons containing PACK -> True
    items_with_pack = {
        "s1": SubtitleItem(sub_id="s1", start_ms=0, end_ms=1000, text_raw="x", text_clean="x",
            brief_v1=BriefMeta(translation_brief="ab", reasons='PACK:{"draft_tl":"x","meaning_tl":"y"}', version="v1")),
        "s2": SubtitleItem(sub_id="s2", start_ms=1000, end_ms=2000, text_raw="y", text_clean="y",
            brief_v1=BriefMeta(translation_brief="cd", reasons='PACK:{"draft_tl":"z"}', version="v1")),
    }
    assert cache_pack_ok(items_with_pack, "v1") is True, "cache with PACK can be used"

    # 90% threshold: 2 with PACK, 1 without -> 2/3 < 0.9 -> False
    items_mixed = dict(items_with_pack)
    items_mixed["s3"] = SubtitleItem(sub_id="s3", start_ms=2000, end_ms=3000, text_raw="z", text_clean="z",
        brief_v1=BriefMeta(translation_brief="ef", reasons="no PACK", version="v1"))
    assert cache_pack_ok(items_mixed, "v1") is False, "below 90% PACK must force regenerate"
    print("[OK] cache_pack_ok: no PACK -> regenerate; with PACK -> use cache; <90% -> regenerate")


def test_parse_pack_from_reasons():
    """PACK in reasons -> dict; missing -> None."""
    pack = {"plain_en": "x", "src_lit": "y"}
    reasons = "PACK:" + json.dumps(pack)
    out = parse_pack_from_reasons(reasons)
    assert out is not None and out.get("plain_en") == "x", out
    assert parse_pack_from_reasons("") is None
    assert parse_pack_from_reasons("no pack here") is None
    print("[OK] parse_pack_from_reasons")


def test_validate_out_block():
    """Valid OUT -> ok; preamble / missing key -> not ok."""
    sub_id = "abc123"
    valid = "<OUT>\nid=abc123\ntgt=聯邦把一切押在戰爭學院上。\ntgt_alt=\nconf=0.85\n</OUT>"
    ok, parsed, reason = validate_out_block(valid, sub_id, simplified=False)
    assert ok, reason
    assert parsed.get("tgt") == "聯邦把一切押在戰爭學院上。"
    assert parsed.get("conf") == "0.85"
    print("[OK] validate_out_block valid")

    with_preamble = "以下是翻譯：\n" + valid
    ok2, _, reason2 = validate_out_block(with_preamble, sub_id, simplified=False)
    assert ok2, "validator should still find <OUT> and pass"
    print("[OK] validate_out_block with preamble (block still valid)")

    bad_id = "<OUT>\nid=wrong\ntgt=x\ntgt_alt=\nconf=0.5\n</OUT>"
    ok3, _, reason3 = validate_out_block(bad_id, sub_id, simplified=False)
    assert not ok3 and "id" in reason3.lower(), reason3
    print("[OK] validate_out_block id mismatch")

    simplified_block = "<OUT>\nid=abc123\ntgt=一句話\nconf=0.8\n</OUT>"
    ok4, parsed4, _ = validate_out_block(simplified_block, sub_id, simplified=True)
    assert ok4 and parsed4.get("tgt") == "一句話"
    print("[OK] validate_out_block simplified (no tgt_alt)")


def test_postprocess_out_extract():
    """_postprocess_stage3_output extracts tgt from <OUT>."""
    raw = "其他廢話\n<OUT>\nid=any\ntgt=這是翻譯結果\n</OUT>\n更多廢話"
    out = _postprocess_stage3_output(raw, "zh-TW", sub_id="")
    assert out == "這是翻譯結果", out
    print("[OK] _postprocess_stage3_output OUT extract")


def test_stage4_fill_draft():
    """_fill_draft_with_suggestions replaces <I1>/<I2> with suggestions."""
    draft = "嗯，<I1>。"
    suggestions = {"I1": "再小心也不為過"}
    out = _fill_draft_with_suggestions(draft, suggestions)
    assert out == "嗯，再小心也不為過。", out
    print("[OK] _fill_draft_with_suggestions I1 filled")


def test_stage4_assemble_csv_template_and_fallback():
    """stage4_assemble_by_main: CSV prompt_config replaces placeholders; fallback when no config."""
    captured = []

    class MockReasonModel:
        def chat(self, messages, temperature=0.1, max_tokens=256, json_mode=False):
            captured.append(list(messages))
            return "一行輸出。"

    mock = MockReasonModel()
    # With prompt_config (CSV-style): placeholders replaced
    class AssembleConfig:
        system_prompt_template = "Output ONE LINE only."
        user_prompt_template = (
            "Target: {target_language}\n"
            "English: {line_en}\n"
            "Context: {ctx_brief}\n"
            "Draft: {draft_prefilled}\n"
            "Suggestions: {suggestions_json}\n"
            "Output ONE LINE in {target_language}."
        )
    out_ja = stage4_assemble_by_main(
        mock,
        "You could never be too careful.",
        "說話者認同。",
        "嗯，<I1>。",
        {"I1": "再小心也不為過"},
        "ja-JP",
        prompt_config=AssembleConfig(),
        role="main_assemble",
    )
    assert out_ja == "一行輸出。", out_ja
    assert len(captured) == 1
    user_content = captured[0][1]["content"]
    assert "ja-JP" in user_content, "target_language placeholder should be ja-JP"
    assert "draft_prefilled" not in user_content, "placeholder must be replaced"
    assert "嗯，再小心也不為過。" in user_content, "draft_prefilled should be filled draft"
    print("[OK] stage4_assemble_by_main CSV template (ja-JP) placeholders replaced")

    captured.clear()
    # Fallback: no prompt_config -> hardcoded prompt
    out_es = stage4_assemble_by_main(
        mock,
        "Never be too careful.",
        "context",
        "Pues, <I1>.",
        {"I1": "nunca es demasiado"},
        "es-ES",
        prompt_config=None,
    )
    assert out_es == "一行輸出。", out_es
    assert len(captured) == 1
    user_content_fb = captured[0][1]["content"]
    assert "es-ES" in user_content_fb, "fallback should still use target_language es-ES"
    assert "Pues, nunca es demasiado." in user_content_fb, "fallback draft filled"
    print("[OK] stage4_assemble_by_main fallback (es-ES) one line")

    # reason_model=None -> return filled draft, no chat
    out_none = stage4_assemble_by_main(
        None,
        "x",
        "",
        "嗯，<I1>。",
        {"I1": "好"},
        "zh-TW",
    )
    assert out_none == "嗯，好。", out_none
    print("[OK] stage4_assemble_by_main reason_model=None returns filled draft")


if __name__ == "__main__":
    test_stage2_json_parse()
    test_stage2_new_schema_parse()
    test_extract_json_and_pack_on_failure()
    test_cache_pack_ok_behavior()
    test_parse_pack_from_reasons()
    test_validate_out_block()
    test_postprocess_out_extract()
    test_stage4_fill_draft()
    test_stage4_assemble_csv_template_and_fallback()
    print("\nAll checks passed.")
