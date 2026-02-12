from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Any
import json
import re

from .srt_utils import clean_srt_text
from .glossary import apply_glossary_post, GlossaryEntry
from .safe_infer import chat_dispatch


def _messages_for_chat(prompt_config: Any, sys_content: str, user_content: str) -> list:
    """Build messages for chat_dispatch. Gemma has no system role: merge into first user turn."""
    chat_format = getattr(prompt_config, "chat_format", "chatml") if prompt_config else "chatml"
    sys_content = (sys_content or "").strip()
    user_content = (user_content or "").strip()
    if chat_format == "gemma":
        combined = (sys_content + "\n\n" + user_content).strip() if sys_content else user_content
        return [{"role": "user", "content": combined}]
    if not sys_content:
        return [{"role": "user", "content": user_content}]
    return [{"role": "system", "content": sys_content}, {"role": "user", "content": user_content}]


@dataclass
class Stage2Result:
    meaning_en: str  # plain_en (or legacy meaning_en)
    notes: str = ""  # may contain "PACK:{...}" for Stage3
    need_vision: Optional[bool] = None  # v1: 是否需視覺
    need_multi_frame_vision: Optional[bool] = None  # v2: 是否需多張視覺
    need_more_context: Optional[bool] = None  # 是否需更多前後文（Run F 方案 C）

def _parse_bool_from_json(val: Any) -> Optional[bool]:
    """Parse boolean from JSON value (bool, str 'true'/'false', int 0/1)."""
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() in ("true", "1", "yes") if val.lower() not in ("false", "0", "no", "") else False
    try:
        return bool(int(val))
    except (TypeError, ValueError):
        return None


def stage2_reason_and_score(
    reason_model,
    line_en: str,
    prev_lines_en: list[str],
    next_lines_en: list[str],
    visual_hint: str | None = None,
    audio_hint: str | None = None,
    prompt_config: Any | None = None,  # ModelPromptConfig from CSV
    target_language: str = "zh-TW",
    brief_version: str = "v1",  # v1 -> need_vision only; v2 -> need_multi_frame_vision only; v3 -> need_more_context only
    log_lines: Optional[list[str]] = None,
    log_label: str = "Stage2",
) -> Stage2Result:
    # Ask the model to output STRICT JSON with meaning; 每階段只要求一個 need（見 brief_version）。
    # Using llama-cpp-python JSON mode.
    # --- Context packing policy ---
    # The user requirement is:
    #   - For every Stage2 input, include exactly 1 previous sentence and 1 next sentence
    #     (except the very first/last subtitle line where one side doesn't exist).
    # We still allow callers to pass wider context; we will surface it in separate sections
    # without changing the required immediate-prev/next.

    prev_1 = prev_lines_en[-1].strip() if prev_lines_en else ""
    next_1 = next_lines_en[0].strip() if next_lines_en else ""

    prev_more = [x.strip() for x in (prev_lines_en[:-1] if len(prev_lines_en) > 1 else []) if x.strip()]
    next_more = [x.strip() for x in (next_lines_en[1:] if len(next_lines_en) > 1 else []) if x.strip()]

    # 建立 context 字串（用於 placeholder 替換）
    context_parts = []
    if prev_1:
        context_parts.append(f"[Prev-1]\n{prev_1}")
    context_parts.append(f"[Current]\n{line_en}")
    if next_1:
        context_parts.append(f"[Next-1]\n{next_1}")
    if prev_more:
        context_parts.append("[Prev-More]\n" + "\n".join([f"- {x}" for x in prev_more]))
    if next_more:
        context_parts.append("[Next-More]\n" + "\n".join([f"- {x}" for x in next_more]))
    
    # Audio cues（音訊線索）- 固定加入，即使為空也標示
    audio_cues_text = audio_hint if audio_hint else "(none)"
    context_parts.append(f"[Audio cues (from speech)]\n{audio_cues_text}")
    
    # Debug log：記錄 audio_hint 是否被注入（僅在前幾句或非空時記錄）
    # 這個 log 會在調用端顯示，這裡只是確保 audio_hint 被正確處理
    
    if visual_hint:
        context_parts.append(f"[Visual Hint]\n{visual_hint}")
    context_str = "\n\n".join(context_parts)

    # 必須提供 CSV prompt 設定，否則拋出錯誤（完全依賴 CSV）
    if not prompt_config or not hasattr(prompt_config, "system_prompt_template") or not hasattr(prompt_config, "user_prompt_template"):
        raise ValueError(
            f"Stage2 prompt config missing. Please ensure model_prompts.csv contains a row "
            f"with role='main' and model_name matching the reason model filename."
        )
    
    # 從 CSV 載入的 prompt（完全參照 CSV）
    sys_content = prompt_config.system_prompt_template
    user_content = prompt_config.user_prompt_template
    
    # 替換 placeholder
    user_content = user_content.replace("{line}", line_en)
    user_content = user_content.replace("{context}", context_str)
    user_content = user_content.replace("{target_language}", target_language)
    if visual_hint:
        user_content = user_content.replace("{visual_hint}", visual_hint)
    else:
        user_content = user_content.replace("{visual_hint}", "")
    
    # 替換 audio_hint placeholder（如果 CSV prompt 中有使用）
    audio_hint_text = audio_hint if audio_hint else "(none)"
    user_content = user_content.replace("{audio_hint}", audio_hint_text)
    
    # 確保最後有 JSON 輸出要求（如果 CSV prompt 沒有的話）
    if "json" not in user_content.lower() and "meaning_tl" not in user_content.lower() and "plain_en" not in user_content.lower():
        user_content += "\n\nReturn JSON ONLY with keys: target_language, tl_instruction, meaning_tl, draft_tl, idiom_requests, ctx_brief. All brief content (meaning_tl, draft_tl, tl_instruction, ctx_brief) must be in ENGLISH only (language-neutral). Do NOT translate into the target language in Stage 2; translation is done in a later stage. Optional: transliteration_requests (array of strings), omit_sfx (boolean). Legacy: plain_en/meaning_en also accepted."

    # 每階段只輸出一個 need，減輕模型負擔。v1→need_vision, v2→need_multi_frame_vision, v3→need_more_context。CSV 維持完全體供使用者一次編輯。
    if brief_version == "v1":
        user_content += "\n\nOutput need_vision (boolean). Rules: need_vision = true if accurate translation would benefit from or require visual context (e.g. who is speaking, spatial layout, on-screen text, action that disambiguates meaning, or anything that cannot be inferred from text and audio alone). need_vision = false if text + context + audio are sufficient for accurate translation."
    elif brief_version == "v2":
        user_content += "\n\nOutput need_multi_frame_vision (boolean). Rules: need_multi_frame_vision = true if single-frame vision was not enough (e.g. motion, change over time, multiple key moments need to be seen to disambiguate). need_multi_frame_vision = false if one frame was sufficient for accurate translation."
    elif brief_version == "v3":
        user_content += "\n\nOutput need_more_context (boolean). Rules: need_more_context = true if resolving referents, tone, or scene (what 'this'/'that' refers to, sarcasm vs sincere, what happened) would need more surrounding lines or cross-sentence context than the single prev/next line provided. need_more_context = false if current context is sufficient."

    messages = _messages_for_chat(prompt_config, sys_content, user_content)
    raw = chat_dispatch(
        reason_model,
        messages,
        max_tokens=600,
        temperature=0.1,
        json_mode=True,
        log_lines=log_lines,
        label=log_label or "Stage2",
    )

    try:
        cleaned_raw = (raw or "").strip()
        if cleaned_raw.startswith("```json"):
            cleaned_raw = cleaned_raw[7:].strip()
        if cleaned_raw.startswith("```"):
            cleaned_raw = cleaned_raw[3:].strip()
        if cleaned_raw.endswith("```"):
            cleaned_raw = cleaned_raw[:-3].strip()

        obj = json.loads(cleaned_raw)

        # translation_brief 必須是英文 brief，供 Stage 3 使用。若模型在 v1 誤將 draft_tl 翻成目標語，改用 meaning_tl。
        draft_tl = str(obj.get("draft_tl", "")).strip()
        meaning_tl = str(obj.get("meaning_tl", "")).strip()
        meaning_en = draft_tl if draft_tl else meaning_tl
        if meaning_en and _is_likely_target_language(meaning_en, target_language) and meaning_tl:
            meaning_en = meaning_tl  # v1 不應輸出翻譯；改用英文 meaning_tl 作為 translation_brief
        if meaning_en and _is_likely_target_language(meaning_en, target_language):
            meaning_en = line_en  # 若仍為目標語則強制用原文，確保 brief 全英文
        if not meaning_en:
            meaning_en = str(obj.get("plain_en", obj.get("meaning_en", ""))).strip()
        if not meaning_en:
            meaning_en = str(obj.get("meaning_en", "")).strip()
        meaning_en = _clean_stage2_output(meaning_en)
        if not meaning_en:
            meaning_en = _clean_stage2_output(cleaned_raw[:500])
        if meaning_en and _is_likely_target_language(meaning_en, target_language):
            meaning_en = line_en

        # tl_instruction: must be English-only; replace with default if model output contains target-language text (e.g. CJK)
        tl_instruction = _sanitize_tl_instruction(str(obj.get("tl_instruction", "")).strip(), target_language)
        if not tl_instruction:
            tl_instruction = _default_tl_instruction(target_language)
        obj = dict(obj)
        obj["tl_instruction"] = tl_instruction

        # 每階段只解析該階段的一個 need
        need_vision = _parse_bool_from_json(obj.get("need_vision")) if brief_version == "v1" else None
        need_multi_frame_vision = _parse_bool_from_json(obj.get("need_multi_frame_vision")) if brief_version == "v2" else None
        need_more_context = _parse_bool_from_json(obj.get("need_more_context")) if brief_version == "v3" else None
        if brief_version == "v1" and need_vision is None:
            need_vision = False
        if brief_version == "v2" and need_multi_frame_vision is None:
            need_multi_frame_vision = False
        if brief_version == "v3" and need_more_context is None:
            need_more_context = False

        notes = str(obj.get("notes", "")).strip()
        # PACK: full JSON for Stage3; 此階段只寫入該版本的一個 need
        pack_obj = {k: v for k, v in obj.items() if k not in ("notes",)}
        if need_vision is not None:
            pack_obj["need_vision"] = need_vision
        if need_multi_frame_vision is not None:
            pack_obj["need_multi_frame_vision"] = need_multi_frame_vision
        if need_more_context is not None:
            pack_obj["need_more_context"] = need_more_context
        pack_str = json.dumps(pack_obj, ensure_ascii=False)
        if pack_str and "PACK:" not in notes:
            notes = f"PACK:{pack_str}" if not notes else (notes + "\nPACK:" + pack_str)

        return Stage2Result(meaning_en=meaning_en, notes=notes, need_vision=need_vision, need_multi_frame_vision=need_multi_frame_vision, need_more_context=need_more_context)
    except Exception:
        # Always try to emit PACK so RunF can parse_pack_from_reasons(); avoid pack=None chain.
        obj = _extract_json_from_raw(raw)
        if obj is not None:
            draft_tl = str(obj.get("draft_tl", "")).strip()
            meaning_tl = str(obj.get("meaning_tl", "")).strip()
            meaning_en = draft_tl or meaning_tl
            if meaning_en and _is_likely_target_language(meaning_en, target_language) and meaning_tl:
                meaning_en = meaning_tl  # v1 不應輸出翻譯；改用英文
            if meaning_en and _is_likely_target_language(meaning_en, target_language):
                meaning_en = line_en
            if not meaning_en:
                meaning_en = str(obj.get("plain_en", obj.get("meaning_en", ""))).strip()
            if not meaning_en:
                meaning_en = str(obj.get("meaning_en", "")).strip()
            meaning_en = _clean_stage2_output(meaning_en)
            if not meaning_en:
                meaning_en = _clean_stage2_output((raw or "").strip()[:500])
            if meaning_en and _is_likely_target_language(meaning_en, target_language):
                meaning_en = line_en
            pack_obj = {k: v for k, v in obj.items() if k != "notes"}
            pack_obj.setdefault("draft_tl", "")
            pack_obj.setdefault("meaning_tl", "")
            pack_obj.setdefault("idiom_requests", [])
            ti = _sanitize_tl_instruction(str(pack_obj.get("tl_instruction", "")).strip(), target_language)
            pack_obj["tl_instruction"] = ti if ti else _default_tl_instruction(target_language)
            pack_obj.setdefault("transliteration_requests", [])
            pack_obj.setdefault("omit_sfx", False)
            need_vision = _parse_bool_from_json(obj.get("need_vision")) if brief_version == "v1" else None
            need_multi_frame_vision = _parse_bool_from_json(obj.get("need_multi_frame_vision")) if brief_version == "v2" else None
            need_more_context = _parse_bool_from_json(obj.get("need_more_context")) if brief_version == "v3" else None
            if brief_version == "v1" and need_vision is None:
                need_vision = False
            if brief_version == "v2" and need_multi_frame_vision is None:
                need_multi_frame_vision = False
            if brief_version == "v3" and need_more_context is None:
                need_more_context = False
            if need_vision is not None:
                pack_obj["need_vision"] = need_vision
            if need_multi_frame_vision is not None:
                pack_obj["need_multi_frame_vision"] = need_multi_frame_vision
            if need_more_context is not None:
                pack_obj["need_more_context"] = need_more_context
            notes = "PACK:" + json.dumps(pack_obj, ensure_ascii=False)
            return Stage2Result(meaning_en=meaning_en, notes=notes, need_vision=need_vision, need_multi_frame_vision=need_multi_frame_vision, need_more_context=need_more_context)
        # No JSON at all: minimal PACK；此階段只寫入該版本的一個 need
        meaning_en = _clean_stage2_output((raw or "").strip()[:500]) or "(parse_failed)"
        minimal_pack = {
            "draft_tl": meaning_en,
            "meaning_tl": meaning_en or "",
            "idiom_requests": [],
            "tl_instruction": "",
            "transliteration_requests": [],
            "omit_sfx": False,
        }
        if brief_version == "v1":
            minimal_pack["need_vision"] = False
        elif brief_version == "v2":
            minimal_pack["need_multi_frame_vision"] = False
        elif brief_version == "v3":
            minimal_pack["need_more_context"] = False
        notes = "PACK:" + json.dumps(minimal_pack, ensure_ascii=False)
        return Stage2Result(
            meaning_en=meaning_en, notes=notes,
            need_vision=False if brief_version == "v1" else None,
            need_multi_frame_vision=False if brief_version == "v2" else None,
            need_more_context=False if brief_version == "v3" else None,
        )


def stage2_reason_and_score_batch(
    reason_model,
    items_batch: list[tuple[str, Any]],  # [(sub_id, SubtitleItem), ...]
    sorted_all: list[tuple[str, Any]],  # full sorted list for prev/next lookup
    target_language: str,
    prompt_config: Any,
    vision_hint_map: Optional[dict[str, str]] = None,
    brief_version: str = "v1",  # v1 -> need_vision only; v2 -> need_multi_frame_vision only; v3 -> need_more_context only
    log_lines: Optional[list[str]] = None,
    log_label: str = "Stage2_batch",
) -> Optional[dict[str, Stage2Result]]:
    """
    Multi-item in one request: single chat with array of items, output JSON {"items": [...]}.
    Returns dict[sub_id, Stage2Result] on success; None on parse failure or length mismatch (caller should binary split).
    """
    if not items_batch:
        return {}
    batch_template = getattr(prompt_config, "batch_user_prompt_template", None) or ""
    if not batch_template.strip():
        return None
    sub_id_to_idx = {sub_id: i for i, (sub_id, _) in enumerate(sorted_all)}
    items_in = []
    for sub_id, item in items_batch:
        idx = sub_id_to_idx.get(sub_id, -1)
        prev_1 = ""
        next_1 = ""
        if 0 <= idx < len(sorted_all):
            if idx > 0:
                _, prev_item = sorted_all[idx - 1]
                prev_1 = (prev_item.text_clean or prev_item.text_raw or "").strip()
            if idx + 1 < len(sorted_all):
                _, next_item = sorted_all[idx + 1]
                next_1 = (next_item.text_clean or next_item.text_raw or "").strip()
        en = (item.text_clean or item.text_raw or "").strip()
        audio = item.get_audio_hint() if hasattr(item, "get_audio_hint") else ""
        if audio == "(none)":
            audio = ""
        visual = (vision_hint_map or {}).get(sub_id, "") or ""
        items_in.append({
            "id": sub_id,
            "prev": prev_1,
            "en": en,
            "next": next_1,
            "audio": audio,
            "visual": visual,
        })
    items_batch_json = json.dumps(items_in, ensure_ascii=False)
    sys_content = prompt_config.system_prompt_template
    user_content = batch_template.replace("{items_batch_json}", items_batch_json).replace("{target_language}", target_language)
    # 每階段只輸出一個 need（v1/v2/v3 各一）
    if brief_version == "v1":
        user_content += "\n\nEach output item MUST include need_vision (boolean). need_vision = true if accurate translation would benefit from or require visual context (e.g. who is speaking, spatial layout, on-screen text, action that disambiguates meaning). need_vision = false if text + context + audio are sufficient."
    elif brief_version == "v2":
        user_content += "\n\nEach output item MUST include need_multi_frame_vision (boolean). need_multi_frame_vision = true if single-frame vision was not enough (e.g. motion, change over time, multiple key moments need to be seen). need_multi_frame_vision = false if one frame was sufficient."
    elif brief_version == "v3":
        user_content += "\n\nEach output item MUST include need_more_context (boolean). need_more_context = true if resolving referents, tone, or scene would need more surrounding lines than the single prev/next provided; false if current context is sufficient."
    messages = _messages_for_chat(prompt_config, sys_content, user_content)
    raw = chat_dispatch(
        reason_model,
        messages,
        max_tokens=max(800, 200 * len(items_batch)),
        temperature=0.1,
        json_mode=True,
        log_lines=log_lines,
        label=log_label,
    )
    try:
        cleaned_raw = (raw or "").strip()
        if cleaned_raw.startswith("```json"):
            cleaned_raw = cleaned_raw[7:].strip()
        if cleaned_raw.startswith("```"):
            cleaned_raw = cleaned_raw[3:].strip()
        if cleaned_raw.endswith("```"):
            cleaned_raw = cleaned_raw[:-3].strip()
        obj = json.loads(cleaned_raw)
        items_out = obj.get("items")
        if not isinstance(items_out, list) or len(items_out) != len(items_batch):
            return None
        result_map = {}
        for i, out in enumerate(items_out):
            if not isinstance(out, dict):
                return None
            sub_id = out.get("id")
            if sub_id is None:
                sub_id = items_batch[i][0]
            draft_tl = str(out.get("draft_tl", "")).strip()
            meaning_tl = str(out.get("meaning_tl", "")).strip()
            meaning_en = draft_tl or meaning_tl
            if meaning_en and _is_likely_target_language(meaning_en, target_language) and meaning_tl:
                meaning_en = meaning_tl  # v1 不應輸出翻譯；改用英文 meaning_tl
            if meaning_en and _is_likely_target_language(meaning_en, target_language):
                meaning_en = (items_batch[i][1].text_clean or "").strip() or meaning_en
            if not meaning_en:
                meaning_en = str(out.get("plain_en", out.get("meaning_en", ""))).strip()
            meaning_en = _clean_stage2_output(meaning_en)
            if meaning_en and _is_likely_target_language(meaning_en, target_language):
                meaning_en = (items_batch[i][1].text_clean or "").strip() or meaning_en
            tl_instruction = _sanitize_tl_instruction(str(out.get("tl_instruction", "")).strip(), target_language)
            if not tl_instruction:
                tl_instruction = _default_tl_instruction(target_language)
            need_vision = _parse_bool_from_json(out.get("need_vision")) if brief_version == "v1" else None
            need_multi_frame_vision = _parse_bool_from_json(out.get("need_multi_frame_vision")) if brief_version == "v2" else None
            need_more_context = _parse_bool_from_json(out.get("need_more_context")) if brief_version == "v3" else None
            if brief_version == "v1" and need_vision is None:
                need_vision = False
            if brief_version == "v2" and need_multi_frame_vision is None:
                need_multi_frame_vision = False
            if brief_version == "v3" and need_more_context is None:
                need_more_context = False
            pack_obj = {k: v for k, v in out.items() if k != "notes"}
            pack_obj.setdefault("tl_instruction", tl_instruction)
            pack_obj.setdefault("draft_tl", draft_tl)
            pack_obj.setdefault("meaning_tl", meaning_tl)
            pack_obj.setdefault("idiom_requests", out.get("idiom_requests", []))
            pack_obj.setdefault("transliteration_requests", out.get("transliteration_requests", []))
            pack_obj.setdefault("omit_sfx", out.get("omit_sfx", False))
            if need_vision is not None:
                pack_obj["need_vision"] = need_vision
            if need_multi_frame_vision is not None:
                pack_obj["need_multi_frame_vision"] = need_multi_frame_vision
            if need_more_context is not None:
                pack_obj["need_more_context"] = need_more_context
            notes = "PACK:" + json.dumps(pack_obj, ensure_ascii=False)
            result_map[sub_id] = Stage2Result(meaning_en=meaning_en, notes=notes, need_vision=need_vision, need_multi_frame_vision=need_multi_frame_vision, need_more_context=need_more_context)
        return result_map
    except Exception:
        return None


def _build_stage3_out_prompt(
    sub_id: str,
    meaning_en: str,
    pack: Optional[dict],
    prev_block: str,
    target_language: str,
    simplified: bool,
    retry_harder: bool,
) -> tuple[str, str]:
    """Build system and user prompt for OUT-block Stage3. Returns (sys_content, user_content)."""
    sys_content = (
        "You MUST output ONLY a single block in this exact format. No other text before or after.\n"
        "<OUT>\n"
        "id={{id}}\n"
        "tgt={{tgt}}\n"
        + ("tgt_alt={{tgt_alt}}\n" if not simplified else "")
        + "conf={{conf}}\n"
        "</OUT>\n"
        "Rules: tgt = one line, no newlines, no explanation. conf = copy main_conf (0-1). "
        "Do NOT output '翻譯：' or '以下是' or any preamble."
    )
    if retry_harder:
        sys_content += "\n\nOutput ONLY the <OUT>...</OUT> block. Nothing else."

    main_conf = "0.8"
    if pack and pack.get("main_conf") is not None:
        try:
            main_conf = str(float(pack.get("main_conf")))
        except (TypeError, ValueError):
            main_conf = "0.8"

    parts = [
        f"id={sub_id}",
        f"meaning_tl={meaning_en[:200]}",
        f"prev_zh={prev_block[:300] if prev_block else '(none)'}",
        f"main_conf={main_conf}",
        f"target_lang={target_language}",
    ]
    if pack:
        parts.append(f"tl_instruction={str(pack.get('tl_instruction', ''))[:400]}")
        parts.append(f"draft_tl={str(pack.get('draft_tl', ''))[:200]}")
        parts.append(f"ctx_brief={str(pack.get('ctx_brief', ''))[:120]}")
        parts.append(f"src_lit={str(pack.get('src_lit', ''))[:200]}")
        parts.append(f"register={str(pack.get('register', ''))}")
        parts.append(f"metaphor={str(pack.get('metaphor', ''))}")
        parts.append(f"style_lock={str(pack.get('style_lock', ''))[:100]}")
        parts.append(f"terms={str(pack.get('terms', ''))[:150]}")
    user_content = "\n".join(parts)
    user_content += "\n\nFill <OUT> with: id=" + sub_id + ", tgt=<one-line translation>, "
    if not simplified:
        user_content += "tgt_alt=<optional alternate>, "
    user_content += "conf=" + main_conf + ". Output ONLY the <OUT> block."
    return sys_content, user_content


def stage3_translate(
    translate_model,
    line_en: str,
    meaning_en: str,
    prev_zh: list[str],
    glossary_entries: list[GlossaryEntry],
    target_language: str,
    prompt_config: Any | None = None,
    log_lines: Optional[list[str]] = None,
    log_label: str = "",
    sub_id: str = "",
    pack: Optional[dict] = None,
    use_out_template: bool = True,
    simplified_out: bool = False,
    retry_harder: bool = False,
) -> str:
    """
    Stage 3: localization. When pack is provided and use_out_template, outputs <OUT> block and returns tgt.
    Otherwise uses CSV prompt and legacy _postprocess_stage3_output.
    Glossary is not injected into the model; output-side replacement is done in run_final_translate.
    """
    prev_block = "\n".join(prev_zh[-2:]) if prev_zh else ""
    if not prompt_config or not hasattr(prompt_config, "system_prompt_template") or not hasattr(prompt_config, "user_prompt_template"):
        raise ValueError(
            "Stage3 prompt config missing. Ensure model_prompts_run_f.csv (or model_prompts.csv) has a row for the translate model with role='localization'."
        )

    use_phrase_suggestion = bool(
        prompt_config
        and ("{tl_instruction}" in (prompt_config.user_prompt_template or "") or "{requests_json}" in (prompt_config.user_prompt_template or ""))
    )

    if pack is not None and use_out_template and sub_id:
        use_phrase_suggestion = False
        sys_content, user_content = _build_stage3_out_prompt(
            sub_id, meaning_en, pack, prev_block, target_language, simplified_out, retry_harder
        )
    else:
        sys_content = prompt_config.system_prompt_template
        user_content = prompt_config.user_prompt_template
        use_phrase_suggestion = "{tl_instruction}" in user_content or "{requests_json}" in user_content
        tl_instruction_val = ""
        if pack is not None:
            tl_instruction_val = str(pack.get("tl_instruction", "")).strip()
        if not tl_instruction_val:
            tl_instruction_val = meaning_en or _default_tl_instruction(target_language)
        requests_json_val = "[]"
        if pack is not None:
            idiom_reqs = pack.get("idiom_requests")
            if isinstance(idiom_reqs, list):
                requests_json_val = json.dumps(idiom_reqs, ensure_ascii=False)
            elif idiom_reqs is not None:
                requests_json_val = json.dumps(idiom_reqs, ensure_ascii=False)
        user_content = user_content.replace("{tl_instruction}", tl_instruction_val)
        user_content = user_content.replace("{requests_json}", requests_json_val)
        user_content = user_content.replace("{line}", line_en)
        user_content = user_content.replace("{meaning_en}", meaning_en)
        user_content = user_content.replace("{prev_zh}", prev_block or "(none)")
        # Run F: all instructions in English; only the model output is in the target language (no prompt language mixing).
        if not use_phrase_suggestion and "only output" not in user_content.lower():
            user_content += f"\n\nOutput ONLY the translated subtitle in the target language (locale: {target_language}). Do not repeat the input or add any explanations. One line only."

    messages = _messages_for_chat(prompt_config, sys_content, user_content)

    prefix = f"[Stage3 {log_label}]".strip() if log_label else "[Stage3]"
    if log_lines is not None:
        log_lines.append(f"{prefix} --- INPUT ---")
        log_lines.append(f"  sub_id: {sub_id!r}")
        log_lines.append(f"  meaning_en: {meaning_en!r}")
        log_lines.append(f"  prev_zh: {repr(prev_block)}")
        log_lines.append(f"{prefix} --- (calling model) ---")

    raw = chat_dispatch(
        translate_model,
        messages,
        max_tokens=256,
        temperature=0.2,
        json_mode=use_phrase_suggestion,
        log_lines=log_lines,
        label=prefix or "Stage3",
    )
    raw = (raw or "").strip()

    sid = sub_id or ""
    if pack is not None and use_out_template and sid:
        ok, parsed, reason = validate_out_block(raw, sid, simplified=simplified_out)
        if ok and parsed.get("tgt"):
            cleaned = parsed["tgt"]
        else:
            if log_lines is not None:
                log_lines.append(f"{prefix} OUT validation failed: {reason}")
            return ""
    else:
        if use_phrase_suggestion and pack is not None and pack.get("draft_tl"):
            filled = _fill_draft_tl_from_phrase_json(raw, pack.get("draft_tl", ""))
            if filled is not None:
                cleaned = filled
            else:
                cleaned = _postprocess_stage3_output(raw, target_language)
        else:
            cleaned = _postprocess_stage3_output(raw, target_language)

    cleaned = apply_glossary_post(cleaned)

    if log_lines is not None:
        log_lines.append(f"{prefix} --- MODEL RAW OUTPUT ---")
        log_lines.append(raw if raw else "(empty)")
        log_lines.append(f"{prefix} --- POSTPROCESSED OUTPUT ---")
        log_lines.append(cleaned if cleaned else "(empty)")
    return cleaned

def _fill_draft_tl_from_phrase_json(raw: str, draft_tl: str) -> Optional[str]:
    """Parse localization JSON (e.g. {\"I1\":\"phrase\",\"I2\":\"\"}) and fill draft_tl slots. Returns filled string or None."""
    if not draft_tl or not raw:
        return None
    s = raw.strip()
    if s.startswith("```"):
        s = (s.split("\n", 1)[-1] if "\n" in s else s[3:]).strip()
        if s.lower().startswith("json"):
            s = s[4:].strip()
        if s.endswith("```"):
            s = s[:-3].strip()
    try:
        obj = json.loads(s)
        if not isinstance(obj, dict):
            return None
    except (json.JSONDecodeError, TypeError):
        return None
    out = draft_tl
    for slot, phrase in obj.items():
        if isinstance(phrase, str) and slot.upper() in ("I1", "I2"):
            tag = f"<{slot.upper()}>"
            out = out.replace(tag, phrase)
    return out if out else None


def _default_tl_instruction(target_language: str) -> str:
    """English-only instruction for Run F when tl_instruction is missing (Run A~D all-English)."""
    if not target_language:
        return "Translate to target language. One line per segment. Natural subtitle style."
    tl = target_language.strip().upper()
    if tl.startswith("ZH-TW") or tl == "ZHTW":
        return "Target: zh-TW. Style: colloquial Taiwanese. One line per segment. Natural subtitle style."
    if tl.startswith("ZH-CN") or tl == "ZHCN":
        return "Target: zh-CN. Style: colloquial Mandarin. One line per segment. Natural subtitle style."
    if tl.startswith("JA") or tl == "JA-JP":
        return "Target: ja-JP. Style: natural Japanese. One line per segment. Natural subtitle style."
    if tl.startswith("ES"):
        return "Target: es-ES. Style: natural Spanish. One line per segment. Natural subtitle style."
    return "Translate to target language. One line per segment. Natural subtitle style."


def _sanitize_tl_instruction(tl_instruction: str, target_language: str) -> str:
    """If tl_instruction contains target-language text (e.g. CJK), replace with English-only default."""
    s = (tl_instruction or "").strip()
    if not s:
        return _default_tl_instruction(target_language)
    # CJK or Japanese kana: model likely output target-language text in tl_instruction
    if re.search(r"[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u309f\u30a0-\u30ff]", s):
        return _default_tl_instruction(target_language)
    return s


def _is_likely_target_language(text: str, target_language: str) -> bool:
    """
    判斷 text 是否看起來像目標語言（非英文 brief）。
    v1 階段應輸出英文 brief；若模型誤將 draft_tl 翻成目標語，下游應改用 meaning_tl。
    """
    if not (text and text.strip()):
        return False
    s = text.strip()
    # 中文（繁/簡）：含 CJK 字元且比例高
    if target_language.upper().startswith("ZH"):
        cjk = sum(1 for c in s if "\u4e00" <= c <= "\u9fff" or "\u3400" <= c <= "\u4dbf")
        return cjk >= 2 or (cjk >= 1 and len(s) <= 15)
    # 日文：平假名/片假名/漢字
    if target_language.upper().startswith("JA"):
        ja = sum(1 for c in s if "\u3040" <= c <= "\u309f" or "\u30a0" <= c <= "\u30ff" or "\u4e00" <= c <= "\u9fff")
        return ja >= 2 or (ja >= 1 and len(s) <= 15)
    # 其他目標語可擴充
    return False


def _clean_stage2_output(text: str) -> str:
    """
    清理 Stage2 輸出，移除可能的提示標籤或說明文字。
    """
    if not text:
        return text
    
    # 移除常見的提示標籤前綴
    prefixes_to_remove = [
        "Meaning: ", "Meaning:", "meaning: ", "meaning:",
        "Explanation: ", "Explanation:", "explanation: ", "explanation:",
        "The meaning is: ", "The meaning is:",
    ]
    
    cleaned = text.strip()
    for prefix in prefixes_to_remove:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
            break
    
    return cleaned


def _extract_json_from_raw(raw: str) -> Optional[dict]:
    """
    Extract a JSON object from raw text (handles truncated JSON).
    Finds first '{' and uses JSONDecoder.raw_decode to get the longest valid object.
    Returns None if no valid dict can be extracted.
    """
    if not raw:
        return None
    s = (raw or "").strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1] if "\n" in s else s[3:]
        if s.lstrip().lower().startswith("json"):
            s = s.lstrip()[4:].lstrip()
        s = s.strip()
        if s.endswith("```"):
            s = s[:-3].strip()
    idx = s.find("{")
    if idx == -1:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(s[idx:])
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def stage3_suggest_local_phrases(
    translate_model,
    requests_json: str,
    tl_instruction: str,
    target_language: str,
    prompt_config: Any,
    log_lines: Optional[list[str]] = None,
    log_label: str = "Stage3_local",
) -> dict:
    """
    Run LOCAL model to produce slot -> suggestion JSON for idiom_requests.
    Reads role=localization template from prompt_config (CSV bound to model).
    User content: only {tl_instruction} + {requests_json}. Robust JSON parse; on failure returns {}.
    """
    if not prompt_config or not hasattr(prompt_config, "system_prompt_template") or not hasattr(prompt_config, "user_prompt_template"):
        return {}
    sys_content = prompt_config.system_prompt_template
    user_content = prompt_config.user_prompt_template
    user_content = user_content.replace("{tl_instruction}", tl_instruction or "")
    user_content = user_content.replace("{requests_json}", requests_json or "[]")
    if "{target_language}" in user_content:
        user_content = user_content.replace("{target_language}", target_language or "zh-TW")
    if "{target_language}" in sys_content:
        sys_content = sys_content.replace("{target_language}", target_language or "zh-TW")
    messages = _messages_for_chat(prompt_config, sys_content, user_content)
    raw = chat_dispatch(
        translate_model,
        messages,
        max_tokens=256,
        temperature=0.2,
        json_mode=True,
        log_lines=log_lines,
        label=log_label or "Stage3_local",
    )
    s = (raw or "").strip()
    if s.startswith("```"):
        s = (s.split("\n", 1)[-1] if "\n" in s else s[3:]).strip()
        if s.lower().startswith("json"):
            s = s[4:].strip()
        if s.endswith("```"):
            s = s[:-3].strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
        return {}
    except (json.JSONDecodeError, TypeError):
        pass
    # Strict fallback: extract first complete JSON object (handles nested braces, weak-model preamble)
    idx = s.find("{")
    if idx >= 0:
        try:
            obj, _ = json.JSONDecoder().raw_decode(s[idx:])
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return {}


def _fill_draft_with_suggestions(draft_tl: str, suggestions: dict) -> str:
    """Replace <I1>, <I2>, ... in draft_tl with suggestions; return one-line string."""
    if not draft_tl:
        return ""
    out = draft_tl
    sug = suggestions or {}
    for slot, phrase in sug.items():
        if not isinstance(phrase, str):
            continue
        slot_key = str(slot).strip().upper()
        if not slot_key:
            continue
        tag = f"<{slot_key}>"
        out = out.replace(tag, phrase)
    return out.strip()


def stage4_assemble_by_main(
    reason_model,
    english_line: str,
    ctx_brief: str,
    draft_tl: str,
    suggestions: dict,
    target_language: str,
    prompt_config: Any = None,
    role: str = "main_assemble",
) -> str:
    """
    Assemble draft_tl + suggestions into one final subtitle line.
    Replaces <I1>/<I2> slots in draft_tl with suggestions, then asks MAIN to polish to one line.
    Uses prompt_config (role=main_assemble) if provided; otherwise fallback to hardcoded prompt.
    """
    filled = _fill_draft_with_suggestions(draft_tl or "", suggestions or {})
    if not filled:
        filled = draft_tl or ""
    if reason_model is None:
        return filled
    suggestions_json = json.dumps(suggestions or {}, ensure_ascii=False)
    if prompt_config and (getattr(prompt_config, "user_prompt_template", None) or "").strip():
        sys_content = (getattr(prompt_config, "system_prompt_template", None) or "").strip()
        user_content = (prompt_config.user_prompt_template or "").strip()
        replacements = {
            "target_language": target_language,
            "line_en": english_line or "",
            "ctx_brief": ctx_brief or "(none)",
            "draft_prefilled": filled or "",
            "suggestions_json": suggestions_json,
        }
        for k, v in replacements.items():
            user_content = user_content.replace("{" + k + "}", str(v))
        if not sys_content:
            sys_content = "Output ONE LINE only: the final subtitle in the target language. No labels, no JSON, no explanation."
    else:
        sys_content = (
            "You are the main reasoning model. Output a single line only: the final subtitle in the target language. "
            "No explanation, no JSON, no markdown. One line, no newlines."
        )
        user_content = (
            f"Target language: {target_language}\n"
            f"Draft line (idiom slots already filled): {filled}\n"
            f"Context: {ctx_brief or '(none)'}\n"
            f"Original English: {english_line}\n\n"
            f"Output exactly one line in {target_language}."
        )
    messages = _messages_for_chat(prompt_config, sys_content, user_content)
    raw = chat_dispatch(
        reason_model,
        messages,
        max_tokens=256,
        temperature=0.1,
        json_mode=False,
        log_lines=None,
        label="Stage4_assemble",
    )
    line = (raw or "").strip()
    if "\n" in line:
        line = line.split("\n")[0].strip()
    return line if line else filled


def parse_pack_from_reasons(reasons: str) -> Optional[dict]:
    """Extract PACK JSON from BriefMeta.reasons (any position). Returns None if not present or invalid."""
    if not reasons:
        return None
    s = reasons.strip()
    idx = s.find("PACK:")
    if idx == -1:
        idx = s.find("PACK\n")
        if idx == -1:
            return None
        idx = s.find("{", idx)
    else:
        idx = s.find("{", idx)
    if idx == -1:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(s[idx:])
        return obj
    except Exception:
        return None


def stage_main_group_translate(
    reason_model,
    target_language: str,
    segments_json: str,
    tl_instruction: str,
    prompt_config: Any | None,
    *,
    expanded_context: Optional[str] = None,
    log_lines: Optional[list[str]] = None,
    log_label: str = "main_group_translate",
    model_path: Optional[str] = None,
    load_params: Optional[dict[str, Any]] = None,
    cfg: Any = None,
    group_segments_count: Optional[int] = None,
) -> Optional[list[dict]]:
    """
    MAIN model: translate a group of segments; output segment_texts (array of {id, text}).
    expanded_context: optional extra prev/next lines when need_more_context (Run F 方案 C).
    Returns segment_texts list on success; None on parse failure or length mismatch.

    如果沒有在 CSV 中定義 main_group_translate prompt（prompt_config 為 None 或沒有 user_prompt_template），
    則退回到一個內建的預設 prompt，以確保仍然會嘗試「英 → 目標語」翻譯，
    而不是直接回傳 None 讓後續只使用英文 PACK / 原文。
    """
    if not reason_model or not segments_json:
        return None

    # 有自訂 prompt_config 時，沿用既有邏輯
    sys_content: str
    user_content: str
    messages: list[dict]
    if prompt_config and (getattr(prompt_config, "user_prompt_template", None) or "").strip():
        sys_content = (getattr(prompt_config, "system_prompt_template", None) or "").strip()
        user_content = (prompt_config.user_prompt_template or "").strip()
        user_content = user_content.replace("{target_language}", target_language or "")
        user_content = user_content.replace("{tl_instruction}", tl_instruction or "")
        user_content = user_content.replace("{segments_json}", segments_json)
        user_content = user_content.replace("{expanded_context}", (expanded_context or "(none)").strip())
        if "<PN__" in segments_json:
            user_content = (
                "All strings of the form <PN__...> are immutable proper-noun anchors: you may read the Key for meaning but must keep them verbatim in output; do not translate, rephrase, split, or remove.\n\n"
                + user_content
            )
        messages = _messages_for_chat(prompt_config, sys_content, user_content)
    else:
        # 預設 fallback：要求模型將 segments_json 內的英文字幕翻成 target_language，
        # 並以 JSON 形式回傳 segment_texts 陣列。
        safe_target = (target_language or "").strip() or "the target language"
        sys_content = (
            "You are a subtitle translation model. "
            "Your task is to translate English subtitle segments into the target language and return ONLY valid JSON. "
            "Do not explain or add comments."
        )
        user_content = (
            "Target language: {target_language}\n"
            "Translation instruction (may be empty or English-only hints from earlier stages):\n"
            "{tl_instruction}\n\n"
            "You are given a JSON array of segments under the key segments_json. "
            "Each element has the shape {{\"id\": string, \"en\": string, \"ms\": number}} where \"en\" is the English subtitle text.\n\n"
            "Translate EACH segment's \"en\" into the target language as natural, fluent subtitles. "
            "Return ONLY JSON in this shape (no extra keys, no text outside JSON):\n"
            "{{\"segment_texts\": [{{\"id\": \"...\", \"text\": \"...\"}}, ...]}}\n\n"
            "segments_json:\n{segments_json}"
        ).format(
            target_language=safe_target,
            tl_instruction=(tl_instruction or "").strip(),
            segments_json=segments_json,
        )
        if "<PN__" in segments_json:
            user_content = (
                "All strings of the form <PN__...> are immutable proper-noun anchors: you may read the Key for meaning but must keep them verbatim in output; do not translate, rephrase, split, or remove.\n\n"
                + user_content
            )
        messages = [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": user_content},
        ]
    raw = chat_dispatch(
        reason_model,
        messages,
        max_tokens=2048,
        temperature=0.1,
        json_mode=True,
        log_lines=log_lines,
        label=log_label,
        model_path=model_path,
        load_params=load_params,
        cfg=cfg,
        group_segments_count=group_segments_count,
    )
    raw = (raw or "").strip()
    s = raw
    if s.startswith("```"):
        s = (s.split("\n", 1)[-1] if "\n" in s else s[3:]).strip()
        if s.lower().startswith("json"):
            s = s[4:].strip()
        if s.endswith("```"):
            s = s[:-3].strip()
    # Drop leading lines until we hit JSON (e.g. "Here is the result:\n{...}")
    for i, line in enumerate(s.split("\n")):
        t = line.strip()
        if t.startswith("{") or t.startswith("["):
            s = "\n".join(s.split("\n")[i:])
            break

    def _fix_trailing_commas(s: str) -> str:
        """Remove trailing commas before ] or } so strict JSON parser can accept."""
        s = re.sub(r",\s*\]", "]", s)
        s = re.sub(r",\s*}", "}", s)
        return s

    def _normalize_segments(segs: list) -> Optional[list[dict]]:
        """Accept list of {id, text} or {id, draft_tl/tl}; return list of {id, text} or None if empty."""
        if not isinstance(segs, list) or not segs:
            return None
        out: list[dict] = []
        for x in segs:
            if not isinstance(x, dict):
                continue
            sid = x.get("id")
            if sid is None:
                continue
            text = (
                (x.get("text") if isinstance(x.get("text"), str) else None)
                or (x.get("draft_tl") if isinstance(x.get("draft_tl"), str) else None)
                or (x.get("tl") if isinstance(x.get("tl"), str) else None)
                or (x.get("content") if isinstance(x.get("content"), str) else None)
                or (x.get("value") if isinstance(x.get("value"), str) else None)
            )
            if text is not None and str(text).strip():
                out.append({"id": str(sid).strip(), "text": str(text).strip()})
        return out if out else None

    def _get_segs_from_obj(obj: Any) -> Optional[list[dict]]:
        if obj is None:
            return None
        if isinstance(obj, list):
            return _normalize_segments(obj)
        if isinstance(obj, dict):
            for key in ("segment_texts", "segments", "translations", "items", "results", "output", "data"):
                segs = obj.get(key)
                if isinstance(segs, list):
                    normalized = _normalize_segments(segs)
                    if normalized:
                        return normalized
        return None

    try:
        obj = json.loads(s)
        segs = _get_segs_from_obj(obj)
        if segs:
            return segs
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        obj = json.loads(_fix_trailing_commas(s))
        segs = _get_segs_from_obj(obj)
        if segs:
            return segs
    except (json.JSONDecodeError, TypeError):
        pass
    # Extract first complete JSON (object or array) via raw_decode to handle trailing text
    for start_char in ("{", "["):
        idx = s.find(start_char)
        if idx >= 0:
            try:
                obj, _ = json.JSONDecoder().raw_decode(s[idx:])
                segs = _get_segs_from_obj(obj)
                if segs:
                    return segs
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
            try:
                obj, _ = json.JSONDecoder().raw_decode(_fix_trailing_commas(s[idx:]))
                segs = _get_segs_from_obj(obj)
                if segs:
                    return segs
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
    # Regex fallback for nested "segment_texts": [...]
    match = re.search(r"\{[\s\S]*\"segment_texts\"\s*:\s*\[[\s\S]*\]\s*[\s\S]*\}", s)
    if match:
        try:
            obj = json.loads(_fix_trailing_commas(match.group()))
            segs = _get_segs_from_obj(obj)
            if segs:
                return segs
        except (json.JSONDecodeError, TypeError):
            pass
    return None


def _polish_line_acceptable(polished: str, original_len: int) -> bool:
    """語言無關保護：長度 >3x 或 <0.3x 原稿則不採用；含 prompt 汙染則不採用。"""
    if not polished or not polished.strip():
        return False
    # 單行、trim
    p = polished.replace("\n", " ").strip()
    if not p:
        return False
    if original_len > 0:
        if len(p) > 3 * original_len or len(p) < 0.3 * original_len:
            return False
    # 明顯 prompt 汙染
    if "lines=" in p or "{tl_instruction}" in p or "JSON" in p:
        return False
    return True


def stage3_polish_local_lines(
    translate_model,
    target_language: str,
    tl_instruction: str,
    lines: list[dict],
    prompt_config: Any | None,
    log_lines: Optional[list[str]] = None,
    log_label: str = "",
    *,
    model_path: Optional[str] = None,
    load_params: Optional[dict[str, Any]] = None,
    cfg: Any = None,
    lines_count: Optional[int] = None,
    transliteration_terms: Optional[list[str]] = None,
    local_polish_mode: Optional[str] = None,
) -> tuple[dict[str, str], list[dict]]:
    """
    LOCAL model: 批次順口化整集字幕（role=local_polish）。
    輸入 lines = [{"id":"...","text":"..."}]，僅含目標語 draft，不得含英文原文。
    transliteration_terms: 主模型標記需音譯的人名/專有名詞，由在地化模型在輸出中音譯。
    local_polish_mode: "weak" = 輸入已去標點，改通順+音譯人名；"strong" = 輸入含標點，改通順/更在地+音譯人名，且每句第一個字前不得有標點。
    回傳 (out, name_translations)：out = {sub_id: polished_one_line}；name_translations = [{"original":"...","translated":"..."}, ...] 供 CSV 輸出。
    """
    out: dict[str, str] = {}
    name_translations: list[dict] = []
    if not translate_model or not lines:
        return (out, name_translations)
    if not prompt_config or not (getattr(prompt_config, "user_prompt_template", None) or "").strip():
        return (out, name_translations)
    lines_json = json.dumps(lines, ensure_ascii=False)
    sys_content = (getattr(prompt_config, "system_prompt_template", None) or "").strip()
    mode = (local_polish_mode or "strong").strip().lower()
    if mode == "weak":
        sys_content = (
            "Input text has punctuation removed. Your task: make each sentence more fluent and transliterate person names where listed. "
            + sys_content
        )
    elif mode == "strong":
        sys_content = (
            "Input text includes punctuation. Your task: make each sentence more fluent and more localized; transliterate person names where listed. "
            "Output rule: the first character of each line must NOT be preceded by any punctuation. "
            + sys_content
        )
    user_content = (prompt_config.user_prompt_template or "").strip()
    user_content = user_content.replace("{tl_instruction}", tl_instruction or "")
    user_content = user_content.replace("{lines_json}", lines_json)
    if transliteration_terms:
        terms_str = ", ".join(str(t).strip() for t in transliteration_terms if str(t).strip())
        if terms_str:
            user_content += "\n\nTransliterate in target language for these terms: " + terms_str + "."
    if "{target_language}" in user_content or "{target_language}" in sys_content:
        user_content = user_content.replace("{target_language}", target_language or "")
        sys_content = sys_content.replace("{target_language}", target_language or "")
    messages = _messages_for_chat(prompt_config, sys_content, user_content)
    n_lines = len(lines)
    max_tokens = min(2048, max(1024, n_lines * 80))
    raw = chat_dispatch(
        translate_model,
        messages,
        max_tokens=max_tokens,
        temperature=0.2,
        json_mode=True,
        log_lines=log_lines,
        label=log_label or "Stage3_polish",
        model_path=model_path,
        load_params=load_params,
        cfg=cfg,
        lines_count=lines_count if lines_count is not None else n_lines,
    )
    raw = (raw or "").strip()
    prefix = f"[Stage3 polish {log_label}]".strip() if log_label else "[Stage3 polish]"
    if log_lines is not None:
        log_lines.append(f"{prefix} raw length={len(raw)}")
    s = raw
    if s.startswith("```"):
        s = (s.split("\n", 1)[-1] if "\n" in s else s[3:]).strip()
        if s.lower().startswith("json"):
            s = s[4:].strip()
        if s.endswith("```"):
            s = s[:-3].strip()
    def _extract_name_translations(obj: Any) -> list[dict]:
        nt = obj.get("name_translations") if isinstance(obj, dict) else None
        if not isinstance(nt, list) or not nt:
            return []
        result: list[dict] = []
        for x in nt:
            if not isinstance(x, dict):
                continue
            orig = x.get("original") if isinstance(x.get("original"), str) else None
            trans = x.get("translated") if isinstance(x.get("translated"), str) else None
            if orig and trans and (orig.strip() and trans.strip()):
                result.append({"original": orig.strip(), "translated": trans.strip()})
        return result

    id_to_original_len = {str(line.get("id", "")): len((line.get("text") or "")) for line in lines}
    try:
        obj = json.loads(s)
        if not isinstance(obj, dict):
            return (out, name_translations)
        name_translations = _extract_name_translations(obj)
        for k, v in obj.items():
            if k == "name_translations" or not isinstance(v, str):
                continue
            v = v.replace("\n", " ").strip()
            if not v:
                continue
            kid = str(k).strip()
            if kid not in id_to_original_len:
                continue  # weak model may return "0","1" etc.; only accept keys from request
            orig_len = id_to_original_len.get(kid, 0)
            if _polish_line_acceptable(v, orig_len):
                out[kid] = v
        return (out, name_translations)
    except (json.JSONDecodeError, TypeError):
        if log_lines is not None:
            log_lines.append(f"{prefix} JSON parse failed")
        pass
    # Strict fallback: extract first complete JSON object (handles nested braces, weak-model preamble)
    idx = s.find("{")
    if idx >= 0:
        try:
            obj, _ = json.JSONDecoder().raw_decode(s[idx:])
            if isinstance(obj, dict):
                name_translations = _extract_name_translations(obj)
                for k, v in obj.items():
                    if k == "name_translations" or not isinstance(v, str):
                        continue
                    v = v.replace("\n", " ").strip()
                    if not v:
                        continue
                    kid = str(k).strip()
                    if kid not in id_to_original_len:
                        continue
                    orig_len = id_to_original_len.get(kid, 0)
                    if _polish_line_acceptable(v, orig_len):
                        out[kid] = v
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return (out, name_translations)


def validate_out_block(text: str, sub_id: str, simplified: bool = False) -> tuple[bool, dict, str]:
    """
    Validate Stage3 <OUT>...</OUT> block.
    simplified: if True, only require id/tgt/conf (no tgt_alt).
    Returns (ok, parsed, reason).
    parsed: {"id": str, "tgt": str, "tgt_alt": str or "", "conf": float}
    """
    if not text or not text.strip():
        return False, {}, "empty"
    raw = text.strip()
    out_start = raw.find("<OUT>")
    out_end = raw.find("</OUT>")
    if out_start < 0 or out_end < 0 or out_end <= out_start:
        return False, {}, "missing <OUT> or </OUT>"
    block = raw[out_start + 5 : out_end].strip()
    parsed = {}
    for line in block.splitlines():
        line = line.strip()
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip().lower(), v.strip()
        if k == "id":
            parsed["id"] = v
        elif k == "tgt":
            parsed["tgt"] = v
        elif k == "tgt_alt":
            parsed["tgt_alt"] = v
        elif k == "conf":
            parsed["conf"] = v
    required = ["id", "tgt", "conf"] if simplified else ["id", "tgt", "tgt_alt", "conf"]
    for r in required:
        if r not in parsed:
            return False, {}, f"missing key {r}"
    if not simplified and "tgt_alt" not in parsed:
        parsed["tgt_alt"] = ""
    if parsed.get("id", "") != sub_id:
        return False, parsed, "id mismatch"
    try:
        c = float(parsed["conf"])
        if not (0 <= c <= 1):
            return False, parsed, "conf out of range"
    except (TypeError, ValueError):
        return False, parsed, "conf not float"
    if len(parsed.get("tgt", "")) > 500:
        return False, parsed, "tgt too long"
    return True, parsed, "ok"


def _postprocess_stage3_output(raw: str, target_language: str = "zh-TW", sub_id: str = "") -> str:
    """
    後處理 Stage3 輸出。若含 <OUT>...</OUT> 則優先提取 tgt；否則沿用原有啟發式。
    """
    if raw and "<OUT>" in raw and "</OUT>" in raw:
        out_start = raw.find("<OUT>")
        out_end = raw.find("</OUT>")
        block = raw[out_start + 5 : out_end].strip()
        for line in block.splitlines():
            line = line.strip()
            if line.lower().startswith("tgt="):
                tgt = line.split("=", 1)[1].strip()
                if tgt:
                    return tgt
    # Legacy heuristic:
    # 1. 若只有一行非空文字，直接回傳。
    # 2. 優先尋找「全形引號」包起來的一行（例如：「這是一句字幕」），取第一個。
    # 3. 移除所有包含提示標籤的行（「英文字幕：」「英文語意說明：」等）
    # 4. 從下往上尋找「不像提示文字」的那一行：
    #    - 不包含說明性關鍵字
    #    - 不是純英文（如果目標語言是中文）
    #    - 長度合理（不是太短或太長）
    # 5. 如果都找不到，嘗試提取第一行看起來像翻譯結果的行
    if not raw:
        return raw

    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        return raw
    if len(lines) == 1:
        # 單行輸出：檢查是否為「已完成宣告」等廢話，若是則視為無效
        single_line = lines[0]
        if ("字幕翻譯流程" in single_line and "已完成" in single_line) or (
            "台灣繁體中文字幕" in single_line and "已完成" in single_line
        ):
            return ""
        # 移除常見的標籤前綴
        for prefix in [
            "英文字幕：", "英文字幕:", "翻譯結果：", "翻譯結果:", "字幕：", "字幕:",
            "Translation:", "Translation：", "Output:", "Output：",
            "台灣繁體中文字幕：", "台灣繁體中文字幕:",
            "繁體中文字幕：", "繁體中文字幕:",
        ]:
            if single_line.startswith(prefix):
                single_line = single_line[len(prefix):].strip()
                break
        return single_line

    # 定義提示標籤關鍵字（更完整）
    bad_keywords = [
        # 中文標籤
        "英文字幕：", "英文字幕:", "English subtitle",
        "英文語意說明：", "英文語意說明:", "English meaning",
        "前文中文字幕：", "前文中文字幕:", "Previous Chinese",
        "術語表：", "術語表:", "Glossary",
        "使用者希望將英文字幕", "User wants to translate",
        "輸入：", "輸入:", "Input:",
        "輸出：", "輸出:", "Output:",
        "翻譯結果：", "翻譯結果:", "Translation result",
        "自然的台灣繁體中文字幕：", "Natural Traditional Chinese",
        "經過多階段字幕翻譯流程後", "After multi-stage subtitle translation",
        "台灣繁體中文字幕：", "台灣繁體中文字幕:",
        "繁體中文字幕：", "繁體中文字幕:",
        # 英文標籤
        "English subtitle:", "Meaning explanation:",
        "Previous translated lines:", "Glossary terms:",
        "You are stage 3", "You are stage3",
        "多階段字幕翻譯流程", "multi-stage subtitle translation",
    ]
    
    # 過濾掉明顯是提示標籤的行
    filtered_lines = []
    for ln in lines:
        # 跳過包含提示標籤的行
        if any(k in ln for k in bad_keywords):
            continue
        # 跳過太短的行（可能是標籤殘留）
        if len(ln) < 3:
            continue
        # 跳過純英文行（如果目標語言是中文）
        if target_language.startswith("zh"):
            # 簡單檢查：如果行中沒有中文字符，可能是提示殘留
            has_chinese = any('\u4e00' <= char <= '\u9fff' for char in ln)
            if not has_chinese and len(ln) > 20:
                # 可能是英文提示，跳過
                continue
        filtered_lines.append(ln)
    
    # 如果過濾後還有行，使用過濾後的行
    if filtered_lines:
        lines = filtered_lines
    
    # 2. 優先尋找「全形引號」包起來的一行
    for ln in lines:
        if ln.startswith("「") and ln.endswith("」") and len(ln) > 2:
            return ln.strip("「」").strip()
    
    # 3. 從結尾往前找「不像提示」的行（更嚴格的檢查）
    for ln in reversed(lines):
        # 檢查是否包含提示關鍵字
        if any(k in ln for k in bad_keywords):
            continue
        # 檢查長度（太長可能是包含了多個部分）
        if len(ln) > 200:
            continue
        # 如果目標語言是中文，確保有中文字符
        if target_language.startswith("zh"):
            has_chinese = any('\u4e00' <= char <= '\u9fff' for char in ln)
            if has_chinese:
                return ln
        else:
            # 非中文目標語言，直接返回
            return ln
    
    # 4. 如果還是找不到，嘗試返回第一行看起來合理的行
    for ln in lines:
        if len(ln) >= 5 and len(ln) <= 200:
            # 移除可能的標籤前綴
            for prefix in ["翻譯結果：", "翻譯結果:", "Translation:", "Output:", "字幕：", "字幕:"]:
                if ln.startswith(prefix):
                    ln = ln[len(prefix):].strip()
                    break
            return ln
    
    # 5. 最後的 fallback：返回原始輸出的第一行（移除標籤）
    first_line = raw.splitlines()[0].strip()
    for prefix in ["翻譯結果：", "翻譯結果:", "Translation:", "Output:", "字幕：", "字幕:",
                   "英文字幕：", "英文字幕:", "English subtitle:"]:
        if first_line.startswith(prefix):
            first_line = first_line[len(prefix):].strip()
            break
    return first_line if first_line else raw.strip()
