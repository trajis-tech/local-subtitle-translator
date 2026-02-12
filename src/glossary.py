from __future__ import annotations
from dataclasses import dataclass, asdict, field
from pathlib import Path
import json
import csv
import io
import re
import xml.etree.ElementTree as ET


@dataclass
class GlossaryEntry:
    en: str
    zh: str = ""  # legacy: zh-TW
    to: str = ""  # optional: current target language output term
    targets: dict = field(default_factory=dict)  # optional: {"ja-JP":"..","es":".."}
    note: str = ""
    enabled: bool = True


def get_glossary_target_term(entry: GlossaryEntry | dict, target_language: str) -> str:
    """依 target_language 選出替換詞：targets[target_language] > to > zh（向後相容）。"""
    if isinstance(entry, dict):
        t = (entry.get("targets") or {}).get(target_language) or entry.get("to") or entry.get("zh") or ""
    else:
        t = (
            (getattr(entry, "targets", None) or {}).get(target_language)
            or (getattr(entry, "to", None) or "")
            or (getattr(entry, "zh", None) or "")
        )
    return (t or "").strip()


def load_glossary(path: str | Path) -> list[GlossaryEntry]:
    p = Path(path)
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    out: list[GlossaryEntry] = []
    for row in data:
        # allow extra keys in row without crashing
        out.append(
            GlossaryEntry(
                en=row.get("en", ""),
                zh=row.get("zh", ""),
                to=row.get("to", ""),
                targets=row.get("targets", {}) or {},
                note=row.get("note", ""),
                enabled=row.get("enabled", True),
            )
        )
    return out

def save_glossary(path: str | Path, entries: list[GlossaryEntry]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(e) for e in entries]
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def import_csv_two_cols(csv_bytes: bytes, encoding="utf-8-sig") -> list[GlossaryEntry]:
    # Expect: col1 English, col2 Traditional Chinese (header optional)
    text = csv_bytes.decode(encoding, errors="replace")
    reader = csv.reader(io.StringIO(text))
    rows = list(reader)
    # If header detected, skip
    if rows and len(rows[0]) >= 2 and ("english" in rows[0][0].lower() or "en" == rows[0][0].lower()):
        rows = rows[1:]
    out: list[GlossaryEntry] = []
    for r in rows:
        if len(r) < 2:
            continue
        en = r[0].strip()
        zh = r[1].strip()
        if not en or not zh:
            continue
        out.append(GlossaryEntry(en=en, zh=zh))
    return out

def import_subtitleedit_multiple_replace_template(template_bytes: bytes) -> list[GlossaryEntry]:
    """Import Subtitle Edit `multiple_replace_groups.template` files.

    Subtitle Edit exports multiple replace groups as XML `.template` files.
    Format: <MultipleSearchAndReplaceItem><FindWhat>...</FindWhat><ReplaceWith>...</ReplaceWith></MultipleSearchAndReplaceItem>
    """
    text = template_bytes.decode("utf-8", errors="replace")
    # Some files may start with BOM or leading whitespace.
    text = text.lstrip("\ufeff\ufeff\ufeff")
    
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        # fallback: try to recover by stripping non-xml prefix/suffix
        m = re.search(r"(<\?xml[\s\S]+)$", text)
        if m:
            try:
                root = ET.fromstring(m.group(1))
            except ET.ParseError:
                return []
        else:
            return []
    except Exception:
        return []

    pairs: list[GlossaryEntry] = []

    # Find all MultipleSearchAndReplaceItem elements
    for item in root.iter():
        if item.tag.endswith("MultipleSearchAndReplaceItem") or "MultipleSearchAndReplaceItem" in item.tag:
            find_what = None
            replace_with = None
            enabled = True
            
            # Look for FindWhat and ReplaceWith child elements
            for child in item:
                tag_lower = child.tag.lower()
                text_content = (child.text or "").strip() if child.text else ""
                
                if "findwhat" in tag_lower or tag_lower.endswith("findwhat"):
                    find_what = text_content
                elif "replacewith" in tag_lower or tag_lower.endswith("replacewith"):
                    replace_with = text_content
                elif "enabled" in tag_lower or tag_lower.endswith("enabled"):
                    # Check if this item is enabled
                    enabled_text = text_content.lower()
                    if enabled_text in ("false", "0", "no", "off"):
                        enabled = False
            
            # Only add if both find and replace are found, and item is enabled
            if find_what and replace_with and enabled:
                pairs.append(GlossaryEntry(
                    en=find_what,
                    zh=replace_with,
                    note="imported from Subtitle Edit template"
                ))

    # Also check for attribute-style pairs (fallback)
    for el in root.iter():
        attrs = {k.lower(): v for k, v in el.attrib.items()}
        find = attrs.get("findwhat") or attrs.get("find") or attrs.get("search") or attrs.get("from")
        repl = attrs.get("replacewith") or attrs.get("replace") or attrs.get("replacement") or attrs.get("to")
        if find and repl:
            pairs.append(GlossaryEntry(en=str(find).strip(), zh=str(repl).strip(), note="imported from Subtitle Edit template"))

    # Dedup by (en, zh) pair
    uniq = {}
    for e in pairs:
        key = (e.en.lower(), e.zh)
        if key not in uniq:
            uniq[key] = e
    return list(uniq.values())

def apply_glossary_pre(text: str, entries: list[GlossaryEntry]) -> str:
    # Pre-replacement: to stabilize proper nouns and technical terms before model sees it.
    # We do conservative, case-insensitive whole-word for ASCII-ish tokens; exact for others.
    out = text
    for e in entries:
        if not e.enabled:
            continue
        en = e.en.strip()
        zh = e.zh.strip()
        if not en or not zh:
            continue
        if re.fullmatch(r"[A-Za-z0-9 _\-\.]+", en):
            pattern = r"\b" + re.escape(en) + r"\b"
            out = re.sub(pattern, f"{{{{GLOSS:{zh}}}}}", out, flags=re.IGNORECASE)
        else:
            out = out.replace(en, f"{{{{GLOSS:{zh}}}}}")
    return out

def apply_glossary_post(text: str) -> str:
    # Convert placeholders back to zh terms.
    return re.sub(r"\{\{GLOSS:([^}]+)\}\}", r"\1", text)


def _to_anchor_key(en: str) -> str:
    """將術語原文壓縮為 CamelCase Key：去空格、少標點、無空格。例：'The Burn' -> 'TheBurn', 'Spore Drive' -> 'SporeDrive'."""
    s = (en or "").strip()
    if not s:
        return "PN"
    # 以非字母數字為分隔，切出單詞，再 CamelCase 相接
    parts = re.sub(r"[^A-Za-z0-9]+", " ", s).strip().split()
    if not parts:
        return "PN"
    return "".join(p[0].upper() + (p[1:].lower() if len(p) > 1 else "") for p in parts)


def _anchor_tag(key: str) -> str:
    return f"<PN__{key}>"


def apply_glossary_pre_anchor(
    text: str,
    entries: list[GlossaryEntry] | list[dict],
    target_language: str,
) -> tuple[str, dict[str, str]]:
    """
    翻譯前：將 Glossary 英文原文替換為不可變錨點 <PN__Key>，並建立 anchor -> 目標語翻譯 的 mapping。
    回傳 (替換後的文字, {anchor 字串: 目標語譯詞})。
    """
    if not text:
        return "", {}
    entries = entries or []
    # 長詞先替換，避免「The」吃掉「The Burn」
    sorted_entries: list[tuple[str, str, str]] = []
    for e in entries:
        if isinstance(e, dict):
            if not e.get("enabled", True):
                continue
            en = (e.get("en") or "").strip()
            to = get_glossary_target_term(e, target_language)
        else:
            if not getattr(e, "enabled", True):
                continue
            en = (getattr(e, "en", None) or "").strip()
            to = get_glossary_target_term(e, target_language)
        if not en:
            continue
        key = _to_anchor_key(en)
        tag = _anchor_tag(key)
        sorted_entries.append((en, tag, to))
    sorted_entries.sort(key=lambda x: -len(x[0]))
    out = text
    anchor_to_target: dict[str, str] = {}
    for en, tag, to in sorted_entries:
        if re.fullmatch(r"[A-Za-z0-9 _\-\.]+", en):
            pattern = r"\b" + re.escape(en) + r"\b"
            if re.search(pattern, out, flags=re.IGNORECASE):
                out = re.sub(pattern, tag, out, flags=re.IGNORECASE)
                anchor_to_target[tag] = to
        else:
            if en in out:
                out = out.replace(en, tag)
                anchor_to_target[tag] = to
    return out, anchor_to_target


def apply_glossary_post_anchor(text: str, anchor_to_target: dict[str, str]) -> str:
    """
    翻譯後：將輸出中的 <PN__Key> 錨點替換為 Glossary 的目標語翻譯。
    """
    if not text or not anchor_to_target:
        return text
    out = text
    for anchor, target in anchor_to_target.items():
        out = out.replace(anchor, target)
    return out


def force_glossary_replace_output(
    text: str,
    entries: list[GlossaryEntry] | list[dict] | None,
    target_language: str,
) -> str:
    """
    Output-side force replace: for enabled entries, replace source term (en) with target (zh) in text.
    ASCII token: whole-word, case-insensitive. Non-ASCII: exact replace.
    No arrow/list formatting, no model. target_language is for future locale use.
    """
    if not text:
        return text
    entries = entries or []
    out = text
    for e in entries:
        if isinstance(e, dict):
            if not e.get("enabled", True):
                continue
            en = (e.get("en") or "").strip()
            zh = get_glossary_target_term(e, target_language)
        else:
            if not getattr(e, "enabled", True):
                continue
            en = (getattr(e, "en", None) or "").strip()
            zh = get_glossary_target_term(e, target_language)
        if not en or not zh:
            continue
        if re.fullmatch(r"[A-Za-z0-9 _\-\.]+", en):
            pattern = r"\b" + re.escape(en) + r"\b"
            out = re.sub(pattern, lambda _: zh, out, flags=re.IGNORECASE)
        else:
            out = out.replace(en, zh)
    return out
