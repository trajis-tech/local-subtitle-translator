"""Ensure model_prompts.csv has UTF-8 BOM encoding."""
from pathlib import Path

CSV_PATH = Path(__file__).parent.parent / "model_prompts.csv"

if CSV_PATH.exists():
    # Read current content (handle legacy encodings)
    raw = CSV_PATH.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        print(f"[OK] UTF-8 BOM already present for {CSV_PATH}")
    else:
        try:
            content = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            # Fallback: common Windows legacy encoding
            content = raw.decode("cp1252")
        # Write with BOM
        CSV_PATH.write_text(content, encoding="utf-8-sig")
        print(f"[OK] Added UTF-8 BOM for {CSV_PATH}")
else:
    print(f"[ERROR] {CSV_PATH} not found")
