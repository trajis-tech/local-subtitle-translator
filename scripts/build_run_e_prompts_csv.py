"""Build model_prompts_run_f.csv from model_prompts.csv (Run F translation roles only).

Run after editing model_prompts.csv if you add or change rows for:
  main_group_translate, main_assemble, local_polish, localization
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN_E_ROLES = {"main_group_translate", "main_assemble", "local_polish", "localization"}

def main():
    src = ROOT / "model_prompts.csv"
    dst = ROOT / "model_prompts_run_f.csv"
    with src.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = [x for x in (reader.fieldnames or []) if x]
        rows = []
        for r in reader:
            role = (r.get("role") or "").strip()
            if role in RUN_E_ROLES:
                rows.append({k: v for k, v in r.items() if k in fieldnames})
    with dst.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {dst.name}")

if __name__ == "__main__":
    main()
