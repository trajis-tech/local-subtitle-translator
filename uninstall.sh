#!/usr/bin/env bash
# =========================================================
#  Local Subtitle Translator - Uninstall (Linux / macOS)
#  Counterpart of uninstall.bat (Windows).
#  Removes only project-local files inside this folder.
# =========================================================
set -euo pipefail
cd "$(dirname "$0")"

echo "=========================================="
echo " Local Subtitle Translator - Uninstall"
echo "=========================================="
echo "This will ONLY remove files inside this folder:"
echo "  - runtime/       (caches, temp)"
echo "  - .venv/         (local Python environment)"
echo "  - models/        (GGUF + Run A audio model)"
echo "  - work/          (intermediate translation results, JSONL)"
echo "  - data/          (glossary)"
echo "  - gradio_cached_examples"
echo "  - config.json, *.log"
echo
read -r -p "Continue? [y/N] " reply
case "${reply}" in
  [yY][eE][sS]|[yY]) ;;
  *) echo "Aborted."; exit 0 ;;
esac

echo
echo "[1/1] Removing project-local files..."

rm -rf runtime
rm -rf .venv
rm -rf models
rm -rf work
rm -rf data
rm -rf gradio_cached_examples

rm -f config.json
rm -f ./*.log 2>/dev/null || true

echo
echo "Uninstall completed."
echo "You can now delete this folder manually if you want."
