"""
子行程 worker：單次載入模型、讀取 stdin 一筆 JSON 請求、執行 model.chat、stdout 回傳結果後 exit。
僅在 heavy request 隔離時由主程式透過 isolated_executor 呼叫，確保 OOM/driver error 不拖死主程式。
"""

from __future__ import annotations
import json
import sys


def main() -> None:
    if len(sys.argv) < 4:
        # argv: model_path, model_type, load_params_path
        out = {"ok": False, "error": "Usage: python -m src.worker_infer <model_path> <model_type> <load_params_path>"}
        sys.stdout.write(json.dumps(out, ensure_ascii=False) + "\n")
        sys.exit(1)
    model_path = sys.argv[1]
    model_type = (sys.argv[2] or "text").strip().lower()
    load_params_path = sys.argv[3]

    try:
        with open(load_params_path, "r", encoding="utf-8") as f:
            load_params = json.load(f)
    except Exception as e:
        out = {"ok": False, "error": f"Failed to load load_params: {e!r}"}
        sys.stdout.write(json.dumps(out, ensure_ascii=False) + "\n")
        sys.exit(0)

    if model_type != "text":
        out = {"ok": False, "error": f"Unsupported model_type: {model_type}"}
        sys.stdout.write(json.dumps(out, ensure_ascii=False) + "\n")
        sys.exit(0)

    # 載入 TextModel（與主程式相同方式）
    from .models import TextModel  # noqa: E402

    chat_format = str(load_params.get("chat_format", "chatml"))
    n_ctx = int(load_params.get("n_ctx", 4096))
    n_gpu_layers = int(load_params.get("n_gpu_layers", 60))
    n_threads = int(load_params.get("n_threads", 8))
    model = TextModel(
        model_path=model_path,
        chat_format=chat_format,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        n_threads=n_threads,
    )

    # stdin 讀一行 JSON：{ "messages": [...], "max_tokens": ..., "temperature": ..., "json_mode": ... }
    try:
        line = sys.stdin.readline()
        if not line:
            out = {"ok": False, "error": "Empty stdin"}
            sys.stdout.write(json.dumps(out, ensure_ascii=False) + "\n")
            sys.exit(0)
        request = json.loads(line.strip())
    except Exception as e:
        out = {"ok": False, "error": f"Invalid request JSON: {e!r}"}
        sys.stdout.write(json.dumps(out, ensure_ascii=False) + "\n")
        sys.exit(0)

    messages = request.get("messages")
    if not messages:
        out = {"ok": False, "error": "Missing or empty messages"}
        sys.stdout.write(json.dumps(out, ensure_ascii=False) + "\n")
        sys.exit(0)

    max_tokens = int(request.get("max_tokens", 512))
    temperature = float(request.get("temperature", 0.2))
    json_mode = bool(request.get("json_mode", False))

    try:
        text = model.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
        )
        out = {"ok": True, "text": (text or "").strip()}
        sys.stdout.write(json.dumps(out, ensure_ascii=False) + "\n")
    except Exception as e:
        out = {"ok": False, "error": str(e)}
        sys.stdout.write(json.dumps(out, ensure_ascii=False) + "\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
