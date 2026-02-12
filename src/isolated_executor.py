"""
主程式端：透過 subprocess 呼叫 worker_infer 執行單次 chat，僅在 heavy request 時使用。
one-shot worker 確保 VRAM 在子行程 exit 後完全釋放，避免主程式被 OOM/driver error 拖死。
"""

from __future__ import annotations
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Tuple


def run_isolated_chat(
    model_path: str,
    load_params: dict[str, Any],
    request_json: dict[str, Any],
    timeout_sec: float = 600.0,
) -> Tuple[bool, str]:
    """
    以子行程執行 python -m src.worker_infer，將 request_json 經 stdin 傳入，回傳 (ok, text_or_error)。

    - model_path: GGUF 模型路徑
    - load_params: TextModel 載入參數，例如 {"chat_format": "chatml", "n_ctx": 8192, "n_gpu_layers": 60, "n_threads": 8}
    - request_json: {"messages": [...], "max_tokens": int, "temperature": float, "json_mode": bool}
    - timeout_sec: 逾時則 kill 子行程，回 (False, "timeout")

    回傳:
    - (True, text): 成功，text 為模型輸出
    - (False, error_message): 失敗（含 timeout、JSON 解析失敗、worker 回傳 ok:false）
    """
    fd, load_params_path = tempfile.mkstemp(suffix=".json", prefix="worker_load_")
    try:
        with open(fd, "w", encoding="utf-8") as f:
            json.dump(load_params, f, ensure_ascii=False)
        request_line = json.dumps(request_json, ensure_ascii=False) + "\n"
        stdin_bytes = request_line.encode("utf-8")
        cmd = [
            sys.executable,
            "-m",
            "src.worker_infer",
            str(model_path),
            "text",
            load_params_path,
        ]
        try:
            result = subprocess.run(
                cmd,
                input=stdin_bytes,
                capture_output=True,
                timeout=timeout_sec,
                cwd=Path(__file__).resolve().parent.parent,
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
        except Exception as e:
            return False, str(e)
        finally:
            try:
                Path(load_params_path).unlink(missing_ok=True)
            except Exception:
                pass

        out = (result.stdout or b"").decode("utf-8", errors="replace").strip()
        if not out:
            err = (result.stderr or b"").decode("utf-8", errors="replace").strip()
            return False, err or "empty stdout"
        try:
            obj = json.loads(out)
        except json.JSONDecodeError as e:
            return False, f"worker stdout not JSON: {e!r}"
        if obj.get("ok") is True:
            return True, (obj.get("text") or "")
        return False, (obj.get("error") or "unknown error")
    finally:
        try:
            Path(load_params_path).unlink(missing_ok=True)
        except Exception:
            pass
