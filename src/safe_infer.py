"""
安全推理包裝器：統一處理 GPU OOM/VRAM 不足時的分級回退，避免崩潰。

提供：NeedReload 例外、chat_dispatch()（統一入口）、safe_chat()（進程內 OOM 回退）、預設 retry_plan。
鏈路：樂觀用滿(99% VRAM) → heavy 隔離(one-shot subprocess) → OOM 回退(降 tokens / CPU / give-up)。

JSON Mode / 語法約束：Run E 階段（main_group_translate、local_polish、localization 等）呼叫時
傳入 json_mode=True，會設定 response_format={"type": "json_object"} 以約束輸出為合法 JSON。
字幕標點與格式（僅 ? ! - ... "" ""、標準電影字幕風格）由 model_prompts.csv 內 prompt 約束；
若需更嚴格語法（如 llama-cpp grammar），可在此處或 models.TextModel.chat 擴充 grammar 參數。
"""

from __future__ import annotations
import gc
import time
from typing import Any, Optional

# OOM 相關字串（用於判斷是否為記憶體不足）
_OOM_PHRASES = (
    "out of memory",
    "CUDA error",
    "ggml_cuda",
    "cudaMalloc",
    "CUDA out of memory",
    "failed to allocate",
    "out of memory;",
)


class NeedReload(Exception):
    """
    表示需要外層重新載入模型後再重試。
    用於 L3（n_ctx）、L4（n_gpu_layers）、L5（CPU 模式 n_gpu_layers=0）。
    外層應 del model; gc.collect(); 依 params 重新建立 TextModel 後再呼叫 safe_chat。
    """
    def __init__(
        self,
        level: str,
        message: str = "",
        n_ctx: Optional[int] = None,
        n_gpu_layers: Optional[int] = None,
        n_batch: Optional[int] = None,
    ):
        self.level = level
        self.message = message
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.n_batch = n_batch
        self.params: dict[str, int] = {}
        if n_ctx is not None:
            self.params["n_ctx"] = n_ctx
        if n_gpu_layers is not None:
            self.params["n_gpu_layers"] = n_gpu_layers
        if n_batch is not None:
            self.params["n_batch"] = n_batch
        super().__init__(message or f"NeedReload({level}, params={self.params})")


def _is_oom_error(exc: BaseException) -> bool:
    """判斷是否為 OOM/VRAM 相關錯誤（RuntimeError/MemoryError 或訊息含 OOM 關鍵字）。"""
    if isinstance(exc, MemoryError):
        return True
    if isinstance(exc, RuntimeError):
        msg = (str(exc) or "").lower()
        return any(phrase in msg for phrase in _OOM_PHRASES)
    msg = (str(exc) or "").lower()
    return any(phrase in msg for phrase in _OOM_PHRASES)


def _release(model: Any) -> None:
    """釋放資源：model.flush_kv_cache()（若有）、或 model.llm.reset()/flush_kv_cache()，再 gc.collect()。"""
    try:
        if getattr(model, "flush_kv_cache", None) is not None and callable(model.flush_kv_cache):
            model.flush_kv_cache()
    except Exception:
        pass
    inner = getattr(model, "llm", None)
    if inner is not None:
        for name in ("flush_kv_cache", "reset"):
            fn = getattr(inner, name, None)
            if fn is not None and callable(fn):
                try:
                    fn()
                except Exception:
                    pass
                break
    gc.collect()


def default_retry_plan() -> list[dict]:
    """
    預設分級回退方案（由輕到重）。
    L1: 降低 max_tokens；L2: 降低 n_batch（若模型支援）；L3/L4/L5: 需重載（raise NeedReload）；L6: 放棄。
    """
    return [
        {"level": "L1", "max_tokens_scale": 0.6},
        {"level": "L2", "n_batch": 256},
        {"level": "L3", "n_ctx": 2048},
        {"level": "L4", "n_gpu_layers": 20},
        {"level": "L5", "n_gpu_layers": 0},
        {"level": "L6"},
    ]


def chat_dispatch(
    model: Any,
    messages: list[dict[str, Any]],
    *,
    max_tokens: int,
    temperature: float = 0.2,
    json_mode: bool = False,
    retry_plan: Optional[list[dict]] = None,
    log_lines: Optional[list[str]] = None,
    label: str = "chat_dispatch",
    sleep_after_oom_sec: float = 0.5,
    model_path: Optional[str] = None,
    load_params: Optional[dict[str, Any]] = None,
    cfg: Any = None,
    lines_count: Optional[int] = None,
    group_segments_count: Optional[int] = None,
) -> str:
    """
    所有 chat 呼叫的統一入口：樂觀用滿 → heavy 隔離 → OOM 回退。

    - 若 cfg.pipeline.isolate_heavy_requests 且 is_heavy_with_reason(...) 且提供 model_path/load_params：
      先走 run_isolated_chat（one-shot subprocess）；成功則回傳。
      失敗則 log「isolated failed, fallback in-process」並 fallback 進程內 safe_chat（降 max_tokens / CPU）。
    - 否則：直接走 in-process safe_chat。

    log_lines 會記錄：heavy 判定原因（token/lines/segments/msg_char）、是否走 isolated、回退級別（降 tokens/CPU/give-up）。
    """
    pipeline = getattr(cfg, "pipeline", None) if cfg is not None else None
    use_isolated = False
    heavy_reason = ""

    if (
        pipeline is not None
        and model_path
        and load_params
        and getattr(pipeline, "isolate_heavy_requests", True)
    ):
        from .heavy_request import is_heavy_with_reason
        from .isolated_executor import run_isolated_chat

        heavy, heavy_reason = is_heavy_with_reason(
            messages,
            max_tokens,
            pipeline,
            lines_count=lines_count,
            group_segments_count=group_segments_count,
        )
        if heavy:
            use_isolated = True
            if log_lines is not None:
                log_lines.append(f"[{label}] heavy: {heavy_reason} (isolate_heavy_requests=on)")
            timeout_sec = getattr(pipeline, "isolate_heavy_timeout_sec", 600.0)
            request_json = {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "json_mode": json_mode,
            }
            ok, text_or_err = run_isolated_chat(
                model_path, load_params, request_json, timeout_sec=timeout_sec
            )
            if ok:
                if log_lines is not None:
                    log_lines.append(f"[{label}] isolated ok")
                return (text_or_err or "").strip()
            if log_lines is not None:
                log_lines.append(f"[{label}] isolated failed: {text_or_err!r}; fallback_level: in-process safe_chat")
            # fallback to in-process safe_chat below
    else:
        if log_lines is not None and pipeline is not None and (not model_path or not load_params):
            # 未提供 model_path/load_params 時不走 isolated，僅 log 一次可選
            pass  # 不重複刷 log

    return safe_chat(
        model,
        messages,
        max_tokens=max_tokens,
        temperature=temperature,
        json_mode=json_mode,
        retry_plan=retry_plan,
        log_lines=log_lines,
        label=label,
        sleep_after_oom_sec=sleep_after_oom_sec,
    )


def safe_chat(
    model: Any,
    messages: list[dict[str, Any]],
    *,
    max_tokens: int,
    temperature: float = 0.2,
    json_mode: bool = False,
    retry_plan: Optional[list[dict]] = None,
    log_lines: Optional[list[str]] = None,
    label: str = "safe_chat",
    sleep_after_oom_sec: float = 0.5,
) -> str:
    """
    進程內 chat + OOM 分級回退（由 chat_dispatch 呼叫，或直接呼叫時不經 heavy 隔離）。

    - 捕捉 RuntimeError / MemoryError 及訊息含 OOM 關鍵字。
    - 每次 OOM 後：_release(model)、gc.collect()、可選 short sleep，再依下一級重試。
    - L1: 僅降低 max_tokens 後重試（fallback_level: reduce max_tokens）。
    - L2: 若 model 支援動態 n_batch，設定後重試。
    - L3/L4/L5: 需重載模型，raise NeedReload；外層重載後再呼叫（fallback_level: reload n_ctx / n_gpu_layers / CPU）。
    - L6: 放棄，回傳空字串（fallback_level: give-up）。

    log_lines: 每次 OOM 與回退級別會 append（含 label、觸發原因、fallback_level）。
    """
    plan = retry_plan if retry_plan is not None else default_retry_plan()
    current_max_tokens = max(64, max_tokens)
    attempt = 0

    while True:
        try:
            # TextModel.chat 或 Llama.create_chat_completion 介面
            if hasattr(model, "chat"):
                out = model.chat(
                    messages,
                    temperature=temperature,
                    max_tokens=current_max_tokens,
                    json_mode=json_mode,
                )
            else:
                kwargs = dict(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=current_max_tokens,
                )
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                out = model.create_chat_completion(**kwargs)["choices"][0]["message"]["content"]
            return (out or "").strip()
        except (RuntimeError, MemoryError) as e:
            if not _is_oom_error(e):
                raise
            reason = str(e).strip() or type(e).__name__
            _release(model)
            gc.collect()
            if sleep_after_oom_sec > 0:
                time.sleep(sleep_after_oom_sec)

            if attempt >= len(plan):
                if log_lines is not None:
                    log_lines.append(f"[{label}] OOM: {reason!r}; no more retry levels; fallback_level: give-up")
                return ""

            # 依序套用回退級別；L1/L2 套用後 break 重試 chat，L3/L4/L5 raise NeedReload，L6 或耗盡則回傳空字串
            will_retry_chat = False
            for step in plan[attempt:]:
                attempt += 1
                level = step.get("level", "")
                if log_lines is not None:
                    log_lines.append(f"[{label}] OOM: {reason!r}; applying retry level {level} (step {attempt}/{len(plan)}).")

                if level == "L1":
                    scale = float(step.get("max_tokens_scale", 0.6))
                    current_max_tokens = max(64, int(current_max_tokens * scale))
                    if log_lines is not None:
                        log_lines.append(f"[{label}] fallback_level: reduce max_tokens -> {current_max_tokens}")
                    will_retry_chat = True
                    break
                if level == "L2":
                    n_batch_val = step.get("n_batch")
                    if n_batch_val is not None:
                        if hasattr(model, "set_n_batch") and callable(model.set_n_batch):
                            try:
                                model.set_n_batch(n_batch_val)
                                if log_lines is not None:
                                    log_lines.append(f"[{label}] fallback_level: n_batch={n_batch_val}")
                                will_retry_chat = True
                                break
                            except Exception:
                                pass
                        if hasattr(model, "llm") and getattr(getattr(model, "llm", None), "set_n_batch", None) is not None:
                            try:
                                model.llm.set_n_batch(n_batch_val)
                                if log_lines is not None:
                                    log_lines.append(f"[{label}] fallback_level: n_batch={n_batch_val}")
                                will_retry_chat = True
                                break
                            except Exception:
                                pass
                    continue
                if level == "L3":
                    if log_lines is not None:
                        log_lines.append(f"[{label}] fallback_level: reload n_ctx={step.get('n_ctx')}")
                    raise NeedReload("L3", f"OOM; reload with n_ctx={step.get('n_ctx')}", n_ctx=step.get("n_ctx"))
                if level == "L4":
                    if log_lines is not None:
                        log_lines.append(f"[{label}] fallback_level: reload n_gpu_layers={step.get('n_gpu_layers')}")
                    raise NeedReload("L4", f"OOM; reload with n_gpu_layers={step.get('n_gpu_layers')}", n_gpu_layers=step.get("n_gpu_layers"))
                if level == "L5":
                    if log_lines is not None:
                        log_lines.append(f"[{label}] fallback_level: reload CPU (n_gpu_layers=0)")
                    raise NeedReload("L5", "OOM; reload with n_gpu_layers=0 (CPU)", n_gpu_layers=0)
                if level == "L6":
                    if log_lines is not None:
                        log_lines.append(f"[{label}] fallback_level: give-up (L6)")
                    return ""
                if log_lines is not None:
                    log_lines.append(f"[{label}] fallback_level: give-up")
                return ""

            if not will_retry_chat:
                return ""
