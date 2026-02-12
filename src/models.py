from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Any

from llama_cpp import Llama
from llama_cpp.llama_chat_format import MoondreamChatHandler


def _check_sharded_model(model_path: str) -> tuple[bool, list[str]]:
    """
    檢查分片模型的所有分片是否存在。
    返回 (is_sharded, missing_shards)
    """
    model_path_obj = Path(model_path)
    
    # 檢查是否為分片模型（格式：*-00001-of-00003.gguf）
    stem = model_path_obj.stem
    if "-00001-of-" in stem:
        # 提取分片信息
        try:
            # 格式：base-00001-of-00003
            parts = stem.split("-00001-of-")
            if len(parts) == 2:
                base = parts[0]
                # 提取總分片數（可能後面還有其他內容）
                total_shards_str = parts[1]
                # 移除可能的後綴，只取數字部分
                total_shards = int(total_shards_str.split("-")[0])
                missing = []
                for i in range(1, total_shards + 1):
                    shard_name = f"{base}-{i:05d}-of-{total_shards:05d}{model_path_obj.suffix}"
                    shard_path = model_path_obj.parent / shard_name
                    if not shard_path.exists():
                        missing.append(str(shard_path))
                return True, missing
        except (ValueError, IndexError) as e:
            # 如果解析失敗，不視為分片模型
            pass
    
    # 如果不是分片模型，檢查文件是否存在
    if not model_path_obj.exists():
        return False, [str(model_path_obj)]
    
    return False, []


def _handle_llama_error(e: Exception, model_path: str) -> Exception:
    """
    處理 Llama 模型加載錯誤，提供更友好的錯誤信息。
    """
    error_msg = str(e)
    
    # Windows CPU 指令集錯誤
    if "0xc000001d" in error_msg or "WinError -1073741795" in error_msg:
        return RuntimeError(
            f"CPU instruction set incompatibility error. This usually means your CPU does not support AVX/AVX2 instruction set.\n"
            f"Solutions:\n"
            f"1. Reinstall llama-cpp-python CPU-only version:\n"
            f"   pip uninstall llama-cpp-python\n"
            f"   pip install llama-cpp-python --no-cache-dir\n"
            f"2. Or use a precompiled version that supports your CPU\n"
            f"3. Model path: {model_path}\n"
            f"Original error: {error_msg}"
        )
    
    # 模型文件加載失敗
    if "Failed to load model" in error_msg:
        is_sharded, missing = _check_sharded_model(model_path)
        if is_sharded and missing:
            return FileNotFoundError(
                f"Sharded model missing the following shard files:\n" + "\n".join(missing) + 
                f"\n\nPlease ensure all shards are in the same directory: {Path(model_path).parent}\n"
                f"Original error: {error_msg}"
            )
        else:
            return FileNotFoundError(
                f"Failed to load model file: {model_path}\n"
                f"Please check if the file exists and is complete.\n"
                f"Original error: {error_msg}"
            )
    
    # 其他錯誤，保持原樣但添加模型路徑信息
    return type(e)(f"Error occurred while loading model ({model_path}): {error_msg}")


class TextModel:
    def __init__(self, model_path: str, chat_format: str, n_ctx: int, n_gpu_layers: int, n_threads: int):
        # 使用 config 的 n_gpu_layers（例如 60）；-1 表示全部上 GPU，VRAM 不足時可能觸發 node_backend_id != -1
        # 若 VRAM 充足可在 config.json 將 n_gpu_layers_reason / n_gpu_layers_translate 設為 -1

        # 檢查分片模型
        is_sharded, missing = _check_sharded_model(model_path)
        if missing:
            raise FileNotFoundError(
                f"Sharded model missing the following shard files:\n" + "\n".join(missing) + 
                f"\n\nPlease ensure all shards are in the same directory: {Path(model_path).parent}"
            )

        resolved = Path(model_path).resolve()
        # Windows 多分片 GGUF 易觸發 GGML_ASSERT(tensor buffer not set)：C++ 依路徑推導其他分片，
        # 改為「先 chdir 到模型目錄、只傳檔名」讓相對路徑一致，避免崩潰。
        old_cwd = None
        if sys.platform == "win32" and is_sharded:
            model_dir = resolved.parent
            old_cwd = os.getcwd()
            os.chdir(model_dir)
            path_for_llama = resolved.name
        elif sys.platform == "win32":
            path_for_llama = str(resolved)
        else:
            path_for_llama = resolved.as_posix()
        try:
            self.llm = Llama(
                model_path=path_for_llama,
                chat_format=chat_format,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                n_threads=n_threads,
                verbose=False,
            )
            # 檢查 GPU 使用情況（如果 n_gpu_layers > 0）
            if n_gpu_layers > 0:
                try:
                    # llama-cpp-python 沒有直接 API 檢查 GPU，但可以通過檢查 backend 來推斷
                    # 這裡我們記錄參數，實際 GPU 使用會在推理時體現
                    self._n_gpu_layers = n_gpu_layers
                except Exception:
                    pass
        except Exception as e:
            raise _handle_llama_error(e, model_path) from e
        finally:
            if old_cwd is not None:
                try:
                    os.chdir(old_cwd)
                except Exception:
                    pass

    def chat(self, messages: list[dict[str, Any]], temperature: float = 0.2, max_tokens: int = 512, json_mode: bool = False):
        kwargs = dict(messages=messages, temperature=temperature, max_tokens=max_tokens)
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        return self.llm.create_chat_completion(**kwargs)["choices"][0]["message"]["content"]

class VisionModel:
    """Moondream2 via llama-cpp-python's chat handler.

    This is the most 'zero-extra-tool' way to do local vision with llama.cpp bindings.
    It requires **local** GGUF files (no auto-download).
    """
    def __init__(
        self,
        text_model_path: str,
        mmproj_path: str,
        n_ctx: int = 2048,
        n_gpu_layers: int = 60,
        n_threads: int = 8,
    ):
        text_model_path = str(Path(text_model_path))
        mmproj_path = str(Path(mmproj_path))

        if not Path(text_model_path).exists():
            raise FileNotFoundError(f"Vision text model not found: {text_model_path}")
        if not Path(mmproj_path).exists():
            raise FileNotFoundError(f"Vision mmproj not found: {mmproj_path}")

        # Most llama-cpp-python multimodal chat handlers accept `clip_model_path`.
        # If your wheel has a slightly different signature, the error message will
        # point you to the exact parameter name.
        chat_handler = MoondreamChatHandler(clip_model_path=mmproj_path)

        path_for_llama = str(Path(text_model_path).resolve().as_posix())
        try:
            self.llm = Llama(
                model_path=path_for_llama,
                chat_handler=chat_handler,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                n_threads=n_threads,
                verbose=False,
            )
        except Exception as e:
            raise _handle_llama_error(e, text_model_path) from e

    def describe_with_grounding(self, image_bytes: bytes, subtitle_text: str) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful vision assistant. Be concise."},
            {"role": "user", "content": [
                {"type": "text", "text":
                    "Analyze the relationship between the image and the subtitle text. "
                    "1) Briefly describe what's happening in the image. "
                    "2) Explain what the subtitle most likely means in THIS visual context (idiom/pun/metaphor if relevant). "
                    f"Subtitle: {subtitle_text!r}. "
                    "Output in 2 short lines prefixed with [Context] and [Meaning]."
                },
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + _b64(image_bytes)}}
            ]}
        ]
        out = self.llm.create_chat_completion(messages=messages, temperature=0.1, max_tokens=256)["choices"][0]["message"]["content"]
        return out.strip()

def _b64(b: bytes) -> str:
    import base64
    return base64.b64encode(b).decode("ascii")
