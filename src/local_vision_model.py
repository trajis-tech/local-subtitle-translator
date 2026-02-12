"""
LocalVisionModel - 通用視覺模型封裝類別，自動檢測並支援所有 llama-cpp-python 支援的視覺模型

此類別會自動檢測模型類型並選擇適當的 ChatHandler，無需手動指定。
支援的模型包括但不限於：Moondream2, LLaVA-v1.6, LLaVA-1.5, BakLLaVA, 以及其他所有
llama-cpp-python 支援的視覺語言模型。
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Union, Optional
import base64
import logging
import importlib

from llama_cpp import Llama

# 設定日誌記錄器
logger = logging.getLogger(__name__)


def _image_to_base64(image_path: str) -> str:
    """
    將圖片檔案轉換為 Base64 編碼字串。
    
    Args:
        image_path: 圖片檔案路徑
        
    Returns:
        Base64 編碼的字串（不含 data URI 前綴）
        
    Raises:
        FileNotFoundError: 當圖片檔案不存在時
        IOError: 當無法讀取圖片檔案時
    """
    image_path_obj = Path(image_path)
    
    if not image_path_obj.exists():
        raise FileNotFoundError(f"圖片檔案不存在: {image_path}")
    
    try:
        with open(image_path_obj, "rb") as f:
            image_bytes = f.read()
        base64_str = base64.b64encode(image_bytes).decode("ascii")
        logger.debug(f"Successfully converted image to Base64: {image_path}")
        return base64_str
    except IOError as e:
        raise IOError(f"Failed to read image file {image_path}: {e}") from e


def _detect_model_type_from_filename(model_path: str) -> Optional[str]:
    """
    根據檔案名稱自動檢測模型類型。
    
    Args:
        model_path: 模型檔案路徑
        
    Returns:
        檢測到的模型類型字串，或 None（無法檢測時）
    """
    model_name_lower = Path(model_path).name.lower()
    
    # 常見的模型類型檢測規則（按優先順序）
    detection_rules = [
        ("moondream", "moondream"),
        ("llava", "llava"),
        ("bakllava", "bakllava"),
        ("minicpm-v", "minicpm-v"),
        ("qwen-vl", "qwen-vl"),
        ("cogvlm", "cogvlm"),
        ("yi-vl", "yi-vl"),
    ]
    
    for keyword, model_type in detection_rules:
        if keyword in model_name_lower:
            logger.debug(f"Detected model type from filename: {model_type} (keyword: {keyword})")
            return model_type
    
    return None


def _get_chat_handler_for_model_type(model_type: Optional[str], clip_model_path: str) -> Any:
    """
    根據模型類型動態導入並創建對應的 ChatHandler。
    
    此函數會嘗試導入所有可能的 ChatHandler，直到找到一個能成功創建的。
    
    Args:
        model_type: 模型類型（例如 'moondream', 'llava', 'bakllava' 等）
        clip_model_path: CLIP/mmproj 模型檔案路徑
        
    Returns:
        ChatHandler 實例
        
    Raises:
        RuntimeError: 當無法找到或創建合適的 ChatHandler 時
    """
    from llama_cpp.llama_chat_format import MoondreamChatHandler
    
    # 定義模型類型到 ChatHandler 的映射
    # 格式: (模型類型關鍵字列表, ChatHandler 類別名稱列表, 參數名稱)
    handler_mappings = [
        # Moondream2
        (["moondream"], ["MoondreamChatHandler"], "clip_model_path"),
        
        # LLaVA 系列（嘗試多個可能的類別名稱）
        (["llava"], ["Llava15ChatHandler", "LlavaChatHandler", "Llava16ChatHandler"], "clip_model_path"),
        
        # BakLLaVA
        (["bakllava"], ["BakllavaChatHandler"], "clip_model_path"),
        
        # MiniCPM-V
        (["minicpm-v", "minicpm"], ["MiniCpmVChatHandler"], "clip_model_path"),
        
        # Qwen-VL
        (["qwen-vl", "qwenvl"], ["QwenVLChatHandler"], "clip_model_path"),
        
        # CogVLM
        (["cogvlm"], ["CogVlmChatHandler"], "clip_model_path"),
        
        # Yi-VL
        (["yi-vl", "yivl"], ["YiVLChatHandler"], "clip_model_path"),
    ]
    
    # 如果提供了 model_type（且不是 "auto"），優先使用
    if model_type and model_type.lower() != "auto":
        model_type_lower = model_type.lower()
        for keywords, handler_names, param_name in handler_mappings:
            if any(kw in model_type_lower for kw in keywords):
                for handler_name in handler_names:
                    try:
                        handler_class = getattr(
                            importlib.import_module("llama_cpp.llama_chat_format"),
                            handler_name,
                            None
                        )
                        if handler_class:
                            kwargs = {param_name: clip_model_path}
                            handler = handler_class(**kwargs)
                            logger.info(f"Successfully initialized {model_type} model using {handler_name}")
                            return handler
                    except (ImportError, AttributeError, TypeError) as e:
                        logger.debug(f"Cannot use {handler_name}: {e}")
                        continue
    
    # 如果沒有指定 model_type 或找不到對應的 handler，嘗試所有可用的
    logger.info("Attempting to auto-detect and load ChatHandler...")
    
    # 首先嘗試 Moondream（最常見）
    try:
        handler = MoondreamChatHandler(clip_model_path=clip_model_path)
        logger.info("Successfully used MoondreamChatHandler (auto-detected)")
        return handler
    except Exception as e:
        logger.debug(f"MoondreamChatHandler failed: {e}")
    
    # 嘗試其他常見的 ChatHandler
    common_handlers = [
        "Llava15ChatHandler",
        "LlavaChatHandler",
        "Llava16ChatHandler",
        "BakllavaChatHandler",
    ]
    
    for handler_name in common_handlers:
        try:
            handler_class = getattr(
                importlib.import_module("llama_cpp.llama_chat_format"),
                handler_name,
                None
            )
            if handler_class:
                handler = handler_class(clip_model_path=clip_model_path)
                logger.info(f"Successfully used {handler_name} (auto-detected)")
                return handler
        except (ImportError, AttributeError, TypeError) as e:
            logger.debug(f"Cannot use {handler_name}: {e}")
            continue
    
    # 如果所有嘗試都失敗，拋出錯誤
    raise RuntimeError(
        f"Unable to find suitable ChatHandler.\n"
        f"Model path: {Path(clip_model_path).name}\n"
        f"Please verify:\n"
        f"1. llama-cpp-python version supports this vision model\n"
        f"2. Corresponding ChatHandler is installed\n"
        f"3. Model file format is correct\n"
        f"Tip: Try updating llama-cpp-python: pip install --upgrade llama-cpp-python"
    )


class LocalVisionModel:
    """
    通用本地視覺模型封裝類別，自動檢測並支援所有 llama-cpp-python 支援的視覺模型。
    
    此類別會自動檢測模型類型並選擇適當的 ChatHandler，無需手動指定。
    支援的模型包括但不限於：
    - Moondream2
    - LLaVA-v1.6 / LLaVA-1.5
    - BakLLaVA
    - MiniCPM-V
    - Qwen-VL
    - CogVLM
    - Yi-VL
    - 以及其他所有 llama-cpp-python 支援的視覺語言模型
    
    範例:
        # 自動檢測模型類型
        model = LocalVisionModel(
            model_path="models/llava-v1.6-mistral-7b.gguf",
            clip_model_path="models/mmproj-model-f16.gguf"
        )
        
        # 手動指定模型類型（可選）
        model = LocalVisionModel(
            model_path="models/moondream2-text-model-f16.gguf",
            clip_model_path="models/moondream2-mmproj-f16.gguf",
            model_type="moondream"
        )
        
        # 分析單張圖片
        result = model.analyze("path/to/image.jpg", "描述這張圖片")
        
        # 分析多張圖片
        result = model.analyze(
            ["path/to/image1.jpg", "path/to/image2.jpg"],
            "比較這兩張圖片的差異"
        )
    """
    
    def __init__(
        self,
        model_path: str,
        clip_model_path: str,
        model_type: Optional[str] = None,
        n_ctx: int | None = None,
        n_threads: int = 8,
    ):
        """
        初始化 LocalVisionModel。
        
        Args:
            model_path: 主模型檔案路徑（GGUF 格式）
            clip_model_path: CLIP/mmproj 模型檔案路徑（GGUF 格式）
            model_type: 模型類型（可選）。若為 None，會自動從檔案名稱檢測。
                        支援的值包括：'moondream', 'llava', 'bakllava', 'minicpm-v', 
                        'qwen-vl', 'cogvlm', 'yi-vl' 等
            n_ctx: 上下文長度。若為 None，會根據模型類型使用預設值：
                   - LLaVA 系列: 8192
                   - 其他模型: 2048
            n_threads: CPU 執行緒數（預設：8）
            
        Raises:
            FileNotFoundError: 當模型檔案不存在時
            RuntimeError: 當模型載入失敗或無法找到合適的 ChatHandler 時
        """
        model_path_obj = Path(model_path)
        clip_model_path_obj = Path(clip_model_path)
        
        # 檢查檔案是否存在
        if not model_path_obj.exists():
            raise FileNotFoundError(f"Main model file not found: {model_path}")
        if not clip_model_path_obj.exists():
            raise FileNotFoundError(f"CLIP/mmproj model file not found: {clip_model_path}")
        
        self.model_path = str(model_path_obj.resolve())
        self.clip_model_path = str(clip_model_path_obj.resolve())

        # 與主模型 / models.VisionModel 一致：傳給 Llama 與 ChatHandler 的路徑使用 as_posix() 正規化，
        # 避免 Windows 反斜線或路徑含空格時底層載入錯誤（例如 CLIP 被誤從主模型路徑載入）。
        self._path_for_llama = str(model_path_obj.resolve().as_posix())
        self._clip_path_for_handler = str(clip_model_path_obj.resolve().as_posix())

        # 自動檢測模型類型（如果未指定）
        if model_type is None:
            model_type = _detect_model_type_from_filename(self.model_path)
            if model_type:
                logger.info(f"Auto-detected model type: {model_type}")
            else:
                logger.warning("Unable to detect model type from filename, will try auto-selecting ChatHandler")
        
        # 保存原始 model_type（用於 n_ctx 判斷）
        detected_type_for_ctx = model_type
        
        self.model_type = model_type or "auto"
        
        # 設定預設 n_ctx（根據檢測到的模型類型）
        if n_ctx is None:
            # 使用 detected_type_for_ctx 而不是 self.model_type，因為後者可能是 "auto"
            if detected_type_for_ctx and "llava" in detected_type_for_ctx.lower():
                n_ctx = 8192  # LLaVA 系列建議使用較大的上下文
            else:
                n_ctx = 2048  # 其他模型預設值
        
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        
        # 動態獲取 ChatHandler
        logger.info(f"Initializing vision model...")
        logger.info(f"  Main model path: {self.model_path}")
        logger.info(f"  CLIP/mmproj path: {self.clip_model_path}")
        logger.info(f"  CLIP/mmproj exists: {Path(self.clip_model_path).exists()}")
        logger.info(f"  Model type: {self.model_type}")
        logger.info(f"  Context length: {self.n_ctx}")
        
        # Verify mmproj file exists (double-check before passing to handler)
        if not Path(self.clip_model_path).exists():
            raise FileNotFoundError(
                f"CLIP/mmproj file does not exist at: {self.clip_model_path}\n"
                f"Please ensure both the main model and mmproj files are in ./models/vision/"
            )
        
        try:
            # 傳入正規化後的 clip 路徑，與 models.VisionModel 一致，避免底層誤用主模型路徑載入 CLIP
            chat_handler = _get_chat_handler_for_model_type(self.model_type, self._clip_path_for_handler)

            # 優先使用 GPU 加速（n_gpu_layers=-1 表示所有層都載入到 GPU）
            # 與主模型載入一致：model_path 使用 as_posix() 正規化路徑（見 models.py TextModel / VisionModel）
            logger.info("Loading model (preferring GPU acceleration)...")
            self.llm = Llama(
                model_path=self._path_for_llama,
                chat_handler=chat_handler,
                n_ctx=self.n_ctx,
                n_gpu_layers=-1,  # -1 = 所有層都載入到 GPU，充分利用 VRAM
                n_threads=self.n_threads,
                verbose=False,
            )
            logger.info(f"✓ Vision model loaded successfully (type: {self.model_type}, preferring GPU)")
            
        except Exception as e:
            # Check for common issues
            has_space_in_path = " " in self.model_path or " " in self.clip_model_path
            space_warning = (
                "\n  ⚠️ WARNING: Path contains spaces. Some llama-cpp-python versions may have issues with spaces in paths.\n"
                "  Try moving models to a path without spaces (e.g. D:\\local-subtitle-translator\\models\\vision\\)"
            ) if has_space_in_path else ""
            
            error_msg = (
                f"Error occurred while loading vision model:\n"
                f"  Main model: {self.model_path}\n"
                f"  CLIP/mmproj: {self.clip_model_path}\n"
                f"  CLIP/mmproj exists: {Path(self.clip_model_path).exists()}\n"
                f"  CLIP/mmproj size: {Path(self.clip_model_path).stat().st_size if Path(self.clip_model_path).exists() else 'N/A'} bytes"
                f"{space_warning}\n"
                f"  Original error: {e}\n"
                f"  CLIP/mmproj: {self.clip_model_path}\n"
                f"  Model type: {self.model_type}\n"
                f"  Error message: {str(e)}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def analyze(
        self,
        image_input: Union[str, list[str]],
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 512,
    ) -> str:
        """
        分析圖片並根據提示詞生成回應。
        
        此方法支援單一圖片或多張圖片輸入。對於多張圖片，會將所有圖片
        都加入到 messages 的 content 中，讓模型能夠同時看到所有圖片。
        
        Args:
            image_input: 圖片路徑（字串）或圖片路徑列表
            prompt: 提示詞（要問模型的問題或指令）
            temperature: 生成溫度（預設：0.1，較低溫度產生更確定性的輸出）
            max_tokens: 最大生成 token 數（預設：512）
            
        Returns:
            模型生成的回應文字
            
        Raises:
            FileNotFoundError: 當圖片檔案不存在時
            ValueError: 當 image_input 為空列表時
            RuntimeError: 當模型推理失敗時
        """
        # 處理單圖或多圖輸入
        if isinstance(image_input, str):
            image_paths = [image_input]
        elif isinstance(image_input, list):
            if len(image_input) == 0:
                raise ValueError("圖片路徑列表不能為空")
            image_paths = image_input
        else:
            raise TypeError(
                f"不支援的 image_input 類型: {type(image_input)}。"
                f"請使用 str（單圖）或 list[str]（多圖）。"
            )
        
        # 檢查所有圖片檔案是否存在
        for img_path in image_paths:
            if not Path(img_path).exists():
                raise FileNotFoundError(f"Image file not found: {img_path}")
        
        # 將所有圖片轉換為 Base64
        logger.debug(f"Processing {len(image_paths)} images...")
        image_base64_list = [_image_to_base64(img_path) for img_path in image_paths]
        
        # 構建符合 OpenAI Chat Format 的 messages
        # 多圖時，在 content 中 append 多個 image_url
        content: list[dict[str, Any]] = [
            {"type": "text", "text": prompt}
        ]
        
        # 為每張圖片添加 image_url
        for base64_str in image_base64_list:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_str}"
                }
            })
        
        messages = [
            {
                "role": "system",
                "content": "You are a helpful vision assistant. Please answer questions concisely and clearly."
            },
            {
                "role": "user",
                "content": content
            }
        ]
        
        # 執行推理
        try:
            logger.debug(f"Running inference ({len(image_paths)} images, prompt: {prompt[:50]}...)")
            response = self.llm.create_chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            result = response["choices"][0]["message"]["content"].strip()
            logger.debug(f"Inference completed, response length: {len(result)} characters")
            return result
            
        except Exception as e:
            error_msg = (
                f"Error occurred during model inference:\n"
                f"  Model type: {self.model_type}\n"
                f"  Image count: {len(image_paths)}\n"
                f"  Error message: {str(e)}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def describe_with_grounding(self, image_bytes: bytes, subtitle_text: str) -> str:
        """
        向後兼容方法：使用圖片位元組和字幕文字進行分析。
        
        此方法保持與舊版 VisionModel 的介面一致，內部使用 analyze 方法。
        
        Args:
            image_bytes: 圖片的 JPEG 位元組資料
            subtitle_text: 字幕文字（用於構建提示詞）
            
        Returns:
            模型生成的回應文字
        """
        import tempfile
        
        # 將位元組資料寫入臨時檔案
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.jpg', delete=False) as tmp_file:
            tmp_file.write(image_bytes)
            tmp_path = tmp_file.name
        
        try:
            # 構建提示詞（與舊版 VisionModel 保持一致）
            prompt = (
                "Analyze the relationship between the image and the subtitle text. "
                "1) Briefly describe what's happening in the image. "
                "2) Explain what the subtitle most likely means in THIS visual context (idiom/pun/metaphor if relevant). "
                f"Subtitle: {subtitle_text!r}. "
                "Output in 2 short lines prefixed with [Context] and [Meaning]."
            )
            
            # 使用 analyze 方法
            result = self.analyze(tmp_path, prompt, temperature=0.1, max_tokens=256)
            return result
        finally:
            # 清理臨時檔案
            try:
                Path(tmp_path).unlink()
            except Exception:
                pass


if __name__ == "__main__":
    """
    使用範例：展示如何初始化不同類型的視覺模型並呼叫 analyze 函數。
    """
    import logging
    
    # 設定日誌級別為 INFO，以便看到載入訊息
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    print("=" * 60)
    print("LocalVisionModel Usage Examples (Auto-detect Model Type)")
    print("=" * 60)
    
    # ========== Example 1: Auto-detect LLaVA model ==========
    print("\n[Example 1] Auto-detect and initialize LLaVA model")
    print("-" * 60)
    try:
        llava_model = LocalVisionModel(
            model_path="models/llava/llava-v1.6-mistral-7b.gguf",
            clip_model_path="models/llava/mmproj-model-f16.gguf",
            # model_type is optional, will auto-detect from filename
        )
        
        result = llava_model.analyze(
            image_input="test_images/sample.jpg",
            prompt="Please describe the contents of this image in detail."
        )
        print(f"LLaVA response:\n{result}\n")
        
    except FileNotFoundError as e:
        print(f"⚠️  File not found (expected, as this is an example): {e}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # ========== Example 2: Auto-detect Moondream2 model ==========
    print("\n[Example 2] Auto-detect and initialize Moondream2 model")
    print("-" * 60)
    try:
        moondream_model = LocalVisionModel(
            model_path="models/moondream2/moondream2-text-model-f16.gguf",
            clip_model_path="models/moondream2/moondream2-mmproj-f16.gguf",
        )
        
        result = moondream_model.analyze(
            image_input=["test_images/frame1.jpg", "test_images/frame2.jpg"],
            prompt="Describe the action changes between these two consecutive frames."
        )
        print(f"Moondream2 response:\n{result}\n")
        
    except FileNotFoundError as e:
        print(f"⚠️  File not found (expected, as this is an example): {e}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # ========== Example 3: Manually specify model type ==========
    print("\n[Example 3] Manually specify model type (optional)")
    print("-" * 60)
    try:
        custom_model = LocalVisionModel(
            model_path="models/custom/custom-vision-model.gguf",
            clip_model_path="models/custom/custom-mmproj.gguf",
            model_type="llava",  # Manually specify type
            n_ctx=4096,  # Custom context length
        )
        
        result = custom_model.analyze(
            image_input="test_images/sample.jpg",
            prompt="What is happening in this image?"
        )
        print(f"Custom model response:\n{result}\n")
        
    except FileNotFoundError as e:
        print(f"⚠️  File not found (expected, as this is an example): {e}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("Examples execution completed")
    print("=" * 60)
