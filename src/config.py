from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
import json
from typing import Any


# -----------------------------
# Sub-configs
# -----------------------------
@dataclass
class LlamaCppConfig:
    # Context / performance knobs
    n_ctx_reason: int = 8192
    n_ctx_translate: int = 4096

    # GPU offload layers (tune by VRAM). -1 = all layers on GPU; number = first N layers on GPU.
    # If you see GGML_ASSERT(node_backend_id != -1), lower this (e.g. 35) or set -1 only when VRAM is ample.
    n_gpu_layers_reason: int = 60
    n_gpu_layers_translate: int = 60

    # CPU threads
    n_threads: int = 8


@dataclass
class PipelineConfig:
    # how many previous EN lines to include in Stage2 prompt
    context_prev_lines: int = 2

    n_frames: int = 3  # Number of frames for multi-frame vision (Run D)
    
    # Work directory for intermediate results (JSONL files)
    work_dir: str = "./work"
    
    # Performance optimization settings
    # Batch processing: process items in batches to optimize memory usage
    batch_size: int | None = None  # None = auto-detect based on available resources
    min_batch_size: int = 8  # Minimum batch size (ensures low-resource systems can run)
    max_batch_size: int = 128  # Maximum batch size
    # Run B Brief: items per single chat request (multi-item in one request); 16~32 recommended to avoid token explosion
    request_batch_size: int = 24
    
    # Parallel processing: use multiple CPU cores for I/O-bound tasks
    enable_parallel: bool = True  # Enable parallel processing
    max_workers: int | None = None  # None = auto-detect (CPU count - 1)
    
    # Memory optimization: release unused data immediately
    enable_memory_optimization: bool = True  # Release brief_v1/v2/v3 after use

    # Heavy request isolation (avoid VRAM/context blow-up)
    # 門檻設高，僅「真的大」請求才 isolate，避免 local_polish 正常 chunk(60 行) 觸發雙載
    heavy_token_threshold: int = 6000  # total estimated tokens (input + max_tokens) >= this → heavy
    heavy_msg_char_threshold: int = 6000  # any message content length >= this → heavy
    heavy_lines_threshold: int = 150  # JSON batch lines count >= this → heavy (e.g. local_polish >150 才 isolate)
    # 注意：group_translate_max_segments 預設是 4，若 heavy_group_segments_threshold 也設 4，
    # 會導致 main_group_translate 很容易觸發 isolated subprocess，進而「同時載入兩份模型」違反單模型規則並增加 OOM 風險。
    # 因此預設設為 > group_translate_max_segments。
    heavy_group_segments_threshold: int = 8  # group segments >= this → heavy
    local_polish_chunk_size: int = 60  # chunk lines per batch for local_polish
    group_translate_max_segments: int = 4  # max segments per group; over this split or cap

    # Heavy request subprocess isolation (avoid OOM/driver error killing main process)
    isolate_heavy_requests: bool = True  # when True, heavy requests run in one-shot subprocess worker
    isolate_heavy_timeout_sec: float = 600.0  # timeout for isolated worker

    # Run F scheme: "full" | "main_led" | "local_led" | "draft_first" (see README)
    run_e_scheme: str = "full"
    # Run E: 是否啟用「更多上下文」備援（need_more_context 時用 prev-3/next-3 更新 brief）
    enable_context_expansion: bool = True

    # Local polish mode: "weak" = input stripped of punctuation, make fluent + transliterate names;
    # "strong" = input keeps punctuation, make fluent + more localized + transliterate names; output: first char of each line must NOT be preceded by punctuation
    local_polish_mode: str = "strong"  # "weak" | "strong"

    # Strip punctuation (Run F final output): when True, remove punctuation; optional protection for decimals/acronyms
    strip_punctuation: bool = True  # when False, do not strip punctuation
    strip_punctuation_keep_decimal: bool = True  # protect e.g. 3.14 from becoming 3 14
    strip_punctuation_keep_acronym: bool = True  # protect e.g. U.S. from becoming U S


@dataclass
class VisionConfig:
    """Optional local vision model (supports all llama-cpp-python vision models).

    We **do not** auto-download vision models. Put the two files below under
    `models_dir` and then enable vision.
    
    Supported models (automatically detected):
    - Moondream2, LLaVA-v1.6/1.5, BakLLaVA, MiniCPM-V, Qwen-VL, CogVLM, Yi-VL
    - Any other vision model supported by llama-cpp-python
    
    The app will automatically detect the model type from filename and select
    the appropriate ChatHandler. No manual configuration needed.
    """

    enabled: bool = False

    # Local filenames under models_dir
    text_model: str = "moondream2-text-model-f16.gguf"
    mmproj_model: str = "moondream2-mmproj-f16.gguf"

    # how many frames to sample inside a subtitle time span
    max_frames_per_sub: int = 1

    # where to sample frames within [start,end], 0..1
    frame_offsets: list[float] = field(default_factory=lambda: [0.5])


@dataclass
class AudioConfig:
    """Audio emotion recognition (Hugging Face Wav2Vec2).
    
    Default model: ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition (RAVDESS, 8 emotions).
    """
    enabled: bool = True
    device: str = "auto"  # "auto" | "cuda" | "cpu"
    device_index: int = 0  # GPU index when device is cuda (e.g. 0 = cuda:0)
    # Hugging Face model id (or local path); default RAVDESS emotion model
    model_id_or_path: str = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
    model_dir: str = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"  # backward compat
    # Cache directory for downloaded checkpoints (optional)
    cache_dir: str = "models/audio"
    # "full_then_slice" = extract full audio once then slice; "per_sentence" = extract per segment
    extract_mode: str = "full_then_slice"
    # Max workers for audio (1 to align with single-model principle)
    max_workers: int = 1
    # Batch size for emotion inference (larger = better GPU utilization, more VRAM)
    batch_size: int = 16
    # 若 True：音訊模型載入失敗時不中止流程，改輸出空 audio_tags 並在 log 顯示大紅警告
    allow_fail: bool = False


@dataclass
class GlossaryConfig:
    json_path: str = "./data/glossary.json"


# -----------------------------
# App config
# -----------------------------
@dataclass
class AppConfig:
    models_dir: str = "./models"

    # GGUF filenames placed under models_dir (legacy, still supported).
    # NOTE: for sharded GGUF, put the *first shard* filename here.
    qwen_model: str = "qwen2.5-14b-instruct-q5_k_m-00001-of-00003.gguf"
    breeze_model: str = "Breeze-7B-Instruct-v1_0.Q5_K_M.gguf"

    # Optional: smarter model layout by directories.
    #
    # Each directory is expected to contain exactly ONE logical model:
    # - If there is a single .gguf file => use that file.
    # - If there are multiple .gguf files => treat them as shards of the SAME model
    #   and automatically pick the correct shard-1 file (e.g. *-00001-of-00003.gguf),
    #   or the largest file as a fallback.
    #
    # This lets the user swap models just by dropping different GGUFs into
    # each folder, without editing filenames in config.json.
    #
    # If any of these are None or empty, the app falls back to the legacy
    # filename-based fields above.
    #
    # 預設直接使用三個「依工作劃分」的資料夾（命名：main / local / vision）：
    #   - ./models/main        → 主 / 推理模型 (Stage2, main model)
    #   - ./models/local       → 在地化 / 翻譯模型 (Stage3, localization model)
    #   - ./models/vision      → 視覺模型 (Moondream2 text + mmproj，都放在同一資料夾)
    #
    # 若單一資料夾裡有多個 .gguf，會被視為同一個模型的切分檔：
    #   - 會優先選擇 *-00001-of-XXX.gguf 作為入口分片
    #   - 找不到時退回選擇檔案大小最大的那一個
    # 內部欄位名稱仍沿用 reason/translate，但實際資料夾是 main/local。
    reason_dir: str | None = "./models/main"
    translate_dir: str | None = "./models/local"
    vision_text_dir: str | None = "./models/vision"
    vision_mmproj_dir: str | None = "./models/vision"

    # IMPORTANT: use default_factory for nested dataclasses
    llama_cpp: LlamaCppConfig = field(default_factory=LlamaCppConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    glossary: GlossaryConfig = field(default_factory=GlossaryConfig)


# -----------------------------
# Loader (config.json -> AppConfig)
# -----------------------------
def _coerce_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(v)


def _update_dataclass(obj: Any, data: dict[str, Any]) -> Any:
    """
    Recursively update a dataclass instance with values from dict.
    Unknown keys are ignored.
    """
    if not is_dataclass(obj) or not isinstance(data, dict):
        return obj

    name_to_field = {f.name: f for f in fields(obj)}

    for k, v in data.items():
        f = name_to_field.get(k)
        if f is None:
            continue

        cur = getattr(obj, k)

        # Nested dataclass
        if is_dataclass(cur) and isinstance(v, dict):
            _update_dataclass(cur, v)
            continue

        # Simple type adjustments
        if isinstance(cur, bool):
            setattr(obj, k, _coerce_bool(v))
            continue

        # List fields (e.g. frame_offsets)
        if isinstance(cur, list) and isinstance(v, list):
            setattr(obj, k, v)
            continue

        # Otherwise just assign
        setattr(obj, k, v)

    return obj


def load_config(path: Path) -> AppConfig:
    """
    Load config.json, overlay onto defaults, return AppConfig.

    Supports partial config.json (only fields you want to override).
    """
    cfg = AppConfig()

    if not path.exists():
        return cfg

    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return cfg

    data = json.loads(raw)
    if not isinstance(data, dict):
        return cfg

    _update_dataclass(cfg, data)
    # Backward compat: if config.json has audio.model_dir but no model_id_or_path, use model_dir
    if isinstance(data.get("audio"), dict):
        ad = data["audio"]
        if "model_dir" in ad and "model_id_or_path" not in ad:
            cfg.audio.model_id_or_path = cfg.audio.model_dir
    return cfg
