"""
Run A: Audio emotion recognition via Hugging Face Wav2Vec2.

Uses ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition (Transformers).
No local model files; model is loaded from Hugging Face Hub.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import os
import shutil
import tempfile
import subprocess


# Default HF model for speech emotion recognition (RAVDESS, 8 emotions)
DEFAULT_AUDIO_MODEL_ID = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"


def _load_checkpoint_state_dict(model_id: str) -> dict[str, Any] | None:
    """
    Load full state_dict from Hub or local path (single file: pytorch_model.bin or model.safetensors).
    Returns None if not found or sharded checkpoint.
    """
    try:
        import torch  # type: ignore
    except Exception:
        return None
    p = Path(model_id)
    if p.is_dir():
        bin_path = p / "pytorch_model.bin"
        safe_path = p / "model.safetensors"
        if safe_path.exists():
            try:
                from safetensors.torch import load_file as safe_load_file  # type: ignore
                return dict(safe_load_file(str(safe_path)))
            except Exception:
                pass
        if bin_path.exists():
            try:
                return dict(torch.load(str(bin_path), map_location="cpu", weights_only=True))
            except Exception:
                pass
        return None
    try:
        from transformers.utils import cached_file  # type: ignore
        for name in ("model.safetensors", "pytorch_model.bin"):
            try:
                path = cached_file(model_id, name)
                if path and Path(path).exists():
                    if name == "model.safetensors":
                        from safetensors.torch import load_file as safe_load_file  # type: ignore
                        return dict(safe_load_file(path))
                    return dict(torch.load(path, map_location="cpu", weights_only=True))
            except Exception:
                continue
    except Exception:
        pass
    return None


def _remap_legacy_classifier_head(state_dict: dict[str, Any]) -> dict[str, Any] | None:
    """
    Map old checkpoint keys (classifier.dense / classifier.output) to new
    (projector / classifier). Returns small state_dict for load_state_dict(..., strict=False).
    """
    if "classifier.dense.weight" not in state_dict or "classifier.output.weight" not in state_dict:
        return None
    out: dict[str, Any] = {}
    out["projector.weight"] = state_dict["classifier.dense.weight"]
    if "classifier.dense.bias" in state_dict:
        out["projector.bias"] = state_dict["classifier.dense.bias"]
    out["classifier.weight"] = state_dict["classifier.output.weight"]
    if "classifier.output.bias" in state_dict:
        out["classifier.bias"] = state_dict["classifier.output.bias"]
    return out


def _load_wav2vec2_emotion_model(model_id: str) -> tuple[Any, bool]:
    """
    Load Wav2Vec2ForSequenceClassification, fixing legacy checkpoint head if present
    (classifier.dense/output -> projector/classifier, and classifier_proj_size from checkpoint).
    Returns (model, use_legacy_fix: bool).
    """
    from transformers import AutoConfig, AutoModelForAudioClassification  # type: ignore
    from transformers import Wav2Vec2ForSequenceClassification  # type: ignore

    raw = _load_checkpoint_state_dict(model_id)
    if raw and "classifier.dense.weight" in raw:
        # Old checkpoint: dense is (proj_size, hidden_size); set config to match.
        proj_size = int(raw["classifier.dense.weight"].shape[0])
        config = AutoConfig.from_pretrained(model_id)
        config.classifier_proj_size = proj_size
        model = Wav2Vec2ForSequenceClassification(config)
        # Full remap: head keys + all keys the model expects (skip old pos_conv_embed etc.)
        model_keys = set(model.state_dict().keys())
        remapped: dict[str, Any] = {}
        for k, v in raw.items():
            if k.startswith("classifier.dense."):
                remapped["projector." + k[len("classifier.dense.") :]] = v
            elif k.startswith("classifier.output."):
                remapped["classifier." + k[len("classifier.output.") :]] = v
            elif k in model_keys:
                remapped[k] = v
        model.load_state_dict(remapped, strict=False)
        return model, True
    model = AutoModelForAudioClassification.from_pretrained(model_id)
    return model, False


def get_audio_stack_versions() -> dict[str, str]:
    """Return dict with python, transformers, torch versions for diagnostics."""
    import sys
    import platform
    out: dict[str, str] = {
        "python": f"{platform.python_version()} ({sys.executable})",
        "transformers": "not installed",
        "torch": "not installed",
    }
    try:
        import transformers as _tf  # type: ignore
        out["transformers"] = getattr(_tf, "__version__", "unknown")
    except Exception:
        pass
    try:
        import torch as _t  # type: ignore
        out["torch"] = getattr(_t, "__version__", "unknown")
    except Exception:
        pass
    return out


@dataclass
class AudioLoadResult:
    """Result of attempting to load the audio model."""
    success: bool
    reason: str
    backend: str  # "cpu" | "cuda"
    model: Any  # AudioModel | None


def _resolve_device(cfg_audio: Any) -> tuple[str, str]:
    """Resolve device from config. Returns (device, reason_or_empty)."""
    from .gpu_torch_diagnostics import has_nvidia_gpu

    device_setting = getattr(cfg_audio, "device", "auto") or "auto"
    try:
        import torch  # type: ignore
        cuda_available = torch.cuda.is_available()
    except Exception:
        cuda_available = False

    if device_setting == "cpu":
        return "cpu", ""
    if device_setting == "cuda":
        if cuda_available:
            return "cuda", ""
        reason = "device='cuda' requested but torch.cuda.is_available()=False."
        if has_nvidia_gpu():
            reason += " Run scripts/install_torch_cuda_windows.bat to install CUDA PyTorch."
        return "cpu", reason
    if cuda_available:
        return "cuda", ""
    if has_nvidia_gpu():
        return "cpu", "NVIDIA GPU detected but CUDA not available; using CPU. Install CUDA PyTorch for GPU."
    return "cpu", ""


def load_audio_model(cfg: Any) -> AudioLoadResult:
    """
    Load Hugging Face audio-classification model from config.
    model_id_or_path: HF model id (e.g. ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition) or path.
    If audio.cache_dir exists and contains config.json (e.g. models/audio from start.bat download), use it.
    """
    cfg_audio = getattr(cfg, "audio", None)
    if cfg_audio is None:
        return AudioLoadResult(success=False, reason="Config has no audio section", backend="cpu", model=None)

    model_id = getattr(cfg_audio, "model_id_or_path", None) or getattr(cfg_audio, "model_dir", None)
    if not model_id:
        return AudioLoadResult(
            success=False,
            reason="audio.model_id_or_path / model_dir not set",
            backend="cpu",
            model=None,
        )

    # Prefer local cache (e.g. models/audio) when start.bat has already downloaded the model
    cache_dir = getattr(cfg_audio, "cache_dir", None)
    if cache_dir:
        cache_path = Path(cache_dir)
        if cache_path.exists() and (cache_path / "config.json").exists():
            model_id = str(cache_path.resolve())

    device, device_reason = _resolve_device(cfg_audio)
    if device_setting := (getattr(cfg_audio, "device", "auto") or "auto"):
        if device_setting == "cuda" and device == "cpu" and device_reason:
            return AudioLoadResult(success=False, reason=device_reason, backend="cpu", model=None)

    device_index = int(getattr(cfg_audio, "device_index", 0))
    try:
        model = AudioModel(model_id=model_id, device=device, device_index=device_index)
        reason = f"loaded on {device}"
        if device == "cuda":
            try:
                import torch  # type: ignore
                name = torch.cuda.get_device_name(device_index)
                if name:
                    reason = f"loaded on cuda:{device_index} ({name})"
            except Exception:
                reason = f"loaded on cuda:{device_index}"
        if device_reason:
            reason = f"{reason} (warning: {device_reason})"
        return AudioLoadResult(success=True, reason=reason, backend=device, model=model)
    except Exception as e:
        reason = str(e).strip() or f"{type(e).__name__}: {repr(e)}"
        if getattr(e, "__cause__", None) is not None:
            c = e.__cause__
            reason = reason + f"\nInner cause: {type(c).__name__}: {c}"
        return AudioLoadResult(success=False, reason=reason, backend="cpu", model=None)


def try_load_audio_model(model_id: str) -> AudioLoadResult:
    """Try to load the audio model by model_id. Prefer load_audio_model(cfg) when config exists."""
    try:
        device = "cpu"
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                device = "cuda"
        except Exception:
            pass
        model = AudioModel(model_id=model_id, device=device, device_index=0)
        return AudioLoadResult(success=True, reason=f"loaded on {device}", backend=device, model=model)
    except Exception as e:
        return AudioLoadResult(
            success=False,
            reason=str(e).strip() or repr(e),
            backend="cpu",
            model=None,
        )


def _infer_tone(emotion: str) -> str:
    """Map emotion to tone for pipeline."""
    m = {
        "angry": "harsh", "calm": "neutral", "disgust": "sarcastic", "fearful": "tense",
        "happy": "cheerful", "neutral": "neutral", "sad": "melancholic", "surprised": "excited",
    }
    return m.get(emotion.lower(), "")


def _infer_intensity(confidence: float) -> str:
    if confidence >= 0.8:
        return "high"
    if confidence >= 0.5:
        return "medium"
    return "low"


def _infer_speaking_style(emotion: str) -> str:
    if emotion.lower() in ("angry", "happy", "surprised"):
        return "expressive"
    if emotion.lower() in ("calm", "neutral", "sad"):
        return "steady"
    return ""


def raw_emotion_to_meta(raw: dict[str, Any]) -> dict[str, Any]:
    """
    Convert raw emotion result (from analyze_emotion / analyze_emotion_batch) to meta dict
    with emotion, tone, intensity, speaking_style, audio_reason (same as analyze_audio_for_item).
    """
    err = raw.get("error")
    if err:
        return {
            "emotion": "", "tone": "", "intensity": "", "speaking_style": "",
            "audio_reason": f"inference error: {err}",
        }
    emotion = (raw.get("emotion") or "").strip()
    confidence = float(raw.get("confidence") or 0.0)
    tone = _infer_tone(emotion)
    intensity = _infer_intensity(confidence)
    speaking_style = _infer_speaking_style(emotion)
    reason = f"Emotion: {emotion} (confidence: {confidence:.2f})"
    if not any([emotion, tone, intensity, speaking_style]):
        reason = "inference returned empty"
    return {
        "emotion": emotion or "neutral",
        "tone": tone,
        "intensity": intensity,
        "speaking_style": speaking_style,
        "audio_reason": reason,
    }


def analyze_audio_for_item(model: Any, waveform: Any, sr: int) -> dict[str, Any]:
    """
    Run inference for one item. Returns emotion, tone, intensity, speaking_style, audio_reason.
    waveform: file path (str) or tensor; sr used when waveform is tensor.
    """
    if model is None:
        return {
            "emotion": "", "tone": "", "intensity": "", "speaking_style": "",
            "audio_reason": "model is None",
        }
    if isinstance(waveform, str):
        raw = model.analyze_emotion(waveform)
    else:
        raw = model.analyze_emotion_from_waveform(waveform, sr)
    emotion = (raw.get("emotion") or "").strip()
    confidence = float(raw.get("confidence") or 0.0)
    err = raw.get("error")
    if err:
        return {
            "emotion": "", "tone": "", "intensity": "", "speaking_style": "",
            "audio_reason": f"inference error: {err}",
        }
    tone = _infer_tone(emotion)
    intensity = _infer_intensity(confidence)
    speaking_style = _infer_speaking_style(emotion)
    reason = f"Emotion: {emotion} (confidence: {confidence:.2f})"
    if not any([emotion, tone, intensity, speaking_style]):
        reason = "inference returned empty"
    return {
        "emotion": emotion or "neutral",
        "tone": tone,
        "intensity": intensity,
        "speaking_style": speaking_style,
        "audio_reason": reason,
    }


def run_audio_env_check(model_id_or_path: str | Path, log_lines: list[str]) -> None:
    """Run environment self-check at Run A start. Logs transformers, torch, ffmpeg."""
    from .gpu_torch_diagnostics import get_torch_diag, format_torch_diag, build_fix_instructions, get_why_cpu_only

    log_lines.append("[Run A] -------- Environment self-check --------")
    vers = get_audio_stack_versions()
    log_lines.append(f"[Run A] python: {vers['python']}")
    log_lines.append(f"[Run A] transformers: {vers['transformers']}, torch: {vers['torch']}")
    log_lines.append(f"[Run A] audio model: {model_id_or_path}")

    diag = get_torch_diag()
    for line in format_torch_diag(diag).split("\n"):
        log_lines.append(f"[Run A] {line}")
    if diag.get("has_nvidia_gpu") and not diag.get("cuda_is_available"):
        why = get_why_cpu_only(diag)
        if why:
            log_lines.append(f"[Run A] Why still CPU only: {why}")
    fix = build_fix_instructions(diag)
    if fix:
        for line in fix.split("\n"):
            log_lines.append(f"[Run A] {line}")

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        log_lines.append(f"[Run A] ffmpeg: found ({ffmpeg_path})")
    else:
        log_lines.append("[Run A] ffmpeg: not found in PATH")
    log_lines.append("[Run A] -------- End environment self-check --------")


class AudioModel:
    """Hugging Face Wav2Vec2 speech emotion recognition (audio-classification pipeline)."""

    def __init__(self, model_id: str, device: str = "cpu", device_index: int = 0):
        """
        Load HF audio-classification pipeline.
        model_id: e.g. ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition
        device: "cpu" | "cuda"
        device_index: GPU index when device=="cuda" (e.g. 0 for cuda:0)
        """
        from transformers import pipeline  # type: ignore
        from transformers import AutoFeatureExtractor  # type: ignore

        self.model_id = model_id
        self._device = device
        self._device_index = device_index
        pipe_device = device_index if device == "cuda" else -1

        # Load model (with legacy head remap if checkpoint uses classifier.dense/output)
        model, _ = _load_wav2vec2_emotion_model(model_id)
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self._pipeline = pipeline(
            "audio-classification",
            model=model,
            feature_extractor=feature_extractor,
            device=pipe_device,
        )

    def _parse_pipeline_output(self, out: Any) -> dict[str, Any]:
        """Convert pipeline output (list of {label, score}) to emotion dict."""
        if not out:
            return {"emotion": "neutral", "confidence": 0.0, "probabilities": {}}
        probs = {item["label"]: float(item["score"]) for item in out}
        top = out[0]
        return {
            "emotion": top["label"],
            "confidence": float(top["score"]),
            "probabilities": probs,
        }

    def analyze_emotion(self, audio_path: str) -> dict[str, Any]:
        """
        Classify emotion from a WAV file.
        Returns dict with emotion, confidence, probabilities.
        """
        try:
            import torch  # type: ignore
            with torch.inference_mode():
                out = self._pipeline(audio_path)
            return self._parse_pipeline_output(out)
        except Exception as e:
            return {"emotion": "", "confidence": 0.0, "probabilities": {}, "error": str(e)}

    def analyze_emotion_batch(self, audio_paths: list[str]) -> list[dict[str, Any]]:
        """
        Classify emotion for multiple WAV files in one call (better GPU utilization).
        Returns list of dicts with emotion, confidence, probabilities (one per path).
        """
        if not audio_paths:
            return []
        try:
            import torch  # type: ignore
            with torch.inference_mode():
                out_list = self._pipeline(audio_paths)
            # pipeline returns list of lists: [[{label, score}, ...], ...]
            results: list[dict[str, Any]] = []
            for out in out_list:
                if isinstance(out, list):
                    results.append(self._parse_pipeline_output(out))
                else:
                    results.append(self._parse_pipeline_output([out] if out else []))
            return results
        except Exception as e:
            return [{"emotion": "", "confidence": 0.0, "probabilities": {}, "error": str(e)}] * len(audio_paths)

    def analyze_emotion_from_waveform(self, waveform: Any, sr: int) -> dict[str, Any]:
        """Run inference from waveform; write temp WAV then classify."""
        import numpy as np  # type: ignore

        target_sr = 16000
        if isinstance(waveform, str):
            return self.analyze_emotion(waveform)
        w = waveform
        if hasattr(w, "numpy"):
            w = np.asarray(w.numpy(), dtype=np.float32)
        elif hasattr(w, "cpu"):
            w = w.cpu().numpy().astype(np.float32)
        else:
            w = np.asarray(w, dtype=np.float32)
        if w.ndim == 1:
            w = w[np.newaxis, :]
        if sr != target_sr:
            from scipy import signal  # type: ignore
            num = int(len(w[0]) * target_sr / sr)
            w = signal.resample(w, num, axis=1).astype(np.float32)
        fd, path = tempfile.mkstemp(suffix=".wav")
        try:
            os.close(fd)
            import soundfile as sf  # type: ignore
            sf.write(path, w.T, target_sr)
            return self.analyze_emotion(path)
        finally:
            try:
                Path(path).unlink(missing_ok=True)
            except Exception:
                pass

    def extract_full_audio(
        self,
        video_path: str,
        output_path: str,
        progress_callback: Any | None = None,
    ) -> str:
        """Extract full video audio to WAV 16kHz mono."""
        if progress_callback:
            progress_callback(0.0, "Starting full audio extraction...")
        cmd = [
            "ffmpeg", "-i", video_path,
            "-ar", "16000", "-ac", "1", "-threads", "0", "-loglevel", "error", "-y", output_path,
        ]
        try:
            if progress_callback:
                progress_callback(0.5, "Extracting audio...")
            subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=600)
            if progress_callback:
                progress_callback(1.0, "Audio extraction complete")
            return output_path
        except subprocess.TimeoutExpired:
            raise RuntimeError("ffmpeg timeout (10min) extracting full audio.") from None
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to extract full audio: {e.stderr}") from e
        except FileNotFoundError:
            raise RuntimeError("ffmpeg not found. Install ffmpeg and add to PATH.") from None

    def extract_segment_from_audio(
        self,
        audio_path: str,
        start_ms: float,
        end_ms: float,
        output_path: str | None = None,
    ) -> str:
        """Slice segment from full audio file."""
        if output_path is None:
            output_path = str(Path(tempfile.gettempdir()) / f"audio_segment_{int(start_ms)}_{int(end_ms)}.wav")
        start_sec = start_ms / 1000.0
        duration = (end_ms - start_ms) / 1000.0
        if duration <= 0 or start_sec < 0:
            raise ValueError(f"Invalid range: start_ms={start_ms}, end_ms={end_ms}")
        cmd = [
            "ffmpeg", "-ss", str(start_sec), "-i", audio_path,
            "-t", str(duration), "-ar", "16000", "-ac", "1", "-threads", "0", "-loglevel", "error", "-y", output_path,
        ]
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=5)
            return output_path
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to extract segment: {e.stderr}") from e
        except FileNotFoundError:
            raise RuntimeError("ffmpeg not found. Install ffmpeg and add to PATH.") from None

    def extract_audio_segment(
        self,
        video_path: str,
        start_ms: float,
        end_ms: float,
        output_path: str | None = None,
    ) -> str:
        """Extract audio segment from video (WAV 16kHz mono)."""
        if output_path is None:
            output_path = str(Path(tempfile.gettempdir()) / f"audio_segment_{int(start_ms)}_{int(end_ms)}.wav")
        start_sec = start_ms / 1000.0
        duration = (end_ms - start_ms) / 1000.0
        if duration <= 0 or start_sec < 0:
            raise ValueError(f"Invalid range: start_ms={start_ms}, end_ms={end_ms}")
        cmd = [
            "ffmpeg", "-ss", str(start_sec), "-i", video_path,
            "-t", str(duration), "-ar", "16000", "-ac", "1", "-threads", "0", "-loglevel", "error", "-y", output_path,
        ]
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
            return output_path
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to extract segment: {e.stderr}") from e
        except FileNotFoundError:
            raise RuntimeError("ffmpeg not found. Install ffmpeg and add to PATH.") from None
