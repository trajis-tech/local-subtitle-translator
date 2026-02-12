"""
GPU / PyTorch 診斷工具：檢測 CUDA 可用性並提供修復指引。

提供：
- get_torch_diag() -> dict：取得 PyTorch/CUDA 診斷資訊
- format_torch_diag(diag) -> str：格式化為可讀 log
- build_fix_instructions(diag) -> str：產生安裝 CUDA 版 torch 的指令
- has_nvidia_gpu() -> bool：檢測是否有 NVIDIA GPU（不依賴 torch）
"""

from __future__ import annotations
import platform
import shutil
import subprocess
from typing import Any


def has_nvidia_gpu() -> bool:
    """
    檢測是否有 NVIDIA GPU（不依賴 torch，用 nvidia-smi）。
    Windows/Linux 皆可。
    """
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return False
    try:
        result = subprocess.run(
            [nvidia_smi, "-L"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # nvidia-smi -L 會列出 GPU，例如 "GPU 0: NVIDIA GeForce RTX 3090 (...)"
        return result.returncode == 0 and "GPU" in result.stdout
    except Exception:
        return False


def get_nvidia_gpu_name() -> str:
    """
    取得 NVIDIA GPU 名稱（不依賴 torch）。
    """
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return ""
    try:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip().split("\n")[0]
    except Exception:
        pass
    return ""


def get_torch_diag() -> dict[str, Any]:
    """
    取得 PyTorch/CUDA 診斷資訊。

    回傳 dict 包含：
    - torch_available: bool（是否能 import torch）
    - torch_version: str
    - torch_cuda_version: str | None（torch.version.cuda）
    - is_cpu_build: bool（torch.version.cuda is None）
    - cuda_is_available: bool（torch.cuda.is_available()）
    - device_count: int
    - device_names: list[str]（所有 GPU 名稱）
    - has_nvidia_gpu: bool（nvidia-smi 偵測）
    - nvidia_gpu_name: str（nvidia-smi 取得）
    - platform: str
    - error: str（若 import 失敗）
    """
    diag: dict[str, Any] = {
        "torch_available": False,
        "torch_version": "",
        "torch_cuda_version": None,
        "is_cpu_build": True,
        "cuda_is_available": False,
        "device_count": 0,
        "device_names": [],
        "has_nvidia_gpu": has_nvidia_gpu(),
        "nvidia_gpu_name": get_nvidia_gpu_name(),
        "platform": platform.system(),
        "error": "",
    }

    try:
        import torch  # type: ignore
        diag["torch_available"] = True
        diag["torch_version"] = getattr(torch, "__version__", "unknown")
        diag["torch_cuda_version"] = getattr(torch.version, "cuda", None)
        diag["is_cpu_build"] = diag["torch_cuda_version"] is None
        diag["cuda_is_available"] = torch.cuda.is_available()
        if diag["cuda_is_available"]:
            diag["device_count"] = torch.cuda.device_count()
            diag["device_names"] = [
                torch.cuda.get_device_name(i) for i in range(diag["device_count"])
            ]
    except Exception as e:
        diag["error"] = f"{type(e).__name__}: {e}"

    return diag


def format_torch_diag(diag: dict[str, Any]) -> str:
    """
    格式化診斷資訊為可讀的多行 log。
    """
    lines = []
    lines.append("[PyTorch/CUDA Diagnostics]")
    lines.append(f"  Platform: {diag.get('platform', 'unknown')}")

    if not diag.get("torch_available"):
        lines.append(f"  torch: import FAILED ({diag.get('error', 'unknown')})")
        return "\n".join(lines)

    lines.append(f"  torch version: {diag.get('torch_version', 'unknown')}")
    cuda_ver = diag.get("torch_cuda_version")
    if cuda_ver:
        lines.append(f"  torch.version.cuda: {cuda_ver}")
    else:
        lines.append("  torch.version.cuda: None (CPU-only build)")

    lines.append(f"  torch.cuda.is_available(): {diag.get('cuda_is_available', False)}")

    if diag.get("cuda_is_available"):
        count = diag.get("device_count", 0)
        lines.append(f"  CUDA device count: {count}")
        for i, name in enumerate(diag.get("device_names", [])):
            lines.append(f"    GPU {i}: {name}")
    else:
        if diag.get("is_cpu_build"):
            lines.append("  ⚠️ This is a CPU-only PyTorch build (no CUDA support)")
        else:
            lines.append("  ⚠️ CUDA build detected but torch.cuda.is_available() is False")
            lines.append("     Possible causes: CUDA driver not installed, driver version mismatch")

    if diag.get("has_nvidia_gpu"):
        lines.append(f"  NVIDIA GPU detected (nvidia-smi): {diag.get('nvidia_gpu_name', 'yes')}")
        if not diag.get("cuda_is_available"):
            lines.append("  ❌ You have a CPU-only PyTorch build. Run scripts\\install_torch_cuda_windows.bat to install CUDA build.")
    else:
        lines.append("  No NVIDIA GPU detected (nvidia-smi not available or no GPU)")

    return "\n".join(lines)


def get_why_cpu_only(diag: dict[str, Any]) -> str:
    """
    回傳「為何目前是 CPU only」的一行說明，方便除錯。
    若 CUDA 可用或無需說明則回傳空字串。
    """
    if diag.get("cuda_is_available"):
        return ""
    if not diag.get("torch_available"):
        return f"torch import failed: {diag.get('error', 'unknown')}"
    cuda_ver = diag.get("torch_cuda_version")
    if cuda_ver is None:
        return (
            "PyTorch is **CPU-only** (torch.version.cuda is None). "
            "Current pip torch has no CUDA support; reinstall using CUDA-specific index."
        )
    return (
            "PyTorch is CUDA build (torch.version.cuda=" + str(cuda_ver) + ") "
            "but torch.cuda.is_available()=False. "
            "Common causes: outdated NVIDIA driver, driver-CUDA version mismatch, or CUDA driver not installed."
        )


def build_fix_instructions(diag: dict[str, Any]) -> str:
    """
    若偵測到 NVIDIA GPU 但 CUDA 不可用，回傳安裝 CUDA 版 torch 的指令。
    若 CUDA 可用或無 NVIDIA GPU，回傳空字串。
    """
    if diag.get("cuda_is_available"):
        return ""  # 已可用，不需修復
    if not diag.get("has_nvidia_gpu"):
        return ""  # 無 NVIDIA GPU，不需 CUDA

    is_windows = diag.get("platform") == "Windows"

    lines = []
    lines.append("")
    lines.append("=" * 70)
    lines.append("NVIDIA GPU detected but PyTorch CUDA is not available.")
    lines.append("Install CUDA-enabled PyTorch to enable GPU acceleration.")
    lines.append("=" * 70)
    lines.append("")

    why = get_why_cpu_only(diag)
    if why:
        lines.append("[Why CPU-only]")
        lines.append("  " + why)
        lines.append("")
    if is_windows:
        lines.append("[Method 1] Run project script (recommended):")
        lines.append("  scripts\\install_torch_cuda_windows.bat")
        lines.append("  Use the same Python as this app (if using .venv, run from project root; script auto-uses .venv\\Scripts\\python.exe)")
        lines.append("")
        lines.append("[Method 2] Manual commands (Stable CUDA 12.8):")
    else:
        lines.append("[Manual install] (Stable CUDA 12.8):")

    lines.append("  pip uninstall -y torch")
    lines.append("  pip install torch --index-url https://download.pytorch.org/whl/cu128")
    lines.append("")
    lines.append("If cu128 is incompatible, try cu126 or cu118:")
    lines.append("  pip install torch --index-url https://download.pytorch.org/whl/cu126")
    lines.append("  pip install torch --index-url https://download.pytorch.org/whl/cu118")
    lines.append("")
    lines.append("Verify after install:")
    if is_windows:
        lines.append("  scripts\\verify_torch_cuda.bat")
        lines.append("  或：python -c \"import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')\"")
    else:
        lines.append("  python -c \"import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')\"")
    lines.append("=" * 70)

    return "\n".join(lines)


def resolve_device(requested: str, allow_fallback: bool = True) -> str:
    """
    解析 device 設定。

    Args:
        requested: "auto" | "cuda" | "cpu"
        allow_fallback: 若為 False 且 requested="cuda" 但不可用，raise RuntimeError

    Returns:
        "cuda" 或 "cpu"
    """
    try:
        import torch  # type: ignore
        cuda_available = torch.cuda.is_available()
    except Exception:
        cuda_available = False

    if requested == "cpu":
        return "cpu"

    if requested == "cuda":
        if cuda_available:
            return "cuda"
        if not allow_fallback:
            diag = get_torch_diag()
            fix = build_fix_instructions(diag)
            msg = (
                "device='cuda' requested but torch.cuda.is_available() is False.\n"
                "Ensure CUDA-enabled PyTorch is installed and NVIDIA driver is configured correctly.\n"
            )
            if fix:
                msg += fix
            else:
                msg += "Run scripts/install_torch_cuda_windows.bat or manually install CUDA-enabled torch."
            raise RuntimeError(msg)
        return "cpu"

    # auto
    return "cuda" if cuda_available else "cpu"
