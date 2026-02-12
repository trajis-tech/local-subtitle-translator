"""
資源檢測和優化工具：自動檢測 CPU、記憶體、GPU 資源，並提供動態調整機制。

此模組提供：
1. GPU/VRAM 檢測（優先）
2. CPU 核心數檢測
3. 可用記憶體檢測
4. 動態批次大小計算（基於 GPU VRAM 優先）
5. 並行度建議
"""

from __future__ import annotations

# VRAM 使用門檻：載入與推理時允許用到 99%，OOM 時由 chat_dispatch → safe_chat + NeedReload 分級回退
VRAM_UTILIZATION_LIMIT = 0.99
import os
import sys
import platform
from typing import Optional, Any


def detect_gpu() -> dict[str, Any]:
    """
    檢測 GPU 可用性和 VRAM 資訊。
    
    Returns:
        包含 GPU 資訊的字典：
        {
            "available": bool,  # GPU 是否可用
            "vram_mb": Optional[int],  # VRAM 大小（MB），如果不可用則為 None
            "device_name": Optional[str],  # GPU 設備名稱
            "driver_version": Optional[str],  # 驅動版本
        }
    """
    gpu_info = {
        "available": False,
        "vram_mb": None,
        "device_name": None,
        "driver_version": None,
    }
    
    # 優先檢測 NVIDIA GPU（最常見）
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            # 解析 nvidia-smi 輸出
            lines = result.stdout.strip().splitlines()
            if lines:
                # 取第一個 GPU 的資訊
                parts = lines[0].split(", ")
                if len(parts) >= 2:
                    gpu_info["device_name"] = parts[0].strip()
                    try:
                        vram_mb = int(parts[1].strip())
                        gpu_info["vram_mb"] = vram_mb
                        gpu_info["available"] = True
                    except ValueError:
                        pass
                    if len(parts) >= 3:
                        gpu_info["driver_version"] = parts[2].strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        # nvidia-smi 不可用或執行失敗
        pass
    
    # 備選：嘗試使用 PyTorch 檢測（如果已安裝）
    if not gpu_info["available"]:
        try:
            import torch
            if torch.cuda.is_available():
                gpu_info["available"] = True
                gpu_info["vram_mb"] = int(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024))
                gpu_info["device_name"] = torch.cuda.get_device_name(0)
        except ImportError:
            pass
    
    return gpu_info


def recommend_llama_params_from_vram(
    vram_mb: int,
    model_kind: str = "reason",
) -> dict[str, Any]:
    """
    僅做最基本 sanity：回傳「模型最大允許」的 n_gpu_layers / n_ctx，不依 VRAM 保守砍參。
    實際 OOM 時由 chat_dispatch（heavy 隔離 → safe_chat）+ NeedReload 分級回退（L1 降 max_tokens → L3/L4/L5 重載降 n_ctx/n_gpu_layers）。
    """
    if model_kind == "reason":
        return {"n_gpu_layers": -1, "n_ctx": 8192}
    return {"n_gpu_layers": -1, "n_ctx": 4096}


def get_cpu_count() -> int:
    """
    獲取 CPU 核心數（邏輯核心）。
    
    Returns:
        CPU 核心數，如果檢測失敗則返回 4（保守預設值）
    """
    try:
        # 優先使用 os.cpu_count()（Python 3.4+）
        count = os.cpu_count()
        if count and count > 0:
            return count
    except Exception:
        pass
    
    # 備選方案：使用 multiprocessing
    try:
        import multiprocessing
        count = multiprocessing.cpu_count()
        if count and count > 0:
            return count
    except Exception:
        pass
    
    # 保守預設值
    return 4


def get_available_memory_mb() -> Optional[int]:
    """
    獲取可用記憶體（MB）。
    
    Returns:
        可用記憶體（MB），如果檢測失敗則返回 None
    """
    try:
        if platform.system() == "Windows":
            # Windows: 使用 psutil（如果可用）或 wmic
            try:
                import psutil
                mem = psutil.virtual_memory()
                return int(mem.available / (1024 * 1024))  # 轉換為 MB
            except ImportError:
                # 備選：使用 wmic（Windows）
                import subprocess
                result = subprocess.run(
                    ["wmic", "OS", "get", "FreePhysicalMemory", "/value"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    for line in result.stdout.splitlines():
                        if line.startswith("FreePhysicalMemory="):
                            kb = int(line.split("=")[1])
                            return kb // 1024  # KB 轉 MB
        else:
            # Linux/macOS: 使用 psutil 或 /proc/meminfo
            try:
                import psutil
                mem = psutil.virtual_memory()
                return int(mem.available / (1024 * 1024))
            except ImportError:
                # 備選：讀取 /proc/meminfo（Linux）
                if os.path.exists("/proc/meminfo"):
                    with open("/proc/meminfo", "r") as f:
                        for line in f:
                            if line.startswith("MemAvailable:"):
                                kb = int(line.split()[1])
                                return kb // 1024
    except Exception:
        pass
    
    return None


def calculate_batch_size(
    total_items: int,
    available_memory_mb: Optional[int] = None,
    cpu_count: Optional[int] = None,
    gpu_info: Optional[dict[str, Any]] = None,
    base_batch_size: int = 32,
    min_batch_size: int = 8,
    max_batch_size: int = 128,
    memory_per_item_mb: float = 0.5,
) -> int:
    """
    根據可用資源動態計算批次大小。
    
    優先順序：GPU VRAM > 系統記憶體 > CPU 核心數
    
    Args:
        total_items: 總項目數
        available_memory_mb: 可用記憶體（MB），如果為 None 則自動檢測
        cpu_count: CPU 核心數，如果為 None 則自動檢測
        gpu_info: GPU 資訊字典（從 detect_gpu() 獲取），如果為 None 則自動檢測
        base_batch_size: 基礎批次大小（預設：32）
        min_batch_size: 最小批次大小（預設：8，確保低資源環境也能運作）
        max_batch_size: 最大批次大小（預設：128）
        memory_per_item_mb: 每個項目預估記憶體佔用（MB，預設：0.5）
    
    Returns:
        計算出的批次大小
    """
    if gpu_info is None:
        gpu_info = detect_gpu()
    
    if cpu_count is None:
        cpu_count = get_cpu_count()
    
    if available_memory_mb is None:
        available_memory_mb = get_available_memory_mb()
    
    # 優先使用 GPU VRAM（如果可用）
    if gpu_info.get("available") and gpu_info.get("vram_mb"):
        vram_mb = gpu_info["vram_mb"]
        # 使用 VRAM 門檻 99%；OOM 時由 chat_dispatch（heavy 隔離）+ safe_chat + NeedReload 回退
        usable_vram_mb = vram_mb * VRAM_UTILIZATION_LIMIT
        # GPU 模式下，每個項目的記憶體佔用較小（因為模型在 GPU 上）
        # 激進模式：降低每個項目的記憶體預估（從 0.3 降到 0.2），允許更大的批次
        gpu_batch = int(usable_vram_mb / (memory_per_item_mb * 0.2))  # GPU 模式下記憶體佔用更小
        # 確保至少使用 base_batch_size（如果計算出的值更小）
        gpu_batch = max(base_batch_size, gpu_batch)
        gpu_batch = max(min_batch_size, min(gpu_batch, max_batch_size, total_items))
        return gpu_batch
    
    # GPU 不可用時，使用 CPU + RAM
    # 基礎批次大小（基於 CPU 核心數）
    # 每個核心處理一個批次，但不要超過總項目數
    cpu_based_batch = min(base_batch_size * max(1, cpu_count // 4), total_items)
    
    # 記憶體限制的批次大小
    memory_based_batch = base_batch_size
    if available_memory_mb:
        # 保留 50% 記憶體給系統和其他進程
        usable_memory_mb = available_memory_mb * 0.5
        memory_based_batch = int(usable_memory_mb / memory_per_item_mb)
        memory_based_batch = max(min_batch_size, min(memory_based_batch, max_batch_size))
    
    # 取兩者的最小值（確保不超過記憶體限制）
    batch_size = min(cpu_based_batch, memory_based_batch)
    
    # 確保在合理範圍內
    batch_size = max(min_batch_size, min(batch_size, max_batch_size, total_items))
    
    return batch_size


def calculate_parallel_workers(
    cpu_count: Optional[int] = None,
    max_workers: Optional[int] = None,
    min_workers: int = 1,
) -> int:
    """
    計算並行工作進程數。
    
    Args:
        cpu_count: CPU 核心數，如果為 None 則自動檢測
        max_workers: 最大工作進程數，如果為 None 則使用 CPU 核心數
        min_workers: 最小工作進程數（預設：1）
    
    Returns:
        建議的並行工作進程數
    """
    if cpu_count is None:
        cpu_count = get_cpu_count()
    
    if max_workers is None:
        # 預設：使用 CPU 核心數，但保留一個核心給系統
        max_workers = max(1, cpu_count - 1)
    
    # 確保在合理範圍內
    workers = max(min_workers, min(max_workers, cpu_count))
    
    return workers


def get_resource_info() -> dict[str, Any]:
    """
    獲取系統資源資訊（優先檢測 GPU）。
    
    Returns:
        包含 GPU、CPU 核心數、可用記憶體等資訊的字典
    """
    gpu_info = detect_gpu()
    cpu_count = get_cpu_count()
    available_memory_mb = get_available_memory_mb()
    
    return {
        "gpu_available": gpu_info["available"],
        "gpu_vram_mb": gpu_info["vram_mb"],
        "gpu_device_name": gpu_info["device_name"],
        "gpu_driver_version": gpu_info["driver_version"],
        "cpu_count": cpu_count,
        "available_memory_mb": available_memory_mb,
        "platform": platform.system(),
        "python_version": sys.version.split()[0],
    }
