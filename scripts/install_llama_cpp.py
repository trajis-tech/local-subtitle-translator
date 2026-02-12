from __future__ import annotations

import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = ROOT / "runtime"
WHEELS_DIR = RUNTIME_DIR / "wheels"
WHEELS_DIR.mkdir(parents=True, exist_ok=True)

# Local build wheel directory (development build)
BUILD_WHEEL_DIR = ROOT / "build_wheel" / "dist"


def _run(cmd: list[str]) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, check=False)
        out = (p.stdout or "") + (p.stderr or "")
        return p.returncode, out.strip()
    except FileNotFoundError:
        return 127, ""


def _pip(args: list[str]) -> int:
    cmd = [sys.executable, "-m", "pip"] + args
    print("\n[CMD]", " ".join(cmd))
    return subprocess.call(cmd)


def _parse_driver_version(v: str) -> tuple[int, int, int]:
    # e.g. "580.38" -> (580, 38, 0)
    parts = re.findall(r"\d+", v)
    nums = [int(x) for x in parts[:3]]
    while len(nums) < 3:
        nums.append(0)
    return nums[0], nums[1], nums[2]


def _detect_nvidia() -> dict:
    rc, out = _run([
        "nvidia-smi",
        "--query-gpu=name,driver_version",
        "--format=csv,noheader",
    ])
    if rc != 0 or not out:
        return {"has_nvidia": False}
    first = out.splitlines()[0]
    # format: "NVIDIA GeForce RTX 5070 Ti, 580.38"
    parts = [p.strip() for p in first.split(",")]
    name = parts[0] if parts else ""
    drv = parts[1] if len(parts) > 1 else ""
    return {"has_nvidia": True, "gpu_name": name, "driver_version": drv}


def _download(url: str, dest: Path) -> None:
    print(f"[INFO] Downloading: {url}")
    with urllib.request.urlopen(url) as r:
        data = r.read()
    dest.write_bytes(data)
    print(f"[INFO] Saved: {dest}")


def main() -> int:
    # Priority 1: Use locally built wheel if available (development build)
    # This wheel is compiled with AVX512 disabled and supports 10/20/30/40/50 series CUDA
    if BUILD_WHEEL_DIR.exists():
        local_wheels = list(BUILD_WHEEL_DIR.glob("llama_cpp_python-*.whl"))
        if local_wheels:
            # Use the first (or most recent) local wheel
            local_wheel = sorted(local_wheels, key=lambda p: p.stat().st_mtime, reverse=True)[0]
            print(f"[INFO] Found locally built wheel: {local_wheel.name}")
            print(f"[INFO] Using local wheel (AVX512 disabled, CUDA support for 10/20/30/40/50 series)")
            rc = _pip(["install", "--no-cache-dir", "--force-reinstall", str(local_wheel)])
            if rc == 0:
                print("[OK] Installed llama-cpp-python (local build wheel).")
                return 0
            print("[WARN] Local wheel install failed, falling back to online wheels...")
    
    # Priority 2: Try online wheels (fallback)
    # Always prefer binary-only installs to avoid Visual Studio build errors.
    # Try Blackwell wheel (RTX 50xx) first when possible.
    info = _detect_nvidia()
    if info.get("has_nvidia"):
        gpu = (info.get("gpu_name") or "").lower()
        drv = (info.get("driver_version") or "").strip()
        print(f"[INFO] Detected NVIDIA GPU: {info.get('gpu_name')} | Driver: {drv}")

        is_blackwell = any(x in gpu for x in [
            "rtx 5050", "rtx 5060", "rtx 5070", "rtx 5080", "rtx 5090",
            "blackwell",
        ])

        if is_blackwell:
            drv_tuple = _parse_driver_version(drv or "0")
            if drv_tuple[0] >= 580:
                # Prebuilt wheel from dougeeai (CUDA 13.0, sm100, py312, win_amd64)
                wheel_name = "llama_cpp_python-0.3.16+cuda13.0.sm100.blackwell-cp312-cp312-win_amd64.whl"
                url = (
                    "https://github.com/dougeeai/llama-cpp-python-wheels/releases/download/"
                    "v0.3.16-cuda13.0-sm100-py312/" + wheel_name
                )
                wheel_path = WHEELS_DIR / wheel_name
                if not wheel_path.exists():
                    try:
                        _download(url, wheel_path)
                    except Exception as e:
                        print(f"[WARN] Failed to download Blackwell wheel: {e}")
                if wheel_path.exists():
                    rc = _pip(["install", "--no-cache-dir", "--force-reinstall", str(wheel_path)])
                    if rc == 0:
                        print("[OK] Installed llama-cpp-python (Blackwell CUDA wheel).")
                        return 0
                    print("[WARN] Blackwell wheel install failed, falling back...")
            else:
                print("[WARN] Blackwell GPU detected but driver < 580.")
                print("       For GPU acceleration on RTX 50xx, update NVIDIA driver to 580+.")
                print("       Falling back to CPU wheel for now.")

        # Generic CUDA wheels (CUDA 12.4) from official abetlen index.
        # This often works for RTX 20/30/40 series; may not for Blackwell.
        print("[INFO] Trying CUDA wheel (cu124) from abetlen index...")
        rc = _pip([
            "install",
            "--no-cache-dir",
            "--only-binary=:all:",
            "--extra-index-url",
            "https://abetlen.github.io/llama-cpp-python/whl/cu124",
            "llama-cpp-python",
        ])
        if rc == 0:
            print("[OK] Installed llama-cpp-python (CUDA wheel).")
            return 0
        print("[WARN] CUDA wheel install failed, falling back to CPU wheel...")

    else:
        print("[INFO] NVIDIA GPU not detected (or nvidia-smi not found). Using CPU wheel.")

    # CPU-only wheel from PyPI (should not require build tools)
    rc = _pip([
        "install",
        "--no-cache-dir",
        "--only-binary=:all:",
        "--force-reinstall",
        "llama-cpp-python",
    ])
    if rc == 0:
        print("[OK] Installed llama-cpp-python (CPU wheel).")
        return 0

    print("[ERROR] Could not install llama-cpp-python.")
    print("        If you are on Windows and pip is trying to compile from source, you may need")
    print("        Visual Studio Build Tools, OR use a prebuilt CUDA wheel.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
