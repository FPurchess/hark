"""Compute device detection for Whisper inference."""

from __future__ import annotations

import subprocess
from typing import TypedDict

__all__ = [
    "DeviceInfo",
    "check_vulkan_support",
    "check_cuda_support",
    "check_pytorch_vulkan",
    "detect_best_device",
    "get_compute_type",
    "get_device_info",
]


class DeviceInfo(TypedDict):
    """Information about detected compute devices."""

    vulkan_hardware: bool
    vulkan_pytorch: bool
    cuda_available: bool
    gpu_name: str | None
    compute_capability: str | None
    recommended_device: str


def check_vulkan_support() -> bool:
    """
    Check if Vulkan is available on the system.

    Returns:
        True if Vulkan support is detected.
    """
    # Check vulkaninfo
    try:
        result = subprocess.run(
            ["vulkaninfo", "--summary"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and "Vulkan Instance" in result.stdout:
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    # Alternative check with vkcube
    try:
        result = subprocess.run(
            ["vkcube", "--help"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    # Check for Vulkan loader library
    try:
        result = subprocess.run(
            ["ldconfig", "-p"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if "libvulkan" in result.stdout:
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    return False


def check_cuda_support() -> tuple[bool, str | None, tuple[int, int] | None]:
    """
    Check if CUDA is available and compatible.

    Returns:
        Tuple of (available, gpu_name, compute_capability).
    """
    try:
        import torch  # pyrefly: ignore[missing-import]

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            compute_cap = torch.cuda.get_device_capability(0)
            return True, gpu_name, compute_cap
    except ImportError:
        pass
    except Exception:
        pass

    return False, None, None


def check_pytorch_vulkan() -> bool:
    """
    Check if PyTorch has Vulkan backend support.

    Returns:
        True if PyTorch Vulkan backend is available.
    """
    try:
        import torch  # pyrefly: ignore[missing-import]

        if hasattr(torch.backends, "vulkan") and torch.backends.vulkan.is_available():
            return True
    except ImportError:
        pass
    except Exception:
        pass

    return False


def detect_best_device(verbose: bool = False) -> str:
    """
    Detect the best available compute device.

    Priority: Vulkan (if PyTorch supports it) -> CUDA (compute >= 7.0) -> CPU

    Args:
        verbose: Print detection details.

    Returns:
        Device string: "vulkan", "cuda", or "cpu".
    """
    if verbose:
        print("Detecting available compute devices...")

    # Check Vulkan
    vulkan_hw = check_vulkan_support()
    vulkan_pytorch = check_pytorch_vulkan()

    if vulkan_hw and vulkan_pytorch:
        if verbose:
            print("Vulkan detected with PyTorch support")
        return "vulkan"

    if vulkan_hw and verbose:
        print("Vulkan hardware detected but PyTorch Vulkan backend not available")

    # Check CUDA
    cuda_available, gpu_name, compute_cap = check_cuda_support()

    if cuda_available and compute_cap is not None:
        if verbose:
            print(f"CUDA GPU detected: {gpu_name} (Compute {compute_cap[0]}.{compute_cap[1]})")

        # Require compute capability 7.0+ for good performance
        if compute_cap[0] >= 7:
            if verbose:
                print("CUDA compatible - using GPU acceleration")
            return "cuda"
        elif verbose:
            print(
                f"GPU compute capability {compute_cap[0]}.{compute_cap[1]} < 7.0 - "
                "falling back to CPU"
            )

    # Fall back to CPU
    if verbose:
        print("Using CPU backend")
    return "cpu"


def get_compute_type(device: str) -> str:
    """
    Get the recommended compute type for a device.

    Args:
        device: Device string ("cuda", "vulkan", or "cpu").

    Returns:
        Compute type string for faster-whisper.
    """
    if device in ("cuda", "vulkan"):
        return "float16"
    return "int8"


def get_device_info() -> DeviceInfo:
    """
    Get detailed device information.

    Returns:
        DeviceInfo with device detection results.
    """
    vulkan_hw = check_vulkan_support()
    vulkan_pytorch = check_pytorch_vulkan()
    cuda_available, gpu_name, compute_cap = check_cuda_support()

    # Determine recommended device using cached values
    if vulkan_hw and vulkan_pytorch:
        recommended = "vulkan"
    elif cuda_available and compute_cap is not None and compute_cap[0] >= 7:
        recommended = "cuda"
    else:
        recommended = "cpu"

    return DeviceInfo(
        vulkan_hardware=vulkan_hw,
        vulkan_pytorch=vulkan_pytorch,
        cuda_available=cuda_available,
        gpu_name=gpu_name,
        compute_capability=f"{compute_cap[0]}.{compute_cap[1]}" if compute_cap else None,
        recommended_device=recommended,
    )
