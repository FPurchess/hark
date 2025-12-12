"""Compute device detection for Whisper inference."""

from typing import TypedDict

__all__ = [
    "DeviceInfo",
    "check_cuda_support",
    "check_mps_support",
    "detect_best_device",
    "get_compute_type",
    "get_device_info",
]


class DeviceInfo(TypedDict):
    """Information about detected compute devices."""

    cuda_available: bool
    mps_available: bool
    gpu_name: str | None
    compute_capability: str | None
    recommended_device: str


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


def check_mps_support() -> bool:
    """
    Check if MPS (Metal Performance Shaders) is available on macOS.

    Returns:
        True if MPS is available (macOS with Apple Silicon or AMD GPU).
    """
    try:
        import torch  # pyrefly: ignore[missing-import]

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return True
    except ImportError:
        pass
    except Exception:
        pass

    return False


def detect_best_device(verbose: bool = False) -> str:
    """
    Detect the best available compute device.

    Priority: CUDA (compute >= 7.0) -> MPS (macOS) -> CPU

    Args:
        verbose: Print detection details.

    Returns:
        Device string: "cuda", "mps", or "cpu".
    """
    if verbose:
        print("Detecting available compute devices...")

    # Check CUDA first (best performance when available)
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
                "checking other backends"
            )

    # Check MPS (macOS Apple Silicon / AMD GPU)
    mps_available = check_mps_support()
    if mps_available:
        if verbose:
            print("MPS (Metal) detected - using Apple GPU acceleration")
        return "mps"

    # Fall back to CPU
    if verbose:
        print("Using CPU backend")
    return "cpu"


def get_compute_type(device: str) -> str:
    """
    Get the recommended compute type for a device.

    Args:
        device: Device string ("cuda", "mps", or "cpu").

    Returns:
        Compute type string for faster-whisper.
    """
    if device in ("cuda", "mps"):
        return "float16"
    return "int8"


def get_device_info() -> DeviceInfo:
    """
    Get detailed device information.

    Returns:
        DeviceInfo with device detection results.
    """
    cuda_available, gpu_name, compute_cap = check_cuda_support()
    mps_available = check_mps_support()

    # Determine recommended device (same priority as detect_best_device)
    if cuda_available and compute_cap is not None and compute_cap[0] >= 7:
        recommended = "cuda"
    elif mps_available:
        recommended = "mps"
    else:
        recommended = "cpu"

    return DeviceInfo(
        cuda_available=cuda_available,
        mps_available=mps_available,
        gpu_name=gpu_name,
        compute_capability=f"{compute_cap[0]}.{compute_cap[1]}" if compute_cap else None,
        recommended_device=recommended,
    )
