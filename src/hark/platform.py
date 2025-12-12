"""Platform detection utilities for cross-platform support."""

import sys
from enum import Enum

__all__ = [
    "Platform",
    "get_platform",
    "is_linux",
    "is_macos",
    "is_windows",
]


class Platform(Enum):
    """Supported operating system platforms."""

    LINUX = "linux"
    MACOS = "darwin"
    WINDOWS = "win32"
    UNKNOWN = "unknown"


def get_platform() -> Platform:
    """
    Detect the current operating system platform.

    Returns:
        Platform enum value for the current OS.
    """
    if sys.platform.startswith("linux"):
        return Platform.LINUX
    elif sys.platform == "darwin":
        return Platform.MACOS
    elif sys.platform == "win32":
        return Platform.WINDOWS
    return Platform.UNKNOWN


def is_linux() -> bool:
    """
    Check if the current platform is Linux.

    Returns:
        True if running on Linux.
    """
    return get_platform() == Platform.LINUX


def is_macos() -> bool:
    """
    Check if the current platform is macOS.

    Returns:
        True if running on macOS.
    """
    return get_platform() == Platform.MACOS


def is_windows() -> bool:
    """
    Check if the current platform is Windows.

    Returns:
        True if running on Windows.
    """
    return get_platform() == Platform.WINDOWS
