"""Type definitions for the recorder module."""

from typing import TypedDict

__all__ = ["AudioDeviceInfo"]


class AudioDeviceInfo(TypedDict):
    """Information about an audio input device."""

    index: int
    name: str
    channels: int
    sample_rate: float
