"""Audio source detection and management for hark."""

import contextlib
import re
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Any, cast

import sounddevice as sd

from hark.audio_backends import RecordingConfig, get_loopback_backend
from hark.audio_backends.base import LoopbackBackend
from hark.platform import is_linux, is_macos, is_windows

__all__ = [
    "InputSource",
    "AudioSourceInfo",
    "find_microphone_device",
    "find_loopback_device",
    "list_loopback_devices",
    "get_devices_for_source",
    "validate_source_availability",
]


class InputSource(Enum):
    """Audio input source modes."""

    MIC = "mic"
    SPEAKER = "speaker"
    BOTH = "both"


@dataclass
class AudioSourceInfo:
    """Information about an audio source."""

    device_index: int | None  # sounddevice index, None for backend-managed devices
    name: str
    channels: int
    sample_rate: float
    is_loopback: bool
    recording_config: RecordingConfig | None = None  # Platform-specific recording config


@lru_cache(maxsize=1)
def _get_loopback_backend() -> LoopbackBackend | None:
    """Get loopback backend lazily, returning None on unsupported platforms."""
    try:
        return get_loopback_backend()
    except NotImplementedError:
        return None


def _is_monitor_device(device_name: str) -> bool:
    """
    Check if device name indicates a monitor/loopback source.

    Args:
        device_name: Name of the audio device.

    Returns:
        True if device appears to be a monitor/loopback source.
    """
    name_lower = device_name.lower()

    # PulseAudio/PipeWire patterns
    if ".monitor" in name_lower:
        return True
    if "monitor of" in name_lower:
        return True

    # Common patterns across platforms
    if re.search(r"\bloopback\b", name_lower):
        return True
    if "stereo mix" in name_lower:
        return True

    return "what u hear" in name_lower


def find_microphone_device() -> AudioSourceInfo | None:
    """
    Find the default microphone device.

    Returns:
        AudioSourceInfo for the default microphone, or None if not found.
    """
    try:
        device_id = sd.default.device[0]
        if device_id is None:
            return None

        device = cast(dict[str, Any], sd.query_devices(device_id))
        if device["max_input_channels"] > 0:
            return AudioSourceInfo(
                device_index=int(device_id),
                name=str(device["name"]),
                channels=int(device["max_input_channels"]),
                sample_rate=float(device["default_samplerate"]),
                is_loopback=False,
                recording_config=None,
            )
    except Exception:
        pass

    return None


def find_loopback_device() -> AudioSourceInfo | None:
    """
    Find a system audio loopback/monitor device.

    On Linux, uses PulseAudio/PipeWire backend to discover monitor sources.
    Falls back to checking sounddevice for other platforms.

    Returns:
        AudioSourceInfo for the first loopback device found, or None if not found.
    """
    # First, try platform-specific loopback backend
    backend = _get_loopback_backend()
    loopback_info = None
    if backend is not None:
        with contextlib.suppress(Exception):
            loopback_info = backend.get_default_loopback()
    if loopback_info is not None and backend is not None:
        return AudioSourceInfo(
            device_index=None,  # Backend device, not sounddevice index
            name=loopback_info.name,
            channels=loopback_info.channels,
            sample_rate=loopback_info.sample_rate,
            is_loopback=True,
            recording_config=backend.get_recording_config(loopback_info.device_id),
        )

    # Fallback: check sounddevice for loopback devices (other platforms)
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        device = cast(dict[str, Any], device)
        if device["max_input_channels"] > 0 and _is_monitor_device(device["name"]):
            return AudioSourceInfo(
                device_index=i,
                name=str(device["name"]),
                channels=int(device["max_input_channels"]),
                sample_rate=float(device["default_samplerate"]),
                is_loopback=True,
                recording_config=None,  # No backend config for sounddevice fallback
            )

    return None


def list_loopback_devices() -> list[AudioSourceInfo]:
    """
    List all available loopback/monitor devices.

    Returns:
        List of AudioSourceInfo for all loopback devices found.
    """
    loopbacks: list[AudioSourceInfo] = []

    # Get loopback devices via platform-specific backend
    backend = _get_loopback_backend()
    backend_devices = backend.list_loopback_devices() if backend else []
    for device_info in backend_devices:
        loopbacks.append(
            AudioSourceInfo(
                device_index=None,
                name=device_info.name,
                channels=device_info.channels,
                sample_rate=device_info.sample_rate,
                is_loopback=True,
                recording_config=(
                    backend.get_recording_config(device_info.device_id) if backend else None
                ),
            )
        )

    # Also check sounddevice for any loopback devices (other platforms)
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        device = cast(dict[str, Any], device)
        # Check if it's an input device with monitor characteristics, avoiding duplicates
        if (
            device["max_input_channels"] > 0
            and _is_monitor_device(device["name"])
            and not any(lb.name == device["name"] for lb in loopbacks)
        ):
            loopbacks.append(
                AudioSourceInfo(
                    device_index=i,
                    name=str(device["name"]),
                    channels=int(device["max_input_channels"]),
                    sample_rate=float(device["default_samplerate"]),
                    is_loopback=True,
                    recording_config=None,  # No backend config for sounddevice fallback
                )
            )

    return loopbacks


def get_devices_for_source(
    source: InputSource,
) -> tuple[AudioSourceInfo | None, AudioSourceInfo | None]:
    """
    Get device(s) for the requested input source.

    Args:
        source: The input source mode.

    Returns:
        Tuple of (mic_device, loopback_device). Either may be None based on source mode.
    """
    mic: AudioSourceInfo | None = None
    loopback: AudioSourceInfo | None = None

    if source in (InputSource.MIC, InputSource.BOTH):
        mic = find_microphone_device()

    if source in (InputSource.SPEAKER, InputSource.BOTH):
        loopback = find_loopback_device()

    return mic, loopback


def validate_source_availability(source: InputSource) -> list[str]:
    """
    Validate that required audio sources are available.

    Args:
        source: The input source mode to validate.

    Returns:
        List of error messages (empty if all required sources are available).
    """
    errors: list[str] = []
    mic, loopback = get_devices_for_source(source)

    if source in (InputSource.MIC, InputSource.BOTH) and mic is None:
        errors.append("No microphone device found")

    if source in (InputSource.SPEAKER, InputSource.BOTH) and loopback is None:
        msg = "No system audio loopback device found. "
        if is_linux():
            msg += "Ensure PulseAudio/PipeWire monitor source is available."
        elif is_macos():
            msg += (
                "Install BlackHole for system audio capture. "
                "See https://github.com/ExistentialAudio/BlackHole for installation."
            )
        elif is_windows():
            msg += (
                "WASAPI loopback device not found. Requires Windows 10 or later. "
                "Ensure your audio output device is working."
            )
        else:
            msg += "System audio capture may not be supported on this platform."
        errors.append(msg)

    return errors
