"""Audio source detection and management for hark."""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

import sounddevice as sd

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

    device_index: int | None  # sounddevice index, None for PulseAudio sources
    name: str
    channels: int
    sample_rate: float
    is_loopback: bool
    pulse_source_name: str | None = None  # PulseAudio source name for loopback


def _get_default_sink() -> str | None:
    """
    Get the default PulseAudio/PipeWire sink name.

    Returns:
        The default sink name, or None if not found.
    """
    try:
        env = os.environ.copy()
        env["LC_ALL"] = "C"

        result = subprocess.run(
            ["pactl", "get-default-sink"],
            capture_output=True,
            text=True,
            timeout=5,
            env=env,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    return None


def _get_pulseaudio_monitors() -> list[dict[str, str]]:
    """
    Get PulseAudio/PipeWire monitor sources via pactl.

    Returns:
        List of dicts with 'name' and 'description' keys,
        sorted with the default sink's monitor first.
    """
    monitors: list[dict[str, str]] = []

    try:
        # Use LC_ALL=C to get consistent English output regardless of locale
        env = os.environ.copy()
        env["LC_ALL"] = "C"

        result = subprocess.run(
            ["pactl", "list", "sources"],
            capture_output=True,
            text=True,
            timeout=5,
            env=env,
        )
        if result.returncode != 0:
            return monitors

        current_source: dict[str, str] = {}
        for line in result.stdout.splitlines():
            line = line.strip()

            # New source entry (LC_ALL=C ensures English output)
            if line.startswith("Source #"):
                if current_source and ".monitor" in current_source.get("name", ""):
                    monitors.append(current_source)
                current_source = {}

            elif line.startswith("Name:"):
                current_source["name"] = line.split(":", 1)[1].strip()

            elif line.startswith("Description:"):
                current_source["description"] = line.split(":", 1)[1].strip()

        # Don't forget the last source
        if current_source and ".monitor" in current_source.get("name", ""):
            monitors.append(current_source)

        # Sort monitors to prefer the default sink's monitor
        default_sink = _get_default_sink()
        if default_sink:
            default_monitor_name = f"{default_sink}.monitor"
            monitors.sort(key=lambda m: 0 if m.get("name") == default_monitor_name else 1)

    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    return monitors


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
                pulse_source_name=None,
            )
    except Exception:
        pass

    return None


def find_loopback_device() -> AudioSourceInfo | None:
    """
    Find a system audio loopback/monitor device.

    On Linux, uses pactl to discover PulseAudio/PipeWire monitor sources.
    Falls back to checking sounddevice for other platforms.

    Returns:
        AudioSourceInfo for the first loopback device found, or None if not found.
    """
    # First, try PulseAudio/PipeWire monitors via pactl (Linux)
    monitors = _get_pulseaudio_monitors()
    if monitors:
        monitor = monitors[0]  # Use first available monitor
        return AudioSourceInfo(
            device_index=None,  # Use pulse device, not index
            name=monitor.get("description", monitor["name"]),
            channels=2,  # Monitors are typically stereo
            sample_rate=44100.0,  # Default, will be overridden by recorder
            is_loopback=True,
            pulse_source_name=monitor["name"],
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
                pulse_source_name=None,
            )

    return None


def list_loopback_devices() -> list[AudioSourceInfo]:
    """
    List all available loopback/monitor devices.

    Returns:
        List of AudioSourceInfo for all loopback devices found.
    """
    loopbacks: list[AudioSourceInfo] = []

    # Get PulseAudio/PipeWire monitors
    monitors = _get_pulseaudio_monitors()
    for monitor in monitors:
        loopbacks.append(
            AudioSourceInfo(
                device_index=None,
                name=monitor.get("description", monitor["name"]),
                channels=2,
                sample_rate=44100.0,
                is_loopback=True,
                pulse_source_name=monitor["name"],
            )
        )

    # Also check sounddevice for any loopback devices
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
                    pulse_source_name=None,
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
        errors.append(
            "No system audio loopback device found. "
            "On Linux, ensure PulseAudio/PipeWire monitor source is available."
        )

    return errors


def setup_loopback_env(loopback: AudioSourceInfo) -> dict[str, str]:
    """
    Get environment variables needed for loopback recording.

    Args:
        loopback: The loopback device info.

    Returns:
        Dict of environment variables to set (may be empty).
    """
    env = os.environ.copy()

    if loopback.pulse_source_name:
        env["PULSE_SOURCE"] = loopback.pulse_source_name

    return env
