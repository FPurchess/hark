"""Core Audio loopback backend for macOS.

Detects BlackHole virtual audio device via sounddevice's Core Audio integration.
Users must install BlackHole separately for system audio capture.
"""

import re
from typing import Any, cast

import sounddevice as sd

from hark.audio_backends.base import LoopbackDeviceInfo, RecordingConfig
from hark.platform import is_macos

__all__ = ["CoreAudioBackend"]

# Cached availability check to avoid repeated query attempts
_SOUNDDEVICE_AVAILABLE: bool | None = None


def _check_sounddevice_available() -> bool:
    """Check if sounddevice can query devices (cached).

    Returns:
        True if sounddevice can enumerate audio devices, False otherwise.
    """
    global _SOUNDDEVICE_AVAILABLE
    if _SOUNDDEVICE_AVAILABLE is None:
        try:
            sd.query_devices()
            _SOUNDDEVICE_AVAILABLE = True
        except Exception:
            _SOUNDDEVICE_AVAILABLE = False
    return _SOUNDDEVICE_AVAILABLE


class CoreAudioBackend:
    """Core Audio loopback backend using sounddevice for BlackHole detection.

    Uses sounddevice (which wraps PortAudio/Core Audio on macOS) to discover
    BlackHole virtual audio devices. BlackHole must be installed separately
    by the user for system audio capture to work.

    BlackHole creates virtual audio devices that can capture system audio
    when configured as a Multi-Output Device in macOS Audio MIDI Setup.
    """

    # Fallback values if device doesn't report them
    _DEFAULT_CHANNELS = 2
    _DEFAULT_SAMPLE_RATE = 44100.0

    # BlackHole detection pattern (only supported virtual device for now)
    _BLACKHOLE_PATTERN = r"blackhole"

    def is_available(self) -> bool:
        """Check if Core Audio is available via sounddevice.

        Returns:
            True if sounddevice can query Core Audio devices on macOS.
        """
        if not _check_sounddevice_available():
            return False

        try:
            if not is_macos():
                return False

            devices = sd.query_devices()
            return len(devices) > 0
        except Exception:
            return False

    def get_default_loopback(self) -> LoopbackDeviceInfo | None:
        """Get the default (first) BlackHole loopback device.

        Returns:
            LoopbackDeviceInfo for the first BlackHole device found,
            or None if no BlackHole device is detected.
        """
        if not _check_sounddevice_available():
            return None

        try:
            devices = self._find_blackhole_devices()
            return devices[0] if devices else None
        except Exception:
            return None

    def list_loopback_devices(self) -> list[LoopbackDeviceInfo]:
        """List all detected BlackHole devices.

        Returns:
            List of LoopbackDeviceInfo sorted alphabetically by name.
        """
        if not _check_sounddevice_available():
            return []

        try:
            return self._find_blackhole_devices()
        except Exception:
            return []

    def get_recording_config(self, device_id: str | int | None) -> RecordingConfig:
        """Get configuration for recording from a BlackHole device.

        On macOS, no environment variables are needed. The device index
        is passed directly to sounddevice.

        Args:
            device_id: The sounddevice device index (integer).

        Returns:
            RecordingConfig with empty env dict and device index.
        """
        # macOS doesn't need environment variables like PULSE_SOURCE
        # The device index is passed directly to sounddevice
        device = device_id if isinstance(device_id, int) else None
        return RecordingConfig(env={}, device=device)

    def _find_blackhole_devices(self) -> list[LoopbackDeviceInfo]:
        """Find all BlackHole virtual audio devices.

        Returns:
            List of LoopbackDeviceInfo for detected BlackHole devices,
            sorted alphabetically by name.
        """
        devices = sd.query_devices()
        blackhole_devices: list[LoopbackDeviceInfo] = []

        for i, device in enumerate(devices):
            device = cast(dict[str, Any], device)

            # Skip output-only devices
            if device["max_input_channels"] <= 0:
                continue

            name = str(device["name"])
            if self._is_blackhole(name):
                info = LoopbackDeviceInfo(
                    name=name,
                    device_id=i,
                    channels=int(device["max_input_channels"]),
                    sample_rate=float(device["default_samplerate"]),
                )
                blackhole_devices.append(info)

        # Sort alphabetically for stability
        blackhole_devices.sort(key=lambda x: x.name)
        return blackhole_devices

    def _is_blackhole(self, device_name: str) -> bool:
        """Check if device name indicates a BlackHole device.

        Args:
            device_name: The device name from sounddevice.

        Returns:
            True if the device appears to be a BlackHole device.
        """
        return bool(re.search(self._BLACKHOLE_PATTERN, device_name, re.IGNORECASE))
