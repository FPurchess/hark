"""WASAPI loopback backend for Windows.

Uses PyAudioWPatch to capture system audio via WASAPI loopback.
This enables recording system audio on Windows 10+ without requiring
virtual audio cables or Stereo Mix.
"""

from typing import Any

from hark.audio_backends.base import LoopbackDeviceInfo, RecordingConfig

__all__ = ["WASAPIBackend"]

# Cached availability check to avoid repeated import attempts
_PYAUDIOWPATCH_AVAILABLE: bool | None = None


def _check_pyaudiowpatch_available() -> bool:
    """Check if PyAudioWPatch is available (cached).

    Returns:
        True if PyAudioWPatch can be imported, False otherwise.
    """
    global _PYAUDIOWPATCH_AVAILABLE
    if _PYAUDIOWPATCH_AVAILABLE is None:
        try:
            import pyaudiowpatch  # noqa: F401  # pyrefly: ignore[missing-import]

            _PYAUDIOWPATCH_AVAILABLE = True
        except ImportError:
            _PYAUDIOWPATCH_AVAILABLE = False
    return _PYAUDIOWPATCH_AVAILABLE


class WASAPIBackend:
    """WASAPI loopback backend using PyAudioWPatch.

    Uses PyAudioWPatch to discover and configure WASAPI loopback devices
    on Windows. WASAPI loopback allows capturing the audio output from
    any speaker device without requiring Stereo Mix or virtual audio drivers.

    Requirements:
        - Windows 10 or later
        - PyAudioWPatch package installed
    """

    # Fallback values if device doesn't report them
    _DEFAULT_CHANNELS = 2
    _DEFAULT_SAMPLE_RATE = 44100.0

    def is_available(self) -> bool:
        """Check if WASAPI loopback is available.

        Returns:
            True if PyAudioWPatch can access WASAPI loopback devices.
        """
        if not _check_pyaudiowpatch_available():
            return False

        try:
            import pyaudiowpatch as pyaudio  # pyrefly: ignore[missing-import]

            with pyaudio.PyAudio() as p:
                # Try to get default loopback - if this works, WASAPI is available
                p.get_default_wasapi_loopback()
            return True
        except (OSError, Exception):
            return False

    def get_default_loopback(self) -> LoopbackDeviceInfo | None:
        """Get the default WASAPI loopback device.

        Returns:
            LoopbackDeviceInfo for the default loopback device, or None
            if no loopback device is available.
        """
        if not _check_pyaudiowpatch_available():
            return None

        try:
            import pyaudiowpatch as pyaudio  # pyrefly: ignore[missing-import]

            with pyaudio.PyAudio() as p:
                device = p.get_default_wasapi_loopback()
                return self._to_device_info(device)
        except (OSError, Exception):
            return None

    def list_loopback_devices(self) -> list[LoopbackDeviceInfo]:
        """List all available WASAPI loopback devices.

        Returns:
            List of LoopbackDeviceInfo for all discovered loopback devices,
            sorted with the default loopback device first.
        """
        if not _check_pyaudiowpatch_available():
            return []

        try:
            import pyaudiowpatch as pyaudio  # pyrefly: ignore[missing-import]

            devices: list[LoopbackDeviceInfo] = []
            default_index: int | None = None

            with pyaudio.PyAudio() as p:
                # Get default loopback index for sorting
                try:
                    default_device = p.get_default_wasapi_loopback()
                    default_index = default_device.get("index")
                except (OSError, Exception):
                    pass

                # Collect all loopback devices
                for device in p.get_loopback_device_info_generator():
                    devices.append(self._to_device_info(device))

            # Sort with default device first
            if default_index is not None:
                devices.sort(key=lambda d: 0 if d.device_id == default_index else 1)

            return devices

        except (OSError, Exception):
            return []

    def get_recording_config(self, device_id: str | int | None) -> RecordingConfig:
        """Get configuration for recording from a WASAPI loopback device.

        For WASAPI loopback, no environment variables are needed. The device
        identifier uses a special format to signal that PyAudioWPatch should
        be used for recording.

        Args:
            device_id: The PyAudioWPatch device index (integer).

        Returns:
            RecordingConfig with empty env dict and wasapi device marker.
        """
        # Use a special marker format to indicate WASAPI loopback
        # The recorder will detect this and use PyAudioWPatch for recording
        device_marker = f"wasapi:{device_id}" if device_id is not None else "wasapi"

        return RecordingConfig(env={}, device=device_marker)

    def _to_device_info(self, device: dict[str, Any]) -> LoopbackDeviceInfo:
        """Convert a PyAudioWPatch device dict to LoopbackDeviceInfo.

        Args:
            device: A PyAudioWPatch device info dictionary.

        Returns:
            LoopbackDeviceInfo with extracted metadata.
        """
        # PyAudioWPatch device dict keys:
        # - index: int
        # - name: str (includes "[Loopback]" suffix)
        # - maxInputChannels: int
        # - defaultSampleRate: float
        # - isLoopbackDevice: bool

        name = device.get("name", "Unknown Device")
        device_index = device.get("index")
        channels = device.get("maxInputChannels", self._DEFAULT_CHANNELS)
        sample_rate = device.get("defaultSampleRate", self._DEFAULT_SAMPLE_RATE)

        return LoopbackDeviceInfo(
            name=name,
            device_id=device_index,
            channels=channels,
            sample_rate=float(sample_rate),
        )
