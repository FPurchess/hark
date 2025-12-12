"""Audio backends for platform-specific audio capture."""

from hark.audio_backends.base import LoopbackBackend, LoopbackDeviceInfo, RecordingConfig
from hark.platform import is_linux, is_macos, is_windows

__all__ = [
    "LoopbackBackend",
    "LoopbackDeviceInfo",
    "RecordingConfig",
    "get_loopback_backend",
]


def get_loopback_backend() -> LoopbackBackend:
    """
    Get the appropriate loopback backend for the current platform.

    Returns the platform-specific backend implementation:
    - Linux: PulseAudioBackend (PulseAudio/PipeWire)
    - macOS: CoreAudioBackend (BlackHole detection)
    - Windows: (Future) WASAPIBackend

    Returns:
        A LoopbackBackend implementation for the current platform.

    Raises:
        NotImplementedError: If no backend is available for the current platform.
    """
    if is_linux():
        from hark.audio_backends.pulseaudio import PulseAudioBackend

        return PulseAudioBackend()

    if is_macos():
        from hark.audio_backends.coreaudio import CoreAudioBackend

        return CoreAudioBackend()

    if is_windows():
        from hark.audio_backends.wasapi import WASAPIBackend

        return WASAPIBackend()

    raise NotImplementedError(
        "Loopback audio capture is not yet supported on this platform. "
        "Currently Linux (PulseAudio/PipeWire), macOS, and Windows are supported."
    )
