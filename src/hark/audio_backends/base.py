"""Abstract base classes and protocols for audio backends.

These define the contracts that audio backend implementations must follow.
Using Protocol allows for structural subtyping (duck typing with type safety).
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

__all__ = [
    "LoopbackDeviceInfo",
    "LoopbackBackend",
    "RecordingConfig",
]


@dataclass
class LoopbackDeviceInfo:
    """Information about a loopback/monitor audio device.

    This is a platform-agnostic representation of a loopback device.
    Each platform backend translates its native device representation
    to this common format.

    Attributes:
        name: Human-readable device name (e.g., "Monitor of Built-in Audio").
        device_id: Platform-specific device identifier. On Linux this may be
            a PulseAudio source name, on Windows a WASAPI device index, etc.
        channels: Number of audio channels (typically 2 for stereo).
        sample_rate: Default sample rate in Hz.
    """

    name: str
    device_id: str | int | None
    channels: int
    sample_rate: float


@dataclass
class RecordingConfig:
    """Platform-specific configuration for recording from a loopback device.

    This encapsulates the environment and device settings needed to record
    from a loopback source. Each platform backend provides its own configuration:

    - Linux (PulseAudio): Sets PULSE_SOURCE env var, uses "pulse" device
    - Windows (WASAPI): No env vars, uses device index directly
    - macOS: No env vars, uses device index for virtual audio device

    Attributes:
        env: Environment variables to set before recording (may be empty).
        device: Device identifier to pass to sounddevice (index, name, or None).
    """

    env: dict[str, str]
    device: int | str | None


@runtime_checkable
class LoopbackBackend(Protocol):
    """Protocol for platform-specific loopback audio capture backends.

    Implementations must provide methods to discover and list loopback
    (monitor/system audio) devices. Each platform has different APIs for this:

    - Linux: PulseAudio/PipeWire monitor sources via `pactl`
    - Windows: WASAPI loopback devices via PyAudioWPatch
    - macOS: Virtual audio devices (BlackHole, etc.) or Core Audio taps

    Example implementation:
        class PulseAudioBackend:
            def get_default_loopback(self) -> LoopbackDeviceInfo | None:
                ...
            def list_loopback_devices(self) -> list[LoopbackDeviceInfo]:
                ...
            def is_available(self) -> bool:
                ...
    """

    def get_default_loopback(self) -> LoopbackDeviceInfo | None:
        """Get the default loopback device for the system.

        On Linux, this returns the monitor source for the default sink.
        On Windows, this returns the default WASAPI loopback device.
        On macOS, this returns the first available virtual audio device.

        Returns:
            LoopbackDeviceInfo for the default loopback device, or None
            if no loopback device is available.
        """
        ...

    def list_loopback_devices(self) -> list[LoopbackDeviceInfo]:
        """List all available loopback devices.

        Returns:
            List of LoopbackDeviceInfo for all discovered loopback devices.
            Returns an empty list if no devices are found.
        """
        ...

    def is_available(self) -> bool:
        """Check if this backend is available on the current system.

        This checks for the presence of required system components (e.g.,
        PulseAudio/PipeWire on Linux, WASAPI on Windows).

        Returns:
            True if the backend can be used on this system.
        """
        ...

    def get_recording_config(self, device_id: str | int | None) -> RecordingConfig:
        """Get configuration for recording from a loopback device.

        Returns the platform-specific environment variables and device
        identifier needed to record from the given loopback device.

        Args:
            device_id: The device identifier from LoopbackDeviceInfo.device_id.

        Returns:
            RecordingConfig with env vars and sounddevice device identifier.
        """
        ...
