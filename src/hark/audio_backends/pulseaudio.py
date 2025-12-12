"""PulseAudio/PipeWire loopback backend for Linux.

This backend uses the pulsectl library (libpulse ctypes bindings) to discover
PulseAudio and PipeWire monitor sources (loopback devices).
"""

from hark.audio_backends.base import LoopbackDeviceInfo, RecordingConfig

__all__ = ["PulseAudioBackend"]

# Cached availability check to avoid repeated import attempts
_PULSECTL_AVAILABLE: bool | None = None


def _check_pulsectl_available() -> bool:
    """Check if pulsectl is available (cached).

    Returns:
        True if pulsectl can be imported, False otherwise.
    """
    global _PULSECTL_AVAILABLE
    if _PULSECTL_AVAILABLE is None:
        try:
            import pulsectl  # noqa: F401

            _PULSECTL_AVAILABLE = True
        except ImportError:
            _PULSECTL_AVAILABLE = False
    return _PULSECTL_AVAILABLE


class PulseAudioBackend:
    """PulseAudio/PipeWire loopback backend using pulsectl.

    Uses pulsectl (libpulse bindings) to discover monitor sources on Linux
    systems running PulseAudio or PipeWire (with PulseAudio compatibility).

    Monitor sources in PulseAudio/PipeWire capture the audio output from
    a sink (speaker), enabling system audio recording.
    """

    # Client name for PulseAudio connection
    _CLIENT_NAME = "hark"

    # Fallback values if source doesn't report them
    _DEFAULT_CHANNELS = 2
    _DEFAULT_SAMPLE_RATE = 44100.0

    def is_available(self) -> bool:
        """Check if PulseAudio/PipeWire is available.

        Returns:
            True if pulsectl can connect to PulseAudio/PipeWire.
        """
        if not _check_pulsectl_available():
            return False

        try:
            import pulsectl

            with pulsectl.Pulse(self._CLIENT_NAME) as pulse:
                pulse.server_info()
            return True
        except Exception:
            return False

    def get_default_loopback(self) -> LoopbackDeviceInfo | None:
        """Get the default loopback device (monitor of default sink).

        Returns:
            LoopbackDeviceInfo for the default sink's monitor, or None
            if no PulseAudio/PipeWire monitors are available.
        """
        if not _check_pulsectl_available():
            return None

        try:
            import pulsectl

            with pulsectl.Pulse(self._CLIENT_NAME) as pulse:
                server_info = pulse.server_info()
                default_sink = server_info.default_sink_name
                default_monitor = f"{default_sink}.monitor" if default_sink else None

                # Find the monitor source for the default sink
                for source in pulse.source_list():
                    if self._is_monitor(source) and source.name == default_monitor:
                        return self._to_device_info(source)

                # Fallback: return first monitor if no default match
                for source in pulse.source_list():
                    if self._is_monitor(source):
                        return self._to_device_info(source)

        except Exception:
            pass

        return None

    def list_loopback_devices(self) -> list[LoopbackDeviceInfo]:
        """List all available PulseAudio/PipeWire monitor sources.

        Returns:
            List of LoopbackDeviceInfo for all discovered monitors,
            sorted with the default sink's monitor first.
        """
        if not _check_pulsectl_available():
            return []

        try:
            import pulsectl

            with pulsectl.Pulse(self._CLIENT_NAME) as pulse:
                server_info = pulse.server_info()
                default_sink = server_info.default_sink_name
                default_monitor = f"{default_sink}.monitor" if default_sink else None

                # Collect all monitor sources
                monitors = [s for s in pulse.source_list() if self._is_monitor(s)]

                # Sort with default monitor first
                if default_monitor:
                    monitors.sort(key=lambda s: 0 if s.name == default_monitor else 1)

                return [self._to_device_info(s) for s in monitors]

        except Exception:
            return []

    def get_recording_config(self, device_id: str | int | None) -> RecordingConfig:
        """Get configuration for recording from a PulseAudio monitor source.

        For PulseAudio, recording from a monitor source requires:
        - Setting PULSE_SOURCE environment variable to the monitor name
        - Using "pulse" as the device identifier for sounddevice

        Args:
            device_id: The PulseAudio source name (e.g., "alsa_output...monitor").

        Returns:
            RecordingConfig with PULSE_SOURCE env var and "pulse" device.
        """
        env: dict[str, str] = {}
        if device_id is not None:
            env["PULSE_SOURCE"] = str(device_id)
        return RecordingConfig(env=env, device="pulse")

    def _is_monitor(self, source) -> bool:
        """Check if a PulseAudio source is a monitor source.

        Args:
            source: A pulsectl PulseSourceInfo object.

        Returns:
            True if the source name contains ".monitor".
        """
        return ".monitor" in source.name

    def _to_device_info(self, source) -> LoopbackDeviceInfo:
        """Convert a pulsectl PulseSourceInfo to LoopbackDeviceInfo.

        Args:
            source: A pulsectl PulseSourceInfo object.

        Returns:
            LoopbackDeviceInfo with extracted metadata.
        """
        # Get channel count from source
        channels = getattr(source, "channel_count", self._DEFAULT_CHANNELS)

        # Get sample rate from sample_spec
        sample_rate = self._DEFAULT_SAMPLE_RATE
        if hasattr(source, "sample_spec") and source.sample_spec:
            rate = getattr(source.sample_spec, "rate", None)
            if rate:
                sample_rate = float(rate)

        # Use description if available, fall back to name
        name = source.description if source.description else source.name

        return LoopbackDeviceInfo(
            name=name,
            device_id=source.name,
            channels=channels,
            sample_rate=sample_rate,
        )
