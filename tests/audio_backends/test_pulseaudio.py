"""Tests for PulseAudio backend - Linux only."""

import sys
from unittest.mock import MagicMock, patch

import pytest

# Skip entire module on non-Linux platforms
pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="PulseAudio tests require Linux")

from hark.audio_backends import LoopbackBackend  # noqa: E402
from hark.audio_backends.pulseaudio import PulseAudioBackend  # noqa: E402


def _create_mock_source(
    name: str,
    description: str | None = None,
    channel_count: int = 2,
    sample_rate: int = 48000,
) -> MagicMock:
    """Create a mock PulseSourceInfo object."""
    source = MagicMock()
    source.name = name
    source.description = description
    source.channel_count = channel_count
    source.sample_spec = MagicMock()
    source.sample_spec.rate = sample_rate
    return source


def _create_mock_server_info(default_sink: str | None = None) -> MagicMock:
    """Create a mock PulseServerInfo object."""
    info = MagicMock()
    info.default_sink_name = default_sink
    return info


class TestPulseAudioBackend:
    """Tests for PulseAudioBackend implementation."""

    def test_implements_loopback_backend(self) -> None:
        """PulseAudioBackend should implement LoopbackBackend protocol."""
        backend = PulseAudioBackend()
        assert isinstance(backend, LoopbackBackend)

    @patch("hark.audio_backends.pulseaudio._check_pulsectl_available", return_value=True)
    @patch("pulsectl.Pulse")
    def test_is_available_success(self, mock_pulse_class: MagicMock, _: MagicMock) -> None:
        """Should return True when pulsectl connects successfully."""
        mock_pulse = MagicMock()
        mock_pulse_class.return_value.__enter__.return_value = mock_pulse
        mock_pulse.server_info.return_value = _create_mock_server_info()

        backend = PulseAudioBackend()
        assert backend.is_available() is True

    @patch("hark.audio_backends.pulseaudio._check_pulsectl_available", return_value=True)
    @patch("pulsectl.Pulse")
    def test_is_available_connection_fails(self, mock_pulse_class: MagicMock, _: MagicMock) -> None:
        """Should return False when PulseAudio connection fails."""
        mock_pulse_class.return_value.__enter__.side_effect = Exception("Connection refused")

        backend = PulseAudioBackend()
        assert backend.is_available() is False

    @patch("hark.audio_backends.pulseaudio._check_pulsectl_available", return_value=False)
    def test_is_available_pulsectl_not_installed(self, _: MagicMock) -> None:
        """Should return False when pulsectl is not installed."""
        backend = PulseAudioBackend()
        assert backend.is_available() is False

    @patch("hark.audio_backends.pulseaudio._check_pulsectl_available", return_value=True)
    @patch("pulsectl.Pulse")
    def test_get_default_loopback_success(self, mock_pulse_class: MagicMock, _: MagicMock) -> None:
        """Should return monitor device info when available."""
        mock_pulse = MagicMock()
        mock_pulse_class.return_value.__enter__.return_value = mock_pulse

        mock_pulse.server_info.return_value = _create_mock_server_info(
            default_sink="alsa_output.pci.analog-stereo"
        )
        mock_pulse.source_list.return_value = [
            _create_mock_source(
                name="alsa_output.pci.analog-stereo.monitor",
                description="Monitor of Built-in Audio Analog Stereo",
                channel_count=2,
                sample_rate=48000,
            ),
            _create_mock_source(
                name="alsa_input.pci.analog-stereo",
                description="Built-in Audio Analog Stereo",
            ),
        ]

        backend = PulseAudioBackend()
        device = backend.get_default_loopback()

        assert device is not None
        assert device.name == "Monitor of Built-in Audio Analog Stereo"
        assert device.device_id == "alsa_output.pci.analog-stereo.monitor"
        assert device.channels == 2
        assert device.sample_rate == 48000.0

    @patch("hark.audio_backends.pulseaudio._check_pulsectl_available", return_value=True)
    @patch("pulsectl.Pulse")
    def test_get_default_loopback_no_monitors(
        self, mock_pulse_class: MagicMock, _: MagicMock
    ) -> None:
        """Should return None when no monitors are available."""
        mock_pulse = MagicMock()
        mock_pulse_class.return_value.__enter__.return_value = mock_pulse

        mock_pulse.server_info.return_value = _create_mock_server_info()
        mock_pulse.source_list.return_value = [
            _create_mock_source(
                name="alsa_input.pci.analog-stereo",
                description="Built-in Audio Analog Stereo",
            ),
        ]

        backend = PulseAudioBackend()
        device = backend.get_default_loopback()
        assert device is None

    @patch("hark.audio_backends.pulseaudio._check_pulsectl_available", return_value=False)
    def test_get_default_loopback_pulsectl_unavailable(self, _: MagicMock) -> None:
        """Should return None when pulsectl is not available."""
        backend = PulseAudioBackend()
        device = backend.get_default_loopback()
        assert device is None

    @patch("hark.audio_backends.pulseaudio._check_pulsectl_available", return_value=True)
    @patch("pulsectl.Pulse")
    def test_get_default_loopback_falls_back_to_first_monitor(
        self, mock_pulse_class: MagicMock, _: MagicMock
    ) -> None:
        """Should fall back to first monitor if default sink's monitor not found."""
        mock_pulse = MagicMock()
        mock_pulse_class.return_value.__enter__.return_value = mock_pulse

        mock_pulse.server_info.return_value = _create_mock_server_info(
            default_sink="nonexistent_sink"
        )
        mock_pulse.source_list.return_value = [
            _create_mock_source(
                name="some_other.monitor",
                description="Some Other Monitor",
            ),
        ]

        backend = PulseAudioBackend()
        device = backend.get_default_loopback()

        assert device is not None
        assert device.device_id == "some_other.monitor"

    @patch("hark.audio_backends.pulseaudio._check_pulsectl_available", return_value=True)
    @patch("pulsectl.Pulse")
    def test_list_loopback_devices_success(self, mock_pulse_class: MagicMock, _: MagicMock) -> None:
        """Should return list of all monitor devices."""
        mock_pulse = MagicMock()
        mock_pulse_class.return_value.__enter__.return_value = mock_pulse

        mock_pulse.server_info.return_value = _create_mock_server_info()
        mock_pulse.source_list.return_value = [
            _create_mock_source(name="sink1.monitor", description="Monitor of Speaker 1"),
            _create_mock_source(name="alsa_input.pci.analog-stereo", description="Microphone"),
            _create_mock_source(name="sink2.monitor", description="Monitor of Speaker 2"),
        ]

        backend = PulseAudioBackend()
        devices = backend.list_loopback_devices()

        assert len(devices) == 2
        assert devices[0].name == "Monitor of Speaker 1"
        assert devices[0].device_id == "sink1.monitor"
        assert devices[1].name == "Monitor of Speaker 2"
        assert devices[1].device_id == "sink2.monitor"

    @patch("hark.audio_backends.pulseaudio._check_pulsectl_available", return_value=False)
    def test_list_loopback_devices_empty_when_unavailable(self, _: MagicMock) -> None:
        """Should return empty list when pulsectl is not available."""
        backend = PulseAudioBackend()
        devices = backend.list_loopback_devices()
        assert devices == []

    @patch("hark.audio_backends.pulseaudio._check_pulsectl_available", return_value=True)
    @patch("pulsectl.Pulse")
    def test_monitors_sorted_by_default_sink(
        self, mock_pulse_class: MagicMock, _: MagicMock
    ) -> None:
        """Should sort monitors with default sink's monitor first."""
        mock_pulse = MagicMock()
        mock_pulse_class.return_value.__enter__.return_value = mock_pulse

        mock_pulse.server_info.return_value = _create_mock_server_info(default_sink="sink2")
        mock_pulse.source_list.return_value = [
            _create_mock_source(name="sink1.monitor", description="Monitor 1"),
            _create_mock_source(name="sink2.monitor", description="Monitor 2"),
        ]

        backend = PulseAudioBackend()
        devices = backend.list_loopback_devices()

        assert len(devices) == 2
        assert devices[0].device_id == "sink2.monitor"
        assert devices[1].device_id == "sink1.monitor"

    @patch("hark.audio_backends.pulseaudio._check_pulsectl_available", return_value=True)
    @patch("pulsectl.Pulse")
    def test_uses_name_when_no_description(self, mock_pulse_class: MagicMock, _: MagicMock) -> None:
        """Should use name when description is not available."""
        mock_pulse = MagicMock()
        mock_pulse_class.return_value.__enter__.return_value = mock_pulse

        mock_pulse.server_info.return_value = _create_mock_server_info()
        mock_pulse.source_list.return_value = [
            _create_mock_source(name="test.monitor", description=None),
        ]

        backend = PulseAudioBackend()
        devices = backend.list_loopback_devices()

        assert len(devices) == 1
        assert devices[0].name == "test.monitor"

    def test_get_recording_config_with_device_id(self) -> None:
        """Should return RecordingConfig with PULSE_SOURCE env var."""
        backend = PulseAudioBackend()
        config = backend.get_recording_config("alsa_output.analog-stereo.monitor")

        assert config.env == {"PULSE_SOURCE": "alsa_output.analog-stereo.monitor"}
        assert config.device == "pulse"

    def test_get_recording_config_with_none_device_id(self) -> None:
        """Should return RecordingConfig with empty env when device_id is None."""
        backend = PulseAudioBackend()
        config = backend.get_recording_config(None)

        assert config.env == {}
        assert config.device == "pulse"

    def test_get_recording_config_with_int_device_id(self) -> None:
        """Should convert int device_id to string for PULSE_SOURCE."""
        backend = PulseAudioBackend()
        config = backend.get_recording_config(42)

        assert config.env == {"PULSE_SOURCE": "42"}
        assert config.device == "pulse"


class TestPulseAudioBackendExtractedValues:
    """Tests for extracted channel count and sample rate (not hardcoded)."""

    @patch("hark.audio_backends.pulseaudio._check_pulsectl_available", return_value=True)
    @patch("pulsectl.Pulse")
    def test_extracts_actual_channel_count(self, mock_pulse_class: MagicMock, _: MagicMock) -> None:
        """Should extract actual channel count from source (not hardcoded)."""
        mock_pulse = MagicMock()
        mock_pulse_class.return_value.__enter__.return_value = mock_pulse

        mock_pulse.server_info.return_value = _create_mock_server_info()
        mock_pulse.source_list.return_value = [
            _create_mock_source(
                name="surround.monitor",
                description="5.1 Surround Monitor",
                channel_count=6,
                sample_rate=48000,
            ),
        ]

        backend = PulseAudioBackend()
        devices = backend.list_loopback_devices()

        assert len(devices) == 1
        assert devices[0].channels == 6

    @patch("hark.audio_backends.pulseaudio._check_pulsectl_available", return_value=True)
    @patch("pulsectl.Pulse")
    def test_extracts_actual_sample_rate(self, mock_pulse_class: MagicMock, _: MagicMock) -> None:
        """Should extract actual sample rate from source (not hardcoded)."""
        mock_pulse = MagicMock()
        mock_pulse_class.return_value.__enter__.return_value = mock_pulse

        mock_pulse.server_info.return_value = _create_mock_server_info()
        mock_pulse.source_list.return_value = [
            _create_mock_source(
                name="hifi.monitor",
                description="Hi-Fi Monitor",
                channel_count=2,
                sample_rate=96000,
            ),
        ]

        backend = PulseAudioBackend()
        devices = backend.list_loopback_devices()

        assert len(devices) == 1
        assert devices[0].sample_rate == 96000.0

    @patch("hark.audio_backends.pulseaudio._check_pulsectl_available", return_value=True)
    @patch("pulsectl.Pulse")
    def test_uses_default_channel_count_when_missing(
        self, mock_pulse_class: MagicMock, _: MagicMock
    ) -> None:
        """Should use default channel count when source doesn't provide it."""
        mock_pulse = MagicMock()
        mock_pulse_class.return_value.__enter__.return_value = mock_pulse

        mock_pulse.server_info.return_value = _create_mock_server_info()
        source = MagicMock()
        source.name = "test.monitor"
        source.description = "Test Monitor"
        del source.channel_count
        source.sample_spec = MagicMock(rate=48000)

        mock_pulse.source_list.return_value = [source]

        backend = PulseAudioBackend()
        devices = backend.list_loopback_devices()

        assert len(devices) == 1
        assert devices[0].channels == 2

    @patch("hark.audio_backends.pulseaudio._check_pulsectl_available", return_value=True)
    @patch("pulsectl.Pulse")
    def test_uses_default_sample_rate_when_missing(
        self, mock_pulse_class: MagicMock, _: MagicMock
    ) -> None:
        """Should use default sample rate when source doesn't provide it."""
        mock_pulse = MagicMock()
        mock_pulse_class.return_value.__enter__.return_value = mock_pulse

        mock_pulse.server_info.return_value = _create_mock_server_info()
        source = MagicMock()
        source.name = "test.monitor"
        source.description = "Test Monitor"
        source.channel_count = 2
        source.sample_spec = None

        mock_pulse.source_list.return_value = [source]

        backend = PulseAudioBackend()
        devices = backend.list_loopback_devices()

        assert len(devices) == 1
        assert devices[0].sample_rate == 44100.0


class TestPulseAudioBackendExports:
    """Tests for pulseaudio module exports."""

    def test_pulseaudio_exports(self) -> None:
        """pulseaudio module should export PulseAudioBackend."""
        from hark.audio_backends import pulseaudio

        assert "PulseAudioBackend" in pulseaudio.__all__
