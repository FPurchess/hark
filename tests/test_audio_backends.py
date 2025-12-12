"""Tests for hark.audio_backends module."""

from unittest.mock import MagicMock, patch

from hark.audio_backends import LoopbackBackend, LoopbackDeviceInfo, RecordingConfig
from hark.audio_backends.coreaudio import CoreAudioBackend
from hark.audio_backends.pulseaudio import PulseAudioBackend
from hark.audio_backends.wasapi import WASAPIBackend


class TestRecordingConfig:
    """Tests for RecordingConfig dataclass."""

    def test_create_with_env_and_device(self) -> None:
        """Should create with environment variables and device."""
        config = RecordingConfig(
            env={"PULSE_SOURCE": "monitor.source"},
            device="pulse",
        )
        assert config.env == {"PULSE_SOURCE": "monitor.source"}
        assert config.device == "pulse"

    def test_create_with_empty_env(self) -> None:
        """Should create with empty environment (Windows/macOS style)."""
        config = RecordingConfig(env={}, device=5)
        assert config.env == {}
        assert config.device == 5

    def test_create_with_none_device(self) -> None:
        """Should create with None device."""
        config = RecordingConfig(env={}, device=None)
        assert config.device is None

    def test_equality(self) -> None:
        """Two RecordingConfig with same values should be equal."""
        config1 = RecordingConfig(env={"KEY": "val"}, device="pulse")
        config2 = RecordingConfig(env={"KEY": "val"}, device="pulse")
        assert config1 == config2

    def test_inequality(self) -> None:
        """Two RecordingConfig with different values should not be equal."""
        config1 = RecordingConfig(env={"KEY": "val1"}, device="pulse")
        config2 = RecordingConfig(env={"KEY": "val2"}, device="pulse")
        assert config1 != config2


class TestLoopbackDeviceInfo:
    """Tests for LoopbackDeviceInfo dataclass."""

    def test_create_with_string_device_id(self) -> None:
        """Should create with string device ID (Linux PulseAudio style)."""
        info = LoopbackDeviceInfo(
            name="Monitor of Built-in Audio",
            device_id="alsa_output.pci-0000_00_1f.3.analog-stereo.monitor",
            channels=2,
            sample_rate=44100.0,
        )
        assert info.name == "Monitor of Built-in Audio"
        assert info.device_id == "alsa_output.pci-0000_00_1f.3.analog-stereo.monitor"
        assert info.channels == 2
        assert info.sample_rate == 44100.0

    def test_create_with_int_device_id(self) -> None:
        """Should create with integer device ID (Windows WASAPI style)."""
        info = LoopbackDeviceInfo(
            name="Speakers (Realtek Audio)",
            device_id=5,
            channels=2,
            sample_rate=48000.0,
        )
        assert info.name == "Speakers (Realtek Audio)"
        assert info.device_id == 5
        assert info.channels == 2
        assert info.sample_rate == 48000.0

    def test_create_with_none_device_id(self) -> None:
        """Should create with None device ID (fallback/virtual devices)."""
        info = LoopbackDeviceInfo(
            name="BlackHole 2ch",
            device_id=None,
            channels=2,
            sample_rate=44100.0,
        )
        assert info.device_id is None

    def test_equality(self) -> None:
        """Two LoopbackDeviceInfo with same values should be equal."""
        info1 = LoopbackDeviceInfo(name="Test", device_id="test", channels=2, sample_rate=44100.0)
        info2 = LoopbackDeviceInfo(name="Test", device_id="test", channels=2, sample_rate=44100.0)
        assert info1 == info2

    def test_inequality(self) -> None:
        """Two LoopbackDeviceInfo with different values should not be equal."""
        info1 = LoopbackDeviceInfo(name="Test1", device_id="test", channels=2, sample_rate=44100.0)
        info2 = LoopbackDeviceInfo(name="Test2", device_id="test", channels=2, sample_rate=44100.0)
        assert info1 != info2


class TestLoopbackBackendProtocol:
    """Tests for LoopbackBackend protocol."""

    def test_protocol_is_runtime_checkable(self) -> None:
        """LoopbackBackend should be runtime checkable."""

        # Creating a class that implements the protocol
        class MockBackend:
            def get_default_loopback(self) -> LoopbackDeviceInfo | None:
                return None

            def list_loopback_devices(self) -> list[LoopbackDeviceInfo]:
                return []

            def is_available(self) -> bool:
                return True

            def get_recording_config(self, device_id: str | int | None) -> RecordingConfig:
                return RecordingConfig(env={}, device=device_id)

        backend = MockBackend()
        assert isinstance(backend, LoopbackBackend)

    def test_protocol_rejects_incomplete_implementation(self) -> None:
        """Objects missing protocol methods should not match."""

        class IncompleteBackend:
            def get_default_loopback(self) -> LoopbackDeviceInfo | None:
                return None

            # Missing list_loopback_devices and is_available

        backend = IncompleteBackend()
        assert not isinstance(backend, LoopbackBackend)

    def test_mock_backend_functionality(self) -> None:
        """Test a mock backend implementing the protocol."""

        class MockBackend:
            def __init__(self, devices: list[LoopbackDeviceInfo]) -> None:
                self._devices = devices

            def get_default_loopback(self) -> LoopbackDeviceInfo | None:
                return self._devices[0] if self._devices else None

            def list_loopback_devices(self) -> list[LoopbackDeviceInfo]:
                return self._devices

            def is_available(self) -> bool:
                return True

            def get_recording_config(self, device_id: str | int | None) -> RecordingConfig:
                return RecordingConfig(env={}, device=device_id)

        devices = [
            LoopbackDeviceInfo(name="Device 1", device_id="dev1", channels=2, sample_rate=44100.0),
            LoopbackDeviceInfo(name="Device 2", device_id="dev2", channels=2, sample_rate=48000.0),
        ]

        backend = MockBackend(devices)

        assert isinstance(backend, LoopbackBackend)
        assert backend.is_available()
        assert backend.get_default_loopback() == devices[0]
        assert backend.list_loopback_devices() == devices

    def test_mock_backend_empty_devices(self) -> None:
        """Test mock backend with no devices."""

        class MockBackend:
            def get_default_loopback(self) -> LoopbackDeviceInfo | None:
                return None

            def list_loopback_devices(self) -> list[LoopbackDeviceInfo]:
                return []

            def is_available(self) -> bool:
                return False

            def get_recording_config(self, device_id: str | int | None) -> RecordingConfig:
                return RecordingConfig(env={}, device=device_id)

        backend = MockBackend()

        assert isinstance(backend, LoopbackBackend)
        assert not backend.is_available()
        assert backend.get_default_loopback() is None
        assert backend.list_loopback_devices() == []


class TestExports:
    """Tests for module exports."""

    def test_all_exports(self) -> None:
        """__all__ should include expected exports."""
        from hark import audio_backends

        expected = {
            "LoopbackBackend",
            "LoopbackDeviceInfo",
            "RecordingConfig",
            "get_loopback_backend",
        }
        assert set(audio_backends.__all__) == expected

    def test_base_exports(self) -> None:
        """base module should export protocol and dataclass."""
        from hark.audio_backends import base

        expected = {"LoopbackBackend", "LoopbackDeviceInfo", "RecordingConfig"}
        assert set(base.__all__) == expected


# Helper functions for creating pulsectl mock objects


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
    def test_is_available_success(self, mock_pulse_class: MagicMock, mock_check: MagicMock) -> None:
        """Should return True when pulsectl connects successfully."""
        mock_pulse = MagicMock()
        mock_pulse_class.return_value.__enter__.return_value = mock_pulse
        mock_pulse.server_info.return_value = _create_mock_server_info()

        backend = PulseAudioBackend()
        assert backend.is_available() is True

    @patch("hark.audio_backends.pulseaudio._check_pulsectl_available", return_value=True)
    @patch("pulsectl.Pulse")
    def test_is_available_connection_fails(
        self, mock_pulse_class: MagicMock, mock_check: MagicMock
    ) -> None:
        """Should return False when PulseAudio connection fails."""
        mock_pulse_class.return_value.__enter__.side_effect = Exception("Connection refused")

        backend = PulseAudioBackend()
        assert backend.is_available() is False

    @patch("hark.audio_backends.pulseaudio._check_pulsectl_available", return_value=False)
    def test_is_available_pulsectl_not_installed(self, mock_check: MagicMock) -> None:
        """Should return False when pulsectl is not installed."""
        backend = PulseAudioBackend()
        assert backend.is_available() is False

    @patch("hark.audio_backends.pulseaudio._check_pulsectl_available", return_value=True)
    @patch("pulsectl.Pulse")
    def test_get_default_loopback_success(
        self, mock_pulse_class: MagicMock, mock_check: MagicMock
    ) -> None:
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
        self, mock_pulse_class: MagicMock, mock_check: MagicMock
    ) -> None:
        """Should return None when no monitors are available."""
        mock_pulse = MagicMock()
        mock_pulse_class.return_value.__enter__.return_value = mock_pulse

        mock_pulse.server_info.return_value = _create_mock_server_info()
        # Only a microphone, no monitors
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
    def test_get_default_loopback_pulsectl_unavailable(self, mock_check: MagicMock) -> None:
        """Should return None when pulsectl is not available."""
        backend = PulseAudioBackend()
        device = backend.get_default_loopback()
        assert device is None

    @patch("hark.audio_backends.pulseaudio._check_pulsectl_available", return_value=True)
    @patch("pulsectl.Pulse")
    def test_get_default_loopback_falls_back_to_first_monitor(
        self, mock_pulse_class: MagicMock, mock_check: MagicMock
    ) -> None:
        """Should fall back to first monitor if default sink's monitor not found."""
        mock_pulse = MagicMock()
        mock_pulse_class.return_value.__enter__.return_value = mock_pulse

        # Default sink doesn't match any monitor
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
    def test_list_loopback_devices_success(
        self, mock_pulse_class: MagicMock, mock_check: MagicMock
    ) -> None:
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
    def test_list_loopback_devices_empty_when_unavailable(self, mock_check: MagicMock) -> None:
        """Should return empty list when pulsectl is not available."""
        backend = PulseAudioBackend()
        devices = backend.list_loopback_devices()
        assert devices == []

    @patch("hark.audio_backends.pulseaudio._check_pulsectl_available", return_value=True)
    @patch("pulsectl.Pulse")
    def test_monitors_sorted_by_default_sink(
        self, mock_pulse_class: MagicMock, mock_check: MagicMock
    ) -> None:
        """Should sort monitors with default sink's monitor first."""
        mock_pulse = MagicMock()
        mock_pulse_class.return_value.__enter__.return_value = mock_pulse

        # Default sink is sink2, so its monitor should be first
        mock_pulse.server_info.return_value = _create_mock_server_info(default_sink="sink2")
        mock_pulse.source_list.return_value = [
            _create_mock_source(name="sink1.monitor", description="Monitor 1"),
            _create_mock_source(name="sink2.monitor", description="Monitor 2"),
        ]

        backend = PulseAudioBackend()
        devices = backend.list_loopback_devices()

        assert len(devices) == 2
        # sink2.monitor should be first (default)
        assert devices[0].device_id == "sink2.monitor"
        assert devices[1].device_id == "sink1.monitor"

    @patch("hark.audio_backends.pulseaudio._check_pulsectl_available", return_value=True)
    @patch("pulsectl.Pulse")
    def test_uses_name_when_no_description(
        self, mock_pulse_class: MagicMock, mock_check: MagicMock
    ) -> None:
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
        # Should use name as fallback when no description
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
    def test_extracts_actual_channel_count(
        self, mock_pulse_class: MagicMock, mock_check: MagicMock
    ) -> None:
        """Should extract actual channel count from source (not hardcoded)."""
        mock_pulse = MagicMock()
        mock_pulse_class.return_value.__enter__.return_value = mock_pulse

        mock_pulse.server_info.return_value = _create_mock_server_info()
        # 5.1 surround sound has 6 channels
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
    def test_extracts_actual_sample_rate(
        self, mock_pulse_class: MagicMock, mock_check: MagicMock
    ) -> None:
        """Should extract actual sample rate from source (not hardcoded)."""
        mock_pulse = MagicMock()
        mock_pulse_class.return_value.__enter__.return_value = mock_pulse

        mock_pulse.server_info.return_value = _create_mock_server_info()
        # High-res audio at 96kHz
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
        self, mock_pulse_class: MagicMock, mock_check: MagicMock
    ) -> None:
        """Should use default channel count when source doesn't provide it."""
        mock_pulse = MagicMock()
        mock_pulse_class.return_value.__enter__.return_value = mock_pulse

        mock_pulse.server_info.return_value = _create_mock_server_info()
        # Source without channel_count attribute
        source = MagicMock()
        source.name = "test.monitor"
        source.description = "Test Monitor"
        # Explicitly delete channel_count to simulate missing attribute
        del source.channel_count
        source.sample_spec = MagicMock(rate=48000)

        mock_pulse.source_list.return_value = [source]

        backend = PulseAudioBackend()
        devices = backend.list_loopback_devices()

        assert len(devices) == 1
        # Should fall back to default (2)
        assert devices[0].channels == 2

    @patch("hark.audio_backends.pulseaudio._check_pulsectl_available", return_value=True)
    @patch("pulsectl.Pulse")
    def test_uses_default_sample_rate_when_missing(
        self, mock_pulse_class: MagicMock, mock_check: MagicMock
    ) -> None:
        """Should use default sample rate when source doesn't provide it."""
        mock_pulse = MagicMock()
        mock_pulse_class.return_value.__enter__.return_value = mock_pulse

        mock_pulse.server_info.return_value = _create_mock_server_info()
        # Source without sample_spec
        source = MagicMock()
        source.name = "test.monitor"
        source.description = "Test Monitor"
        source.channel_count = 2
        source.sample_spec = None

        mock_pulse.source_list.return_value = [source]

        backend = PulseAudioBackend()
        devices = backend.list_loopback_devices()

        assert len(devices) == 1
        # Should fall back to default (44100.0)
        assert devices[0].sample_rate == 44100.0


class TestPulseAudioBackendExports:
    """Tests for pulseaudio module exports."""

    def test_pulseaudio_exports(self) -> None:
        """pulseaudio module should export PulseAudioBackend."""
        from hark.audio_backends import pulseaudio

        assert "PulseAudioBackend" in pulseaudio.__all__


# Helper functions for creating sounddevice mock objects


def _create_mock_sd_device(
    name: str,
    max_input_channels: int = 2,
    max_output_channels: int = 2,
    default_samplerate: float = 48000.0,
) -> dict:
    """Create a mock sounddevice device dictionary."""
    return {
        "name": name,
        "max_input_channels": max_input_channels,
        "max_output_channels": max_output_channels,
        "default_samplerate": default_samplerate,
    }


class TestCoreAudioBackend:
    """Tests for CoreAudioBackend implementation."""

    def test_implements_loopback_backend(self) -> None:
        """CoreAudioBackend should implement LoopbackBackend protocol."""
        backend = CoreAudioBackend()
        assert isinstance(backend, LoopbackBackend)

    @patch("hark.audio_backends.coreaudio.is_macos", return_value=True)
    @patch("hark.audio_backends.coreaudio._check_sounddevice_available", return_value=True)
    @patch("hark.audio_backends.coreaudio.sd.query_devices")
    def test_is_available_success(
        self, mock_query: MagicMock, mock_check: MagicMock, mock_is_macos: MagicMock
    ) -> None:
        """Should return True on macOS with sounddevice working."""
        mock_query.return_value = [
            _create_mock_sd_device("Built-in Microphone"),
        ]

        backend = CoreAudioBackend()
        assert backend.is_available() is True

    @patch("hark.audio_backends.coreaudio._check_sounddevice_available", return_value=False)
    def test_is_available_sounddevice_unavailable(self, mock_check: MagicMock) -> None:
        """Should return False when sounddevice unavailable."""
        backend = CoreAudioBackend()
        assert backend.is_available() is False

    @patch("hark.audio_backends.coreaudio.is_macos", return_value=False)
    @patch("hark.audio_backends.coreaudio._check_sounddevice_available", return_value=True)
    @patch("hark.audio_backends.coreaudio.sd.query_devices")
    def test_is_available_wrong_platform(
        self, mock_query: MagicMock, mock_check: MagicMock, mock_is_macos: MagicMock
    ) -> None:
        """Should return False on non-macOS platforms."""
        mock_query.return_value = []
        backend = CoreAudioBackend()
        assert backend.is_available() is False

    @patch("hark.audio_backends.coreaudio._check_sounddevice_available", return_value=True)
    @patch("hark.audio_backends.coreaudio.sd.query_devices")
    def test_get_default_loopback_blackhole_2ch(
        self, mock_query: MagicMock, mock_check: MagicMock
    ) -> None:
        """Should detect BlackHole 2ch as default loopback."""
        mock_query.return_value = [
            _create_mock_sd_device("Built-in Microphone", max_input_channels=2),
            _create_mock_sd_device("BlackHole 2ch", max_input_channels=2),
        ]

        backend = CoreAudioBackend()
        device = backend.get_default_loopback()

        assert device is not None
        assert device.name == "BlackHole 2ch"
        assert device.device_id == 1
        assert device.channels == 2
        assert device.sample_rate == 48000.0

    @patch("hark.audio_backends.coreaudio._check_sounddevice_available", return_value=True)
    @patch("hark.audio_backends.coreaudio.sd.query_devices")
    def test_get_default_loopback_blackhole_16ch(
        self, mock_query: MagicMock, mock_check: MagicMock
    ) -> None:
        """Should detect BlackHole 16ch."""
        mock_query.return_value = [
            _create_mock_sd_device(
                "BlackHole 16ch", max_input_channels=16, default_samplerate=48000.0
            ),
        ]

        backend = CoreAudioBackend()
        device = backend.get_default_loopback()

        assert device is not None
        assert device.name == "BlackHole 16ch"
        assert device.channels == 16

    @patch("hark.audio_backends.coreaudio._check_sounddevice_available", return_value=True)
    @patch("hark.audio_backends.coreaudio.sd.query_devices")
    def test_get_default_loopback_none_found(
        self, mock_query: MagicMock, mock_check: MagicMock
    ) -> None:
        """Should return None when no BlackHole devices found."""
        mock_query.return_value = [
            _create_mock_sd_device("Built-in Microphone", max_input_channels=2),
            _create_mock_sd_device("USB Audio Device", max_input_channels=2),
        ]

        backend = CoreAudioBackend()
        device = backend.get_default_loopback()
        assert device is None

    @patch("hark.audio_backends.coreaudio._check_sounddevice_available", return_value=False)
    def test_get_default_loopback_unavailable(self, mock_check: MagicMock) -> None:
        """Should return None when sounddevice unavailable."""
        backend = CoreAudioBackend()
        assert backend.get_default_loopback() is None

    @patch("hark.audio_backends.coreaudio._check_sounddevice_available", return_value=True)
    @patch("hark.audio_backends.coreaudio.sd.query_devices")
    def test_list_loopback_devices_multiple_blackhole(
        self, mock_query: MagicMock, mock_check: MagicMock
    ) -> None:
        """Should list all BlackHole devices sorted alphabetically."""
        mock_query.return_value = [
            _create_mock_sd_device("Built-in Microphone", max_input_channels=2),
            _create_mock_sd_device("BlackHole 16ch", max_input_channels=16),
            _create_mock_sd_device("BlackHole 2ch", max_input_channels=2),
        ]

        backend = CoreAudioBackend()
        devices = backend.list_loopback_devices()

        assert len(devices) == 2
        # Should be sorted alphabetically
        assert devices[0].name == "BlackHole 16ch"
        assert devices[1].name == "BlackHole 2ch"

    @patch("hark.audio_backends.coreaudio._check_sounddevice_available", return_value=True)
    @patch("hark.audio_backends.coreaudio.sd.query_devices")
    def test_list_loopback_devices_empty_when_no_blackhole(
        self, mock_query: MagicMock, mock_check: MagicMock
    ) -> None:
        """Should return empty list when no BlackHole devices."""
        mock_query.return_value = [
            _create_mock_sd_device("Built-in Microphone", max_input_channels=2),
        ]

        backend = CoreAudioBackend()
        devices = backend.list_loopback_devices()
        assert devices == []

    @patch("hark.audio_backends.coreaudio._check_sounddevice_available", return_value=False)
    def test_list_loopback_devices_unavailable(self, mock_check: MagicMock) -> None:
        """Should return empty list when sounddevice unavailable."""
        backend = CoreAudioBackend()
        devices = backend.list_loopback_devices()
        assert devices == []

    @patch("hark.audio_backends.coreaudio._check_sounddevice_available", return_value=True)
    @patch("hark.audio_backends.coreaudio.sd.query_devices")
    def test_skips_output_only_devices(self, mock_query: MagicMock, mock_check: MagicMock) -> None:
        """Should skip devices with no input channels."""
        mock_query.return_value = [
            _create_mock_sd_device(
                "BlackHole 2ch Output",
                max_input_channels=0,
                max_output_channels=2,
            ),
            _create_mock_sd_device(
                "BlackHole 16ch",
                max_input_channels=16,
                max_output_channels=16,
            ),
        ]

        backend = CoreAudioBackend()
        devices = backend.list_loopback_devices()

        assert len(devices) == 1
        assert devices[0].name == "BlackHole 16ch"

    def test_get_recording_config_with_int_device_id(self) -> None:
        """Should return RecordingConfig with empty env and device index."""
        backend = CoreAudioBackend()
        config = backend.get_recording_config(5)

        assert config.env == {}
        assert config.device == 5

    def test_get_recording_config_with_none_device_id(self) -> None:
        """Should return RecordingConfig with None device when device_id is None."""
        backend = CoreAudioBackend()
        config = backend.get_recording_config(None)

        assert config.env == {}
        assert config.device is None

    def test_get_recording_config_with_string_device_id(self) -> None:
        """Should return None device for non-integer device_id."""
        backend = CoreAudioBackend()
        config = backend.get_recording_config("some_string")

        assert config.env == {}
        assert config.device is None

    def test_is_blackhole_case_insensitive(self) -> None:
        """Device detection should be case-insensitive."""
        backend = CoreAudioBackend()
        assert backend._is_blackhole("BLACKHOLE 2CH") is True
        assert backend._is_blackhole("BlackHole 2ch") is True
        assert backend._is_blackhole("blackhole 2ch") is True
        assert backend._is_blackhole("Blackhole16ch") is True

    def test_is_blackhole_variants(self) -> None:
        """Should detect all BlackHole channel variants."""
        backend = CoreAudioBackend()
        assert backend._is_blackhole("BlackHole 2ch") is True
        assert backend._is_blackhole("BlackHole 16ch") is True
        assert backend._is_blackhole("BlackHole 64ch") is True
        assert backend._is_blackhole("BlackHole") is True

    def test_is_blackhole_non_blackhole_devices_rejected(self) -> None:
        """Should not detect regular audio devices as BlackHole."""
        backend = CoreAudioBackend()
        assert backend._is_blackhole("Built-in Microphone") is False
        assert backend._is_blackhole("MacBook Pro Microphone") is False
        assert backend._is_blackhole("USB Audio Device") is False
        assert backend._is_blackhole("AirPods Pro") is False
        assert backend._is_blackhole("Soundflower (2ch)") is False
        assert backend._is_blackhole("Loopback Audio") is False


class TestCoreAudioBackendExports:
    """Tests for coreaudio module exports."""

    def test_coreaudio_exports(self) -> None:
        """coreaudio module should export CoreAudioBackend."""
        from hark.audio_backends import coreaudio

        assert "CoreAudioBackend" in coreaudio.__all__


# Helper functions for creating PyAudioWPatch mock objects


def _create_mock_wasapi_device(
    index: int,
    name: str,
    max_input_channels: int = 2,
    default_sample_rate: float = 48000.0,
    is_loopback: bool = True,
) -> dict:
    """Create a mock PyAudioWPatch device dictionary."""
    return {
        "index": index,
        "name": name,
        "maxInputChannels": max_input_channels,
        "defaultSampleRate": default_sample_rate,
        "isLoopbackDevice": is_loopback,
    }


class TestWASAPIBackend:
    """Tests for WASAPIBackend implementation."""

    def test_implements_loopback_backend(self) -> None:
        """WASAPIBackend should implement LoopbackBackend protocol."""
        backend = WASAPIBackend()
        assert isinstance(backend, LoopbackBackend)

    def test_is_available_success(self) -> None:
        """Should return True when PyAudioWPatch works."""
        import sys

        # Create mock pyaudiowpatch module
        mock_pyaudio_module = MagicMock()
        mock_pyaudio_instance = MagicMock()
        mock_pyaudio_module.PyAudio.return_value.__enter__.return_value = mock_pyaudio_instance
        mock_pyaudio_instance.get_default_wasapi_loopback.return_value = _create_mock_wasapi_device(
            0, "Speakers [Loopback]"
        )

        with (
            patch.dict(sys.modules, {"pyaudiowpatch": mock_pyaudio_module}),
            patch("hark.audio_backends.wasapi._check_pyaudiowpatch_available", return_value=True),
        ):
            backend = WASAPIBackend()
            assert backend.is_available() is True

    @patch("hark.audio_backends.wasapi._check_pyaudiowpatch_available", return_value=False)
    def test_is_available_pyaudiowpatch_not_installed(self, mock_check: MagicMock) -> None:
        """Should return False when PyAudioWPatch is not installed."""
        backend = WASAPIBackend()
        assert backend.is_available() is False

    def test_is_available_no_loopback_device(self) -> None:
        """Should return False when no WASAPI loopback device available."""
        import sys

        # Create mock pyaudiowpatch module
        mock_pyaudio_module = MagicMock()
        mock_pyaudio_instance = MagicMock()
        mock_pyaudio_module.PyAudio.return_value.__enter__.return_value = mock_pyaudio_instance
        mock_pyaudio_instance.get_default_wasapi_loopback.side_effect = OSError(
            "No loopback device"
        )

        with (
            patch.dict(sys.modules, {"pyaudiowpatch": mock_pyaudio_module}),
            patch("hark.audio_backends.wasapi._check_pyaudiowpatch_available", return_value=True),
        ):
            backend = WASAPIBackend()
            assert backend.is_available() is False

    def test_get_default_loopback_success(self) -> None:
        """Should return loopback device info when available."""
        import sys

        # Create mock pyaudiowpatch module
        mock_pyaudio_module = MagicMock()
        mock_pyaudio_instance = MagicMock()
        mock_pyaudio_module.PyAudio.return_value.__enter__.return_value = mock_pyaudio_instance
        mock_pyaudio_instance.get_default_wasapi_loopback.return_value = _create_mock_wasapi_device(
            index=3,
            name="Speakers (Realtek High Definition Audio) [Loopback]",
            max_input_channels=2,
            default_sample_rate=48000.0,
        )

        with (
            patch.dict(sys.modules, {"pyaudiowpatch": mock_pyaudio_module}),
            patch("hark.audio_backends.wasapi._check_pyaudiowpatch_available", return_value=True),
        ):
            backend = WASAPIBackend()
            device = backend.get_default_loopback()

            assert device is not None
            assert device.name == "Speakers (Realtek High Definition Audio) [Loopback]"
            assert device.device_id == 3
            assert device.channels == 2
            assert device.sample_rate == 48000.0

    def test_get_default_loopback_returns_none_on_error(self) -> None:
        """Should return None when no loopback available."""
        import sys

        # Create mock pyaudiowpatch module
        mock_pyaudio_module = MagicMock()
        mock_pyaudio_instance = MagicMock()
        mock_pyaudio_module.PyAudio.return_value.__enter__.return_value = mock_pyaudio_instance
        mock_pyaudio_instance.get_default_wasapi_loopback.side_effect = OSError("No device")

        with (
            patch.dict(sys.modules, {"pyaudiowpatch": mock_pyaudio_module}),
            patch("hark.audio_backends.wasapi._check_pyaudiowpatch_available", return_value=True),
        ):
            backend = WASAPIBackend()
            device = backend.get_default_loopback()
            assert device is None

    @patch("hark.audio_backends.wasapi._check_pyaudiowpatch_available", return_value=False)
    def test_get_default_loopback_unavailable(self, mock_check: MagicMock) -> None:
        """Should return None when PyAudioWPatch unavailable."""
        backend = WASAPIBackend()
        device = backend.get_default_loopback()
        assert device is None

    def test_list_loopback_devices_success(self) -> None:
        """Should list all WASAPI loopback devices."""
        import sys

        mock_devices = [
            _create_mock_wasapi_device(0, "Speakers [Loopback]"),
            _create_mock_wasapi_device(1, "Headphones [Loopback]"),
        ]

        # Create mock pyaudiowpatch module
        mock_pyaudio_module = MagicMock()
        mock_pyaudio_instance = MagicMock()
        mock_pyaudio_module.PyAudio.return_value.__enter__.return_value = mock_pyaudio_instance
        mock_pyaudio_instance.get_loopback_device_info_generator.return_value = iter(mock_devices)
        mock_pyaudio_instance.get_default_wasapi_loopback.return_value = mock_devices[0]

        with (
            patch.dict(sys.modules, {"pyaudiowpatch": mock_pyaudio_module}),
            patch("hark.audio_backends.wasapi._check_pyaudiowpatch_available", return_value=True),
        ):
            backend = WASAPIBackend()
            devices = backend.list_loopback_devices()

            assert len(devices) == 2
            assert devices[0].name == "Speakers [Loopback]"
            assert devices[1].name == "Headphones [Loopback]"

    def test_list_loopback_devices_sorted_by_default(self) -> None:
        """Should sort with default loopback first."""
        import sys

        mock_devices = [
            _create_mock_wasapi_device(0, "Speakers [Loopback]"),
            _create_mock_wasapi_device(1, "Headphones [Loopback]"),  # Default
        ]

        # Create mock pyaudiowpatch module
        mock_pyaudio_module = MagicMock()
        mock_pyaudio_instance = MagicMock()
        mock_pyaudio_module.PyAudio.return_value.__enter__.return_value = mock_pyaudio_instance
        mock_pyaudio_instance.get_loopback_device_info_generator.return_value = iter(mock_devices)
        mock_pyaudio_instance.get_default_wasapi_loopback.return_value = mock_devices[1]

        with (
            patch.dict(sys.modules, {"pyaudiowpatch": mock_pyaudio_module}),
            patch("hark.audio_backends.wasapi._check_pyaudiowpatch_available", return_value=True),
        ):
            backend = WASAPIBackend()
            devices = backend.list_loopback_devices()

            assert len(devices) == 2
            # Headphones should be first (it's the default)
            assert devices[0].device_id == 1
            assert devices[1].device_id == 0

    @patch("hark.audio_backends.wasapi._check_pyaudiowpatch_available", return_value=False)
    def test_list_loopback_devices_unavailable(self, mock_check: MagicMock) -> None:
        """Should return empty list when PyAudioWPatch unavailable."""
        backend = WASAPIBackend()
        devices = backend.list_loopback_devices()
        assert devices == []

    def test_list_loopback_devices_empty_when_no_devices(self) -> None:
        """Should return empty list when no loopback devices found."""
        import sys

        # Create mock pyaudiowpatch module
        mock_pyaudio_module = MagicMock()
        mock_pyaudio_instance = MagicMock()
        mock_pyaudio_module.PyAudio.return_value.__enter__.return_value = mock_pyaudio_instance
        mock_pyaudio_instance.get_loopback_device_info_generator.return_value = iter([])
        mock_pyaudio_instance.get_default_wasapi_loopback.side_effect = OSError("No default")

        with (
            patch.dict(sys.modules, {"pyaudiowpatch": mock_pyaudio_module}),
            patch("hark.audio_backends.wasapi._check_pyaudiowpatch_available", return_value=True),
        ):
            backend = WASAPIBackend()
            devices = backend.list_loopback_devices()
            assert devices == []

    def test_get_recording_config_with_device_id(self) -> None:
        """Should return RecordingConfig with wasapi marker."""
        backend = WASAPIBackend()
        config = backend.get_recording_config(5)

        assert config.env == {}
        assert config.device == "wasapi:5"

    def test_get_recording_config_with_none_device_id(self) -> None:
        """Should return RecordingConfig with wasapi marker (no index)."""
        backend = WASAPIBackend()
        config = backend.get_recording_config(None)

        assert config.env == {}
        assert config.device == "wasapi"

    def test_get_recording_config_with_string_device_id(self) -> None:
        """Should handle string device_id in wasapi marker."""
        backend = WASAPIBackend()
        config = backend.get_recording_config("some_string")

        assert config.env == {}
        assert config.device == "wasapi:some_string"


class TestWASAPIBackendExports:
    """Tests for wasapi module exports."""

    def test_wasapi_exports(self) -> None:
        """wasapi module should export WASAPIBackend."""
        from hark.audio_backends import wasapi

        assert "WASAPIBackend" in wasapi.__all__


class TestGetLoopbackBackend:
    """Tests for get_loopback_backend platform dispatch function."""

    def test_returns_pulseaudio_on_linux(self) -> None:
        """Should return PulseAudioBackend on Linux."""
        with patch("hark.audio_backends.is_linux", return_value=True):
            from hark.audio_backends import get_loopback_backend

            backend = get_loopback_backend()
            assert isinstance(backend, PulseAudioBackend)
            assert isinstance(backend, LoopbackBackend)

    def test_returns_coreaudio_on_macos(self) -> None:
        """Should return CoreAudioBackend on macOS."""
        with (
            patch("hark.audio_backends.is_linux", return_value=False),
            patch("hark.audio_backends.is_macos", return_value=True),
        ):
            from hark.audio_backends import get_loopback_backend

            backend = get_loopback_backend()
            assert isinstance(backend, CoreAudioBackend)
            assert isinstance(backend, LoopbackBackend)

    def test_returns_wasapi_on_windows(self) -> None:
        """Should return WASAPIBackend on Windows."""
        with (
            patch("hark.audio_backends.is_linux", return_value=False),
            patch("hark.audio_backends.is_macos", return_value=False),
            patch("hark.audio_backends.is_windows", return_value=True),
        ):
            from hark.audio_backends import get_loopback_backend

            backend = get_loopback_backend()
            assert isinstance(backend, WASAPIBackend)
            assert isinstance(backend, LoopbackBackend)

    def test_raises_on_unsupported_platform(self) -> None:
        """Should raise NotImplementedError on unsupported platforms."""
        import pytest

        with (
            patch("hark.audio_backends.is_linux", return_value=False),
            patch("hark.audio_backends.is_macos", return_value=False),
            patch("hark.audio_backends.is_windows", return_value=False),
        ):
            from hark.audio_backends import get_loopback_backend

            with pytest.raises(NotImplementedError) as exc_info:
                get_loopback_backend()

            assert "not yet supported" in str(exc_info.value)

    def test_get_loopback_backend_in_exports(self) -> None:
        """get_loopback_backend should be in module exports."""
        from hark import audio_backends

        assert "get_loopback_backend" in audio_backends.__all__
