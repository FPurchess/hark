"""Tests for hark.audio_sources module."""

from unittest.mock import MagicMock, patch

import pytest

from hark.audio_backends import LoopbackDeviceInfo, RecordingConfig
from hark.audio_sources import (
    AudioSourceInfo,
    InputSource,
    _get_loopback_backend,
    find_loopback_device,
    find_microphone_device,
    get_devices_for_source,
    list_loopback_devices,
    validate_source_availability,
)


@pytest.fixture(autouse=True)
def clear_loopback_backend_cache():
    """Clear the cached loopback backend before each test."""
    _get_loopback_backend.cache_clear()
    yield
    _get_loopback_backend.cache_clear()


def _create_mock_loopback_info(
    name: str = "Monitor of Test Device",
    device_id: str = "alsa_output.test.monitor",
    channels: int = 2,
    sample_rate: float = 48000.0,
) -> LoopbackDeviceInfo:
    """Create a mock LoopbackDeviceInfo for testing."""
    return LoopbackDeviceInfo(
        name=name,
        device_id=device_id,
        channels=channels,
        sample_rate=sample_rate,
    )


class TestInputSourceEnum:
    """Tests for InputSource enum."""

    def test_mic_value(self) -> None:
        """MIC should have value 'mic'."""
        assert InputSource.MIC.value == "mic"

    def test_speaker_value(self) -> None:
        """SPEAKER should have value 'speaker'."""
        assert InputSource.SPEAKER.value == "speaker"

    def test_both_value(self) -> None:
        """BOTH should have value 'both'."""
        assert InputSource.BOTH.value == "both"

    def test_from_string(self) -> None:
        """Should be constructable from string."""
        assert InputSource("mic") == InputSource.MIC
        assert InputSource("speaker") == InputSource.SPEAKER
        assert InputSource("both") == InputSource.BOTH


class TestAudioSourceInfo:
    """Tests for AudioSourceInfo dataclass."""

    def test_mic_source_info(self) -> None:
        """Should store microphone source info."""
        info = AudioSourceInfo(
            device_index=0,
            name="Test Microphone",
            channels=2,
            sample_rate=44100.0,
            is_loopback=False,
        )
        assert info.device_index == 0
        assert info.name == "Test Microphone"
        assert info.channels == 2
        assert info.sample_rate == 44100.0
        assert info.is_loopback is False
        assert info.recording_config is None

    def test_loopback_source_info(self) -> None:
        """Should store loopback source info with recording_config."""
        config = RecordingConfig(
            env={"PULSE_SOURCE": "alsa_output.analog-stereo.monitor"},
            device="pulse",
        )
        info = AudioSourceInfo(
            device_index=None,
            name="Monitor of Built-in Audio",
            channels=2,
            sample_rate=44100.0,
            is_loopback=True,
            recording_config=config,
        )
        assert info.device_index is None
        assert info.is_loopback is True
        assert info.recording_config is not None
        assert info.recording_config.env == {"PULSE_SOURCE": "alsa_output.analog-stereo.monitor"}
        assert info.recording_config.device == "pulse"


class TestFindMicrophoneDevice:
    """Tests for find_microphone_device function."""

    def test_returns_default_device(self) -> None:
        """Should return default input device."""
        mock_device = {
            "name": "Default Microphone",
            "max_input_channels": 2,
            "default_samplerate": 44100.0,
        }

        with (
            patch("hark.audio_sources.sd.default.device", [0, 1]),
            patch("hark.audio_sources.sd.query_devices", return_value=mock_device),
        ):
            device = find_microphone_device()

            assert device is not None
            assert device.name == "Default Microphone"
            assert device.device_index == 0
            assert device.is_loopback is False

    def test_returns_none_when_no_default(self) -> None:
        """Should return None when no default device."""
        with patch("hark.audio_sources.sd.default") as mock_default:
            mock_default.device = [None, None]
            device = find_microphone_device()
            assert device is None

    def test_returns_none_on_exception(self) -> None:
        """Should return None on exception."""
        with patch("hark.audio_sources.sd.default") as mock_default:
            mock_default.device = [0, 1]
            with patch("hark.audio_sources.sd.query_devices", side_effect=Exception("Error")):
                device = find_microphone_device()
                assert device is None

    def test_returns_none_for_output_only_device(self) -> None:
        """Should return None if default device has no input channels."""
        mock_device = {
            "name": "Speaker",
            "max_input_channels": 0,
            "default_samplerate": 44100.0,
        }

        with (
            patch("hark.audio_sources.sd.default.device", [0, 1]),
            patch("hark.audio_sources.sd.query_devices", return_value=mock_device),
        ):
            device = find_microphone_device()
            assert device is None


class TestFindLoopbackDevice:
    """Tests for find_loopback_device function."""

    def test_finds_pulseaudio_monitor(self) -> None:
        """Should find PulseAudio monitor via backend."""
        mock_loopback = _create_mock_loopback_info(
            name="Monitor of Built-in Audio",
            device_id="alsa_output.pci.analog-stereo.monitor",
        )
        mock_backend = MagicMock()
        mock_backend.get_default_loopback.return_value = mock_loopback
        mock_backend.get_recording_config.return_value = RecordingConfig(
            env={"PULSE_SOURCE": "alsa_output.pci.analog-stereo.monitor"},
            device="pulse",
        )

        with patch("hark.audio_sources.get_loopback_backend", return_value=mock_backend):
            device = find_loopback_device()

            assert device is not None
            assert device.is_loopback is True
            assert device.recording_config is not None
            assert device.recording_config.env == {
                "PULSE_SOURCE": "alsa_output.pci.analog-stereo.monitor"
            }
            assert device.recording_config.device == "pulse"
            assert "Monitor" in device.name

    def test_falls_back_to_sounddevice(self) -> None:
        """Should fall back to sounddevice when backend returns nothing."""
        mock_backend = MagicMock()
        mock_backend.get_default_loopback.return_value = None

        mock_devices = [
            {
                "name": "USB Microphone",
                "max_input_channels": 2,
                "default_samplerate": 44100.0,
            },
            {
                "name": "Stereo Mix",
                "max_input_channels": 2,
                "default_samplerate": 44100.0,
            },
        ]

        with (
            patch("hark.audio_sources.get_loopback_backend", return_value=mock_backend),
            patch("hark.audio_sources.sd.query_devices", return_value=mock_devices),
        ):
            device = find_loopback_device()

            assert device is not None
            assert device.name == "Stereo Mix"
            assert device.device_index == 1
            assert device.is_loopback is True

    def test_returns_none_when_no_loopback(self) -> None:
        """Should return None when no loopback device found."""
        mock_backend = MagicMock()
        mock_backend.get_default_loopback.return_value = None

        mock_devices = [
            {
                "name": "USB Microphone",
                "max_input_channels": 2,
                "default_samplerate": 44100.0,
            },
        ]

        with (
            patch("hark.audio_sources.get_loopback_backend", return_value=mock_backend),
            patch("hark.audio_sources.sd.query_devices", return_value=mock_devices),
        ):
            device = find_loopback_device()
            assert device is None

    def test_handles_backend_not_available(self) -> None:
        """Should handle backend not available gracefully."""
        mock_devices = [
            {
                "name": "Loopback Device",
                "max_input_channels": 2,
                "default_samplerate": 44100.0,
            },
        ]

        with (
            patch(
                "hark.audio_sources.get_loopback_backend",
                side_effect=NotImplementedError,
            ),
            patch("hark.audio_sources.sd.query_devices", return_value=mock_devices),
        ):
            device = find_loopback_device()
            assert device is not None
            assert device.name == "Loopback Device"

    def test_handles_backend_exception(self) -> None:
        """Should handle backend exceptions gracefully."""
        mock_backend = MagicMock()
        mock_backend.get_default_loopback.side_effect = Exception("Backend error")

        mock_devices = [
            {
                "name": "What U Hear",
                "max_input_channels": 2,
                "default_samplerate": 44100.0,
            },
        ]

        with (
            patch("hark.audio_sources.get_loopback_backend", return_value=mock_backend),
            patch("hark.audio_sources.sd.query_devices", return_value=mock_devices),
        ):
            device = find_loopback_device()
            assert device is not None
            assert device.name == "What U Hear"

    def test_prefers_default_sink_monitor(self) -> None:
        """Should prefer the monitor for the default sink (via backend)."""
        # Backend returns the correct default sink's monitor
        mock_loopback = _create_mock_loopback_info(
            name="Monitor of USB Device B",
            device_id="alsa_output.usb-device-B.analog-stereo.monitor",
        )
        mock_backend = MagicMock()
        mock_backend.get_default_loopback.return_value = mock_loopback
        mock_backend.get_recording_config.return_value = RecordingConfig(
            env={"PULSE_SOURCE": "alsa_output.usb-device-B.analog-stereo.monitor"},
            device="pulse",
        )

        with patch("hark.audio_sources.get_loopback_backend", return_value=mock_backend):
            device = find_loopback_device()

            assert device is not None
            assert device.recording_config is not None
            assert device.recording_config.env == {
                "PULSE_SOURCE": "alsa_output.usb-device-B.analog-stereo.monitor"
            }
            assert "Device B" in device.name

    def test_falls_back_to_first_monitor_when_no_default_sink(self) -> None:
        """Should fall back to first monitor when default sink can't be determined."""
        # Backend returns first available monitor
        mock_loopback = _create_mock_loopback_info(
            name="Monitor of USB Device A",
            device_id="alsa_output.usb-device-A.analog-stereo.monitor",
        )
        mock_backend = MagicMock()
        mock_backend.get_default_loopback.return_value = mock_loopback
        mock_backend.get_recording_config.return_value = RecordingConfig(
            env={"PULSE_SOURCE": "alsa_output.usb-device-A.analog-stereo.monitor"},
            device="pulse",
        )

        with patch("hark.audio_sources.get_loopback_backend", return_value=mock_backend):
            device = find_loopback_device()

            assert device is not None
            assert device.recording_config is not None
            assert device.recording_config.env == {
                "PULSE_SOURCE": "alsa_output.usb-device-A.analog-stereo.monitor"
            }


class TestListLoopbackDevices:
    """Tests for list_loopback_devices function."""

    def test_lists_all_monitors(self) -> None:
        """Should list all available monitor devices."""
        mock_devices = [
            _create_mock_loopback_info(
                name="Monitor of USB Audio",
                device_id="alsa_output.usb-device.analog-stereo.monitor",
            ),
            _create_mock_loopback_info(
                name="Monitor of HDMI Audio",
                device_id="alsa_output.pci.hdmi-stereo.monitor",
            ),
        ]
        mock_backend = MagicMock()
        mock_backend.list_loopback_devices.return_value = mock_devices
        mock_backend.get_recording_config.side_effect = lambda device_id: RecordingConfig(
            env={"PULSE_SOURCE": device_id},
            device="pulse",
        )

        with (
            patch("hark.audio_sources.get_loopback_backend", return_value=mock_backend),
            patch("hark.audio_sources.sd.query_devices", return_value=[]),
        ):
            devices = list_loopback_devices()

            assert len(devices) == 2
            assert all(d.is_loopback for d in devices)

    def test_empty_when_no_monitors(self) -> None:
        """Should return empty list when no monitors."""
        mock_backend = MagicMock()
        mock_backend.list_loopback_devices.return_value = []

        with (
            patch("hark.audio_sources.get_loopback_backend", return_value=mock_backend),
            patch("hark.audio_sources.sd.query_devices", return_value=[]),
        ):
            devices = list_loopback_devices()
            assert devices == []


class TestGetDevicesForSource:
    """Tests for get_devices_for_source function."""

    def test_mic_mode_returns_mic_only(self) -> None:
        """MIC mode should return only mic device."""
        mock_mic = AudioSourceInfo(0, "Mic", 1, 16000, False)

        with (
            patch("hark.audio_sources.find_microphone_device", return_value=mock_mic),
            patch("hark.audio_sources.find_loopback_device", return_value=None),
        ):
            mic, loopback = get_devices_for_source(InputSource.MIC)

            assert mic is mock_mic
            assert loopback is None

    def test_speaker_mode_returns_loopback_only(self) -> None:
        """SPEAKER mode should return only loopback device."""
        mock_loopback = AudioSourceInfo(None, "Monitor", 2, 44100, True)

        with (
            patch("hark.audio_sources.find_microphone_device", return_value=None),
            patch("hark.audio_sources.find_loopback_device", return_value=mock_loopback),
        ):
            mic, loopback = get_devices_for_source(InputSource.SPEAKER)

            assert mic is None
            assert loopback is mock_loopback

    def test_both_mode_returns_both(self) -> None:
        """BOTH mode should return both devices."""
        mock_mic = AudioSourceInfo(0, "Mic", 1, 16000, False)
        mock_loopback = AudioSourceInfo(None, "Monitor", 2, 44100, True)

        with (
            patch("hark.audio_sources.find_microphone_device", return_value=mock_mic),
            patch("hark.audio_sources.find_loopback_device", return_value=mock_loopback),
        ):
            mic, loopback = get_devices_for_source(InputSource.BOTH)

            assert mic is mock_mic
            assert loopback is mock_loopback


class TestValidateSourceAvailability:
    """Tests for validate_source_availability function."""

    def test_mic_mode_valid(self) -> None:
        """MIC mode with available mic should have no errors."""
        mock_mic = AudioSourceInfo(0, "Mic", 1, 16000, False)

        with patch("hark.audio_sources.find_microphone_device", return_value=mock_mic):
            errors = validate_source_availability(InputSource.MIC)
            assert errors == []

    def test_mic_mode_no_mic(self) -> None:
        """MIC mode without mic should have error."""
        with patch("hark.audio_sources.find_microphone_device", return_value=None):
            errors = validate_source_availability(InputSource.MIC)
            assert len(errors) == 1
            assert "microphone" in errors[0].lower()

    def test_speaker_mode_valid(self) -> None:
        """SPEAKER mode with available loopback should have no errors."""
        mock_loopback = AudioSourceInfo(None, "Monitor", 2, 44100, True)

        with patch("hark.audio_sources.find_loopback_device", return_value=mock_loopback):
            errors = validate_source_availability(InputSource.SPEAKER)
            assert errors == []

    def test_speaker_mode_no_loopback(self) -> None:
        """SPEAKER mode without loopback should have error."""
        with patch("hark.audio_sources.find_loopback_device", return_value=None):
            errors = validate_source_availability(InputSource.SPEAKER)
            assert len(errors) == 1
            assert "loopback" in errors[0].lower()

    def test_both_mode_valid(self) -> None:
        """BOTH mode with both devices should have no errors."""
        mock_mic = AudioSourceInfo(0, "Mic", 1, 16000, False)
        mock_loopback = AudioSourceInfo(None, "Monitor", 2, 44100, True)

        with (
            patch("hark.audio_sources.find_microphone_device", return_value=mock_mic),
            patch("hark.audio_sources.find_loopback_device", return_value=mock_loopback),
        ):
            errors = validate_source_availability(InputSource.BOTH)
            assert errors == []

    def test_both_mode_missing_mic(self) -> None:
        """BOTH mode without mic should have error."""
        mock_loopback = AudioSourceInfo(None, "Monitor", 2, 44100, True)

        with (
            patch("hark.audio_sources.find_microphone_device", return_value=None),
            patch("hark.audio_sources.find_loopback_device", return_value=mock_loopback),
        ):
            errors = validate_source_availability(InputSource.BOTH)
            assert len(errors) == 1
            assert "microphone" in errors[0].lower()

    def test_both_mode_missing_loopback(self) -> None:
        """BOTH mode without loopback should have error."""
        mock_mic = AudioSourceInfo(0, "Mic", 1, 16000, False)

        with (
            patch("hark.audio_sources.find_microphone_device", return_value=mock_mic),
            patch("hark.audio_sources.find_loopback_device", return_value=None),
        ):
            errors = validate_source_availability(InputSource.BOTH)
            assert len(errors) == 1
            assert "loopback" in errors[0].lower()

    def test_both_mode_missing_both(self) -> None:
        """BOTH mode without either device should have two errors."""
        with (
            patch("hark.audio_sources.find_microphone_device", return_value=None),
            patch("hark.audio_sources.find_loopback_device", return_value=None),
        ):
            errors = validate_source_availability(InputSource.BOTH)
            assert len(errors) == 2

    def test_loopback_error_message_linux(self) -> None:
        """Error message should mention PulseAudio on Linux."""
        with (
            patch("hark.audio_sources.find_loopback_device", return_value=None),
            patch("hark.audio_sources.is_linux", return_value=True),
            patch("hark.audio_sources.is_macos", return_value=False),
            patch("hark.audio_sources.is_windows", return_value=False),
        ):
            errors = validate_source_availability(InputSource.SPEAKER)
            assert len(errors) == 1
            assert "PulseAudio" in errors[0] or "PipeWire" in errors[0]

    def test_loopback_error_message_macos(self) -> None:
        """Error message should mention BlackHole on macOS."""
        with (
            patch("hark.audio_sources.find_loopback_device", return_value=None),
            patch("hark.audio_sources.is_linux", return_value=False),
            patch("hark.audio_sources.is_macos", return_value=True),
            patch("hark.audio_sources.is_windows", return_value=False),
        ):
            errors = validate_source_availability(InputSource.SPEAKER)
            assert len(errors) == 1
            assert "BlackHole" in errors[0]

    def test_loopback_error_message_windows(self) -> None:
        """Error message should mention WASAPI on Windows."""
        with (
            patch("hark.audio_sources.find_loopback_device", return_value=None),
            patch("hark.audio_sources.is_linux", return_value=False),
            patch("hark.audio_sources.is_macos", return_value=False),
            patch("hark.audio_sources.is_windows", return_value=True),
        ):
            errors = validate_source_availability(InputSource.SPEAKER)
            assert len(errors) == 1
            assert "WASAPI" in errors[0]


class TestMonitorDeviceDetection:
    """Tests for monitor device name pattern detection."""

    def test_detects_pulseaudio_monitor(self) -> None:
        """Should detect .monitor suffix via backend."""
        mock_loopback = _create_mock_loopback_info(
            name="Monitor of Built-in Audio",
            device_id="alsa_output.pci-0000_00_1f.3.analog-stereo.monitor",
        )
        mock_backend = MagicMock()
        mock_backend.get_default_loopback.return_value = mock_loopback
        mock_backend.get_recording_config.return_value = RecordingConfig(
            env={"PULSE_SOURCE": "alsa_output.pci-0000_00_1f.3.analog-stereo.monitor"},
            device="pulse",
        )

        with patch("hark.audio_sources.get_loopback_backend", return_value=mock_backend):
            device = find_loopback_device()
            assert device is not None

    def test_detects_monitor_of_prefix(self) -> None:
        """Should detect 'Monitor of' in description via sounddevice fallback."""
        mock_backend = MagicMock()
        mock_backend.get_default_loopback.return_value = None

        mock_devices = [
            {
                "name": "Monitor of Built-in Audio",
                "max_input_channels": 2,
                "default_samplerate": 44100.0,
            },
        ]

        with (
            patch("hark.audio_sources.get_loopback_backend", return_value=mock_backend),
            patch("hark.audio_sources.sd.query_devices", return_value=mock_devices),
        ):
            device = find_loopback_device()
            assert device is not None

    def test_detects_stereo_mix(self) -> None:
        """Should detect 'Stereo Mix' (Windows) via sounddevice fallback."""
        mock_backend = MagicMock()
        mock_backend.get_default_loopback.return_value = None

        mock_devices = [
            {
                "name": "Stereo Mix",
                "max_input_channels": 2,
                "default_samplerate": 44100.0,
            },
        ]

        with (
            patch("hark.audio_sources.get_loopback_backend", return_value=mock_backend),
            patch("hark.audio_sources.sd.query_devices", return_value=mock_devices),
        ):
            device = find_loopback_device()
            assert device is not None
            assert device.name == "Stereo Mix"

    def test_detects_loopback_keyword(self) -> None:
        """Should detect 'loopback' keyword via sounddevice fallback."""
        mock_backend = MagicMock()
        mock_backend.get_default_loopback.return_value = None

        mock_devices = [
            {
                "name": "Virtual Loopback Device",
                "max_input_channels": 2,
                "default_samplerate": 44100.0,
            },
        ]

        with (
            patch("hark.audio_sources.get_loopback_backend", return_value=mock_backend),
            patch("hark.audio_sources.sd.query_devices", return_value=mock_devices),
        ):
            device = find_loopback_device()
            assert device is not None
