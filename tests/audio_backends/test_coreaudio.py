"""Tests for CoreAudio backend - macOS only."""

import sys
from unittest.mock import MagicMock, patch

import pytest

# Skip entire module on non-macOS platforms
pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="CoreAudio tests require macOS")

from hark.audio_backends import LoopbackBackend  # noqa: E402
from hark.audio_backends.coreaudio import CoreAudioBackend  # noqa: E402


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
    def test_is_available_success(self, mock_query: MagicMock, _: MagicMock, __: MagicMock) -> None:
        """Should return True on macOS with sounddevice working."""
        mock_query.return_value = [
            _create_mock_sd_device("Built-in Microphone"),
        ]

        backend = CoreAudioBackend()
        assert backend.is_available() is True

    @patch("hark.audio_backends.coreaudio._check_sounddevice_available", return_value=False)
    def test_is_available_sounddevice_unavailable(self, _: MagicMock) -> None:
        """Should return False when sounddevice unavailable."""
        backend = CoreAudioBackend()
        assert backend.is_available() is False

    @patch("hark.audio_backends.coreaudio.is_macos", return_value=False)
    @patch("hark.audio_backends.coreaudio._check_sounddevice_available", return_value=True)
    @patch("hark.audio_backends.coreaudio.sd.query_devices")
    def test_is_available_wrong_platform(
        self, mock_query: MagicMock, _: MagicMock, __: MagicMock
    ) -> None:
        """Should return False on non-macOS platforms."""
        mock_query.return_value = []
        backend = CoreAudioBackend()
        assert backend.is_available() is False

    @patch("hark.audio_backends.coreaudio._check_sounddevice_available", return_value=True)
    @patch("hark.audio_backends.coreaudio.sd.query_devices")
    def test_get_default_loopback_blackhole_2ch(self, mock_query: MagicMock, _: MagicMock) -> None:
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
    def test_get_default_loopback_blackhole_16ch(self, mock_query: MagicMock, _: MagicMock) -> None:
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
    def test_get_default_loopback_none_found(self, mock_query: MagicMock, _: MagicMock) -> None:
        """Should return None when no BlackHole devices found."""
        mock_query.return_value = [
            _create_mock_sd_device("Built-in Microphone", max_input_channels=2),
            _create_mock_sd_device("USB Audio Device", max_input_channels=2),
        ]

        backend = CoreAudioBackend()
        device = backend.get_default_loopback()
        assert device is None

    @patch("hark.audio_backends.coreaudio._check_sounddevice_available", return_value=False)
    def test_get_default_loopback_unavailable(self, _: MagicMock) -> None:
        """Should return None when sounddevice unavailable."""
        backend = CoreAudioBackend()
        assert backend.get_default_loopback() is None

    @patch("hark.audio_backends.coreaudio._check_sounddevice_available", return_value=True)
    @patch("hark.audio_backends.coreaudio.sd.query_devices")
    def test_list_loopback_devices_multiple_blackhole(
        self, mock_query: MagicMock, _: MagicMock
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
        self, mock_query: MagicMock, _: MagicMock
    ) -> None:
        """Should return empty list when no BlackHole devices."""
        mock_query.return_value = [
            _create_mock_sd_device("Built-in Microphone", max_input_channels=2),
        ]

        backend = CoreAudioBackend()
        devices = backend.list_loopback_devices()
        assert devices == []

    @patch("hark.audio_backends.coreaudio._check_sounddevice_available", return_value=False)
    def test_list_loopback_devices_unavailable(self, _: MagicMock) -> None:
        """Should return empty list when sounddevice unavailable."""
        backend = CoreAudioBackend()
        devices = backend.list_loopback_devices()
        assert devices == []

    @patch("hark.audio_backends.coreaudio._check_sounddevice_available", return_value=True)
    @patch("hark.audio_backends.coreaudio.sd.query_devices")
    def test_skips_output_only_devices(self, mock_query: MagicMock, _: MagicMock) -> None:
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
