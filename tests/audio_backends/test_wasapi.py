"""Tests for WASAPI backend - Windows only."""

import sys
from unittest.mock import MagicMock, patch

import pytest

# Skip entire module on non-Windows platforms
pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="WASAPI tests require Windows")

from hark.audio_backends import LoopbackBackend  # noqa: E402
from hark.audio_backends.wasapi import WASAPIBackend  # noqa: E402


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
    def test_is_available_pyaudiowpatch_not_installed(self, _: MagicMock) -> None:
        """Should return False when PyAudioWPatch is not installed."""
        backend = WASAPIBackend()
        assert backend.is_available() is False

    def test_is_available_no_loopback_device(self) -> None:
        """Should return False when no WASAPI loopback device available."""
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
    def test_get_default_loopback_unavailable(self, _: MagicMock) -> None:
        """Should return None when PyAudioWPatch unavailable."""
        backend = WASAPIBackend()
        device = backend.get_default_loopback()
        assert device is None

    def test_list_loopback_devices_success(self) -> None:
        """Should list all WASAPI loopback devices."""
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
    def test_list_loopback_devices_unavailable(self, _: MagicMock) -> None:
        """Should return empty list when PyAudioWPatch unavailable."""
        backend = WASAPIBackend()
        devices = backend.list_loopback_devices()
        assert devices == []

    def test_list_loopback_devices_empty_when_no_devices(self) -> None:
        """Should return empty list when no loopback devices found."""
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
