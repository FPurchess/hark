"""Tests for audio_backends base module - protocol, dataclasses, exports."""

from unittest.mock import patch

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
