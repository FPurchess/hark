"""Tests for mrec_cli.recorder module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import sounddevice as sd

from mrec_cli.exceptions import AudioDeviceBusyError, NoMicrophoneError
from mrec_cli.recorder import AudioRecorder


class TestAudioRecorderInit:
    """Tests for AudioRecorder initialization."""

    def test_default_parameters(self) -> None:
        """Should set default parameters correctly."""
        recorder = AudioRecorder()
        assert recorder._sample_rate == 16000
        assert recorder._channels == 1
        assert recorder._max_duration == 600

    def test_custom_parameters(self, tmp_path: Path) -> None:
        """Should accept custom parameters."""
        recorder = AudioRecorder(
            sample_rate=48000,
            channels=2,
            max_duration=300,
            temp_dir=tmp_path,
            buffer_size=2048,
        )
        assert recorder._sample_rate == 48000
        assert recorder._channels == 2
        assert recorder._max_duration == 300
        assert recorder._temp_dir == tmp_path
        assert recorder._buffer_size == 2048

    def test_is_recording_initially_false(self) -> None:
        """is_recording should be False initially."""
        recorder = AudioRecorder()
        assert recorder.is_recording is False

    def test_level_callback_stored(self) -> None:
        """Level callback should be stored."""
        callback = MagicMock()
        recorder = AudioRecorder(level_callback=callback)
        assert recorder._level_callback is callback


class TestAudioRecorderStart:
    """Tests for start method."""

    def test_creates_temp_dir(self, tmp_path: Path) -> None:
        """Should create temp directory if it doesn't exist."""
        temp_dir = tmp_path / "new_dir"
        recorder = AudioRecorder(temp_dir=temp_dir)

        with patch("sounddevice.InputStream") as mock_stream_cls:
            mock_stream = MagicMock()
            mock_stream_cls.return_value = mock_stream
            with patch("soundfile.SoundFile"):
                recorder.start()

        assert temp_dir.exists()
        recorder.stop()

    def test_creates_temp_file(self, tmp_path: Path) -> None:
        """Should create temp file with mkstemp."""
        recorder = AudioRecorder(temp_dir=tmp_path)

        with patch("sounddevice.InputStream") as mock_stream_cls:
            mock_stream = MagicMock()
            mock_stream_cls.return_value = mock_stream
            with patch("soundfile.SoundFile"):
                recorder.start()

        assert recorder._temp_file is not None
        assert recorder._temp_file.suffix == ".wav"
        recorder.stop()

    def test_opens_soundfile(self, tmp_path: Path) -> None:
        """Should open SoundFile for writing."""
        recorder = AudioRecorder(temp_dir=tmp_path, sample_rate=16000, channels=1)

        with patch("sounddevice.InputStream") as mock_stream_cls:
            mock_stream = MagicMock()
            mock_stream_cls.return_value = mock_stream
            with patch("soundfile.SoundFile") as mock_sf:
                recorder.start()
                mock_sf.assert_called_once()
                call_kwargs = mock_sf.call_args[1]
                assert call_kwargs["mode"] == "w"
                assert call_kwargs["samplerate"] == 16000
                assert call_kwargs["channels"] == 1

        recorder.stop()

    def test_opens_input_stream(self, tmp_path: Path) -> None:
        """Should open InputStream with correct parameters."""
        recorder = AudioRecorder(temp_dir=tmp_path, sample_rate=16000, channels=1, buffer_size=1024)

        with patch("sounddevice.InputStream") as mock_stream_cls:
            mock_stream = MagicMock()
            mock_stream_cls.return_value = mock_stream
            with patch("soundfile.SoundFile"):
                recorder.start()

                mock_stream_cls.assert_called_once()
                call_kwargs = mock_stream_cls.call_args[1]
                assert call_kwargs["samplerate"] == 16000
                assert call_kwargs["channels"] == 1
                assert call_kwargs["blocksize"] == 1024
                mock_stream.start.assert_called_once()

        recorder.stop()

    def test_sets_recording_flag(self, tmp_path: Path) -> None:
        """Should set is_recording to True."""
        recorder = AudioRecorder(temp_dir=tmp_path)

        with patch("sounddevice.InputStream") as mock_stream_cls:
            mock_stream = MagicMock()
            mock_stream_cls.return_value = mock_stream
            with patch("soundfile.SoundFile"):
                recorder.start()
                assert recorder.is_recording is True

        recorder.stop()

    def test_no_microphone_error(self, tmp_path: Path) -> None:
        """Should raise NoMicrophoneError when no mic is available."""
        recorder = AudioRecorder(temp_dir=tmp_path)

        with (
            patch("soundfile.SoundFile"),
            patch("sounddevice.InputStream") as mock_stream_cls,
        ):
            mock_stream_cls.side_effect = sd.PortAudioError("No Default Input Device")

            with pytest.raises(NoMicrophoneError):
                recorder.start()

    def test_device_busy_error(self, tmp_path: Path) -> None:
        """Should raise AudioDeviceBusyError when device is busy."""
        recorder = AudioRecorder(temp_dir=tmp_path)

        with (
            patch("soundfile.SoundFile"),
            patch("sounddevice.InputStream") as mock_stream_cls,
        ):
            mock_stream_cls.side_effect = sd.PortAudioError("Device unavailable")

            with pytest.raises(AudioDeviceBusyError):
                recorder.start()

    def test_idempotent_start(self, tmp_path: Path) -> None:
        """Second start() should be no-op if already recording."""
        recorder = AudioRecorder(temp_dir=tmp_path)

        with patch("sounddevice.InputStream") as mock_stream_cls:
            mock_stream = MagicMock()
            mock_stream_cls.return_value = mock_stream
            with patch("soundfile.SoundFile"):
                recorder.start()
                recorder.start()  # Second call

                # Should only be called once
                assert mock_stream_cls.call_count == 1

        recorder.stop()

    def test_cleanup_on_stream_error(self, tmp_path: Path) -> None:
        """Should clean up resources if stream creation fails."""
        recorder = AudioRecorder(temp_dir=tmp_path)

        with patch("soundfile.SoundFile") as mock_sf:
            mock_file = MagicMock()
            mock_sf.return_value = mock_file
            with patch("sounddevice.InputStream") as mock_stream_cls:
                mock_stream_cls.side_effect = Exception("Stream error")

                with pytest.raises(AudioDeviceBusyError):
                    recorder.start()

                # Sound file should be closed
                mock_file.close.assert_called()


class TestAudioRecorderStop:
    """Tests for stop method."""

    def test_returns_temp_file_path(self, tmp_path: Path) -> None:
        """Should return path to temp file."""
        recorder = AudioRecorder(temp_dir=tmp_path)

        with patch("sounddevice.InputStream") as mock_stream_cls:
            mock_stream = MagicMock()
            mock_stream_cls.return_value = mock_stream
            with patch("soundfile.SoundFile"):
                recorder.start()
                result = recorder.stop()

                assert isinstance(result, Path)
                assert result == recorder._temp_file

    def test_closes_stream(self, tmp_path: Path) -> None:
        """Should stop and close the stream."""
        recorder = AudioRecorder(temp_dir=tmp_path)

        with patch("sounddevice.InputStream") as mock_stream_cls:
            mock_stream = MagicMock()
            mock_stream_cls.return_value = mock_stream
            with patch("soundfile.SoundFile"):
                recorder.start()
                recorder.stop()

                mock_stream.stop.assert_called_once()
                mock_stream.close.assert_called_once()

    def test_closes_soundfile(self, tmp_path: Path) -> None:
        """Should close the SoundFile."""
        recorder = AudioRecorder(temp_dir=tmp_path)

        with patch("sounddevice.InputStream") as mock_stream_cls:
            mock_stream = MagicMock()
            mock_stream_cls.return_value = mock_stream
            with patch("soundfile.SoundFile") as mock_sf:
                mock_file = MagicMock()
                mock_sf.return_value = mock_file
                recorder.start()
                recorder.stop()

                mock_file.close.assert_called_once()

    def test_sets_recording_flag_false(self, tmp_path: Path) -> None:
        """Should set is_recording to False."""
        recorder = AudioRecorder(temp_dir=tmp_path)

        with patch("sounddevice.InputStream") as mock_stream_cls:
            mock_stream = MagicMock()
            mock_stream_cls.return_value = mock_stream
            with patch("soundfile.SoundFile"):
                recorder.start()
                recorder.stop()

                assert recorder.is_recording is False

    def test_not_started_raises_error(self) -> None:
        """Should raise RuntimeError if never started."""
        recorder = AudioRecorder()

        with pytest.raises(RuntimeError) as exc_info:
            recorder.stop()
        assert "never started" in str(exc_info.value).lower()


class TestGetDuration:
    """Tests for get_duration method."""

    def test_not_started_returns_zero(self) -> None:
        """Should return 0.0 if not started."""
        recorder = AudioRecorder()
        assert recorder.get_duration() == 0.0

    def test_while_recording_returns_elapsed_time(self, tmp_path: Path) -> None:
        """Should return elapsed time while recording."""
        recorder = AudioRecorder(temp_dir=tmp_path)

        with patch("sounddevice.InputStream") as mock_stream_cls:
            mock_stream = MagicMock()
            mock_stream_cls.return_value = mock_stream
            with patch("soundfile.SoundFile"), patch("time.time") as mock_time:
                mock_time.return_value = 100.0
                recorder.start()

                mock_time.return_value = 105.0
                duration = recorder.get_duration()

                assert duration == 5.0

        recorder.stop()

    def test_after_stop_returns_calculated_duration(self, tmp_path: Path) -> None:
        """Should return calculated duration after stop."""
        recorder = AudioRecorder(temp_dir=tmp_path, sample_rate=16000)

        with patch("sounddevice.InputStream") as mock_stream_cls:
            mock_stream = MagicMock()
            mock_stream_cls.return_value = mock_stream
            with patch("soundfile.SoundFile"):
                recorder.start()
                # Set frames written AFTER start (since start resets it)
                recorder._frames_written = 32000  # 2 seconds worth
                recorder.stop()

                duration = recorder.get_duration()
                assert duration == 2.0


class TestAudioCallback:
    """Tests for _audio_callback method."""

    def test_calls_level_callback(self, tmp_path: Path) -> None:
        """Should call level_callback with RMS value."""
        callback = MagicMock()
        recorder = AudioRecorder(temp_dir=tmp_path, level_callback=callback)

        with patch("sounddevice.InputStream") as mock_stream_cls:
            mock_stream = MagicMock()
            mock_stream_cls.return_value = mock_stream
            with patch("soundfile.SoundFile") as mock_sf:
                mock_file = MagicMock()
                mock_file.closed = False
                mock_sf.return_value = mock_file
                recorder.start()

                # Simulate callback
                test_data = np.array([[0.5], [0.5], [0.5]], dtype=np.float32)
                recorder._audio_callback(test_data, 3, {}, sd.CallbackFlags())

                callback.assert_called()
                # RMS of 0.5, 0.5, 0.5 = 0.5
                assert abs(callback.call_args[0][0] - 0.5) < 0.01

        recorder.stop()

    def test_writes_to_file(self, tmp_path: Path) -> None:
        """Should write data to sound file."""
        recorder = AudioRecorder(temp_dir=tmp_path)

        with patch("sounddevice.InputStream") as mock_stream_cls:
            mock_stream = MagicMock()
            mock_stream_cls.return_value = mock_stream
            with patch("soundfile.SoundFile") as mock_sf:
                mock_file = MagicMock()
                mock_file.closed = False
                mock_sf.return_value = mock_file
                recorder.start()

                test_data = np.zeros((100, 1), dtype=np.float32)
                recorder._audio_callback(test_data, 100, {}, sd.CallbackFlags())

                mock_file.write.assert_called()

        recorder.stop()

    def test_checks_max_duration(self, tmp_path: Path) -> None:
        """Should stop recording at max duration."""
        recorder = AudioRecorder(temp_dir=tmp_path, max_duration=5)

        with patch("sounddevice.InputStream") as mock_stream_cls:
            mock_stream = MagicMock()
            mock_stream_cls.return_value = mock_stream
            with patch("soundfile.SoundFile") as mock_sf:
                mock_file = MagicMock()
                mock_file.closed = False
                mock_sf.return_value = mock_file
                with patch("time.time") as mock_time:
                    mock_time.return_value = 100.0
                    recorder.start()

                    # Simulate callback after max duration
                    mock_time.return_value = 106.0  # 6 seconds later
                    test_data = np.zeros((100, 1), dtype=np.float32)
                    recorder._audio_callback(test_data, 100, {}, sd.CallbackFlags())

                    # Should have stopped recording
                    assert recorder._is_recording is False

        recorder.stop()

    def test_does_not_write_when_not_recording(self, tmp_path: Path) -> None:
        """Should not write when not recording."""
        recorder = AudioRecorder(temp_dir=tmp_path)

        with patch("soundfile.SoundFile") as mock_sf:
            mock_file = MagicMock()
            mock_file.closed = False
            mock_sf.return_value = mock_file

            # Callback without starting
            test_data = np.zeros((100, 1), dtype=np.float32)
            recorder._audio_callback(test_data, 100, {}, sd.CallbackFlags())

            mock_file.write.assert_not_called()


class TestStaticMethods:
    """Tests for static methods."""

    def test_list_devices_returns_input_devices(self) -> None:
        """Should return list of input devices."""
        mock_devices = [
            {"name": "Microphone", "max_input_channels": 2, "default_samplerate": 44100},
            {"name": "Speaker", "max_input_channels": 0, "default_samplerate": 44100},
            {"name": "USB Mic", "max_input_channels": 1, "default_samplerate": 48000},
        ]

        with patch("sounddevice.query_devices", return_value=mock_devices):
            devices = AudioRecorder.list_devices()

            # Should only include input devices (channels > 0)
            assert len(devices) == 2
            assert devices[0]["name"] == "Microphone"
            assert devices[1]["name"] == "USB Mic"

    def test_list_devices_format(self) -> None:
        """Should return devices with expected keys."""
        mock_devices = [
            {"name": "Microphone", "max_input_channels": 2, "default_samplerate": 44100},
        ]

        with patch("sounddevice.query_devices", return_value=mock_devices):
            devices = AudioRecorder.list_devices()

            assert "index" in devices[0]
            assert "name" in devices[0]
            assert "channels" in devices[0]
            assert "sample_rate" in devices[0]

    def test_get_default_device_returns_device(self) -> None:
        """Should return default device info."""
        mock_device = {
            "name": "Default Mic",
            "max_input_channels": 2,
            "default_samplerate": 44100,
        }

        with (
            patch("sounddevice.default.device", [0, 1]),  # Input, Output
            patch("sounddevice.query_devices", return_value=mock_device),
        ):
            device = AudioRecorder.get_default_device()

            assert device is not None
            assert device["name"] == "Default Mic"

    def test_get_default_device_returns_none_on_error(self) -> None:
        """Should return None if no default device."""
        # Mock the entire get_default_device function internals
        with (
            patch.object(sd.default, "device", new=[None, None]),
            patch("sounddevice.query_devices", side_effect=Exception("No device")),
        ):
            device = AudioRecorder.get_default_device()
            assert device is None

    def test_get_default_device_handles_exception(self) -> None:
        """Should return None on exception."""
        # Patch query_devices to raise an exception
        with patch("sounddevice.query_devices", side_effect=Exception("Error")):
            device = AudioRecorder.get_default_device()
            assert device is None


class TestCleanup:
    """Tests for cleanup functionality."""

    def test_cleanup_closes_stream(self, tmp_path: Path) -> None:
        """Cleanup should close stream."""
        recorder = AudioRecorder(temp_dir=tmp_path)

        mock_stream = MagicMock()
        recorder._stream = mock_stream

        recorder._cleanup()

        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()
        assert recorder._stream is None

    def test_cleanup_closes_sound_file(self, tmp_path: Path) -> None:
        """Cleanup should close sound file."""
        recorder = AudioRecorder(temp_dir=tmp_path)

        mock_file = MagicMock()
        recorder._sound_file = mock_file

        recorder._cleanup()

        mock_file.close.assert_called_once()
        assert recorder._sound_file is None

    def test_cleanup_removes_temp_file(self, tmp_path: Path) -> None:
        """Cleanup should remove temp file."""
        recorder = AudioRecorder(temp_dir=tmp_path)

        temp_file = tmp_path / "test.wav"
        temp_file.touch()
        recorder._temp_file = temp_file

        recorder._cleanup()

        assert not temp_file.exists()
        assert recorder._temp_file is None

    def test_cleanup_handles_exceptions(self, tmp_path: Path) -> None:
        """Cleanup should handle exceptions gracefully."""
        recorder = AudioRecorder(temp_dir=tmp_path)

        mock_stream = MagicMock()
        mock_stream.stop.side_effect = Exception("Stop error")
        recorder._stream = mock_stream

        # Should not raise
        recorder._cleanup()
        assert recorder._stream is None
