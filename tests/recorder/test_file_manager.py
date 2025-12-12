"""Tests for RecordingFileManager component."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from hark.exceptions import AudioDeviceBusyError
from hark.recorder import RecordingFileManager


class TestRecordingFileManagerInit:
    """Tests for RecordingFileManager initialization."""

    def test_stores_config(self, tmp_path: Path) -> None:
        """Should store configuration parameters."""
        manager = RecordingFileManager(tmp_path, 16000, 2)
        assert manager._temp_dir == tmp_path
        assert manager._sample_rate == 16000
        assert manager._channels == 2

    def test_initial_state(self, tmp_path: Path) -> None:
        """Should initialize with None/0 values."""
        manager = RecordingFileManager(tmp_path, 16000, 1)
        assert manager.file_path is None
        assert manager.frames_written == 0
        assert manager.is_open is False


class TestRecordingFileManagerCreate:
    """Tests for RecordingFileManager.create() method."""

    def test_creates_temp_dir(self, tmp_path: Path) -> None:
        """Should create temp directory if it doesn't exist."""
        temp_dir = tmp_path / "new_dir"
        manager = RecordingFileManager(temp_dir, 16000, 1)

        with patch("soundfile.SoundFile"):
            manager.create()

        assert temp_dir.exists()

    def test_creates_wav_file(self, tmp_path: Path) -> None:
        """Should create a .wav file."""
        manager = RecordingFileManager(tmp_path, 16000, 1)

        with patch("soundfile.SoundFile"):
            result = manager.create()

        assert result.suffix == ".wav"
        assert manager.file_path == result

    def test_opens_soundfile_with_correct_params(self, tmp_path: Path) -> None:
        """Should open SoundFile with correct parameters."""
        manager = RecordingFileManager(tmp_path, 48000, 2)

        with patch("soundfile.SoundFile") as mock_sf:
            manager.create()

            mock_sf.assert_called_once()
            call_kwargs = mock_sf.call_args[1]
            assert call_kwargs["mode"] == "w"
            assert call_kwargs["samplerate"] == 48000
            assert call_kwargs["channels"] == 2
            assert call_kwargs["format"] == "WAV"
            assert call_kwargs["subtype"] == "FLOAT"

    def test_raises_on_soundfile_error(self, tmp_path: Path) -> None:
        """Should raise AudioDeviceBusyError on file creation error."""
        manager = RecordingFileManager(tmp_path, 16000, 1)

        with patch("soundfile.SoundFile", side_effect=Exception("File error")):
            with pytest.raises(AudioDeviceBusyError) as exc_info:
                manager.create()

            assert "Failed to create audio file" in str(exc_info.value)

    def test_cleans_up_on_error(self, tmp_path: Path) -> None:
        """Should clean up temp file on error."""
        manager = RecordingFileManager(tmp_path, 16000, 1)

        # Create will fail, temp file should be removed
        with patch("soundfile.SoundFile", side_effect=Exception("File error")):
            with pytest.raises(AudioDeviceBusyError):
                manager.create()

        assert manager.file_path is None


class TestRecordingFileManagerWrite:
    """Tests for RecordingFileManager.write() method."""

    def test_writes_data(self, tmp_path: Path) -> None:
        """Should write audio data to file."""
        manager = RecordingFileManager(tmp_path, 16000, 1)

        mock_file = MagicMock()
        mock_file.closed = False
        manager._sound_file = mock_file

        data = np.zeros((100, 1), dtype=np.float32)
        frames = manager.write(data)

        mock_file.write.assert_called_once()
        assert frames == 100

    def test_increments_frames_written(self, tmp_path: Path) -> None:
        """Should track total frames written."""
        manager = RecordingFileManager(tmp_path, 16000, 1)

        mock_file = MagicMock()
        mock_file.closed = False
        manager._sound_file = mock_file

        data1 = np.zeros((100, 1), dtype=np.float32)
        data2 = np.zeros((50, 1), dtype=np.float32)

        manager.write(data1)
        manager.write(data2)

        assert manager.frames_written == 150

    def test_returns_zero_if_file_closed(self, tmp_path: Path) -> None:
        """Should return 0 if file is closed."""
        manager = RecordingFileManager(tmp_path, 16000, 1)

        mock_file = MagicMock()
        mock_file.closed = True
        manager._sound_file = mock_file

        data = np.zeros((100, 1), dtype=np.float32)
        frames = manager.write(data)

        assert frames == 0
        mock_file.write.assert_not_called()

    def test_returns_zero_if_no_file(self, tmp_path: Path) -> None:
        """Should return 0 if file is None."""
        manager = RecordingFileManager(tmp_path, 16000, 1)

        data = np.zeros((100, 1), dtype=np.float32)
        frames = manager.write(data)

        assert frames == 0

    def test_handles_write_error(self, tmp_path: Path) -> None:
        """Should handle write errors gracefully."""
        manager = RecordingFileManager(tmp_path, 16000, 1)

        mock_file = MagicMock()
        mock_file.closed = False
        mock_file.write.side_effect = sf.SoundFileError("Write failed")
        manager._sound_file = mock_file

        data = np.zeros((100, 1), dtype=np.float32)
        frames = manager.write(data)

        assert frames == 0


class TestRecordingFileManagerClose:
    """Tests for RecordingFileManager.close() method."""

    def test_closes_sound_file(self, tmp_path: Path) -> None:
        """Should close the sound file."""
        manager = RecordingFileManager(tmp_path, 16000, 1)

        mock_file = MagicMock()
        manager._sound_file = mock_file

        manager.close()

        mock_file.close.assert_called_once()
        assert manager._sound_file is None

    def test_handles_close_error(self, tmp_path: Path) -> None:
        """Should handle close errors gracefully."""
        manager = RecordingFileManager(tmp_path, 16000, 1)

        mock_file = MagicMock()
        mock_file.close.side_effect = OSError("Close failed")
        manager._sound_file = mock_file

        # Should not raise
        manager.close()
        assert manager._sound_file is None

    def test_noop_if_no_file(self, tmp_path: Path) -> None:
        """Should do nothing if no file."""
        manager = RecordingFileManager(tmp_path, 16000, 1)

        # Should not raise
        manager.close()


class TestRecordingFileManagerCleanup:
    """Tests for RecordingFileManager.cleanup() method."""

    def test_closes_and_removes_file(self, tmp_path: Path) -> None:
        """Should close file and remove temp file."""
        manager = RecordingFileManager(tmp_path, 16000, 1)

        # Create a real temp file
        temp_file = tmp_path / "test.wav"
        temp_file.touch()
        manager._temp_file = temp_file

        mock_file = MagicMock()
        manager._sound_file = mock_file

        manager.cleanup()

        mock_file.close.assert_called_once()
        assert not temp_file.exists()
        assert manager._temp_file is None

    def test_handles_missing_temp_file(self, tmp_path: Path) -> None:
        """Should handle already deleted temp file."""
        manager = RecordingFileManager(tmp_path, 16000, 1)

        # File doesn't exist
        manager._temp_file = tmp_path / "nonexistent.wav"

        # Should not raise
        manager.cleanup()
        assert manager._temp_file is None


class TestRecordingFileManagerProperties:
    """Tests for RecordingFileManager properties."""

    def test_is_open_true_when_file_open(self, tmp_path: Path) -> None:
        """is_open should be True when file is open."""
        manager = RecordingFileManager(tmp_path, 16000, 1)

        mock_file = MagicMock()
        mock_file.closed = False
        manager._sound_file = mock_file

        assert manager.is_open is True

    def test_is_open_false_when_file_closed(self, tmp_path: Path) -> None:
        """is_open should be False when file is closed."""
        manager = RecordingFileManager(tmp_path, 16000, 1)

        mock_file = MagicMock()
        mock_file.closed = True
        manager._sound_file = mock_file

        assert manager.is_open is False

    def test_is_open_false_when_no_file(self, tmp_path: Path) -> None:
        """is_open should be False when no file."""
        manager = RecordingFileManager(tmp_path, 16000, 1)
        assert manager.is_open is False
