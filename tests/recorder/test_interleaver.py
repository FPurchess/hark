"""Tests for DualStreamInterleaver component."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from hark.recorder import DualStreamInterleaver, RecordingFileManager


class TestDualStreamInterleaverInit:
    """Tests for DualStreamInterleaver initialization."""

    def test_stores_file_manager(self, tmp_path: Path) -> None:
        """Should store file manager reference."""
        file_manager = RecordingFileManager(tmp_path, 16000, 2)
        interleaver = DualStreamInterleaver(file_manager)
        assert interleaver._file_manager is file_manager

    def test_initial_state(self, tmp_path: Path) -> None:
        """Should initialize with empty buffers."""
        file_manager = RecordingFileManager(tmp_path, 16000, 2)
        interleaver = DualStreamInterleaver(file_manager)
        assert interleaver.mic_buffer == []
        assert interleaver.speaker_buffer == []
        assert interleaver._thread is None


class TestDualStreamInterleaverAddData:
    """Tests for DualStreamInterleaver add_*_data methods."""

    def test_add_mic_data(self, tmp_path: Path) -> None:
        """Should add data to mic buffer."""
        file_manager = RecordingFileManager(tmp_path, 16000, 2)
        interleaver = DualStreamInterleaver(file_manager)

        data = np.array([[0.5], [0.5]], dtype=np.float32)
        interleaver.add_mic_data(data)

        assert len(interleaver.mic_buffer) == 1
        np.testing.assert_array_equal(interleaver.mic_buffer[0], data)

    def test_add_speaker_data(self, tmp_path: Path) -> None:
        """Should add data to speaker buffer."""
        file_manager = RecordingFileManager(tmp_path, 16000, 2)
        interleaver = DualStreamInterleaver(file_manager)

        data = np.array([[0.3], [0.3]], dtype=np.float32)
        interleaver.add_speaker_data(data)

        assert len(interleaver.speaker_buffer) == 1
        np.testing.assert_array_equal(interleaver.speaker_buffer[0], data)

    def test_copies_data(self, tmp_path: Path) -> None:
        """Should copy data to prevent external modification."""
        file_manager = RecordingFileManager(tmp_path, 16000, 2)
        interleaver = DualStreamInterleaver(file_manager)

        data = np.array([[0.5]], dtype=np.float32)
        interleaver.add_mic_data(data)

        # Modify original
        data[0, 0] = 0.0

        # Buffer should be unchanged
        assert interleaver.mic_buffer[0][0, 0] == 0.5


class TestDualStreamInterleaverStartStop:
    """Tests for DualStreamInterleaver start/stop methods."""

    def test_start_creates_thread(self, tmp_path: Path) -> None:
        """Should create and start interleaving thread."""
        file_manager = RecordingFileManager(tmp_path, 16000, 2)
        interleaver = DualStreamInterleaver(file_manager)

        interleaver.start()

        assert interleaver._thread is not None
        assert interleaver._thread.is_alive()

        # Clean up
        interleaver.stop()

    def test_start_clears_buffers(self, tmp_path: Path) -> None:
        """Should clear buffers on start."""
        file_manager = RecordingFileManager(tmp_path, 16000, 2)
        interleaver = DualStreamInterleaver(file_manager)

        # Add some data
        interleaver._mic_buffer = [np.zeros((10, 1))]
        interleaver._speaker_buffer = [np.zeros((10, 1))]

        interleaver.start()

        assert len(interleaver.mic_buffer) == 0
        assert len(interleaver.speaker_buffer) == 0

        # Clean up
        interleaver.stop()

    def test_stop_joins_thread(self, tmp_path: Path) -> None:
        """Should stop and join thread."""
        file_manager = RecordingFileManager(tmp_path, 16000, 2)
        interleaver = DualStreamInterleaver(file_manager)

        interleaver.start()
        interleaver.stop()

        assert interleaver._thread is None

    def test_stop_flushes_remaining(self, tmp_path: Path) -> None:
        """Should flush remaining buffers on stop."""
        file_manager = RecordingFileManager(tmp_path, 16000, 2)
        mock_file = MagicMock()
        mock_file.closed = False
        file_manager._sound_file = mock_file

        interleaver = DualStreamInterleaver(file_manager)
        interleaver.start()

        # Add data after start
        mic_chunk = np.array([[0.5]], dtype=np.float32)
        speaker_chunk = np.array([[0.3]], dtype=np.float32)
        interleaver.add_mic_data(mic_chunk)
        interleaver.add_speaker_data(speaker_chunk)

        interleaver.stop()

        # Should have written the interleaved data
        assert mock_file.write.called


class TestDualStreamInterleaverProcessing:
    """Tests for DualStreamInterleaver buffer processing."""

    def test_interleaves_to_stereo(self, tmp_path: Path) -> None:
        """Should create stereo from mono channels."""
        file_manager = RecordingFileManager(tmp_path, 16000, 2)
        mock_file = MagicMock()
        mock_file.closed = False
        file_manager._sound_file = mock_file

        interleaver = DualStreamInterleaver(file_manager)

        mic_chunk = np.array([[0.5], [0.5]], dtype=np.float32)
        speaker_chunk = np.array([[0.3], [0.3]], dtype=np.float32)
        interleaver._mic_buffer = [mic_chunk]
        interleaver._speaker_buffer = [speaker_chunk]

        interleaver._process_buffers()

        # Check stereo output
        written_data = mock_file.write.call_args[0][0]
        assert written_data.shape == (2, 2)  # 2 samples, 2 channels
        np.testing.assert_array_almost_equal(written_data[:, 0], [0.5, 0.5])  # L = mic
        np.testing.assert_array_almost_equal(written_data[:, 1], [0.3, 0.3])  # R = speaker

    def test_truncates_to_shorter_chunk(self, tmp_path: Path) -> None:
        """Should truncate to shorter chunk length."""
        file_manager = RecordingFileManager(tmp_path, 16000, 2)
        mock_file = MagicMock()
        mock_file.closed = False
        file_manager._sound_file = mock_file

        interleaver = DualStreamInterleaver(file_manager)

        mic_chunk = np.array([[0.5], [0.5], [0.5]], dtype=np.float32)  # 3 samples
        speaker_chunk = np.array([[0.3], [0.3]], dtype=np.float32)  # 2 samples
        interleaver._mic_buffer = [mic_chunk]
        interleaver._speaker_buffer = [speaker_chunk]

        interleaver._process_buffers()

        written_data = mock_file.write.call_args[0][0]
        assert written_data.shape[0] == 2  # Truncated to 2

    def test_processes_multiple_chunks(self, tmp_path: Path) -> None:
        """Should process multiple chunk pairs."""
        file_manager = RecordingFileManager(tmp_path, 16000, 2)
        mock_file = MagicMock()
        mock_file.closed = False
        file_manager._sound_file = mock_file

        interleaver = DualStreamInterleaver(file_manager)

        # Add 2 chunks to each buffer
        interleaver._mic_buffer = [
            np.array([[0.5]], dtype=np.float32),
            np.array([[0.6]], dtype=np.float32),
        ]
        interleaver._speaker_buffer = [
            np.array([[0.3]], dtype=np.float32),
            np.array([[0.4]], dtype=np.float32),
        ]

        interleaver._process_buffers()

        # Should have written twice
        assert mock_file.write.call_count == 2

    def test_waits_for_matching_chunks(self, tmp_path: Path) -> None:
        """Should not write until both buffers have data."""
        file_manager = RecordingFileManager(tmp_path, 16000, 2)
        mock_file = MagicMock()
        mock_file.closed = False
        file_manager._sound_file = mock_file

        interleaver = DualStreamInterleaver(file_manager)

        # Only mic data
        interleaver._mic_buffer = [np.array([[0.5]], dtype=np.float32)]
        interleaver._speaker_buffer = []

        interleaver._process_buffers()

        mock_file.write.assert_not_called()
        # Mic data should still be in buffer
        assert len(interleaver._mic_buffer) == 1


class TestDualStreamInterleaverFlush:
    """Tests for DualStreamInterleaver._flush_remaining method."""

    def test_flushes_matched_pairs(self, tmp_path: Path) -> None:
        """Should write remaining matched pairs."""
        file_manager = RecordingFileManager(tmp_path, 16000, 2)
        mock_file = MagicMock()
        mock_file.closed = False
        file_manager._sound_file = mock_file

        interleaver = DualStreamInterleaver(file_manager)

        interleaver._mic_buffer = [np.array([[0.5]], dtype=np.float32)]
        interleaver._speaker_buffer = [np.array([[0.3]], dtype=np.float32)]

        interleaver._flush_remaining()

        mock_file.write.assert_called_once()

    def test_clears_unmatched_data(self, tmp_path: Path) -> None:
        """Should clear remaining unmatched data."""
        file_manager = RecordingFileManager(tmp_path, 16000, 2)
        interleaver = DualStreamInterleaver(file_manager)

        # More mic data than speaker
        interleaver._mic_buffer = [
            np.array([[0.5]], dtype=np.float32),
            np.array([[0.6]], dtype=np.float32),
        ]
        interleaver._speaker_buffer = [np.array([[0.3]], dtype=np.float32)]

        interleaver._flush_remaining()

        assert len(interleaver._mic_buffer) == 0
        assert len(interleaver._speaker_buffer) == 0
