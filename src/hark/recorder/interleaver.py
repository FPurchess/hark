"""Dual-stream audio interleaving for hark."""

import threading
import time

import numpy as np

from hark.recorder.file_manager import RecordingFileManager

__all__ = ["DualStreamInterleaver"]


class DualStreamInterleaver:
    """
    Manages buffer interleaving for dual-stream (mic + speaker) recording.

    Responsibilities:
    - Managing separate mic and speaker buffers
    - Running interleaving thread
    - Creating stereo audio from mono inputs (L=mic, R=speaker)
    - Flushing remaining buffers on stop
    """

    def __init__(self, file_manager: RecordingFileManager) -> None:
        """
        Initialize the interleaver.

        Args:
            file_manager: RecordingFileManager to write interleaved audio to.
        """
        self._file_manager = file_manager
        self._mic_buffer: list[np.ndarray] = []
        self._speaker_buffer: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    @property
    def mic_buffer(self) -> list[np.ndarray]:
        """Get mic buffer (for testing)."""
        return self._mic_buffer

    @property
    def speaker_buffer(self) -> list[np.ndarray]:
        """Get speaker buffer (for testing)."""
        return self._speaker_buffer

    def add_mic_data(self, data: np.ndarray) -> None:
        """
        Add audio data to the mic buffer.

        Args:
            data: Audio data as numpy array.
        """
        with self._lock:
            self._mic_buffer.append(data.copy())

    def add_speaker_data(self, data: np.ndarray) -> None:
        """
        Add audio data to the speaker buffer.

        Args:
            data: Audio data as numpy array.
        """
        with self._lock:
            self._speaker_buffer.append(data.copy())

    def start(self) -> None:
        """Start the interleaving thread."""
        self._stop_event.clear()
        self._mic_buffer.clear()
        self._speaker_buffer.clear()
        self._thread = threading.Thread(
            target=self._interleave_loop,
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the interleaving thread and flush remaining buffers."""
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join(timeout=1.0)
            self._thread = None

        # Flush any remaining matched buffers
        self._flush_remaining()

    def _interleave_loop(self) -> None:
        """Thread loop that interleaves mic and speaker buffers."""
        while not self._stop_event.is_set():
            self._process_buffers()
            # Prevent busy-waiting
            time.sleep(0.01)

    def _process_buffers(self) -> None:
        """Process available buffer pairs and write to file."""
        with self._lock:
            min_chunks = min(len(self._mic_buffer), len(self._speaker_buffer))

            for _ in range(min_chunks):
                mic_chunk = self._mic_buffer.pop(0)
                speaker_chunk = self._speaker_buffer.pop(0)

                # Ensure same length (take minimum)
                min_len = min(len(mic_chunk), len(speaker_chunk))
                mic_chunk = mic_chunk[:min_len]
                speaker_chunk = speaker_chunk[:min_len]

                # Interleave: L=mic, R=speaker
                stereo = np.column_stack(
                    [
                        mic_chunk.flatten(),
                        speaker_chunk.flatten(),
                    ]
                )

                self._file_manager.write(stereo)

    def _flush_remaining(self) -> None:
        """Flush any remaining matched buffer pairs."""
        with self._lock:
            min_chunks = min(len(self._mic_buffer), len(self._speaker_buffer))

            for _ in range(min_chunks):
                mic_chunk = self._mic_buffer.pop(0)
                speaker_chunk = self._speaker_buffer.pop(0)

                min_len = min(len(mic_chunk), len(speaker_chunk))
                mic_chunk = mic_chunk[:min_len]
                speaker_chunk = speaker_chunk[:min_len]

                stereo = np.column_stack(
                    [
                        mic_chunk.flatten(),
                        speaker_chunk.flatten(),
                    ]
                )

                self._file_manager.write(stereo)

            # Clear any remaining unmatched buffers
            self._mic_buffer.clear()
            self._speaker_buffer.clear()
