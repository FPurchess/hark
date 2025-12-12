"""Recording file management for hark."""

import contextlib
import os
import tempfile
import threading
from pathlib import Path

import numpy as np
import soundfile as sf

from hark.exceptions import AudioDeviceBusyError

__all__ = ["RecordingFileManager"]


class RecordingFileManager:
    """
    Manages temporary audio file creation, writing, and cleanup.

    Responsibilities:
    - Creating temp directory and WAV file
    - Opening SoundFile for writing
    - Thread-safe audio data writing
    - Tracking frames written
    - Closing and cleaning up resources
    """

    def __init__(
        self,
        temp_dir: Path,
        sample_rate: int,
        channels: int,
    ) -> None:
        """
        Initialize the file manager.

        Args:
            temp_dir: Directory for temporary audio files.
            sample_rate: Audio sample rate in Hz.
            channels: Number of audio channels.
        """
        self._temp_dir = temp_dir
        self._sample_rate = sample_rate
        self._channels = channels
        self._temp_file: Path | None = None
        self._sound_file: sf.SoundFile | None = None
        self._lock = threading.Lock()
        self._frames_written = 0

    @property
    def file_path(self) -> Path | None:
        """Get the path to the temporary file."""
        return self._temp_file

    @property
    def frames_written(self) -> int:
        """Get the number of frames written."""
        return self._frames_written

    @property
    def is_open(self) -> bool:
        """Check if the file is open and writable."""
        return self._sound_file is not None and not self._sound_file.closed

    def create(self) -> Path:
        """
        Create temp directory and file, open SoundFile for writing.

        Returns:
            Path to the created temporary file.

        Raises:
            AudioDeviceBusyError: If file creation fails.
        """
        # Ensure temp directory exists
        self._temp_dir.mkdir(parents=True, exist_ok=True)

        # Create temporary file
        fd, temp_path = tempfile.mkstemp(suffix=".wav", dir=self._temp_dir)
        self._temp_file = Path(temp_path)

        # Close the file descriptor from mkstemp
        os.close(fd)

        # Open sound file for writing
        try:
            self._sound_file = sf.SoundFile(
                self._temp_file,
                mode="w",
                samplerate=self._sample_rate,
                channels=self._channels,
                format="WAV",
                subtype="FLOAT",
            )
        except Exception as e:
            self._temp_file.unlink(missing_ok=True)
            self._temp_file = None
            raise AudioDeviceBusyError(f"Failed to create audio file: {e}") from e

        self._frames_written = 0
        return self._temp_file

    def write(self, data: np.ndarray) -> int:
        """
        Write audio data to the file (thread-safe).

        Args:
            data: Audio data as numpy array.

        Returns:
            Number of frames written.
        """
        with self._lock:
            if self._sound_file is not None and not self._sound_file.closed:
                try:
                    self._sound_file.write(data)
                    frames = len(data)
                    self._frames_written += frames
                    return frames
                except (sf.SoundFileError, OSError):
                    pass
        return 0

    def close(self) -> None:
        """Close the SoundFile (does not delete the temp file)."""
        if self._sound_file is not None:
            with contextlib.suppress(OSError):
                self._sound_file.close()
            self._sound_file = None

    def cleanup(self) -> None:
        """Close the file and remove the temporary file."""
        self.close()
        if self._temp_file is not None:
            with contextlib.suppress(Exception):
                self._temp_file.unlink(missing_ok=True)
            self._temp_file = None
