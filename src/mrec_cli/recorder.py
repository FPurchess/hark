"""Audio recording for mrec-cli."""

from __future__ import annotations

import contextlib
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypedDict, cast

import numpy as np
import sounddevice as sd
import soundfile as sf

from mrec_cli.constants import DEFAULT_BUFFER_SIZE, DEFAULT_TEMP_DIR
from mrec_cli.exceptions import AudioDeviceBusyError, NoMicrophoneError

__all__ = [
    "AudioDeviceInfo",
    "AudioRecorder",
]


class AudioDeviceInfo(TypedDict):
    """Information about an audio input device."""

    index: int
    name: str
    channels: int
    sample_rate: float


class AudioRecorder:
    """
    Records audio from microphone with real-time level monitoring.

    Streams audio to a temporary WAV file to handle long recordings
    without running out of memory.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        max_duration: int = 600,
        level_callback: Callable[[float], None] | None = None,
        temp_dir: Path = DEFAULT_TEMP_DIR,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
    ) -> None:
        """
        Initialize the audio recorder.

        Args:
            sample_rate: Audio sample rate in Hz.
            channels: Number of audio channels (1 for mono, 2 for stereo).
            max_duration: Maximum recording duration in seconds.
            level_callback: Callback function that receives RMS level (0.0-1.0).
            temp_dir: Directory for temporary audio files.
            buffer_size: Audio buffer size for sounddevice.
        """
        self._sample_rate = sample_rate
        self._channels = channels
        self._max_duration = max_duration
        self._level_callback = level_callback
        self._temp_dir = temp_dir
        self._buffer_size = buffer_size

        self._is_recording = False
        self._start_time: float | None = None
        self._stream: sd.InputStream | None = None
        self._temp_file: Path | None = None
        self._sound_file: sf.SoundFile | None = None
        self._lock = threading.Lock()
        self._frames_written = 0

    @property
    def is_recording(self) -> bool:
        """Check if recording is in progress."""
        return self._is_recording

    def start(self) -> None:
        """
        Start recording audio.

        Raises:
            NoMicrophoneError: If no microphone is available.
            AudioDeviceBusyError: If the audio device is busy.
        """
        if self._is_recording:
            return

        # Ensure temp directory exists
        self._temp_dir.mkdir(parents=True, exist_ok=True)

        # Create temporary file
        fd, temp_path = tempfile.mkstemp(suffix=".wav", dir=self._temp_dir)
        self._temp_file = Path(temp_path)

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
            raise AudioDeviceBusyError(f"Failed to create audio file: {e}") from e

        # Start audio stream
        try:
            self._stream = sd.InputStream(
                callback=self._audio_callback,
                channels=self._channels,
                samplerate=self._sample_rate,
                blocksize=self._buffer_size,
                dtype=np.float32,
                latency="low",
            )
            self._stream.start()
        except sd.PortAudioError as e:
            self._cleanup()
            if "No Default Input Device" in str(e) or "Invalid device" in str(e):
                raise NoMicrophoneError("No microphone detected") from e
            raise AudioDeviceBusyError(f"Audio device error: {e}") from e
        except Exception as e:
            self._cleanup()
            raise AudioDeviceBusyError(f"Failed to start audio stream: {e}") from e

        self._is_recording = True
        self._start_time = time.time()
        self._frames_written = 0

    def stop(self) -> Path:
        """
        Stop recording and return the path to the recorded audio file.

        Returns:
            Path to the temporary WAV file containing the recording.
        """
        if not self._is_recording:
            if self._temp_file:
                return self._temp_file
            raise RuntimeError("Recording was never started")

        self._is_recording = False

        # Stop and close stream
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except (sd.PortAudioError, OSError):
                pass
            self._stream = None

        # Close sound file
        if self._sound_file is not None:
            with contextlib.suppress(OSError):
                self._sound_file.close()
            self._sound_file = None

        if self._temp_file is None:
            raise RuntimeError("No temp file created")

        return self._temp_file

    def get_duration(self) -> float:
        """
        Get the current recording duration in seconds.

        Returns:
            Recording duration in seconds.
        """
        if self._start_time is None:
            return 0.0
        if self._is_recording:
            return time.time() - self._start_time
        # If stopped, calculate from frames written
        return self._frames_written / self._sample_rate

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: dict,
        status: sd.CallbackFlags,
    ) -> None:
        """
        Callback for sounddevice InputStream.

        Args:
            indata: Input audio data.
            frames: Number of frames.
            time_info: Time information.
            status: Status flags.
        """
        if status:
            # Log status issues (could add logging here)
            pass

        if not self._is_recording:
            return

        # Check max duration
        if self._start_time and (time.time() - self._start_time) >= self._max_duration:
            self._is_recording = False
            return

        # Calculate RMS level for UI feedback
        if self._level_callback:
            rms = float(np.sqrt(np.mean(indata**2)))
            self._level_callback(rms)

        # Write to file (thread-safe)
        with self._lock:
            if self._sound_file is not None and not self._sound_file.closed:
                try:
                    self._sound_file.write(indata)
                    self._frames_written += len(indata)
                except (sf.SoundFileError, OSError):
                    pass

    def _cleanup(self) -> None:
        """Clean up resources."""
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        if self._sound_file is not None:
            with contextlib.suppress(Exception):
                self._sound_file.close()
            self._sound_file = None

        if self._temp_file is not None:
            with contextlib.suppress(Exception):
                self._temp_file.unlink(missing_ok=True)
            self._temp_file = None

    @staticmethod
    def list_devices() -> list[AudioDeviceInfo]:
        """
        List available audio input devices.

        Returns:
            List of AudioDeviceInfo dictionaries.
        """
        devices: list[AudioDeviceInfo] = []
        for i, device in enumerate(sd.query_devices()):
            if device["max_input_channels"] > 0:
                devices.append(
                    AudioDeviceInfo(
                        index=i,
                        name=device["name"],
                        channels=device["max_input_channels"],
                        sample_rate=device["default_samplerate"],
                    )
                )
        return devices

    @staticmethod
    def get_default_device() -> AudioDeviceInfo | None:
        """
        Get the default audio input device.

        Returns:
            AudioDeviceInfo, or None if no default device.
        """
        try:
            device_id = sd.default.device[0]
            if device_id is None:
                return None
            device = cast(dict[str, Any], sd.query_devices(device_id))
            return AudioDeviceInfo(
                index=int(device_id),
                name=str(device["name"]),
                channels=int(device["max_input_channels"]),
                sample_rate=float(device["default_samplerate"]),
            )
        except Exception:
            return None
