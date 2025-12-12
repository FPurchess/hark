"""Audio recording for hark."""

import contextlib
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import sounddevice as sd
import soundfile as sf

from hark.audio_sources import (
    AudioSourceInfo,
    InputSource,
    get_devices_for_source,
    validate_source_availability,
)
from hark.constants import (
    DEFAULT_BUFFER_SIZE,
    DEFAULT_INPUT_SOURCE,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_TEMP_DIR,
)
from hark.exceptions import AudioDeviceBusyError, NoLoopbackDeviceError, NoMicrophoneError
from hark.platform import is_windows
from hark.recorder.file_manager import RecordingFileManager
from hark.recorder.interleaver import DualStreamInterleaver
from hark.recorder.types import AudioDeviceInfo
from hark.utils import env_vars

# WASAPI stream availability check (Windows only)
_WASAPI_STREAM_AVAILABLE: bool | None = None


def _is_wasapi_device(device: str | int | None) -> tuple[bool, int | None]:
    """Check if device identifier indicates a WASAPI loopback device.

    Args:
        device: Device identifier from RecordingConfig.

    Returns:
        Tuple of (is_wasapi, device_index). device_index is None if
        the marker doesn't include an index.
    """
    if not isinstance(device, str) or not device.startswith("wasapi"):
        return False, None
    if ":" in device:
        try:
            return True, int(device.split(":")[1])
        except (ValueError, IndexError):
            return True, None
    return True, None


def _check_wasapi_available() -> bool:
    """Check if WASAPI streaming is available (cached)."""
    global _WASAPI_STREAM_AVAILABLE
    if _WASAPI_STREAM_AVAILABLE is None:
        if not is_windows():
            _WASAPI_STREAM_AVAILABLE = False
        else:
            try:
                import pyaudiowpatch  # noqa: F401  # pyrefly: ignore[missing-import]

                _WASAPI_STREAM_AVAILABLE = True
            except ImportError:
                _WASAPI_STREAM_AVAILABLE = False
    return _WASAPI_STREAM_AVAILABLE


__all__ = ["AudioRecorder"]


class AudioRecorder:
    """
    Records audio from microphone and/or system audio with real-time level monitoring.

    Streams audio to a temporary WAV file to handle long recordings
    without running out of memory. Supports three input modes:
    - mic: Microphone only (default)
    - speaker: System audio/loopback only
    - both: Simultaneous mic + speaker capture to stereo (L=mic, R=speaker)

    This class uses composition with focused components:
    - RecordingFileManager: Handles temp file creation, writing, and cleanup
    - DualStreamInterleaver: Handles buffer management for dual-stream mode
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        channels: int = 1,
        max_duration: int = 600,
        level_callback: Callable[[float], None] | None = None,
        temp_dir: Path = DEFAULT_TEMP_DIR,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        input_source: str = DEFAULT_INPUT_SOURCE,
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
            input_source: Input source mode ("mic", "speaker", or "both").
        """
        self._sample_rate = sample_rate
        self._channels = channels
        self._max_duration = max_duration
        self._level_callback = level_callback
        self._temp_dir = temp_dir
        self._buffer_size = buffer_size
        self._input_source = InputSource(input_source)

        # Recording state
        self._is_recording = False
        self._start_time: float | None = None

        # Components (created on start)
        self._file_manager: RecordingFileManager | None = None
        self._interleaver: DualStreamInterleaver | None = None

        # Stream management
        self._stream: sd.InputStream | None = None
        self._mic_stream: sd.InputStream | None = None
        self._speaker_stream: sd.InputStream | None = None

        # Device info
        self._mic_device: AudioSourceInfo | None = None
        self._speaker_device: AudioSourceInfo | None = None

        # WASAPI stream management (Windows)
        self._wasapi_stream: Any = None
        self._pyaudio_instance: Any = None

    @property
    def is_recording(self) -> bool:
        """Check if recording is in progress."""
        return self._is_recording

    # Backwards-compatible properties for internal state access
    @property
    def _temp_file(self) -> Path | None:
        """Get temp file path (backwards compatibility)."""
        return self._file_manager.file_path if self._file_manager else None

    @_temp_file.setter
    def _temp_file(self, value: Path | None) -> None:
        """Set temp file path - only used during cleanup."""
        if self._file_manager is not None and value is None:
            self._file_manager._temp_file = None

    @property
    def _sound_file(self) -> sf.SoundFile | None:
        """Get sound file (backwards compatibility)."""
        return self._file_manager._sound_file if self._file_manager else None

    @_sound_file.setter
    def _sound_file(self, value: sf.SoundFile | None) -> None:
        """Set sound file - only used during cleanup or tests."""
        if self._file_manager is not None:
            self._file_manager._sound_file = value

    @property
    def _frames_written(self) -> int:
        """Get frames written (backwards compatibility)."""
        return self._file_manager.frames_written if self._file_manager else 0

    @_frames_written.setter
    def _frames_written(self, value: int) -> None:
        """Set frames written - only used by tests."""
        if self._file_manager is not None:
            self._file_manager._frames_written = value

    @property
    def _mic_buffer(self) -> list[np.ndarray]:
        """Get mic buffer (backwards compatibility)."""
        return self._interleaver.mic_buffer if self._interleaver else []

    @_mic_buffer.setter
    def _mic_buffer(self, value: list[np.ndarray]) -> None:
        """Set mic buffer - only used by tests."""
        if self._interleaver is not None:
            self._interleaver._mic_buffer = value

    @property
    def _speaker_buffer(self) -> list[np.ndarray]:
        """Get speaker buffer (backwards compatibility)."""
        return self._interleaver.speaker_buffer if self._interleaver else []

    @_speaker_buffer.setter
    def _speaker_buffer(self, value: list[np.ndarray]) -> None:
        """Set speaker buffer - only used by tests."""
        if self._interleaver is not None:
            self._interleaver._speaker_buffer = value

    @property
    def _stop_interleave(self) -> threading.Event:
        """Get stop interleave event (backwards compatibility)."""
        return self._interleaver._stop_event if self._interleaver else threading.Event()

    @property
    def _interleave_thread(self) -> threading.Thread | None:
        """Get interleave thread (backwards compatibility)."""
        return self._interleaver._thread if self._interleaver else None

    @_interleave_thread.setter
    def _interleave_thread(self, value: threading.Thread | None) -> None:
        """Set interleave thread - only used by tests."""
        if self._interleaver is not None:
            self._interleaver._thread = value

    @property
    def _lock(self) -> threading.Lock:
        """Get lock (backwards compatibility)."""
        if self._file_manager is not None:
            return self._file_manager._lock
        return threading.Lock()

    def start(self) -> None:
        """
        Start recording audio from configured source(s).

        Raises:
            NoMicrophoneError: If no microphone is available (for mic/both modes).
            NoLoopbackDeviceError: If no loopback device is available (for speaker/both modes).
            AudioDeviceBusyError: If an audio device is busy.
        """
        if self._is_recording:
            return

        # Validate source availability
        errors = validate_source_availability(self._input_source)
        if errors:
            error_msg = errors[0]
            if "loopback" in error_msg.lower():
                raise NoLoopbackDeviceError()
            raise NoMicrophoneError(error_msg)

        # Get devices for the configured source
        self._mic_device, self._speaker_device = get_devices_for_source(self._input_source)

        # Create file manager and initialize recording file
        self._file_manager = RecordingFileManager(
            temp_dir=self._temp_dir,
            sample_rate=self._sample_rate,
            channels=self._channels,
        )
        self._file_manager.create()

        # Create interleaver for dual-stream mode
        if self._input_source == InputSource.BOTH:
            self._interleaver = DualStreamInterleaver(self._file_manager)

        # Start appropriate stream(s) based on input source
        try:
            if self._input_source == InputSource.BOTH:
                self._start_dual_streams()
            else:
                self._start_single_stream()
        except sd.PortAudioError as e:
            self._cleanup()
            if "No Default Input Device" in str(e) or "Invalid device" in str(e):
                if self._input_source == InputSource.SPEAKER:
                    raise NoLoopbackDeviceError() from e
                raise NoMicrophoneError("No microphone detected") from e
            raise AudioDeviceBusyError(f"Audio device error: {e}") from e
        except Exception as e:
            self._cleanup()
            raise AudioDeviceBusyError(f"Failed to start audio stream: {e}") from e

        self._is_recording = True
        self._start_time = time.time()

    def _start_single_stream(self) -> None:
        """Start a single input stream for mic or speaker mode."""
        # Determine which device and env vars to use
        recording_env: dict[str, str] = {}
        is_wasapi = False
        wasapi_index: int | None = None

        if self._input_source == InputSource.MIC:
            device: int | str | None = self._mic_device.device_index if self._mic_device else None
        else:  # SPEAKER
            # Use backend-specific recording config if available
            if self._speaker_device and self._speaker_device.recording_config:
                recording_env = self._speaker_device.recording_config.env
                device = self._speaker_device.recording_config.device
                is_wasapi, wasapi_index = _is_wasapi_device(device)
            else:
                device = self._speaker_device.device_index if self._speaker_device else None

        # WASAPI loopback (Windows)
        if is_wasapi:
            if not _check_wasapi_available():
                raise AudioDeviceBusyError(
                    "WASAPI loopback requires PyAudioWPatch. "
                    "Install with: pip install PyAudioWPatch"
                )
            self._start_wasapi_stream(wasapi_index)
            return

        # Use env_vars context manager to temporarily set env vars during stream creation
        # This ensures PULSE_SOURCE etc. are only set when needed and cleaned up after
        with env_vars(recording_env):
            self._stream = sd.InputStream(
                device=device,
                callback=self._audio_callback,
                channels=self._channels,
                samplerate=self._sample_rate,
                blocksize=self._buffer_size,
                dtype=np.float32,
                latency="low",
            )
            self._stream.start()

    def _start_dual_streams(self) -> None:
        """Start both mic and speaker streams for combined capture."""
        # Start interleaver
        if self._interleaver:
            self._interleaver.start()

        # Start mic stream (mono, will be left channel)
        if self._mic_device:
            self._mic_stream = sd.InputStream(
                device=self._mic_device.device_index,
                callback=self._mic_callback,
                channels=1,
                samplerate=self._sample_rate,
                blocksize=self._buffer_size,
                dtype=np.float32,
                latency="low",
            )

        # Start speaker stream (mono, will be right channel)
        if self._speaker_device:
            # Use backend-specific recording config if available
            recording_env: dict[str, str] = {}
            if self._speaker_device.recording_config:
                recording_env = self._speaker_device.recording_config.env
                speaker_device: int | str | None = self._speaker_device.recording_config.device
            else:
                speaker_device = self._speaker_device.device_index

            # Check for WASAPI loopback (Windows)
            is_wasapi, wasapi_index = _is_wasapi_device(speaker_device)
            if is_wasapi:
                if not _check_wasapi_available():
                    raise AudioDeviceBusyError(
                        "WASAPI loopback requires PyAudioWPatch. "
                        "Install with: pip install PyAudioWPatch"
                    )
                self._start_wasapi_speaker_stream(wasapi_index)
            else:
                # Use env_vars context manager to temporarily set env vars during stream creation
                # This ensures PULSE_SOURCE etc. are only set when needed and cleaned up after
                with env_vars(recording_env):
                    self._speaker_stream = sd.InputStream(
                        device=speaker_device,
                        callback=self._speaker_callback,
                        channels=1,
                        samplerate=self._sample_rate,
                        blocksize=self._buffer_size,
                        dtype=np.float32,
                        latency="low",
                    )

        # Start streams
        if self._mic_stream:
            self._mic_stream.start()
        if self._speaker_stream:
            self._speaker_stream.start()

    def _mic_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: dict,
        status: sd.CallbackFlags,
    ) -> None:
        """Callback for microphone stream in dual-stream mode."""
        if not self._is_recording:
            return

        # Check max duration
        if self._start_time and (time.time() - self._start_time) >= self._max_duration:
            self._is_recording = False
            return

        # Calculate RMS level for UI feedback (use mic for level in both mode)
        if self._level_callback:
            rms = float(np.sqrt(np.mean(indata**2)))
            self._level_callback(rms)

        # Add to interleaver buffer
        if self._interleaver:
            self._interleaver.add_mic_data(indata)

    def _speaker_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: dict,
        status: sd.CallbackFlags,
    ) -> None:
        """Callback for speaker/loopback stream in dual-stream mode."""
        if not self._is_recording:
            return

        # Add to interleaver buffer
        if self._interleaver:
            self._interleaver.add_speaker_data(indata)

    def _start_wasapi_stream(self, device_index: int | None) -> None:
        """Start a WASAPI loopback stream on Windows.

        Args:
            device_index: PyAudioWPatch device index, or None for default.
        """
        import pyaudiowpatch as pyaudio  # pyrefly: ignore[missing-import]

        self._pyaudio_instance = pyaudio.PyAudio()

        # Get device info
        if device_index is not None:
            device_info = self._pyaudio_instance.get_device_info_by_index(device_index)
        else:
            device_info = self._pyaudio_instance.get_default_wasapi_loopback()
            device_index = device_info["index"]

        # Open stream with callback
        self._wasapi_stream = self._pyaudio_instance.open(
            format=pyaudio.paFloat32,
            channels=min(self._channels, int(device_info["maxInputChannels"])),
            rate=int(device_info["defaultSampleRate"]),
            input=True,
            input_device_index=device_index,
            frames_per_buffer=self._buffer_size,
            stream_callback=self._wasapi_callback,
        )
        self._wasapi_stream.start_stream()

    def _wasapi_callback(
        self,
        in_data: bytes | None,
        frame_count: int,
        time_info: dict[str, float],
        status: int,
    ) -> tuple[None, int]:
        """Callback for PyAudioWPatch WASAPI stream.

        Args:
            in_data: Raw audio bytes.
            frame_count: Number of frames.
            time_info: Time information dict.
            status: Status flags.

        Returns:
            Tuple of (None, continue_flag).
        """
        import pyaudiowpatch as pyaudio  # pyrefly: ignore[missing-import]

        if not self._is_recording:
            return (None, pyaudio.paComplete)

        # Check max duration
        if self._start_time and (time.time() - self._start_time) >= self._max_duration:
            self._is_recording = False
            return (None, pyaudio.paComplete)

        if in_data is None:
            return (None, pyaudio.paContinue)

        # Convert bytes to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        audio_data = audio_data.reshape(-1, self._channels)

        # Calculate RMS level for UI feedback
        if self._level_callback:
            rms = float(np.sqrt(np.mean(audio_data**2)))
            self._level_callback(rms)

        # Write to file via file manager
        if self._file_manager:
            self._file_manager.write(audio_data)

        return (None, pyaudio.paContinue)

    def _start_wasapi_speaker_stream(self, device_index: int | None) -> None:
        """Start a WASAPI loopback stream for speaker in dual-stream mode.

        Args:
            device_index: PyAudioWPatch device index, or None for default.
        """
        import pyaudiowpatch as pyaudio  # pyrefly: ignore[missing-import]

        self._pyaudio_instance = pyaudio.PyAudio()

        # Get device info
        if device_index is not None:
            device_info = self._pyaudio_instance.get_device_info_by_index(device_index)
        else:
            device_info = self._pyaudio_instance.get_default_wasapi_loopback()
            device_index = device_info["index"]

        # Open stream with callback for speaker (mono for interleaving)
        self._wasapi_stream = self._pyaudio_instance.open(
            format=pyaudio.paFloat32,
            channels=1,  # Mono for dual-stream interleaving
            rate=int(device_info["defaultSampleRate"]),
            input=True,
            input_device_index=device_index,
            frames_per_buffer=self._buffer_size,
            stream_callback=self._wasapi_speaker_callback,
        )
        self._wasapi_stream.start_stream()

    def _wasapi_speaker_callback(
        self,
        in_data: bytes | None,
        frame_count: int,
        time_info: dict[str, float],
        status: int,
    ) -> tuple[None, int]:
        """Callback for PyAudioWPatch WASAPI stream in dual-stream mode.

        Args:
            in_data: Raw audio bytes.
            frame_count: Number of frames.
            time_info: Time information dict.
            status: Status flags.

        Returns:
            Tuple of (None, continue_flag).
        """
        import pyaudiowpatch as pyaudio  # pyrefly: ignore[missing-import]

        if not self._is_recording:
            return (None, pyaudio.paComplete)

        if in_data is None:
            return (None, pyaudio.paContinue)

        # Convert bytes to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        audio_data = audio_data.reshape(-1, 1)  # Mono

        # Add to interleaver buffer
        if self._interleaver:
            self._interleaver.add_speaker_data(audio_data)

        return (None, pyaudio.paContinue)

    def _interleave_buffers(self) -> None:
        """Thread that interleaves mic and speaker buffers into stereo.

        Note: This method is kept for backwards compatibility with tests.
        The actual interleaving is now handled by DualStreamInterleaver.
        """
        if self._interleaver:
            self._interleaver._interleave_loop()

    def stop(self) -> Path:
        """
        Stop recording and return the path to the recorded audio file.

        Returns:
            Path to the temporary WAV file containing the recording.
        """
        if not self._is_recording:
            if self._file_manager and self._file_manager.file_path:
                return self._file_manager.file_path
            raise RuntimeError("Recording was never started")

        self._is_recording = False

        # Stop interleaver thread if running
        if self._interleaver is not None:
            self._interleaver.stop()

        # Stop and close single stream
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except (sd.PortAudioError, OSError):
                pass
            self._stream = None

        # Stop and close dual streams
        if self._mic_stream is not None:
            try:
                self._mic_stream.stop()
                self._mic_stream.close()
            except (sd.PortAudioError, OSError):
                pass
            self._mic_stream = None

        if self._speaker_stream is not None:
            try:
                self._speaker_stream.stop()
                self._speaker_stream.close()
            except (sd.PortAudioError, OSError):
                pass
            self._speaker_stream = None

        # Stop and close WASAPI stream (Windows)
        if self._wasapi_stream is not None:
            try:
                self._wasapi_stream.stop_stream()
                self._wasapi_stream.close()
            except Exception:
                pass
            self._wasapi_stream = None

        if self._pyaudio_instance is not None:
            with contextlib.suppress(Exception):
                self._pyaudio_instance.terminate()
            self._pyaudio_instance = None

        # Close file manager
        if self._file_manager is not None:
            self._file_manager.close()

        if self._file_manager is None or self._file_manager.file_path is None:
            raise RuntimeError("No temp file created")

        return self._file_manager.file_path

    def _flush_remaining_buffers(self) -> None:
        """Flush any remaining data in the dual-stream buffers.

        Note: This method is kept for backwards compatibility with tests.
        The actual flushing is now handled by DualStreamInterleaver.stop().
        """
        if self._interleaver:
            self._interleaver._flush_remaining()

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
        frames = self._file_manager.frames_written if self._file_manager else 0
        return frames / self._sample_rate

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

        # Write to file via file manager
        if self._file_manager:
            self._file_manager.write(indata)

    def _cleanup(self) -> None:
        """Clean up resources."""
        # Stop interleaver
        if self._interleaver is not None:
            self._interleaver.stop()
            self._interleaver = None

        # Clean up single stream
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        # Clean up dual streams
        if self._mic_stream is not None:
            try:
                self._mic_stream.stop()
                self._mic_stream.close()
            except Exception:
                pass
            self._mic_stream = None

        if self._speaker_stream is not None:
            try:
                self._speaker_stream.stop()
                self._speaker_stream.close()
            except Exception:
                pass
            self._speaker_stream = None

        # Clean up WASAPI stream (Windows)
        if self._wasapi_stream is not None:
            try:
                self._wasapi_stream.stop_stream()
                self._wasapi_stream.close()
            except Exception:
                pass
            self._wasapi_stream = None

        if self._pyaudio_instance is not None:
            with contextlib.suppress(Exception):
                self._pyaudio_instance.terminate()
            self._pyaudio_instance = None

        # Clean up file manager
        if self._file_manager is not None:
            self._file_manager.cleanup()
            self._file_manager = None

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
