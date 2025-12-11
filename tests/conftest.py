"""Shared pytest fixtures for hark tests."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from hark.config import (
    HarkConfig,
    InterfaceConfig,
    NoiseReductionConfig,
    NormalizationConfig,
    OutputConfig,
    PreprocessingConfig,
    RecordingConfig,
    SilenceTrimmingConfig,
    WhisperConfig,
)
from hark.transcriber import TranscriptionResult, TranscriptionSegment, WordSegment


@pytest.fixture
def sample_audio() -> np.ndarray:
    """Generate sample audio data (1 second of sine wave at 440Hz)."""
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    # 440Hz sine wave with some noise
    audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(len(t)).astype(np.float32)
    return audio


@pytest.fixture
def silent_audio() -> np.ndarray:
    """Generate silent audio data (1 second of near-silence)."""
    sample_rate = 16000
    duration = 1.0
    # Very quiet noise (effectively silent)
    audio = 1e-9 * np.random.randn(int(sample_rate * duration)).astype(np.float32)
    return audio


@pytest.fixture
def stereo_audio() -> np.ndarray:
    """Generate stereo audio data."""
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    left = 0.5 * np.sin(2 * np.pi * 440 * t)
    right = 0.5 * np.sin(2 * np.pi * 880 * t)
    return np.column_stack([left, right]).astype(np.float32)


@pytest.fixture
def sample_transcription_result() -> TranscriptionResult:
    """Create a sample transcription result for testing."""
    return TranscriptionResult(
        text="Hello world. This is a test.",
        segments=[
            TranscriptionSegment(
                start=0.0,
                end=1.5,
                text="Hello world.",
                words=[
                    WordSegment(start=0.0, end=0.5, word="Hello"),
                    WordSegment(start=0.6, end=1.0, word="world."),
                ],
            ),
            TranscriptionSegment(
                start=1.6,
                end=3.0,
                text="This is a test.",
                words=[
                    WordSegment(start=1.6, end=1.8, word="This"),
                    WordSegment(start=1.9, end=2.1, word="is"),
                    WordSegment(start=2.2, end=2.4, word="a"),
                    WordSegment(start=2.5, end=3.0, word="test."),
                ],
            ),
        ],
        language="en",
        language_probability=0.95,
        duration=3.0,
    )


@pytest.fixture
def empty_transcription_result() -> TranscriptionResult:
    """Create an empty transcription result."""
    return TranscriptionResult(
        text="",
        segments=[],
        language="en",
        language_probability=0.5,
        duration=0.0,
    )


@pytest.fixture
def long_transcription_result() -> TranscriptionResult:
    """Create a transcription result with duration > 1 hour."""
    return TranscriptionResult(
        text="This is a long recording.",
        segments=[
            TranscriptionSegment(
                start=0.0,
                end=3661.5,  # Just over 1 hour
                text="This is a long recording.",
                words=[],
            ),
        ],
        language="en",
        language_probability=0.99,
        duration=3661.5,
    )


@pytest.fixture
def default_config() -> HarkConfig:
    """Create a default HarkConfig for testing."""
    return HarkConfig()


@pytest.fixture
def custom_config() -> HarkConfig:
    """Create a custom HarkConfig with non-default values."""
    return HarkConfig(
        recording=RecordingConfig(
            sample_rate=44100,
            channels=2,
            max_duration=300,
        ),
        whisper=WhisperConfig(
            model="large-v3",
            language="de",
            device="cuda",
        ),
        preprocessing=PreprocessingConfig(
            noise_reduction=NoiseReductionConfig(enabled=False, strength=0.3),
            normalization=NormalizationConfig(enabled=True, target_level_db=-15.0),
            silence_trimming=SilenceTrimmingConfig(enabled=False, threshold_db=-50.0),
        ),
        output=OutputConfig(
            format="markdown",
            timestamps=True,
            append_mode=True,
        ),
        interface=InterfaceConfig(
            quiet=True,
            verbose=False,
        ),
    )


@pytest.fixture
def sample_config_yaml() -> str:
    """Sample YAML configuration content."""
    return """\
recording:
  sample_rate: 16000
  channels: 1
  max_duration: 600

whisper:
  model: base
  language: auto
  device: auto

preprocessing:
  noise_reduction:
    enabled: true
    strength: 0.5
  normalization:
    enabled: true
    target_level: -20
  silence_trimming:
    enabled: true
    threshold: -40
    min_silence_duration: 0.5

output:
  format: plain
  timestamps: false
  append_mode: false
  encoding: utf-8

interface:
  quiet: false
  verbose: false
  color_output: true
"""


@pytest.fixture
def partial_config_yaml() -> str:
    """Partial YAML configuration (missing sections)."""
    return """\
whisper:
  model: small
  language: en

output:
  format: markdown
"""


@pytest.fixture
def invalid_config_yaml() -> str:
    """Invalid YAML content."""
    return """\
recording:
  sample_rate: not_a_number
  channels: [invalid
"""


@pytest.fixture
def cli_args_namespace() -> argparse.Namespace:
    """Create a sample argparse.Namespace with CLI arguments."""
    return argparse.Namespace(
        output_file="output.txt",
        max_duration=120,
        sample_rate=None,
        channels=None,
        input_source=None,
        lang="en",
        model="small",
        no_noise_reduction=True,
        no_normalize=False,
        no_trim_silence=False,
        noise_strength=0.7,
        timestamps=True,
        format="markdown",
        append=False,
        quiet=False,
        verbose=True,
        config=None,
    )


@pytest.fixture
def empty_cli_args_namespace() -> argparse.Namespace:
    """Create an empty argparse.Namespace (all None/False)."""
    return argparse.Namespace(
        output_file=None,
        max_duration=None,
        sample_rate=None,
        channels=None,
        input_source=None,
        lang=None,
        model=None,
        no_noise_reduction=False,
        no_normalize=False,
        no_trim_silence=False,
        noise_strength=None,
        timestamps=False,
        format=None,
        append=False,
        quiet=False,
        verbose=False,
        config=None,
    )


@pytest.fixture
def temp_audio_file(tmp_path: Path, sample_audio: np.ndarray) -> Path:
    """Create a temporary WAV file with sample audio."""
    import soundfile as sf

    audio_path = tmp_path / "test_audio.wav"
    sf.write(audio_path, sample_audio, 16000)
    return audio_path


@pytest.fixture
def temp_silent_audio_file(tmp_path: Path, silent_audio: np.ndarray) -> Path:
    """Create a temporary WAV file with silent audio."""
    import soundfile as sf

    audio_path = tmp_path / "silent_audio.wav"
    sf.write(audio_path, silent_audio, 16000)
    return audio_path


@pytest.fixture
def temp_config_file(tmp_path: Path, sample_config_yaml: str) -> Path:
    """Create a temporary config file."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(sample_config_yaml)
    return config_path


@pytest.fixture
def mock_whisper_model() -> MagicMock:
    """Create a mock WhisperModel for faster-whisper."""
    mock = MagicMock()

    # Mock segment object
    mock_segment = MagicMock()
    mock_segment.start = 0.0
    mock_segment.end = 1.5
    mock_segment.text = " Hello world."
    mock_segment.words = None

    # Mock info object
    mock_info = MagicMock()
    mock_info.language = "en"
    mock_info.language_probability = 0.95

    # transcribe returns (generator, info)
    mock.transcribe.return_value = (iter([mock_segment]), mock_info)

    return mock


@pytest.fixture
def mock_sounddevice() -> MagicMock:
    """Create a mock for sounddevice module."""
    mock = MagicMock()

    # Mock InputStream
    mock_stream = MagicMock()
    mock_stream.start = MagicMock()
    mock_stream.stop = MagicMock()
    mock_stream.close = MagicMock()
    mock.InputStream.return_value = mock_stream

    # Mock default device
    mock.default.device = (0, 0)

    # Mock query_devices
    mock.query_devices.return_value = [
        {
            "name": "Test Microphone",
            "max_input_channels": 2,
            "max_output_channels": 0,
            "default_samplerate": 44100.0,
        },
        {
            "name": "Test Speaker",
            "max_input_channels": 0,
            "max_output_channels": 2,
            "default_samplerate": 44100.0,
        },
    ]

    return mock


@pytest.fixture
def mock_soundfile() -> MagicMock:
    """Create a mock for soundfile module."""
    mock = MagicMock()

    # Mock SoundFile context manager
    mock_sf = MagicMock()
    mock_sf.write = MagicMock()
    mock_sf.close = MagicMock()
    mock_sf.closed = False
    mock.SoundFile.return_value = mock_sf

    # Mock read function
    mock.read.return_value = (np.zeros(16000, dtype=np.float32), 16000)

    return mock


@pytest.fixture
def mock_stereo_config() -> MagicMock:
    """Create mock HarkConfig for stereo processor tests."""
    config = MagicMock()
    config.whisper.model = "base"
    config.whisper.device = "cpu"
    config.whisper.language = "auto"
    config.model_cache_dir = Path("/tmp/models")
    config.diarization.hf_token = "test_token"
    config.diarization.local_speaker_name = "LOCAL"
    return config


@pytest.fixture
def diarized_segment_factory():
    """Factory for creating test DiarizedSegment objects."""
    from hark.diarizer import DiarizedSegment

    def _make(
        start: float,
        end: float,
        text: str,
        speaker: str,
        words: list | None = None,
    ) -> DiarizedSegment:
        return DiarizedSegment(
            start=start,
            end=end,
            text=text,
            speaker=speaker,
            words=words or [],
        )

    return _make


@pytest.fixture
def sample_diarization_result():
    """Create a sample DiarizationResult for testing."""
    from hark.diarizer import DiarizationResult, DiarizedSegment

    return DiarizationResult(
        segments=[
            DiarizedSegment(
                start=0.0,
                end=2.0,
                text="Hello from speaker one.",
                speaker="SPEAKER_01",
                words=[],
            ),
            DiarizedSegment(
                start=2.5,
                end=4.0,
                text="Response from speaker two.",
                speaker="SPEAKER_02",
                words=[],
            ),
        ],
        speakers=["SPEAKER_01", "SPEAKER_02"],
        language="en",
        language_probability=0.95,
        duration=4.0,
    )
