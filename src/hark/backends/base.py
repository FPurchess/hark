"""Abstract base classes and protocols for backends.

These define the contracts that backend implementations must follow.
Using Protocol allows for structural subtyping (duck typing with type safety).
"""

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from hark.constants import (
    DEFAULT_BEAM_SIZE,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_VAD_FILTER,
    DEFAULT_VAD_MIN_SILENCE_MS,
)

# ============================================================
# Data Classes for Transcription
# ============================================================


@dataclass
class WordInfo:
    """Information about a single word."""

    start: float
    end: float
    word: str
    probability: float = 1.0


@dataclass
class TranscriptionSegment:
    """A transcription segment from any backend."""

    start: float
    end: float
    text: str
    words: list[WordInfo] = field(default_factory=list)


@dataclass
class TranscriptionOutput:
    """Output from transcription."""

    segments: list[TranscriptionSegment]
    language: str
    language_probability: float
    duration: float = 0.0

    @property
    def text(self) -> str:
        """Get full transcription text."""
        return " ".join(seg.text for seg in self.segments).strip()


# ============================================================
# Data Classes for Diarization
# ============================================================


@dataclass
class DiarizedSegment:
    """A segment with speaker information."""

    start: float
    end: float
    text: str
    speaker: str
    words: list[WordInfo] = field(default_factory=list)


@dataclass
class DiarizationOutput:
    """Output from diarization."""

    segments: list[DiarizedSegment]
    speakers: list[str]
    language: str
    language_probability: float
    duration: float


# ============================================================
# Backend Protocols
# ============================================================


@runtime_checkable
class TranscriptionBackend(Protocol):
    """Protocol for transcription backends.

    Implementations must provide methods to load a model and transcribe audio.
    The protocol uses structural subtyping - any class with matching methods
    will satisfy the protocol.

    Example implementation:
        class MyBackend:
            def load_model(self, model_name, device, compute_type, download_root):
                ...
            def transcribe(self, audio, language=None, word_timestamps=False):
                ...
            def is_loaded(self):
                ...
    """

    def load_model(
        self,
        model_name: str,
        device: str,
        compute_type: str,
        download_root: str,
    ) -> None:
        """Load the transcription model.

        Args:
            model_name: Name of the model to load (e.g., "base", "small", "medium").
            device: Device to use ("cpu", "cuda", "auto").
            compute_type: Compute type (e.g., "int8", "float16").
            download_root: Directory to cache downloaded models.
        """
        ...

    def transcribe(
        self,
        audio: np.ndarray,
        language: str | None = None,
        word_timestamps: bool = False,
        beam_size: int = DEFAULT_BEAM_SIZE,
        vad_filter: bool = DEFAULT_VAD_FILTER,
        vad_min_silence_ms: int = DEFAULT_VAD_MIN_SILENCE_MS,
    ) -> TranscriptionOutput:
        """Transcribe audio.

        Args:
            audio: Audio data as float32 numpy array, mono, 16kHz.
            language: Language code or None for auto-detection.
            word_timestamps: Whether to include word-level timestamps.
            beam_size: Beam size for decoding (default: 5).
            vad_filter: Enable VAD filtering (default: True).
            vad_min_silence_ms: Minimum silence duration in ms for VAD (default: 500).

        Returns:
            TranscriptionOutput with segments, language, and probability.

        Raises:
            RuntimeError: If model is not loaded.
        """
        ...

    def is_loaded(self) -> bool:
        """Check if model is loaded.

        Returns:
            True if model is loaded and ready for transcription.
        """
        ...


@runtime_checkable
class DiarizationBackend(Protocol):
    """Protocol for diarization backends.

    Implementations must provide methods to load models and perform
    transcription with speaker diarization.
    """

    def load_model(
        self,
        model_name: str,
        device: str,
        compute_type: str,
        download_root: str,
        hf_token: str,
    ) -> None:
        """Load diarization models.

        Args:
            model_name: Name of the model to load.
            device: Device to use ("cpu", "cuda", "auto").
            compute_type: Compute type.
            download_root: Directory to cache downloaded models.
            hf_token: HuggingFace token for accessing gated models.
        """
        ...

    def transcribe_and_diarize(
        self,
        audio: np.ndarray,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        language: str | None = None,
        num_speakers: int | None = None,
    ) -> DiarizationOutput:
        """Transcribe and diarize audio.

        Args:
            audio: Audio data as float32 numpy array.
            sample_rate: Sample rate of audio.
            language: Language code or None for auto-detection.
            num_speakers: Expected number of speakers, or None for auto.

        Returns:
            DiarizationOutput with speaker-labeled segments.

        Raises:
            RuntimeError: If model is not loaded.
        """
        ...

    def is_loaded(self) -> bool:
        """Check if models are loaded.

        Returns:
            True if models are loaded and ready for diarization.
        """
        ...
