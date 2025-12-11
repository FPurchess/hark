"""Transcription engine using faster-whisper."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from hark.constants import DEFAULT_MODEL_CACHE_DIR, VALID_MODELS
from hark.device import detect_best_device, get_compute_type
from hark.exceptions import ModelDownloadError, ModelNotFoundError, TranscriptionError

if TYPE_CHECKING:
    from hark.backends.base import TranscriptionBackend

__all__ = [
    "WordSegment",
    "TranscriptionSegment",
    "TranscriptionResult",
    "Transcriber",
]


@dataclass
class WordSegment:
    """A single word with timing information."""

    start: float
    end: float
    word: str


@dataclass
class TranscriptionSegment:
    """A single transcription segment."""

    start: float
    end: float
    text: str
    words: list[WordSegment] = field(default_factory=list)


@dataclass
class TranscriptionResult:
    """Complete transcription result."""

    text: str
    segments: list[TranscriptionSegment]
    language: str
    language_probability: float
    duration: float


class Transcriber:
    """Transcription engine using faster-whisper.

    Supports dependency injection for testing via the `backend` parameter.
    If no backend is provided, uses faster-whisper directly (default behavior).

    Example with dependency injection:
        from hark.backends import FasterWhisperBackend
        backend = FasterWhisperBackend()
        transcriber = Transcriber(backend=backend)

    Example with mock for testing:
        mock_backend = MagicMock(spec=TranscriptionBackend)
        transcriber = Transcriber(backend=mock_backend)
    """

    def __init__(
        self,
        model_name: str = "base",
        device: str = "auto",
        compute_type: str | None = None,
        model_cache_dir: Path | None = None,
        backend: TranscriptionBackend | None = None,
    ) -> None:
        """
        Initialize the transcriber.

        Args:
            model_name: Whisper model name (tiny, base, small, medium, large, etc.).
            device: Compute device (auto, cpu, cuda, vulkan).
            compute_type: Compute type for inference (auto-detected if None).
            model_cache_dir: Directory for caching models.
            backend: Optional backend for dependency injection. If provided,
                     the backend handles model loading and transcription.
                     If None, uses faster-whisper directly (default).
        """
        if model_name not in VALID_MODELS:
            raise ValueError(f"Invalid model '{model_name}'. Valid: {', '.join(VALID_MODELS)}")

        self._model_name = model_name
        self._requested_device = device
        self._compute_type = compute_type
        self._cache_dir = model_cache_dir or DEFAULT_MODEL_CACHE_DIR
        self._model = None
        self._actual_device: str | None = None
        self._backend = backend

    def load_model(self) -> None:
        """
        Load the Whisper model.

        Downloads the model if not cached.

        Raises:
            ModelNotFoundError: If model cannot be found or loaded.
            ModelDownloadError: If model download fails.
        """
        # Resolve device
        if self._requested_device == "auto":
            self._actual_device = detect_best_device()
        else:
            self._actual_device = self._requested_device

        # Handle Vulkan - faster-whisper doesn't directly support it,
        # fall back to CPU for now
        if self._actual_device == "vulkan":
            self._actual_device = "cpu"

        # Determine compute type
        compute_type = self._compute_type or get_compute_type(self._actual_device)

        # Ensure cache directory exists
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # If using injected backend, delegate to it
        if self._backend is not None:
            try:
                self._backend.load_model(
                    model_name=self._model_name,
                    device=self._actual_device,
                    compute_type=compute_type,
                    download_root=str(self._cache_dir),
                )
            except Exception as e:
                error_str = str(e).lower()
                if "download" in error_str or "network" in error_str:
                    raise ModelDownloadError(f"Failed to download model: {e}") from e
                raise ModelNotFoundError(f"Failed to load model: {e}") from e
            return

        # Default: use faster-whisper directly
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise ModelNotFoundError(
                "faster-whisper is not installed. Install it with: pip install faster-whisper"
            ) from e

        try:
            self._model = WhisperModel(
                self._model_name,
                device=self._actual_device,
                compute_type=compute_type,
                download_root=str(self._cache_dir),
            )
        except Exception as e:
            error_str = str(e).lower()
            if "download" in error_str or "network" in error_str:
                raise ModelDownloadError(f"Failed to download model: {e}") from e
            raise ModelNotFoundError(f"Failed to load model: {e}") from e

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: str | None = None,
        word_timestamps: bool = False,
        progress_callback: Callable[[float], None] | None = None,
    ) -> TranscriptionResult:
        """
        Transcribe audio.

        Args:
            audio: Audio data as numpy array (float32).
            sample_rate: Audio sample rate in Hz.
            language: Language code (None for auto-detection).
            word_timestamps: Include word-level timestamps.
            progress_callback: Callback for progress updates (0.0-1.0).

        Returns:
            TranscriptionResult with text and segments.

        Raises:
            TranscriptionError: If transcription fails.
        """
        # Ensure model is loaded
        if not self.is_model_loaded():
            self.load_model()

        # Resample to 16kHz if needed (Whisper requirement)
        if sample_rate != 16000:
            try:
                import librosa

                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            except ImportError as e:
                raise TranscriptionError(
                    "Audio sample rate must be 16000Hz, or librosa must be installed for resampling"
                ) from e

        # Ensure float32
        audio = audio.astype(np.float32)

        # Calculate total duration for progress estimation
        total_duration = len(audio) / 16000

        # If using injected backend, delegate to it
        if self._backend is not None:
            try:
                result = self._backend.transcribe(
                    audio=audio,
                    language=language,
                    word_timestamps=word_timestamps,
                )

                # Convert backend result to TranscriptionResult
                segments = [
                    TranscriptionSegment(
                        start=seg.start,
                        end=seg.end,
                        text=seg.text,
                        words=[
                            WordSegment(start=w.start, end=w.end, word=w.word) for w in seg.words
                        ],
                    )
                    for seg in result.segments
                ]

                if progress_callback:
                    progress_callback(1.0)

                return TranscriptionResult(
                    text=result.text,
                    segments=segments,
                    language=result.language,
                    language_probability=result.language_probability,
                    duration=total_duration,
                )
            except Exception as e:
                raise TranscriptionError(f"Transcription failed: {e}") from e

        # Default path: use faster-whisper directly
        assert self._model is not None  # for type checker after load_model()

        try:
            # Transcribe with faster-whisper
            segments_generator, info = self._model.transcribe(
                audio,
                language=language,
                word_timestamps=word_timestamps,
                beam_size=5,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
            )

            # Collect segments
            segments = []
            full_text_parts = []

            for segment in segments_generator:
                # Create segment
                words = []
                if segment.words:
                    words = [
                        WordSegment(start=w.start, end=w.end, word=w.word) for w in segment.words
                    ]

                segments.append(
                    TranscriptionSegment(
                        start=segment.start,
                        end=segment.end,
                        text=segment.text.strip(),
                        words=words,
                    )
                )
                full_text_parts.append(segment.text)

                # Progress callback
                if progress_callback and total_duration > 0:
                    progress = min(segment.end / total_duration, 1.0)
                    progress_callback(progress)

            # Final progress
            if progress_callback:
                progress_callback(1.0)

            # If language was explicitly specified, confidence is 100%
            language_probability = 1.0 if language else info.language_probability

            return TranscriptionResult(
                text=" ".join(full_text_parts).strip(),
                segments=segments,
                language=info.language,
                language_probability=language_probability,
                duration=total_duration,
            )

        except Exception as e:
            raise TranscriptionError(f"Transcription failed: {e}") from e

    def is_model_loaded(self) -> bool:
        """Check if the model is loaded."""
        if self._backend is not None:
            return self._backend.is_loaded()
        return self._model is not None

    @property
    def device(self) -> str | None:
        """Get the actual device being used."""
        return self._actual_device

    @staticmethod
    def list_models() -> list[str]:
        """List available model names."""
        return list(VALID_MODELS)
