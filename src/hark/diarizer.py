"""Speaker diarization using WhisperX.

Supports dependency injection for testing via the `backend` parameter.
If no backend is provided, uses WhisperX directly (default behavior).
"""

from __future__ import annotations

import contextlib
import io
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from hark.constants import DEFAULT_MODEL_CACHE_DIR, VALID_MODELS
from hark.device import detect_best_device, get_compute_type
from hark.exceptions import (
    DependencyMissingError,
    DiarizationError,
    GatedModelError,
    MissingTokenError,
)

if TYPE_CHECKING:
    from hark.backends.base import DiarizationBackend


@contextlib.contextmanager
def _suppress_output():
    """Temporarily suppress stdout/stderr to hide noisy library output."""
    # Save original streams
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        # Redirect to devnull
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        yield
    finally:
        # Restore original streams
        sys.stdout = old_stdout
        sys.stderr = old_stderr


__all__ = [
    "WordSegment",
    "DiarizedSegment",
    "DiarizationResult",
    "Diarizer",
]


@dataclass
class WordSegment:
    """A single word with timing and speaker information."""

    start: float
    end: float
    word: str
    speaker: str | None = None


@dataclass
class DiarizedSegment:
    """A transcription segment with speaker information."""

    start: float
    end: float
    text: str
    speaker: str  # "SPEAKER_01", "SPEAKER_02", or custom name
    words: list[WordSegment] = field(default_factory=list)


@dataclass
class DiarizationResult:
    """Complete diarization result."""

    segments: list[DiarizedSegment]
    speakers: list[str]  # Unique speaker labels detected
    language: str
    language_probability: float
    duration: float


class Diarizer:
    """WhisperX-based transcription with speaker diarization.

    Supports dependency injection for testing via the `backend` parameter.
    If no backend is provided, uses WhisperX directly (default behavior).

    Example with dependency injection:
        from hark.backends import WhisperXBackend
        backend = WhisperXBackend()
        diarizer = Diarizer(backend=backend)

    Example with mock for testing:
        mock_backend = MagicMock(spec=DiarizationBackend)
        diarizer = Diarizer(backend=mock_backend)
    """

    def __init__(
        self,
        model_name: str = "base",
        device: str = "auto",
        compute_type: str | None = None,
        hf_token: str | None = None,
        num_speakers: int | None = None,
        model_cache_dir: Path | None = None,
        backend: DiarizationBackend | None = None,
    ) -> None:
        """
        Initialize the diarizer.

        Args:
            model_name: Whisper model name (tiny, base, small, medium, large, etc.).
            device: Compute device (auto, cpu, cuda).
            compute_type: Compute type for inference (auto-detected if None).
            hf_token: HuggingFace token for pyannote models.
            num_speakers: Expected number of speakers (helps accuracy).
            model_cache_dir: Directory for caching models.
            backend: Optional backend for dependency injection. If provided,
                     the backend handles model loading and diarization.
                     If None, uses WhisperX directly (default).
        """
        if model_name not in VALID_MODELS:
            raise ValueError(f"Invalid model '{model_name}'. Valid: {', '.join(VALID_MODELS)}")

        self._model_name = model_name
        self._requested_device = device
        self._compute_type = compute_type
        self._hf_token = hf_token
        self._num_speakers = num_speakers
        self._cache_dir = model_cache_dir or DEFAULT_MODEL_CACHE_DIR
        self._backend = backend

        # Cached models (lazy loaded)
        self._whisperx_model = None
        self._actual_device: str | None = None
        self._actual_compute_type: str | None = None

    def _check_dependencies(self) -> None:
        """Check if WhisperX is installed."""
        try:
            import whisperx  # type: ignore[import-not-found]  # noqa: F401
        except ImportError as e:
            raise DependencyMissingError() from e

    def _check_token(self) -> None:
        """Check if HF token is configured."""
        if not self._hf_token:
            raise MissingTokenError()

    def _resolve_device(self) -> str:
        """Resolve the compute device to use."""
        if self._requested_device == "auto":
            device = detect_best_device()
            # WhisperX doesn't support Vulkan, fall back to CPU
            if device == "vulkan":
                return "cpu"
            return device
        return self._requested_device

    def _load_model(self):
        """Load and cache the WhisperX model."""
        if self._whisperx_model is not None:
            return self._whisperx_model

        import whisperx  # type: ignore[import-not-found]

        self._actual_device = self._resolve_device()
        self._actual_compute_type = self._compute_type or get_compute_type(self._actual_device)

        # Suppress noisy version mismatch warnings from bundled VAD model
        with _suppress_output():
            self._whisperx_model = whisperx.load_model(
                self._model_name,
                device=self._actual_device,
                compute_type=self._actual_compute_type,
                download_root=str(self._cache_dir),
            )

        return self._whisperx_model

    def transcribe_and_diarize(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: str | None = None,
    ) -> DiarizationResult:
        """
        Transcribe audio with speaker diarization.

        Uses WhisperX pipeline:
        1. Transcribe with faster-whisper
        2. Align with wav2vec2
        3. Diarize with pyannote
        4. Assign speakers to words

        Args:
            audio: Audio data as numpy array.
            sample_rate: Sample rate of audio (default 16000).
            language: Language code or None for auto-detect.

        Returns:
            DiarizationResult with speaker-labeled segments.

        Raises:
            DependencyMissingError: If WhisperX is not installed.
            MissingTokenError: If HF token is not configured.
            DiarizationError: If diarization fails.
        """
        # Ensure audio is float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # If using injected backend, delegate to it
        if self._backend is not None:
            # Load model if needed
            if not self._backend.is_loaded():
                self._actual_device = self._resolve_device()
                self._actual_compute_type = self._compute_type or get_compute_type(
                    self._actual_device
                )
                try:
                    self._backend.load_model(
                        model_name=self._model_name,
                        device=self._actual_device,
                        compute_type=self._actual_compute_type,
                        download_root=str(self._cache_dir),
                        hf_token=self._hf_token or "",
                    )
                except Exception as e:
                    raise DiarizationError(f"Failed to load model: {e}") from e

            try:
                result = self._backend.transcribe_and_diarize(
                    audio=audio,
                    sample_rate=sample_rate,
                    language=language,
                    num_speakers=self._num_speakers,
                )

                # Convert backend result to DiarizationResult
                segments = [
                    DiarizedSegment(
                        start=seg.start,
                        end=seg.end,
                        text=seg.text,
                        speaker=seg.speaker,
                        words=[
                            WordSegment(
                                start=w.start,
                                end=w.end,
                                word=w.word,
                                speaker=seg.speaker,
                            )
                            for w in seg.words
                        ],
                    )
                    for seg in result.segments
                ]

                return DiarizationResult(
                    segments=segments,
                    speakers=result.speakers,
                    language=result.language,
                    language_probability=result.language_probability,
                    duration=result.duration,
                )
            except Exception as e:
                raise DiarizationError(f"Diarization failed: {e}") from e

        # Default path: use WhisperX directly
        self._check_dependencies()
        self._check_token()

        try:
            import whisperx  # type: ignore[import-not-found]
            import whisperx.diarize  # type: ignore[import-not-found]

            # Load model (cached after first call)
            model = self._load_model()
            device = self._actual_device

            # Transcribe (suppress noisy library output)
            with _suppress_output():
                result = model.transcribe(audio, batch_size=16, language=language)
            detected_language = result.get("language", "unknown")

            # Align (get word-level timestamps)
            with _suppress_output():
                model_a, metadata = whisperx.load_align_model(
                    language_code=detected_language,
                    device=device,
                )
                result = whisperx.align(
                    result["segments"],
                    model_a,
                    metadata,
                    audio,
                    device,
                    return_char_alignments=False,
                )

            # Diarize (suppress noisy library output)
            with _suppress_output():
                diarize_model = whisperx.diarize.DiarizationPipeline(
                    use_auth_token=self._hf_token,
                    device=device,
                )

            # DiarizationPipeline returns None if the model couldn't be loaded
            # (usually because the user hasn't accepted the gated model license)
            if diarize_model is None:
                raise GatedModelError()

            diarize_kwargs = {}
            if self._num_speakers is not None:
                diarize_kwargs["min_speakers"] = self._num_speakers
                diarize_kwargs["max_speakers"] = self._num_speakers

            with _suppress_output():
                diarize_segments = diarize_model(
                    audio,
                    **diarize_kwargs,  # pyrefly: ignore[bad-argument-type]
                )

            # Assign speakers to words
            result = whisperx.assign_word_speakers(diarize_segments, result)

            # Convert to our data structures
            return self._convert_result(result, detected_language, language)

        except (DependencyMissingError, MissingTokenError, GatedModelError):
            raise
        except Exception as e:
            raise DiarizationError(f"Diarization failed: {e}") from e

    def _convert_result(
        self,
        whisperx_result: dict,
        detected_language: str,
        explicit_language: str | None,
    ) -> DiarizationResult:
        """Convert WhisperX output to DiarizationResult.

        Args:
            whisperx_result: Raw result from WhisperX.
            detected_language: Language detected by WhisperX.
            explicit_language: Language explicitly specified by user (or None).

        Returns:
            Standardized DiarizationResult.
        """
        segments: list[DiarizedSegment] = []
        speakers_seen: set[str] = set()

        for seg in whisperx_result.get("segments", []):
            speaker = seg.get("speaker", "UNKNOWN")

            # Convert SPEAKER_00 to SPEAKER_01 (1-indexed for remote speakers)
            if speaker.startswith("SPEAKER_"):
                try:
                    num = int(speaker.split("_")[1])
                    speaker = f"SPEAKER_{num + 1:02d}"
                except (IndexError, ValueError):
                    pass

            speakers_seen.add(speaker)

            # Extract word-level information if available
            words: list[WordSegment] = []
            for word_info in seg.get("words", []):
                word_speaker = word_info.get("speaker")
                # Apply same 1-indexing to word speakers
                if word_speaker and word_speaker.startswith("SPEAKER_"):
                    try:
                        num = int(word_speaker.split("_")[1])
                        word_speaker = f"SPEAKER_{num + 1:02d}"
                    except (IndexError, ValueError):
                        pass

                words.append(
                    WordSegment(
                        start=word_info.get("start", 0.0),
                        end=word_info.get("end", 0.0),
                        word=word_info.get("word", ""),
                        speaker=word_speaker,
                    )
                )

            segments.append(
                DiarizedSegment(
                    start=seg.get("start", 0.0),
                    end=seg.get("end", 0.0),
                    text=seg.get("text", "").strip(),
                    speaker=speaker,
                    words=words,
                )
            )

        duration = segments[-1].end if segments else 0.0

        # If language was explicitly specified, confidence is 100%
        language_probability = 1.0 if explicit_language else 0.0

        return DiarizationResult(
            segments=segments,
            speakers=sorted(speakers_seen),
            language=detected_language,
            language_probability=language_probability,
            duration=duration,
        )
