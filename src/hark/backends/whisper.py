"""Faster-whisper backend implementation."""

import numpy as np

from hark.backends.base import (
    TranscriptionOutput,
    TranscriptionSegment,
    WordInfo,
)
from hark.constants import DEFAULT_BEAM_SIZE, DEFAULT_VAD_FILTER, DEFAULT_VAD_MIN_SILENCE_MS


class FasterWhisperBackend:
    """Wrapper around faster-whisper for transcription.

    This backend provides a clean interface to faster-whisper, isolating
    the external dependency from business logic. It handles:
    - Model loading with proper parameters
    - Audio transcription with word timestamps
    - Conversion to standardized output format

    Example:
        backend = FasterWhisperBackend()
        backend.load_model("base", "cpu", "int8", "/cache")
        result = backend.transcribe(audio_array, language="en")
    """

    def __init__(self) -> None:
        """Initialize the backend."""
        self._model = None

    def load_model(
        self,
        model_name: str,
        device: str,
        compute_type: str,
        download_root: str,
    ) -> None:
        """Load the WhisperModel.

        Args:
            model_name: Model size (e.g., "base", "small", "medium", "large-v3").
            device: Device to use ("cpu", "cuda", "auto").
            compute_type: Compute type (e.g., "int8", "float16", "float32").
            download_root: Directory for model cache.
        """
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            download_root=download_root,
        )

    def transcribe(
        self,
        audio: np.ndarray,
        language: str | None = None,
        word_timestamps: bool = False,
        beam_size: int = DEFAULT_BEAM_SIZE,
        vad_filter: bool = DEFAULT_VAD_FILTER,
        vad_min_silence_ms: int = DEFAULT_VAD_MIN_SILENCE_MS,
    ) -> TranscriptionOutput:
        """Transcribe audio using faster-whisper.

        Args:
            audio: Audio data as float32 numpy array (mono, 16kHz).
            language: Language code or None for auto-detection.
            word_timestamps: Include word-level timestamps.
            beam_size: Beam size for decoding (default: 5).
            vad_filter: Enable VAD filtering (default: True).
            vad_min_silence_ms: Minimum silence duration in ms for VAD (default: 500).

        Returns:
            TranscriptionOutput with segments and language info.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Transcribe
        segments_gen, info = self._model.transcribe(
            audio,
            language=language,
            word_timestamps=word_timestamps,
            beam_size=beam_size,
            vad_filter=vad_filter,
            vad_parameters={"min_silence_duration_ms": vad_min_silence_ms},
        )

        # Collect segments
        segments = []
        for seg in segments_gen:
            words = []
            if seg.words:
                words = [
                    WordInfo(
                        start=w.start,
                        end=w.end,
                        word=w.word,
                        probability=getattr(w, "probability", 1.0),
                    )
                    for w in seg.words
                ]

            segments.append(
                TranscriptionSegment(
                    start=seg.start,
                    end=seg.end,
                    text=seg.text.strip(),
                    words=words,
                )
            )

        # Calculate duration
        duration = len(audio) / 16000 if len(audio) > 0 else 0.0

        # If language was explicitly specified, confidence is 100%
        language_probability = 1.0 if language else info.language_probability

        return TranscriptionOutput(
            segments=segments,
            language=info.language,
            language_probability=language_probability,
            duration=duration,
        )

    def is_loaded(self) -> bool:
        """Check if model is loaded.

        Returns:
            True if model is loaded and ready for transcription.
        """
        return self._model is not None
