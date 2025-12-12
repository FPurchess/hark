"""WhisperX backend implementation for diarization."""

import numpy as np

from hark.backends.base import (
    DiarizationOutput,
    DiarizedSegment,
    WordInfo,
)
from hark.constants import DEFAULT_SAMPLE_RATE, UNKNOWN_LANGUAGE_PROBABILITY
from hark.utils import renumber_speaker, suppress_output


class WhisperXBackend:
    """Wrapper around whisperx for diarization.

    This backend provides a clean interface to whisperx, isolating the
    external dependency from business logic. It handles:
    - Model loading with proper parameters
    - Transcription, alignment, and diarization pipeline
    - Suppression of noisy library output
    - Conversion to standardized output format

    Example:
        backend = WhisperXBackend()
        backend.load_model("base", "cuda", "float16", "/cache", "hf_token")
        result = backend.transcribe_and_diarize(audio_array, language="en")
    """

    def __init__(self) -> None:
        """Initialize the backend."""
        self._model = None
        self._device: str | None = None
        self._hf_token: str | None = None

    def load_model(
        self,
        model_name: str,
        device: str,
        compute_type: str,
        download_root: str,
        hf_token: str,
    ) -> None:
        """Load WhisperX model.

        Args:
            model_name: Model size (e.g., "base", "small", "medium").
            device: Device to use ("cpu", "cuda").
            compute_type: Compute type.
            download_root: Directory for model cache.
            hf_token: HuggingFace token for pyannote models.
        """
        import whisperx

        self._device = device
        self._hf_token = hf_token

        with suppress_output():
            self._model = whisperx.load_model(
                model_name,
                device=device,
                compute_type=compute_type,
                download_root=download_root,
            )

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
            sample_rate: Sample rate of audio (default 16000).
            language: Language code or None for auto-detection.
            num_speakers: Expected number of speakers, or None for auto.

        Returns:
            DiarizationOutput with speaker-labeled segments.

        Raises:
            RuntimeError: If model is not loaded.
        """
        import whisperx
        import whisperx.diarize  # type: ignore[import-not-found]

        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Transcribe
        with suppress_output():
            result = self._model.transcribe(audio, batch_size=16, language=language)
        detected_language = result.get("language", "unknown")

        # Align (get word-level timestamps)
        with suppress_output():
            model_a, metadata = whisperx.load_align_model(
                language_code=detected_language,
                device=self._device,
            )
            result = whisperx.align(
                result["segments"],
                model_a,
                metadata,
                audio,
                self._device,
                return_char_alignments=False,
            )

        # Diarize - CRITICAL: use whisperx.diarize.DiarizationPipeline
        with suppress_output():
            diarize_model = whisperx.diarize.DiarizationPipeline(
                use_auth_token=self._hf_token,
                device=self._device,
            )

        if diarize_model is None:
            raise RuntimeError(
                "Failed to load diarization model. Check your HuggingFace token and model access."
            )

        diarize_kwargs = {}
        if num_speakers is not None:
            diarize_kwargs["min_speakers"] = num_speakers
            diarize_kwargs["max_speakers"] = num_speakers

        with suppress_output():
            diarize_segments = diarize_model(
                audio,
                **diarize_kwargs,  # pyrefly: ignore[bad-argument-type]
            )

        # Assign speakers to words
        result = whisperx.assign_word_speakers(diarize_segments, result)

        # Convert to output format
        return self._convert_result(result, detected_language, language)

    def _convert_result(
        self,
        whisperx_result: dict,
        detected_language: str,
        explicit_language: str | None,
    ) -> DiarizationOutput:
        """Convert whisperx output to DiarizationOutput.

        Args:
            whisperx_result: Raw result from whisperx.
            detected_language: Language detected by whisperx.
            explicit_language: Language explicitly specified by user (or None).

        Returns:
            Standardized DiarizationOutput.
        """
        segments = []
        speakers_seen: set[str] = set()

        for seg in whisperx_result.get("segments", []):
            speaker = seg.get("speaker", "UNKNOWN")

            # 1-index speakers (SPEAKER_00 -> SPEAKER_01)
            speaker = renumber_speaker(speaker)

            speakers_seen.add(speaker)

            # Extract words if available
            words = []
            if "words" in seg:
                for w in seg["words"]:
                    words.append(
                        WordInfo(
                            start=w.get("start", 0.0),
                            end=w.get("end", 0.0),
                            word=w.get("word", ""),
                            probability=w.get("score", 1.0),
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
        # Otherwise, mark as unknown (WhisperX doesn't expose language probability)
        language_probability = 1.0 if explicit_language else UNKNOWN_LANGUAGE_PROBABILITY

        return DiarizationOutput(
            segments=segments,
            speakers=sorted(speakers_seen),
            language=detected_language,
            language_probability=language_probability,
            duration=duration,
        )

    def is_loaded(self) -> bool:
        """Check if models are loaded.

        Returns:
            True if model is loaded and ready for diarization.
        """
        return self._model is not None
