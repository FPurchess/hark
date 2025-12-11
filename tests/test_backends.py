"""Tests for backend wrappers and dependency injection."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from hark.backends.base import (
    DiarizationOutput,
    DiarizedSegment,
    TranscriptionOutput,
    TranscriptionSegment,
    WordInfo,
)
from hark.backends.whisper import FasterWhisperBackend
from hark.backends.whisperx import WhisperXBackend
from hark.diarizer import DiarizationResult, Diarizer
from hark.transcriber import Transcriber, TranscriptionResult


class TestFasterWhisperBackend:
    """Tests for FasterWhisperBackend wrapper."""

    def test_init_not_loaded(self) -> None:
        """Backend should not be loaded initially."""
        backend = FasterWhisperBackend()
        assert backend.is_loaded() is False

    def test_load_model_creates_whisper_model(self) -> None:
        """load_model should create WhisperModel with correct parameters."""
        mock_whisper_module = MagicMock()
        mock_model_class = MagicMock()
        mock_whisper_module.WhisperModel = mock_model_class

        with patch.dict("sys.modules", {"faster_whisper": mock_whisper_module}):
            backend = FasterWhisperBackend()
            backend.load_model(
                model_name="base",
                device="cpu",
                compute_type="int8",
                download_root="/cache",
            )

            mock_model_class.assert_called_once_with(
                "base",
                device="cpu",
                compute_type="int8",
                download_root="/cache",
            )
            assert backend.is_loaded() is True

    def test_transcribe_raises_if_not_loaded(self) -> None:
        """transcribe should raise RuntimeError if model not loaded."""
        backend = FasterWhisperBackend()
        audio = np.zeros(16000, dtype=np.float32)

        with pytest.raises(RuntimeError) as exc_info:
            backend.transcribe(audio)
        assert "not loaded" in str(exc_info.value).lower()

    def test_transcribe_returns_transcription_output(self) -> None:
        """transcribe should return TranscriptionOutput with segments."""
        mock_whisper_module = MagicMock()
        mock_model = MagicMock()
        mock_whisper_module.WhisperModel.return_value = mock_model

        # Mock segment
        mock_segment = MagicMock()
        mock_segment.start = 0.0
        mock_segment.end = 1.5
        mock_segment.text = " Hello world"
        mock_segment.words = None

        # Mock info
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.95

        mock_model.transcribe.return_value = (iter([mock_segment]), mock_info)

        with patch.dict("sys.modules", {"faster_whisper": mock_whisper_module}):
            backend = FasterWhisperBackend()
            backend.load_model("base", "cpu", "int8", "/cache")

            audio = np.zeros(16000, dtype=np.float32)
            result = backend.transcribe(audio, language="en")

            assert isinstance(result, TranscriptionOutput)
            assert len(result.segments) == 1
            assert result.segments[0].text == "Hello world"
            assert result.language == "en"
            assert result.language_probability == 1.0  # explicit language

    def test_transcribe_includes_word_timestamps(self) -> None:
        """transcribe should include word timestamps when available."""
        mock_whisper_module = MagicMock()
        mock_model = MagicMock()
        mock_whisper_module.WhisperModel.return_value = mock_model

        # Mock words
        mock_word1 = MagicMock()
        mock_word1.start = 0.0
        mock_word1.end = 0.5
        mock_word1.word = "Hello"
        mock_word1.probability = 0.98

        mock_word2 = MagicMock()
        mock_word2.start = 0.6
        mock_word2.end = 1.0
        mock_word2.word = "world"
        mock_word2.probability = 0.97

        mock_segment = MagicMock()
        mock_segment.start = 0.0
        mock_segment.end = 1.0
        mock_segment.text = " Hello world"
        mock_segment.words = [mock_word1, mock_word2]

        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.95

        mock_model.transcribe.return_value = (iter([mock_segment]), mock_info)

        with patch.dict("sys.modules", {"faster_whisper": mock_whisper_module}):
            backend = FasterWhisperBackend()
            backend.load_model("base", "cpu", "int8", "/cache")

            audio = np.zeros(16000, dtype=np.float32)
            result = backend.transcribe(audio, word_timestamps=True)

            assert len(result.segments[0].words) == 2
            assert result.segments[0].words[0].word == "Hello"
            assert result.segments[0].words[1].word == "world"

    def test_transcribe_converts_to_float32(self) -> None:
        """transcribe should convert audio to float32."""
        mock_whisper_module = MagicMock()
        mock_model = MagicMock()
        mock_whisper_module.WhisperModel.return_value = mock_model

        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.95
        mock_model.transcribe.return_value = (iter([]), mock_info)

        with patch.dict("sys.modules", {"faster_whisper": mock_whisper_module}):
            backend = FasterWhisperBackend()
            backend.load_model("base", "cpu", "int8", "/cache")

            # Provide int16 audio
            int16_audio = np.zeros(16000, dtype=np.int16)
            backend.transcribe(int16_audio)

            # Check model received float32
            call_args = mock_model.transcribe.call_args
            received_audio = call_args[0][0]
            assert received_audio.dtype == np.float32


class TestWhisperXBackend:
    """Tests for WhisperXBackend wrapper."""

    def test_init_not_loaded(self) -> None:
        """Backend should not be loaded initially."""
        backend = WhisperXBackend()
        assert backend.is_loaded() is False

    def test_load_model_creates_whisperx_model(self) -> None:
        """load_model should create whisperx model with correct parameters."""
        mock_whisperx = MagicMock()

        with patch.dict("sys.modules", {"whisperx": mock_whisperx}):
            backend = WhisperXBackend()
            backend.load_model(
                model_name="base",
                device="cpu",
                compute_type="int8",
                download_root="/cache",
                hf_token="test_token",
            )

            mock_whisperx.load_model.assert_called_once()
            assert backend.is_loaded() is True

    def test_transcribe_and_diarize_raises_if_not_loaded(self) -> None:
        """transcribe_and_diarize should raise RuntimeError if model not loaded."""
        backend = WhisperXBackend()
        audio = np.zeros(16000, dtype=np.float32)

        with pytest.raises(RuntimeError) as exc_info:
            backend.transcribe_and_diarize(audio)
        assert "not loaded" in str(exc_info.value).lower()

    def test_transcribe_and_diarize_returns_diarization_output(self) -> None:
        """transcribe_and_diarize should return DiarizationOutput."""
        mock_whisperx = MagicMock()
        mock_model = MagicMock()
        mock_whisperx.load_model.return_value = mock_model

        # Mock transcription result
        mock_model.transcribe.return_value = {
            "language": "en",
            "segments": [{"start": 0.0, "end": 1.0, "text": "Hello"}],
        }

        # Mock align result
        mock_whisperx.load_align_model.return_value = (MagicMock(), MagicMock())
        mock_whisperx.align.return_value = {
            "segments": [{"start": 0.0, "end": 1.0, "text": "Hello"}],
        }

        # Mock diarization
        mock_diarize_pipeline = MagicMock()
        mock_diarize = MagicMock()
        mock_diarize.DiarizationPipeline.return_value = mock_diarize_pipeline
        mock_whisperx.diarize = mock_diarize

        # Mock assign_word_speakers result
        mock_whisperx.assign_word_speakers.return_value = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": " Hello",
                    "speaker": "SPEAKER_00",
                    "words": [],
                }
            ]
        }

        with patch.dict("sys.modules", {"whisperx": mock_whisperx}):
            backend = WhisperXBackend()
            backend.load_model("base", "cpu", "int8", "/cache", "test_token")

            audio = np.zeros(16000, dtype=np.float32)
            result = backend.transcribe_and_diarize(audio, language="en")

            assert isinstance(result, DiarizationOutput)
            assert len(result.segments) == 1
            # SPEAKER_00 should become SPEAKER_01 (1-indexed)
            assert result.segments[0].speaker == "SPEAKER_01"
            assert "SPEAKER_01" in result.speakers


class TestTranscriberWithBackendDI:
    """Tests for Transcriber with dependency injection."""

    def test_uses_injected_backend(self, sample_audio: np.ndarray) -> None:
        """Transcriber should delegate to injected backend."""
        mock_backend = MagicMock()
        mock_backend.is_loaded.return_value = True
        mock_backend.transcribe.return_value = TranscriptionOutput(
            segments=[
                TranscriptionSegment(
                    start=0.0,
                    end=1.0,
                    text="Hello",
                    words=[],
                )
            ],
            language="en",
            language_probability=0.95,
            duration=1.0,
        )

        transcriber = Transcriber(backend=mock_backend)
        result = transcriber.transcribe(sample_audio)

        mock_backend.transcribe.assert_called_once()
        assert isinstance(result, TranscriptionResult)
        assert result.text == "Hello"

    def test_is_model_loaded_delegates_to_backend(self) -> None:
        """is_model_loaded should delegate to backend."""
        mock_backend = MagicMock()
        mock_backend.is_loaded.return_value = True

        transcriber = Transcriber(backend=mock_backend)
        assert transcriber.is_model_loaded() is True

        mock_backend.is_loaded.return_value = False
        assert transcriber.is_model_loaded() is False

    def test_load_model_delegates_to_backend(self) -> None:
        """load_model should delegate to backend."""
        mock_backend = MagicMock()

        with patch("hark.transcriber.detect_best_device", return_value="cpu"):
            transcriber = Transcriber(backend=mock_backend)
            transcriber.load_model()

            mock_backend.load_model.assert_called_once()
            call_kwargs = mock_backend.load_model.call_args[1]
            assert call_kwargs["model_name"] == "base"
            assert call_kwargs["device"] == "cpu"

    def test_backend_error_raises_transcription_error(self, sample_audio: np.ndarray) -> None:
        """Backend errors should be wrapped in TranscriptionError."""
        from hark.exceptions import TranscriptionError

        mock_backend = MagicMock()
        mock_backend.is_loaded.return_value = True
        mock_backend.transcribe.side_effect = RuntimeError("Backend failed")

        transcriber = Transcriber(backend=mock_backend)

        with pytest.raises(TranscriptionError) as exc_info:
            transcriber.transcribe(sample_audio)
        assert "failed" in str(exc_info.value).lower()

    def test_without_backend_uses_faster_whisper(self) -> None:
        """Without backend, should use faster-whisper directly."""
        mock_whisper_module = MagicMock()

        with (
            patch("hark.transcriber.detect_best_device", return_value="cpu"),
            patch.dict("sys.modules", {"faster_whisper": mock_whisper_module}),
        ):
            transcriber = Transcriber()  # No backend
            transcriber.load_model()

            # Should have created WhisperModel
            mock_whisper_module.WhisperModel.assert_called_once()


class TestDiarizerWithBackendDI:
    """Tests for Diarizer with dependency injection."""

    def test_uses_injected_backend(self, sample_audio: np.ndarray) -> None:
        """Diarizer should delegate to injected backend."""
        mock_backend = MagicMock()
        mock_backend.is_loaded.return_value = True
        mock_backend.transcribe_and_diarize.return_value = DiarizationOutput(
            segments=[
                DiarizedSegment(
                    start=0.0,
                    end=1.0,
                    text="Hello",
                    speaker="SPEAKER_01",
                    words=[],
                )
            ],
            speakers=["SPEAKER_01"],
            language="en",
            language_probability=0.95,
            duration=1.0,
        )

        diarizer = Diarizer(backend=mock_backend)
        result = diarizer.transcribe_and_diarize(sample_audio)

        mock_backend.transcribe_and_diarize.assert_called_once()
        assert isinstance(result, DiarizationResult)
        assert result.speakers == ["SPEAKER_01"]

    def test_backend_load_model_called_if_not_loaded(self, sample_audio: np.ndarray) -> None:
        """Backend.load_model should be called if not loaded."""
        mock_backend = MagicMock()
        mock_backend.is_loaded.return_value = False
        mock_backend.transcribe_and_diarize.return_value = DiarizationOutput(
            segments=[],
            speakers=[],
            language="en",
            language_probability=0.95,
            duration=0.0,
        )

        with patch("hark.diarizer.detect_best_device", return_value="cpu"):
            diarizer = Diarizer(backend=mock_backend, hf_token="test")
            diarizer.transcribe_and_diarize(sample_audio)

            mock_backend.load_model.assert_called_once()
            call_kwargs = mock_backend.load_model.call_args[1]
            assert call_kwargs["model_name"] == "base"

    def test_backend_error_raises_diarization_error(self, sample_audio: np.ndarray) -> None:
        """Backend errors should be wrapped in DiarizationError."""
        from hark.exceptions import DiarizationError

        mock_backend = MagicMock()
        mock_backend.is_loaded.return_value = True
        mock_backend.transcribe_and_diarize.side_effect = RuntimeError("Backend failed")

        diarizer = Diarizer(backend=mock_backend)

        with pytest.raises(DiarizationError) as exc_info:
            diarizer.transcribe_and_diarize(sample_audio)
        assert "failed" in str(exc_info.value).lower()

    def test_without_backend_checks_dependencies(self) -> None:
        """Without backend, should check for whisperx dependency."""
        from hark.exceptions import DependencyMissingError

        # Mock whisperx as not installed
        with patch.dict("sys.modules", {"whisperx": None}):
            diarizer = Diarizer(hf_token="test")  # No backend

            with pytest.raises(DependencyMissingError):
                diarizer.transcribe_and_diarize(np.zeros(16000, dtype=np.float32))


class TestBackendDataClasses:
    """Tests for backend data classes."""

    def test_word_info_creation(self) -> None:
        """WordInfo should be created with all fields."""
        word = WordInfo(start=0.0, end=0.5, word="Hello", probability=0.98)
        assert word.start == 0.0
        assert word.end == 0.5
        assert word.word == "Hello"
        assert word.probability == 0.98

    def test_word_info_default_probability(self) -> None:
        """WordInfo should default probability to 1.0."""
        word = WordInfo(start=0.0, end=0.5, word="Hello")
        assert word.probability == 1.0

    def test_transcription_segment_creation(self) -> None:
        """TranscriptionSegment should be created with all fields."""
        words = [WordInfo(start=0.0, end=0.5, word="Hello")]
        segment = TranscriptionSegment(start=0.0, end=1.0, text="Hello world", words=words)
        assert segment.start == 0.0
        assert segment.end == 1.0
        assert segment.text == "Hello world"
        assert len(segment.words) == 1

    def test_transcription_output_text_property(self) -> None:
        """TranscriptionOutput.text should concatenate segment texts."""
        segments = [
            TranscriptionSegment(start=0.0, end=1.0, text="Hello"),
            TranscriptionSegment(start=1.0, end=2.0, text="world"),
        ]
        output = TranscriptionOutput(
            segments=segments,
            language="en",
            language_probability=0.95,
            duration=2.0,
        )
        assert output.text == "Hello world"

    def test_diarized_segment_creation(self) -> None:
        """DiarizedSegment should be created with speaker field."""
        segment = DiarizedSegment(start=0.0, end=1.0, text="Hello", speaker="SPEAKER_01", words=[])
        assert segment.speaker == "SPEAKER_01"

    def test_diarization_output_creation(self) -> None:
        """DiarizationOutput should be created with speakers list."""
        segments = [
            DiarizedSegment(start=0.0, end=1.0, text="Hello", speaker="SPEAKER_01", words=[]),
        ]
        output = DiarizationOutput(
            segments=segments,
            speakers=["SPEAKER_01"],
            language="en",
            language_probability=0.95,
            duration=1.0,
        )
        assert output.speakers == ["SPEAKER_01"]
        assert len(output.segments) == 1
