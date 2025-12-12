"""Tests for FasterWhisperBackend wrapper."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from hark.backends.base import TranscriptionOutput
from hark.backends.whisper import FasterWhisperBackend


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
