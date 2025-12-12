"""Tests for WhisperXBackend wrapper."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from hark.backends.base import DiarizationOutput
from hark.backends.whisperx import WhisperXBackend


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
