"""Tests for hark.transcriber module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from hark.constants import VALID_MODELS
from hark.exceptions import ModelDownloadError, ModelNotFoundError, TranscriptionError
from hark.transcriber import (
    Transcriber,
    TranscriptionResult,
    TranscriptionSegment,
    WordSegment,
)


class TestTranscriberInitialization:
    """Tests for Transcriber initialization."""

    def test_valid_model_accepted(self) -> None:
        """Should accept valid model names."""
        for model in VALID_MODELS:
            transcriber = Transcriber(model_name=model)
            assert transcriber._model_name == model

    def test_invalid_model_raises(self) -> None:
        """Should raise ValueError for invalid model name."""
        with pytest.raises(ValueError) as exc_info:
            Transcriber(model_name="invalid_model")
        assert "invalid model" in str(exc_info.value).lower()
        assert "invalid_model" in str(exc_info.value)

    def test_is_model_loaded_initially_false(self) -> None:
        """Model should not be loaded initially."""
        transcriber = Transcriber()
        assert transcriber.is_model_loaded() is False

    def test_default_device_is_auto(self) -> None:
        """Default device should be 'auto'."""
        transcriber = Transcriber()
        assert transcriber._requested_device == "auto"

    def test_custom_cache_dir(self, tmp_path: Path) -> None:
        """Should accept custom cache directory."""
        cache_dir = tmp_path / "models"
        transcriber = Transcriber(model_cache_dir=cache_dir)
        assert transcriber._cache_dir == cache_dir


class TestLoadModel:
    """Tests for load_model method."""

    def test_auto_device_calls_detect_best_device(self) -> None:
        """Should call detect_best_device when device is 'auto'."""
        mock_whisper = MagicMock()

        with (
            patch("hark.transcriber.detect_best_device", return_value="cpu") as mock_detect,
            patch.dict("sys.modules", {"faster_whisper": mock_whisper}),
        ):
            transcriber = Transcriber(device="auto")
            transcriber.load_model()
            mock_detect.assert_called_once()

    def test_vulkan_falls_back_to_cpu(self) -> None:
        """Vulkan device should fall back to CPU."""
        mock_whisper = MagicMock()

        with (
            patch("hark.transcriber.detect_best_device", return_value="vulkan"),
            patch.dict("sys.modules", {"faster_whisper": mock_whisper}),
        ):
            transcriber = Transcriber(device="auto")
            transcriber.load_model()
            assert transcriber._actual_device == "cpu"

    def test_creates_cache_directory(self, tmp_path: Path) -> None:
        """Should create cache directory if it doesn't exist."""
        mock_whisper = MagicMock()
        cache_dir = tmp_path / "new_cache" / "models"

        with (
            patch("hark.transcriber.detect_best_device", return_value="cpu"),
            patch.dict("sys.modules", {"faster_whisper": mock_whisper}),
        ):
            transcriber = Transcriber(model_cache_dir=cache_dir)
            transcriber.load_model()
            assert cache_dir.exists()

    def test_instantiates_whisper_model(self) -> None:
        """Should instantiate WhisperModel with correct parameters."""
        mock_whisper_module = MagicMock()
        mock_model_class = MagicMock()
        mock_whisper_module.WhisperModel = mock_model_class

        with (
            patch("hark.transcriber.detect_best_device", return_value="cpu"),
            patch.dict("sys.modules", {"faster_whisper": mock_whisper_module}),
        ):
            transcriber = Transcriber(model_name="small", device="cpu")
            transcriber.load_model()

            mock_model_class.assert_called_once()
            call_args = mock_model_class.call_args
            assert call_args[0][0] == "small"
            assert call_args[1]["device"] == "cpu"

    def test_faster_whisper_not_installed(self) -> None:
        """Should raise ModelNotFoundError if faster-whisper not installed."""
        # Need to patch detect_best_device to avoid subprocess calls during test
        # The import error should trigger when trying to import faster_whisper
        original_import = (
            __builtins__["__import__"]
            if isinstance(__builtins__, dict)
            else __builtins__.__import__
        )

        def selective_import(name, *args, **kwargs):
            if name == "faster_whisper":
                raise ImportError("No module named 'faster_whisper'")
            return original_import(name, *args, **kwargs)

        with (
            patch("hark.transcriber.detect_best_device", return_value="cpu"),
            patch.dict("sys.modules", {"faster_whisper": None}),
            patch("builtins.__import__", side_effect=selective_import),
        ):
            transcriber = Transcriber()
            with pytest.raises(ModelNotFoundError) as exc_info:
                transcriber.load_model()
            assert "faster-whisper" in str(exc_info.value)

    def test_download_error_raises_model_download_error(self) -> None:
        """Should raise ModelDownloadError on download failure."""
        mock_whisper_module = MagicMock()
        mock_whisper_module.WhisperModel.side_effect = Exception("Network download failed")

        with (
            patch("hark.transcriber.detect_best_device", return_value="cpu"),
            patch.dict("sys.modules", {"faster_whisper": mock_whisper_module}),
        ):
            transcriber = Transcriber()
            with pytest.raises(ModelDownloadError):
                transcriber.load_model()

    def test_generic_load_error_raises_model_not_found(self) -> None:
        """Should raise ModelNotFoundError on generic load failure."""
        mock_whisper_module = MagicMock()
        mock_whisper_module.WhisperModel.side_effect = Exception("Generic error")

        with (
            patch("hark.transcriber.detect_best_device", return_value="cpu"),
            patch.dict("sys.modules", {"faster_whisper": mock_whisper_module}),
        ):
            transcriber = Transcriber()
            with pytest.raises(ModelNotFoundError):
                transcriber.load_model()


class TestTranscribe:
    """Tests for transcribe method."""

    @pytest.fixture
    def mock_transcriber(self) -> Transcriber:
        """Create a transcriber with mocked model."""
        mock_whisper_module = MagicMock()

        with (
            patch("hark.transcriber.detect_best_device", return_value="cpu"),
            patch.dict("sys.modules", {"faster_whisper": mock_whisper_module}),
        ):
            transcriber = Transcriber()
            transcriber.load_model()

        return transcriber

    def test_lazy_loads_model(self, sample_audio: np.ndarray) -> None:
        """Should lazy-load model if not already loaded."""
        mock_whisper_module = MagicMock()
        mock_model = MagicMock()
        mock_whisper_module.WhisperModel.return_value = mock_model

        # Mock transcribe to return empty generator and info
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.95
        mock_model.transcribe.return_value = (iter([]), mock_info)

        with (
            patch("hark.transcriber.detect_best_device", return_value="cpu"),
            patch.dict("sys.modules", {"faster_whisper": mock_whisper_module}),
        ):
            transcriber = Transcriber()
            assert transcriber.is_model_loaded() is False
            transcriber.transcribe(sample_audio)
            assert transcriber.is_model_loaded() is True

    def test_resamples_non_16khz_audio(self, sample_audio: np.ndarray) -> None:
        """Should resample audio if sample rate is not 16kHz."""
        mock_whisper_module = MagicMock()
        mock_model = MagicMock()
        mock_whisper_module.WhisperModel.return_value = mock_model

        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.95
        mock_model.transcribe.return_value = (iter([]), mock_info)

        with (
            patch("hark.transcriber.detect_best_device", return_value="cpu"),
            patch.dict("sys.modules", {"faster_whisper": mock_whisper_module}),
            patch("librosa.resample") as mock_resample,
        ):
            mock_resample.return_value = sample_audio
            transcriber = Transcriber()
            transcriber.load_model()
            transcriber.transcribe(sample_audio, sample_rate=48000)

            mock_resample.assert_called_once()
            call_kwargs = mock_resample.call_args[1]
            assert call_kwargs["orig_sr"] == 48000
            assert call_kwargs["target_sr"] == 16000

    def test_ensures_float32(self, sample_audio: np.ndarray) -> None:
        """Should convert audio to float32."""
        mock_whisper_module = MagicMock()
        mock_model = MagicMock()
        mock_whisper_module.WhisperModel.return_value = mock_model

        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.95
        mock_model.transcribe.return_value = (iter([]), mock_info)

        int16_audio = (sample_audio * 32767).astype(np.int16)

        with (
            patch("hark.transcriber.detect_best_device", return_value="cpu"),
            patch.dict("sys.modules", {"faster_whisper": mock_whisper_module}),
        ):
            transcriber = Transcriber()
            transcriber.load_model()
            transcriber.transcribe(int16_audio)

            # Check that model received float32
            call_args = mock_model.transcribe.call_args
            received_audio = call_args[0][0]
            assert received_audio.dtype == np.float32

    def test_calls_model_transcribe(
        self, sample_audio: np.ndarray, mock_whisper_model: MagicMock
    ) -> None:
        """Should call model.transcribe with correct parameters."""
        mock_whisper_module = MagicMock()
        mock_whisper_module.WhisperModel.return_value = mock_whisper_model

        with (
            patch("hark.transcriber.detect_best_device", return_value="cpu"),
            patch.dict("sys.modules", {"faster_whisper": mock_whisper_module}),
        ):
            transcriber = Transcriber()
            transcriber.load_model()
            transcriber.transcribe(sample_audio, language="en", word_timestamps=True)

            mock_whisper_model.transcribe.assert_called_once()
            call_kwargs = mock_whisper_model.transcribe.call_args[1]
            assert call_kwargs["language"] == "en"
            assert call_kwargs["word_timestamps"] is True

    def test_collects_segments(self, sample_audio: np.ndarray) -> None:
        """Should collect segments from generator."""
        mock_whisper_module = MagicMock()
        mock_model = MagicMock()
        mock_whisper_module.WhisperModel.return_value = mock_model

        # Create mock segments
        mock_segment1 = MagicMock()
        mock_segment1.start = 0.0
        mock_segment1.end = 1.5
        mock_segment1.text = " Hello"
        mock_segment1.words = None

        mock_segment2 = MagicMock()
        mock_segment2.start = 1.6
        mock_segment2.end = 3.0
        mock_segment2.text = " World"
        mock_segment2.words = None

        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.95
        mock_model.transcribe.return_value = (iter([mock_segment1, mock_segment2]), mock_info)

        with (
            patch("hark.transcriber.detect_best_device", return_value="cpu"),
            patch.dict("sys.modules", {"faster_whisper": mock_whisper_module}),
        ):
            transcriber = Transcriber()
            transcriber.load_model()
            result = transcriber.transcribe(sample_audio)

            assert len(result.segments) == 2
            assert result.segments[0].text == "Hello"
            assert result.segments[1].text == "World"

    def test_includes_word_timestamps(self, sample_audio: np.ndarray) -> None:
        """Should include word timestamps when available."""
        mock_whisper_module = MagicMock()
        mock_model = MagicMock()
        mock_whisper_module.WhisperModel.return_value = mock_model

        # Create mock segment with words
        mock_word1 = MagicMock()
        mock_word1.start = 0.0
        mock_word1.end = 0.5
        mock_word1.word = "Hello"

        mock_word2 = MagicMock()
        mock_word2.start = 0.6
        mock_word2.end = 1.0
        mock_word2.word = "World"

        mock_segment = MagicMock()
        mock_segment.start = 0.0
        mock_segment.end = 1.0
        mock_segment.text = " Hello World"
        mock_segment.words = [mock_word1, mock_word2]

        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.95
        mock_model.transcribe.return_value = (iter([mock_segment]), mock_info)

        with (
            patch("hark.transcriber.detect_best_device", return_value="cpu"),
            patch.dict("sys.modules", {"faster_whisper": mock_whisper_module}),
        ):
            transcriber = Transcriber()
            transcriber.load_model()
            result = transcriber.transcribe(sample_audio, word_timestamps=True)

            assert len(result.segments[0].words) == 2
            assert result.segments[0].words[0].word == "Hello"
            assert result.segments[0].words[1].word == "World"

    def test_progress_callback_invoked(self, sample_audio: np.ndarray) -> None:
        """Should invoke progress callback during transcription."""
        mock_whisper_module = MagicMock()
        mock_model = MagicMock()
        mock_whisper_module.WhisperModel.return_value = mock_model

        mock_segment = MagicMock()
        mock_segment.start = 0.0
        mock_segment.end = 0.5
        mock_segment.text = " Test"
        mock_segment.words = None

        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.95
        mock_model.transcribe.return_value = (iter([mock_segment]), mock_info)

        callback = MagicMock()

        with (
            patch("hark.transcriber.detect_best_device", return_value="cpu"),
            patch.dict("sys.modules", {"faster_whisper": mock_whisper_module}),
        ):
            transcriber = Transcriber()
            transcriber.load_model()
            transcriber.transcribe(sample_audio, progress_callback=callback)

            # Should be called at least twice (during segment and final 1.0)
            assert callback.call_count >= 2

    def test_returns_transcription_result(self, sample_audio: np.ndarray) -> None:
        """Should return complete TranscriptionResult."""
        mock_whisper_module = MagicMock()
        mock_model = MagicMock()
        mock_whisper_module.WhisperModel.return_value = mock_model

        mock_segment = MagicMock()
        mock_segment.start = 0.0
        mock_segment.end = 1.0
        mock_segment.text = " Hello"
        mock_segment.words = None

        mock_info = MagicMock()
        mock_info.language = "fr"
        mock_info.language_probability = 0.87
        mock_model.transcribe.return_value = (iter([mock_segment]), mock_info)

        with (
            patch("hark.transcriber.detect_best_device", return_value="cpu"),
            patch.dict("sys.modules", {"faster_whisper": mock_whisper_module}),
        ):
            transcriber = Transcriber()
            transcriber.load_model()
            result = transcriber.transcribe(sample_audio)

            assert isinstance(result, TranscriptionResult)
            assert result.text == "Hello"
            assert result.language == "fr"
            assert result.language_probability == 0.87
            assert result.duration > 0

    def test_transcription_error_raised(self, sample_audio: np.ndarray) -> None:
        """Should raise TranscriptionError on failure."""
        mock_whisper_module = MagicMock()
        mock_model = MagicMock()
        mock_whisper_module.WhisperModel.return_value = mock_model
        mock_model.transcribe.side_effect = RuntimeError("Transcription failed")

        with (
            patch("hark.transcriber.detect_best_device", return_value="cpu"),
            patch.dict("sys.modules", {"faster_whisper": mock_whisper_module}),
        ):
            transcriber = Transcriber()
            transcriber.load_model()

            with pytest.raises(TranscriptionError) as exc_info:
                transcriber.transcribe(sample_audio)
                assert "transcription failed" in str(exc_info.value).lower()

    def test_language_probability_100_when_language_specified(
        self, sample_audio: np.ndarray
    ) -> None:
        """Should set language_probability to 1.0 when language is explicitly specified."""
        mock_whisper_module = MagicMock()
        mock_model = MagicMock()
        mock_whisper_module.WhisperModel.return_value = mock_model

        mock_segment = MagicMock()
        mock_segment.start = 0.0
        mock_segment.end = 1.0
        mock_segment.text = "Hallo"
        mock_segment.words = []

        mock_info = MagicMock()
        mock_info.language = "de"
        mock_info.language_probability = 0.65  # Model's detected probability
        mock_model.transcribe.return_value = (iter([mock_segment]), mock_info)

        with (
            patch("hark.transcriber.detect_best_device", return_value="cpu"),
            patch.dict("sys.modules", {"faster_whisper": mock_whisper_module}),
        ):
            transcriber = Transcriber()
            transcriber.load_model()
            # Explicitly specify language
            result = transcriber.transcribe(sample_audio, language="de")

            # Should be 100% confidence since language was explicitly specified
            assert result.language_probability == 1.0
            assert result.language == "de"

    def test_language_probability_from_model_when_auto_detect(
        self, sample_audio: np.ndarray
    ) -> None:
        """Should use model's language_probability when language is auto-detected."""
        mock_whisper_module = MagicMock()
        mock_model = MagicMock()
        mock_whisper_module.WhisperModel.return_value = mock_model

        mock_segment = MagicMock()
        mock_segment.start = 0.0
        mock_segment.end = 1.0
        mock_segment.text = "Hello"
        mock_segment.words = []

        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.73
        mock_model.transcribe.return_value = (iter([mock_segment]), mock_info)

        with (
            patch("hark.transcriber.detect_best_device", return_value="cpu"),
            patch.dict("sys.modules", {"faster_whisper": mock_whisper_module}),
        ):
            transcriber = Transcriber()
            transcriber.load_model()
            # No language specified - auto-detect
            result = transcriber.transcribe(sample_audio, language=None)

            # Should use model's detected probability
            assert result.language_probability == 0.73
            assert result.language == "en"


class TestTranscriberMethods:
    """Tests for other Transcriber methods."""

    def test_is_model_loaded_true_after_load(self) -> None:
        """is_model_loaded should return True after loading."""
        mock_whisper_module = MagicMock()

        with (
            patch("hark.transcriber.detect_best_device", return_value="cpu"),
            patch.dict("sys.modules", {"faster_whisper": mock_whisper_module}),
        ):
            transcriber = Transcriber()
            transcriber.load_model()
            assert transcriber.is_model_loaded() is True

    def test_device_property_returns_actual_device(self) -> None:
        """device property should return actual device after loading."""
        mock_whisper_module = MagicMock()

        with (
            patch("hark.transcriber.detect_best_device", return_value="cuda"),
            patch.dict("sys.modules", {"faster_whisper": mock_whisper_module}),
        ):
            transcriber = Transcriber(device="auto")
            transcriber.load_model()
            assert transcriber.device == "cuda"

    def test_device_property_none_before_load(self) -> None:
        """device property should return None before loading."""
        transcriber = Transcriber()
        assert transcriber.device is None

    def test_list_models_returns_valid_models(self) -> None:
        """list_models should return VALID_MODELS."""
        models = Transcriber.list_models()
        assert models == list(VALID_MODELS)


class TestDataclasses:
    """Tests for transcriber dataclasses."""

    def test_word_segment_creation(self) -> None:
        """Should create WordSegment with all fields."""
        word = WordSegment(start=0.0, end=0.5, word="Hello")
        assert word.start == 0.0
        assert word.end == 0.5
        assert word.word == "Hello"

    def test_transcription_segment_creation(self) -> None:
        """Should create TranscriptionSegment with all fields."""
        words = [WordSegment(start=0.0, end=0.5, word="Hello")]
        segment = TranscriptionSegment(start=0.0, end=1.0, text="Hello world", words=words)
        assert segment.start == 0.0
        assert segment.end == 1.0
        assert segment.text == "Hello world"
        assert len(segment.words) == 1

    def test_transcription_segment_default_words(self) -> None:
        """TranscriptionSegment should default to empty words list."""
        segment = TranscriptionSegment(start=0.0, end=1.0, text="Test")
        assert segment.words == []

    def test_transcription_result_creation(self) -> None:
        """Should create TranscriptionResult with all fields."""
        segments = [TranscriptionSegment(start=0.0, end=1.0, text="Hello")]
        result = TranscriptionResult(
            text="Hello",
            segments=segments,
            language="en",
            language_probability=0.95,
            duration=1.0,
        )
        assert result.text == "Hello"
        assert len(result.segments) == 1
        assert result.language == "en"
        assert result.language_probability == 0.95
        assert result.duration == 1.0
