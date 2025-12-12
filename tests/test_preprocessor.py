"""Tests for hark.preprocessor module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from hark.config import (
    NoiseReductionConfig,
    NormalizationConfig,
    PreprocessingConfig,
    SilenceTrimmingConfig,
)
from hark.exceptions import PreprocessingError
from hark.preprocessor import (
    AudioPreprocessor,
    PreprocessingResult,
    normalize_audio,
    reduce_noise,
    trim_silence,
)


class TestReduceNoise:
    """Tests for reduce_noise function."""

    def test_applies_noisereduce(self, sample_audio: np.ndarray) -> None:
        """Should call noisereduce.reduce_noise with correct parameters."""
        mock_nr = MagicMock()
        mock_nr.reduce_noise.return_value = sample_audio

        with patch.dict("sys.modules", {"noisereduce": mock_nr}):
            # Need to reimport to pick up mock
            from hark import preprocessor

            preprocessor.reduce_noise(sample_audio, 16000, strength=0.7)

            mock_nr.reduce_noise.assert_called_once()
            call_kwargs = mock_nr.reduce_noise.call_args[1]
            assert call_kwargs["sr"] == 16000
            assert call_kwargs["prop_decrease"] == 0.7
            assert call_kwargs["stationary"] is True

    def test_strength_mapping_zero(self, sample_audio: np.ndarray) -> None:
        """Strength 0.0 should result in prop_decrease=0.0."""
        mock_nr = MagicMock()
        mock_nr.reduce_noise.return_value = sample_audio

        with patch.dict("sys.modules", {"noisereduce": mock_nr}):
            from hark import preprocessor

            preprocessor.reduce_noise(sample_audio, 16000, strength=0.0)
            call_kwargs = mock_nr.reduce_noise.call_args[1]
            assert call_kwargs["prop_decrease"] == 0.0

    def test_strength_mapping_full(self, sample_audio: np.ndarray) -> None:
        """Strength 1.0 should result in prop_decrease=1.0."""
        mock_nr = MagicMock()
        mock_nr.reduce_noise.return_value = sample_audio

        with patch.dict("sys.modules", {"noisereduce": mock_nr}):
            from hark import preprocessor

            preprocessor.reduce_noise(sample_audio, 16000, strength=1.0)
            call_kwargs = mock_nr.reduce_noise.call_args[1]
            assert call_kwargs["prop_decrease"] == 1.0

    def test_returns_original_if_noisereduce_not_installed(self, sample_audio: np.ndarray) -> None:
        """Should return original audio if noisereduce is not installed."""
        with patch.dict("sys.modules", {"noisereduce": None}):
            result = reduce_noise(sample_audio, 16000, strength=0.5)
            np.testing.assert_array_equal(result, sample_audio)

    def test_raises_preprocessing_error_on_exception(self, sample_audio: np.ndarray) -> None:
        """Should raise PreprocessingError on other exceptions."""
        mock_nr = MagicMock()
        mock_nr.reduce_noise.side_effect = RuntimeError("NR failed")

        with patch.dict("sys.modules", {"noisereduce": mock_nr}):
            from hark import preprocessor

            with pytest.raises(PreprocessingError) as exc_info:
                preprocessor.reduce_noise(sample_audio, 16000)
            assert "noise reduction failed" in str(exc_info.value).lower()

    def test_output_dtype_float32(self, sample_audio: np.ndarray) -> None:
        """Output should be float32."""
        mock_nr = MagicMock()
        mock_nr.reduce_noise.return_value = sample_audio.astype(np.float64)

        with patch.dict("sys.modules", {"noisereduce": mock_nr}):
            from hark import preprocessor

            result = preprocessor.reduce_noise(sample_audio, 16000)
            assert result.dtype == np.float32


class TestNormalizeAudio:
    """Tests for normalize_audio function."""

    def test_empty_array_unchanged(self) -> None:
        """Empty array should be returned unchanged."""
        empty = np.array([], dtype=np.float32)
        result = normalize_audio(empty, target_db=-20.0)
        assert len(result) == 0

    def test_silent_audio_unchanged(self) -> None:
        """Audio with RMS < 1e-10 should be returned unchanged."""
        silent = np.zeros(1000, dtype=np.float32)
        result = normalize_audio(silent, target_db=-20.0)
        np.testing.assert_array_equal(result, silent)

    def test_applies_correct_gain(self) -> None:
        """Should apply correct gain to reach target dB."""
        # Create audio with known RMS
        # RMS = 0.1 → -20 dB
        audio = np.ones(1000, dtype=np.float32) * 0.1
        result = normalize_audio(audio, target_db=-20.0)

        # Result should have RMS ≈ 0.1 (already at -20 dB)
        result_rms = np.sqrt(np.mean(result**2))
        # -20 dB = 10^(-20/20) = 0.1
        expected_rms = 10 ** (-20 / 20)
        assert abs(result_rms - expected_rms) < 0.01

    def test_output_clipped(self) -> None:
        """Output should be clipped to [-1, 1]."""
        # Create quiet audio that will be boosted significantly
        audio = np.ones(1000, dtype=np.float32) * 0.001
        result = normalize_audio(audio, target_db=-3.0)

        # Should be clipped
        assert np.max(result) <= 1.0
        assert np.min(result) >= -1.0

    def test_output_dtype_float32(self) -> None:
        """Output should be float32."""
        audio = np.ones(1000, dtype=np.float64) * 0.5
        result = normalize_audio(audio, target_db=-20.0)
        assert result.dtype == np.float32

    def test_negative_target_db_works(self) -> None:
        """Negative target dB values should work correctly."""
        audio = np.random.randn(1000).astype(np.float32) * 0.1
        result = normalize_audio(audio, target_db=-30.0)

        result_rms = np.sqrt(np.mean(result**2))
        expected_rms = 10 ** (-30 / 20)
        # Allow some tolerance due to clipping
        assert abs(result_rms - expected_rms) < 0.01


class TestTrimSilence:
    """Tests for trim_silence function."""

    def test_empty_array_returns_empty(self) -> None:
        """Empty array should return (empty, 0.0)."""
        empty = np.array([], dtype=np.float32)
        result, trimmed = trim_silence(empty, 16000)
        assert len(result) == 0
        assert trimmed == 0.0

    def test_uses_librosa_split(self, sample_audio: np.ndarray) -> None:
        """Should use librosa.effects.split for silence detection."""
        with patch("librosa.effects.split") as mock_split:
            mock_split.return_value = np.array([[0, 1000], [2000, 3000]])
            trim_silence(sample_audio, 16000)
            mock_split.assert_called_once()

    def test_all_silence_returns_original(self) -> None:
        """Audio that is all silence should return original."""
        silent = np.zeros(16000, dtype=np.float32)

        with patch("librosa.effects.split") as mock_split:
            mock_split.return_value = np.array([]).reshape(0, 2)
            result, trimmed = trim_silence(silent, 16000)

            np.testing.assert_array_equal(result, silent)
            assert trimmed == 0.0

    def test_concatenates_segments_with_gaps(self) -> None:
        """Non-silent segments should be concatenated with small gaps."""
        audio = np.ones(32000, dtype=np.float32)

        with patch("librosa.effects.split") as mock_split:
            # Two segments: 0-8000, 16000-24000
            mock_split.return_value = np.array([[0, 8000], [16000, 24000]])
            result, trimmed = trim_silence(audio, 16000)

            # Result should be segment1 + gap (1600 samples @ 16kHz) + segment2
            expected_len = 8000 + 1600 + 8000  # 100ms gap = 1600 samples
            assert len(result) == expected_len

    def test_returns_trimmed_seconds(self) -> None:
        """Should return correct number of seconds trimmed."""
        audio = np.ones(32000, dtype=np.float32)  # 2 seconds

        with patch("librosa.effects.split") as mock_split:
            # One segment: first second only
            mock_split.return_value = np.array([[0, 16000]])
            result, trimmed = trim_silence(audio, 16000)

            # Original 2s, result 1s, so trimmed ≈ 1s
            assert trimmed > 0.9

    def test_raises_preprocessing_error_on_exception(self, sample_audio: np.ndarray) -> None:
        """Should raise PreprocessingError on exception."""
        with patch("librosa.effects.split") as mock_split:
            mock_split.side_effect = RuntimeError("Split failed")

            with pytest.raises(PreprocessingError) as exc_info:
                trim_silence(sample_audio, 16000)
            assert "silence trimming failed" in str(exc_info.value).lower()

    def test_output_dtype_float32(self) -> None:
        """Output should be float32."""
        audio = np.ones(16000, dtype=np.float64)

        with patch("librosa.effects.split") as mock_split:
            mock_split.return_value = np.array([[0, 8000]])
            result, _ = trim_silence(audio, 16000)
            assert result.dtype == np.float32


class TestAudioPreprocessor:
    """Tests for AudioPreprocessor class."""

    @pytest.fixture
    def full_config(self) -> PreprocessingConfig:
        """Config with all preprocessing enabled."""
        return PreprocessingConfig(
            noise_reduction=NoiseReductionConfig(enabled=True, strength=0.5),
            normalization=NormalizationConfig(enabled=True, target_level_db=-20.0),
            silence_trimming=SilenceTrimmingConfig(
                enabled=True, threshold_db=-40.0, min_silence_duration=0.5
            ),
        )

    @pytest.fixture
    def disabled_config(self) -> PreprocessingConfig:
        """Config with all preprocessing disabled."""
        return PreprocessingConfig(
            noise_reduction=NoiseReductionConfig(enabled=False),
            normalization=NormalizationConfig(enabled=False),
            silence_trimming=SilenceTrimmingConfig(enabled=False),
        )

    def test_loads_audio_file(
        self, temp_audio_file: Path, full_config: PreprocessingConfig
    ) -> None:
        """Should load audio file using soundfile."""
        preprocessor = AudioPreprocessor(full_config)

        with (
            patch("soundfile.read") as mock_read,
            patch("hark.preprocessor.reduce_noise", return_value=np.zeros(16000)),
            patch("librosa.effects.split", return_value=np.array([[0, 16000]])),
        ):
            mock_read.return_value = (np.zeros(16000, dtype=np.float32), 16000)
            preprocessor.process(temp_audio_file, 16000)
            mock_read.assert_called_once()

    def test_converts_stereo_to_mono(
        self, temp_audio_file: Path, full_config: PreprocessingConfig
    ) -> None:
        """Should convert stereo audio to mono."""
        stereo_audio = np.random.randn(16000, 2).astype(np.float32)

        with patch("soundfile.read") as mock_read:
            mock_read.return_value = (stereo_audio, 16000)
            with patch("hark.preprocessor.reduce_noise") as mock_nr:
                mock_nr.return_value = np.zeros(16000, dtype=np.float32)
                with patch("librosa.effects.split", return_value=np.array([[0, 16000]])):
                    preprocessor = AudioPreprocessor(full_config)
                    preprocessor.process(temp_audio_file, 16000)

                    # reduce_noise should receive mono audio
                    call_audio = mock_nr.call_args[0][0]
                    assert len(call_audio.shape) == 1

    def test_resamples_if_needed(
        self, temp_audio_file: Path, full_config: PreprocessingConfig
    ) -> None:
        """Should resample audio if sample rate differs."""
        audio_48k = np.zeros(48000, dtype=np.float32)

        with (
            patch("soundfile.read") as mock_read,
            patch("librosa.resample") as mock_resample,
            patch("hark.preprocessor.reduce_noise", return_value=np.zeros(16000)),
            patch("librosa.effects.split", return_value=np.array([[0, 16000]])),
        ):
            mock_read.return_value = (audio_48k, 48000)  # 48kHz file
            mock_resample.return_value = np.zeros(16000, dtype=np.float32)
            preprocessor = AudioPreprocessor(full_config)
            preprocessor.process(temp_audio_file, 16000)

            mock_resample.assert_called_once()
            call_kwargs = mock_resample.call_args[1]
            assert call_kwargs["orig_sr"] == 48000
            assert call_kwargs["target_sr"] == 16000

    def test_applies_noise_reduction_when_enabled(
        self, temp_audio_file: Path, full_config: PreprocessingConfig
    ) -> None:
        """Should apply noise reduction when enabled in config."""
        with (
            patch("soundfile.read") as mock_read,
            patch("hark.preprocessor.reduce_noise") as mock_nr,
            patch("librosa.effects.split", return_value=np.array([[0, 16000]])),
        ):
            mock_read.return_value = (np.zeros(16000, dtype=np.float32), 16000)
            mock_nr.return_value = np.zeros(16000, dtype=np.float32)
            preprocessor = AudioPreprocessor(full_config)
            _, result = preprocessor.process(temp_audio_file, 16000)

            mock_nr.assert_called_once()
            assert result.noise_reduction_applied is True

    def test_applies_normalization_when_enabled(
        self, temp_audio_file: Path, full_config: PreprocessingConfig
    ) -> None:
        """Should apply normalization when enabled in config."""
        with (
            patch("soundfile.read") as mock_read,
            patch("hark.preprocessor.reduce_noise", return_value=np.zeros(16000)),
            patch("hark.preprocessor.normalize_audio") as mock_norm,
            patch("librosa.effects.split", return_value=np.array([[0, 16000]])),
        ):
            mock_read.return_value = (np.zeros(16000, dtype=np.float32), 16000)
            mock_norm.return_value = np.zeros(16000, dtype=np.float32)
            preprocessor = AudioPreprocessor(full_config)
            _, result = preprocessor.process(temp_audio_file, 16000)

            mock_norm.assert_called_once()
            assert result.normalization_applied is True

    def test_applies_silence_trimming_when_enabled(
        self, temp_audio_file: Path, full_config: PreprocessingConfig
    ) -> None:
        """Should apply silence trimming when enabled in config."""
        with (
            patch("soundfile.read") as mock_read,
            patch("hark.preprocessor.reduce_noise", return_value=np.zeros(16000)),
            patch("hark.preprocessor.trim_silence") as mock_trim,
        ):
            mock_read.return_value = (np.zeros(16000, dtype=np.float32), 16000)
            mock_trim.return_value = (np.zeros(8000, dtype=np.float32), 0.5)
            preprocessor = AudioPreprocessor(full_config)
            _, result = preprocessor.process(temp_audio_file, 16000)

            mock_trim.assert_called_once()
            assert result.silence_trimmed_seconds == 0.5

    def test_skips_disabled_steps(
        self, temp_audio_file: Path, disabled_config: PreprocessingConfig
    ) -> None:
        """Should skip disabled preprocessing steps."""
        with (
            patch("soundfile.read") as mock_read,
            patch("hark.preprocessor.reduce_noise") as mock_nr,
            patch("hark.preprocessor.normalize_audio") as mock_norm,
            patch("hark.preprocessor.trim_silence") as mock_trim,
        ):
            mock_read.return_value = (np.zeros(16000, dtype=np.float32), 16000)
            preprocessor = AudioPreprocessor(disabled_config)
            _, result = preprocessor.process(temp_audio_file, 16000)

            mock_nr.assert_not_called()
            mock_norm.assert_not_called()
            mock_trim.assert_not_called()
            assert result.noise_reduction_applied is False
            assert result.normalization_applied is False
            assert result.silence_trimmed_seconds == 0.0

    def test_calls_progress_callback(
        self, temp_audio_file: Path, full_config: PreprocessingConfig
    ) -> None:
        """Should call progress callback for each step."""
        callback = MagicMock()

        with (
            patch("soundfile.read") as mock_read,
            patch("hark.preprocessor.reduce_noise", return_value=np.zeros(16000)),
            patch("librosa.effects.split", return_value=np.array([[0, 16000]])),
        ):
            mock_read.return_value = (np.zeros(16000, dtype=np.float32), 16000)
            preprocessor = AudioPreprocessor(full_config)
            preprocessor.process(temp_audio_file, 16000, progress_callback=callback)

            # Should be called for each step (start and end)
            assert callback.call_count >= 6  # 3 steps × 2 calls each

    def test_returns_preprocessing_result(
        self, temp_audio_file: Path, full_config: PreprocessingConfig
    ) -> None:
        """Should return populated PreprocessingResult."""
        with (
            patch("soundfile.read") as mock_read,
            patch("hark.preprocessor.reduce_noise", return_value=np.zeros(16000)),
            patch("librosa.effects.split", return_value=np.array([[0, 16000]])),
        ):
            mock_read.return_value = (np.zeros(16000, dtype=np.float32), 16000)
            preprocessor = AudioPreprocessor(full_config)
            audio, result = preprocessor.process(temp_audio_file, 16000)

            assert isinstance(result, PreprocessingResult)
            assert isinstance(audio, np.ndarray)
            assert result.original_duration > 0

    def test_file_not_found_raises(self, full_config: PreprocessingConfig, tmp_path: Path) -> None:
        """Should raise PreprocessingError if file not found."""
        preprocessor = AudioPreprocessor(full_config)
        nonexistent = tmp_path / "nonexistent.wav"

        with pytest.raises(PreprocessingError) as exc_info:
            preprocessor.process(nonexistent, 16000)
        assert "failed to load" in str(exc_info.value).lower()


class TestPreprocessingResult:
    """Tests for PreprocessingResult dataclass."""

    def test_dataclass_creation(self) -> None:
        """Should create PreprocessingResult with all fields."""
        result = PreprocessingResult(
            original_duration=10.0,
            processed_duration=8.5,
            noise_reduction_applied=True,
            normalization_applied=True,
            silence_trimmed_seconds=1.5,
        )

        assert result.original_duration == 10.0
        assert result.processed_duration == 8.5
        assert result.noise_reduction_applied is True
        assert result.normalization_applied is True
        assert result.silence_trimmed_seconds == 1.5

    def test_zero_values(self) -> None:
        """Should handle zero values."""
        result = PreprocessingResult(
            original_duration=0.0,
            processed_duration=0.0,
            noise_reduction_applied=False,
            normalization_applied=False,
            silence_trimmed_seconds=0.0,
        )

        assert result.original_duration == 0.0
        assert result.silence_trimmed_seconds == 0.0
