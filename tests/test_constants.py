"""Tests for mrec_cli.constants module."""

import re
from pathlib import Path

from mrec_cli.constants import (
    DEFAULT_BUFFER_SIZE,
    DEFAULT_CACHE_DIR,
    DEFAULT_CHANNELS,
    DEFAULT_CONFIG_DIR,
    DEFAULT_CONFIG_PATH,
    DEFAULT_ENCODING,
    DEFAULT_LANGUAGE,
    DEFAULT_MAX_DURATION,
    DEFAULT_MIN_SILENCE_DURATION,
    DEFAULT_MODEL,
    DEFAULT_MODEL_CACHE_DIR,
    DEFAULT_NOISE_STRENGTH,
    DEFAULT_OUTPUT_FORMAT,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SILENCE_THRESHOLD_DB,
    DEFAULT_TARGET_LEVEL_DB,
    DEFAULT_TEMP_DIR,
    EXIT_ERROR,
    EXIT_INTERRUPT,
    EXIT_SUCCESS,
    MIN_RECORDING_DURATION,
    VALID_MODELS,
    VALID_OUTPUT_FORMATS,
    VERSION,
)


class TestVersion:
    """Tests for VERSION constant."""

    def test_version_format(self) -> None:
        """VERSION should be a valid semver string."""
        # Basic semver pattern: X.Y.Z with optional pre-release
        pattern = r"^\d+\.\d+\.\d+(-[a-zA-Z0-9.]+)?$"
        assert re.match(pattern, VERSION), f"VERSION '{VERSION}' is not valid semver"

    def test_version_is_string(self) -> None:
        """VERSION should be a string."""
        assert isinstance(VERSION, str)


class TestValidModels:
    """Tests for VALID_MODELS constant."""

    def test_valid_models_is_list(self) -> None:
        """VALID_MODELS should be a list."""
        assert isinstance(VALID_MODELS, list)

    def test_valid_models_contains_expected(self) -> None:
        """VALID_MODELS should contain all expected model names."""
        expected = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
        for model in expected:
            assert model in VALID_MODELS, f"'{model}' missing from VALID_MODELS"

    def test_valid_models_count(self) -> None:
        """VALID_MODELS should have exactly 7 models."""
        assert len(VALID_MODELS) == 7

    def test_default_model_in_valid_models(self) -> None:
        """DEFAULT_MODEL should be in VALID_MODELS."""
        assert DEFAULT_MODEL in VALID_MODELS


class TestValidOutputFormats:
    """Tests for VALID_OUTPUT_FORMATS constant."""

    def test_valid_output_formats_is_list(self) -> None:
        """VALID_OUTPUT_FORMATS should be a list."""
        assert isinstance(VALID_OUTPUT_FORMATS, list)

    def test_valid_output_formats_contains_expected(self) -> None:
        """VALID_OUTPUT_FORMATS should contain expected formats."""
        expected = ["plain", "markdown", "srt"]
        for fmt in expected:
            assert fmt in VALID_OUTPUT_FORMATS, f"'{fmt}' missing from VALID_OUTPUT_FORMATS"

    def test_valid_output_formats_count(self) -> None:
        """VALID_OUTPUT_FORMATS should have exactly 3 formats."""
        assert len(VALID_OUTPUT_FORMATS) == 3

    def test_default_format_in_valid_formats(self) -> None:
        """DEFAULT_OUTPUT_FORMAT should be in VALID_OUTPUT_FORMATS."""
        assert DEFAULT_OUTPUT_FORMAT in VALID_OUTPUT_FORMATS


class TestExitCodes:
    """Tests for exit code constants."""

    def test_exit_success(self) -> None:
        """EXIT_SUCCESS should be 0."""
        assert EXIT_SUCCESS == 0

    def test_exit_error(self) -> None:
        """EXIT_ERROR should be 1."""
        assert EXIT_ERROR == 1

    def test_exit_interrupt(self) -> None:
        """EXIT_INTERRUPT should be 130 (standard Ctrl+C code)."""
        assert EXIT_INTERRUPT == 130

    def test_exit_codes_are_integers(self) -> None:
        """All exit codes should be integers."""
        assert isinstance(EXIT_SUCCESS, int)
        assert isinstance(EXIT_ERROR, int)
        assert isinstance(EXIT_INTERRUPT, int)


class TestPathConstants:
    """Tests for path constants."""

    def test_config_dir_is_path(self) -> None:
        """DEFAULT_CONFIG_DIR should be a Path."""
        assert isinstance(DEFAULT_CONFIG_DIR, Path)

    def test_config_path_is_path(self) -> None:
        """DEFAULT_CONFIG_PATH should be a Path."""
        assert isinstance(DEFAULT_CONFIG_PATH, Path)

    def test_cache_dir_is_path(self) -> None:
        """DEFAULT_CACHE_DIR should be a Path."""
        assert isinstance(DEFAULT_CACHE_DIR, Path)

    def test_model_cache_dir_is_path(self) -> None:
        """DEFAULT_MODEL_CACHE_DIR should be a Path."""
        assert isinstance(DEFAULT_MODEL_CACHE_DIR, Path)

    def test_temp_dir_is_path(self) -> None:
        """DEFAULT_TEMP_DIR should be a Path."""
        assert isinstance(DEFAULT_TEMP_DIR, Path)

    def test_config_path_in_config_dir(self) -> None:
        """DEFAULT_CONFIG_PATH should be under DEFAULT_CONFIG_DIR."""
        assert DEFAULT_CONFIG_PATH.parent == DEFAULT_CONFIG_DIR

    def test_model_cache_in_cache_dir(self) -> None:
        """DEFAULT_MODEL_CACHE_DIR should be under DEFAULT_CACHE_DIR."""
        assert DEFAULT_CACHE_DIR in DEFAULT_MODEL_CACHE_DIR.parents

    def test_config_dir_contains_mrec(self) -> None:
        """DEFAULT_CONFIG_DIR should contain 'mrec' in path."""
        assert "mrec" in str(DEFAULT_CONFIG_DIR)

    def test_temp_dir_path(self) -> None:
        """DEFAULT_TEMP_DIR should be /tmp/mrec."""
        assert str(DEFAULT_TEMP_DIR) == "/tmp/mrec"


class TestAudioDefaults:
    """Tests for audio-related default constants."""

    def test_sample_rate_is_16khz(self) -> None:
        """DEFAULT_SAMPLE_RATE should be 16000 (Whisper's expected rate)."""
        assert DEFAULT_SAMPLE_RATE == 16000

    def test_channels_is_mono(self) -> None:
        """DEFAULT_CHANNELS should be 1 (mono)."""
        assert DEFAULT_CHANNELS == 1

    def test_max_duration_is_reasonable(self) -> None:
        """DEFAULT_MAX_DURATION should be 600 seconds (10 minutes)."""
        assert DEFAULT_MAX_DURATION == 600

    def test_buffer_size_is_power_of_two(self) -> None:
        """DEFAULT_BUFFER_SIZE should be a power of 2 for efficiency."""
        assert DEFAULT_BUFFER_SIZE > 0
        assert (DEFAULT_BUFFER_SIZE & (DEFAULT_BUFFER_SIZE - 1)) == 0

    def test_min_recording_duration(self) -> None:
        """MIN_RECORDING_DURATION should be 0.5 seconds."""
        assert MIN_RECORDING_DURATION == 0.5


class TestPreprocessingDefaults:
    """Tests for preprocessing default constants."""

    def test_noise_strength_in_range(self) -> None:
        """DEFAULT_NOISE_STRENGTH should be between 0 and 1."""
        assert 0.0 <= DEFAULT_NOISE_STRENGTH <= 1.0

    def test_target_level_db_negative(self) -> None:
        """DEFAULT_TARGET_LEVEL_DB should be negative (dB below 0)."""
        assert DEFAULT_TARGET_LEVEL_DB < 0

    def test_silence_threshold_db_negative(self) -> None:
        """DEFAULT_SILENCE_THRESHOLD_DB should be negative."""
        assert DEFAULT_SILENCE_THRESHOLD_DB < 0

    def test_min_silence_duration_positive(self) -> None:
        """DEFAULT_MIN_SILENCE_DURATION should be positive."""
        assert DEFAULT_MIN_SILENCE_DURATION > 0


class TestOtherDefaults:
    """Tests for other default constants."""

    def test_default_language(self) -> None:
        """DEFAULT_LANGUAGE should be 'auto'."""
        assert DEFAULT_LANGUAGE == "auto"

    def test_default_model(self) -> None:
        """DEFAULT_MODEL should be 'base'."""
        assert DEFAULT_MODEL == "base"

    def test_default_encoding(self) -> None:
        """DEFAULT_ENCODING should be 'utf-8'."""
        assert DEFAULT_ENCODING == "utf-8"

    def test_default_output_format(self) -> None:
        """DEFAULT_OUTPUT_FORMAT should be 'plain'."""
        assert DEFAULT_OUTPUT_FORMAT == "plain"
