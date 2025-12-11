"""Tests for hark.config module."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from hark.config import (
    HarkConfig,
    InterfaceConfig,
    NoiseReductionConfig,
    NormalizationConfig,
    OutputConfig,
    PreprocessingConfig,
    RecordingConfig,
    SilenceTrimmingConfig,
    WhisperConfig,
    create_default_config_file,
    ensure_directories,
    get_default_config_path,
    load_config,
    merge_cli_args,
    validate_config,
)
from hark.constants import (
    DEFAULT_CHANNELS,
    DEFAULT_CONFIG_PATH,
    DEFAULT_INPUT_SOURCE,
    DEFAULT_LANGUAGE,
    DEFAULT_MAX_DURATION,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_FORMAT,
    DEFAULT_SAMPLE_RATE,
)
from hark.exceptions import ConfigError


class TestDataclassDefaults:
    """Tests for configuration dataclass default values."""

    def test_recording_config_defaults(self) -> None:
        """RecordingConfig should have correct default values."""
        config = RecordingConfig()
        assert config.sample_rate == DEFAULT_SAMPLE_RATE
        assert config.channels == DEFAULT_CHANNELS
        assert config.max_duration == DEFAULT_MAX_DURATION
        assert config.input_source == DEFAULT_INPUT_SOURCE

    def test_whisper_config_defaults(self) -> None:
        """WhisperConfig should have correct default values."""
        config = WhisperConfig()
        assert config.model == DEFAULT_MODEL
        assert config.language == DEFAULT_LANGUAGE
        assert config.device == "auto"

    def test_noise_reduction_config_defaults(self) -> None:
        """NoiseReductionConfig should have correct default values."""
        config = NoiseReductionConfig()
        assert config.enabled is True
        assert config.strength == 0.5

    def test_normalization_config_defaults(self) -> None:
        """NormalizationConfig should have correct default values."""
        config = NormalizationConfig()
        assert config.enabled is True
        assert config.target_level_db == -20.0

    def test_silence_trimming_config_defaults(self) -> None:
        """SilenceTrimmingConfig should have correct default values."""
        config = SilenceTrimmingConfig()
        assert config.enabled is True
        assert config.threshold_db == -40.0
        assert config.min_silence_duration == 0.5

    def test_preprocessing_config_nested(self) -> None:
        """PreprocessingConfig should have nested config objects."""
        config = PreprocessingConfig()
        assert isinstance(config.noise_reduction, NoiseReductionConfig)
        assert isinstance(config.normalization, NormalizationConfig)
        assert isinstance(config.silence_trimming, SilenceTrimmingConfig)

    def test_output_config_defaults(self) -> None:
        """OutputConfig should have correct default values."""
        config = OutputConfig()
        assert config.format == DEFAULT_OUTPUT_FORMAT
        assert config.timestamps is False
        assert config.append_mode is False
        assert config.encoding == "utf-8"

    def test_interface_config_defaults(self) -> None:
        """InterfaceConfig should have correct default values."""
        config = InterfaceConfig()
        assert config.quiet is False
        assert config.verbose is False
        assert config.color_output is True

    def test_hark_config_factory_defaults(self) -> None:
        """HarkConfig should use default_factory for nested configs."""
        config = HarkConfig()
        assert isinstance(config.recording, RecordingConfig)
        assert isinstance(config.whisper, WhisperConfig)
        assert isinstance(config.preprocessing, PreprocessingConfig)
        assert isinstance(config.output, OutputConfig)
        assert isinstance(config.interface, InterfaceConfig)


class TestGetDefaultConfigPath:
    """Tests for get_default_config_path function."""

    def test_returns_path(self) -> None:
        """Should return a Path object."""
        result = get_default_config_path()
        assert isinstance(result, Path)

    def test_returns_default_path(self) -> None:
        """Should return DEFAULT_CONFIG_PATH."""
        result = get_default_config_path()
        assert result == DEFAULT_CONFIG_PATH


class TestLoadConfig:
    """Tests for load_config function."""

    def test_no_file_returns_defaults(self, tmp_path: Path) -> None:
        """When config file doesn't exist, should return default HarkConfig."""
        nonexistent = tmp_path / "nonexistent.yaml"
        config = load_config(nonexistent)
        assert isinstance(config, HarkConfig)
        assert config.whisper.model == DEFAULT_MODEL

    def test_empty_yaml_returns_defaults(self, tmp_path: Path) -> None:
        """Empty YAML file should return defaults."""
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")
        config = load_config(config_file)
        assert isinstance(config, HarkConfig)

    def test_valid_yaml_parsed(self, tmp_path: Path, sample_config_yaml: str) -> None:
        """Valid YAML should be parsed correctly."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(sample_config_yaml)
        config = load_config(config_file)
        assert config.whisper.model == "base"
        assert config.recording.sample_rate == 16000

    def test_partial_yaml_uses_defaults(self, tmp_path: Path, partial_config_yaml: str) -> None:
        """Missing sections should use defaults."""
        config_file = tmp_path / "partial.yaml"
        config_file.write_text(partial_config_yaml)
        config = load_config(config_file)
        # Specified values
        assert config.whisper.model == "small"
        assert config.whisper.language == "en"
        assert config.output.format == "markdown"
        # Default values (not specified)
        assert config.recording.sample_rate == DEFAULT_SAMPLE_RATE

    def test_invalid_yaml_raises(self, tmp_path: Path) -> None:
        """Malformed YAML should raise ConfigError."""
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("invalid: [yaml: syntax")
        with pytest.raises(ConfigError):
            load_config(config_file)

    def test_io_error_raises(self, tmp_path: Path) -> None:
        """Read errors should raise ConfigError."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: value")

        with patch("builtins.open", side_effect=OSError("Read error")):
            with pytest.raises(ConfigError) as exc_info:
                load_config(config_file)
            assert "Read error" in str(exc_info.value) or "read" in str(exc_info.value).lower()

    def test_custom_path_used(self, tmp_path: Path) -> None:
        """Custom config path should be used."""
        custom_path = tmp_path / "custom" / "config.yaml"
        custom_path.parent.mkdir(parents=True)
        custom_path.write_text("whisper:\n  model: large-v3")
        config = load_config(custom_path)
        assert config.whisper.model == "large-v3"

    def test_none_path_uses_default(self) -> None:
        """None path should use default path."""
        with patch.object(Path, "exists", return_value=False):
            config = load_config(None)
            assert isinstance(config, HarkConfig)


class TestMergeCliArgs:
    """Tests for merge_cli_args function."""

    def test_recording_options_override(
        self, default_config: HarkConfig, cli_args_namespace: argparse.Namespace
    ) -> None:
        """Recording options should override config."""
        config = merge_cli_args(default_config, cli_args_namespace)
        assert config.recording.max_duration == 120

    def test_whisper_options_override(
        self, default_config: HarkConfig, cli_args_namespace: argparse.Namespace
    ) -> None:
        """Whisper options should override config."""
        config = merge_cli_args(default_config, cli_args_namespace)
        assert config.whisper.language == "en"
        assert config.whisper.model == "small"

    def test_preprocessing_flags(
        self, default_config: HarkConfig, cli_args_namespace: argparse.Namespace
    ) -> None:
        """Preprocessing flags should disable features."""
        config = merge_cli_args(default_config, cli_args_namespace)
        assert config.preprocessing.noise_reduction.enabled is False

    def test_noise_strength_override(
        self, default_config: HarkConfig, cli_args_namespace: argparse.Namespace
    ) -> None:
        """Noise strength should override config value."""
        config = merge_cli_args(default_config, cli_args_namespace)
        assert config.preprocessing.noise_reduction.strength == 0.7

    def test_output_options_override(
        self, default_config: HarkConfig, cli_args_namespace: argparse.Namespace
    ) -> None:
        """Output options should override config."""
        config = merge_cli_args(default_config, cli_args_namespace)
        assert config.output.timestamps is True
        assert config.output.format == "markdown"

    def test_interface_options_override(
        self, default_config: HarkConfig, cli_args_namespace: argparse.Namespace
    ) -> None:
        """Interface options should override config."""
        config = merge_cli_args(default_config, cli_args_namespace)
        assert config.interface.verbose is True

    def test_none_values_ignored(
        self, default_config: HarkConfig, empty_cli_args_namespace: argparse.Namespace
    ) -> None:
        """None values in args should not override config."""
        original_sample_rate = default_config.recording.sample_rate
        config = merge_cli_args(default_config, empty_cli_args_namespace)
        assert config.recording.sample_rate == original_sample_rate

    def test_preserves_unset_values(
        self, custom_config: HarkConfig, empty_cli_args_namespace: argparse.Namespace
    ) -> None:
        """Config values should be preserved when args are not set."""
        config = merge_cli_args(custom_config, empty_cli_args_namespace)
        assert config.recording.sample_rate == 44100
        assert config.whisper.model == "large-v3"

    def test_input_source_override(
        self, default_config: HarkConfig, empty_cli_args_namespace: argparse.Namespace
    ) -> None:
        """Input source CLI arg should override config."""
        empty_cli_args_namespace.input_source = "speaker"
        config = merge_cli_args(default_config, empty_cli_args_namespace)
        assert config.recording.input_source == "speaker"

    def test_input_source_none_preserves_default(
        self, default_config: HarkConfig, empty_cli_args_namespace: argparse.Namespace
    ) -> None:
        """None input_source arg should preserve config value."""
        empty_cli_args_namespace.input_source = None
        config = merge_cli_args(default_config, empty_cli_args_namespace)
        assert config.recording.input_source == "mic"

    def test_input_source_both_auto_sets_stereo(
        self, default_config: HarkConfig, empty_cli_args_namespace: argparse.Namespace
    ) -> None:
        """Input source 'both' should auto-set channels to 2 (stereo)."""
        empty_cli_args_namespace.input_source = "both"
        empty_cli_args_namespace.channels = None
        config = merge_cli_args(default_config, empty_cli_args_namespace)
        assert config.recording.input_source == "both"
        assert config.recording.channels == 2

    def test_input_source_both_respects_explicit_channels(
        self, default_config: HarkConfig, empty_cli_args_namespace: argparse.Namespace
    ) -> None:
        """Explicit channels arg should override auto-stereo for 'both' mode."""
        empty_cli_args_namespace.input_source = "both"
        empty_cli_args_namespace.channels = 2
        config = merge_cli_args(default_config, empty_cli_args_namespace)
        assert config.recording.channels == 2


class TestValidateConfig:
    """Tests for validate_config function."""

    def test_valid_config_returns_empty(self, default_config: HarkConfig) -> None:
        """Valid config should return empty error list."""
        errors = validate_config(default_config)
        assert errors == []

    def test_sample_rate_low(self, default_config: HarkConfig) -> None:
        """Sample rate < 8000 should produce error."""
        default_config.recording.sample_rate = 7999
        errors = validate_config(default_config)
        assert len(errors) >= 1
        assert any("sample" in e.lower() and "rate" in e.lower() for e in errors)

    def test_sample_rate_high(self, default_config: HarkConfig) -> None:
        """Sample rate > 48000 should produce error."""
        default_config.recording.sample_rate = 48001
        errors = validate_config(default_config)
        assert len(errors) >= 1

    def test_sample_rate_boundary_low(self, default_config: HarkConfig) -> None:
        """Sample rate of 8000 should be valid."""
        default_config.recording.sample_rate = 8000
        errors = validate_config(default_config)
        assert errors == []

    def test_sample_rate_boundary_high(self, default_config: HarkConfig) -> None:
        """Sample rate of 48000 should be valid."""
        default_config.recording.sample_rate = 48000
        errors = validate_config(default_config)
        assert errors == []

    def test_invalid_channels(self, default_config: HarkConfig) -> None:
        """Channels not 1 or 2 should produce error."""
        default_config.recording.channels = 3
        errors = validate_config(default_config)
        assert len(errors) >= 1
        assert any("channel" in e.lower() for e in errors)

    def test_negative_max_duration(self, default_config: HarkConfig) -> None:
        """Negative max_duration should produce error."""
        default_config.recording.max_duration = -1
        errors = validate_config(default_config)
        assert len(errors) >= 1

    def test_zero_max_duration(self, default_config: HarkConfig) -> None:
        """Zero max_duration should produce error."""
        default_config.recording.max_duration = 0
        errors = validate_config(default_config)
        assert len(errors) >= 1

    def test_invalid_model(self, default_config: HarkConfig) -> None:
        """Invalid model name should produce error."""
        default_config.whisper.model = "invalid-model"
        errors = validate_config(default_config)
        assert len(errors) >= 1
        assert any("model" in e.lower() for e in errors)

    def test_invalid_device(self, default_config: HarkConfig) -> None:
        """Invalid device should produce error."""
        default_config.whisper.device = "tpu"
        errors = validate_config(default_config)
        assert len(errors) >= 1
        assert any("device" in e.lower() for e in errors)

    def test_noise_strength_negative(self, default_config: HarkConfig) -> None:
        """Noise strength < 0 should produce error."""
        default_config.preprocessing.noise_reduction.strength = -0.1
        errors = validate_config(default_config)
        assert len(errors) >= 1

    def test_noise_strength_over_one(self, default_config: HarkConfig) -> None:
        """Noise strength > 1 should produce error."""
        default_config.preprocessing.noise_reduction.strength = 1.1
        errors = validate_config(default_config)
        assert len(errors) >= 1

    def test_positive_target_db(self, default_config: HarkConfig) -> None:
        """Target level dB > 0 should produce error."""
        default_config.preprocessing.normalization.target_level_db = 5.0
        errors = validate_config(default_config)
        assert len(errors) >= 1

    def test_invalid_format(self, default_config: HarkConfig) -> None:
        """Invalid output format should produce error."""
        default_config.output.format = "pdf"
        errors = validate_config(default_config)
        assert len(errors) >= 1
        assert any("format" in e.lower() for e in errors)

    def test_invalid_input_source(self, default_config: HarkConfig) -> None:
        """Invalid input source should produce error."""
        default_config.recording.input_source = "invalid"
        errors = validate_config(default_config)
        assert len(errors) >= 1
        assert any("input" in e.lower() and "source" in e.lower() for e in errors)

    def test_valid_input_sources(self, default_config: HarkConfig) -> None:
        """Valid input sources should not produce errors."""
        for source in ["mic", "speaker", "both"]:
            default_config.recording.input_source = source
            if source == "both":
                default_config.recording.channels = 2
            else:
                default_config.recording.channels = 1
            errors = validate_config(default_config)
            assert errors == [], f"Unexpected errors for input_source={source}: {errors}"

    def test_both_mode_requires_stereo(self, default_config: HarkConfig) -> None:
        """Input source 'both' should require channels=2."""
        default_config.recording.input_source = "both"
        default_config.recording.channels = 1
        errors = validate_config(default_config)
        assert len(errors) >= 1
        assert any("channel" in e.lower() for e in errors)

    def test_both_mode_with_stereo_valid(self, default_config: HarkConfig) -> None:
        """Input source 'both' with channels=2 should be valid."""
        default_config.recording.input_source = "both"
        default_config.recording.channels = 2
        errors = validate_config(default_config)
        assert errors == []

    def test_multiple_errors(self, default_config: HarkConfig) -> None:
        """Multiple issues should produce multiple errors."""
        default_config.recording.sample_rate = 1000
        default_config.recording.channels = 5
        default_config.whisper.model = "invalid"
        errors = validate_config(default_config)
        assert len(errors) >= 3


class TestEnsureDirectories:
    """Tests for ensure_directories function."""

    def test_creates_config_dir(self, default_config: HarkConfig, tmp_path: Path) -> None:
        """Should create config directory."""
        with patch("hark.config.DEFAULT_CONFIG_DIR", tmp_path / "config" / "hark"):
            ensure_directories(default_config)
            assert (tmp_path / "config" / "hark").exists()

    def test_creates_temp_dir(self, default_config: HarkConfig, tmp_path: Path) -> None:
        """Should create temp directory."""
        default_config.temp_directory = tmp_path / "temp" / "hark"
        with patch("hark.config.DEFAULT_CONFIG_DIR", tmp_path / "config" / "hark"):
            ensure_directories(default_config)
            assert default_config.temp_directory.exists()

    def test_creates_cache_dir(self, default_config: HarkConfig, tmp_path: Path) -> None:
        """Should create model cache directory."""
        default_config.model_cache_dir = tmp_path / "cache" / "hark" / "models"
        with patch("hark.config.DEFAULT_CONFIG_DIR", tmp_path / "config" / "hark"):
            ensure_directories(default_config)
            assert default_config.model_cache_dir.exists()

    def test_idempotent(self, default_config: HarkConfig, tmp_path: Path) -> None:
        """Should not error if directories already exist."""
        default_config.temp_directory = tmp_path / "temp"
        default_config.model_cache_dir = tmp_path / "cache"
        default_config.temp_directory.mkdir(parents=True)
        default_config.model_cache_dir.mkdir(parents=True)

        with patch("hark.config.DEFAULT_CONFIG_DIR", tmp_path / "config"):
            # Should not raise
            ensure_directories(default_config)
            ensure_directories(default_config)


class TestCreateDefaultConfigFile:
    """Tests for create_default_config_file function."""

    def test_creates_file(self, tmp_path: Path) -> None:
        """Should create config file."""
        config_path = tmp_path / "config.yaml"
        result = create_default_config_file(config_path)
        assert result == config_path
        assert config_path.exists()

    def test_content_is_valid_yaml(self, tmp_path: Path) -> None:
        """Created file should contain valid YAML."""
        config_path = tmp_path / "config.yaml"
        create_default_config_file(config_path)
        content = config_path.read_text()
        # Should not raise
        parsed = yaml.safe_load(content)
        assert parsed is not None

    def test_content_has_expected_structure(self, tmp_path: Path) -> None:
        """Created file should have expected sections."""
        config_path = tmp_path / "config.yaml"
        create_default_config_file(config_path)
        content = config_path.read_text()
        assert "recording:" in content
        assert "whisper:" in content
        assert "preprocessing:" in content
        assert "output:" in content
        assert "interface:" in content

    def test_custom_path_used(self, tmp_path: Path) -> None:
        """Should use provided path."""
        custom_path = tmp_path / "custom" / "dir" / "config.yaml"
        result = create_default_config_file(custom_path)
        assert result == custom_path
        assert custom_path.exists()

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """Should create parent directories if needed."""
        nested_path = tmp_path / "a" / "b" / "c" / "config.yaml"
        create_default_config_file(nested_path)
        assert nested_path.exists()


class TestConfigYamlParsing:
    """Tests for YAML parsing edge cases."""

    def test_parses_different_sample_rates(self, tmp_path: Path) -> None:
        """Should parse various sample rate values."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("recording:\n  sample_rate: 44100")
        config = load_config(config_file)
        assert config.recording.sample_rate == 44100

    def test_parses_boolean_values(self, tmp_path: Path) -> None:
        """Should parse boolean values correctly."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("interface:\n  quiet: true\n  verbose: false\n  color_output: yes")
        config = load_config(config_file)
        assert config.interface.quiet is True
        assert config.interface.verbose is False
        assert config.interface.color_output is True

    def test_parses_float_values(self, tmp_path: Path) -> None:
        """Should parse float values correctly."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("preprocessing:\n  noise_reduction:\n    strength: 0.75")
        config = load_config(config_file)
        assert config.preprocessing.noise_reduction.strength == 0.75

    def test_expands_home_in_paths(self, tmp_path: Path) -> None:
        """Should expand ~ in cache paths."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("cache:\n  model_cache_dir: ~/.cache/test")
        config = load_config(config_file)
        # Path should be expanded
        assert "~" not in str(config.model_cache_dir)

    def test_parses_input_source(self, tmp_path: Path) -> None:
        """Should parse input_source from YAML."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("recording:\n  input_source: speaker")
        config = load_config(config_file)
        assert config.recording.input_source == "speaker"

    def test_parses_input_source_both(self, tmp_path: Path) -> None:
        """Should parse input_source 'both' from YAML."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("recording:\n  input_source: both\n  channels: 2")
        config = load_config(config_file)
        assert config.recording.input_source == "both"
        assert config.recording.channels == 2
