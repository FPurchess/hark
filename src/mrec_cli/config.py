"""Configuration management for mrec-cli."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "RecordingConfig",
    "WhisperConfig",
    "NoiseReductionConfig",
    "NormalizationConfig",
    "SilenceTrimmingConfig",
    "PreprocessingConfig",
    "OutputConfig",
    "InterfaceConfig",
    "MrecConfig",
    "get_default_config_path",
    "load_config",
    "merge_cli_args",
    "validate_config",
    "ensure_directories",
    "create_default_config_file",
]

from mrec_cli.constants import (
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
    VALID_MODELS,
    VALID_OUTPUT_FORMATS,
)
from mrec_cli.exceptions import ConfigError


@dataclass
class RecordingConfig:
    """Audio recording configuration."""

    sample_rate: int = DEFAULT_SAMPLE_RATE
    channels: int = DEFAULT_CHANNELS
    max_duration: int = DEFAULT_MAX_DURATION


@dataclass
class WhisperConfig:
    """Whisper model configuration."""

    model: str = DEFAULT_MODEL
    language: str = DEFAULT_LANGUAGE
    device: str = "auto"  # auto, cpu, cuda, vulkan


@dataclass
class NoiseReductionConfig:
    """Noise reduction settings."""

    enabled: bool = True
    strength: float = DEFAULT_NOISE_STRENGTH


@dataclass
class NormalizationConfig:
    """Audio normalization settings."""

    enabled: bool = True
    target_level_db: float = DEFAULT_TARGET_LEVEL_DB


@dataclass
class SilenceTrimmingConfig:
    """Silence trimming settings."""

    enabled: bool = True
    threshold_db: float = DEFAULT_SILENCE_THRESHOLD_DB
    min_silence_duration: float = DEFAULT_MIN_SILENCE_DURATION


@dataclass
class PreprocessingConfig:
    """Audio preprocessing configuration."""

    noise_reduction: NoiseReductionConfig = field(default_factory=NoiseReductionConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    silence_trimming: SilenceTrimmingConfig = field(default_factory=SilenceTrimmingConfig)


@dataclass
class OutputConfig:
    """Output configuration."""

    format: str = DEFAULT_OUTPUT_FORMAT
    timestamps: bool = False
    append_mode: bool = False
    encoding: str = DEFAULT_ENCODING


@dataclass
class InterfaceConfig:
    """Interface configuration."""

    quiet: bool = False
    verbose: bool = False
    color_output: bool = True


@dataclass
class MrecConfig:
    """Main configuration class."""

    recording: RecordingConfig = field(default_factory=RecordingConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    interface: InterfaceConfig = field(default_factory=InterfaceConfig)
    temp_directory: Path = DEFAULT_TEMP_DIR
    model_cache_dir: Path = DEFAULT_MODEL_CACHE_DIR


def get_default_config_path() -> Path:
    """Return the default configuration file path."""
    return DEFAULT_CONFIG_PATH


def _dict_to_config(data: dict[str, Any]) -> MrecConfig:
    """Convert a dictionary to MrecConfig."""
    config = MrecConfig()

    if "recording" in data:
        rec = data["recording"]
        config.recording = RecordingConfig(
            sample_rate=rec.get("sample_rate", DEFAULT_SAMPLE_RATE),
            channels=rec.get("channels", DEFAULT_CHANNELS),
            max_duration=rec.get("max_duration", DEFAULT_MAX_DURATION),
        )

    if "whisper" in data:
        w = data["whisper"]
        config.whisper = WhisperConfig(
            model=w.get("model", DEFAULT_MODEL),
            language=w.get("language", DEFAULT_LANGUAGE),
            device=w.get("device", "auto"),
        )

    if "preprocessing" in data:
        p = data["preprocessing"]
        nr = p.get("noise_reduction", {})
        norm = p.get("normalization", {})
        st = p.get("silence_trimming", {})

        config.preprocessing = PreprocessingConfig(
            noise_reduction=NoiseReductionConfig(
                enabled=nr.get("enabled", True),
                strength=nr.get("strength", DEFAULT_NOISE_STRENGTH),
            ),
            normalization=NormalizationConfig(
                enabled=norm.get("enabled", True),
                target_level_db=norm.get("target_level", DEFAULT_TARGET_LEVEL_DB),
            ),
            silence_trimming=SilenceTrimmingConfig(
                enabled=st.get("enabled", True),
                threshold_db=st.get("threshold", DEFAULT_SILENCE_THRESHOLD_DB),
                min_silence_duration=st.get("min_silence_duration", DEFAULT_MIN_SILENCE_DURATION),
            ),
        )

    if "output" in data:
        o = data["output"]
        config.output = OutputConfig(
            format=o.get("format", DEFAULT_OUTPUT_FORMAT),
            timestamps=o.get("timestamps", False),
            append_mode=o.get("append_mode", False),
            encoding=o.get("encoding", DEFAULT_ENCODING),
        )

    if "interface" in data:
        i = data["interface"]
        config.interface = InterfaceConfig(
            quiet=i.get("quiet", False),
            verbose=i.get("verbose", False),
            color_output=i.get("color_output", True),
        )

    if "performance" in data:
        perf = data["performance"]
        if "temp_directory" in perf:
            config.temp_directory = Path(perf["temp_directory"])

    if "cache" in data:
        cache = data["cache"]
        if "model_cache_dir" in cache:
            config.model_cache_dir = Path(cache["model_cache_dir"]).expanduser()

    return config


def load_config(config_path: Path | None = None) -> MrecConfig:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, uses default path.

    Returns:
        MrecConfig with loaded or default values.

    Raises:
        ConfigError: If config file exists but cannot be parsed.
    """
    path = config_path or get_default_config_path()

    if not path.exists():
        return MrecConfig()

    try:
        with open(path) as f:
            data = yaml.safe_load(f)

        if data is None:
            return MrecConfig()

        return _dict_to_config(data)

    except yaml.YAMLError as e:
        raise ConfigError(f"Failed to parse config file {path}: {e}") from e
    except OSError as e:
        raise ConfigError(f"Failed to read config file {path}: {e}") from e


def merge_cli_args(config: MrecConfig, args: argparse.Namespace) -> MrecConfig:
    """
    Merge CLI arguments into config (CLI takes precedence).

    Args:
        config: Base configuration.
        args: Parsed CLI arguments.

    Returns:
        Updated configuration with CLI overrides.
    """
    # Recording options
    if getattr(args, "max_duration", None) is not None:
        config.recording.max_duration = args.max_duration
    if getattr(args, "sample_rate", None) is not None:
        config.recording.sample_rate = args.sample_rate
    if getattr(args, "channels", None) is not None:
        config.recording.channels = args.channels

    # Whisper options
    if getattr(args, "lang", None) is not None:
        config.whisper.language = args.lang
    if getattr(args, "model", None) is not None:
        config.whisper.model = args.model

    # Preprocessing options
    if getattr(args, "no_noise_reduction", False):
        config.preprocessing.noise_reduction.enabled = False
    if getattr(args, "no_normalize", False):
        config.preprocessing.normalization.enabled = False
    if getattr(args, "no_trim_silence", False):
        config.preprocessing.silence_trimming.enabled = False
    if getattr(args, "noise_strength", None) is not None:
        config.preprocessing.noise_reduction.strength = args.noise_strength

    # Output options
    if getattr(args, "timestamps", False):
        config.output.timestamps = True
    if getattr(args, "format", None) is not None:
        config.output.format = args.format
    if getattr(args, "append", False):
        config.output.append_mode = True

    # Interface options
    if getattr(args, "quiet", False):
        config.interface.quiet = True
    if getattr(args, "verbose", False):
        config.interface.verbose = True

    return config


def validate_config(config: MrecConfig) -> list[str]:
    """
    Validate configuration values.

    Args:
        config: Configuration to validate.

    Returns:
        List of error messages (empty if valid).
    """
    errors: list[str] = []

    # Recording validation
    if not (8000 <= config.recording.sample_rate <= 48000):
        errors.append(
            f"Sample rate must be between 8000 and 48000 Hz, got {config.recording.sample_rate}"
        )
    if config.recording.channels not in (1, 2):
        errors.append(f"Channels must be 1 or 2, got {config.recording.channels}")
    if config.recording.max_duration <= 0:
        errors.append(f"Max duration must be positive, got {config.recording.max_duration}")

    # Whisper validation
    if config.whisper.model not in VALID_MODELS:
        errors.append(
            f"Invalid model '{config.whisper.model}'. Valid models: {', '.join(VALID_MODELS)}"
        )
    if config.whisper.device not in ("auto", "cpu", "cuda", "vulkan"):
        errors.append(f"Invalid device '{config.whisper.device}'. Valid: auto, cpu, cuda, vulkan")

    # Preprocessing validation
    noise_strength = config.preprocessing.noise_reduction.strength
    if not (0.0 <= noise_strength <= 1.0):
        errors.append(f"Noise strength must be between 0.0 and 1.0, got {noise_strength}")
    target_db = config.preprocessing.normalization.target_level_db
    if target_db > 0:
        errors.append(f"Target level must be <= 0 dB, got {target_db}")

    # Output validation
    if config.output.format not in VALID_OUTPUT_FORMATS:
        errors.append(
            f"Invalid format '{config.output.format}'. Valid: {', '.join(VALID_OUTPUT_FORMATS)}"
        )

    return errors


def ensure_directories(config: MrecConfig) -> None:
    """
    Create necessary directories if they don't exist.

    Args:
        config: Configuration with directory paths.
    """
    DEFAULT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config.temp_directory.mkdir(parents=True, exist_ok=True)
    config.model_cache_dir.mkdir(parents=True, exist_ok=True)


def create_default_config_file(path: Path | None = None) -> Path:
    """
    Create a default configuration file.

    Args:
        path: Path for the config file. If None, uses default path.

    Returns:
        Path to the created config file.
    """
    config_path = path or get_default_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)

    default_config = """\
# MREC-CLI Configuration
# See CLI_SPECIFICATION.md for full documentation

# Audio Recording Settings
recording:
  sample_rate: 16000
  channels: 1
  max_duration: 600  # 10 minutes

# Whisper Model Settings
whisper:
  model: base  # tiny, base, small, medium, large, large-v2, large-v3
  language: auto  # or specific language code (en, de, etc.)
  device: auto  # cpu, cuda, vulkan, or auto-detect

# Audio Preprocessing
preprocessing:
  noise_reduction:
    enabled: true
    strength: 0.5  # 0.0-1.0

  normalization:
    enabled: true
    target_level: -20  # dB

  silence_trimming:
    enabled: true
    threshold: -40  # dB
    min_silence_duration: 0.5  # seconds

# Output Settings
output:
  format: plain  # plain, markdown, srt
  timestamps: false
  append_mode: false
  encoding: utf-8

# Interface Settings
interface:
  quiet: false
  verbose: false
  color_output: true

# Performance Settings
performance:
  temp_directory: /tmp/mrec

# Cache Settings
cache:
  model_cache_dir: ~/.cache/mrec/models
"""

    with open(config_path, "w") as f:
        f.write(default_config)

    return config_path
