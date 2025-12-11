"""Configuration management for hark."""

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
    "DiarizationConfig",
    "HarkConfig",
    "get_default_config_path",
    "load_config",
    "merge_cli_args",
    "validate_config",
    "ensure_directories",
    "create_default_config_file",
]

from hark.constants import (
    DEFAULT_CHANNELS,
    DEFAULT_CONFIG_DIR,
    DEFAULT_CONFIG_PATH,
    DEFAULT_DIARIZATION_MODEL,
    DEFAULT_ENCODING,
    DEFAULT_INPUT_SOURCE,
    DEFAULT_LANGUAGE,
    DEFAULT_MAX_DURATION,
    DEFAULT_MIN_SILENCE_DURATION,
    DEFAULT_MODEL,
    DEFAULT_MODEL_CACHE_DIR,
    DEFAULT_NOISE_STRENGTH,
    DEFAULT_OUTPUT_FORMAT,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SILENCE_THRESHOLD_DB,
    DEFAULT_SPEAKERS_DIR,
    DEFAULT_TARGET_LEVEL_DB,
    DEFAULT_TEMP_DIR,
    VALID_INPUT_SOURCES,
    VALID_MODELS,
    VALID_OUTPUT_FORMATS,
)
from hark.exceptions import ConfigError


@dataclass
class RecordingConfig:
    """Audio recording configuration."""

    sample_rate: int = DEFAULT_SAMPLE_RATE
    channels: int = DEFAULT_CHANNELS
    max_duration: int = DEFAULT_MAX_DURATION
    input_source: str = DEFAULT_INPUT_SOURCE


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
class DiarizationConfig:
    """Speaker diarization configuration."""

    hf_token: str | None = None
    model: str = DEFAULT_DIARIZATION_MODEL
    local_speaker_name: str | None = None  # Name for SPEAKER_00 in --input both
    speakers_dir: Path = field(default_factory=lambda: DEFAULT_SPEAKERS_DIR)


@dataclass
class HarkConfig:
    """Main configuration class."""

    recording: RecordingConfig = field(default_factory=RecordingConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    interface: InterfaceConfig = field(default_factory=InterfaceConfig)
    diarization: DiarizationConfig = field(default_factory=DiarizationConfig)
    temp_directory: Path = DEFAULT_TEMP_DIR
    model_cache_dir: Path = DEFAULT_MODEL_CACHE_DIR


def get_default_config_path() -> Path:
    """Return the default configuration file path."""
    return DEFAULT_CONFIG_PATH


def _dict_to_config(data: dict[str, Any]) -> HarkConfig:
    """Convert a dictionary to HarkConfig."""
    config = HarkConfig()

    if "recording" in data:
        rec = data["recording"]
        config.recording = RecordingConfig(
            sample_rate=rec.get("sample_rate", DEFAULT_SAMPLE_RATE),
            channels=rec.get("channels", DEFAULT_CHANNELS),
            max_duration=rec.get("max_duration", DEFAULT_MAX_DURATION),
            input_source=rec.get("input_source", DEFAULT_INPUT_SOURCE),
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

    if "diarization" in data:
        d = data["diarization"]
        config.diarization = DiarizationConfig(
            hf_token=d.get("hf_token"),
            model=d.get("model", DEFAULT_DIARIZATION_MODEL),
            local_speaker_name=d.get("local_speaker_name"),
            speakers_dir=Path(d.get("speakers_dir", DEFAULT_SPEAKERS_DIR)).expanduser(),
        )

    return config


def load_config(config_path: Path | None = None) -> HarkConfig:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, uses default path.

    Returns:
        HarkConfig with loaded or default values.

    Raises:
        ConfigError: If config file exists but cannot be parsed.
    """
    path = config_path or get_default_config_path()

    if not path.exists():
        return HarkConfig()

    try:
        with open(path) as f:
            data = yaml.safe_load(f)

        if data is None:
            return HarkConfig()

        return _dict_to_config(data)

    except yaml.YAMLError as e:
        raise ConfigError(f"Failed to parse config file {path}: {e}") from e
    except OSError as e:
        raise ConfigError(f"Failed to read config file {path}: {e}") from e


def merge_cli_args(config: HarkConfig, args: argparse.Namespace) -> HarkConfig:
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
    if getattr(args, "input_source", None) is not None:
        config.recording.input_source = args.input_source

    # Auto-set channels to 2 when using 'both' input source (stereo required)
    if config.recording.input_source == "both" and getattr(args, "channels", None) is None:
        config.recording.channels = 2

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


def validate_config(config: HarkConfig) -> list[str]:
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
    if config.recording.input_source not in VALID_INPUT_SOURCES:
        errors.append(
            f"Invalid input source '{config.recording.input_source}'. "
            f"Valid options: {', '.join(VALID_INPUT_SOURCES)}"
        )
    if config.recording.input_source == "both" and config.recording.channels != 2:
        errors.append("Channels must be 2 when input_source is 'both' (stereo: L=mic, R=speaker)")

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


def ensure_directories(config: HarkConfig) -> None:
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
# Hark Configuration

# Audio Recording Settings
recording:
  sample_rate: 16000
  channels: 1
  max_duration: 600  # 10 minutes
  input_source: mic  # mic, speaker, or both

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
  temp_directory: /tmp/hark

# Cache Settings
cache:
  model_cache_dir: ~/.cache/hark/models

# Speaker Diarization Settings (requires: pip install hark-cli[diarization])
# diarization:
#   hf_token: "hf_xxxxxxxxxxxxx"  # Required: HuggingFace token
#   model: "pyannote/speaker-diarization-3.1"
#   local_speaker_name: null  # Name for local mic in --input both (null = SPEAKER_00)
#   speakers_dir: ~/.config/hark/speakers  # Voice profile storage
"""

    with open(config_path, "w") as f:
        f.write(default_config)

    return config_path
