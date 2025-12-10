"""Default constants and configuration values."""

from pathlib import Path

__all__ = [
    "DEFAULT_SAMPLE_RATE",
    "DEFAULT_CHANNELS",
    "DEFAULT_BIT_DEPTH",
    "DEFAULT_MAX_DURATION",
    "DEFAULT_BUFFER_SIZE",
    "DEFAULT_INPUT_SOURCE",
    "VALID_INPUT_SOURCES",
    "DEFAULT_MODEL",
    "DEFAULT_LANGUAGE",
    "VALID_MODELS",
    "DEFAULT_NOISE_STRENGTH",
    "DEFAULT_TARGET_LEVEL_DB",
    "DEFAULT_SILENCE_THRESHOLD_DB",
    "DEFAULT_MIN_SILENCE_DURATION",
    "DEFAULT_OUTPUT_FORMAT",
    "VALID_OUTPUT_FORMATS",
    "DEFAULT_ENCODING",
    "DEFAULT_CONFIG_DIR",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_CACHE_DIR",
    "DEFAULT_MODEL_CACHE_DIR",
    "DEFAULT_TEMP_DIR",
    "DEFAULT_SPEAKERS_DIR",
    "DEFAULT_DIARIZATION_MODEL",
    "EXIT_SUCCESS",
    "EXIT_ERROR",
    "EXIT_INTERRUPT",
    "MIN_RECORDING_DURATION",
]

# Audio recording defaults
DEFAULT_SAMPLE_RATE = 16000  # Whisper's expected sample rate
DEFAULT_CHANNELS = 1
DEFAULT_BIT_DEPTH = 16
DEFAULT_MAX_DURATION = 600  # 10 minutes
DEFAULT_BUFFER_SIZE = 4096
DEFAULT_INPUT_SOURCE = "mic"
VALID_INPUT_SOURCES = ["mic", "speaker", "both"]

# Whisper defaults
DEFAULT_MODEL = "base"
DEFAULT_LANGUAGE = "auto"
VALID_MODELS = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]

# Preprocessing defaults
DEFAULT_NOISE_STRENGTH = 0.5
DEFAULT_TARGET_LEVEL_DB = -20.0
DEFAULT_SILENCE_THRESHOLD_DB = -40.0
DEFAULT_MIN_SILENCE_DURATION = 0.5

# Output defaults
DEFAULT_OUTPUT_FORMAT = "plain"
VALID_OUTPUT_FORMATS = ["plain", "markdown", "srt"]
DEFAULT_ENCODING = "utf-8"

# Paths
DEFAULT_CONFIG_DIR = Path.home() / ".config" / "hark"
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_DIR / "config.yaml"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "hark"
DEFAULT_MODEL_CACHE_DIR = DEFAULT_CACHE_DIR / "models"
DEFAULT_TEMP_DIR = Path("/tmp/hark")
DEFAULT_SPEAKERS_DIR = DEFAULT_CONFIG_DIR / "speakers"

# Diarization defaults
DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"

# Exit codes
EXIT_SUCCESS = 0
EXIT_ERROR = 1
EXIT_INTERRUPT = 130  # Standard exit code for Ctrl+C

# Minimum recording duration (seconds)
MIN_RECORDING_DURATION = 0.5
