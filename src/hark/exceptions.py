"""Custom exceptions for hark."""

__all__ = [
    "HarkError",
    "ConfigError",
    "ConfigNotFoundError",
    "ConfigValidationError",
    "AudioError",
    "NoMicrophoneError",
    "NoLoopbackDeviceError",
    "AudioDeviceBusyError",
    "RecordingTooShortError",
    "PreprocessingError",
    "TranscriptionError",
    "ModelNotFoundError",
    "ModelDownloadError",
    "InsufficientDiskSpaceError",
    "OutputError",
    "DiarizationError",
    "DependencyMissingError",
    "MissingTokenError",
    "GatedModelError",
]


class HarkError(Exception):
    """Base exception for hark."""

    pass


class ConfigError(HarkError):
    """Configuration-related errors."""

    pass


class ConfigNotFoundError(ConfigError):
    """Configuration file not found."""

    pass


class ConfigValidationError(ConfigError):
    """Configuration validation failed."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__(f"Configuration validation failed: {'; '.join(errors)}")


class AudioError(HarkError):
    """Audio recording/processing errors."""

    pass


class NoMicrophoneError(AudioError):
    """No microphone detected."""

    pass


class NoLoopbackDeviceError(AudioError):
    """No system audio loopback device found."""

    def __init__(self, message: str | None = None) -> None:
        default_msg = (
            "No system audio loopback device found.\n\n"
            "On Linux with PulseAudio, ensure a monitor source is available:\n"
            "  $ pactl list sources | grep -i monitor\n\n"
            "On PipeWire, monitor sources should be automatically available."
        )
        super().__init__(message or default_msg)


class AudioDeviceBusyError(AudioError):
    """Audio device is busy or unavailable."""

    pass


class RecordingTooShortError(AudioError):
    """Recording is too short to process."""

    pass


class PreprocessingError(HarkError):
    """Audio preprocessing errors."""

    pass


class TranscriptionError(HarkError):
    """Transcription-related errors."""

    pass


class ModelNotFoundError(TranscriptionError):
    """Whisper model not found or failed to load."""

    pass


class ModelDownloadError(TranscriptionError):
    """Failed to download Whisper model."""

    pass


class InsufficientDiskSpaceError(HarkError):
    """Insufficient disk space for operation."""

    def __init__(self, required_mb: float, available_mb: float) -> None:
        self.required_mb = required_mb
        self.available_mb = available_mb
        super().__init__(
            f"Insufficient disk space: need {required_mb:.0f}MB, have {available_mb:.0f}MB"
        )


class OutputError(HarkError):
    """Output-related errors."""

    pass


class DiarizationError(HarkError):
    """Speaker diarization errors."""

    pass


class DependencyMissingError(DiarizationError):
    """Required dependency for diarization is not installed."""

    def __init__(self, message: str | None = None) -> None:
        default_msg = (
            "Diarization requires additional dependencies.\n\n"
            "\033[1mInstall with:\033[0m\n"
            "  \033[93mpip install hark-cli[diarization]\033[0m"
        )
        super().__init__(message or default_msg)


class MissingTokenError(DiarizationError):
    """HuggingFace token is required but not configured."""

    def __init__(self, message: str | None = None) -> None:
        default_msg = (
            "Diarization requires a HuggingFace token.\n\n"
            "\033[1mSet up your token:\033[0m\n"
            "  1️⃣  Create account at \033[94mhttps://huggingface.co\033[0m\n"
            "  2️⃣  Accept model licenses:\n"
            "      • \033[94mhttps://huggingface.co/pyannote/segmentation-3.0\033[0m\n"
            "      • \033[94mhttps://huggingface.co/pyannote/speaker-diarization-3.1\033[0m\n"
            "  3️⃣  Create token at \033[94mhttps://huggingface.co/settings/tokens\033[0m\n"
            "  4️⃣  Add to \033[93m~/.config/hark/config.yaml\033[0m:\n"
            "      \033[2mdiarization:\n"
            "        hf_token: your_token_here\033[0m"
        )
        super().__init__(message or default_msg)


class GatedModelError(DiarizationError):
    """Pyannote model access not granted - user must accept license on HuggingFace."""

    def __init__(self, message: str | None = None) -> None:
        default_msg = (
            "Speaker diarization model access denied.\n\n"
            "The pyannote models are gated and require accepting the license terms.\n\n"
            "\033[1mTo fix this:\033[0m\n"
            "  1️⃣  Visit \033[94mhttps://huggingface.co/pyannote/speaker-diarization-3.1\033[0m\n"
            "  2️⃣  Log in with your HuggingFace account\n"
            "  3️⃣  Click 'Agree and access repository' to accept the license\n"
            "  4️⃣  Also accept the segmentation model license:\n"
            "      \033[94mhttps://huggingface.co/pyannote/segmentation-3.0\033[0m\n\n"
            "Then retry your command. ✨"
        )
        super().__init__(message or default_msg)
