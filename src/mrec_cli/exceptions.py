"""Custom exceptions for mrec-cli."""

__all__ = [
    "MrecError",
    "ConfigError",
    "ConfigNotFoundError",
    "ConfigValidationError",
    "AudioError",
    "NoMicrophoneError",
    "AudioDeviceBusyError",
    "RecordingTooShortError",
    "PreprocessingError",
    "TranscriptionError",
    "ModelNotFoundError",
    "ModelDownloadError",
    "InsufficientDiskSpaceError",
    "OutputError",
]


class MrecError(Exception):
    """Base exception for mrec-cli."""

    pass


class ConfigError(MrecError):
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


class AudioError(MrecError):
    """Audio recording/processing errors."""

    pass


class NoMicrophoneError(AudioError):
    """No microphone detected."""

    pass


class AudioDeviceBusyError(AudioError):
    """Audio device is busy or unavailable."""

    pass


class RecordingTooShortError(AudioError):
    """Recording is too short to process."""

    pass


class PreprocessingError(MrecError):
    """Audio preprocessing errors."""

    pass


class TranscriptionError(MrecError):
    """Transcription-related errors."""

    pass


class ModelNotFoundError(TranscriptionError):
    """Whisper model not found or failed to load."""

    pass


class ModelDownloadError(TranscriptionError):
    """Failed to download Whisper model."""

    pass


class InsufficientDiskSpaceError(MrecError):
    """Insufficient disk space for operation."""

    def __init__(self, required_mb: float, available_mb: float) -> None:
        self.required_mb = required_mb
        self.available_mb = available_mb
        super().__init__(
            f"Insufficient disk space: need {required_mb:.0f}MB, have {available_mb:.0f}MB"
        )


class OutputError(MrecError):
    """Output-related errors."""

    pass
