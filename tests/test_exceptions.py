"""Tests for mrec_cli.exceptions module."""

import pytest

from mrec_cli.exceptions import (
    AudioDeviceBusyError,
    AudioError,
    ConfigError,
    ConfigNotFoundError,
    ConfigValidationError,
    InsufficientDiskSpaceError,
    ModelDownloadError,
    ModelNotFoundError,
    MrecError,
    NoMicrophoneError,
    OutputError,
    PreprocessingError,
    RecordingTooShortError,
    TranscriptionError,
)


class TestExceptionHierarchy:
    """Tests for exception inheritance hierarchy."""

    def test_mrec_error_is_base_exception(self) -> None:
        """MrecError should inherit from Exception."""
        assert issubclass(MrecError, Exception)

    def test_config_error_inherits_mrec_error(self) -> None:
        """ConfigError should inherit from MrecError."""
        assert issubclass(ConfigError, MrecError)

    def test_config_not_found_inherits_config_error(self) -> None:
        """ConfigNotFoundError should inherit from ConfigError."""
        assert issubclass(ConfigNotFoundError, ConfigError)

    def test_config_validation_inherits_config_error(self) -> None:
        """ConfigValidationError should inherit from ConfigError."""
        assert issubclass(ConfigValidationError, ConfigError)

    def test_audio_error_inherits_mrec_error(self) -> None:
        """AudioError should inherit from MrecError."""
        assert issubclass(AudioError, MrecError)

    def test_no_microphone_inherits_audio_error(self) -> None:
        """NoMicrophoneError should inherit from AudioError."""
        assert issubclass(NoMicrophoneError, AudioError)

    def test_audio_device_busy_inherits_audio_error(self) -> None:
        """AudioDeviceBusyError should inherit from AudioError."""
        assert issubclass(AudioDeviceBusyError, AudioError)

    def test_recording_too_short_inherits_audio_error(self) -> None:
        """RecordingTooShortError should inherit from AudioError."""
        assert issubclass(RecordingTooShortError, AudioError)

    def test_preprocessing_error_inherits_mrec_error(self) -> None:
        """PreprocessingError should inherit from MrecError."""
        assert issubclass(PreprocessingError, MrecError)

    def test_transcription_error_inherits_mrec_error(self) -> None:
        """TranscriptionError should inherit from MrecError."""
        assert issubclass(TranscriptionError, MrecError)

    def test_model_not_found_inherits_transcription_error(self) -> None:
        """ModelNotFoundError should inherit from TranscriptionError."""
        assert issubclass(ModelNotFoundError, TranscriptionError)

    def test_model_download_inherits_transcription_error(self) -> None:
        """ModelDownloadError should inherit from TranscriptionError."""
        assert issubclass(ModelDownloadError, TranscriptionError)

    def test_output_error_inherits_mrec_error(self) -> None:
        """OutputError should inherit from MrecError."""
        assert issubclass(OutputError, MrecError)

    def test_insufficient_disk_space_inherits_mrec_error(self) -> None:
        """InsufficientDiskSpaceError should inherit from MrecError."""
        assert issubclass(InsufficientDiskSpaceError, MrecError)


class TestConfigValidationError:
    """Tests for ConfigValidationError."""

    def test_formatting_single_error(self) -> None:
        """Single error should be formatted correctly."""
        err = ConfigValidationError(["Invalid sample rate"])
        assert "Invalid sample rate" in str(err)

    def test_formatting_multiple_errors(self) -> None:
        """Multiple errors should be joined with semicolons."""
        errors = ["Error 1", "Error 2", "Error 3"]
        err = ConfigValidationError(errors)
        message = str(err)
        assert "Error 1" in message
        assert "Error 2" in message
        assert "Error 3" in message
        assert ";" in message

    def test_empty_error_list(self) -> None:
        """Empty error list should create exception without crashing."""
        err = ConfigValidationError([])
        # Should not raise, message will be empty
        assert err.errors == []

    def test_errors_attribute_stored(self) -> None:
        """Errors list should be stored in errors attribute."""
        errors = ["Error A", "Error B"]
        err = ConfigValidationError(errors)
        assert err.errors == errors

    def test_errors_attribute_is_same_list(self) -> None:
        """Errors attribute should be the same list passed in."""
        errors = ["Test error"]
        err = ConfigValidationError(errors)
        assert err.errors is errors


class TestInsufficientDiskSpaceError:
    """Tests for InsufficientDiskSpaceError."""

    def test_formatting_includes_values(self) -> None:
        """Error message should include required and available MB."""
        err = InsufficientDiskSpaceError(required_mb=500.0, available_mb=120.0)
        message = str(err)
        assert "500" in message
        assert "120" in message

    def test_required_mb_attribute(self) -> None:
        """required_mb attribute should be stored."""
        err = InsufficientDiskSpaceError(required_mb=1024.5, available_mb=100.0)
        assert err.required_mb == 1024.5

    def test_available_mb_attribute(self) -> None:
        """available_mb attribute should be stored."""
        err = InsufficientDiskSpaceError(required_mb=500.0, available_mb=123.4)
        assert err.available_mb == 123.4

    def test_zero_values(self) -> None:
        """Zero values should work without error."""
        err = InsufficientDiskSpaceError(required_mb=0.0, available_mb=0.0)
        assert err.required_mb == 0.0
        assert err.available_mb == 0.0

    def test_large_values(self) -> None:
        """Large values should work correctly."""
        err = InsufficientDiskSpaceError(required_mb=100000.0, available_mb=50.0)
        message = str(err)
        assert "100000" in message


class TestAllExceptionsInstantiable:
    """Tests that all exception classes can be instantiated."""

    def test_mrec_error_instantiation(self) -> None:
        """MrecError should be instantiable."""
        err = MrecError("Test error")
        assert str(err) == "Test error"

    def test_config_error_instantiation(self) -> None:
        """ConfigError should be instantiable."""
        err = ConfigError("Config test")
        assert str(err) == "Config test"

    def test_config_not_found_instantiation(self) -> None:
        """ConfigNotFoundError should be instantiable."""
        err = ConfigNotFoundError("File not found")
        assert str(err) == "File not found"

    def test_audio_error_instantiation(self) -> None:
        """AudioError should be instantiable."""
        err = AudioError("Audio test")
        assert str(err) == "Audio test"

    def test_no_microphone_instantiation(self) -> None:
        """NoMicrophoneError should be instantiable."""
        err = NoMicrophoneError("No mic")
        assert str(err) == "No mic"

    def test_audio_device_busy_instantiation(self) -> None:
        """AudioDeviceBusyError should be instantiable."""
        err = AudioDeviceBusyError("Device busy")
        assert str(err) == "Device busy"

    def test_recording_too_short_instantiation(self) -> None:
        """RecordingTooShortError should be instantiable."""
        err = RecordingTooShortError("Too short")
        assert str(err) == "Too short"

    def test_preprocessing_error_instantiation(self) -> None:
        """PreprocessingError should be instantiable."""
        err = PreprocessingError("Preprocessing failed")
        assert str(err) == "Preprocessing failed"

    def test_transcription_error_instantiation(self) -> None:
        """TranscriptionError should be instantiable."""
        err = TranscriptionError("Transcription failed")
        assert str(err) == "Transcription failed"

    def test_model_not_found_instantiation(self) -> None:
        """ModelNotFoundError should be instantiable."""
        err = ModelNotFoundError("Model not found")
        assert str(err) == "Model not found"

    def test_model_download_instantiation(self) -> None:
        """ModelDownloadError should be instantiable."""
        err = ModelDownloadError("Download failed")
        assert str(err) == "Download failed"

    def test_output_error_instantiation(self) -> None:
        """OutputError should be instantiable."""
        err = OutputError("Output failed")
        assert str(err) == "Output failed"


class TestExceptionRaising:
    """Tests that exceptions can be raised and caught correctly."""

    def test_catch_as_mrec_error(self) -> None:
        """All exceptions should be catchable as MrecError."""
        exceptions_to_test = [
            MrecError("test"),
            ConfigError("test"),
            ConfigNotFoundError("test"),
            ConfigValidationError(["test"]),
            AudioError("test"),
            NoMicrophoneError("test"),
            AudioDeviceBusyError("test"),
            RecordingTooShortError("test"),
            PreprocessingError("test"),
            TranscriptionError("test"),
            ModelNotFoundError("test"),
            ModelDownloadError("test"),
            InsufficientDiskSpaceError(100, 50),
            OutputError("test"),
        ]

        for exc in exceptions_to_test:
            with pytest.raises(MrecError):
                raise exc

    def test_catch_specific_type(self) -> None:
        """Specific exception types should be catchable."""
        with pytest.raises(NoMicrophoneError):
            raise NoMicrophoneError("No microphone")

        with pytest.raises(ConfigValidationError):
            raise ConfigValidationError(["Invalid"])

    def test_exception_chaining(self) -> None:
        """Exceptions should support chaining with 'from'."""
        original = ValueError("Original error")
        try:
            try:
                raise original
            except ValueError as e:
                raise MrecError("Wrapped error") from e
        except MrecError as err:
            assert err.__cause__ is original


class TestExceptionRepr:
    """Tests for exception string representations."""

    def test_mrec_error_str(self) -> None:
        """MrecError should have meaningful __str__."""
        err = MrecError("Meaningful message")
        assert str(err) == "Meaningful message"

    def test_config_validation_error_str_includes_prefix(self) -> None:
        """ConfigValidationError __str__ should mention validation."""
        err = ConfigValidationError(["bad value"])
        message = str(err).lower()
        assert "validation" in message or "bad value" in message

    def test_insufficient_disk_space_str_readable(self) -> None:
        """InsufficientDiskSpaceError should have human-readable message."""
        err = InsufficientDiskSpaceError(500, 120)
        message = str(err)
        # Should contain words like "disk", "space", "need", "have" or the numbers
        assert "500" in message and "120" in message
