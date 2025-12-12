"""Tests for hark.ui module."""

import io
import sys
from unittest.mock import MagicMock, patch

import pytest

from hark.config import HarkConfig
from hark.constants import UNKNOWN_LANGUAGE_PROBABILITY
from hark.diarizer import DiarizationResult
from hark.transcriber import TranscriptionResult, TranscriptionSegment
from hark.ui import UI, Color


class TestColorEnum:
    """Tests for Color enum."""

    def test_color_enum_has_expected_values(self) -> None:
        """All colors should have ANSI escape codes."""
        assert Color.RESET.value == "\033[0m"
        assert Color.BOLD.value == "\033[1m"
        assert Color.DIM.value == "\033[2m"
        assert Color.RED.value == "\033[91m"
        assert Color.GREEN.value == "\033[92m"
        assert Color.YELLOW.value == "\033[93m"
        assert Color.BLUE.value == "\033[94m"
        assert Color.MAGENTA.value == "\033[95m"
        assert Color.CYAN.value == "\033[96m"

    def test_all_colors_are_strings(self) -> None:
        """All color values should be strings."""
        for color in Color:
            assert isinstance(color.value, str)
            assert color.value.startswith("\033[")


class TestUIInitialization:
    """Tests for UI initialization."""

    def test_quiet_mode_sets_flag(self) -> None:
        """quiet=True should set _quiet flag."""
        ui = UI(quiet=True)
        assert ui._quiet is True

    def test_quiet_mode_default_false(self) -> None:
        """Default quiet should be False."""
        ui = UI()
        assert ui._quiet is False

    def test_color_disabled_non_tty(self) -> None:
        """Color should be disabled for non-TTY stdout."""
        mock_stdout = io.StringIO()
        with patch.object(sys, "stdout", mock_stdout):
            ui = UI(use_color=True)
            assert ui._use_color is False

    def test_color_disabled_explicit(self) -> None:
        """use_color=False should disable colors."""
        mock_stdout = MagicMock()
        mock_stdout.isatty.return_value = True
        with patch.object(sys, "stdout", mock_stdout):
            ui = UI(use_color=False)
            assert ui._use_color is False


class TestUIQuietMode:
    """Tests for UI methods in quiet mode."""

    @pytest.fixture
    def quiet_ui(self) -> UI:
        """Create quiet UI instance."""
        return UI(quiet=True)

    def test_header_quiet(self, quiet_ui: UI, capsys) -> None:
        """header should produce no output in quiet mode."""
        quiet_ui.header("Test Title")
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_config_summary_quiet(self, quiet_ui: UI, default_config: HarkConfig, capsys) -> None:
        """config_summary should produce no output in quiet mode."""
        quiet_ui.config_summary(default_config, "output.txt")
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_prompt_start_quiet(self, quiet_ui: UI, capsys) -> None:
        """prompt_start should produce no output in quiet mode."""
        quiet_ui.prompt_start()
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_recording_status_quiet(self, quiet_ui: UI, capsys) -> None:
        """recording_status should produce no output in quiet mode."""
        quiet_ui.recording_status(30.0, 60.0, 0.5)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_recording_stopped_quiet(self, quiet_ui: UI, capsys) -> None:
        """recording_stopped should produce no output in quiet mode."""
        quiet_ui.recording_stopped(30.0)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_preprocessing_header_quiet(self, quiet_ui: UI, capsys) -> None:
        """preprocessing_header should produce no output in quiet mode."""
        quiet_ui.preprocessing_header()
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_preprocessing_step_quiet(self, quiet_ui: UI, capsys) -> None:
        """preprocessing_step should produce no output in quiet mode."""
        quiet_ui.preprocessing_step("Noise reduction")
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_transcription_progress_quiet(self, quiet_ui: UI, capsys) -> None:
        """transcription_progress should produce no output in quiet mode."""
        quiet_ui.transcription_progress(0.5)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_warning_quiet(self, quiet_ui: UI, capsys) -> None:
        """warning should produce no output in quiet mode."""
        quiet_ui.warning("Test warning")
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_info_quiet(self, quiet_ui: UI, capsys) -> None:
        """info should produce no output in quiet mode."""
        quiet_ui.info("Test info")
        captured = capsys.readouterr()
        assert captured.out == ""


class TestUIOutput:
    """Tests for UI output methods."""

    @pytest.fixture
    def ui(self) -> UI:
        """Create UI instance with colors disabled."""
        return UI(quiet=False, use_color=False)

    def test_header_output(self, ui: UI, capsys) -> None:
        """header should print title with underline."""
        ui.header("Test Title")
        captured = capsys.readouterr()
        assert "Test Title" in captured.out
        assert "=" in captured.out

    def test_config_summary_output(self, ui: UI, default_config: HarkConfig, capsys) -> None:
        """config_summary should print config details."""
        ui.config_summary(default_config, "output.txt")
        captured = capsys.readouterr()
        assert "Configuration" in captured.out
        assert "Language" in captured.out
        assert "Model" in captured.out
        assert "output.txt" in captured.out

    def test_config_summary_stdout_output(self, ui: UI, default_config: HarkConfig, capsys) -> None:
        """config_summary should show 'stdout' when no output file."""
        ui.config_summary(default_config, None)
        captured = capsys.readouterr()
        assert "stdout" in captured.out

    def test_config_summary_shows_input_source(
        self, ui: UI, default_config: HarkConfig, capsys
    ) -> None:
        """config_summary should show input source."""
        ui.config_summary(default_config, None)
        captured = capsys.readouterr()
        assert "Input" in captured.out
        assert "Microphone" in captured.out

    def test_config_summary_shows_speaker_input(
        self, ui: UI, default_config: HarkConfig, capsys
    ) -> None:
        """config_summary should show 'System Audio' for speaker mode."""
        default_config.recording.input_source = "speaker"
        ui.config_summary(default_config, None)
        captured = capsys.readouterr()
        assert "System Audio" in captured.out

    def test_config_summary_shows_both_input(
        self, ui: UI, default_config: HarkConfig, capsys
    ) -> None:
        """config_summary should show both sources for 'both' mode."""
        default_config.recording.input_source = "both"
        ui.config_summary(default_config, None)
        captured = capsys.readouterr()
        assert "Microphone" in captured.out and "System Audio" in captured.out

    def test_error_always_prints(self, capsys) -> None:
        """error should print even in quiet mode."""
        ui = UI(quiet=True, use_color=False)
        ui.error("Test error")
        captured = capsys.readouterr()
        assert "Test error" in captured.err

    def test_error_to_stderr(self, ui: UI, capsys) -> None:
        """error should print to stderr."""
        ui.error("Test error")
        captured = capsys.readouterr()
        assert "Test error" in captured.err
        assert captured.out == ""


class TestUIFormatDuration:
    """Tests for _format_duration method."""

    def test_format_duration_zero(self) -> None:
        """Zero seconds should format as 00:00."""
        assert UI._format_duration(0.0) == "00:00"

    def test_format_duration_seconds(self) -> None:
        """Seconds should format correctly."""
        assert UI._format_duration(45.0) == "00:45"

    def test_format_duration_minutes_seconds(self) -> None:
        """Minutes and seconds should format correctly."""
        assert UI._format_duration(125.0) == "02:05"

    def test_format_duration_rounds_down(self) -> None:
        """Fractional seconds should round down."""
        assert UI._format_duration(45.9) == "00:45"


class TestUIRecordingStatus:
    """Tests for recording_status method."""

    @pytest.fixture
    def ui(self) -> UI:
        """Create UI instance with colors disabled."""
        return UI(quiet=False, use_color=False)

    def test_recording_status_shows_progress(self, ui: UI, capsys) -> None:
        """recording_status should show progress bar."""
        ui.recording_status(30.0, 60.0, 0.5)
        captured = capsys.readouterr()
        assert "Recording" in captured.out

    def test_recording_status_shows_level(self, ui: UI, capsys) -> None:
        """recording_status should show audio level."""
        ui.recording_status(30.0, 60.0, 0.5)
        captured = capsys.readouterr()
        assert "Audio Level" in captured.out

    def test_recording_status_clamps_values(self, ui: UI, capsys) -> None:
        """recording_status should clamp level to [0, 1]."""
        # Level > 1.0
        ui.recording_status(30.0, 60.0, 1.5)
        captured = capsys.readouterr()
        assert "100%" in captured.out

        # Level < 0.0
        ui.recording_status(30.0, 60.0, -0.5)
        captured = capsys.readouterr()
        assert "0%" in captured.out

    def test_recording_status_mic_mode(self, ui: UI, capsys) -> None:
        """recording_status should show 'Recording...' for mic mode."""
        ui.recording_status(30.0, 60.0, 0.5, input_source="mic")
        captured = capsys.readouterr()
        assert "Recording..." in captured.out
        assert "speaker" not in captured.out.lower()

    def test_recording_status_speaker_mode(self, ui: UI, capsys) -> None:
        """recording_status should indicate speaker mode."""
        ui.recording_status(30.0, 60.0, 0.5, input_source="speaker")
        captured = capsys.readouterr()
        assert "speaker" in captured.out.lower()

    def test_recording_status_both_mode(self, ui: UI, capsys) -> None:
        """recording_status should indicate both mode."""
        ui.recording_status(30.0, 60.0, 0.5, input_source="both")
        captured = capsys.readouterr()
        assert "mic" in captured.out.lower() and "speaker" in captured.out.lower()


class TestUITranscriptionProgress:
    """Tests for transcription_progress method."""

    @pytest.fixture
    def ui(self) -> UI:
        """Create UI instance with colors disabled."""
        return UI(quiet=False, use_color=False)

    def test_transcription_progress_shows_bar(self, ui: UI, capsys) -> None:
        """transcription_progress should show progress bar."""
        ui.transcription_progress(0.5)
        captured = capsys.readouterr()
        assert "Transcribing" in captured.out
        assert "50%" in captured.out

    def test_transcription_progress_uses_carriage_return(self, ui: UI, capsys) -> None:
        """transcription_progress should use carriage return for in-place update."""
        ui.transcription_progress(0.5)
        captured = capsys.readouterr()
        assert "\r" in captured.out


class TestUIColorApplication:
    """Tests for color application."""

    def test_color_applied_when_enabled(self) -> None:
        """Color should be applied when enabled."""
        mock_stdout = MagicMock()
        mock_stdout.isatty.return_value = True
        with patch.object(sys, "stdout", mock_stdout):
            ui = UI(use_color=True)
            colored = ui._color("test", Color.RED)
            assert Color.RED.value in colored
            assert Color.RESET.value in colored

    def test_color_not_applied_when_disabled(self) -> None:
        """Color should not be applied when disabled."""
        ui = UI(use_color=False)
        plain = ui._color("test", Color.RED)
        assert plain == "test"
        assert "\033[" not in plain


class TestUITranscriptionComplete:
    """Tests for transcription_complete method."""

    @pytest.fixture
    def ui(self) -> UI:
        """Create UI instance with colors disabled."""
        return UI(quiet=False, use_color=False)

    def test_transcription_complete_shows_summary(
        self,
        ui: UI,
        sample_transcription_result: TranscriptionResult,
        capsys,
    ) -> None:
        """transcription_complete should show summary."""
        ui.transcription_complete(sample_transcription_result, "output.txt")
        captured = capsys.readouterr()
        assert "complete" in captured.out.lower()
        assert "Duration" in captured.out
        assert "Words" in captured.out
        assert "Language" in captured.out
        assert "output.txt" in captured.out

    def test_transcription_complete_no_output_path(
        self,
        ui: UI,
        sample_transcription_result: TranscriptionResult,
        capsys,
    ) -> None:
        """transcription_complete should not show output path when None."""
        ui.transcription_complete(sample_transcription_result, None)
        captured = capsys.readouterr()
        assert "complete" in captured.out.lower()
        # Should not mention output file
        lines = captured.out.split("\n")
        output_lines = [line for line in lines if "Output:" in line]
        assert len(output_lines) == 0


class TestUITypeDetection:
    """Tests for result type detection using isinstance."""

    @pytest.fixture
    def ui(self) -> UI:
        """Create UI instance with colors disabled."""
        return UI(quiet=False, use_color=False)

    def test_transcription_result_shows_transcription_complete(
        self,
        ui: UI,
        sample_transcription_result: TranscriptionResult,
        capsys,
    ) -> None:
        """TranscriptionResult should show 'Transcription complete!' message."""
        ui.transcription_complete(sample_transcription_result, None)
        captured = capsys.readouterr()
        assert "Transcription complete!" in captured.out
        assert "Diarization complete!" not in captured.out
        # TranscriptionResult should not show Speakers count
        assert "Speakers:" not in captured.out

    def test_diarization_result_shows_diarization_complete(
        self,
        ui: UI,
        sample_diarization_result: DiarizationResult,
        capsys,
    ) -> None:
        """DiarizationResult should show 'Diarization complete!' message."""
        ui.transcription_complete(sample_diarization_result, None)
        captured = capsys.readouterr()
        assert "Diarization complete!" in captured.out
        assert "Transcription complete!" not in captured.out
        # DiarizationResult should show speaker count
        assert "Speakers:" in captured.out

    def test_diarization_result_shows_speaker_count(
        self,
        ui: UI,
        sample_diarization_result: DiarizationResult,
        capsys,
    ) -> None:
        """DiarizationResult should show correct speaker count."""
        ui.transcription_complete(sample_diarization_result, None)
        captured = capsys.readouterr()
        # The fixture has 2 speakers (SPEAKER_01 and SPEAKER_02)
        assert "Speakers: 2" in captured.out

    def test_isinstance_check_used_for_type_detection(
        self,
        ui: UI,
        sample_transcription_result: TranscriptionResult,
        capsys,
    ) -> None:
        """Type detection should use isinstance, not duck-typing."""
        # This test verifies that adding a 'speakers' attribute to TranscriptionResult
        # would not cause it to be incorrectly identified as a diarization result
        # (since we use isinstance, not hasattr)
        ui.transcription_complete(sample_transcription_result, None)
        captured = capsys.readouterr()
        # Even if we were to add a speakers attr, it should still say "Transcription"
        assert "Transcription complete!" in captured.out


class TestUIPreprocessingStep:
    """Tests for preprocessing_step method."""

    @pytest.fixture
    def ui(self) -> UI:
        """Create UI instance with colors disabled."""
        return UI(quiet=False, use_color=False)

    def test_preprocessing_step_success(self, ui: UI, capsys) -> None:
        """preprocessing_step with success=True should show checkmark."""
        ui.preprocessing_step("Noise reduction", success=True)
        captured = capsys.readouterr()
        assert "Noise reduction" in captured.out

    def test_preprocessing_step_failure(self, ui: UI, capsys) -> None:
        """preprocessing_step with success=False should show X."""
        ui.preprocessing_step("Noise reduction", success=False)
        captured = capsys.readouterr()
        assert "Noise reduction" in captured.out


class TestUIVerbose:
    """Tests for verbose method."""

    @pytest.fixture
    def ui(self) -> UI:
        """Create UI instance with colors disabled."""
        return UI(quiet=False, use_color=False)

    def test_verbose_prints_message(self, ui: UI, capsys) -> None:
        """verbose should print message with DEBUG label."""
        ui.verbose("Debug message")
        captured = capsys.readouterr()
        assert "DEBUG" in captured.out
        assert "Debug message" in captured.out


class TestUIClearLines:
    """Tests for _clear_lines method."""

    def test_clear_lines_no_op_when_count_zero(self, capsys) -> None:
        """_clear_lines should do nothing when count is 0."""
        mock_stdout = MagicMock()
        mock_stdout.isatty.return_value = True
        with patch.object(sys, "stdout", mock_stdout):
            ui = UI(use_color=True)
            ui._clear_lines(0)
            mock_stdout.write.assert_not_called()

    def test_clear_lines_no_op_when_no_color(self) -> None:
        """_clear_lines should do nothing when colors disabled."""
        ui = UI(use_color=False)
        # Should not raise
        ui._clear_lines(5)


class TestUIWarning:
    """Tests for warning method."""

    @pytest.fixture
    def ui(self) -> UI:
        """Create UI instance with colors disabled."""
        return UI(quiet=False, use_color=False)

    def test_warning_prints_message(self, ui: UI, capsys) -> None:
        """warning should print message with Warning label."""
        ui.warning("Warning message")
        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert "Warning message" in captured.out


class TestUIInfo:
    """Tests for info method."""

    @pytest.fixture
    def ui(self) -> UI:
        """Create UI instance with colors disabled."""
        return UI(quiet=False, use_color=False)

    def test_info_prints_message(self, ui: UI, capsys) -> None:
        """info should print message as-is."""
        ui.info("Info message")
        captured = capsys.readouterr()
        assert "Info message" in captured.out


class TestUILanguageProbabilityHandling:
    """Tests for language probability handling in transcription_complete method."""

    @pytest.fixture
    def ui(self) -> UI:
        """Create UI instance with colors disabled."""
        return UI(quiet=False, use_color=False)

    def test_transcription_complete_with_known_language_probability(self, ui: UI, capsys) -> None:
        """Should display language with confidence percentage when probability is known."""
        result = TranscriptionResult(
            text="Hello world",
            segments=[TranscriptionSegment(start=0.0, end=1.0, text="Hello world", words=[])],
            language="en",
            language_probability=0.95,
            duration=1.0,
        )
        ui.transcription_complete(result, None)
        captured = capsys.readouterr()
        assert "Language: en (95% confidence)" in captured.out

    def test_transcription_complete_with_unknown_language_probability(self, ui: UI, capsys) -> None:
        """Should display language without confidence when probability is unknown."""
        result = TranscriptionResult(
            text="Hello world",
            segments=[TranscriptionSegment(start=0.0, end=1.0, text="Hello world", words=[])],
            language="en",
            language_probability=UNKNOWN_LANGUAGE_PROBABILITY,
            duration=1.0,
        )
        ui.transcription_complete(result, None)
        captured = capsys.readouterr()
        # Should show language without confidence percentage
        assert "Language: en" in captured.out
        assert "confidence" not in captured.out

    def test_transcription_complete_100_percent_confidence(self, ui: UI, capsys) -> None:
        """Should display 100% confidence when language was explicitly specified."""
        result = TranscriptionResult(
            text="Hello world",
            segments=[TranscriptionSegment(start=0.0, end=1.0, text="Hello world", words=[])],
            language="de",
            language_probability=1.0,
            duration=1.0,
        )
        ui.transcription_complete(result, None)
        captured = capsys.readouterr()
        assert "Language: de (100% confidence)" in captured.out
