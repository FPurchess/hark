"""Tests for hark.cli module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from hark.cli import create_parser, main, run_workflow
from hark.config import HarkConfig
from hark.constants import (
    EXIT_ERROR,
    EXIT_INTERRUPT,
    EXIT_SUCCESS,
)
from hark.exceptions import HarkError


class TestCreateParser:
    """Tests for create_parser function."""

    def test_positional_output_file_optional(self) -> None:
        """Output file should be optional positional argument."""
        parser = create_parser()

        # Without output file
        args = parser.parse_args([])
        assert args.output_file is None

        # With output file
        args = parser.parse_args(["output.txt"])
        assert args.output_file == "output.txt"

    def test_recording_options(self) -> None:
        """Recording options should be parsed correctly."""
        parser = create_parser()
        args = parser.parse_args(
            [
                "--max-duration",
                "300",
                "--sample-rate",
                "48000",
                "--channels",
                "2",
            ]
        )

        assert args.max_duration == 300
        assert args.sample_rate == 48000
        assert args.channels == 2

    def test_input_source_option(self) -> None:
        """--input option should be parsed correctly."""
        parser = create_parser()

        # Test mic (default)
        args = parser.parse_args(["--input", "mic"])
        assert args.input_source == "mic"

        # Test speaker
        args = parser.parse_args(["--input", "speaker"])
        assert args.input_source == "speaker"

        # Test both
        args = parser.parse_args(["--input", "both"])
        assert args.input_source == "both"

    def test_input_source_invalid_rejected(self) -> None:
        """Invalid --input value should be rejected."""
        parser = create_parser()

        with pytest.raises(SystemExit):
            parser.parse_args(["--input", "invalid"])

    def test_input_source_default_none(self) -> None:
        """--input should default to None (not specified)."""
        parser = create_parser()
        args = parser.parse_args([])
        assert args.input_source is None

    def test_language_options(self) -> None:
        """Language options should be parsed correctly."""
        parser = create_parser()
        args = parser.parse_args(["--lang", "de", "--model", "large-v3"])

        assert args.lang == "de"
        assert args.model == "large-v3"

    def test_model_choices(self) -> None:
        """Invalid model should be rejected."""
        parser = create_parser()

        with pytest.raises(SystemExit):
            parser.parse_args(["--model", "invalid_model"])

    def test_preprocessing_options(self) -> None:
        """Preprocessing options should be parsed correctly."""
        parser = create_parser()
        args = parser.parse_args(
            [
                "--no-noise-reduction",
                "--no-normalize",
                "--no-trim-silence",
                "--noise-strength",
                "0.7",
            ]
        )

        assert args.no_noise_reduction is True
        assert args.no_normalize is True
        assert args.no_trim_silence is True
        assert args.noise_strength == 0.7

    def test_output_options(self) -> None:
        """Output options should be parsed correctly."""
        parser = create_parser()
        args = parser.parse_args(
            [
                "--timestamps",
                "--format",
                "srt",
                "--append",
            ]
        )

        assert args.timestamps is True
        assert args.format == "srt"
        assert args.append is True

    def test_format_choices(self) -> None:
        """Invalid format should be rejected."""
        parser = create_parser()

        with pytest.raises(SystemExit):
            parser.parse_args(["--format", "invalid_format"])

    def test_interface_options(self) -> None:
        """Interface options should be parsed correctly."""
        parser = create_parser()

        args = parser.parse_args(["-q", "-v", "--config", "/path/to/config.yaml"])
        assert args.quiet is True
        assert args.verbose is True
        assert args.config == "/path/to/config.yaml"

    def test_version_flag(self, capsys) -> None:
        """--version should show version and exit."""
        parser = create_parser()

        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--version"])
        assert exc_info.value.code == 0

    def test_help_flag(self, capsys) -> None:
        """--help should show help and exit."""
        parser = create_parser()

        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--help"])
        assert exc_info.value.code == 0


class TestMain:
    """Tests for main function."""

    def test_no_args_uses_defaults(self) -> None:
        """main() with no args should use default config."""
        with patch("hark.cli.run_workflow", return_value=EXIT_SUCCESS) as mock_run:
            with patch("hark.cli.load_config") as mock_load:
                mock_load.return_value = HarkConfig()
                with patch("hark.cli.merge_cli_args", return_value=HarkConfig()):
                    with patch("hark.cli.validate_config", return_value=[]):
                        with patch("hark.cli.ensure_directories"):
                            result = main([])

                            assert result == EXIT_SUCCESS
                            mock_run.assert_called_once()

    def test_with_output_file(self) -> None:
        """main() with output file should pass it to workflow."""
        with patch("hark.cli.run_workflow", return_value=EXIT_SUCCESS) as mock_run:
            with patch("hark.cli.load_config", return_value=HarkConfig()):
                with patch("hark.cli.merge_cli_args", return_value=HarkConfig()):
                    with patch("hark.cli.validate_config", return_value=[]):
                        with patch("hark.cli.ensure_directories"):
                            main(["output.txt"])

                            call_args = mock_run.call_args
                            assert call_args[0][1] == "output.txt"

    def test_custom_config(self) -> None:
        """--config should load custom config file."""
        with patch("hark.cli.run_workflow", return_value=EXIT_SUCCESS):
            with patch("hark.cli.load_config") as mock_load:
                mock_load.return_value = HarkConfig()
                with patch("hark.cli.merge_cli_args", return_value=HarkConfig()):
                    with patch("hark.cli.validate_config", return_value=[]):
                        with patch("hark.cli.ensure_directories"):
                            main(["--config", "/custom/config.yaml"])

                            mock_load.assert_called_once()
                            call_path = mock_load.call_args[0][0]
                            assert call_path == Path("/custom/config.yaml")

    def test_config_validation_error(self, capsys) -> None:
        """Validation errors should print errors and return EXIT_ERROR."""
        with patch("hark.cli.load_config", return_value=HarkConfig()):
            with patch("hark.cli.merge_cli_args", return_value=HarkConfig()):
                with patch("hark.cli.validate_config", return_value=["Error 1", "Error 2"]):
                    result = main([])

                    assert result == EXIT_ERROR
                    captured = capsys.readouterr()
                    assert "Error 1" in captured.err
                    assert "Error 2" in captured.err

    def test_keyboard_interrupt(self) -> None:
        """KeyboardInterrupt should return EXIT_INTERRUPT."""
        with patch("hark.cli.load_config", side_effect=KeyboardInterrupt):
            result = main([])
            assert result == EXIT_INTERRUPT

    def test_hark_error(self, capsys) -> None:
        """HarkError should print error and return EXIT_ERROR."""
        with patch("hark.cli.load_config", side_effect=HarkError("Test error")):
            result = main([])

            assert result == EXIT_ERROR
            captured = capsys.readouterr()
            assert "Test error" in captured.err

    def test_unexpected_error(self, capsys) -> None:
        """Unexpected error should return EXIT_ERROR."""
        with patch("hark.cli.load_config", side_effect=RuntimeError("Unexpected")):
            result = main([])

            assert result == EXIT_ERROR
            captured = capsys.readouterr()
            assert "Unexpected" in captured.err

    def test_verbose_shows_traceback(self, capsys) -> None:
        """Verbose mode should show traceback on unexpected error."""
        with patch("hark.cli.load_config", side_effect=RuntimeError("Test error")):
            result = main(["-v"])

            assert result == EXIT_ERROR
            captured = capsys.readouterr()
            assert "Traceback" in captured.err


class TestRunWorkflow:
    """Tests for run_workflow function."""

    @pytest.fixture
    def mock_ui(self) -> MagicMock:
        """Create mock UI."""
        return MagicMock()

    @pytest.fixture
    def mock_recorder(self) -> MagicMock:
        """Create mock recorder."""
        mock = MagicMock()
        mock.is_recording = True
        mock.get_duration.return_value = 5.0
        mock.stop.return_value = Path("/tmp/test.wav")
        return mock

    @pytest.fixture
    def mock_preprocessor_result(self) -> MagicMock:
        """Create mock preprocessing result."""
        result = MagicMock()
        result.silence_trimmed_seconds = 0.5
        return result

    @pytest.fixture
    def mock_transcription_result(self) -> MagicMock:
        """Create mock transcription result."""
        result = MagicMock()
        result.text = "Hello world"
        result.language = "en"
        result.language_probability = 0.95
        result.duration = 5.0
        result.segments = []
        return result

    def test_displays_header(self, default_config: HarkConfig, mock_ui: MagicMock) -> None:
        """Should display header."""
        with patch("hark.cli.KeypressHandler") as mock_keys:
            mock_keys.return_value.__enter__.return_value.get_key.return_value = None
            # Make it raise KeyboardInterrupt to exit early
            mock_keys.return_value.__enter__.return_value.get_key.side_effect = KeyboardInterrupt

            run_workflow(default_config, None, mock_ui, verbose=False)

            mock_ui.header.assert_called()

    def test_displays_config_summary(self, default_config: HarkConfig, mock_ui: MagicMock) -> None:
        """Should display config summary."""
        with patch("hark.cli.KeypressHandler") as mock_keys:
            mock_keys.return_value.__enter__.return_value.get_key.side_effect = KeyboardInterrupt

            run_workflow(default_config, "output.txt", mock_ui, verbose=False)

            mock_ui.config_summary.assert_called_with(default_config, "output.txt")

    def test_waits_for_space_keypress(self, default_config: HarkConfig, mock_ui: MagicMock) -> None:
        """Should wait for space key to start recording."""
        call_count = [0]

        def side_effect(timeout=0.1):
            call_count[0] += 1
            if call_count[0] < 3:
                return None  # No key pressed
            return " "  # Space pressed

        with patch("hark.cli.KeypressHandler") as mock_keys:
            mock_handler = MagicMock()
            mock_handler.get_key.side_effect = side_effect
            mock_keys.return_value.__enter__.return_value = mock_handler

            with patch("hark.cli.AudioRecorder") as mock_recorder_cls:
                mock_recorder = MagicMock()
                mock_recorder.is_recording = False  # Stop immediately
                mock_recorder.get_duration.return_value = 0.1
                mock_recorder.stop.return_value = Path("/tmp/test.wav")
                mock_recorder_cls.return_value = mock_recorder

                run_workflow(default_config, None, mock_ui, verbose=False)

    def test_ctrl_c_during_wait_returns_interrupt(
        self, default_config: HarkConfig, mock_ui: MagicMock
    ) -> None:
        """Ctrl+C before recording should return EXIT_INTERRUPT."""
        with patch("hark.cli.KeypressHandler") as mock_keys:
            mock_handler = MagicMock()
            mock_handler.get_key.return_value = "\x03"  # Ctrl+C
            mock_keys.return_value.__enter__.return_value = mock_handler

            result = run_workflow(default_config, None, mock_ui, verbose=False)

            assert result == EXIT_INTERRUPT

    def test_starts_recording(self, default_config: HarkConfig, mock_ui: MagicMock) -> None:
        """Should start recording after space is pressed."""
        with patch("hark.cli.KeypressHandler") as mock_keys:
            mock_handler = MagicMock()
            mock_handler.get_key.side_effect = [" ", KeyboardInterrupt]
            mock_keys.return_value.__enter__.return_value = mock_handler

            with patch("hark.cli.AudioRecorder") as mock_recorder_cls:
                mock_recorder = MagicMock()
                mock_recorder.is_recording = False
                mock_recorder.get_duration.return_value = 0.1
                mock_recorder.stop.return_value = Path("/tmp/test.wav")
                mock_recorder_cls.return_value = mock_recorder

                run_workflow(default_config, None, mock_ui, verbose=False)

                mock_recorder.start.assert_called_once()

    def test_shows_recording_status(self, default_config: HarkConfig, mock_ui: MagicMock) -> None:
        """Should show recording status during recording."""
        call_count = [0]

        def is_recording_side_effect():
            call_count[0] += 1
            return call_count[0] < 3  # Record for 2 iterations

        with patch("hark.cli.KeypressHandler") as mock_keys:
            mock_handler = MagicMock()
            mock_handler.get_key.side_effect = [" "] + [None] * 10
            mock_keys.return_value.__enter__.return_value = mock_handler

            with patch("hark.cli.AudioRecorder") as mock_recorder_cls:
                mock_recorder = MagicMock()
                type(mock_recorder).is_recording = property(lambda self: is_recording_side_effect())
                mock_recorder.get_duration.return_value = 1.0
                mock_recorder.stop.return_value = Path("/tmp/test.wav")
                mock_recorder_cls.return_value = mock_recorder

                with patch("time.sleep"):
                    with patch("hark.cli.AudioPreprocessor") as mock_prep_cls:
                        mock_prep = MagicMock()
                        mock_result = MagicMock()
                        mock_result.silence_trimmed_seconds = 0.0
                        mock_prep.process.return_value = (np.zeros(1000), mock_result)
                        mock_prep_cls.return_value = mock_prep

                        with patch("hark.cli.Transcriber") as mock_trans_cls:
                            mock_trans = MagicMock()
                            mock_trans_result = MagicMock()
                            mock_trans_result.text = "Test"
                            mock_trans_result.language = "en"
                            mock_trans_result.language_probability = 0.9
                            mock_trans_result.duration = 1.0
                            mock_trans.transcribe.return_value = mock_trans_result
                            mock_trans_cls.return_value = mock_trans

                            with patch("hark.cli.get_formatter") as mock_fmt:
                                mock_fmt.return_value.format.return_value = "Test"

                                run_workflow(default_config, None, mock_ui, verbose=False)

                mock_ui.recording_status.assert_called()

    def test_recording_stopped_message(
        self, default_config: HarkConfig, mock_ui: MagicMock
    ) -> None:
        """Should show recording stopped message."""
        with patch("hark.cli.KeypressHandler") as mock_keys:
            mock_handler = MagicMock()
            mock_handler.get_key.side_effect = [" ", KeyboardInterrupt]
            mock_keys.return_value.__enter__.return_value = mock_handler

            with patch("hark.cli.AudioRecorder") as mock_recorder_cls:
                mock_recorder = MagicMock()
                mock_recorder.is_recording = False
                mock_recorder.get_duration.return_value = 5.0
                mock_recorder.stop.return_value = Path("/tmp/test.wav")
                mock_recorder_cls.return_value = mock_recorder

                with patch("hark.cli.AudioPreprocessor") as mock_prep_cls:
                    mock_prep = MagicMock()
                    mock_result = MagicMock()
                    mock_result.silence_trimmed_seconds = 0.0
                    mock_prep.process.return_value = (np.zeros(1000), mock_result)
                    mock_prep_cls.return_value = mock_prep

                    with patch("hark.cli.Transcriber") as mock_trans_cls:
                        mock_trans = MagicMock()
                        mock_trans_result = MagicMock()
                        mock_trans_result.text = "Test"
                        mock_trans_result.language = "en"
                        mock_trans_result.language_probability = 0.9
                        mock_trans_result.duration = 5.0
                        mock_trans.transcribe.return_value = mock_trans_result
                        mock_trans_cls.return_value = mock_trans

                        with patch("hark.cli.get_formatter") as mock_formatter:
                            mock_formatter.return_value.format.return_value = "Test"

                            run_workflow(default_config, None, mock_ui, verbose=False)

                mock_ui.recording_stopped.assert_called_with(5.0)

    def test_checks_min_duration(self, default_config: HarkConfig, mock_ui: MagicMock) -> None:
        """Should return EXIT_ERROR if recording too short."""
        with patch("hark.cli.KeypressHandler") as mock_keys:
            mock_handler = MagicMock()
            mock_handler.get_key.side_effect = [" ", KeyboardInterrupt]
            mock_keys.return_value.__enter__.return_value = mock_handler

            with patch("hark.cli.AudioRecorder") as mock_recorder_cls:
                mock_recorder = MagicMock()
                mock_recorder.is_recording = False
                mock_recorder.get_duration.return_value = 0.1  # Too short
                mock_recorder.stop.return_value = Path("/tmp/test.wav")
                mock_recorder_cls.return_value = mock_recorder

                result = run_workflow(default_config, None, mock_ui, verbose=False)

                assert result == EXIT_ERROR
                mock_ui.error.assert_called()

    def test_runs_preprocessor(self, default_config: HarkConfig, mock_ui: MagicMock) -> None:
        """Should run AudioPreprocessor.process."""
        with patch("hark.cli.KeypressHandler") as mock_keys:
            mock_handler = MagicMock()
            mock_handler.get_key.side_effect = [" ", KeyboardInterrupt]
            mock_keys.return_value.__enter__.return_value = mock_handler

            with patch("hark.cli.AudioRecorder") as mock_recorder_cls:
                mock_recorder = MagicMock()
                mock_recorder.is_recording = False
                mock_recorder.get_duration.return_value = 5.0
                mock_recorder.stop.return_value = Path("/tmp/test.wav")
                mock_recorder_cls.return_value = mock_recorder

                with patch("hark.cli.AudioPreprocessor") as mock_prep_cls:
                    mock_prep = MagicMock()
                    mock_result = MagicMock()
                    mock_result.silence_trimmed_seconds = 0.0
                    mock_prep.process.return_value = (np.zeros(1000), mock_result)
                    mock_prep_cls.return_value = mock_prep

                    with patch("hark.cli.Transcriber") as mock_trans_cls:
                        mock_trans = MagicMock()
                        mock_trans_result = MagicMock()
                        mock_trans_result.text = "Test"
                        mock_trans_result.language = "en"
                        mock_trans_result.language_probability = 0.9
                        mock_trans_result.duration = 5.0
                        mock_trans.transcribe.return_value = mock_trans_result
                        mock_trans_cls.return_value = mock_trans

                        with patch("hark.cli.get_formatter") as mock_fmt:
                            mock_fmt.return_value.format.return_value = "Test"

                            run_workflow(default_config, None, mock_ui, verbose=False)

                    mock_prep.process.assert_called_once()

    def test_preprocessing_header_shown(
        self, default_config: HarkConfig, mock_ui: MagicMock
    ) -> None:
        """Should show preprocessing header."""
        with patch("hark.cli.KeypressHandler") as mock_keys:
            mock_handler = MagicMock()
            mock_handler.get_key.side_effect = [" ", KeyboardInterrupt]
            mock_keys.return_value.__enter__.return_value = mock_handler

            with patch("hark.cli.AudioRecorder") as mock_recorder_cls:
                mock_recorder = MagicMock()
                mock_recorder.is_recording = False
                mock_recorder.get_duration.return_value = 5.0
                mock_recorder.stop.return_value = Path("/tmp/test.wav")
                mock_recorder_cls.return_value = mock_recorder

                with patch("hark.cli.AudioPreprocessor") as mock_prep_cls:
                    mock_prep = MagicMock()
                    mock_result = MagicMock()
                    mock_result.silence_trimmed_seconds = 0.0
                    mock_prep.process.return_value = (np.zeros(1000), mock_result)
                    mock_prep_cls.return_value = mock_prep

                    with patch("hark.cli.Transcriber") as mock_trans_cls:
                        mock_trans = MagicMock()
                        mock_trans_result = MagicMock()
                        mock_trans_result.text = "Test"
                        mock_trans_result.language = "en"
                        mock_trans_result.language_probability = 0.9
                        mock_trans_result.duration = 5.0
                        mock_trans.transcribe.return_value = mock_trans_result
                        mock_trans_cls.return_value = mock_trans

                        with patch("hark.cli.get_formatter") as mock_fmt:
                            mock_fmt.return_value.format.return_value = "Test"

                            run_workflow(default_config, None, mock_ui, verbose=False)

                mock_ui.preprocessing_header.assert_called()

    def test_loads_model(self, default_config: HarkConfig, mock_ui: MagicMock) -> None:
        """Should load transcriber model."""
        with patch("hark.cli.KeypressHandler") as mock_keys:
            mock_handler = MagicMock()
            mock_handler.get_key.side_effect = [" ", KeyboardInterrupt]
            mock_keys.return_value.__enter__.return_value = mock_handler

            with patch("hark.cli.AudioRecorder") as mock_recorder_cls:
                mock_recorder = MagicMock()
                mock_recorder.is_recording = False
                mock_recorder.get_duration.return_value = 5.0
                mock_recorder.stop.return_value = Path("/tmp/test.wav")
                mock_recorder_cls.return_value = mock_recorder

                with patch("hark.cli.AudioPreprocessor") as mock_prep_cls:
                    mock_prep = MagicMock()
                    mock_result = MagicMock()
                    mock_result.silence_trimmed_seconds = 0.0
                    mock_prep.process.return_value = (np.zeros(1000), mock_result)
                    mock_prep_cls.return_value = mock_prep

                    with patch("hark.cli.Transcriber") as mock_trans_cls:
                        mock_trans = MagicMock()
                        mock_trans_result = MagicMock()
                        mock_trans_result.text = "Test"
                        mock_trans_result.language = "en"
                        mock_trans_result.language_probability = 0.9
                        mock_trans_result.duration = 5.0
                        mock_trans.transcribe.return_value = mock_trans_result
                        mock_trans_cls.return_value = mock_trans

                        with patch("hark.cli.get_formatter") as mock_fmt:
                            mock_fmt.return_value.format.return_value = "Test"

                            run_workflow(default_config, None, mock_ui, verbose=False)

                        mock_trans.load_model.assert_called_once()

    def test_transcribes_audio(self, default_config: HarkConfig, mock_ui: MagicMock) -> None:
        """Should transcribe audio."""
        with patch("hark.cli.KeypressHandler") as mock_keys:
            mock_handler = MagicMock()
            mock_handler.get_key.side_effect = [" ", KeyboardInterrupt]
            mock_keys.return_value.__enter__.return_value = mock_handler

            with patch("hark.cli.AudioRecorder") as mock_recorder_cls:
                mock_recorder = MagicMock()
                mock_recorder.is_recording = False
                mock_recorder.get_duration.return_value = 5.0
                mock_recorder.stop.return_value = Path("/tmp/test.wav")
                mock_recorder_cls.return_value = mock_recorder

                with patch("hark.cli.AudioPreprocessor") as mock_prep_cls:
                    mock_prep = MagicMock()
                    mock_result = MagicMock()
                    mock_result.silence_trimmed_seconds = 0.0
                    processed_audio = np.zeros(1000)
                    mock_prep.process.return_value = (processed_audio, mock_result)
                    mock_prep_cls.return_value = mock_prep

                    with patch("hark.cli.Transcriber") as mock_trans_cls:
                        mock_trans = MagicMock()
                        mock_trans_result = MagicMock()
                        mock_trans_result.text = "Test"
                        mock_trans_result.language = "en"
                        mock_trans_result.language_probability = 0.9
                        mock_trans_result.duration = 5.0
                        mock_trans.transcribe.return_value = mock_trans_result
                        mock_trans_cls.return_value = mock_trans

                        with patch("hark.cli.get_formatter") as mock_fmt:
                            mock_fmt.return_value.format.return_value = "Test"

                            run_workflow(default_config, None, mock_ui, verbose=False)

                        mock_trans.transcribe.assert_called_once()
                        call_kwargs = mock_trans.transcribe.call_args[1]
                        assert "audio" in call_kwargs
                        assert "sample_rate" in call_kwargs

    def test_formats_output(self, default_config: HarkConfig, mock_ui: MagicMock) -> None:
        """Should format output using formatter."""
        with patch("hark.cli.KeypressHandler") as mock_keys:
            mock_handler = MagicMock()
            mock_handler.get_key.side_effect = [" ", KeyboardInterrupt]
            mock_keys.return_value.__enter__.return_value = mock_handler

            with patch("hark.cli.AudioRecorder") as mock_recorder_cls:
                mock_recorder = MagicMock()
                mock_recorder.is_recording = False
                mock_recorder.get_duration.return_value = 5.0
                mock_recorder.stop.return_value = Path("/tmp/test.wav")
                mock_recorder_cls.return_value = mock_recorder

                with patch("hark.cli.AudioPreprocessor") as mock_prep_cls:
                    mock_prep = MagicMock()
                    mock_result = MagicMock()
                    mock_result.silence_trimmed_seconds = 0.0
                    mock_prep.process.return_value = (np.zeros(1000), mock_result)
                    mock_prep_cls.return_value = mock_prep

                    with patch("hark.cli.Transcriber") as mock_trans_cls:
                        mock_trans = MagicMock()
                        mock_trans_result = MagicMock()
                        mock_trans_result.text = "Test"
                        mock_trans_result.language = "en"
                        mock_trans_result.language_probability = 0.9
                        mock_trans_result.duration = 5.0
                        mock_trans.transcribe.return_value = mock_trans_result
                        mock_trans_cls.return_value = mock_trans

                        with patch("hark.cli.get_formatter") as mock_fmt:
                            mock_formatter = MagicMock()
                            mock_formatter.format.return_value = "Formatted output"
                            mock_fmt.return_value = mock_formatter

                            run_workflow(default_config, None, mock_ui, verbose=False)

                            mock_fmt.assert_called_once()
                            mock_formatter.format.assert_called_with(mock_trans_result)

    def test_writes_to_file(
        self, default_config: HarkConfig, mock_ui: MagicMock, tmp_path: Path
    ) -> None:
        """Should write output to file."""
        output_file = tmp_path / "output.txt"

        with patch("hark.cli.KeypressHandler") as mock_keys:
            mock_handler = MagicMock()
            mock_handler.get_key.side_effect = [" ", KeyboardInterrupt]
            mock_keys.return_value.__enter__.return_value = mock_handler

            with patch("hark.cli.AudioRecorder") as mock_recorder_cls:
                mock_recorder = MagicMock()
                mock_recorder.is_recording = False
                mock_recorder.get_duration.return_value = 5.0
                mock_recorder.stop.return_value = Path("/tmp/test.wav")
                mock_recorder_cls.return_value = mock_recorder

                with patch("hark.cli.AudioPreprocessor") as mock_prep_cls:
                    mock_prep = MagicMock()
                    mock_result = MagicMock()
                    mock_result.silence_trimmed_seconds = 0.0
                    mock_prep.process.return_value = (np.zeros(1000), mock_result)
                    mock_prep_cls.return_value = mock_prep

                    with patch("hark.cli.Transcriber") as mock_trans_cls:
                        mock_trans = MagicMock()
                        mock_trans_result = MagicMock()
                        mock_trans_result.text = "Test transcription"
                        mock_trans_result.language = "en"
                        mock_trans_result.language_probability = 0.9
                        mock_trans_result.duration = 5.0
                        mock_trans.transcribe.return_value = mock_trans_result
                        mock_trans_cls.return_value = mock_trans

                        with patch("hark.cli.get_formatter") as mock_fmt:
                            mock_fmt.return_value.format.return_value = "Formatted text"

                            run_workflow(default_config, str(output_file), mock_ui, verbose=False)

        assert output_file.exists()
        content = output_file.read_text()
        assert "Formatted text" in content

    def test_writes_to_stdout(self, default_config: HarkConfig, mock_ui: MagicMock, capsys) -> None:
        """Should print output to stdout when no file."""
        with patch("hark.cli.KeypressHandler") as mock_keys:
            mock_handler = MagicMock()
            mock_handler.get_key.side_effect = [" ", KeyboardInterrupt]
            mock_keys.return_value.__enter__.return_value = mock_handler

            with patch("hark.cli.AudioRecorder") as mock_recorder_cls:
                mock_recorder = MagicMock()
                mock_recorder.is_recording = False
                mock_recorder.get_duration.return_value = 5.0
                mock_recorder.stop.return_value = Path("/tmp/test.wav")
                mock_recorder_cls.return_value = mock_recorder

                with patch("hark.cli.AudioPreprocessor") as mock_prep_cls:
                    mock_prep = MagicMock()
                    mock_result = MagicMock()
                    mock_result.silence_trimmed_seconds = 0.0
                    mock_prep.process.return_value = (np.zeros(1000), mock_result)
                    mock_prep_cls.return_value = mock_prep

                    with patch("hark.cli.Transcriber") as mock_trans_cls:
                        mock_trans = MagicMock()
                        mock_trans_result = MagicMock()
                        mock_trans_result.text = "Test"
                        mock_trans_result.language = "en"
                        mock_trans_result.language_probability = 0.9
                        mock_trans_result.duration = 5.0
                        mock_trans.transcribe.return_value = mock_trans_result
                        mock_trans_cls.return_value = mock_trans

                        with patch("hark.cli.get_formatter") as mock_fmt:
                            mock_fmt.return_value.format.return_value = "Stdout output"

                            run_workflow(default_config, None, mock_ui, verbose=False)

        captured = capsys.readouterr()
        assert "Stdout output" in captured.out

    def test_transcription_complete_message(
        self, default_config: HarkConfig, mock_ui: MagicMock
    ) -> None:
        """Should show transcription complete message."""
        with patch("hark.cli.KeypressHandler") as mock_keys:
            mock_handler = MagicMock()
            mock_handler.get_key.side_effect = [" ", KeyboardInterrupt]
            mock_keys.return_value.__enter__.return_value = mock_handler

            with patch("hark.cli.AudioRecorder") as mock_recorder_cls:
                mock_recorder = MagicMock()
                mock_recorder.is_recording = False
                mock_recorder.get_duration.return_value = 5.0
                mock_recorder.stop.return_value = Path("/tmp/test.wav")
                mock_recorder_cls.return_value = mock_recorder

                with patch("hark.cli.AudioPreprocessor") as mock_prep_cls:
                    mock_prep = MagicMock()
                    mock_result = MagicMock()
                    mock_result.silence_trimmed_seconds = 0.0
                    mock_prep.process.return_value = (np.zeros(1000), mock_result)
                    mock_prep_cls.return_value = mock_prep

                    with patch("hark.cli.Transcriber") as mock_trans_cls:
                        mock_trans = MagicMock()
                        mock_trans_result = MagicMock()
                        mock_trans_result.text = "Test"
                        mock_trans_result.language = "en"
                        mock_trans_result.language_probability = 0.9
                        mock_trans_result.duration = 5.0
                        mock_trans.transcribe.return_value = mock_trans_result
                        mock_trans_cls.return_value = mock_trans

                        with patch("hark.cli.get_formatter") as mock_fmt:
                            mock_fmt.return_value.format.return_value = "Test"

                            run_workflow(default_config, None, mock_ui, verbose=False)

                mock_ui.transcription_complete.assert_called()

    def test_cleans_up_temp_file(
        self, default_config: HarkConfig, mock_ui: MagicMock, tmp_path: Path
    ) -> None:
        """Should clean up temp file."""
        temp_file = tmp_path / "recording.wav"
        temp_file.touch()

        with patch("hark.cli.KeypressHandler") as mock_keys:
            mock_handler = MagicMock()
            mock_handler.get_key.side_effect = [" ", KeyboardInterrupt]
            mock_keys.return_value.__enter__.return_value = mock_handler

            with patch("hark.cli.AudioRecorder") as mock_recorder_cls:
                mock_recorder = MagicMock()
                mock_recorder.is_recording = False
                mock_recorder.get_duration.return_value = 5.0
                mock_recorder.stop.return_value = temp_file
                mock_recorder_cls.return_value = mock_recorder

                with patch("hark.cli.AudioPreprocessor") as mock_prep_cls:
                    mock_prep = MagicMock()
                    mock_result = MagicMock()
                    mock_result.silence_trimmed_seconds = 0.0
                    mock_prep.process.return_value = (np.zeros(1000), mock_result)
                    mock_prep_cls.return_value = mock_prep

                    with patch("hark.cli.Transcriber") as mock_trans_cls:
                        mock_trans = MagicMock()
                        mock_trans_result = MagicMock()
                        mock_trans_result.text = "Test"
                        mock_trans_result.language = "en"
                        mock_trans_result.language_probability = 0.9
                        mock_trans_result.duration = 5.0
                        mock_trans.transcribe.return_value = mock_trans_result
                        mock_trans_cls.return_value = mock_trans

                        with patch("hark.cli.get_formatter") as mock_fmt:
                            mock_fmt.return_value.format.return_value = "Test"

                            run_workflow(default_config, None, mock_ui, verbose=False)

        assert not temp_file.exists()

    def test_returns_success(self, default_config: HarkConfig, mock_ui: MagicMock) -> None:
        """Should return EXIT_SUCCESS on success."""
        with patch("hark.cli.KeypressHandler") as mock_keys:
            mock_handler = MagicMock()
            mock_handler.get_key.side_effect = [" ", KeyboardInterrupt]
            mock_keys.return_value.__enter__.return_value = mock_handler

            with patch("hark.cli.AudioRecorder") as mock_recorder_cls:
                mock_recorder = MagicMock()
                mock_recorder.is_recording = False
                mock_recorder.get_duration.return_value = 5.0
                mock_recorder.stop.return_value = Path("/tmp/test.wav")
                mock_recorder_cls.return_value = mock_recorder

                with patch("hark.cli.AudioPreprocessor") as mock_prep_cls:
                    mock_prep = MagicMock()
                    mock_result = MagicMock()
                    mock_result.silence_trimmed_seconds = 0.0
                    mock_prep.process.return_value = (np.zeros(1000), mock_result)
                    mock_prep_cls.return_value = mock_prep

                    with patch("hark.cli.Transcriber") as mock_trans_cls:
                        mock_trans = MagicMock()
                        mock_trans_result = MagicMock()
                        mock_trans_result.text = "Test"
                        mock_trans_result.language = "en"
                        mock_trans_result.language_probability = 0.9
                        mock_trans_result.duration = 5.0
                        mock_trans.transcribe.return_value = mock_trans_result
                        mock_trans_cls.return_value = mock_trans

                        with patch("hark.cli.get_formatter") as mock_fmt:
                            mock_fmt.return_value.format.return_value = "Test"

                            result = run_workflow(default_config, None, mock_ui, verbose=False)

        assert result == EXIT_SUCCESS


class TestLanguageHandling:
    """Tests for language handling in run_workflow."""

    @pytest.fixture
    def mock_ui(self) -> MagicMock:
        """Create mock UI."""
        return MagicMock()

    def test_language_auto_passed_as_none(self, mock_ui: MagicMock) -> None:
        """Language 'auto' should be passed as None to transcriber."""
        config = HarkConfig()
        config.whisper.language = "auto"

        with patch("hark.cli.KeypressHandler") as mock_keys:
            mock_handler = MagicMock()
            mock_handler.get_key.side_effect = [" ", KeyboardInterrupt]
            mock_keys.return_value.__enter__.return_value = mock_handler

            with patch("hark.cli.AudioRecorder") as mock_recorder_cls:
                mock_recorder = MagicMock()
                mock_recorder.is_recording = False
                mock_recorder.get_duration.return_value = 5.0
                mock_recorder.stop.return_value = Path("/tmp/test.wav")
                mock_recorder_cls.return_value = mock_recorder

                with patch("hark.cli.AudioPreprocessor") as mock_prep_cls:
                    mock_prep = MagicMock()
                    mock_result = MagicMock()
                    mock_result.silence_trimmed_seconds = 0.0
                    mock_prep.process.return_value = (np.zeros(1000), mock_result)
                    mock_prep_cls.return_value = mock_prep

                    with patch("hark.cli.Transcriber") as mock_trans_cls:
                        mock_trans = MagicMock()
                        mock_trans_result = MagicMock()
                        mock_trans_result.text = "Test"
                        mock_trans_result.language = "en"
                        mock_trans_result.language_probability = 0.9
                        mock_trans_result.duration = 5.0
                        mock_trans.transcribe.return_value = mock_trans_result
                        mock_trans_cls.return_value = mock_trans

                        with patch("hark.cli.get_formatter") as mock_fmt:
                            mock_fmt.return_value.format.return_value = "Test"

                            run_workflow(config, None, mock_ui, verbose=False)

                        call_kwargs = mock_trans.transcribe.call_args[1]
                        assert call_kwargs["language"] is None

    def test_specific_language_passed(self, mock_ui: MagicMock) -> None:
        """Specific language should be passed to transcriber."""
        config = HarkConfig()
        config.whisper.language = "de"

        with patch("hark.cli.KeypressHandler") as mock_keys:
            mock_handler = MagicMock()
            mock_handler.get_key.side_effect = [" ", KeyboardInterrupt]
            mock_keys.return_value.__enter__.return_value = mock_handler

            with patch("hark.cli.AudioRecorder") as mock_recorder_cls:
                mock_recorder = MagicMock()
                mock_recorder.is_recording = False
                mock_recorder.get_duration.return_value = 5.0
                mock_recorder.stop.return_value = Path("/tmp/test.wav")
                mock_recorder_cls.return_value = mock_recorder

                with patch("hark.cli.AudioPreprocessor") as mock_prep_cls:
                    mock_prep = MagicMock()
                    mock_result = MagicMock()
                    mock_result.silence_trimmed_seconds = 0.0
                    mock_prep.process.return_value = (np.zeros(1000), mock_result)
                    mock_prep_cls.return_value = mock_prep

                    with patch("hark.cli.Transcriber") as mock_trans_cls:
                        mock_trans = MagicMock()
                        mock_trans_result = MagicMock()
                        mock_trans_result.text = "Test"
                        mock_trans_result.language = "de"
                        mock_trans_result.language_probability = 0.9
                        mock_trans_result.duration = 5.0
                        mock_trans.transcribe.return_value = mock_trans_result
                        mock_trans_cls.return_value = mock_trans

                        with patch("hark.cli.get_formatter") as mock_fmt:
                            mock_fmt.return_value.format.return_value = "Test"

                            run_workflow(config, None, mock_ui, verbose=False)

                        call_kwargs = mock_trans.transcribe.call_args[1]
                        assert call_kwargs["language"] == "de"
