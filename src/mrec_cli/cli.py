"""Main CLI entry point for mrec-cli."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

__all__ = [
    "create_parser",
    "run_workflow",
    "main",
]

from mrec_cli import __version__
from mrec_cli.config import (
    MrecConfig,
    ensure_directories,
    load_config,
    merge_cli_args,
    validate_config,
)
from mrec_cli.constants import (
    EXIT_ERROR,
    EXIT_INTERRUPT,
    EXIT_SUCCESS,
    MIN_RECORDING_DURATION,
    VALID_MODELS,
    VALID_OUTPUT_FORMATS,
)
from mrec_cli.exceptions import MrecError
from mrec_cli.formatter import get_formatter
from mrec_cli.keypress import KeypressHandler
from mrec_cli.preprocessor import AudioPreprocessor
from mrec_cli.recorder import AudioRecorder
from mrec_cli.transcriber import Transcriber, TranscriptionResult
from mrec_cli.ui import UI


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="mrec-cli",
        description="Speech-to-text recording and transcription",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  mrec-cli                          Record and output to stdout
  mrec-cli output.txt               Record and save to file
  mrec-cli --lang de speech.md      Record with German language
  mrec-cli --model large-v3 out.txt Use large model for better accuracy
  mrec-cli --timestamps notes.md    Include timestamps in output
  mrec-cli --format srt subs.srt    Output as SRT subtitles
""",
    )

    # Positional argument
    parser.add_argument(
        "output_file",
        nargs="?",
        help="Output file path (default: stdout)",
    )

    # Recording options
    recording = parser.add_argument_group("Recording Options")
    recording.add_argument(
        "--max-duration",
        type=int,
        metavar="SECONDS",
        help="Maximum recording duration (default: 600)",
    )
    recording.add_argument(
        "--sample-rate",
        type=int,
        metavar="HZ",
        help="Audio sample rate (default: 16000)",
    )
    recording.add_argument(
        "--channels",
        type=int,
        metavar="NUM",
        help="Audio channels (default: 1)",
    )

    # Language options
    language = parser.add_argument_group("Language Options")
    language.add_argument(
        "--lang",
        metavar="LANGUAGE",
        help="Language code (en, de, auto) (default: auto)",
    )
    language.add_argument(
        "--model",
        choices=VALID_MODELS,
        help=f"Whisper model (default: base). Options: {', '.join(VALID_MODELS)}",
    )

    # Preprocessing options
    preprocess = parser.add_argument_group("Preprocessing Options")
    preprocess.add_argument(
        "--no-noise-reduction",
        action="store_true",
        help="Disable noise reduction",
    )
    preprocess.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable audio normalization",
    )
    preprocess.add_argument(
        "--no-trim-silence",
        action="store_true",
        help="Disable silence trimming",
    )
    preprocess.add_argument(
        "--noise-strength",
        type=float,
        metavar="FLOAT",
        help="Noise reduction strength 0.0-1.0 (default: 0.5)",
    )

    # Output options
    output = parser.add_argument_group("Output Options")
    output.add_argument(
        "--timestamps",
        action="store_true",
        help="Include timestamps in output",
    )
    output.add_argument(
        "--format",
        choices=VALID_OUTPUT_FORMATS,
        help=f"Output format (default: plain). Options: {', '.join(VALID_OUTPUT_FORMATS)}",
    )
    output.add_argument(
        "--append",
        action="store_true",
        help="Append to output file instead of overwriting",
    )

    # Interface options
    interface = parser.add_argument_group("Interface Options")
    interface.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Minimal output, no progress indicators",
    )
    interface.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Detailed processing information",
    )
    interface.add_argument(
        "--config",
        metavar="CONFIG_FILE",
        help="Use custom configuration file",
    )

    # Other options
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    return parser


def _wait_for_start_signal(ui: UI) -> bool:
    """
    Wait for user to press space to start recording.

    Args:
        ui: UI handler.

    Returns:
        True if space was pressed, False if interrupted.
    """
    ui.prompt_start()
    try:
        with KeypressHandler() as keys:
            while True:
                key = keys.get_key(timeout=0.1)
                if key == " ":
                    return True
                elif key == "\x03":  # Ctrl+C
                    raise KeyboardInterrupt
    except KeyboardInterrupt:
        ui.info("\nCancelled.")
        return False


def _record_audio(ui: UI, config: MrecConfig) -> tuple[Path, float] | None:
    """
    Record audio from microphone.

    Args:
        ui: UI handler.
        config: Application configuration.

    Returns:
        Tuple of (audio_path, duration) or None if interrupted too early.
    """
    # Set up level tracking for UI
    current_level = [0.0]

    def level_callback(level: float) -> None:
        current_level[0] = min(level * 10, 1.0)

    recorder = AudioRecorder(
        sample_rate=config.recording.sample_rate,
        channels=config.recording.channels,
        max_duration=config.recording.max_duration,
        level_callback=level_callback,
        temp_dir=config.temp_directory,
    )

    ui.info("")  # New line after prompt
    recorder.start()

    # Recording loop with UI updates
    try:
        with KeypressHandler() as keys:
            while recorder.is_recording:
                ui.recording_status(
                    elapsed=recorder.get_duration(),
                    max_duration=config.recording.max_duration,
                    level=current_level[0],
                )
                time.sleep(0.05)
                keys.get_key(timeout=0.01)
    except KeyboardInterrupt:
        pass  # Normal stop via Ctrl+C

    audio_path = recorder.stop()
    duration = recorder.get_duration()
    ui.recording_stopped(duration)

    return audio_path, duration


def _preprocess_audio(ui: UI, config: MrecConfig, audio_path: Path) -> tuple[np.ndarray, float]:
    """
    Preprocess recorded audio.

    Args:
        ui: UI handler.
        config: Application configuration.
        audio_path: Path to audio file.

    Returns:
        Tuple of (processed_audio, silence_trimmed_seconds).
    """
    ui.preprocessing_header()

    preprocessor = AudioPreprocessor(config.preprocessing)
    processed_audio, preprocess_result = preprocessor.process(
        audio_path=audio_path,
        sample_rate=config.recording.sample_rate,
    )

    if config.preprocessing.noise_reduction.enabled:
        ui.preprocessing_step("Noise reduction applied")
    if config.preprocessing.normalization.enabled:
        ui.preprocessing_step("Audio normalized")
    if config.preprocessing.silence_trimming.enabled:
        trimmed = preprocess_result.silence_trimmed_seconds
        ui.preprocessing_step(f"Silence trimmed ({trimmed:.1f}s removed)")

    return processed_audio, preprocess_result.silence_trimmed_seconds


def _transcribe_audio(ui: UI, config: MrecConfig, audio: np.ndarray) -> TranscriptionResult:
    """
    Load model and transcribe audio.

    Args:
        ui: UI handler.
        config: Application configuration.
        audio: Processed audio data.

    Returns:
        Transcription result.
    """
    ui.info(f"\nLoading Whisper model '{config.whisper.model}'...")

    transcriber = Transcriber(
        model_name=config.whisper.model,
        device=config.whisper.device,
        model_cache_dir=config.model_cache_dir,
    )
    transcriber.load_model()
    ui.info("Model loaded.")

    ui.info("\nTranscribing audio...")

    def progress_callback(progress: float) -> None:
        ui.transcription_progress(progress)

    language = config.whisper.language if config.whisper.language != "auto" else None
    return transcriber.transcribe(
        audio=audio,
        sample_rate=config.recording.sample_rate,
        language=language,
        progress_callback=progress_callback,
    )


def _write_output(
    ui: UI,
    config: MrecConfig,
    result: TranscriptionResult,
    output_file: str | None,
) -> None:
    """
    Format and write transcription output.

    Args:
        ui: UI handler.
        config: Application configuration.
        result: Transcription result.
        output_file: Output file path (None for stdout).
    """
    formatter = get_formatter(
        format_name=config.output.format,
        include_timestamps=config.output.timestamps,
    )
    output_text = formatter.format(result)

    if output_file:
        output_path = Path(output_file)
        mode = "a" if config.output.append_mode else "w"
        with output_path.open(mode, encoding=config.output.encoding) as f:
            f.write(output_text)
            if config.output.append_mode:
                f.write("\n")
        ui.transcription_complete(result, str(output_path))
    else:
        ui.info("")  # New line before output
        print(output_text)
        if not config.interface.quiet:
            ui.transcription_complete(result, None)


def run_workflow(config: MrecConfig, output_file: str | None, ui: UI, verbose: bool) -> int:
    """
    Run the main recording/transcription workflow.

    Args:
        config: Application configuration.
        output_file: Output file path (None for stdout).
        ui: UI handler.
        verbose: Enable verbose output.

    Returns:
        Exit code.
    """
    # Display header and config
    ui.header("MREC-CLI - Speech-to-Text Recording")
    ui.config_summary(config, output_file)

    # Wait for user to start
    if not _wait_for_start_signal(ui):
        return EXIT_INTERRUPT

    # Record audio
    recording_result = _record_audio(ui, config)
    if recording_result is None:
        return EXIT_INTERRUPT

    audio_path, duration = recording_result

    # Check minimum duration
    if duration < MIN_RECORDING_DURATION:
        ui.error(f"Recording too short ({duration:.1f}s < {MIN_RECORDING_DURATION}s)")
        return EXIT_ERROR

    # Preprocess audio
    processed_audio, _ = _preprocess_audio(ui, config, audio_path)

    # Transcribe
    result = _transcribe_audio(ui, config, processed_audio)

    # Write output
    _write_output(ui, config, result, output_file)

    # Cleanup temp file
    try:
        audio_path.unlink()
        if verbose:
            ui.verbose(f"Cleaned up temp file: {audio_path}")
    except OSError:
        pass

    return EXIT_SUCCESS


def main(argv: list[str] | None = None) -> int:
    """
    Main entry point.

    Args:
        argv: Command line arguments (None for sys.argv).

    Returns:
        Exit code.
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    # Initialize UI early for error reporting
    ui = UI(quiet=args.quiet, use_color=True)

    try:
        # Load and merge config
        config_path = Path(args.config) if args.config else None
        config = load_config(config_path)
        config = merge_cli_args(config, args)

        # Validate config
        errors = validate_config(config)
        if errors:
            for error in errors:
                ui.error(error)
            return EXIT_ERROR

        # Ensure directories exist
        ensure_directories(config)

        # Run main workflow
        return run_workflow(config, args.output_file, ui, verbose=args.verbose)

    except KeyboardInterrupt:
        ui.info("\nCancelled.")
        return EXIT_INTERRUPT
    except MrecError as e:
        ui.error(str(e))
        return EXIT_ERROR
    except Exception as e:
        ui.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
