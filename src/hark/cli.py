"""Main CLI entry point for hark."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from hark.diarizer import DiarizationResult

__all__ = [
    "create_parser",
    "run_workflow",
    "main",
]

from hark import __version__
from hark.config import (
    HarkConfig,
    ensure_directories,
    load_config,
    merge_cli_args,
    validate_config,
)
from hark.constants import (
    EXIT_ERROR,
    EXIT_INTERRUPT,
    EXIT_SUCCESS,
    MIN_RECORDING_DURATION,
    VALID_MODELS,
    VALID_OUTPUT_FORMATS,
)
from hark.exceptions import DependencyMissingError, HarkError, MissingTokenError
from hark.formatter import get_formatter
from hark.keypress import KeypressHandler
from hark.preprocessor import AudioPreprocessor
from hark.recorder import AudioRecorder
from hark.transcriber import Transcriber, TranscriptionResult
from hark.ui import UI

# Type alias for results
FormattableResult = TranscriptionResult


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="hark",
        description="Speech-to-text recording and transcription",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  hark                                  Record and output to stdout
  hark output.txt                       Record and save to file
  hark --lang de speech.md              Record with German language
  hark --model large-v3 out.txt         Use large model for better accuracy
  hark --timestamps notes.md            Include timestamps in output
  hark --format srt subs.srt            Output as SRT subtitles
  hark --diarize --input speaker out.txt  Transcribe with speaker detection
  hark --diarize --speakers 3 meeting.md  Hint: 3 speakers expected
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
        "--input",
        dest="input_source",
        choices=["mic", "speaker", "both"],
        metavar="SOURCE",
        help="Audio input source: mic (default), speaker (system audio), or both (stereo)",
    )
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

    # Diarization options
    diarization = parser.add_argument_group("Diarization Options")
    diarization.add_argument(
        "--diarize",
        action="store_true",
        help="Enable speaker diarization (requires --input speaker or both)",
    )
    diarization.add_argument(
        "--speakers",
        type=int,
        metavar="N",
        help="Expected number of speakers (helps diarization accuracy)",
    )
    diarization.add_argument(
        "--no-interactive",
        action="store_true",
        help="Skip interactive speaker naming prompt",
    )

    # Other options
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    return parser


def _validate_diarization_args(args: argparse.Namespace, config: HarkConfig, ui: UI) -> bool:
    """
    Validate diarization-related arguments.

    Args:
        args: Parsed CLI arguments.
        config: Application configuration.
        ui: UI handler for error messages.

    Returns:
        True if validation passes, False otherwise.
    """
    if not getattr(args, "diarize", False):
        return True

    # Check input source
    input_source = getattr(args, "input_source", None) or config.recording.input_source
    if input_source == "mic":
        ui.error(
            "Diarization requires --input speaker or --input both.\n"
            "Use --input mic without --diarize for single-speaker recordings."
        )
        return False

    # Check dependencies
    try:
        import whisperx  # type: ignore[import-not-found]  # noqa: F401
    except ImportError as e:
        raise DependencyMissingError() from e

    # Check HF token
    if not config.diarization.hf_token:
        raise MissingTokenError()

    return True


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


def _record_audio(ui: UI, config: HarkConfig) -> tuple[Path, float] | None:
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
        input_source=config.recording.input_source,
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
                    input_source=config.recording.input_source,
                )
                time.sleep(0.05)
                keys.get_key(timeout=0.01)
    except KeyboardInterrupt:
        pass  # Normal stop via Ctrl+C

    audio_path = recorder.stop()
    duration = recorder.get_duration()
    ui.recording_stopped(duration)

    return audio_path, duration


def _preprocess_audio(
    ui: UI,
    config: HarkConfig,
    audio_path: Path,
    preserve_stereo: bool = False,
) -> tuple[np.ndarray, float]:
    """
    Preprocess recorded audio.

    Args:
        ui: UI handler.
        config: Application configuration.
        audio_path: Path to audio file.
        preserve_stereo: If True, preserve stereo channels for diarization.

    Returns:
        Tuple of (processed_audio, silence_trimmed_seconds).
    """
    ui.preprocessing_header()

    preprocessor = AudioPreprocessor(config.preprocessing)
    processed_audio, preprocess_result = preprocessor.process(
        audio_path=audio_path,
        sample_rate=config.recording.sample_rate,
        preserve_stereo=preserve_stereo,
    )

    if config.preprocessing.noise_reduction.enabled:
        ui.preprocessing_step("Noise reduction applied")
    if config.preprocessing.normalization.enabled:
        ui.preprocessing_step("Audio normalized")
    if config.preprocessing.silence_trimming.enabled:
        trimmed = preprocess_result.silence_trimmed_seconds
        ui.preprocessing_step(f"Silence trimmed ({trimmed:.1f}s removed)")

    return processed_audio, preprocess_result.silence_trimmed_seconds


def _transcribe_audio(ui: UI, config: HarkConfig, audio: np.ndarray) -> TranscriptionResult:
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


def _diarize_audio(
    ui: UI,
    config: HarkConfig,
    audio: np.ndarray,
    num_speakers: int | None = None,
) -> DiarizationResult:
    """
    Transcribe and diarize audio using WhisperX.

    Args:
        ui: UI handler.
        config: Application configuration.
        audio: Processed audio data.
        num_speakers: Expected number of speakers (hint).

    Returns:
        DiarizationResult with speaker-labeled segments.
    """
    from hark.diarizer import Diarizer

    ui.info(f"\nLoading Whisper model '{config.whisper.model}' with diarization...")

    diarizer = Diarizer(
        model_name=config.whisper.model,
        device=config.whisper.device,
        hf_token=config.diarization.hf_token,
        num_speakers=num_speakers,
        model_cache_dir=config.model_cache_dir,
    )

    ui.info("Model loaded. Transcribing and diarizing...")

    language = config.whisper.language if config.whisper.language != "auto" else None
    return diarizer.transcribe_and_diarize(
        audio=audio,
        sample_rate=config.recording.sample_rate,
        language=language,
    )


def _process_stereo_diarization(
    ui: UI,
    config: HarkConfig,
    audio: np.ndarray,
    num_speakers: int | None = None,
) -> DiarizationResult:
    """
    Process stereo audio with separate handling for local/remote channels.

    Args:
        ui: UI handler.
        config: Application configuration.
        audio: Stereo audio data (L=mic, R=speaker).
        num_speakers: Expected number of remote speakers.

    Returns:
        DiarizationResult with all speakers labeled.
    """
    from hark.stereo_processor import StereoProcessor

    ui.info("\nProcessing stereo audio (L=mic, R=speaker)...")

    processor = StereoProcessor(config=config, num_speakers=num_speakers)

    ui.info(f"Loading Whisper model '{config.whisper.model}'...")
    ui.info("Transcribing local channel...")
    ui.info("Diarizing remote channel...")

    return processor.process(
        stereo_audio=audio,
        sample_rate=config.recording.sample_rate,
    )


def _write_output(
    ui: UI,
    config: HarkConfig,
    result: TranscriptionResult | DiarizationResult,
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


def run_workflow(
    config: HarkConfig,
    output_file: str | None,
    ui: UI,
    verbose: bool,
    diarize: bool = False,
    num_speakers: int | None = None,
    no_interactive: bool = False,
) -> int:
    """
    Run the main recording/transcription workflow.

    Args:
        config: Application configuration.
        output_file: Output file path (None for stdout).
        ui: UI handler.
        verbose: Enable verbose output.
        diarize: Enable speaker diarization.
        num_speakers: Expected number of speakers (hint for diarization).
        no_interactive: Skip interactive speaker naming.

    Returns:
        Exit code.
    """
    # Display header and config
    ui.header("Hark - Speech-to-Text Recording")
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
    # Preserve stereo if diarizing with --input both (need separate channels)
    preserve_stereo = diarize and config.recording.input_source == "both"
    processed_audio, _ = _preprocess_audio(ui, config, audio_path, preserve_stereo=preserve_stereo)

    # Transcribe or diarize
    result: TranscriptionResult | DiarizationResult
    if diarize:
        if config.recording.input_source == "both":
            # Stereo mode: process channels separately
            result = _process_stereo_diarization(ui, config, processed_audio, num_speakers)
        else:
            # Mono mode (speaker input): diarize directly
            result = _diarize_audio(ui, config, processed_audio, num_speakers)

        # Show speaker summary
        ui.info(f"\nDetected {len(result.speakers)} speaker(s): {', '.join(result.speakers)}")

        # Interactive speaker naming (unless --no-interactive)
        if not no_interactive:
            from hark.interactive import interactive_speaker_naming

            local_speaker = config.diarization.local_speaker_name or "SPEAKER_00"
            result = interactive_speaker_naming(
                result,
                quiet=config.interface.quiet,
                local_speaker_name=local_speaker,
                ui=ui,
            )
    else:
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

        # Validate diarization args
        if not _validate_diarization_args(args, config, ui):
            return EXIT_ERROR

        # Ensure directories exist
        ensure_directories(config)

        # Run main workflow
        return run_workflow(
            config,
            args.output_file,
            ui,
            verbose=args.verbose,
            diarize=getattr(args, "diarize", False),
            num_speakers=getattr(args, "speakers", None),
            no_interactive=getattr(args, "no_interactive", False),
        )

    except KeyboardInterrupt:
        ui.info("\nCancelled.")
        return EXIT_INTERRUPT
    except HarkError as e:
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
