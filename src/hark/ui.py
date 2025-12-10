"""Terminal UI for hark."""

from __future__ import annotations

import sys
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hark.config import HarkConfig
    from hark.diarizer import DiarizationResult
    from hark.transcriber import TranscriptionResult

__all__ = [
    "Color",
    "UI",
    "HEADER_WIDTH",
    "RECORDING_BAR_WIDTH",
    "LEVEL_METER_WIDTH",
    "TRANSCRIPTION_BAR_WIDTH",
]

# UI display constants
HEADER_WIDTH = 60
RECORDING_BAR_WIDTH = 30
LEVEL_METER_WIDTH = 16
TRANSCRIPTION_BAR_WIDTH = 40


class Color(Enum):
    """ANSI color codes."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"


class UI:
    """Terminal UI handler."""

    def __init__(self, quiet: bool = False, use_color: bool = True) -> None:
        """
        Initialize UI handler.

        Args:
            quiet: Suppress non-essential output.
            use_color: Use ANSI colors (auto-disabled if not a TTY).
        """
        self._quiet = quiet
        self._use_color = use_color and sys.stdout.isatty()
        self._last_recording_line_count = 0

    def _color(self, text: str, color: Color) -> str:
        """Apply color to text if colors enabled."""
        if not self._use_color:
            return text
        return f"{color.value}{text}{Color.RESET.value}"

    def _clear_lines(self, count: int) -> None:
        """Clear previous lines for updating display."""
        if count > 0 and self._use_color:
            # Move up and clear each line
            for _ in range(count):
                sys.stdout.write("\033[1A")  # Move up
                sys.stdout.write("\033[2K")  # Clear line

    def header(self, title: str) -> None:
        """Print header."""
        if self._quiet:
            return
        print()
        print(self._color(title, Color.BOLD))
        print("=" * HEADER_WIDTH)

    def config_summary(self, config: HarkConfig, output_file: str | None) -> None:
        """Print configuration summary."""
        if self._quiet:
            return

        # Format input source display
        input_source_labels = {
            "mic": "Microphone",
            "speaker": "System Audio",
            "both": "Microphone + System Audio (stereo)",
        }
        input_label = input_source_labels.get(
            config.recording.input_source,
            config.recording.input_source,
        )

        print()
        print("Configuration:")
        print(f"  - Input: {input_label}")
        print(f"  - Language: {config.whisper.language}")
        print(f"  - Model: {config.whisper.model}")
        print(f"  - Output: {output_file or 'stdout'}")
        print(f"  - Max Duration: {self._format_duration(config.recording.max_duration)}")

        preprocessing = []
        if config.preprocessing.noise_reduction.enabled:
            preprocessing.append("noise reduction")
        if config.preprocessing.normalization.enabled:
            preprocessing.append("normalization")
        if config.preprocessing.silence_trimming.enabled:
            preprocessing.append("silence trimming")

        prep_str = ", ".join(preprocessing) if preprocessing else "disabled"
        print(f"  - Preprocessing: {prep_str}")

    def prompt_start(self) -> None:
        """Print start prompt."""
        if self._quiet:
            return
        space = self._color("SPACE", Color.CYAN)
        ctrl_c = self._color("Ctrl+C", Color.YELLOW)
        print(f"\nPress {space} to start recording, {ctrl_c} to cancel...")

    def recording_status(
        self,
        elapsed: float,
        max_duration: float,
        level: float,
        input_source: str = "mic",
    ) -> None:
        """
        Update recording status display.

        Args:
            elapsed: Elapsed recording time in seconds.
            max_duration: Maximum recording duration in seconds.
            level: Audio level (0.0-1.0).
            input_source: Input source mode ("mic", "speaker", or "both").
        """
        if self._quiet:
            return

        # Clear previous output
        self._clear_lines(self._last_recording_line_count)

        # Progress bar
        progress = min(elapsed / max_duration, 1.0)
        filled = int(RECORDING_BAR_WIDTH * progress)
        bar = "\u2588" * filled + "\u2591" * (RECORDING_BAR_WIDTH - filled)

        # Level meter
        level_clamped = min(max(level, 0.0), 1.0)
        level_bars = int(level_clamped * LEVEL_METER_WIDTH)
        level_meter = "\u2588" * level_bars + "\u2591" * (LEVEL_METER_WIDTH - level_bars)
        level_pct = int(level_clamped * 100)

        # Time display
        time_str = f"{self._format_duration(elapsed)} / {self._format_duration(max_duration)}"

        # Recording indicator and label based on input source
        if input_source == "both":
            rec_indicator = self._color("\u25cf", Color.MAGENTA)
            rec_label = "Recording (mic+speaker)..."
        elif input_source == "speaker":
            rec_indicator = self._color("\u25cf", Color.CYAN)
            rec_label = "Recording (speaker)..."
        else:
            rec_indicator = self._color("\u25cf", Color.RED)
            rec_label = "Recording..."

        # Build output
        lines = [
            f"{rec_indicator} {rec_label} {time_str} {bar}",
            f"  Audio Level: {level_meter} ({level_pct:3d}%)",
        ]

        for line in lines:
            print(line)

        self._last_recording_line_count = len(lines)
        sys.stdout.flush()

    def recording_stopped(self, duration: float) -> None:
        """Print recording stopped message."""
        if self._quiet:
            return

        # Clear the recording status
        self._clear_lines(self._last_recording_line_count)
        self._last_recording_line_count = 0

        stop_indicator = self._color("\u25a0", Color.BLUE)
        print(f"{stop_indicator} Recording stopped ({self._format_duration(duration)})")

    def preprocessing_header(self) -> None:
        """Print preprocessing header."""
        if self._quiet:
            return
        print(f"\n{self._color('Processing audio...', Color.CYAN)}")

    def preprocessing_step(self, step: str, success: bool = True) -> None:
        """Print preprocessing step result."""
        if self._quiet:
            return
        icon = self._color("\u2713", Color.GREEN) if success else self._color("\u2717", Color.RED)
        print(f"  {icon} {step}")

    def transcription_progress(self, progress: float) -> None:
        """
        Update transcription progress bar.

        Args:
            progress: Progress value (0.0-1.0).
        """
        if self._quiet:
            return

        progress_clamped = min(max(progress, 0.0), 1.0)
        filled = int(TRANSCRIPTION_BAR_WIDTH * progress_clamped)
        bar = "\u2588" * filled + "\u2591" * (TRANSCRIPTION_BAR_WIDTH - filled)
        pct = int(progress_clamped * 100)

        # Use carriage return for in-place update
        print(f"\r  Transcribing: {bar} {pct:3d}%", end="", flush=True)

    def transcription_complete(
        self,
        result: TranscriptionResult | DiarizationResult,
        output_path: str | None,
    ) -> None:
        """Print transcription complete summary."""
        if self._quiet:
            return

        # New line after progress bar
        print()
        print()

        # Check if this is a diarization result (has speakers attribute)
        is_diarization = hasattr(result, "speakers")

        if is_diarization:
            check = self._color("\u2713 Diarization complete!", Color.GREEN)
        else:
            check = self._color("\u2713 Transcription complete!", Color.GREEN)
        print(check)

        # Calculate word count from text or segments
        if hasattr(result, "text") and result.text:
            word_count = sum(1 for _ in result.text.split())
        else:
            # DiarizationResult: count words from segments
            word_count = sum(len(seg.text.split()) for seg in result.segments if seg.text)

        print(f"  - Duration: {result.duration:.1f} seconds")
        print(f"  - Words: {word_count}")
        print(f"  - Language: {result.language} ({result.language_probability:.0%} confidence)")

        # Show speaker count for diarization
        if is_diarization:
            print(f"  - Speakers: {len(result.speakers)}")  # type: ignore[union-attr]

        if output_path:
            print(f"  - Output: {output_path}")

    def error(self, message: str) -> None:
        """Print error message."""
        error_label = self._color("Error:", Color.RED)
        print(f"{error_label} {message}", file=sys.stderr)

    def warning(self, message: str) -> None:
        """Print warning message."""
        if self._quiet:
            return
        warning_label = self._color("Warning:", Color.YELLOW)
        print(f"{warning_label} {message}")

    def info(self, message: str) -> None:
        """Print info message."""
        if self._quiet:
            return
        print(message)

    def verbose(self, message: str) -> None:
        """Print verbose/debug message."""
        debug_label = self._color("[DEBUG]", Color.MAGENTA)
        print(f"{debug_label} {message}")

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format seconds as MM:SS."""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"
