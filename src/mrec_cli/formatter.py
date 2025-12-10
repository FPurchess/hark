"""Output formatters for mrec-cli."""

from __future__ import annotations

from abc import ABC, abstractmethod

from mrec_cli.transcriber import TranscriptionResult

__all__ = [
    "OutputFormatter",
    "PlainFormatter",
    "MarkdownFormatter",
    "SRTFormatter",
    "get_formatter",
]


class OutputFormatter(ABC):
    """Base class for output formatters."""

    @abstractmethod
    def format(self, result: TranscriptionResult) -> str:
        """
        Format transcription result to string.

        Args:
            result: Transcription result to format.

        Returns:
            Formatted string.
        """
        pass


class PlainFormatter(OutputFormatter):
    """Plain text formatter."""

    def __init__(self, include_timestamps: bool = False) -> None:
        """
        Initialize plain formatter.

        Args:
            include_timestamps: Include timestamps in output.
        """
        self._include_timestamps = include_timestamps

    def format(self, result: TranscriptionResult) -> str:
        """Format as plain text."""
        if not self._include_timestamps:
            return result.text

        lines: list[str] = []
        for segment in result.segments:
            timestamp = f"[{self._format_time(segment.start)} --> {self._format_time(segment.end)}]"
            lines.append(f"{timestamp} {segment.text}")

        return "\n".join(lines)

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds as MM:SS.mmm"""
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins:02d}:{secs:06.3f}"


class MarkdownFormatter(OutputFormatter):
    """Markdown formatter."""

    def __init__(self, include_timestamps: bool = False) -> None:
        """
        Initialize markdown formatter.

        Args:
            include_timestamps: Include timestamps in output.
        """
        self._include_timestamps = include_timestamps

    def format(self, result: TranscriptionResult) -> str:
        """Format as markdown."""
        lines = ["# Transcription", ""]

        if self._include_timestamps:
            for segment in result.segments:
                timestamp = self._format_time(segment.start)
                lines.append(f"**[{timestamp}]** {segment.text}")
                lines.append("")
        else:
            lines.append(result.text)
            lines.append("")

        # Add metadata footer
        lines.extend(
            [
                "---",
                "",
                f"*Language: {result.language} ({result.language_probability:.0%} confidence)*  ",
                f"*Duration: {result.duration:.1f}s*",
            ]
        )

        return "\n".join(lines)

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds as MM:SS"""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"


class SRTFormatter(OutputFormatter):
    """SRT subtitle formatter."""

    def format(self, result: TranscriptionResult) -> str:
        """Format as SRT subtitles."""
        lines: list[str] = []

        for i, segment in enumerate(result.segments, 1):
            # Sequence number
            lines.append(str(i))

            # Timestamps
            start_time = self._format_srt_time(segment.start)
            end_time = self._format_srt_time(segment.end)
            lines.append(f"{start_time} --> {end_time}")

            # Text
            lines.append(segment.text)

            # Blank line between entries
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _format_srt_time(seconds: float) -> str:
        """Format seconds as HH:MM:SS,mmm (SRT format)."""
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = seconds % 60
        # SRT uses comma for milliseconds
        return f"{hours:02d}:{mins:02d}:{secs:06.3f}".replace(".", ",")


def get_formatter(format_name: str, include_timestamps: bool = False) -> OutputFormatter:
    """
    Get formatter instance by name.

    Args:
        format_name: Format name (plain, markdown, srt).
        include_timestamps: Include timestamps in output.

    Returns:
        OutputFormatter instance.

    Raises:
        ValueError: If format name is invalid.
    """
    match format_name:
        case "plain":
            return PlainFormatter(include_timestamps=include_timestamps)
        case "markdown":
            return MarkdownFormatter(include_timestamps=include_timestamps)
        case "srt":
            # SRT always has timestamps
            return SRTFormatter()
        case _:
            raise ValueError(f"Unknown format: {format_name}. Valid formats: plain, markdown, srt")
