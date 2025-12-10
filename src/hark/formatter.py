"""Output formatters for hark."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from hark.transcriber import TranscriptionResult

if TYPE_CHECKING:
    from hark.diarizer import DiarizationResult

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
    def format(self, result: TranscriptionResult | DiarizationResult) -> str:
        """
        Format transcription or diarization result to string.

        Args:
            result: Transcription or diarization result to format.

        Returns:
            Formatted string.
        """
        pass

    def _is_diarization_result(self, result: TranscriptionResult | DiarizationResult) -> bool:
        """Check if result is a DiarizationResult."""
        return hasattr(result, "speakers")


class PlainFormatter(OutputFormatter):
    """Plain text formatter."""

    def __init__(self, include_timestamps: bool = False) -> None:
        """
        Initialize plain formatter.

        Args:
            include_timestamps: Include timestamps in output.
        """
        self._include_timestamps = include_timestamps

    def format(self, result: TranscriptionResult | DiarizationResult) -> str:
        """Format as plain text."""
        if self._is_diarization_result(result):
            return self._format_diarization(result)  # type: ignore[arg-type]
        return self._format_transcription(result)  # type: ignore[arg-type]

    def _format_transcription(self, result: TranscriptionResult) -> str:
        """Format transcription result."""
        if not self._include_timestamps:
            return result.text

        lines: list[str] = []
        for segment in result.segments:
            timestamp = f"[{self._format_time(segment.start)} --> {self._format_time(segment.end)}]"
            lines.append(f"{timestamp} {segment.text}")

        return "\n".join(lines)

    def _format_diarization(self, result: DiarizationResult) -> str:
        """Format diarization result with speaker labels."""
        lines: list[str] = []

        for segment in result.segments:
            speaker = segment.speaker

            # Handle overlapping speakers (e.g., "SPEAKER_01 + SPEAKER_02")
            if self._include_timestamps:
                timestamp = self._format_time_short(segment.start)
                if " + " in speaker:
                    lines.append(f"[{timestamp}] [{speaker}] [overlapping] {segment.text}")
                else:
                    lines.append(f"[{timestamp}] [{speaker}] {segment.text}")
            else:
                if " + " in speaker:
                    lines.append(f"[{speaker}] [overlapping] {segment.text}")
                else:
                    lines.append(f"[{speaker}] {segment.text}")

        return "\n".join(lines)

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds as MM:SS.mmm"""
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins:02d}:{secs:06.3f}"

    @staticmethod
    def _format_time_short(seconds: float) -> str:
        """Format seconds as HH:MM:SS for diarization output."""
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours:02d}:{mins:02d}:{secs:02d}"
        return f"{mins:02d}:{secs:02d}"


class MarkdownFormatter(OutputFormatter):
    """Markdown formatter."""

    def __init__(self, include_timestamps: bool = False) -> None:
        """
        Initialize markdown formatter.

        Args:
            include_timestamps: Include timestamps in output.
        """
        self._include_timestamps = include_timestamps

    def format(self, result: TranscriptionResult | DiarizationResult) -> str:
        """Format as markdown."""
        if self._is_diarization_result(result):
            return self._format_diarization(result)  # type: ignore[arg-type]
        return self._format_transcription(result)  # type: ignore[arg-type]

    def _format_transcription(self, result: TranscriptionResult) -> str:
        """Format transcription result."""
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

    def _format_diarization(self, result: DiarizationResult) -> str:
        """Format diarization result with speaker labels."""
        lines = ["# Meeting Transcript", ""]

        current_speaker: str | None = None
        current_text_parts: list[str] = []
        current_timestamp = ""

        def format_speaker_header(speaker: str, timestamp: str) -> str:
            """Format speaker header with optional timestamp."""
            if self._include_timestamps:
                return f"**{speaker}** ({timestamp})"
            return f"**{speaker}**"

        for segment in result.segments:
            speaker = segment.speaker
            timestamp = self._format_time(segment.start)

            # Handle overlapping speakers
            if " + " in speaker:
                # Flush current speaker's text
                if current_speaker and current_text_parts:
                    lines.append(format_speaker_header(current_speaker, current_timestamp))
                    lines.append(" ".join(current_text_parts))
                    lines.append("")
                    current_text_parts = []
                    current_speaker = None

                # Add overlap as separate block
                header = format_speaker_header(speaker, timestamp)
                lines.append(f"{header} *[overlapping]*")
                lines.append(segment.text)
                lines.append("")
            elif speaker != current_speaker:
                # Flush previous speaker's text
                if current_speaker and current_text_parts:
                    lines.append(format_speaker_header(current_speaker, current_timestamp))
                    lines.append(" ".join(current_text_parts))
                    lines.append("")

                # Start new speaker
                current_speaker = speaker
                current_timestamp = timestamp
                current_text_parts = [segment.text]
            else:
                # Same speaker, accumulate text
                current_text_parts.append(segment.text)

        # Flush final speaker
        if current_speaker and current_text_parts:
            lines.append(format_speaker_header(current_speaker, current_timestamp))
            lines.append(" ".join(current_text_parts))
            lines.append("")

        # Add metadata footer
        speaker_count = len(result.speakers)
        duration_str = self._format_duration(result.duration)
        confidence = result.language_probability
        lines.extend(
            [
                "---",
                "",
                f"*{speaker_count} speakers detected • Duration: {duration_str} • "
                f"Language: {result.language} ({confidence:.0%} confidence)*",
            ]
        )

        return "\n".join(lines)

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds as MM:SS"""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format duration as M:SS or H:MM:SS."""
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours}:{mins:02d}:{secs:02d}"
        return f"{mins}:{secs:02d}"


class SRTFormatter(OutputFormatter):
    """SRT subtitle formatter."""

    def format(self, result: TranscriptionResult | DiarizationResult) -> str:
        """Format as SRT subtitles."""
        if self._is_diarization_result(result):
            return self._format_diarization(result)  # type: ignore[arg-type]
        return self._format_transcription(result)  # type: ignore[arg-type]

    def _format_transcription(self, result: TranscriptionResult) -> str:
        """Format transcription result as SRT."""
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

    def _format_diarization(self, result: DiarizationResult) -> str:
        """Format diarization result as SRT with speaker labels."""
        lines: list[str] = []

        for i, segment in enumerate(result.segments, 1):
            # Sequence number
            lines.append(str(i))

            # Timestamps
            start_time = self._format_srt_time(segment.start)
            end_time = self._format_srt_time(segment.end)
            lines.append(f"{start_time} --> {end_time}")

            # Text with speaker label
            speaker = segment.speaker
            if " + " in speaker:
                # Overlapping speakers
                lines.append(f"[{speaker}] [overlapping]")
                lines.append(segment.text)
            else:
                lines.append(f"[{speaker}] {segment.text}")

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
