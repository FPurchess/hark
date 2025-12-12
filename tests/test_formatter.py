"""Tests for hark.formatter module."""

import pytest

from hark.constants import UNKNOWN_LANGUAGE_PROBABILITY
from hark.diarizer import DiarizationResult
from hark.formatter import (
    MarkdownFormatter,
    OutputFormatter,
    PlainFormatter,
    SRTFormatter,
    get_formatter,
)
from hark.transcriber import TranscriptionResult, TranscriptionSegment


class TestPlainFormatter:
    """Tests for PlainFormatter class."""

    def test_no_timestamps_returns_text_only(
        self, sample_transcription_result: TranscriptionResult
    ) -> None:
        """Without timestamps, should return only the text."""
        formatter = PlainFormatter(include_timestamps=False)
        result = formatter.format(sample_transcription_result)
        assert result == "Hello world. This is a test."

    def test_with_timestamps_format(self, sample_transcription_result: TranscriptionResult) -> None:
        """With timestamps, each segment should have [start --> end] prefix."""
        formatter = PlainFormatter(include_timestamps=True)
        result = formatter.format(sample_transcription_result)

        # Should have timestamps in format [MM:SS.mmm --> MM:SS.mmm]
        assert "[00:00.000 --> 00:01.500]" in result
        assert "[00:01.600 --> 00:03.000]" in result
        assert "Hello world." in result
        assert "This is a test." in result

    def test_time_format_minutes_seconds_ms(self) -> None:
        """Time format should be MM:SS.mmm."""
        result = TranscriptionResult(
            text="Test",
            segments=[
                TranscriptionSegment(start=65.5, end=125.123, text="Test", words=[]),
            ],
            language="en",
            language_probability=0.9,
            duration=125.123,
        )
        formatter = PlainFormatter(include_timestamps=True)
        output = formatter.format(result)

        # 65.5s = 01:05.500, 125.123s = 02:05.123
        assert "[01:05.500 --> 02:05.123]" in output

    def test_empty_text(self, empty_transcription_result: TranscriptionResult) -> None:
        """Empty text should return empty string."""
        formatter = PlainFormatter(include_timestamps=False)
        result = formatter.format(empty_transcription_result)
        assert result == ""

    def test_no_segments_with_timestamps(
        self, empty_transcription_result: TranscriptionResult
    ) -> None:
        """Empty segments with timestamps should return empty string."""
        formatter = PlainFormatter(include_timestamps=True)
        result = formatter.format(empty_transcription_result)
        assert result == ""

    def test_format_time_static_method(self) -> None:
        """_format_time should format correctly."""
        assert PlainFormatter._format_time(0.0) == "00:00.000"
        assert PlainFormatter._format_time(5.5) == "00:05.500"
        assert PlainFormatter._format_time(65.123) == "01:05.123"
        assert PlainFormatter._format_time(3599.999) == "59:59.999"


class TestMarkdownFormatter:
    """Tests for MarkdownFormatter class."""

    def test_header_present(self, sample_transcription_result: TranscriptionResult) -> None:
        """Output should start with '# Transcription' header."""
        formatter = MarkdownFormatter(include_timestamps=False)
        result = formatter.format(sample_transcription_result)
        assert result.startswith("# Transcription")

    def test_no_timestamps_plain_text_body(
        self, sample_transcription_result: TranscriptionResult
    ) -> None:
        """Without timestamps, body should be plain text."""
        formatter = MarkdownFormatter(include_timestamps=False)
        result = formatter.format(sample_transcription_result)
        assert "Hello world. This is a test." in result

    def test_with_timestamps_bold_prefix(
        self, sample_transcription_result: TranscriptionResult
    ) -> None:
        """With timestamps, segments should have **[MM:SS]** prefix."""
        formatter = MarkdownFormatter(include_timestamps=True)
        result = formatter.format(sample_transcription_result)
        assert "**[00:00]**" in result
        assert "**[00:01]**" in result

    def test_metadata_footer_language(
        self, sample_transcription_result: TranscriptionResult
    ) -> None:
        """Footer should include language and confidence."""
        formatter = MarkdownFormatter(include_timestamps=False)
        result = formatter.format(sample_transcription_result)
        assert "en" in result
        assert "95%" in result

    def test_metadata_footer_duration(
        self, sample_transcription_result: TranscriptionResult
    ) -> None:
        """Footer should include duration."""
        formatter = MarkdownFormatter(include_timestamps=False)
        result = formatter.format(sample_transcription_result)
        assert "3.0s" in result

    def test_time_format_mm_ss(self) -> None:
        """Time format should be MM:SS."""
        assert MarkdownFormatter._format_time(0.0) == "00:00"
        assert MarkdownFormatter._format_time(5.5) == "00:05"
        assert MarkdownFormatter._format_time(65.9) == "01:05"

    def test_separator_line_present(self, sample_transcription_result: TranscriptionResult) -> None:
        """Should have --- separator before metadata."""
        formatter = MarkdownFormatter(include_timestamps=False)
        result = formatter.format(sample_transcription_result)
        assert "---" in result


class TestSRTFormatter:
    """Tests for SRTFormatter class."""

    def test_sequence_numbers_start_at_one(
        self, sample_transcription_result: TranscriptionResult
    ) -> None:
        """Sequence numbers should start at 1."""
        formatter = SRTFormatter()
        result = formatter.format(sample_transcription_result)
        lines = result.split("\n")
        assert lines[0] == "1"

    def test_sequence_numbers_increment(
        self, sample_transcription_result: TranscriptionResult
    ) -> None:
        """Sequence numbers should increment for each segment."""
        formatter = SRTFormatter()
        result = formatter.format(sample_transcription_result)
        # Should have "1" and "2" as sequence numbers
        assert "\n1\n" in f"\n{result}" or result.startswith("1\n")
        assert "\n2\n" in result

    def test_time_format_hh_mm_ss_comma_mmm(
        self, sample_transcription_result: TranscriptionResult
    ) -> None:
        """Time format should be HH:MM:SS,mmm with comma for ms."""
        formatter = SRTFormatter()
        result = formatter.format(sample_transcription_result)
        # Should have format like "00:00:00,000 --> 00:00:01,500"
        assert "00:00:00,000 --> 00:00:01,500" in result

    def test_always_has_timestamps(self, sample_transcription_result: TranscriptionResult) -> None:
        """SRT always includes timestamps regardless of parameter."""
        # SRTFormatter doesn't take include_timestamps parameter
        formatter = SRTFormatter()
        result = formatter.format(sample_transcription_result)
        assert "-->" in result

    def test_blank_lines_between_entries(
        self, sample_transcription_result: TranscriptionResult
    ) -> None:
        """Should have blank lines between entries."""
        formatter = SRTFormatter()
        result = formatter.format(sample_transcription_result)
        # Between entries, there should be blank line
        assert "\n\n" in result

    def test_empty_segments_no_output(
        self, empty_transcription_result: TranscriptionResult
    ) -> None:
        """Empty segments should produce empty or minimal output."""
        formatter = SRTFormatter()
        result = formatter.format(empty_transcription_result)
        # Either empty string or just newlines
        assert result.strip() == "" or result == "\n"

    def test_format_srt_time_static_method(self) -> None:
        """_format_srt_time should format correctly."""
        assert SRTFormatter._format_srt_time(0.0) == "00:00:00,000"
        assert SRTFormatter._format_srt_time(5.5) == "00:00:05,500"
        assert SRTFormatter._format_srt_time(65.123) == "00:01:05,123"
        assert SRTFormatter._format_srt_time(3661.5) == "01:01:01,500"

    def test_long_duration_hours(self, long_transcription_result: TranscriptionResult) -> None:
        """Should handle durations > 1 hour."""
        formatter = SRTFormatter()
        result = formatter.format(long_transcription_result)
        # 3661.5 seconds = 01:01:01,500
        assert "01:01:01,500" in result


class TestGetFormatter:
    """Tests for get_formatter factory function."""

    def test_get_plain_formatter(self) -> None:
        """'plain' should return PlainFormatter."""
        formatter = get_formatter("plain")
        assert isinstance(formatter, PlainFormatter)

    def test_get_markdown_formatter(self) -> None:
        """'markdown' should return MarkdownFormatter."""
        formatter = get_formatter("markdown")
        assert isinstance(formatter, MarkdownFormatter)

    def test_get_srt_formatter(self) -> None:
        """'srt' should return SRTFormatter."""
        formatter = get_formatter("srt")
        assert isinstance(formatter, SRTFormatter)

    def test_invalid_format_raises(self) -> None:
        """Invalid format name should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            get_formatter("invalid")
        assert "invalid" in str(exc_info.value).lower()

    def test_timestamps_passed_to_plain(self) -> None:
        """include_timestamps should be passed to PlainFormatter."""
        formatter = get_formatter("plain", include_timestamps=True)
        assert isinstance(formatter, PlainFormatter)
        assert formatter._include_timestamps is True

    def test_timestamps_passed_to_markdown(self) -> None:
        """include_timestamps should be passed to MarkdownFormatter."""
        formatter = get_formatter("markdown", include_timestamps=True)
        assert isinstance(formatter, MarkdownFormatter)
        assert formatter._include_timestamps is True

    def test_srt_ignores_timestamps_param(self) -> None:
        """SRT formatter should be returned without timestamps param."""
        formatter = get_formatter("srt", include_timestamps=False)
        assert isinstance(formatter, SRTFormatter)
        # SRTFormatter always includes timestamps


class TestOutputFormatterBase:
    """Tests for OutputFormatter base class."""

    def test_is_abstract(self) -> None:
        """OutputFormatter should be abstract."""
        # Can't directly instantiate (would need to implement format)
        assert hasattr(OutputFormatter, "format")

    def test_all_formatters_are_subclass(self) -> None:
        """All formatters should be subclasses of OutputFormatter."""
        assert issubclass(PlainFormatter, OutputFormatter)
        assert issubclass(MarkdownFormatter, OutputFormatter)
        assert issubclass(SRTFormatter, OutputFormatter)


class TestEdgeCases:
    """Tests for edge cases in formatting."""

    def test_long_duration_over_one_hour(
        self, long_transcription_result: TranscriptionResult
    ) -> None:
        """Formatters should handle durations > 1 hour."""
        # Plain formatter with timestamps
        plain = PlainFormatter(include_timestamps=True)
        result = plain.format(long_transcription_result)
        # Should have time like 61:01.500 for plain format
        assert "61:01.500" in result

    def test_zero_duration_segment(self) -> None:
        """Should handle segment with 0 duration."""
        result = TranscriptionResult(
            text="Word",
            segments=[
                TranscriptionSegment(start=0.0, end=0.0, text="Word", words=[]),
            ],
            language="en",
            language_probability=0.9,
            duration=0.0,
        )
        formatter = PlainFormatter(include_timestamps=True)
        output = formatter.format(result)
        assert "[00:00.000 --> 00:00.000]" in output

    def test_special_characters_in_text(self) -> None:
        """Should handle special characters (quotes, newlines)."""
        result = TranscriptionResult(
            text='He said "Hello"\nNew line here',
            segments=[
                TranscriptionSegment(
                    start=0.0,
                    end=1.0,
                    text='He said "Hello"\nNew line here',
                    words=[],
                ),
            ],
            language="en",
            language_probability=0.9,
            duration=1.0,
        )
        formatter = PlainFormatter(include_timestamps=False)
        output = formatter.format(result)
        assert '"Hello"' in output
        assert "\n" in output

    def test_language_probability_percentage(self) -> None:
        """Language probability should be shown as percentage in markdown."""
        result = TranscriptionResult(
            text="Test",
            segments=[],
            language="en",
            language_probability=0.95,
            duration=1.0,
        )
        formatter = MarkdownFormatter(include_timestamps=False)
        output = formatter.format(result)
        assert "95%" in output

    def test_language_probability_zero(self) -> None:
        """Should handle 0% language probability."""
        result = TranscriptionResult(
            text="Test",
            segments=[],
            language="unknown",
            language_probability=0.0,
            duration=1.0,
        )
        formatter = MarkdownFormatter(include_timestamps=False)
        output = formatter.format(result)
        assert "0%" in output

    def test_language_probability_full(self) -> None:
        """Should handle 100% language probability."""
        result = TranscriptionResult(
            text="Test",
            segments=[],
            language="en",
            language_probability=1.0,
            duration=1.0,
        )
        formatter = MarkdownFormatter(include_timestamps=False)
        output = formatter.format(result)
        assert "100%" in output

    def test_language_probability_unknown(self) -> None:
        """Should handle unknown language probability (from diarization backends)."""
        result = TranscriptionResult(
            text="Test",
            segments=[],
            language="en",
            language_probability=UNKNOWN_LANGUAGE_PROBABILITY,
            duration=1.0,
        )
        formatter = MarkdownFormatter(include_timestamps=False)
        output = formatter.format(result)
        # Should show language without confidence percentage
        assert "Language: en" in output
        assert "confidence" not in output

    def test_unicode_text(self) -> None:
        """Should handle unicode text correctly."""
        result = TranscriptionResult(
            text="Bonjour monde",
            segments=[
                TranscriptionSegment(start=0.0, end=1.0, text="Bonjour monde", words=[]),
            ],
            language="fr",
            language_probability=0.9,
            duration=1.0,
        )
        formatter = PlainFormatter(include_timestamps=False)
        output = formatter.format(result)
        assert output == "Bonjour monde"

    def test_very_long_text(self) -> None:
        """Should handle very long text without truncation."""
        long_text = "word " * 1000
        result = TranscriptionResult(
            text=long_text.strip(),
            segments=[
                TranscriptionSegment(start=0.0, end=600.0, text=long_text.strip(), words=[]),
            ],
            language="en",
            language_probability=0.9,
            duration=600.0,
        )
        formatter = PlainFormatter(include_timestamps=False)
        output = formatter.format(result)
        assert len(output) == len(long_text.strip())


class TestTypeDetection:
    """Tests for result type detection using isinstance."""

    def test_transcription_result_not_diarization(
        self, sample_transcription_result: TranscriptionResult
    ) -> None:
        """TranscriptionResult should not be detected as diarization."""
        formatter = PlainFormatter()
        assert formatter._is_diarization_result(sample_transcription_result) is False

    def test_diarization_result_is_diarization(
        self, sample_diarization_result: DiarizationResult
    ) -> None:
        """DiarizationResult should be detected as diarization."""
        formatter = PlainFormatter()
        assert formatter._is_diarization_result(sample_diarization_result) is True

    def test_isinstance_check_not_hasattr(
        self, sample_transcription_result: TranscriptionResult
    ) -> None:
        """Type detection should use isinstance, not hasattr."""
        # Even if TranscriptionResult had a 'speakers' attribute, it should still
        # be identified correctly by type, not by duck-typing
        formatter = PlainFormatter()
        # Verify the method uses isinstance (implementation detail check)
        # The actual test is that TranscriptionResult is not detected as diarization
        assert formatter._is_diarization_result(sample_transcription_result) is False

    def test_markdown_formatter_type_detection(
        self,
        sample_transcription_result: TranscriptionResult,
        sample_diarization_result: DiarizationResult,
    ) -> None:
        """MarkdownFormatter should correctly detect result types."""
        formatter = MarkdownFormatter()
        assert formatter._is_diarization_result(sample_transcription_result) is False
        assert formatter._is_diarization_result(sample_diarization_result) is True

    def test_srt_formatter_type_detection(
        self,
        sample_transcription_result: TranscriptionResult,
        sample_diarization_result: DiarizationResult,
    ) -> None:
        """SRTFormatter should correctly detect result types."""
        formatter = SRTFormatter()
        assert formatter._is_diarization_result(sample_transcription_result) is False
        assert formatter._is_diarization_result(sample_diarization_result) is True


class TestDiarizationFormatting:
    """Tests for formatting DiarizationResult."""

    def test_plain_formatter_diarization(
        self, sample_diarization_result: DiarizationResult
    ) -> None:
        """PlainFormatter should format diarization with speaker labels."""
        formatter = PlainFormatter(include_timestamps=False)
        output = formatter.format(sample_diarization_result)
        # Should have speaker labels
        assert "[SPEAKER_01]" in output
        assert "[SPEAKER_02]" in output
        # Should have text from fixture
        assert "Hello from speaker one" in output
        assert "Response from speaker two" in output

    def test_plain_formatter_diarization_with_timestamps(
        self, sample_diarization_result: DiarizationResult
    ) -> None:
        """PlainFormatter should include timestamps when requested."""
        formatter = PlainFormatter(include_timestamps=True)
        output = formatter.format(sample_diarization_result)
        # Should have timestamps in HH:MM:SS format
        assert "[00:00]" in output
        assert "[SPEAKER_01]" in output

    def test_markdown_formatter_diarization(
        self, sample_diarization_result: DiarizationResult
    ) -> None:
        """MarkdownFormatter should format diarization with meeting transcript header."""
        formatter = MarkdownFormatter(include_timestamps=False)
        output = formatter.format(sample_diarization_result)
        # Should have meeting transcript header
        assert "# Meeting Transcript" in output
        # Should have speaker labels in bold
        assert "**SPEAKER_01**" in output
        assert "**SPEAKER_02**" in output
        # Should have speaker count in footer
        assert "2 speakers detected" in output

    def test_srt_formatter_diarization(self, sample_diarization_result: DiarizationResult) -> None:
        """SRTFormatter should format diarization with speaker labels in subtitles."""
        formatter = SRTFormatter()
        output = formatter.format(sample_diarization_result)
        # Should have sequence numbers
        assert "1\n" in output
        # Should have timestamps
        assert "-->" in output
        # Should have speaker labels
        assert "[SPEAKER_01]" in output
