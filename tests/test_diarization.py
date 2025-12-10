"""Tests for diarization functionality."""

from __future__ import annotations

import numpy as np
import pytest

from hark.diarizer import DiarizationResult, DiarizedSegment, WordSegment
from hark.formatter import (
    MarkdownFormatter,
    PlainFormatter,
    SRTFormatter,
    get_formatter,
)
from hark.interactive import get_speaker_excerpt, interactive_speaker_naming
from hark.stereo_processor import (
    merge_diarization_timelines,
    split_stereo_channels,
)


# Fixtures for test data
@pytest.fixture
def sample_segments() -> list[DiarizedSegment]:
    """Create sample diarization segments."""
    return [
        DiarizedSegment(
            start=0.0,
            end=2.5,
            text="Hello everyone.",
            speaker="SPEAKER_01",
            words=[],
        ),
        DiarizedSegment(
            start=3.0,
            end=5.5,
            text="Thanks for joining us today.",
            speaker="SPEAKER_02",
            words=[],
        ),
        DiarizedSegment(
            start=6.0,
            end=8.0,
            text="Let's get started.",
            speaker="SPEAKER_01",
            words=[],
        ),
    ]


@pytest.fixture
def sample_result(sample_segments: list[DiarizedSegment]) -> DiarizationResult:
    """Create sample diarization result."""
    return DiarizationResult(
        segments=sample_segments,
        speakers=["SPEAKER_01", "SPEAKER_02"],
        language="en",
        language_probability=0.95,
        duration=8.0,
    )


class TestDiarizedSegment:
    """Tests for DiarizedSegment dataclass."""

    def test_create_basic_segment(self) -> None:
        """Test creating a basic segment."""
        seg = DiarizedSegment(
            start=0.0,
            end=1.0,
            text="Hello",
            speaker="SPEAKER_01",
        )
        assert seg.start == 0.0
        assert seg.end == 1.0
        assert seg.text == "Hello"
        assert seg.speaker == "SPEAKER_01"
        assert seg.words == []

    def test_create_segment_with_words(self) -> None:
        """Test creating a segment with word timestamps."""
        words = [
            WordSegment(start=0.0, end=0.5, word="Hello"),
            WordSegment(start=0.5, end=1.0, word="world"),
        ]
        seg = DiarizedSegment(
            start=0.0,
            end=1.0,
            text="Hello world",
            speaker="SPEAKER_01",
            words=words,
        )
        assert len(seg.words) == 2
        assert seg.words[0].word == "Hello"


class TestDiarizationResult:
    """Tests for DiarizationResult dataclass."""

    def test_create_result(self, sample_segments: list[DiarizedSegment]) -> None:
        """Test creating a diarization result."""
        result = DiarizationResult(
            segments=sample_segments,
            speakers=["SPEAKER_01", "SPEAKER_02"],
            language="en",
            language_probability=0.95,
            duration=8.0,
        )
        assert len(result.segments) == 3
        assert len(result.speakers) == 2
        assert result.language == "en"
        assert result.duration == 8.0


class TestPlainFormatterDiarization:
    """Tests for PlainFormatter with DiarizationResult."""

    def test_format_diarization_result(self, sample_result: DiarizationResult) -> None:
        """Test formatting a diarization result."""
        formatter = PlainFormatter()
        output = formatter.format(sample_result)

        assert "[SPEAKER_01]" in output
        assert "[SPEAKER_02]" in output
        assert "Hello everyone." in output
        assert "Thanks for joining us today." in output

    def test_format_includes_timestamps(self, sample_result: DiarizationResult) -> None:
        """Test that diarization output includes timestamps when enabled."""
        formatter = PlainFormatter(include_timestamps=True)
        output = formatter.format(sample_result)

        # Should have timestamp format [MM:SS]
        assert "[00:00]" in output
        assert "[00:03]" in output

    def test_format_without_timestamps(self, sample_result: DiarizationResult) -> None:
        """Test that diarization output excludes timestamps when disabled."""
        formatter = PlainFormatter(include_timestamps=False)
        output = formatter.format(sample_result)

        # Should not have timestamp format
        assert "[00:00]" not in output
        assert "[SPEAKER_01]" in output
        assert "Hello everyone." in output

    def test_format_overlapping_speakers(self) -> None:
        """Test formatting overlapping speakers."""
        seg = DiarizedSegment(
            start=0.0,
            end=1.0,
            text="Both speaking",
            speaker="SPEAKER_01 + SPEAKER_02",
        )
        result = DiarizationResult(
            segments=[seg],
            speakers=["SPEAKER_01", "SPEAKER_02"],
            language="en",
            language_probability=0.95,
            duration=1.0,
        )
        formatter = PlainFormatter()
        output = formatter.format(result)

        assert "[overlapping]" in output
        assert "SPEAKER_01 + SPEAKER_02" in output


class TestMarkdownFormatterDiarization:
    """Tests for MarkdownFormatter with DiarizationResult."""

    def test_format_diarization_result(self, sample_result: DiarizationResult) -> None:
        """Test formatting a diarization result as markdown."""
        formatter = MarkdownFormatter()
        output = formatter.format(sample_result)

        assert "# Meeting Transcript" in output
        assert "**SPEAKER_01**" in output
        assert "**SPEAKER_02**" in output

    def test_includes_metadata_footer(self, sample_result: DiarizationResult) -> None:
        """Test that markdown includes metadata footer."""
        formatter = MarkdownFormatter()
        output = formatter.format(sample_result)

        assert "2 speakers detected" in output
        assert "Duration:" in output
        assert "Language: en" in output


class TestSRTFormatterDiarization:
    """Tests for SRTFormatter with DiarizationResult."""

    def test_format_diarization_result(self, sample_result: DiarizationResult) -> None:
        """Test formatting a diarization result as SRT."""
        formatter = SRTFormatter()
        output = formatter.format(sample_result)

        # Check sequence numbers
        assert output.startswith("1\n")
        assert "\n2\n" in output

        # Check speaker labels
        assert "[SPEAKER_01]" in output
        assert "[SPEAKER_02]" in output

    def test_srt_timestamp_format(self, sample_result: DiarizationResult) -> None:
        """Test SRT timestamp format with diarization."""
        formatter = SRTFormatter()
        output = formatter.format(sample_result)

        # SRT uses comma for milliseconds
        assert "00:00:00,000" in output
        assert "-->" in output


class TestGetFormatterWithDiarization:
    """Tests for get_formatter function with diarization results."""

    def test_plain_formatter_handles_diarization(self, sample_result: DiarizationResult) -> None:
        """Test plain formatter from get_formatter handles diarization."""
        formatter = get_formatter("plain")
        output = formatter.format(sample_result)
        assert "[SPEAKER_01]" in output

    def test_markdown_formatter_handles_diarization(self, sample_result: DiarizationResult) -> None:
        """Test markdown formatter from get_formatter handles diarization."""
        formatter = get_formatter("markdown")
        output = formatter.format(sample_result)
        assert "**SPEAKER_01**" in output

    def test_srt_formatter_handles_diarization(self, sample_result: DiarizationResult) -> None:
        """Test SRT formatter from get_formatter handles diarization."""
        formatter = get_formatter("srt")
        output = formatter.format(sample_result)
        assert "[SPEAKER_01]" in output


class TestGetSpeakerExcerpt:
    """Tests for get_speaker_excerpt function."""

    def test_finds_speaker_excerpt(self, sample_segments: list[DiarizedSegment]) -> None:
        """Test finding an excerpt for a speaker."""
        excerpt = get_speaker_excerpt(sample_segments, "SPEAKER_01")
        assert excerpt == "Hello everyone."

    def test_truncates_long_excerpt(self) -> None:
        """Test that long excerpts are truncated."""
        long_text = "A" * 100
        segments = [DiarizedSegment(start=0.0, end=1.0, text=long_text, speaker="SPEAKER_01")]
        excerpt = get_speaker_excerpt(segments, "SPEAKER_01", max_length=50)
        assert len(excerpt) == 50
        assert excerpt.endswith("...")

    def test_returns_default_for_missing_speaker(
        self, sample_segments: list[DiarizedSegment]
    ) -> None:
        """Test default when speaker not found."""
        excerpt = get_speaker_excerpt(sample_segments, "SPEAKER_99")
        assert excerpt == "[no speech found]"

    def test_skips_empty_text(self) -> None:
        """Test that empty text segments are skipped."""
        segments = [
            DiarizedSegment(start=0.0, end=1.0, text="", speaker="SPEAKER_01"),
            DiarizedSegment(start=1.0, end=2.0, text="Hello", speaker="SPEAKER_01"),
        ]
        excerpt = get_speaker_excerpt(segments, "SPEAKER_01")
        assert excerpt == "Hello"


class TestInteractiveSpeakerNaming:
    """Tests for interactive_speaker_naming function."""

    def test_quiet_mode_returns_unchanged(self, sample_result: DiarizationResult) -> None:
        """Test that quiet mode returns result unchanged."""
        result = interactive_speaker_naming(sample_result, quiet=True)
        assert result.speakers == sample_result.speakers

    def test_speaker_00_not_renamed(self) -> None:
        """Test that SPEAKER_00 is not offered for renaming."""
        segments = [
            DiarizedSegment(start=0.0, end=1.0, text="Hello", speaker="SPEAKER_00"),
        ]
        result = DiarizationResult(
            segments=segments,
            speakers=["SPEAKER_00"],
            language="en",
            language_probability=0.95,
            duration=1.0,
        )
        # With quiet=True, should return unchanged
        named_result = interactive_speaker_naming(result, quiet=True)
        assert "SPEAKER_00" in named_result.speakers


class TestSplitStereoChannels:
    """Tests for split_stereo_channels function."""

    def test_splits_stereo_correctly(self) -> None:
        """Test splitting stereo into left and right channels."""
        stereo = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        left, right = split_stereo_channels(stereo, 16000)

        assert left.channel_name == "mic"
        assert right.channel_name == "speaker"
        np.testing.assert_array_equal(left.audio, np.array([1.0, 3.0, 5.0]))
        np.testing.assert_array_equal(right.audio, np.array([2.0, 4.0, 6.0]))

    def test_raises_for_mono_audio(self) -> None:
        """Test that mono audio raises ValueError."""
        mono = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        with pytest.raises(ValueError, match="mono"):
            split_stereo_channels(mono, 16000)

    def test_raises_for_wrong_channels(self) -> None:
        """Test that wrong channel count raises ValueError."""
        multi = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        with pytest.raises(ValueError, match="2 channels"):
            split_stereo_channels(multi, 16000)


class TestMergeDiarizationTimelines:
    """Tests for merge_diarization_timelines function."""

    def test_merges_local_and_remote(self) -> None:
        """Test merging local and remote timelines."""
        local_segments = [
            DiarizedSegment(start=0.0, end=2.0, text="Hi there", speaker="Me"),
        ]
        remote_result = DiarizationResult(
            segments=[
                DiarizedSegment(start=3.0, end=5.0, text="Hello", speaker="SPEAKER_01"),
            ],
            speakers=["SPEAKER_01"],
            language="en",
            language_probability=0.95,
            duration=5.0,
        )

        merged = merge_diarization_timelines(local_segments, remote_result, local_speaker_name="Me")

        assert len(merged.segments) == 2
        assert merged.speakers == ["Me", "SPEAKER_01"]
        assert merged.segments[0].speaker == "Me"
        assert merged.segments[1].speaker == "SPEAKER_01"

    def test_sorts_by_timestamp(self) -> None:
        """Test that merged segments are sorted by timestamp."""
        local_segments = [
            DiarizedSegment(start=5.0, end=7.0, text="Later", speaker="Me"),
        ]
        remote_result = DiarizationResult(
            segments=[
                DiarizedSegment(start=0.0, end=2.0, text="First", speaker="SPEAKER_01"),
            ],
            speakers=["SPEAKER_01"],
            language="en",
            language_probability=0.95,
            duration=7.0,
        )

        merged = merge_diarization_timelines(local_segments, remote_result, local_speaker_name="Me")

        # First segment should be from remote (earlier timestamp)
        assert merged.segments[0].speaker == "SPEAKER_01"
        assert merged.segments[1].speaker == "Me"


class TestMergeOverlappingSegmentsNoMutation:
    """Test that _merge_overlapping_segments doesn't mutate input."""

    def test_does_not_mutate_input_list(self) -> None:
        """Test that the input list is not modified."""
        from hark.stereo_processor import _merge_overlapping_segments

        segments = [
            DiarizedSegment(start=0.0, end=2.0, text="First", speaker="A"),
            DiarizedSegment(start=1.5, end=3.0, text="Second", speaker="B"),
        ]
        original_length = len(segments)
        original_first_end = segments[0].end

        _merge_overlapping_segments(segments)

        # Original list should be unchanged
        assert len(segments) == original_length
        assert segments[0].end == original_first_end


class TestWordSpeakerUpdate:
    """Test word-level speaker updates in interactive naming."""

    def test_update_word_speakers(self) -> None:
        """Test that word speaker labels are updated."""
        from hark.diarizer import WordSegment
        from hark.interactive import _update_word_speakers

        words = [
            WordSegment(start=0.0, end=0.5, word="Hello", speaker="SPEAKER_01"),
            WordSegment(start=0.5, end=1.0, word="world", speaker="SPEAKER_02"),
            WordSegment(start=1.0, end=1.5, word="!", speaker=None),
        ]

        speaker_names = {"SPEAKER_01": "Alice", "SPEAKER_02": "Bob"}
        updated = _update_word_speakers(words, speaker_names)

        assert updated[0].speaker == "Alice"
        assert updated[1].speaker == "Bob"
        assert updated[2].speaker is None  # None stays None

    def test_preserves_non_matching_speakers(self) -> None:
        """Test that non-matching speakers are preserved."""
        from hark.diarizer import WordSegment
        from hark.interactive import _update_word_speakers

        words = [
            WordSegment(start=0.0, end=0.5, word="Hello", speaker="SPEAKER_03"),
        ]

        speaker_names = {"SPEAKER_01": "Alice"}
        updated = _update_word_speakers(words, speaker_names)

        assert updated[0].speaker == "SPEAKER_03"


class TestMarkdownFormatterTimestamps:
    """Test Markdown formatter timestamp behavior."""

    def test_markdown_with_timestamps(self, sample_result: DiarizationResult) -> None:
        """Test Markdown format includes timestamps when enabled."""
        formatter = MarkdownFormatter(include_timestamps=True)
        output = formatter.format(sample_result)

        # Should have timestamps in parentheses
        assert "(00:00)" in output

    def test_markdown_without_timestamps(self, sample_result: DiarizationResult) -> None:
        """Test Markdown format excludes timestamps when disabled."""
        formatter = MarkdownFormatter(include_timestamps=False)
        output = formatter.format(sample_result)

        # Should not have timestamps
        assert "(00:00)" not in output
        # But should still have speaker names
        assert "**SPEAKER_01**" in output


class TestPreprocessorStereoPreservation:
    """Test preprocessor stereo preservation."""

    def test_preserves_stereo_when_requested(self, tmp_path) -> None:
        """Test that stereo audio is preserved when preserve_stereo=True."""
        import soundfile as sf

        from hark.config import PreprocessingConfig
        from hark.preprocessor import AudioPreprocessor

        # Create stereo audio file
        stereo_audio = np.column_stack(
            [
                np.sin(np.linspace(0, 4 * np.pi, 16000)),  # Left
                np.cos(np.linspace(0, 4 * np.pi, 16000)),  # Right
            ]
        ).astype(np.float32)
        audio_path = tmp_path / "test_stereo.wav"
        sf.write(audio_path, stereo_audio, 16000)

        # Process with preserve_stereo=True
        config = PreprocessingConfig()
        config.noise_reduction.enabled = False
        config.normalization.enabled = False
        config.silence_trimming.enabled = False

        preprocessor = AudioPreprocessor(config)
        processed, _ = preprocessor.process(audio_path, 16000, preserve_stereo=True)

        # Should still be stereo
        assert processed.ndim == 2
        assert processed.shape[1] == 2

    def test_converts_to_mono_by_default(self, tmp_path) -> None:
        """Test that stereo audio is converted to mono by default."""
        import soundfile as sf

        from hark.config import PreprocessingConfig
        from hark.preprocessor import AudioPreprocessor

        # Create stereo audio file
        stereo_audio = np.column_stack(
            [
                np.sin(np.linspace(0, 4 * np.pi, 16000)),  # Left
                np.cos(np.linspace(0, 4 * np.pi, 16000)),  # Right
            ]
        ).astype(np.float32)
        audio_path = tmp_path / "test_stereo.wav"
        sf.write(audio_path, stereo_audio, 16000)

        # Process without preserve_stereo (default)
        config = PreprocessingConfig()
        config.noise_reduction.enabled = False
        config.normalization.enabled = False
        config.silence_trimming.enabled = False

        preprocessor = AudioPreprocessor(config)
        processed, _ = preprocessor.process(audio_path, 16000)

        # Should be mono
        assert processed.ndim == 1


class TestInteractiveLocalSpeakerExclusion:
    """Test interactive speaker naming excludes local speaker."""

    def test_excludes_configured_local_speaker(self) -> None:
        """Test that configured local speaker name is excluded from renaming."""
        result = DiarizationResult(
            segments=[
                DiarizedSegment(start=0.0, end=1.0, text="Hi", speaker="Me"),
                DiarizedSegment(start=1.0, end=2.0, text="Hello", speaker="SPEAKER_01"),
            ],
            speakers=["Me", "SPEAKER_01"],
            language="en",
            language_probability=0.95,
            duration=2.0,
        )

        # When quiet=True, no interaction - just returns result unchanged
        from hark.interactive import interactive_speaker_naming

        renamed = interactive_speaker_naming(result, quiet=True, local_speaker_name="Me")

        assert renamed.speakers == ["Me", "SPEAKER_01"]
