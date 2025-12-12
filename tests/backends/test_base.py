"""Tests for backend data classes."""

from hark.backends.base import (
    DiarizationOutput,
    DiarizedSegment,
    TranscriptionOutput,
    TranscriptionSegment,
    WordInfo,
)


class TestBackendDataClasses:
    """Tests for backend data classes."""

    def test_word_info_creation(self) -> None:
        """WordInfo should be created with all fields."""
        word = WordInfo(start=0.0, end=0.5, word="Hello", probability=0.98)
        assert word.start == 0.0
        assert word.end == 0.5
        assert word.word == "Hello"
        assert word.probability == 0.98

    def test_word_info_default_probability(self) -> None:
        """WordInfo should default probability to 1.0."""
        word = WordInfo(start=0.0, end=0.5, word="Hello")
        assert word.probability == 1.0

    def test_transcription_segment_creation(self) -> None:
        """TranscriptionSegment should be created with all fields."""
        words = [WordInfo(start=0.0, end=0.5, word="Hello")]
        segment = TranscriptionSegment(start=0.0, end=1.0, text="Hello world", words=words)
        assert segment.start == 0.0
        assert segment.end == 1.0
        assert segment.text == "Hello world"
        assert len(segment.words) == 1

    def test_transcription_output_text_property(self) -> None:
        """TranscriptionOutput.text should concatenate segment texts."""
        segments = [
            TranscriptionSegment(start=0.0, end=1.0, text="Hello"),
            TranscriptionSegment(start=1.0, end=2.0, text="world"),
        ]
        output = TranscriptionOutput(
            segments=segments,
            language="en",
            language_probability=0.95,
            duration=2.0,
        )
        assert output.text == "Hello world"

    def test_diarized_segment_creation(self) -> None:
        """DiarizedSegment should be created with speaker field."""
        segment = DiarizedSegment(start=0.0, end=1.0, text="Hello", speaker="SPEAKER_01", words=[])
        assert segment.speaker == "SPEAKER_01"

    def test_diarization_output_creation(self) -> None:
        """DiarizationOutput should be created with speakers list."""
        segments = [
            DiarizedSegment(start=0.0, end=1.0, text="Hello", speaker="SPEAKER_01", words=[]),
        ]
        output = DiarizationOutput(
            segments=segments,
            speakers=["SPEAKER_01"],
            language="en",
            language_probability=0.95,
            duration=1.0,
        )
        assert output.speakers == ["SPEAKER_01"]
        assert len(output.segments) == 1
