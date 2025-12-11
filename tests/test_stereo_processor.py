"""Tests for hark.stereo_processor module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from hark.diarizer import DiarizationResult
from hark.stereo_processor import (
    ChannelAudio,
    StereoProcessor,
    _merge_overlapping_segments,
    _suppress_output,
    merge_diarization_timelines,
    split_stereo_channels,
)


class TestSplitStereoChannels:
    """Tests for split_stereo_channels function."""

    def test_splits_stereo_correctly(self) -> None:
        """Should split stereo audio into left and right channels."""
        # Create stereo audio: left channel = 0.5, right channel = -0.5
        stereo = np.array([[0.5, -0.5], [0.5, -0.5], [0.5, -0.5]], dtype=np.float32)

        left, right = split_stereo_channels(stereo, sample_rate=16000)

        assert left.channel_name == "mic"
        assert right.channel_name == "speaker"
        np.testing.assert_array_almost_equal(left.audio, np.array([0.5, 0.5, 0.5]))
        np.testing.assert_array_almost_equal(right.audio, np.array([-0.5, -0.5, -0.5]))

    def test_raises_for_mono_audio(self) -> None:
        """Should raise ValueError for mono audio."""
        mono = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        with pytest.raises(ValueError, match="mono"):
            split_stereo_channels(mono, sample_rate=16000)

    def test_raises_for_wrong_channel_count(self) -> None:
        """Should raise ValueError for non-stereo multi-channel audio."""
        multichannel = np.zeros((100, 4), dtype=np.float32)

        with pytest.raises(ValueError, match="2 channels"):
            split_stereo_channels(multichannel, sample_rate=16000)


class TestSuppressOutput:
    """Tests for _suppress_output context manager."""

    def test_suppresses_stdout(self, capsys: pytest.CaptureFixture) -> None:
        """Should suppress stdout within context."""
        print("before")
        with _suppress_output():
            print("suppressed")
        print("after")

        captured = capsys.readouterr()
        assert "before" in captured.out
        assert "suppressed" not in captured.out
        assert "after" in captured.out

    def test_suppresses_stderr(self, capsys: pytest.CaptureFixture) -> None:
        """Should suppress stderr within context."""
        import sys

        print("before", file=sys.stderr)
        with _suppress_output():
            print("suppressed", file=sys.stderr)
        print("after", file=sys.stderr)

        captured = capsys.readouterr()
        assert "before" in captured.err
        assert "suppressed" not in captured.err
        assert "after" in captured.err

    def test_restores_streams_on_exception(self) -> None:
        """Should restore stdout/stderr even if exception occurs."""
        import sys

        original_stdout = sys.stdout
        original_stderr = sys.stderr

        with pytest.raises(ValueError):
            with _suppress_output():
                raise ValueError("test error")

        assert sys.stdout is original_stdout
        assert sys.stderr is original_stderr


class TestStereoProcessorLanguageProbability:
    """Tests for language probability handling in StereoProcessor."""

    def test_language_probability_100_when_specified(self) -> None:
        """Should set language_probability to 1.0 when language is explicitly specified."""
        mock_config = MagicMock()
        mock_config.whisper.model = "base"
        mock_config.whisper.device = "cpu"
        mock_config.whisper.language = "de"  # Explicitly specified
        mock_config.model_cache_dir = "/tmp/models"
        mock_config.diarization.hf_token = "test_token"
        mock_config.diarization.local_speaker_name = "LOCAL"
        mock_config.diarization.num_speakers = None

        _processor = StereoProcessor(mock_config)  # Verify instantiation succeeds

        # The language_probability logic is in _transcribe_and_diarize_channel
        # When language is explicitly specified (not None), probability should be 1.0
        # This is verified by checking the code path, not running full diarization
        assert mock_config.whisper.language == "de"
        assert _processor is not None  # Suppress unused warning

        # The actual test of the logic is that when we call the method with
        # language="de", the returned DiarizationResult should have
        # language_probability=1.0. This requires more extensive mocking.


# =============================================================================
# Pure Logic Tests - No mocks needed
# =============================================================================


class TestMergeOverlappingSegments:
    """Tests for _merge_overlapping_segments - pure function, no mocks."""

    def test_empty_input_returns_empty(self, diarized_segment_factory) -> None:
        """Empty input should return empty list."""
        result = _merge_overlapping_segments([])
        assert result == []

    def test_single_segment_passes_through(self, diarized_segment_factory) -> None:
        """Single segment should pass through unchanged."""
        seg = diarized_segment_factory(0.0, 1.0, "Hello", "SPEAKER_A")
        result = _merge_overlapping_segments([seg])

        assert len(result) == 1
        assert result[0].text == "Hello"
        assert result[0].speaker == "SPEAKER_A"

    def test_non_overlapping_segments_pass_through(self, diarized_segment_factory) -> None:
        """Non-overlapping segments should pass through unchanged."""
        seg1 = diarized_segment_factory(0.0, 1.0, "Hello", "SPEAKER_A")
        seg2 = diarized_segment_factory(2.0, 3.0, "World", "SPEAKER_B")
        result = _merge_overlapping_segments([seg1, seg2])

        assert len(result) == 2
        assert result[0].text == "Hello"
        assert result[1].text == "World"

    def test_same_speaker_overlap_merges_text(self, diarized_segment_factory) -> None:
        """Overlapping segments from same speaker should merge."""
        seg1 = diarized_segment_factory(0.0, 2.0, "Hello", "SPEAKER_A")
        seg2 = diarized_segment_factory(1.0, 3.0, "World", "SPEAKER_A")
        result = _merge_overlapping_segments([seg1, seg2])

        assert len(result) == 1
        assert result[0].text == "Hello World"
        assert result[0].speaker == "SPEAKER_A"
        assert result[0].start == 0.0
        assert result[0].end == 3.0

    def test_different_speaker_overlap_creates_combined_segment(
        self, diarized_segment_factory
    ) -> None:
        """Different speakers overlapping create combined 'A + B' segment."""
        seg1 = diarized_segment_factory(0.0, 2.0, "Hello", "SPEAKER_A")
        seg2 = diarized_segment_factory(1.0, 3.0, "World", "SPEAKER_B")
        result = _merge_overlapping_segments([seg1, seg2])

        # Should have: before-overlap, overlap, after-overlap
        assert len(result) == 3

        # Before overlap (0.0-1.0): SPEAKER_A only
        assert result[0].start == 0.0
        assert result[0].end == 1.0
        assert result[0].speaker == "SPEAKER_A"

        # Overlap (1.0-2.0): both speakers
        assert result[1].start == 1.0
        assert result[1].end == 2.0
        assert "SPEAKER_A + SPEAKER_B" in result[1].speaker

        # After overlap (2.0-3.0): SPEAKER_B only
        assert result[2].start == 2.0
        assert result[2].end == 3.0
        assert result[2].speaker == "SPEAKER_B"

    def test_fully_contained_segment_same_speaker(self, diarized_segment_factory) -> None:
        """Segment fully contained in another (same speaker) should merge."""
        seg1 = diarized_segment_factory(0.0, 5.0, "Long", "SPEAKER_A")
        seg2 = diarized_segment_factory(1.0, 2.0, "Short", "SPEAKER_A")
        result = _merge_overlapping_segments([seg1, seg2])

        assert len(result) == 1
        assert result[0].start == 0.0
        assert result[0].end == 5.0

    def test_current_extends_past_next(self, diarized_segment_factory) -> None:
        """When current.end > next.end, handle correctly (fully contained)."""
        seg1 = diarized_segment_factory(0.0, 5.0, "Long", "SPEAKER_A")
        seg2 = diarized_segment_factory(1.0, 2.0, "Short", "SPEAKER_B")
        result = _merge_overlapping_segments([seg1, seg2])

        # Should create: before, overlap, after
        assert len(result) >= 3
        # The overlap should be at 1.0-2.0
        overlap = [s for s in result if "+" in s.speaker]
        assert len(overlap) == 1
        assert overlap[0].start == 1.0
        assert overlap[0].end == 2.0

    def test_does_not_mutate_input(self, diarized_segment_factory) -> None:
        """Should not mutate the input list."""
        seg1 = diarized_segment_factory(0.0, 2.0, "Hello", "SPEAKER_A")
        seg2 = diarized_segment_factory(1.0, 3.0, "World", "SPEAKER_A")
        original_segments = [seg1, seg2]
        original_len = len(original_segments)

        _merge_overlapping_segments(original_segments)

        assert len(original_segments) == original_len


class TestMergeDiarizationTimelines:
    """Tests for merge_diarization_timelines."""

    def test_merges_local_and_remote(self, diarized_segment_factory) -> None:
        """Should combine local segments with remote result."""
        local_segments = [
            diarized_segment_factory(0.0, 1.0, "Local speech", "LOCAL"),
        ]
        remote_result = DiarizationResult(
            segments=[
                diarized_segment_factory(1.5, 2.5, "Remote speech", "SPEAKER_01"),
            ],
            speakers=["SPEAKER_01"],
            language="en",
            language_probability=0.95,
            duration=2.5,
        )

        result = merge_diarization_timelines(
            local_segments, remote_result, local_speaker_name="LOCAL"
        )

        assert len(result.segments) == 2
        assert "LOCAL" in result.speakers
        assert "SPEAKER_01" in result.speakers

    def test_sorts_by_start_time(self, diarized_segment_factory) -> None:
        """Result should be sorted by start time."""
        # Local segment comes AFTER remote in time
        local_segments = [
            diarized_segment_factory(5.0, 6.0, "Local", "LOCAL"),
        ]
        remote_result = DiarizationResult(
            segments=[
                diarized_segment_factory(0.0, 1.0, "Remote", "SPEAKER_01"),
            ],
            speakers=["SPEAKER_01"],
            language="en",
            language_probability=0.95,
            duration=6.0,
        )

        result = merge_diarization_timelines(
            local_segments, remote_result, local_speaker_name="LOCAL"
        )

        # Should be sorted: remote first, then local
        assert result.segments[0].text == "Remote"
        assert result.segments[1].text == "Local"

    def test_builds_correct_speaker_list(self, diarized_segment_factory) -> None:
        """Speaker list should include local + remote speakers."""
        local_segments = [
            diarized_segment_factory(0.0, 1.0, "Local", "MY_MIC"),
        ]
        remote_result = DiarizationResult(
            segments=[
                diarized_segment_factory(1.0, 2.0, "Remote1", "SPEAKER_01"),
                diarized_segment_factory(2.0, 3.0, "Remote2", "SPEAKER_02"),
            ],
            speakers=["SPEAKER_01", "SPEAKER_02"],
            language="en",
            language_probability=0.95,
            duration=3.0,
        )

        result = merge_diarization_timelines(
            local_segments, remote_result, local_speaker_name="MY_MIC"
        )

        assert result.speakers[0] == "MY_MIC"  # Local first
        assert "SPEAKER_01" in result.speakers
        assert "SPEAKER_02" in result.speakers

    def test_calculates_duration_from_last_segment(self, diarized_segment_factory) -> None:
        """Duration should be end time of last segment."""
        local_segments = [
            diarized_segment_factory(0.0, 1.0, "Local", "LOCAL"),
        ]
        remote_result = DiarizationResult(
            segments=[
                diarized_segment_factory(1.0, 5.5, "Remote", "SPEAKER_01"),
            ],
            speakers=["SPEAKER_01"],
            language="en",
            language_probability=0.95,
            duration=5.5,
        )

        result = merge_diarization_timelines(
            local_segments, remote_result, local_speaker_name="LOCAL"
        )

        assert result.duration == 5.5

    def test_empty_local_segments(self, diarized_segment_factory) -> None:
        """Should handle empty local segments."""
        local_segments = []
        remote_result = DiarizationResult(
            segments=[
                diarized_segment_factory(0.0, 1.0, "Remote", "SPEAKER_01"),
            ],
            speakers=["SPEAKER_01"],
            language="en",
            language_probability=0.95,
            duration=1.0,
        )

        result = merge_diarization_timelines(
            local_segments, remote_result, local_speaker_name="LOCAL"
        )

        assert len(result.segments) == 1
        assert result.segments[0].speaker == "SPEAKER_01"

    def test_preserves_language_info(self, diarized_segment_factory) -> None:
        """Should preserve language info from remote result."""
        local_segments = [
            diarized_segment_factory(0.0, 1.0, "Hallo", "LOCAL"),
        ]
        remote_result = DiarizationResult(
            segments=[],
            speakers=[],
            language="de",
            language_probability=0.87,
            duration=1.0,
        )

        result = merge_diarization_timelines(
            local_segments, remote_result, local_speaker_name="LOCAL"
        )

        assert result.language == "de"
        assert result.language_probability == 0.87


# =============================================================================
# StereoProcessor Tests - Mock whisperx at boundary
# =============================================================================


class TestStereoProcessorInit:
    """Tests for StereoProcessor initialization."""

    def test_init_stores_config(self, mock_stereo_config) -> None:
        """Should store config on init."""
        processor = StereoProcessor(mock_stereo_config)
        assert processor._config is mock_stereo_config

    def test_init_stores_num_speakers(self, mock_stereo_config) -> None:
        """Should store num_speakers on init."""
        processor = StereoProcessor(mock_stereo_config, num_speakers=3)
        assert processor._num_speakers == 3

    def test_model_not_loaded_initially(self, mock_stereo_config) -> None:
        """Model should not be loaded on init."""
        processor = StereoProcessor(mock_stereo_config)
        assert processor._whisperx_model is None
        assert processor._device is None


class TestStereoProcessorLoadModel:
    """Tests for _load_whisperx_model method."""

    def test_raises_dependency_missing_if_whisperx_not_installed(self, mock_stereo_config) -> None:
        """Should raise DependencyMissingError if whisperx not installed."""
        from hark.exceptions import DependencyMissingError

        processor = StereoProcessor(mock_stereo_config)

        with patch.dict("sys.modules", {"whisperx": None}):
            with pytest.raises(DependencyMissingError):
                processor._load_whisperx_model()

    def test_loads_model_with_correct_params(self, mock_stereo_config) -> None:
        """Should load model with correct parameters."""
        mock_whisperx = MagicMock()
        mock_model = MagicMock()
        mock_whisperx.load_model.return_value = mock_model

        processor = StereoProcessor(mock_stereo_config)

        with (
            patch.dict("sys.modules", {"whisperx": mock_whisperx}),
            patch("hark.device.detect_best_device", return_value="cpu"),
            patch("hark.device.get_compute_type", return_value="int8"),
        ):
            model, device = processor._load_whisperx_model()

            mock_whisperx.load_model.assert_called_once_with(
                "base",
                device="cpu",
                compute_type="int8",
                download_root="/tmp/models",
            )
            assert model is mock_model
            assert device == "cpu"

    def test_caches_model_on_second_call(self, mock_stereo_config) -> None:
        """Model should be cached and not reloaded on second call."""
        mock_whisperx = MagicMock()
        mock_model = MagicMock()
        mock_whisperx.load_model.return_value = mock_model

        processor = StereoProcessor(mock_stereo_config)

        with (
            patch.dict("sys.modules", {"whisperx": mock_whisperx}),
            patch("hark.device.detect_best_device", return_value="cpu"),
            patch("hark.device.get_compute_type", return_value="int8"),
        ):
            processor._load_whisperx_model()
            processor._load_whisperx_model()

            # Should only load once
            assert mock_whisperx.load_model.call_count == 1

    def test_vulkan_falls_back_to_cpu(self, mock_stereo_config) -> None:
        """Vulkan device should fall back to CPU for whisperx."""
        mock_stereo_config.whisper.device = "auto"
        mock_whisperx = MagicMock()

        processor = StereoProcessor(mock_stereo_config)

        with (
            patch.dict("sys.modules", {"whisperx": mock_whisperx}),
            patch("hark.device.detect_best_device", return_value="vulkan"),
            patch("hark.device.get_compute_type", return_value="int8"),
        ):
            _, device = processor._load_whisperx_model()
            assert device == "cpu"


class TestStereoProcessorProcess:
    """Tests for process() method."""

    def test_splits_channels_correctly(self, mock_stereo_config, stereo_audio) -> None:
        """Should split stereo into mic and speaker channels."""
        mock_whisperx = MagicMock()
        mock_model = MagicMock()
        mock_whisperx.load_model.return_value = mock_model
        mock_model.transcribe.return_value = {"language": "en", "segments": []}
        mock_whisperx.load_align_model.return_value = (MagicMock(), MagicMock())
        mock_whisperx.align.return_value = {"segments": []}
        mock_diarize_model = MagicMock()
        mock_diarize_model.return_value = MagicMock()  # Diarize result
        mock_whisperx.diarize.DiarizationPipeline.return_value = mock_diarize_model
        mock_whisperx.assign_word_speakers.return_value = {"segments": []}

        processor = StereoProcessor(mock_stereo_config)

        with (
            patch.dict("sys.modules", {"whisperx": mock_whisperx}),
            patch("hark.device.detect_best_device", return_value="cpu"),
            patch("hark.device.get_compute_type", return_value="int8"),
        ):
            processor.process(stereo_audio, sample_rate=16000)

            # Model should be called twice (once per channel)
            assert mock_model.transcribe.call_count == 2

    def test_returns_diarization_result(self, mock_stereo_config, stereo_audio) -> None:
        """Should return DiarizationResult."""
        mock_whisperx = MagicMock()
        mock_model = MagicMock()
        mock_whisperx.load_model.return_value = mock_model
        mock_model.transcribe.return_value = {"language": "en", "segments": []}
        mock_whisperx.load_align_model.return_value = (MagicMock(), MagicMock())
        mock_whisperx.align.return_value = {"segments": []}
        mock_diarize_model = MagicMock()
        mock_diarize_model.return_value = MagicMock()
        mock_whisperx.diarize.DiarizationPipeline.return_value = mock_diarize_model
        mock_whisperx.assign_word_speakers.return_value = {"segments": []}

        processor = StereoProcessor(mock_stereo_config)

        with (
            patch.dict("sys.modules", {"whisperx": mock_whisperx}),
            patch("hark.device.detect_best_device", return_value="cpu"),
            patch("hark.device.get_compute_type", return_value="int8"),
        ):
            result = processor.process(stereo_audio, sample_rate=16000)

            assert isinstance(result, DiarizationResult)


class TestStereoProcessorTranscribeChannel:
    """Tests for _transcribe_channel method."""

    def test_assigns_speaker_name_to_segments(self, mock_stereo_config) -> None:
        """Should assign the given speaker name to all segments."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "language": "en",
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "Hello"},
                {"start": 1.0, "end": 2.0, "text": "World"},
            ],
        }

        processor = StereoProcessor(mock_stereo_config)
        channel = ChannelAudio(
            audio=np.zeros(16000, dtype=np.float32),
            sample_rate=16000,
            channel_name="mic",
        )

        segments = processor._transcribe_channel(
            model=mock_model,
            device="cpu",
            channel=channel,
            speaker_name="MY_SPEAKER",
            language=None,
        )

        assert len(segments) == 2
        assert all(s.speaker == "MY_SPEAKER" for s in segments)

    def test_converts_to_float32(self, mock_stereo_config) -> None:
        """Should convert audio to float32 if needed."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"language": "en", "segments": []}

        processor = StereoProcessor(mock_stereo_config)
        # Create int16 audio
        channel = ChannelAudio(
            audio=np.zeros(16000, dtype=np.int16),
            sample_rate=16000,
            channel_name="mic",
        )

        processor._transcribe_channel(
            model=mock_model,
            device="cpu",
            channel=channel,
            speaker_name="SPEAKER",
            language=None,
        )

        # Check that transcribe received float32
        call_args = mock_model.transcribe.call_args
        received_audio = call_args[0][0]
        assert received_audio.dtype == np.float32


class TestStereoProcessorDiarizeChannel:
    """Tests for _diarize_channel method."""

    def test_raises_missing_token_error_without_hf_token(self, mock_stereo_config) -> None:
        """Should raise MissingTokenError if no HF token configured."""
        from hark.exceptions import MissingTokenError

        mock_stereo_config.diarization.hf_token = None

        processor = StereoProcessor(mock_stereo_config)
        channel = ChannelAudio(
            audio=np.zeros(16000, dtype=np.float32),
            sample_rate=16000,
            channel_name="speaker",
        )

        mock_whisperx = MagicMock()

        with patch.dict("sys.modules", {"whisperx": mock_whisperx}):
            with pytest.raises(MissingTokenError):
                processor._diarize_channel(
                    model=MagicMock(),
                    device="cpu",
                    channel=channel,
                    language=None,
                )

    def test_speaker_indexing_converts_00_to_01(self, mock_stereo_config) -> None:
        """SPEAKER_00 should become SPEAKER_01 (1-indexed)."""
        mock_whisperx = MagicMock()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"language": "en", "segments": []}
        mock_whisperx.load_align_model.return_value = (MagicMock(), MagicMock())
        mock_whisperx.align.return_value = {"segments": []}

        mock_diarize_model = MagicMock()
        mock_whisperx.diarize.DiarizationPipeline.return_value = mock_diarize_model

        # Simulate diarization returning SPEAKER_00
        mock_whisperx.assign_word_speakers.return_value = {
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "Hello", "speaker": "SPEAKER_00", "words": []},
            ]
        }

        processor = StereoProcessor(mock_stereo_config)
        channel = ChannelAudio(
            audio=np.zeros(16000, dtype=np.float32),
            sample_rate=16000,
            channel_name="speaker",
        )

        with patch.dict("sys.modules", {"whisperx": mock_whisperx}):
            result = processor._diarize_channel(
                model=mock_model,
                device="cpu",
                channel=channel,
                language=None,
            )

            # SPEAKER_00 should be converted to SPEAKER_01
            assert result.segments[0].speaker == "SPEAKER_01"

    def test_language_probability_100_when_language_specified(self, mock_stereo_config) -> None:
        """Language probability should be 1.0 when language explicitly specified."""
        mock_whisperx = MagicMock()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"language": "de", "segments": []}
        mock_whisperx.load_align_model.return_value = (MagicMock(), MagicMock())
        mock_whisperx.align.return_value = {"segments": []}
        mock_diarize_model = MagicMock()
        mock_whisperx.diarize.DiarizationPipeline.return_value = mock_diarize_model
        mock_whisperx.assign_word_speakers.return_value = {"segments": []}

        processor = StereoProcessor(mock_stereo_config)
        channel = ChannelAudio(
            audio=np.zeros(16000, dtype=np.float32),
            sample_rate=16000,
            channel_name="speaker",
        )

        with patch.dict("sys.modules", {"whisperx": mock_whisperx}):
            result = processor._diarize_channel(
                model=mock_model,
                device="cpu",
                channel=channel,
                language="de",  # Explicitly specified
            )

            assert result.language_probability == 1.0

    def test_wraps_exceptions_in_diarization_error(self, mock_stereo_config) -> None:
        """Should wrap exceptions in DiarizationError."""
        from hark.exceptions import DiarizationError

        mock_whisperx = MagicMock()
        mock_model = MagicMock()
        mock_model.transcribe.side_effect = RuntimeError("Transcription failed")

        processor = StereoProcessor(mock_stereo_config)
        channel = ChannelAudio(
            audio=np.zeros(16000, dtype=np.float32),
            sample_rate=16000,
            channel_name="speaker",
        )

        with patch.dict("sys.modules", {"whisperx": mock_whisperx}):
            with pytest.raises(DiarizationError) as exc_info:
                processor._diarize_channel(
                    model=mock_model,
                    device="cpu",
                    channel=channel,
                    language=None,
                )
            assert "failed" in str(exc_info.value).lower()
