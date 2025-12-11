"""Integration tests for hark.

These tests use small generated audio files to test the full preprocessing
and formatting pipelines without mocking internal components.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from hark.config import (
    HarkConfig,
    NoiseReductionConfig,
    NormalizationConfig,
    PreprocessingConfig,
    SilenceTrimmingConfig,
)
from hark.formatter import MarkdownFormatter, PlainFormatter, SRTFormatter, get_formatter
from hark.preprocessor import AudioPreprocessor, PreprocessingResult
from hark.transcriber import TranscriptionResult, TranscriptionSegment


@pytest.fixture
def speech_audio_file(tmp_path: Path) -> Path:
    """Create a test audio file with simulated speech-like audio."""
    # Generate 1 second of audio at 16kHz with some variation
    sample_rate = 16000
    duration = 1.0
    samples = int(sample_rate * duration)

    # Create audio with varying amplitude (simulates speech patterns)
    t = np.linspace(0, duration, samples)
    # Mix of frequencies with amplitude envelope
    envelope = np.sin(np.pi * t / duration) ** 2  # Ramp up and down
    audio = envelope * (
        0.3 * np.sin(2 * np.pi * 440 * t)  # Fundamental
        + 0.15 * np.sin(2 * np.pi * 880 * t)  # Harmonic
        + 0.05 * np.random.randn(samples)  # Some noise
    )
    audio = audio.astype(np.float32)

    # Save to file
    audio_path = tmp_path / "test_speech.wav"
    sf.write(audio_path, audio, sample_rate)
    return audio_path


@pytest.fixture
def silence_audio_file(tmp_path: Path) -> Path:
    """Create a test audio file with near-silence."""
    sample_rate = 16000
    duration = 1.0
    samples = int(sample_rate * duration)

    # Very quiet audio (simulates silence)
    audio = np.random.randn(samples).astype(np.float32) * 0.001

    audio_path = tmp_path / "test_silence.wav"
    sf.write(audio_path, audio, sample_rate)
    return audio_path


@pytest.fixture
def stereo_audio_file(tmp_path: Path) -> Path:
    """Create a stereo test audio file."""
    sample_rate = 16000
    duration = 1.0
    samples = int(sample_rate * duration)

    # Stereo audio
    left = np.sin(2 * np.pi * 440 * np.linspace(0, duration, samples))
    right = np.sin(2 * np.pi * 660 * np.linspace(0, duration, samples))
    audio = np.column_stack([left, right]).astype(np.float32)

    audio_path = tmp_path / "test_stereo.wav"
    sf.write(audio_path, audio, sample_rate)
    return audio_path


class TestPreprocessorIntegration:
    """Integration tests for AudioPreprocessor with real audio files."""

    def test_process_with_all_steps(self, speech_audio_file: Path) -> None:
        """Should process audio with all preprocessing steps enabled."""
        config = PreprocessingConfig(
            noise_reduction=NoiseReductionConfig(enabled=True, strength=0.3),
            normalization=NormalizationConfig(enabled=True, target_level_db=-20.0),
            silence_trimming=SilenceTrimmingConfig(enabled=True),
        )

        preprocessor = AudioPreprocessor(config)
        audio, result = preprocessor.process(speech_audio_file, sample_rate=16000)

        assert isinstance(audio, np.ndarray)
        assert isinstance(result, PreprocessingResult)
        assert result.noise_reduction_applied is True
        assert result.normalization_applied is True
        assert result.original_duration > 0

    def test_process_with_no_preprocessing(self, speech_audio_file: Path) -> None:
        """Should process audio with all preprocessing disabled."""
        config = PreprocessingConfig(
            noise_reduction=NoiseReductionConfig(enabled=False),
            normalization=NormalizationConfig(enabled=False),
            silence_trimming=SilenceTrimmingConfig(enabled=False),
        )

        preprocessor = AudioPreprocessor(config)
        audio, result = preprocessor.process(speech_audio_file, sample_rate=16000)

        assert isinstance(audio, np.ndarray)
        assert result.noise_reduction_applied is False
        assert result.normalization_applied is False
        assert result.silence_trimmed_seconds == 0.0

    def test_process_stereo_converts_to_mono(self, stereo_audio_file: Path) -> None:
        """Should convert stereo audio to mono."""
        config = PreprocessingConfig(
            noise_reduction=NoiseReductionConfig(enabled=False),
            normalization=NormalizationConfig(enabled=False),
            silence_trimming=SilenceTrimmingConfig(enabled=False),
        )

        preprocessor = AudioPreprocessor(config)
        audio, result = preprocessor.process(stereo_audio_file, sample_rate=16000)

        # Should be 1D array (mono)
        assert len(audio.shape) == 1

    def test_process_resamples_different_rate(self, tmp_path: Path) -> None:
        """Should resample audio from different sample rate."""
        # Create 48kHz audio file
        sample_rate = 48000
        duration = 1.0
        samples = int(sample_rate * duration)
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, samples))
        audio = audio.astype(np.float32)

        audio_path = tmp_path / "test_48k.wav"
        sf.write(audio_path, audio, sample_rate)

        config = PreprocessingConfig(
            noise_reduction=NoiseReductionConfig(enabled=False),
            normalization=NormalizationConfig(enabled=False),
            silence_trimming=SilenceTrimmingConfig(enabled=False),
        )

        preprocessor = AudioPreprocessor(config)
        processed, result = preprocessor.process(audio_path, sample_rate=16000)

        # Should be resampled to ~16000 samples (1 second at 16kHz)
        assert 15000 < len(processed) < 17000

    def test_process_with_progress_callback(self, speech_audio_file: Path) -> None:
        """Should call progress callback during processing."""
        config = PreprocessingConfig(
            noise_reduction=NoiseReductionConfig(enabled=True, strength=0.3),
            normalization=NormalizationConfig(enabled=True),
            silence_trimming=SilenceTrimmingConfig(enabled=True),
        )

        callback_calls = []

        def progress_callback(step: str, progress: float) -> None:
            callback_calls.append((step, progress))

        preprocessor = AudioPreprocessor(config)
        preprocessor.process(
            speech_audio_file, sample_rate=16000, progress_callback=progress_callback
        )

        # Should have called back for each step
        steps = [call[0] for call in callback_calls]
        assert "noise_reduction" in steps
        assert "normalization" in steps
        assert "silence_trimming" in steps


class TestFormatterIntegration:
    """Integration tests for formatters with TranscriptionResult."""

    @pytest.fixture
    def full_result(self) -> TranscriptionResult:
        """Create a full transcription result."""
        return TranscriptionResult(
            text="Hello world. This is a test transcription.",
            segments=[
                TranscriptionSegment(start=0.0, end=1.5, text="Hello world.", words=[]),
                TranscriptionSegment(
                    start=1.6, end=3.5, text="This is a test transcription.", words=[]
                ),
            ],
            language="en",
            language_probability=0.95,
            duration=3.5,
        )

    def test_plain_formatter_no_timestamps(self, full_result: TranscriptionResult) -> None:
        """PlainFormatter should output text only."""
        formatter = PlainFormatter(include_timestamps=False)
        output = formatter.format(full_result)

        assert "Hello world." in output
        assert "This is a test transcription." in output
        assert "[" not in output  # No timestamps

    def test_plain_formatter_with_timestamps(self, full_result: TranscriptionResult) -> None:
        """PlainFormatter should include timestamps when enabled."""
        formatter = PlainFormatter(include_timestamps=True)
        output = formatter.format(full_result)

        assert "[00:00.000 --> 00:01.500]" in output
        assert "Hello world." in output

    def test_markdown_formatter_has_header(self, full_result: TranscriptionResult) -> None:
        """MarkdownFormatter should include header."""
        formatter = MarkdownFormatter(include_timestamps=False)
        output = formatter.format(full_result)

        assert output.startswith("# Transcription")
        assert "Hello world." in output

    def test_markdown_formatter_has_metadata(self, full_result: TranscriptionResult) -> None:
        """MarkdownFormatter should include metadata."""
        formatter = MarkdownFormatter(include_timestamps=False)
        output = formatter.format(full_result)

        assert "en" in output
        assert "95%" in output
        assert "3.5s" in output

    def test_srt_formatter_has_sequence_numbers(self, full_result: TranscriptionResult) -> None:
        """SRTFormatter should include sequence numbers."""
        formatter = SRTFormatter()
        output = formatter.format(full_result)

        lines = output.split("\n")
        assert lines[0] == "1"
        assert "2" in output

    def test_srt_formatter_has_timestamps(self, full_result: TranscriptionResult) -> None:
        """SRTFormatter should have HH:MM:SS,mmm timestamps."""
        formatter = SRTFormatter()
        output = formatter.format(full_result)

        assert "00:00:00,000 --> 00:00:01,500" in output

    def test_get_formatter_plain(self) -> None:
        """get_formatter('plain') should return PlainFormatter."""
        formatter = get_formatter("plain")
        assert isinstance(formatter, PlainFormatter)

    def test_get_formatter_markdown(self) -> None:
        """get_formatter('markdown') should return MarkdownFormatter."""
        formatter = get_formatter("markdown")
        assert isinstance(formatter, MarkdownFormatter)

    def test_get_formatter_srt(self) -> None:
        """get_formatter('srt') should return SRTFormatter."""
        formatter = get_formatter("srt")
        assert isinstance(formatter, SRTFormatter)


class TestConfigIntegration:
    """Integration tests for configuration system."""

    def test_default_config_valid(self) -> None:
        """Default HarkConfig should pass validation."""
        from hark.config import validate_config

        config = HarkConfig()
        errors = validate_config(config)
        assert errors == []

    def test_config_merge_preserves_defaults(self) -> None:
        """CLI args merge should preserve unset values."""
        import argparse

        from hark.config import merge_cli_args

        config = HarkConfig()
        original_model = config.whisper.model

        # Args with no model specified
        args = argparse.Namespace(
            lang=None,
            model=None,
            max_duration=None,
            sample_rate=None,
            channels=None,
            no_noise_reduction=False,
            no_normalize=False,
            no_trim_silence=False,
            noise_strength=None,
            timestamps=False,
            format=None,
            append=False,
            quiet=False,
            verbose=False,
        )

        merged = merge_cli_args(config, args)
        assert merged.whisper.model == original_model


class TestEndToEndFlow:
    """End-to-end integration tests."""

    def test_preprocessing_to_formatter_flow(self, speech_audio_file: Path) -> None:
        """Test flow from preprocessing to formatted output."""
        # Preprocess audio
        config = PreprocessingConfig(
            noise_reduction=NoiseReductionConfig(enabled=True, strength=0.3),
            normalization=NormalizationConfig(enabled=True, target_level_db=-20.0),
            silence_trimming=SilenceTrimmingConfig(enabled=False),
        )

        preprocessor = AudioPreprocessor(config)
        audio, preprocess_result = preprocessor.process(speech_audio_file, sample_rate=16000)

        # Create a mock transcription result (can't actually transcribe without model)
        transcription = TranscriptionResult(
            text="Test transcription result",
            segments=[
                TranscriptionSegment(
                    start=0.0,
                    end=preprocess_result.processed_duration,
                    text="Test transcription result",
                    words=[],
                ),
            ],
            language="en",
            language_probability=0.95,
            duration=preprocess_result.processed_duration,
        )

        # Format output
        formatter = PlainFormatter(include_timestamps=False)
        output = formatter.format(transcription)

        assert "Test transcription result" in output

    def test_file_output_roundtrip(self, tmp_path: Path) -> None:
        """Test writing and reading output file."""
        output_file = tmp_path / "output.txt"

        # Create transcription result
        result = TranscriptionResult(
            text="Test output text",
            segments=[TranscriptionSegment(start=0.0, end=1.0, text="Test output text", words=[])],
            language="en",
            language_probability=0.9,
            duration=1.0,
        )

        # Format and write
        formatter = PlainFormatter(include_timestamps=False)
        output_text = formatter.format(result)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(output_text)

        # Read back
        content = output_file.read_text(encoding="utf-8")
        assert "Test output text" in content

    def test_append_mode_output(self, tmp_path: Path) -> None:
        """Test appending to output file."""
        output_file = tmp_path / "append_test.txt"

        # Write first result
        result1 = TranscriptionResult(
            text="First transcription",
            segments=[],
            language="en",
            language_probability=0.9,
            duration=1.0,
        )
        formatter = PlainFormatter(include_timestamps=False)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(formatter.format(result1))
            f.write("\n")

        # Append second result
        result2 = TranscriptionResult(
            text="Second transcription",
            segments=[],
            language="en",
            language_probability=0.9,
            duration=1.0,
        )

        with open(output_file, "a", encoding="utf-8") as f:
            f.write(formatter.format(result2))
            f.write("\n")

        # Verify both are present
        content = output_file.read_text(encoding="utf-8")
        assert "First transcription" in content
        assert "Second transcription" in content

    def test_all_output_formats(self) -> None:
        """Test all output formats produce valid output."""
        result = TranscriptionResult(
            text="Format test",
            segments=[TranscriptionSegment(start=0.0, end=1.0, text="Format test", words=[])],
            language="en",
            language_probability=0.95,
            duration=1.0,
        )

        formats = ["plain", "markdown", "srt"]
        for fmt in formats:
            formatter = get_formatter(fmt)
            output = formatter.format(result)
            assert len(output) > 0, f"Format '{fmt}' produced empty output"
            assert "Format test" in output or "test" in output.lower()


class TestDataFlowIntegration:
    """Test component interactions without external deps."""

    def test_preprocessor_output_valid_for_transcriber(self, speech_audio_file: Path) -> None:
        """Preprocessor output should meet transcriber input requirements."""
        config = PreprocessingConfig(
            noise_reduction=NoiseReductionConfig(enabled=True, strength=0.3),
            normalization=NormalizationConfig(enabled=True, target_level_db=-20.0),
            silence_trimming=SilenceTrimmingConfig(enabled=False),
        )

        preprocessor = AudioPreprocessor(config)
        audio, _ = preprocessor.process(speech_audio_file, sample_rate=16000)

        # Verify audio meets transcriber requirements
        assert audio.dtype == np.float32, "Audio must be float32"
        assert audio.ndim == 1, "Audio must be mono (1D)"
        assert len(audio) > 0, "Audio must have samples"
        assert np.isfinite(audio).all(), "Audio must not contain NaN/Inf"

    def test_preprocessor_output_valid_for_diarizer(self, speech_audio_file: Path) -> None:
        """Preprocessor output should meet diarizer input requirements."""
        config = PreprocessingConfig(
            noise_reduction=NoiseReductionConfig(enabled=True, strength=0.3),
            normalization=NormalizationConfig(enabled=True, target_level_db=-20.0),
            silence_trimming=SilenceTrimmingConfig(enabled=False),
        )

        preprocessor = AudioPreprocessor(config)
        audio, _ = preprocessor.process(speech_audio_file, sample_rate=16000)

        # Verify audio meets diarizer requirements (same as transcriber)
        assert audio.dtype == np.float32, "Audio must be float32"
        assert audio.ndim == 1, "Audio must be mono (1D)"
        assert len(audio) > 0, "Audio must have samples"

    def test_stereo_audio_produces_two_channels(self, tmp_path: Path) -> None:
        """Stereo audio should be loadable and splittable into two channels."""
        # Create stereo audio with different content per channel
        sample_rate = 16000
        duration = 1.0
        samples = int(sample_rate * duration)

        # Left channel: 440Hz, Right channel: 880Hz
        t = np.linspace(0, duration, samples)
        left = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        right = np.sin(2 * np.pi * 880 * t).astype(np.float32)
        stereo = np.column_stack([left, right])

        audio_path = tmp_path / "stereo_test.wav"
        sf.write(audio_path, stereo, sample_rate)

        # Read back and verify channels are distinct
        audio, sr = sf.read(audio_path, dtype="float32")
        assert audio.shape[1] == 2, "Should have 2 channels"

        left_channel = audio[:, 0]
        right_channel = audio[:, 1]

        # Channels should be different (different frequencies)
        assert not np.allclose(left_channel, right_channel), "Channels should differ"

        # Both channels should be valid for processing
        assert left_channel.dtype == np.float32
        assert right_channel.dtype == np.float32
        assert len(left_channel) == len(right_channel) == samples


class TestConfigWiring:
    """Test configuration correctly affects components."""

    def test_noise_reduction_config_affects_output(self, speech_audio_file: Path) -> None:
        """Different noise reduction config should produce different output."""
        config_on = PreprocessingConfig(
            noise_reduction=NoiseReductionConfig(enabled=True, strength=0.5),
            normalization=NormalizationConfig(enabled=False),
            silence_trimming=SilenceTrimmingConfig(enabled=False),
        )
        config_off = PreprocessingConfig(
            noise_reduction=NoiseReductionConfig(enabled=False),
            normalization=NormalizationConfig(enabled=False),
            silence_trimming=SilenceTrimmingConfig(enabled=False),
        )

        preprocessor_on = AudioPreprocessor(config_on)
        preprocessor_off = AudioPreprocessor(config_off)

        audio_on, result_on = preprocessor_on.process(speech_audio_file, sample_rate=16000)
        audio_off, result_off = preprocessor_off.process(speech_audio_file, sample_rate=16000)

        assert result_on.noise_reduction_applied is True
        assert result_off.noise_reduction_applied is False
        # Audio should be different (noise reduction modifies the signal)
        assert not np.allclose(audio_on, audio_off, atol=1e-6)

    def test_normalization_config_affects_output(self, speech_audio_file: Path) -> None:
        """Different normalization config should produce different output."""
        config_on = PreprocessingConfig(
            noise_reduction=NoiseReductionConfig(enabled=False),
            normalization=NormalizationConfig(enabled=True, target_level_db=-20.0),
            silence_trimming=SilenceTrimmingConfig(enabled=False),
        )
        config_off = PreprocessingConfig(
            noise_reduction=NoiseReductionConfig(enabled=False),
            normalization=NormalizationConfig(enabled=False),
            silence_trimming=SilenceTrimmingConfig(enabled=False),
        )

        preprocessor_on = AudioPreprocessor(config_on)
        preprocessor_off = AudioPreprocessor(config_off)

        audio_on, result_on = preprocessor_on.process(speech_audio_file, sample_rate=16000)
        audio_off, result_off = preprocessor_off.process(speech_audio_file, sample_rate=16000)

        assert result_on.normalization_applied is True
        assert result_off.normalization_applied is False
        # Audio should be different (normalization changes amplitude)
        # Note: We can't use np.allclose because normalized audio may have same shape
        # but different absolute values
        assert np.abs(audio_on).max() != np.abs(audio_off).max()

    def test_sample_rate_affects_output_length(self, tmp_path: Path) -> None:
        """Different target sample rates should produce different output lengths."""
        # Create a 48kHz audio file
        source_rate = 48000
        duration = 1.0
        samples = int(source_rate * duration)
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, samples))
        audio = audio.astype(np.float32)

        audio_path = tmp_path / "test_48k.wav"
        sf.write(audio_path, audio, source_rate)

        config = PreprocessingConfig(
            noise_reduction=NoiseReductionConfig(enabled=False),
            normalization=NormalizationConfig(enabled=False),
            silence_trimming=SilenceTrimmingConfig(enabled=False),
        )

        preprocessor = AudioPreprocessor(config)

        # Process at 16kHz
        audio_16k, _ = preprocessor.process(audio_path, sample_rate=16000)

        # Should be resampled to 16kHz (16000 samples for 1 second)
        expected_samples = int(16000 * duration)
        assert abs(len(audio_16k) - expected_samples) < 100  # Allow small tolerance


class TestSilenceHandling:
    """Tests for handling silent audio."""

    def test_preprocessing_silent_audio(self, silence_audio_file: Path) -> None:
        """Should handle mostly silent audio gracefully."""
        config = PreprocessingConfig(
            noise_reduction=NoiseReductionConfig(enabled=True, strength=0.5),
            normalization=NormalizationConfig(enabled=True, target_level_db=-20.0),
            silence_trimming=SilenceTrimmingConfig(enabled=True),
        )

        preprocessor = AudioPreprocessor(config)
        audio, result = preprocessor.process(silence_audio_file, sample_rate=16000)

        # Should not crash, should return some audio
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0


class TestEdgeCases:
    """Tests for edge cases."""

    def test_very_short_audio(self, tmp_path: Path) -> None:
        """Should handle very short audio."""
        # Create 0.1 second audio
        sample_rate = 16000
        samples = int(sample_rate * 0.1)
        audio = np.random.randn(samples).astype(np.float32) * 0.1

        audio_path = tmp_path / "short.wav"
        sf.write(audio_path, audio, sample_rate)

        config = PreprocessingConfig(
            noise_reduction=NoiseReductionConfig(enabled=False),
            normalization=NormalizationConfig(enabled=False),
            silence_trimming=SilenceTrimmingConfig(enabled=False),
        )

        preprocessor = AudioPreprocessor(config)
        processed, result = preprocessor.process(audio_path, sample_rate=16000)

        assert len(processed) > 0

    def test_empty_transcription_result(self) -> None:
        """Should handle empty transcription result."""
        result = TranscriptionResult(
            text="",
            segments=[],
            language="unknown",
            language_probability=0.0,
            duration=0.0,
        )

        for fmt in ["plain", "markdown", "srt"]:
            formatter = get_formatter(fmt)
            output = formatter.format(result)
            # Should not crash, may be empty or minimal
            assert isinstance(output, str)

    def test_unicode_in_transcription(self) -> None:
        """Should handle unicode characters in transcription."""
        result = TranscriptionResult(
            text="Bonjour le monde! \u4e16\u754c\u4f60\u597d!",
            segments=[
                TranscriptionSegment(
                    start=0.0,
                    end=2.0,
                    text="Bonjour le monde! \u4e16\u754c\u4f60\u597d!",
                    words=[],
                )
            ],
            language="multi",
            language_probability=0.5,
            duration=2.0,
        )

        formatter = PlainFormatter(include_timestamps=False)
        output = formatter.format(result)

        assert "Bonjour" in output
        assert "\u4e16\u754c" in output  # Chinese characters

    def test_special_characters_in_output(self) -> None:
        """Should handle special characters like quotes and newlines."""
        result = TranscriptionResult(
            text='He said "Hello"\nNew line',
            segments=[
                TranscriptionSegment(start=0.0, end=1.0, text='He said "Hello"\nNew line', words=[])
            ],
            language="en",
            language_probability=0.9,
            duration=1.0,
        )

        formatter = PlainFormatter(include_timestamps=False)
        output = formatter.format(result)

        assert '"Hello"' in output
        assert "\n" in output
