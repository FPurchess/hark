"""Audio preprocessing for hark."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from hark.config import PreprocessingConfig
from hark.exceptions import PreprocessingError

__all__ = [
    "PreprocessingResult",
    "reduce_noise",
    "normalize_audio",
    "trim_silence",
    "AudioPreprocessor",
]


@dataclass
class PreprocessingResult:
    """Results from preprocessing."""

    original_duration: float
    processed_duration: float
    noise_reduction_applied: bool
    normalization_applied: bool
    silence_trimmed_seconds: float


def reduce_noise(
    audio: np.ndarray,
    sample_rate: int,
    strength: float = 0.5,
) -> np.ndarray:
    """
    Apply spectral gating noise reduction.

    Args:
        audio: Audio data as numpy array.
        sample_rate: Sample rate in Hz.
        strength: Noise reduction strength (0.0-1.0).

    Returns:
        Noise-reduced audio.
    """
    try:
        import noisereduce as nr

        # Map strength to prop_decrease parameter
        # strength 0.0 = no reduction, 1.0 = maximum reduction
        prop_decrease = strength

        reduced = nr.reduce_noise(
            y=audio,
            sr=sample_rate,
            prop_decrease=prop_decrease,
            stationary=True,  # Assume stationary background noise
        )

        return np.asarray(reduced, dtype=np.float32)

    except ImportError:
        # noisereduce not installed, return original
        return audio
    except Exception as e:
        raise PreprocessingError(f"Noise reduction failed: {e}") from e


def normalize_audio(
    audio: np.ndarray,
    target_db: float = -20.0,
) -> np.ndarray:
    """
    Normalize audio to target dB level using RMS normalization.

    Args:
        audio: Audio data as numpy array.
        target_db: Target RMS level in dB (should be <= 0).

    Returns:
        Normalized audio.
    """
    if len(audio) == 0:
        return audio

    # Calculate current RMS
    rms = np.sqrt(np.mean(audio**2))

    if rms < 1e-10:
        # Audio is essentially silent
        return audio

    # Convert RMS to dB
    current_db = 20 * np.log10(rms)

    # Calculate gain needed
    gain_db = target_db - current_db
    gain_linear = 10 ** (gain_db / 20)

    # Apply gain with clipping protection
    normalized = audio * gain_linear
    normalized = np.clip(normalized, -1.0, 1.0)

    return normalized.astype(np.float32)


def trim_silence(
    audio: np.ndarray,
    sample_rate: int,
    threshold_db: float = -40.0,
    min_silence_duration: float = 0.5,
) -> tuple[np.ndarray, float]:
    """
    Trim leading/trailing silence and compress long internal silences.

    For stereo audio, uses combined channel energy to determine silence regions,
    then trims both channels identically to keep them synchronized.

    Args:
        audio: Audio data as numpy array (mono or stereo).
        sample_rate: Sample rate in Hz.
        threshold_db: Silence threshold in dB.
        min_silence_duration: Minimum silence duration to consider (unused, kept for API).

    Returns:
        Tuple of (trimmed_audio, seconds_trimmed).
    """
    if len(audio) == 0:
        return audio, 0.0

    is_stereo = audio.ndim == 2

    # For silence detection, use mono (combined channels for stereo)
    if is_stereo:
        # Use max of both channels for silence detection
        detection_audio = np.max(np.abs(audio), axis=1)
        original_duration = audio.shape[0] / sample_rate
    else:
        detection_audio = audio
        original_duration = len(audio) / sample_rate

    try:
        # Use librosa for silence detection
        # top_db is the threshold below reference to consider as silence
        # Reference is the maximum amplitude
        intervals = librosa.effects.split(
            detection_audio,
            top_db=abs(threshold_db),
            frame_length=2048,
            hop_length=512,
        )

        if len(intervals) == 0:
            # All silence
            return audio, 0.0

        # Concatenate non-silent segments with small gaps
        gap_samples = int(sample_rate * 0.1)  # 100ms gap between segments

        if is_stereo:
            trimmed_parts: list[np.ndarray] = []
            silence_gap = np.zeros((gap_samples, 2), dtype=np.float32)

            for i, (start, end) in enumerate(intervals):
                trimmed_parts.append(audio[start:end])
                if i < len(intervals) - 1:
                    trimmed_parts.append(silence_gap)

            if not trimmed_parts:
                return audio, 0.0

            trimmed = np.concatenate(trimmed_parts, axis=0)
            trimmed_duration = trimmed.shape[0] / sample_rate
        else:
            trimmed_parts_mono: list[np.ndarray] = []
            silence_gap_mono = np.zeros(gap_samples, dtype=np.float32)

            for i, (start, end) in enumerate(intervals):
                trimmed_parts_mono.append(audio[start:end])
                if i < len(intervals) - 1:
                    trimmed_parts_mono.append(silence_gap_mono)

            if not trimmed_parts_mono:
                return audio, 0.0

            trimmed = np.concatenate(trimmed_parts_mono)
            trimmed_duration = len(trimmed) / sample_rate

        seconds_trimmed = original_duration - trimmed_duration

        return trimmed.astype(np.float32), max(0.0, seconds_trimmed)

    except Exception as e:
        raise PreprocessingError(f"Silence trimming failed: {e}") from e


class AudioPreprocessor:
    """Audio preprocessing pipeline."""

    def __init__(self, config: PreprocessingConfig) -> None:
        """
        Initialize preprocessor.

        Args:
            config: Preprocessing configuration.
        """
        self._config = config

    def process(
        self,
        audio_path: Path,
        sample_rate: int,
        progress_callback: Callable[[str, float], None] | None = None,
        preserve_stereo: bool = False,
    ) -> tuple[np.ndarray, PreprocessingResult]:
        """
        Process audio through the preprocessing pipeline.

        Args:
            audio_path: Path to the audio file.
            sample_rate: Expected sample rate.
            progress_callback: Callback for progress updates (step_name, progress).
            preserve_stereo: If True, preserve stereo channels instead of converting to mono.
                           Each channel is processed independently but kept synchronized.

        Returns:
            Tuple of (processed_audio, result_info).

        Raises:
            PreprocessingError: If preprocessing fails.
        """
        # Load audio from file
        try:
            audio, file_sr = sf.read(audio_path, dtype="float32")

            is_stereo = audio.ndim == 2 and audio.shape[1] == 2

            if is_stereo and not preserve_stereo:
                # Convert stereo to mono by averaging channels
                audio = np.mean(audio, axis=1)
                is_stereo = False

            # Resample if needed
            if file_sr != sample_rate:
                if is_stereo:
                    # Resample each channel separately
                    left = librosa.resample(audio[:, 0], orig_sr=file_sr, target_sr=sample_rate)
                    right = librosa.resample(audio[:, 1], orig_sr=file_sr, target_sr=sample_rate)
                    audio = np.column_stack((left, right))
                else:
                    audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sample_rate)

            audio = audio.astype(np.float32)

        except Exception as e:
            raise PreprocessingError(f"Failed to load audio file: {e}") from e

        original_duration = audio.shape[0] / sample_rate if is_stereo else len(audio) / sample_rate

        noise_reduction_applied = False
        normalization_applied = False
        silence_trimmed = 0.0

        # Apply noise reduction
        if self._config.noise_reduction.enabled:
            if progress_callback:
                progress_callback("noise_reduction", 0.0)

            if is_stereo:
                # Process each channel separately
                left = reduce_noise(
                    audio[:, 0],
                    sample_rate,
                    strength=self._config.noise_reduction.strength,
                )
                right = reduce_noise(
                    audio[:, 1],
                    sample_rate,
                    strength=self._config.noise_reduction.strength,
                )
                audio = np.column_stack((left, right))
            else:
                audio = reduce_noise(
                    audio,
                    sample_rate,
                    strength=self._config.noise_reduction.strength,
                )
            noise_reduction_applied = True

            if progress_callback:
                progress_callback("noise_reduction", 1.0)

        # Apply normalization
        if self._config.normalization.enabled:
            if progress_callback:
                progress_callback("normalization", 0.0)

            if is_stereo:
                # Process each channel separately
                left = normalize_audio(
                    audio[:, 0],
                    target_db=self._config.normalization.target_level_db,
                )
                right = normalize_audio(
                    audio[:, 1],
                    target_db=self._config.normalization.target_level_db,
                )
                audio = np.column_stack((left, right))
            else:
                audio = normalize_audio(
                    audio,
                    target_db=self._config.normalization.target_level_db,
                )
            normalization_applied = True

            if progress_callback:
                progress_callback("normalization", 1.0)

        # Apply silence trimming (handles stereo natively)
        if self._config.silence_trimming.enabled:
            if progress_callback:
                progress_callback("silence_trimming", 0.0)

            audio, silence_trimmed = trim_silence(
                audio,
                sample_rate,
                threshold_db=self._config.silence_trimming.threshold_db,
                min_silence_duration=self._config.silence_trimming.min_silence_duration,
            )

            if progress_callback:
                progress_callback("silence_trimming", 1.0)

        processed_duration = audio.shape[0] / sample_rate if is_stereo else len(audio) / sample_rate

        result = PreprocessingResult(
            original_duration=original_duration,
            processed_duration=processed_duration,
            noise_reduction_applied=noise_reduction_applied,
            normalization_applied=normalization_applied,
            silence_trimmed_seconds=silence_trimmed,
        )

        return audio, result
