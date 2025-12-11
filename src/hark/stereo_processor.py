"""Stereo audio processing for diarization with local/remote speaker separation."""

from __future__ import annotations

import contextlib
import io
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np


@contextlib.contextmanager
def _suppress_output():
    """Temporarily suppress stdout/stderr to hide noisy library output."""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


if TYPE_CHECKING:
    from hark.config import HarkConfig
    from hark.diarizer import DiarizationResult, DiarizedSegment

__all__ = [
    "StereoProcessor",
    "split_stereo_channels",
    "merge_diarization_timelines",
]


@dataclass
class ChannelAudio:
    """Audio data from a single channel."""

    audio: np.ndarray
    sample_rate: int
    channel_name: str  # "mic" or "speaker"


def split_stereo_channels(
    stereo_audio: np.ndarray,
    sample_rate: int,
) -> tuple[ChannelAudio, ChannelAudio]:
    """
    Split stereo audio into left (mic) and right (speaker) channels.

    Args:
        stereo_audio: Stereo audio array with shape (samples, 2).
        sample_rate: Audio sample rate.

    Returns:
        Tuple of (left_channel, right_channel) as ChannelAudio.

    Raises:
        ValueError: If audio is not stereo (2 channels).
    """
    if stereo_audio.ndim == 1:
        raise ValueError("Audio is mono, not stereo. Use --input both for stereo recording.")

    if stereo_audio.shape[1] != 2:
        raise ValueError(f"Expected 2 channels, got {stereo_audio.shape[1]}")

    left = ChannelAudio(
        audio=stereo_audio[:, 0].copy(),
        sample_rate=sample_rate,
        channel_name="mic",
    )
    right = ChannelAudio(
        audio=stereo_audio[:, 1].copy(),
        sample_rate=sample_rate,
        channel_name="speaker",
    )

    return left, right


def merge_diarization_timelines(
    local_segments: list[DiarizedSegment],
    remote_result: DiarizationResult,
    local_speaker_name: str = "SPEAKER_00",
) -> DiarizationResult:
    """
    Merge local (mic) and remote (speaker) diarization timelines.

    Args:
        local_segments: Segments from local microphone (single speaker).
        remote_result: Diarization result from remote speaker channel.
        local_speaker_name: Name/label for the local speaker.

    Returns:
        Combined DiarizationResult with all speakers.
    """
    from hark.diarizer import DiarizationResult

    # Combine all segments
    all_segments = local_segments + remote_result.segments

    # Sort by start time
    all_segments.sort(key=lambda s: s.start)

    # Detect and handle overlaps
    merged_segments = _merge_overlapping_segments(all_segments)

    # Build speaker list
    speakers = [local_speaker_name] + [s for s in remote_result.speakers if s != local_speaker_name]

    # Calculate total duration
    duration = max(s.end for s in merged_segments) if merged_segments else 0.0

    return DiarizationResult(
        segments=merged_segments,
        speakers=speakers,
        language=remote_result.language,
        language_probability=remote_result.language_probability,
        duration=duration,
    )


def _merge_overlapping_segments(
    segments: list[DiarizedSegment],
) -> list[DiarizedSegment]:
    """
    Merge segments that overlap in time.

    When two speakers speak simultaneously, creates a combined segment
    with speaker label "SPEAKER_A + SPEAKER_B".

    Note: This function does not modify the input list.

    Args:
        segments: List of segments sorted by start time.

    Returns:
        List of segments with overlaps merged.
    """
    from hark.diarizer import DiarizedSegment

    if not segments:
        return []

    # Work on a copy to avoid mutating the input
    working_segments = [
        DiarizedSegment(
            start=s.start,
            end=s.end,
            text=s.text,
            speaker=s.speaker,
            words=list(s.words),
        )
        for s in segments
    ]

    result: list[DiarizedSegment] = []
    i = 0

    while i < len(working_segments):
        current = working_segments[i]

        # Check if next segment overlaps
        if i + 1 < len(working_segments):
            next_seg = working_segments[i + 1]

            # Check for overlap
            if next_seg.start < current.end:
                # Different speakers = overlap, same speaker = merge
                if next_seg.speaker != current.speaker:
                    # Create overlap segment
                    overlap_start = next_seg.start
                    overlap_end = min(current.end, next_seg.end)

                    # Part before overlap (current speaker only)
                    if current.start < overlap_start:
                        result.append(
                            DiarizedSegment(
                                start=current.start,
                                end=overlap_start,
                                text=current.text,
                                speaker=current.speaker,
                                words=current.words,
                            )
                        )

                    # Overlap part (both speakers)
                    result.append(
                        DiarizedSegment(
                            start=overlap_start,
                            end=overlap_end,
                            text=f"{current.text} / {next_seg.text}",
                            speaker=f"{current.speaker} + {next_seg.speaker}",
                            words=[],
                        )
                    )

                    # Part after overlap
                    if current.end > overlap_end:
                        # Current continues after next ends
                        result.append(
                            DiarizedSegment(
                                start=overlap_end,
                                end=current.end,
                                text=current.text,
                                speaker=current.speaker,
                                words=current.words,
                            )
                        )
                    if next_seg.end > overlap_end:
                        # Next continues after current ends - will be handled in next iteration
                        working_segments[i + 1] = DiarizedSegment(
                            start=overlap_end,
                            end=next_seg.end,
                            text=next_seg.text,
                            speaker=next_seg.speaker,
                            words=next_seg.words,
                        )
                    else:
                        # Skip next segment as it's fully contained in overlap
                        i += 1

                    i += 1
                    continue
                else:
                    # Same speaker, merge the text
                    merged = DiarizedSegment(
                        start=current.start,
                        end=max(current.end, next_seg.end),
                        text=f"{current.text} {next_seg.text}",
                        speaker=current.speaker,
                        words=current.words + next_seg.words,
                    )
                    # Replace current with merged and remove next
                    working_segments[i] = merged
                    working_segments.pop(i + 1)
                    continue

        # No overlap, add segment as-is
        result.append(current)
        i += 1

    return result


class StereoProcessor:
    """Process stereo audio with separate handling for local and remote channels.

    Uses a single WhisperX model instance for both channels to minimize memory usage.
    The left channel (mic) is transcribed only, while the right channel (speaker)
    undergoes full diarization.
    """

    def __init__(
        self,
        config: HarkConfig,
        num_speakers: int | None = None,
    ) -> None:
        """
        Initialize the stereo processor.

        Args:
            config: Application configuration.
            num_speakers: Expected number of remote speakers (hint for diarization).
        """
        self._config = config
        self._num_speakers = num_speakers
        self._whisperx_model = None
        self._device: str | None = None

    def _load_whisperx_model(self) -> tuple:
        """Load the WhisperX model once for reuse across both channels."""
        # Return cached model if already loaded
        if self._whisperx_model is not None and self._device is not None:
            return self._whisperx_model, self._device

        from hark.device import detect_best_device, get_compute_type
        from hark.exceptions import DependencyMissingError

        try:
            import whisperx  # type: ignore[import-not-found]
        except ImportError as e:
            raise DependencyMissingError() from e

        # Resolve device
        device = self._config.whisper.device
        if device == "auto":
            device = detect_best_device()
            # WhisperX doesn't support Vulkan, fall back to CPU
            if device == "vulkan":
                device = "cpu"

        self._device = device
        compute_type = get_compute_type(device)

        # Load model once (suppress noisy version mismatch warnings)
        with _suppress_output():
            self._whisperx_model = whisperx.load_model(
                self._config.whisper.model,
                device=device,
                compute_type=compute_type,
                download_root=str(self._config.model_cache_dir),
            )

        return self._whisperx_model, self._device

    def process(
        self,
        stereo_audio: np.ndarray,
        sample_rate: int,
    ) -> DiarizationResult:
        """
        Process stereo audio with diarization.

        Left channel (mic) is transcribed as local speaker (using shared model).
        Right channel (speaker) is transcribed and diarized for remote speakers.

        Args:
            stereo_audio: Stereo audio array with shape (samples, 2).
            sample_rate: Audio sample rate.

        Returns:
            Combined DiarizationResult with all speakers labeled.
        """
        # Load model once, shared between both channels
        model, device = self._load_whisperx_model()

        # Split channels
        left, right = split_stereo_channels(stereo_audio, sample_rate)

        # Get local speaker name from config
        local_speaker = self._config.diarization.local_speaker_name or "SPEAKER_00"

        # Determine language
        language = (
            self._config.whisper.language if self._config.whisper.language != "auto" else None
        )

        # Process left channel (local user) - transcribe only, no diarization
        local_segments = self._transcribe_channel(
            model=model,
            device=device,
            channel=left,
            speaker_name=local_speaker,
            language=language,
        )

        # Process right channel (remote) - full diarization
        remote_result = self._diarize_channel(
            model=model,
            device=device,
            channel=right,
            language=language,
        )

        # Merge timelines
        return merge_diarization_timelines(
            local_segments=local_segments,
            remote_result=remote_result,
            local_speaker_name=local_speaker,
        )

    def _transcribe_channel(
        self,
        model,  # whisperx model
        device: str,
        channel: ChannelAudio,
        speaker_name: str,
        language: str | None,
    ) -> list[DiarizedSegment]:
        """
        Transcribe a channel without diarization using the shared WhisperX model.

        Args:
            model: Loaded WhisperX model.
            device: Compute device.
            channel: Audio data from the channel.
            speaker_name: Name to assign to all segments.
            language: Language code or None for auto-detect.

        Returns:
            List of DiarizedSegment with the given speaker label.
        """
        from hark.diarizer import DiarizedSegment

        # Ensure audio is float32
        audio = channel.audio
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Transcribe using the shared model
        result = model.transcribe(audio, batch_size=16, language=language)

        # Convert to DiarizedSegment with speaker label
        segments: list[DiarizedSegment] = []
        for seg in result.get("segments", []):
            segments.append(
                DiarizedSegment(
                    start=seg.get("start", 0.0),
                    end=seg.get("end", 0.0),
                    text=seg.get("text", "").strip(),
                    speaker=speaker_name,
                    words=[],
                )
            )

        return segments

    def _diarize_channel(
        self,
        model,  # whisperx model
        device: str,
        channel: ChannelAudio,
        language: str | None,
    ) -> DiarizationResult:
        """
        Transcribe and diarize a channel using the shared WhisperX model.

        Args:
            model: Loaded WhisperX model.
            device: Compute device.
            channel: Audio data from the channel.
            language: Language code or None for auto-detect.

        Returns:
            DiarizationResult with speaker labels SPEAKER_01, SPEAKER_02, etc.
        """
        import whisperx  # type: ignore[import-not-found]
        import whisperx.diarize  # type: ignore[import-not-found]

        from hark.diarizer import DiarizationResult, DiarizedSegment, WordSegment
        from hark.exceptions import DiarizationError, MissingTokenError

        # Check HF token
        hf_token = self._config.diarization.hf_token
        if not hf_token:
            raise MissingTokenError()

        try:
            # Ensure audio is float32
            audio = channel.audio
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            # Transcribe using the shared model
            result = model.transcribe(audio, batch_size=16, language=language)
            detected_language = result.get("language", "unknown")

            # Align (get word-level timestamps)
            with _suppress_output():
                model_a, metadata = whisperx.load_align_model(
                    language_code=detected_language,
                    device=device,
                )
            result = whisperx.align(
                result["segments"],
                model_a,
                metadata,
                audio,
                device,
                return_char_alignments=False,
            )

            # Diarize (suppress noisy version mismatch warnings)
            with _suppress_output():
                diarize_model = whisperx.diarize.DiarizationPipeline(
                    use_auth_token=hf_token,
                    device=device,
                )

            diarize_kwargs = {}
            if self._num_speakers is not None:
                diarize_kwargs["min_speakers"] = self._num_speakers
                diarize_kwargs["max_speakers"] = self._num_speakers

            with _suppress_output():
                diarize_segments = diarize_model(
                    audio,
                    **diarize_kwargs,  # pyrefly: ignore[bad-argument-type]
                )

            # Assign speakers to words
            result = whisperx.assign_word_speakers(diarize_segments, result)

            # Convert to our data structures
            segments: list[DiarizedSegment] = []
            speakers_seen: set[str] = set()

            for seg in result.get("segments", []):
                speaker = seg.get("speaker", "UNKNOWN")

                # Convert SPEAKER_00 to SPEAKER_01 (1-indexed for remote speakers)
                if speaker.startswith("SPEAKER_"):
                    try:
                        num = int(speaker.split("_")[1])
                        speaker = f"SPEAKER_{num + 1:02d}"
                    except (IndexError, ValueError):
                        pass

                speakers_seen.add(speaker)

                # Extract word-level information if available
                words: list[WordSegment] = []
                for word_info in seg.get("words", []):
                    word_speaker = word_info.get("speaker")
                    # Apply same 1-indexing to word speakers
                    if word_speaker and word_speaker.startswith("SPEAKER_"):
                        try:
                            num = int(word_speaker.split("_")[1])
                            word_speaker = f"SPEAKER_{num + 1:02d}"
                        except (IndexError, ValueError):
                            pass

                    words.append(
                        WordSegment(
                            start=word_info.get("start", 0.0),
                            end=word_info.get("end", 0.0),
                            word=word_info.get("word", ""),
                            speaker=word_speaker,
                        )
                    )

                segments.append(
                    DiarizedSegment(
                        start=seg.get("start", 0.0),
                        end=seg.get("end", 0.0),
                        text=seg.get("text", "").strip(),
                        speaker=speaker,
                        words=words,
                    )
                )

            duration = segments[-1].end if segments else 0.0

            # If language was explicitly specified, confidence is 100%
            language_probability = 1.0 if language else 0.0

            return DiarizationResult(
                segments=segments,
                speakers=sorted(speakers_seen),
                language=detected_language,
                language_probability=language_probability,
                duration=duration,
            )

        except MissingTokenError:
            raise
        except Exception as e:
            raise DiarizationError(f"Diarization failed: {e}") from e
