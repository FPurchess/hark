"""Interactive speaker naming for diarization results."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hark.diarizer import DiarizationResult, DiarizedSegment, WordSegment
    from hark.ui import UI

__all__ = [
    "interactive_speaker_naming",
    "get_speaker_excerpt",
]


def get_speaker_excerpt(
    segments: list[DiarizedSegment],
    speaker: str,
    max_length: int = 80,
) -> str:
    """
    Get a text excerpt from a speaker's first segment.

    Args:
        segments: List of diarization segments.
        speaker: Speaker label to find excerpt for.
        max_length: Maximum length of excerpt.

    Returns:
        Text excerpt or "[no speech found]" if no segments.
    """
    for seg in segments:
        if seg.speaker == speaker and seg.text.strip():
            text = seg.text.strip()
            if len(text) > max_length:
                return text[: max_length - 3] + "..."
            return text
    return "[no speech found]"


def _update_word_speakers(
    words: list[WordSegment],
    speaker_names: dict[str, str],
) -> list[WordSegment]:
    """Update speaker labels in word-level segments."""
    from hark.diarizer import WordSegment

    updated_words = []
    for word in words:
        if word.speaker and word.speaker in speaker_names:
            updated_words.append(
                WordSegment(
                    start=word.start,
                    end=word.end,
                    word=word.word,
                    speaker=speaker_names[word.speaker],
                )
            )
        else:
            updated_words.append(word)
    return updated_words


def interactive_speaker_naming(
    result: DiarizationResult,
    quiet: bool = False,
    local_speaker_name: str = "SPEAKER_00",
    ui: UI | None = None,
) -> DiarizationResult:
    """
    Interactively prompt user to name detected speakers.

    Shows excerpt from each speaker and asks for a name.
    User can enter:
    - A name: Renames the speaker
    - "skip" or empty: Keeps original label
    - "done": Stops prompting, keeps remaining labels as-is

    Args:
        result: Diarization result with speaker labels.
        quiet: If True, skip interactive prompts.
        local_speaker_name: Name of the local speaker (excluded from renaming).
        ui: UI handler for output (uses print if None).

    Returns:
        DiarizationResult with updated speaker names.
    """
    from hark.diarizer import DiarizationResult, DiarizedSegment

    def output(msg: str) -> None:
        """Output message via UI or print."""
        if ui:
            ui.info(msg)
        else:
            print(msg)

    if quiet:
        return result

    # Filter to speakers that can be renamed
    # Exclude local speaker (whatever it's named) and any already-named speakers
    renameable_speakers = [
        s for s in result.speakers if s != local_speaker_name and s.startswith("SPEAKER_")
    ]

    if not renameable_speakers:
        return result

    output(f"\nDetected {len(renameable_speakers)} speaker(s) to identify.\n")

    # Map old labels to new names
    speaker_names: dict[str, str] = {}

    for speaker in renameable_speakers:
        excerpt = get_speaker_excerpt(result.segments, speaker)
        output(f'{speaker} said: "{excerpt}"')

        try:
            response = input("Who is this? [name/skip/done]: ").strip()
        except (KeyboardInterrupt, EOFError):
            output("\nSkipping remaining speakers.")
            break

        if response.lower() == "done":
            break
        elif response.lower() == "skip" or not response:
            continue
        else:
            speaker_names[speaker] = response
            output(f"  -> Renamed to: {response}")

    if not speaker_names:
        return result

    # Apply renames to segments and their words
    updated_segments: list[DiarizedSegment] = []
    for seg in result.segments:
        new_speaker = speaker_names.get(seg.speaker, seg.speaker)
        # Update word-level speaker labels too
        new_words = _update_word_speakers(seg.words, speaker_names)

        if seg.speaker in speaker_names or seg.words != new_words:
            updated_segments.append(
                DiarizedSegment(
                    start=seg.start,
                    end=seg.end,
                    text=seg.text,
                    speaker=new_speaker,
                    words=new_words,
                )
            )
        else:
            updated_segments.append(seg)

    # Update speakers list
    updated_speakers = [speaker_names.get(s, s) for s in result.speakers]

    return DiarizationResult(
        segments=updated_segments,
        speakers=updated_speakers,
        language=result.language,
        language_probability=result.language_probability,
        duration=result.duration,
    )
