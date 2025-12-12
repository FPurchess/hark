"""Shared utilities for hark.

This module provides common utility functions used across the codebase,
consolidating duplicated functionality into a single source of truth.
"""

import contextlib
import io
import os
import sys
from collections.abc import Generator

__all__ = [
    "suppress_output",
    "renumber_speaker",
    "env_vars",
]


@contextlib.contextmanager
def suppress_output() -> Generator[None, None, None]:
    """
    Context manager to temporarily suppress stdout and stderr.

    This is useful for silencing noisy library output (e.g., WhisperX,
    pyannote) that prints warnings or progress information that clutters
    the user interface.

    Example:
        with suppress_output():
            model = whisperx.load_model(...)

    Yields:
        None
    """
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def renumber_speaker(speaker: str, offset: int = 1) -> str:
    """
    Renumber a speaker label by adding an offset to its numeric suffix.

    Speaker labels from diarization models typically use 0-indexed naming
    (SPEAKER_00, SPEAKER_01, etc.). This function converts them to 1-indexed
    or applies any other offset for consistency.

    Args:
        speaker: Original speaker label (e.g., "SPEAKER_00").
        offset: Number to add to the speaker index (default: 1).

    Returns:
        Renumbered speaker label (e.g., "SPEAKER_01"), or the original
        label if it doesn't match the expected pattern.

    Examples:
        >>> renumber_speaker("SPEAKER_00")
        'SPEAKER_01'
        >>> renumber_speaker("SPEAKER_05", offset=1)
        'SPEAKER_06'
        >>> renumber_speaker("SPEAKER_00", offset=0)
        'SPEAKER_00'
        >>> renumber_speaker("UNKNOWN")
        'UNKNOWN'
        >>> renumber_speaker("SPEAKER_invalid")
        'SPEAKER_invalid'
    """
    if not speaker.startswith("SPEAKER_"):
        return speaker

    try:
        # Extract the numeric part after "SPEAKER_"
        num_str = speaker.split("_")[1]
        num = int(num_str)
        return f"SPEAKER_{num + offset:02d}"
    except (IndexError, ValueError):
        # Return original if parsing fails
        return speaker


@contextlib.contextmanager
def env_vars(variables: dict[str, str]) -> Generator[None, None, None]:
    """
    Context manager to temporarily set environment variables.

    On entry, sets the specified environment variables. On exit (whether
    normal or due to an exception), restores the original values. Variables
    that didn't exist before are removed; variables that had values are
    restored to those values.

    This is useful for setting environment variables like PULSE_SOURCE
    that need to be active only during a specific operation (e.g., audio
    recording) without polluting the global environment.

    Args:
        variables: Dictionary of environment variable names to values.

    Yields:
        None

    Example:
        with env_vars({"PULSE_SOURCE": "monitor_source"}):
            # PULSE_SOURCE is set here
            record_audio()
        # PULSE_SOURCE is restored to its original state (or removed)

    Example with multiple variables:
        with env_vars({"VAR1": "value1", "VAR2": "value2"}):
            do_something()
    """
    # Store original values (None if variable didn't exist)
    original_values: dict[str, str | None] = {}
    for key in variables:
        original_values[key] = os.environ.get(key)

    try:
        # Set new values
        os.environ.update(variables)
        yield
    finally:
        # Restore original state
        for key, original_value in original_values.items():
            if original_value is None:
                # Variable didn't exist before, remove it
                os.environ.pop(key, None)
            else:
                # Restore original value
                os.environ[key] = original_value
