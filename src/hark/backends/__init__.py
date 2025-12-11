"""Backend interfaces for external dependencies.

This module provides thin wrappers around external libraries (faster-whisper,
whisperx, sounddevice) with well-defined interfaces. This enables:

1. Clear separation between business logic and external dependencies
2. Type-safe interfaces that can be tested independently
3. Easy mocking at well-defined boundaries
4. Contract testing to verify API compatibility
"""

from hark.backends.base import (
    DiarizationBackend,
    DiarizationOutput,
    DiarizedSegment,
    TranscriptionBackend,
    TranscriptionOutput,
    TranscriptionSegment,
    WordInfo,
)
from hark.backends.whisper import FasterWhisperBackend
from hark.backends.whisperx import WhisperXBackend

__all__ = [
    # Protocols
    "TranscriptionBackend",
    "DiarizationBackend",
    # Implementations
    "FasterWhisperBackend",
    "WhisperXBackend",
    # Data classes
    "TranscriptionOutput",
    "TranscriptionSegment",
    "WordInfo",
    "DiarizationOutput",
    "DiarizedSegment",
]
