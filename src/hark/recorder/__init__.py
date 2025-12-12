"""
Audio recording module for hark.

This module provides audio recording functionality with support for:
- Microphone recording
- System audio/loopback recording
- Simultaneous mic + speaker recording (dual-stream)

Components:
- AudioRecorder: Main recording class with start/stop/duration tracking
- RecordingFileManager: Manages temporary WAV file creation and writing
- DualStreamInterleaver: Handles buffer interleaving for dual-stream mode
- AudioDeviceInfo: Type definition for device information
"""

from hark.recorder.file_manager import RecordingFileManager
from hark.recorder.interleaver import DualStreamInterleaver
from hark.recorder.recorder import AudioRecorder
from hark.recorder.types import AudioDeviceInfo

__all__ = [
    "AudioDeviceInfo",
    "AudioRecorder",
    "DualStreamInterleaver",
    "RecordingFileManager",
]
