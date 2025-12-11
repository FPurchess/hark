"""Hark: 100% offline, Whisper-powered voice notes from your terminal."""

import logging
import os
import warnings

# PyTorch 2.6+ defaults to weights_only=True, but pyannote models use many
# types (omegaconf, builtins) that aren't in the safe list. Use PyTorch's
# official environment variable to disable this for trusted HuggingFace models.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

# Suppress noisy warnings from dependencies
warnings.filterwarnings("ignore", message=".*torchaudio._backend.list_audio_backends.*")
warnings.filterwarnings("ignore", message=".*std\\(\\): degrees of freedom.*")
warnings.filterwarnings("ignore", message=".*TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD.*")
warnings.filterwarnings("ignore", message=".*Model was trained with pyannote.*")
warnings.filterwarnings("ignore", message=".*Model was trained with torch.*")
warnings.filterwarnings("ignore", message=".*Lightning automatically upgraded.*")

# Suppress noisy logging from dependencies (use ERROR level to be more aggressive)
for logger_name in [
    "whisperx",
    "whisperx.asr",
    "whisperx.vads",
    "whisperx.vads.pyannote",
    "whisperx.diarize",
    "pyannote",
    "pyannote.audio",
    "speechbrain",
    "faster_whisper",
    "lightning_fabric",
    "pytorch_lightning",
]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

__version__ = "0.2.0"
