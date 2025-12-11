"""Contract tests for external dependencies.

These tests verify that the APIs we depend on actually exist and match
our expectations. They run only when the optional dependency is installed.

NO MODEL DOWNLOADS - these tests verify import paths and signatures only.

These tests would have caught bugs like:
- whisperx.DiarizationPipeline vs whisperx.diarize.DiarizationPipeline
- API changes in faster-whisper/whisperx updates
"""

from __future__ import annotations

import inspect

import pytest

# ============================================================
# Skip helpers for optional dependencies
# ============================================================


def _whisperx_available() -> bool:
    """Check if whisperx is installed."""
    try:
        import whisperx  # noqa: F401

        return True
    except ImportError:
        return False


def _faster_whisper_available() -> bool:
    """Check if faster-whisper is installed."""
    try:
        import faster_whisper  # noqa: F401

        return True
    except ImportError:
        return False


requires_whisperx = pytest.mark.skipif(
    not _whisperx_available(),
    reason="whisperx not installed",
)

requires_faster_whisper = pytest.mark.skipif(
    not _faster_whisper_available(),
    reason="faster-whisper not installed",
)


# ============================================================
# WhisperX Contract Tests
# ============================================================


@requires_whisperx
@pytest.mark.contract_test
class TestWhisperXContracts:
    """Verify whisperx API surface matches our expectations.

    These tests verify import paths and function signatures without
    downloading any models.
    """

    def test_diarization_pipeline_import_path(self) -> None:
        """CRITICAL: Verify DiarizationPipeline import path.

        This is the exact bug that slipped through - we use:
            whisperx.diarize.DiarizationPipeline
        NOT:
            whisperx.DiarizationPipeline

        See: src/hark/diarizer.py:228, src/hark/stereo_processor.py:448
        """
        from whisperx import diarize

        assert hasattr(diarize, "DiarizationPipeline"), (
            "whisperx.diarize.DiarizationPipeline not found. The whisperx API may have changed."
        )

    def test_load_model_exists(self) -> None:
        """Verify whisperx.load_model function exists."""
        import whisperx

        assert hasattr(whisperx, "load_model"), "whisperx.load_model not found"
        assert callable(whisperx.load_model), "whisperx.load_model not callable"

    def test_load_model_signature(self) -> None:
        """Verify load_model accepts expected parameters.

        We use these params in diarizer.py and stereo_processor.py:
        - whisper_arch (model name)
        - device
        - compute_type
        - download_root
        """
        import whisperx

        sig = inspect.signature(whisperx.load_model)
        params = set(sig.parameters.keys())

        # Check required params exist (or **kwargs allows them)
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )

        required_params = ["whisper_arch", "device", "compute_type", "download_root"]
        for param in required_params:
            assert param in params or has_var_keyword, (
                f"whisperx.load_model missing parameter: {param}. Available: {params}"
            )

    def test_load_align_model_exists(self) -> None:
        """Verify whisperx.load_align_model function exists."""
        import whisperx

        assert hasattr(whisperx, "load_align_model"), "whisperx.load_align_model not found"
        assert callable(whisperx.load_align_model)

    def test_align_function_exists(self) -> None:
        """Verify whisperx.align function exists."""
        import whisperx

        assert hasattr(whisperx, "align"), "whisperx.align not found"
        assert callable(whisperx.align)

    def test_assign_word_speakers_exists(self) -> None:
        """Verify whisperx.assign_word_speakers function exists."""
        import whisperx

        assert hasattr(whisperx, "assign_word_speakers"), "whisperx.assign_word_speakers not found"
        assert callable(whisperx.assign_word_speakers)

    def test_diarization_pipeline_is_class(self) -> None:
        """Verify DiarizationPipeline is a class we can instantiate."""
        from whisperx.diarize import DiarizationPipeline

        assert isinstance(DiarizationPipeline, type), "DiarizationPipeline should be a class"


# ============================================================
# Faster-Whisper Contract Tests
# ============================================================


@requires_faster_whisper
@pytest.mark.contract_test
class TestFasterWhisperContracts:
    """Verify faster-whisper API surface matches our expectations."""

    def test_whisper_model_import(self) -> None:
        """Verify WhisperModel can be imported."""
        from faster_whisper import WhisperModel

        assert WhisperModel is not None
        assert isinstance(WhisperModel, type)

    def test_whisper_model_signature(self) -> None:
        """Verify WhisperModel constructor accepts expected params.

        We use these in transcriber.py:
        - model_size_or_path
        - device
        - compute_type
        - download_root
        """
        from faster_whisper import WhisperModel

        sig = inspect.signature(WhisperModel.__init__)
        params = set(sig.parameters.keys())

        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )

        required_params = ["model_size_or_path", "device", "compute_type", "download_root"]
        for param in required_params:
            assert param in params or has_var_keyword, (
                f"WhisperModel.__init__ missing parameter: {param}. Available: {params}"
            )

    def test_transcribe_method_exists(self) -> None:
        """Verify WhisperModel.transcribe method exists."""
        from faster_whisper import WhisperModel

        assert hasattr(WhisperModel, "transcribe"), "WhisperModel.transcribe not found"


# ============================================================
# SoundDevice Contract Tests (always available - core dep)
# ============================================================


@pytest.mark.contract_test
class TestSoundDeviceContracts:
    """Verify sounddevice API surface matches expectations."""

    def test_input_stream_exists(self) -> None:
        """Verify InputStream class exists."""
        import sounddevice as sd

        assert hasattr(sd, "InputStream"), "sounddevice.InputStream not found"

    def test_query_devices_exists(self) -> None:
        """Verify query_devices function exists."""
        import sounddevice as sd

        assert hasattr(sd, "query_devices"), "sounddevice.query_devices not found"
        assert callable(sd.query_devices)

    def test_default_device_attribute(self) -> None:
        """Verify default.device attribute exists."""
        import sounddevice as sd

        assert hasattr(sd, "default"), "sounddevice.default not found"
        assert hasattr(sd.default, "device"), "sounddevice.default.device not found"

    def test_callback_flags_exists(self) -> None:
        """Verify CallbackFlags exists."""
        import sounddevice as sd

        assert hasattr(sd, "CallbackFlags"), "sounddevice.CallbackFlags not found"

    def test_port_audio_error_exists(self) -> None:
        """Verify PortAudioError exception exists."""
        import sounddevice as sd

        assert hasattr(sd, "PortAudioError"), "sounddevice.PortAudioError not found"


# ============================================================
# Librosa Contract Tests (core dep)
# ============================================================


@pytest.mark.contract_test
class TestLibrosaContracts:
    """Verify librosa API surface matches expectations."""

    def test_resample_exists(self) -> None:
        """Verify librosa.resample function exists."""
        import librosa

        assert hasattr(librosa, "resample"), "librosa.resample not found"
        assert callable(librosa.resample)

    def test_effects_split_exists(self) -> None:
        """Verify librosa.effects.split exists (used for silence detection)."""
        import librosa

        assert hasattr(librosa, "effects"), "librosa.effects not found"
        assert hasattr(librosa.effects, "split"), "librosa.effects.split not found"
        assert callable(librosa.effects.split)

    def test_db_to_amplitude_exists(self) -> None:
        """Verify librosa.db_to_amplitude exists (used in normalization)."""
        import librosa

        assert hasattr(librosa, "db_to_amplitude"), "librosa.db_to_amplitude not found"


# ============================================================
# Noisereduce Contract Tests (core dep)
# ============================================================


@pytest.mark.contract_test
class TestNoisereduceContracts:
    """Verify noisereduce API surface matches expectations."""

    def test_reduce_noise_exists(self) -> None:
        """Verify noisereduce.reduce_noise function exists."""
        import noisereduce as nr

        assert hasattr(nr, "reduce_noise"), "noisereduce.reduce_noise not found"
        assert callable(nr.reduce_noise)

    def test_reduce_noise_signature(self) -> None:
        """Verify reduce_noise accepts expected parameters.

        We use these in preprocessor.py:
        - y (audio data)
        - sr (sample rate)
        - prop_decrease (strength)
        - stationary
        """
        import noisereduce as nr

        sig = inspect.signature(nr.reduce_noise)
        params = set(sig.parameters.keys())

        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )

        required_params = ["y", "sr", "prop_decrease", "stationary"]
        for param in required_params:
            assert param in params or has_var_keyword, (
                f"noisereduce.reduce_noise missing parameter: {param}. Available: {params}"
            )


# ============================================================
# Soundfile Contract Tests (core dep)
# ============================================================


@pytest.mark.contract_test
class TestSoundfileContracts:
    """Verify soundfile API surface matches expectations."""

    def test_soundfile_class_exists(self) -> None:
        """Verify SoundFile class exists."""
        import soundfile as sf

        assert hasattr(sf, "SoundFile"), "soundfile.SoundFile not found"

    def test_read_function_exists(self) -> None:
        """Verify soundfile.read function exists."""
        import soundfile as sf

        assert hasattr(sf, "read"), "soundfile.read not found"
        assert callable(sf.read)

    def test_write_function_exists(self) -> None:
        """Verify soundfile.write function exists."""
        import soundfile as sf

        assert hasattr(sf, "write"), "soundfile.write not found"
        assert callable(sf.write)
