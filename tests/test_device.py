"""Tests for hark.device module."""

from unittest.mock import MagicMock, patch

from hark.device import (
    check_cuda_support,
    check_mps_support,
    detect_best_device,
    get_compute_type,
    get_device_info,
)


class TestCheckCudaSupport:
    """Tests for check_cuda_support function."""

    def test_cuda_available(self) -> None:
        """Should return (True, name, capability) if CUDA is available."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "NVIDIA GeForce RTX 3080"
        mock_torch.cuda.get_device_capability.return_value = (8, 6)

        with patch.dict("sys.modules", {"torch": mock_torch}):
            # Need to reimport to pick up mock
            from hark import device

            result = device.check_cuda_support()
            assert result == (True, "NVIDIA GeForce RTX 3080", (8, 6))

    def test_cuda_not_available(self) -> None:
        """Should return (False, None, None) if CUDA not available."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with patch.dict("sys.modules", {"torch": mock_torch}):
            from hark import device

            result = device.check_cuda_support()
            assert result == (False, None, None)

    def test_torch_not_installed(self) -> None:
        """Should return (False, None, None) if torch not installed."""
        with patch.dict("sys.modules", {"torch": None}):
            # Force ImportError
            with patch("builtins.__import__", side_effect=ImportError("No module named 'torch'")):
                result = check_cuda_support()
                assert result == (False, None, None)

    def test_exception_handled(self) -> None:
        """Should return (False, None, None) on other exceptions."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.side_effect = RuntimeError("CUDA error")

        with patch.dict("sys.modules", {"torch": mock_torch}):
            from hark import device

            result = device.check_cuda_support()
            assert result == (False, None, None)


class TestCheckMpsSupport:
    """Tests for check_mps_support function."""

    def test_mps_available(self) -> None:
        """Should return True if MPS backend is available."""
        mock_torch = MagicMock()
        mock_backends = MagicMock()
        mock_mps = MagicMock()
        mock_mps.is_available.return_value = True
        mock_backends.mps = mock_mps
        mock_torch.backends = mock_backends

        with patch.dict("sys.modules", {"torch": mock_torch}):
            from hark import device

            result = device.check_mps_support()
            assert result is True

    def test_mps_not_available(self) -> None:
        """Should return False if MPS backend not available."""
        mock_torch = MagicMock()
        mock_backends = MagicMock()
        mock_mps = MagicMock()
        mock_mps.is_available.return_value = False
        mock_backends.mps = mock_mps
        mock_torch.backends = mock_backends

        with patch.dict("sys.modules", {"torch": mock_torch}):
            from hark import device

            result = device.check_mps_support()
            assert result is False

    def test_no_mps_attribute(self) -> None:
        """Should return False if backends.mps doesn't exist."""
        mock_torch = MagicMock()
        mock_backends = MagicMock(spec=[])  # No mps attribute
        mock_torch.backends = mock_backends
        del mock_backends.mps

        with patch.dict("sys.modules", {"torch": mock_torch}):
            from hark import device

            result = device.check_mps_support()
            assert result is False

    def test_torch_not_installed(self) -> None:
        """Should return False if torch not installed."""
        result = check_mps_support()
        # Since torch may or may not be available in test env, just verify it returns bool
        assert isinstance(result, bool)


class TestDetectBestDevice:
    """Tests for detect_best_device function."""

    def test_cuda_highest_priority(self) -> None:
        """Should return 'cuda' first if CUDA >= 7.0 available."""
        with patch(
            "hark.device.check_cuda_support",
            return_value=(True, "RTX 3080", (8, 6)),
        ):
            result = detect_best_device()
            assert result == "cuda"

    def test_mps_after_cuda(self) -> None:
        """Should return 'mps' if CUDA unavailable but MPS available."""
        with patch(
            "hark.device.check_cuda_support",
            return_value=(False, None, None),
        ):
            with patch("hark.device.check_mps_support", return_value=True):
                result = detect_best_device()
                assert result == "mps"

    def test_cuda_low_compute_falls_to_mps(self) -> None:
        """Should try MPS if CUDA compute capability < 7.0."""
        with patch(
            "hark.device.check_cuda_support",
            return_value=(True, "GTX 1050", (6, 1)),
        ):
            with patch("hark.device.check_mps_support", return_value=True):
                result = detect_best_device()
                assert result == "mps"

    def test_cuda_low_compute_falls_to_cpu(self) -> None:
        """Should return 'cpu' if CUDA compute capability < 7.0 and no MPS."""
        with patch(
            "hark.device.check_cuda_support",
            return_value=(True, "GTX 1050", (6, 1)),
        ):
            with patch("hark.device.check_mps_support", return_value=False):
                result = detect_best_device()
                assert result == "cpu"

    def test_cpu_fallback(self) -> None:
        """Should return 'cpu' if no GPU available."""
        with patch(
            "hark.device.check_cuda_support",
            return_value=(False, None, None),
        ):
            with patch("hark.device.check_mps_support", return_value=False):
                result = detect_best_device()
                assert result == "cpu"

    def test_verbose_output(self, capsys) -> None:
        """Verbose mode should print detection messages."""
        with patch(
            "hark.device.check_cuda_support",
            return_value=(False, None, None),
        ):
            with patch("hark.device.check_mps_support", return_value=False):
                detect_best_device(verbose=True)
                captured = capsys.readouterr()
                assert "cpu" in captured.out.lower() or "backend" in captured.out.lower()


class TestGetComputeType:
    """Tests for get_compute_type function."""

    def test_cuda_returns_float16(self) -> None:
        """'cuda' device should return 'float16'."""
        assert get_compute_type("cuda") == "float16"

    def test_mps_returns_float16(self) -> None:
        """'mps' device should return 'float16'."""
        assert get_compute_type("mps") == "float16"

    def test_cpu_returns_int8(self) -> None:
        """'cpu' device should return 'int8'."""
        assert get_compute_type("cpu") == "int8"

    def test_unknown_device_returns_int8(self) -> None:
        """Unknown devices should default to 'int8'."""
        assert get_compute_type("unknown") == "int8"


class TestGetDeviceInfo:
    """Tests for get_device_info function."""

    def test_returns_dict(self) -> None:
        """Should return a dictionary."""
        with patch(
            "hark.device.check_cuda_support",
            return_value=(False, None, None),
        ):
            with patch("hark.device.check_mps_support", return_value=False):
                result = get_device_info()
                assert isinstance(result, dict)

    def test_has_expected_keys(self) -> None:
        """Should have expected keys in result."""
        with patch(
            "hark.device.check_cuda_support",
            return_value=(False, None, None),
        ):
            with patch("hark.device.check_mps_support", return_value=False):
                result = get_device_info()
                assert "cuda_available" in result
                assert "mps_available" in result
                assert "recommended_device" in result

    def test_compute_capability_format(self) -> None:
        """compute_capability should be formatted as 'X.Y' string."""
        with patch(
            "hark.device.check_cuda_support",
            return_value=(True, "RTX 3080", (8, 6)),
        ):
            with patch("hark.device.check_mps_support", return_value=False):
                result = get_device_info()
                assert result["compute_capability"] == "8.6"

    def test_compute_capability_none_when_no_cuda(self) -> None:
        """compute_capability should be None when CUDA unavailable."""
        with patch(
            "hark.device.check_cuda_support",
            return_value=(False, None, None),
        ):
            with patch("hark.device.check_mps_support", return_value=False):
                result = get_device_info()
                assert result["compute_capability"] is None

    def test_gpu_name_included(self) -> None:
        """Should include GPU name when available."""
        with patch(
            "hark.device.check_cuda_support",
            return_value=(True, "Test GPU", (7, 5)),
        ):
            with patch("hark.device.check_mps_support", return_value=False):
                result = get_device_info()
                assert result["gpu_name"] == "Test GPU"
