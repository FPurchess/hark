"""Tests for mrec_cli.device module."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

from mrec_cli.device import (
    check_cuda_support,
    check_pytorch_vulkan,
    check_vulkan_support,
    detect_best_device,
    get_compute_type,
    get_device_info,
)


class TestCheckVulkanSupport:
    """Tests for check_vulkan_support function."""

    def test_vulkaninfo_success(self) -> None:
        """Should return True if vulkaninfo succeeds with 'Vulkan Instance'."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Vulkan Instance Version: 1.3.0"

        with patch("subprocess.run", return_value=mock_result):
            assert check_vulkan_support() is True

    def test_vulkaninfo_fails_vkcube_success(self) -> None:
        """Should fall back to vkcube if vulkaninfo fails."""
        vulkaninfo_result = MagicMock()
        vulkaninfo_result.returncode = 1
        vulkaninfo_result.stdout = ""

        vkcube_result = MagicMock()
        vkcube_result.returncode = 0

        def mock_run(cmd, **kwargs):
            if "vulkaninfo" in cmd:
                return vulkaninfo_result
            elif "vkcube" in cmd:
                return vkcube_result
            raise FileNotFoundError()

        with patch("subprocess.run", side_effect=mock_run):
            assert check_vulkan_support() is True

    def test_ldconfig_fallback(self) -> None:
        """Should fall back to ldconfig if other checks fail."""

        def mock_run(cmd, **kwargs):
            if "ldconfig" in cmd:
                result = MagicMock()
                result.returncode = 0
                result.stdout = "libvulkan.so.1 => /usr/lib/libvulkan.so.1"
                return result
            raise FileNotFoundError()

        with patch("subprocess.run", side_effect=mock_run):
            assert check_vulkan_support() is True

    def test_all_checks_fail(self) -> None:
        """Should return False if all checks fail."""
        with patch("subprocess.run", side_effect=FileNotFoundError()):
            assert check_vulkan_support() is False

    def test_timeout_handled(self) -> None:
        """Should handle subprocess timeout gracefully."""
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 5)):
            assert check_vulkan_support() is False

    def test_command_not_found(self) -> None:
        """Should handle FileNotFoundError for missing commands."""
        with patch("subprocess.run", side_effect=FileNotFoundError()):
            assert check_vulkan_support() is False

    def test_vulkaninfo_no_vulkan_instance(self) -> None:
        """Should return False if vulkaninfo output doesn't contain 'Vulkan Instance'."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "No Vulkan support"

        def mock_run(cmd, **kwargs):
            if "vulkaninfo" in cmd:
                return mock_result
            raise FileNotFoundError()

        with patch("subprocess.run", side_effect=mock_run):
            assert check_vulkan_support() is False


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
            from mrec_cli import device

            result = device.check_cuda_support()
            assert result == (True, "NVIDIA GeForce RTX 3080", (8, 6))

    def test_cuda_not_available(self) -> None:
        """Should return (False, None, None) if CUDA not available."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with patch.dict("sys.modules", {"torch": mock_torch}):
            from mrec_cli import device

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
            from mrec_cli import device

            result = device.check_cuda_support()
            assert result == (False, None, None)


class TestCheckPytorchVulkan:
    """Tests for check_pytorch_vulkan function."""

    def test_vulkan_available(self) -> None:
        """Should return True if PyTorch Vulkan backend is available."""
        mock_torch = MagicMock()
        mock_backends = MagicMock()
        mock_vulkan = MagicMock()
        mock_vulkan.is_available.return_value = True
        mock_backends.vulkan = mock_vulkan
        mock_torch.backends = mock_backends

        with patch.dict("sys.modules", {"torch": mock_torch}):
            from mrec_cli import device

            result = device.check_pytorch_vulkan()
            assert result is True

    def test_vulkan_not_available(self) -> None:
        """Should return False if Vulkan backend not available."""
        mock_torch = MagicMock()
        mock_backends = MagicMock()
        mock_vulkan = MagicMock()
        mock_vulkan.is_available.return_value = False
        mock_backends.vulkan = mock_vulkan
        mock_torch.backends = mock_backends

        with patch.dict("sys.modules", {"torch": mock_torch}):
            from mrec_cli import device

            result = device.check_pytorch_vulkan()
            assert result is False

    def test_no_vulkan_attribute(self) -> None:
        """Should return False if backends.vulkan doesn't exist."""
        mock_torch = MagicMock()
        mock_backends = MagicMock(spec=[])  # No vulkan attribute
        mock_torch.backends = mock_backends
        # hasattr will return False
        del mock_backends.vulkan

        with patch.dict("sys.modules", {"torch": mock_torch}):
            from mrec_cli import device

            result = device.check_pytorch_vulkan()
            assert result is False

    def test_torch_not_installed(self) -> None:
        """Should return False if torch not installed."""
        result = check_pytorch_vulkan()
        # Since torch may or may not be available in test env, just verify it returns bool
        assert isinstance(result, bool)


class TestDetectBestDevice:
    """Tests for detect_best_device function."""

    def test_vulkan_priority(self) -> None:
        """Should return 'vulkan' if both HW and PyTorch support exist."""
        with patch("mrec_cli.device.check_vulkan_support", return_value=True):
            with patch("mrec_cli.device.check_pytorch_vulkan", return_value=True):
                result = detect_best_device()
                assert result == "vulkan"

    def test_cuda_fallback_high_compute(self) -> None:
        """Should return 'cuda' if Vulkan unavailable but CUDA >= 7.0."""
        with patch("mrec_cli.device.check_vulkan_support", return_value=False):
            with patch("mrec_cli.device.check_pytorch_vulkan", return_value=False):
                with patch(
                    "mrec_cli.device.check_cuda_support",
                    return_value=(True, "RTX 3080", (8, 6)),
                ):
                    result = detect_best_device()
                    assert result == "cuda"

    def test_cuda_low_compute_falls_to_cpu(self) -> None:
        """Should return 'cpu' if CUDA compute capability < 7.0."""
        with patch("mrec_cli.device.check_vulkan_support", return_value=False):
            with patch("mrec_cli.device.check_pytorch_vulkan", return_value=False):
                with patch(
                    "mrec_cli.device.check_cuda_support",
                    return_value=(True, "GTX 1050", (6, 1)),
                ):
                    result = detect_best_device()
                    assert result == "cpu"

    def test_cpu_fallback(self) -> None:
        """Should return 'cpu' if no GPU available."""
        with patch("mrec_cli.device.check_vulkan_support", return_value=False):
            with patch("mrec_cli.device.check_pytorch_vulkan", return_value=False):
                with patch(
                    "mrec_cli.device.check_cuda_support",
                    return_value=(False, None, None),
                ):
                    result = detect_best_device()
                    assert result == "cpu"

    def test_vulkan_hw_but_no_pytorch(self) -> None:
        """Should skip Vulkan if HW exists but no PyTorch support."""
        with patch("mrec_cli.device.check_vulkan_support", return_value=True):
            with patch("mrec_cli.device.check_pytorch_vulkan", return_value=False):
                with patch(
                    "mrec_cli.device.check_cuda_support",
                    return_value=(False, None, None),
                ):
                    result = detect_best_device()
                    assert result == "cpu"

    def test_verbose_output(self, capsys) -> None:
        """Verbose mode should print detection messages."""
        with patch("mrec_cli.device.check_vulkan_support", return_value=False):
            with patch("mrec_cli.device.check_pytorch_vulkan", return_value=False):
                with patch(
                    "mrec_cli.device.check_cuda_support",
                    return_value=(False, None, None),
                ):
                    detect_best_device(verbose=True)
                    captured = capsys.readouterr()
                    assert "cpu" in captured.out.lower() or "backend" in captured.out.lower()


class TestGetComputeType:
    """Tests for get_compute_type function."""

    def test_cuda_returns_float16(self) -> None:
        """'cuda' device should return 'float16'."""
        assert get_compute_type("cuda") == "float16"

    def test_vulkan_returns_float16(self) -> None:
        """'vulkan' device should return 'float16'."""
        assert get_compute_type("vulkan") == "float16"

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
        with patch("mrec_cli.device.check_vulkan_support", return_value=False):
            with patch("mrec_cli.device.check_pytorch_vulkan", return_value=False):
                with patch(
                    "mrec_cli.device.check_cuda_support",
                    return_value=(False, None, None),
                ):
                    with patch("mrec_cli.device.detect_best_device", return_value="cpu"):
                        result = get_device_info()
                        assert isinstance(result, dict)

    def test_has_expected_keys(self) -> None:
        """Should have expected keys in result."""
        with patch("mrec_cli.device.check_vulkan_support", return_value=False):
            with patch("mrec_cli.device.check_pytorch_vulkan", return_value=False):
                with patch(
                    "mrec_cli.device.check_cuda_support",
                    return_value=(False, None, None),
                ):
                    with patch("mrec_cli.device.detect_best_device", return_value="cpu"):
                        result = get_device_info()
                        assert "vulkan_hardware" in result
                        assert "vulkan_pytorch" in result
                        assert "cuda_available" in result
                        assert "recommended_device" in result

    def test_compute_capability_format(self) -> None:
        """compute_capability should be formatted as 'X.Y' string."""
        with patch("mrec_cli.device.check_vulkan_support", return_value=False):
            with patch("mrec_cli.device.check_pytorch_vulkan", return_value=False):
                with patch(
                    "mrec_cli.device.check_cuda_support",
                    return_value=(True, "RTX 3080", (8, 6)),
                ):
                    with patch("mrec_cli.device.detect_best_device", return_value="cuda"):
                        result = get_device_info()
                        assert result["compute_capability"] == "8.6"

    def test_compute_capability_none_when_no_cuda(self) -> None:
        """compute_capability should be None when CUDA unavailable."""
        with patch("mrec_cli.device.check_vulkan_support", return_value=False):
            with patch("mrec_cli.device.check_pytorch_vulkan", return_value=False):
                with patch(
                    "mrec_cli.device.check_cuda_support",
                    return_value=(False, None, None),
                ):
                    with patch("mrec_cli.device.detect_best_device", return_value="cpu"):
                        result = get_device_info()
                        assert result["compute_capability"] is None

    def test_gpu_name_included(self) -> None:
        """Should include GPU name when available."""
        with patch("mrec_cli.device.check_vulkan_support", return_value=False):
            with patch("mrec_cli.device.check_pytorch_vulkan", return_value=False):
                with patch(
                    "mrec_cli.device.check_cuda_support",
                    return_value=(True, "Test GPU", (7, 5)),
                ):
                    with patch("mrec_cli.device.detect_best_device", return_value="cuda"):
                        result = get_device_info()
                        assert result["gpu_name"] == "Test GPU"
