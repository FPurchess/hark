"""Tests for hark.platform module."""

from unittest.mock import patch

from hark.platform import (
    Platform,
    get_platform,
    is_linux,
    is_macos,
    is_windows,
)


class TestPlatformEnum:
    """Tests for Platform enum."""

    def test_platform_linux_value(self) -> None:
        """Platform.LINUX should have value 'linux'."""
        assert Platform.LINUX.value == "linux"

    def test_platform_macos_value(self) -> None:
        """Platform.MACOS should have value 'darwin'."""
        assert Platform.MACOS.value == "darwin"

    def test_platform_windows_value(self) -> None:
        """Platform.WINDOWS should have value 'win32'."""
        assert Platform.WINDOWS.value == "win32"

    def test_platform_unknown_value(self) -> None:
        """Platform.UNKNOWN should have value 'unknown'."""
        assert Platform.UNKNOWN.value == "unknown"

    def test_all_platforms_unique(self) -> None:
        """All platform values should be unique."""
        values = [p.value for p in Platform]
        assert len(values) == len(set(values))


class TestGetPlatform:
    """Tests for get_platform function."""

    def test_linux_detection(self) -> None:
        """Should detect Linux platform."""
        with patch("hark.platform.sys.platform", "linux"):
            assert get_platform() == Platform.LINUX

    def test_linux_detection_with_suffix(self) -> None:
        """Should detect Linux platform with kernel version suffix."""
        with patch("hark.platform.sys.platform", "linux2"):
            assert get_platform() == Platform.LINUX

    def test_linux_detection_with_arch(self) -> None:
        """Should detect Linux platform with architecture suffix."""
        with patch("hark.platform.sys.platform", "linux-aarch64"):
            assert get_platform() == Platform.LINUX

    def test_macos_detection(self) -> None:
        """Should detect macOS platform."""
        with patch("hark.platform.sys.platform", "darwin"):
            assert get_platform() == Platform.MACOS

    def test_windows_detection(self) -> None:
        """Should detect Windows platform."""
        with patch("hark.platform.sys.platform", "win32"):
            assert get_platform() == Platform.WINDOWS

    def test_unknown_platform(self) -> None:
        """Should return UNKNOWN for unrecognized platforms."""
        with patch("hark.platform.sys.platform", "freebsd"):
            assert get_platform() == Platform.UNKNOWN

    def test_cygwin_is_unknown(self) -> None:
        """Cygwin should be detected as UNKNOWN (not Linux or Windows)."""
        with patch("hark.platform.sys.platform", "cygwin"):
            assert get_platform() == Platform.UNKNOWN


class TestIsLinux:
    """Tests for is_linux function."""

    def test_is_linux_on_linux(self) -> None:
        """Should return True on Linux."""
        with patch("hark.platform.sys.platform", "linux"):
            assert is_linux() is True

    def test_is_linux_on_macos(self) -> None:
        """Should return False on macOS."""
        with patch("hark.platform.sys.platform", "darwin"):
            assert is_linux() is False

    def test_is_linux_on_windows(self) -> None:
        """Should return False on Windows."""
        with patch("hark.platform.sys.platform", "win32"):
            assert is_linux() is False

    def test_is_linux_on_unknown(self) -> None:
        """Should return False on unknown platforms."""
        with patch("hark.platform.sys.platform", "freebsd"):
            assert is_linux() is False


class TestIsMacos:
    """Tests for is_macos function."""

    def test_is_macos_on_linux(self) -> None:
        """Should return False on Linux."""
        with patch("hark.platform.sys.platform", "linux"):
            assert is_macos() is False

    def test_is_macos_on_macos(self) -> None:
        """Should return True on macOS."""
        with patch("hark.platform.sys.platform", "darwin"):
            assert is_macos() is True

    def test_is_macos_on_windows(self) -> None:
        """Should return False on Windows."""
        with patch("hark.platform.sys.platform", "win32"):
            assert is_macos() is False

    def test_is_macos_on_unknown(self) -> None:
        """Should return False on unknown platforms."""
        with patch("hark.platform.sys.platform", "freebsd"):
            assert is_macos() is False


class TestIsWindows:
    """Tests for is_windows function."""

    def test_is_windows_on_linux(self) -> None:
        """Should return False on Linux."""
        with patch("hark.platform.sys.platform", "linux"):
            assert is_windows() is False

    def test_is_windows_on_macos(self) -> None:
        """Should return False on macOS."""
        with patch("hark.platform.sys.platform", "darwin"):
            assert is_windows() is False

    def test_is_windows_on_windows(self) -> None:
        """Should return True on Windows."""
        with patch("hark.platform.sys.platform", "win32"):
            assert is_windows() is True

    def test_is_windows_on_unknown(self) -> None:
        """Should return False on unknown platforms."""
        with patch("hark.platform.sys.platform", "freebsd"):
            assert is_windows() is False


class TestExports:
    """Tests for module exports."""

    def test_all_exports(self) -> None:
        """__all__ should include all public functions and classes."""
        from hark import platform

        expected = {"Platform", "get_platform", "is_linux", "is_macos", "is_windows"}
        assert set(platform.__all__) == expected
