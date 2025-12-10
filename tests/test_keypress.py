"""Tests for mrec_cli.keypress module."""

from __future__ import annotations

import io
import sys
from unittest.mock import MagicMock, patch

from mrec_cli.keypress import (
    KeypressHandler,
    check_keypress_nowait,
    raw_terminal,
    wait_for_keypress,
)


class TestRawTerminal:
    """Tests for raw_terminal context manager."""

    def test_non_tty_yields_without_modification(self) -> None:
        """Non-TTY stdin should yield without changing settings."""
        mock_stdin = io.StringIO()

        with patch.object(sys, "stdin", mock_stdin), patch("termios.tcgetattr") as mock_get:
            with patch("termios.tcsetattr") as mock_set:
                with patch("tty.setcbreak") as mock_cbreak:
                    with raw_terminal():
                        pass

                    # Should not be called for non-TTY
                    mock_get.assert_not_called()
                    mock_set.assert_not_called()
                    mock_cbreak.assert_not_called()

    def test_tty_sets_cbreak_mode(self) -> None:
        """TTY stdin should have cbreak mode set."""
        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = True
        mock_stdin.fileno.return_value = 0

        with patch.object(sys, "stdin", mock_stdin):
            with patch("termios.tcgetattr", return_value=[0, 1, 2, 3, 4, 5, 6]):
                with patch("termios.tcsetattr"):
                    with patch("tty.setcbreak") as mock_cbreak:
                        with raw_terminal():
                            mock_cbreak.assert_called_once_with(0)

    def test_restores_settings_on_exit(self) -> None:
        """Should restore original settings on exit."""
        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = True
        mock_stdin.fileno.return_value = 0
        old_settings = [1, 2, 3, 4, 5, 6, 7]

        with patch.object(sys, "stdin", mock_stdin):
            with patch("termios.tcgetattr", return_value=old_settings):
                with patch("termios.tcsetattr") as mock_set:
                    with patch("tty.setcbreak"):
                        with raw_terminal():
                            pass

                        # Should restore with TCSADRAIN
                        mock_set.assert_called()
                        call_args = mock_set.call_args[0]
                        assert call_args[2] == old_settings

    def test_restores_settings_on_exception(self) -> None:
        """Should restore settings even if exception occurs."""
        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = True
        mock_stdin.fileno.return_value = 0
        old_settings = [1, 2, 3, 4, 5, 6, 7]

        with patch.object(sys, "stdin", mock_stdin):
            with patch("termios.tcgetattr", return_value=old_settings):
                with patch("termios.tcsetattr") as mock_set:
                    with patch("tty.setcbreak"):
                        try:
                            with raw_terminal():
                                raise ValueError("Test error")
                        except ValueError:
                            pass

                        # Should still restore settings
                        mock_set.assert_called()


class TestWaitForKeypress:
    """Tests for wait_for_keypress function."""

    def test_non_tty_returns_false(self) -> None:
        """Non-TTY stdin should return False immediately."""
        mock_stdin = io.StringIO()

        with patch.object(sys, "stdin", mock_stdin):
            result = wait_for_keypress()
            assert result is False

    def test_target_key_pressed_returns_true(self) -> None:
        """Should return True when target key is pressed."""
        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = True
        mock_stdin.fileno.return_value = 0
        mock_stdin.read.return_value = " "

        with patch.object(sys, "stdin", mock_stdin):
            with patch("termios.tcgetattr", return_value=[]):
                with patch("termios.tcsetattr"):
                    with patch("tty.setcbreak"):
                        result = wait_for_keypress(target_key=" ")
                        assert result is True

    def test_different_key_returns_false(self) -> None:
        """Should return False when different key is pressed."""
        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = True
        mock_stdin.fileno.return_value = 0
        mock_stdin.read.return_value = "x"

        with patch.object(sys, "stdin", mock_stdin):
            with patch("termios.tcgetattr", return_value=[]):
                with patch("termios.tcsetattr"):
                    with patch("tty.setcbreak"):
                        result = wait_for_keypress(target_key=" ")
                        assert result is False

    def test_timeout_returns_false(self) -> None:
        """Should return False when timeout expires."""
        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = True
        mock_stdin.fileno.return_value = 0

        with patch.object(sys, "stdin", mock_stdin):
            with patch("termios.tcgetattr", return_value=[]):
                with patch("termios.tcsetattr"):
                    with patch("tty.setcbreak"):
                        with patch("select.select", return_value=([], [], [])):
                            result = wait_for_keypress(timeout=0.1)
                            assert result is False

    def test_space_is_default_target(self) -> None:
        """Default target key should be space."""
        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = True
        mock_stdin.fileno.return_value = 0
        mock_stdin.read.return_value = " "

        with patch.object(sys, "stdin", mock_stdin):
            with patch("termios.tcgetattr", return_value=[]):
                with patch("termios.tcsetattr"):
                    with patch("tty.setcbreak"):
                        result = wait_for_keypress()  # No target_key specified
                        assert result is True


class TestCheckKeypressNowait:
    """Tests for check_keypress_nowait function."""

    def test_non_tty_returns_false(self) -> None:
        """Non-TTY stdin should return False immediately."""
        mock_stdin = io.StringIO()

        with patch.object(sys, "stdin", mock_stdin):
            result = check_keypress_nowait()
            assert result is False

    def test_key_available_returns_true(self) -> None:
        """Should return True when target key is available."""
        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = True
        mock_stdin.fileno.return_value = 0
        mock_stdin.read.return_value = " "

        with patch.object(sys, "stdin", mock_stdin):
            with patch("termios.tcgetattr", return_value=[]):
                with patch("termios.tcsetattr"):
                    with patch("tty.setcbreak"):
                        with patch("select.select", return_value=([mock_stdin], [], [])):
                            result = check_keypress_nowait()
                            assert result is True

    def test_no_key_returns_false(self) -> None:
        """Should return False when no key is available."""
        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = True
        mock_stdin.fileno.return_value = 0

        with patch.object(sys, "stdin", mock_stdin):
            with patch("termios.tcgetattr", return_value=[]):
                with patch("termios.tcsetattr"):
                    with patch("tty.setcbreak"):
                        with patch("select.select", return_value=([], [], [])):
                            result = check_keypress_nowait()
                            assert result is False


class TestKeypressHandler:
    """Tests for KeypressHandler class."""

    def test_context_manager_enter_exit(self) -> None:
        """Should work as context manager."""
        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = True
        mock_stdin.fileno.return_value = 0

        with patch.object(sys, "stdin", mock_stdin):
            with patch("termios.tcgetattr", return_value=[]):
                with patch("termios.tcsetattr"):
                    with patch("tty.setcbreak"):
                        with KeypressHandler() as handler:
                            assert handler._active is True
                        assert handler._active is False

    def test_non_tty_works(self) -> None:
        """Should work with non-TTY stdin."""
        mock_stdin = io.StringIO()

        with patch.object(sys, "stdin", mock_stdin), KeypressHandler() as handler:
            assert handler._active is True
            assert handler._is_tty is False

    def test_get_key_returns_none_when_inactive(self) -> None:
        """get_key should return None when not active."""
        handler = KeypressHandler()
        result = handler.get_key()
        assert result is None

    def test_get_key_returns_none_non_tty(self) -> None:
        """get_key should return None for non-TTY."""
        mock_stdin = io.StringIO()

        with patch.object(sys, "stdin", mock_stdin), KeypressHandler() as handler:
            result = handler.get_key()
            assert result is None

    def test_get_key_timeout_returns_none(self) -> None:
        """get_key should return None on timeout."""
        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = True
        mock_stdin.fileno.return_value = 0

        with patch.object(sys, "stdin", mock_stdin):
            with patch("termios.tcgetattr", return_value=[]):
                with patch("termios.tcsetattr"):
                    with patch("tty.setcbreak"):
                        with patch("select.select", return_value=([], [], [])):
                            with KeypressHandler() as handler:
                                result = handler.get_key(timeout=0.1)
                                assert result is None

    def test_get_key_returns_char(self) -> None:
        """get_key should return pressed character."""
        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = True
        mock_stdin.fileno.return_value = 0
        mock_stdin.read.return_value = "a"

        with patch.object(sys, "stdin", mock_stdin):
            with patch("termios.tcgetattr", return_value=[]):
                with patch("termios.tcsetattr"):
                    with patch("tty.setcbreak"):
                        with patch("select.select", return_value=([mock_stdin], [], [])):
                            with KeypressHandler() as handler:
                                result = handler.get_key()
                                assert result == "a"

    def test_flush_input_non_tty(self) -> None:
        """flush_input should do nothing for non-TTY."""
        mock_stdin = io.StringIO()

        with patch.object(sys, "stdin", mock_stdin), patch("termios.tcflush") as mock_flush:
            with KeypressHandler() as handler:
                handler.flush_input()
                mock_flush.assert_not_called()

    def test_flush_input_handles_errors(self) -> None:
        """flush_input should handle termios errors."""
        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = True
        mock_stdin.fileno.return_value = 0

        import termios as real_termios

        with patch.object(sys, "stdin", mock_stdin):
            with patch("termios.tcgetattr", return_value=[]):
                with patch("termios.tcsetattr"):
                    with patch("tty.setcbreak"):
                        with patch("termios.tcflush", side_effect=real_termios.error("Error")):
                            with KeypressHandler() as handler:
                                # Should not raise
                                handler.flush_input()


class TestKeypressHandlerSettingsRestoration:
    """Tests for KeypressHandler settings restoration."""

    def test_restores_settings_on_normal_exit(self) -> None:
        """Should restore settings on normal exit."""
        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = True
        mock_stdin.fileno.return_value = 0
        old_settings = [1, 2, 3, 4, 5, 6, 7]

        with patch.object(sys, "stdin", mock_stdin):
            with patch("termios.tcgetattr", return_value=old_settings):
                with patch("termios.tcsetattr") as mock_set:
                    with patch("tty.setcbreak"):
                        with KeypressHandler():
                            pass

                        mock_set.assert_called()
                        call_args = mock_set.call_args[0]
                        assert call_args[2] == old_settings

    def test_restores_settings_on_exception(self) -> None:
        """Should restore settings even on exception."""
        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = True
        mock_stdin.fileno.return_value = 0
        old_settings = [1, 2, 3, 4, 5, 6, 7]

        with patch.object(sys, "stdin", mock_stdin):
            with patch("termios.tcgetattr", return_value=old_settings):
                with patch("termios.tcsetattr") as mock_set:
                    with patch("tty.setcbreak"):
                        try:
                            with KeypressHandler():
                                raise ValueError("Test")
                        except ValueError:
                            pass

                        mock_set.assert_called()
