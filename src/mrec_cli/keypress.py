"""Terminal keypress detection for mrec-cli."""

from __future__ import annotations

import contextlib
import select
import sys
import termios
import tty
from collections.abc import Generator
from contextlib import contextmanager

__all__ = [
    "raw_terminal",
    "wait_for_keypress",
    "check_keypress_nowait",
    "KeypressHandler",
]


@contextmanager
def raw_terminal() -> Generator[None, None, None]:
    """
    Context manager for raw terminal mode.

    Uses setcbreak instead of setraw to preserve Ctrl+C handling.
    """
    if not sys.stdin.isatty():
        yield
        return

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def wait_for_keypress(target_key: str = " ", timeout: float | None = None) -> bool:
    """
    Wait for a specific keypress.

    Args:
        target_key: The key to wait for (default: space).
        timeout: Optional timeout in seconds.

    Returns:
        True if target key was pressed, False if timeout or different key.
    """
    if not sys.stdin.isatty():
        return False

    with raw_terminal():
        if timeout is not None:
            ready, _, _ = select.select([sys.stdin], [], [], timeout)
            if not ready:
                return False

        key = sys.stdin.read(1)
        return key == target_key


def check_keypress_nowait(target_key: str = " ") -> bool:
    """
    Non-blocking check if a specific key was pressed.

    Args:
        target_key: The key to check for (default: space).

    Returns:
        True if target key was pressed, False otherwise.
    """
    if not sys.stdin.isatty():
        return False

    with raw_terminal():
        ready, _, _ = select.select([sys.stdin], [], [], 0)
        if ready:
            key = sys.stdin.read(1)
            return key == target_key
        return False


class KeypressHandler:
    """
    Context manager for continuous keypress monitoring.

    Usage:
        with KeypressHandler() as handler:
            while True:
                key = handler.get_key(timeout=0.1)
                if key == " ":
                    break
    """

    def __init__(self) -> None:
        self._old_settings: list[int] | None = None
        self._active = False
        self._is_tty = sys.stdin.isatty()

    def __enter__(self) -> KeypressHandler:
        if not self._is_tty:
            self._active = True
            return self

        fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        self._active = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        if self._old_settings is not None:
            termios.tcsetattr(
                sys.stdin.fileno(),
                termios.TCSADRAIN,
                self._old_settings,  # pyrefly: ignore[bad-argument-type]
            )
        self._active = False

    def get_key(self, timeout: float = 0.1) -> str | None:
        """
        Get pressed key with timeout.

        Args:
            timeout: Time to wait for a key in seconds.

        Returns:
            The pressed key character, or None if no key was pressed.
        """
        if not self._active:
            return None

        if not self._is_tty:
            return None

        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        if ready:
            return sys.stdin.read(1)
        return None

    def flush_input(self) -> None:
        """Flush any pending input from stdin."""
        if not self._is_tty:
            return

        with contextlib.suppress(termios.error):
            termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
