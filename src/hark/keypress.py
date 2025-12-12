"""Terminal keypress detection for hark."""

import contextlib
import sys
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING

from hark.platform import is_windows

__all__ = [
    "raw_terminal",
    "wait_for_keypress",
    "check_keypress_nowait",
    "KeypressHandler",
]

# Conditional imports based on platform
if is_windows():
    import msvcrt
else:
    import select
    import termios
    import tty

# Type hints for platform-specific modules (for IDE support)
if TYPE_CHECKING:
    import msvcrt
    import select
    import termios
    import tty


@contextmanager
def raw_terminal() -> Generator[None, None, None]:
    """
    Context manager for raw terminal mode.

    On POSIX: Uses setcbreak to preserve Ctrl+C handling.
    On Windows: No-op (msvcrt functions handle raw input natively).
    """
    if not sys.stdin.isatty():
        yield
        return

    if is_windows():
        # Windows: msvcrt functions already work in "raw" mode
        # No terminal mode change needed
        yield
        return

    # POSIX implementation
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def _wait_for_keypress_windows(target_key: str, timeout: float | None) -> bool:
    """Windows implementation using msvcrt."""
    start = time.time()
    poll_interval = 0.01  # 10ms polling

    while True:
        if msvcrt.kbhit():  # pyrefly: ignore[missing-attribute]
            # getwch() returns str (Unicode), getch() returns bytes
            key = msvcrt.getwch()  # pyrefly: ignore[missing-attribute]
            return key == target_key

        if timeout is not None and (time.time() - start) >= timeout:
            return False

        time.sleep(poll_interval)


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

    if is_windows():
        return _wait_for_keypress_windows(target_key, timeout)

    # POSIX implementation
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

    if is_windows():
        if msvcrt.kbhit():  # pyrefly: ignore[missing-attribute]
            key = msvcrt.getwch()  # pyrefly: ignore[missing-attribute]
            return key == target_key
        return False

    # POSIX implementation
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

    def __enter__(self) -> "KeypressHandler":
        if not self._is_tty:
            self._active = True
            return self

        if is_windows():
            # Windows: No setup needed, msvcrt works directly
            self._active = True
            return self

        # POSIX: Save terminal settings and enter cbreak mode
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
        if is_windows():
            # Windows: No cleanup needed
            self._active = False
            return

        # POSIX: Restore terminal settings
        if self._old_settings is not None:
            termios.tcsetattr(
                sys.stdin.fileno(),
                termios.TCSADRAIN,
                self._old_settings,  # pyrefly: ignore[bad-argument-type]
            )
        self._active = False

    def _get_key_windows(self, timeout: float) -> str | None:
        """Windows implementation of get_key with timeout."""
        start = time.time()
        poll_interval = 0.01  # 10ms polling

        while (time.time() - start) < timeout:
            if msvcrt.kbhit():  # pyrefly: ignore[missing-attribute]
                return msvcrt.getwch()  # pyrefly: ignore[missing-attribute]
            time.sleep(poll_interval)

        return None

    def get_key(self, timeout: float = 0.1) -> str | None:
        """
        Get pressed key with timeout.

        Args:
            timeout: Time to wait for a key in seconds.

        Returns:
            The pressed key character, or None if no key was pressed.
        """
        if not self._active or not self._is_tty:
            return None

        if is_windows():
            return self._get_key_windows(timeout)

        # POSIX implementation
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        if ready:
            return sys.stdin.read(1)
        return None

    def flush_input(self) -> None:
        """Flush any pending input from stdin."""
        if not self._is_tty:
            return

        if is_windows():
            # Windows: Consume all pending keypresses
            while msvcrt.kbhit():  # pyrefly: ignore[missing-attribute]
                msvcrt.getch()  # pyrefly: ignore[missing-attribute]
            return

        # POSIX: Flush terminal input buffer
        with contextlib.suppress(termios.error):
            termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
