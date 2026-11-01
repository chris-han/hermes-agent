"""Core locking/long-path primitives; provenance in filesystem-provenance.md."""
import errno
import ntpath
import os
from pathlib import Path
import time

_LOCK_POLL_SECONDS = 0.05

_VERBATIM = "\\\\?\\"
_VERBATIM_UNC = "\\\\?\\UNC\\"


def _long_form(text: str) -> str:
    if text.startswith(_VERBATIM):
        return text
    # Windows stops normalising a verbatim path, so it must already be absolute,
    # backslashed and free of `..` before the prefix goes on.
    text = ntpath.abspath(text)
    if text.startswith("\\\\"):
        return _VERBATIM_UNC + text[2:]
    return _VERBATIM + text


def long_root(path: Path) -> Path:
    """Spell a core-owned root so Windows file calls beneath it are not cut at
    MAX_PATH when LongPathsEnabled is 0 (the default). Every path joined onto the
    result inherits the spelling; relative paths and digests are unchanged."""
    return Path(_long_form(os.fspath(path))) if os.name == "nt" else path


def lock_fd(fd: int, *, wait: bool, timeout: float | None = None) -> bool:
    """Take the byte lock; ``timeout`` bounds the retry loop (None waits forever, 0 tries once)."""
    deadline = None if timeout is None else time.monotonic() + timeout
    if os.name == "nt":
        import msvcrt
        while True:
            try:
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                return True
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                    raise
            if not wait or (deadline is not None and time.monotonic() >= deadline):
                return False
            time.sleep(_LOCK_POLL_SECONDS)
    else:
        import fcntl
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            except BlockingIOError:
                pass
            if not wait or (deadline is not None and time.monotonic() >= deadline):
                return False
            time.sleep(_LOCK_POLL_SECONDS)
