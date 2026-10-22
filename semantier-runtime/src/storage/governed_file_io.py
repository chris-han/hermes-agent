"""File primitives for the governed immutable publishers, not a store/authority owner.

All participating writers use stable sidecar locks. Readers never lock immutable
committed paths. Errors propagate; synchronization is never silently skipped.
"""
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import re
import shutil

from pm.filesystem import lock_fd, long_root


@contextmanager
def exclusive_file_lock(path: Path, *, wait: bool = True):
    # No truncation/unlink: replacing a sidecar could split the lock identity.
    # Python's descriptors are non-inheritable; close releases the OS lock even
    # when the body/cleanup raises. Windows locks byte zero, including beyond EOF.
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o666)
    acquired = False
    try:
        if not lock_fd(descriptor, wait=wait):
            raise BlockingIOError("governed storage lock is held")
        acquired = True
        yield
    finally:
        try:
            if acquired:
                if os.name == 'nt':
                    import msvcrt
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    # Preserve explicit POSIX scope release even if fork copied
                    # this open file description; close alone would not do so.
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def sync_directory(path: Path) -> None:
    """Request OS synchronization; never equate this with tested power-loss survival.

    Win32 requires a writable directory handle and BACKUP_SEMANTICS. A read-only
    handle fails FlushFileBuffers with access denied even when opening succeeds.
    Unsupported filesystem/access behavior is an error, not a weaker fallback.
    """
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                      ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        kernel.CreateFileW.restype = wintypes.HANDLE
        kernel.FlushFileBuffers.argtypes = [wintypes.HANDLE]
        kernel.FlushFileBuffers.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        # GENERIC_WRITE, FILE_SHARE_READ|WRITE|DELETE, OPEN_EXISTING,
        # FILE_FLAG_BACKUP_SEMANTICS. This is not a volume/admin flush.
        handle = kernel.CreateFileW(str(long_root(path.absolute())), 0x40000000, 7, None, 3, 0x02000000, None)
        if handle == wintypes.HANDLE(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if not kernel.FlushFileBuffers(handle):
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            if not kernel.CloseHandle(handle):
                raise ctypes.WinError(ctypes.get_last_error())
    else:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def clean_publication_temps(target: Path) -> None:
    """Reap only this publisher's exact UUID staging names while its lock is held.

    A killed process may leave uncommitted bytes. Never promote them, follow a
    symlink, or touch the stable lock file/another publisher's staging names.
    """
    pattern = re.compile(re.escape("." + target.name + ".") + r"[0-9a-f]{32}\.tmp\Z")
    for entry in target.parent.iterdir():
        if not pattern.fullmatch(entry.name):
            continue
        if entry.is_symlink() or (os.name == 'nt' and entry.is_junction()):
            raise ValueError("governed publication temporary must not be a symlink")
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()
