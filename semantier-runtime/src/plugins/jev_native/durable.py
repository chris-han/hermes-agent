"""POSIX, file-backed one-shot receipts; no transport or credential handling.

The service owns the stable operation-directory mapping and its governed parent.
Call reserve() successfully before starting any worker. A consumed directory is
never reusable, including after torn writes. Raw hashes/completeness come from
the caller; this layer hashes only the already-redacted bytes it actually stores.
"""
from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import re
import stat
import uuid
from datetime import datetime, timezone
from pathlib import Path


class OperationBusy(RuntimeError):
    pass


class OperationConsumed(RuntimeError):
    pass


class BindingMismatch(RuntimeError):
    pass


class EvidenceCorrupt(OperationConsumed):
    pass


class UnsafePath(RuntimeError):
    pass


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _regular(fd, directory_fd, name):
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid():
        raise UnsafePath("EVIDENCE_NOT_OWNED_REGULAR_FILE")
    if info.st_nlink == 1:
        return
    # A crash between atomic link publication and temporary unlink leaves two
    # names for the same owned inode. Accept only that exact in-directory pair;
    # an external hard link (including a linked lock) remains forbidden.
    pattern = r"\.(response\.redacted\.bin|terminal\.json)\.[0-9a-f]{32}\.tmp"
    match = re.fullmatch(pattern, name)
    if match:
        peers = [match.group(1)]
    elif name in ("response.redacted.bin", "terminal.json"):
        peers = [entry for entry in os.listdir(directory_fd)
                 if re.fullmatch(pattern, entry) and entry.startswith("." + name + ".")]
    else:
        peers = []
    for peer in peers:
        other = os.stat(peer, dir_fd=directory_fd, follow_symlinks=False)
        if info.st_nlink == 2 and (other.st_dev, other.st_ino) == (info.st_dev, info.st_ino):
            return
    raise UnsafePath("EVIDENCE_EXTERNAL_HARDLINK")


def _open_file(directory_fd, name, flags, mode=0o600):
    try:
        fd = os.open(name, flags | os.O_NOFOLLOW | os.O_NONBLOCK,
                     mode, dir_fd=directory_fd)
    except OSError as error:
        if error.errno in (errno.ELOOP, errno.ENOTDIR):
            raise UnsafePath("SYMLINK_EVIDENCE") from None
        raise
    try:
        _regular(fd, directory_fd, name)
    except BaseException:
        os.close(fd)
        raise
    return fd


def _open_directory(directory, *, create=False):
    """Walk with directory FDs: never resolve an ancestor or leaf symlink."""
    path = Path(directory)
    if ".." in path.parts:
        raise UnsafePath("PARENT_TRAVERSAL")
    path = path.absolute()
    if path == Path("/"):
        raise UnsafePath("OPERATION_DIRECTORY_REQUIRED")
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for index, component in enumerate(path.parts[1:]):
            last = index == len(path.parts) - 2
            if last and create:
                try:
                    os.mkdir(component, mode=0o700, dir_fd=fd)
                except FileExistsError:
                    pass
                # Also covers a concurrent creator not yet fsyncing its parent.
                os.fsync(fd)
            try:
                child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                dir_fd=fd)
            except OSError as error:
                if error.errno in (errno.ELOOP, errno.ENOTDIR):
                    raise UnsafePath("SYMLINK_DIRECTORY") from None
                raise
            os.close(fd)
            fd = child
        info = os.fstat(fd)
        if info.st_uid != os.geteuid() or info.st_mode & 0o022:
            raise UnsafePath("OPERATION_DIRECTORY_NOT_PRIVATE_OWNER")
        return fd
    except BaseException:
        os.close(fd)
        raise


def _read(directory_fd, name):
    try:
        fd = _open_file(directory_fd, name, os.O_RDONLY)
    except FileNotFoundError:
        return None
    with os.fdopen(fd, "rb") as stream:
        return stream.read()


def _object(raw):
    try:
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError
        return value
    except (ValueError, TypeError, UnicodeDecodeError):
        raise EvidenceCorrupt("INVALID_EVIDENCE_JSON") from None


def _reservation_state(directory_fd):
    raw = _read(directory_fd, "reservation.json")
    if raw is None:
        return "absent"
    try:
        value = _object(raw)
        valid = (value.get("schema_version") == "JevReservationV1"
                 and value.get("attempts_consumed") == 1
                 and value.get("retry_allowance") == 0
                 and isinstance(value.get("binding"), dict))
    except EvidenceCorrupt:
        valid = False
    return "valid" if valid else "corrupt"


def inspect(directory):
    """Read existing evidence without creating a directory, lock, or receipt.

    Inspection is a snapshot, not permission to dispatch. Callers must use the
    exclusive Operation context for reservation, completion, or recovery.
    """
    fd = _open_directory(directory)
    try:
        binding_raw = _read(fd, "binding.json")
        terminal_raw = _read(fd, "terminal.json")
        return {
            "consumed": any(name != "operation.lock" for name in os.listdir(fd)),
            "binding": _object(binding_raw) if binding_raw is not None else None,
            "reservation_state": _reservation_state(fd),
            "terminal": _object(terminal_raw) if terminal_raw is not None else None,
        }
    finally:
        os.close(fd)


class Operation:
    """An exclusive owner of one caller-supplied, stable operation directory."""

    def __init__(self, directory, binding: dict, request_bytes: bytes):
        self.directory = Path(directory)
        self._binding = _object(_json(binding))  # Immutable caller-input snapshot.
        required = ("operation_id", "profile_home", "request_sha256", "model", "endpoint")
        if any(not isinstance(self._binding.get(key), str) or not self._binding[key]
               for key in required):
            raise BindingMismatch("REQUIRED_BINDING_MISSING")
        if not isinstance(request_bytes, bytes):
            raise TypeError("request_bytes must be bytes")
        self._request = request_bytes
        if self._binding["request_sha256"] != _sha(request_bytes):
            raise BindingMismatch("REQUEST_HASH_MISMATCH")
        self._directory_fd = None
        self._lock_fd = None
        self._reserved = False
        self._finalizing = False

    def _binding_state(self, raw):
        if raw is None:
            return "absent"
        try:
            value = _object(raw)
        except EvidenceCorrupt:
            return "corrupt"
        # Every available binding field is a constraint, even in a partial
        # object. Recovery cannot relabel a known profile/model/request.
        if any(key in self._binding and item != self._binding[key]
               for key, item in value.items()):
            raise BindingMismatch("OPERATION_BINDING_CHANGED")
        if value == self._binding:
            return "intact"
        if set(self._binding).issubset(value):
            raise BindingMismatch("OPERATION_BINDING_CHANGED")
        return "incomplete"

    def _request_state(self, raw):
        if raw is None:
            return "absent"
        if raw == self._request:
            return "intact"
        if self._request.startswith(raw):
            return "incomplete"
        try:
            json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            return "corrupt"
        # The service prepares JSON requests. A complete different JSON value
        # is conflicting evidence, rather than a torn write to reinterpret.
        raise BindingMismatch("OPERATION_REQUEST_CHANGED")

    def _open(self, *, recovering=False):
        if self._directory_fd is not None:
            raise OperationBusy("OPERATION_ALREADY_OPEN")
        self._directory_fd = _open_directory(self.directory, create=True)
        try:
            self._lock_fd = _open_file(
                self._directory_fd, "operation.lock", os.O_RDWR | os.O_CREAT)
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise OperationBusy("OPERATION_IN_PROGRESS") from None
            os.fsync(self._lock_fd)
            os.fsync(self._directory_fd)
            for name in os.listdir(self._directory_fd):
                fd = _open_file(self._directory_fd, name, os.O_RDONLY)
                os.close(fd)
            self._recovery_evidence = {
                "binding.json": self._binding_state(_read(self._directory_fd, "binding.json")),
                "request.bin": self._request_state(_read(self._directory_fd, "request.bin")),
            }
            if not recovering and any(state in ("corrupt", "incomplete")
                                      for state in self._recovery_evidence.values()):
                raise EvidenceCorrupt("INCOMPLETE_OPERATION_EVIDENCE")
            reservation = _read(self._directory_fd, "reservation.json")
            if reservation is not None:
                try:
                    reservation_binding = _object(reservation).get("binding")
                except EvidenceCorrupt:
                    reservation_binding = None
                if isinstance(reservation_binding, dict):
                    self._binding_state(_json(reservation_binding))
            terminal = _read(self._directory_fd, "terminal.json")
            if terminal is not None and _object(terminal).get("binding") != self._binding:
                raise BindingMismatch("TERMINAL_BINDING_CHANGED")
        except BaseException:
            self._close()
            raise

    def _close(self):
        if self._lock_fd is not None:
            # Close, never unlink or replace the lock inode.
            os.close(self._lock_fd)
            self._lock_fd = None
        if self._directory_fd is not None:
            os.close(self._directory_fd)
            self._directory_fd = None

    def _consumed(self):
        return any(name != "operation.lock" for name in os.listdir(self._directory_fd))

    def __enter__(self):
        self._open()
        if self._consumed():
            self._close()
            raise OperationConsumed("OPERATION_ALREADY_CONSUMED")
        return self

    def __exit__(self, *exc):
        self._close()

    def _write_once(self, name, raw, *, atomic=False):
        temporary = "." + name + "." + uuid.uuid4().hex + ".tmp" if atomic else name
        fd = _open_file(self._directory_fd, temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        try:
            remaining = memoryview(raw)
            while remaining:
                written = os.write(fd, remaining)
                if written <= 0:
                    raise OSError("SHORT_EVIDENCE_WRITE")
                remaining = remaining[written:]
            os.fsync(fd)
        finally:
            os.close(fd)
        if atomic:
            # Atomic publication without replacement, even if a final already exists.
            os.link(temporary, name, src_dir_fd=self._directory_fd,
                    dst_dir_fd=self._directory_fd, follow_symlinks=False)
            os.unlink(temporary, dir_fd=self._directory_fd)
        os.fsync(self._directory_fd)

    def reserve(self):
        """Durably consume the sole attempt; on any failure, do not dispatch."""
        if self._directory_fd is None:
            raise RuntimeError("OPERATION_NOT_OPEN")
        if self._reserved or self._consumed():
            raise OperationConsumed("OPERATION_ALREADY_CONSUMED")
        self._write_once("binding.json", _json(self._binding))
        self._write_once("request.bin", self._request)
        reservation = {
            "schema_version": "JevReservationV1", "reserved_at": _now(),
            "binding": self._binding, "attempts_consumed": 1, "retry_allowance": 0,
        }
        self._write_once("reservation.json", _json(reservation))
        self._reserved = True

    def _seal(self, receipt, response_bytes, *, recovering=False):
        if not isinstance(response_bytes, bytes):
            raise TypeError("response_bytes must be bytes")
        terminal = _object(_json(receipt))
        terminal.setdefault("raw_response_sha256", None)
        terminal.setdefault("response_complete", None)
        terminal.update(
            schema_version="JevOperationTerminalV1", sealed_at=_now(),
            binding=self._binding, attempts_consumed=1, retry_allowance=0,
            retained_response_sha256=_sha(response_bytes),
            retained_response_bytes=len(response_bytes),
        )
        terminal_raw = _json(terminal)
        existing = _read(self._directory_fd, "response.redacted.bin")
        if existing is None:
            self._write_once("response.redacted.bin", response_bytes, atomic=True)
        elif not recovering or existing != response_bytes:
            raise EvidenceCorrupt("RESPONSE_ALREADY_EXISTS")
        self._write_once("terminal.json", terminal_raw, atomic=True)
        return terminal

    def complete(self, receipt: dict, response_bytes: bytes):
        """Return only after response evidence and terminal are durable."""
        if self._directory_fd is None or not self._reserved:
            raise RuntimeError("DURABLE_RESERVATION_REQUIRED")
        if self._finalizing:
            raise OperationConsumed("TERMINAL_ALREADY_ATTEMPTED")
        self._finalizing = True
        return self._seal(receipt, response_bytes)

    def recover(self):
        """Seal consumed state, retaining torn evidence; never dispatch or retry."""
        self._open(recovering=True)
        try:
            if not self._consumed():
                raise RuntimeError("NO_OPERATION_TO_RECOVER")
            raw = _read(self._directory_fd, "terminal.json")
            if raw is not None:
                terminal = _object(raw)
                response = _read(self._directory_fd, "response.redacted.bin")
                if (response is None
                        or terminal.get("retained_response_sha256") != _sha(response)
                        or terminal.get("retained_response_bytes") != len(response)):
                    raise EvidenceCorrupt("TERMINAL_RESPONSE_MISMATCH")
                # A previous completion might have failed its final fsync.
                for name in ("response.redacted.bin", "terminal.json"):
                    fd = _open_file(self._directory_fd, name, os.O_RDONLY)
                    try:
                        os.fsync(fd)
                    finally:
                        os.close(fd)
                os.fsync(self._directory_fd)
                return terminal
            response = _read(self._directory_fd, "response.redacted.bin") or b""
            return self._seal({
                "disposition": "INTERRUPTED_UNKNOWN",
                "reason": "CONSUMED_WITHOUT_TERMINAL",
                "wire_attempts": None,
                "remote_outcome": "unknown",
                "reservation_state": _reservation_state(self._directory_fd),
                "recovery_evidence": self._recovery_evidence,
                "raw_response_sha256": None,
                "response_complete": None,
            }, response, recovering=True)
        finally:
            self._close()
