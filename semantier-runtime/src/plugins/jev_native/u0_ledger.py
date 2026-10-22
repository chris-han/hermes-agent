"""Finite offline U0 accounting around the existing one-shot Operation.

The caller owns the fixed governed parent and transport/conformance decisions.
Neither constructing this guard nor a reservation grants paid-call authority.
No path, profile, home, result, or caller-supplied cost can renew an allowance.
"""
from __future__ import annotations

from contextlib import contextmanager
from decimal import Decimal
import errno
import fcntl
import os
from pathlib import Path
import re

from . import durable

EXPERIMENT_ID = "u0-reference-conformance-v1"
PROPOSED_BUDGET_USD = Decimal("0.20")
ROLE_RESERVES_USD = {"R1": Decimal("0.008781824"), "R2": Decimal("0.005505024")}
_MODELS = {"R1": "qwen3.5-plus", "R2": "jev-1.13.0"}
_ENDPOINTS = {"R1": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
              "R2": "https://api.typesafe.ai/v1/systemone"}
_ENTRY_FIELDS = ("role", "case_id", "request_sha256", "model", "endpoint")
_OLD_SPECIMEN = "e160b92c55bdf0c16eb17668a2233fdb21aa3fe794b02c08afa7348627eac19d"
_SOURCE_MANIFEST = "2758a15245f01e96c11d6a4b47acb50ccd46d33eedbdb5fe94751d462174710e"
_SOURCE_READINESS = "0cda99c62a0289b5875da9081f93d39910bddff99a71b9c3fccf582b7dec1c5f"


class ExperimentStopped(durable.OperationConsumed):
    pass


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _same_inode(left, right):
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _read_object(fd, name):
    raw = durable._read(fd, name)
    if raw is None:
        raise durable.EvidenceCorrupt("U0_INCOMPLETE_EVIDENCE")
    value = durable._object(raw)
    if durable._json(value) != raw:
        raise durable.EvidenceCorrupt("U0_NONCANONICAL_EVIDENCE")
    return value


def _child_fd(fd, name):
    try:
        child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
    except OSError as error:
        if error.errno in (errno.ELOOP, errno.ENOTDIR):
            raise durable.UnsafePath("U0_UNSAFE_CHILD_DIRECTORY") from None
        raise
    info = os.fstat(child)
    if info.st_uid != os.geteuid() or info.st_mode & 0o022:
        os.close(child)
        raise durable.UnsafePath("U0_CHILD_DIRECTORY_NOT_PRIVATE_OWNER")
    return child


class _Scope:
    """Only locking/anchoring here; all evidence IO reuses durable primitives."""

    _write_once = durable.Operation._write_once

    def __init__(self, root, *, read_only=False):
        self.root = root
        self.read_only = read_only
        self.directory = root / EXPERIMENT_ID
        self._parent_fd = self._directory_fd = self._lock_fd = None

    def __enter__(self):
        try:
            self._parent_fd = durable._open_directory(self.root)
            if not self.read_only:
                try:
                    os.mkdir(EXPERIMENT_ID, mode=0o700, dir_fd=self._parent_fd)
                except FileExistsError:
                    pass
                os.fsync(self._parent_fd)
            self._directory_fd = _child_fd(self._parent_fd, EXPERIMENT_ID)
            self._lock_fd = durable._open_file(
                self._directory_fd, "experiment.lock", os.O_RDWR | (0 if self.read_only else os.O_CREAT))
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise durable.OperationBusy("U0_EXPERIMENT_IN_PROGRESS") from None
            if not self.read_only:
                os.fsync(self._lock_fd)
                os.fsync(self._directory_fd)
            self.ensure_current()
            return self
        except BaseException:
            self.__exit__()
            raise

    def __exit__(self, *exc):
        for attr in ("_lock_fd", "_directory_fd", "_parent_fd"):
            fd = getattr(self, attr)
            if fd is not None:
                os.close(fd)
                setattr(self, attr, None)

    def ensure_current(self):
        probe = durable._open_directory(self.directory)
        try:
            if not _same_inode(os.fstat(probe), os.fstat(self._directory_fd)):
                raise durable.UnsafePath("U0_SCOPE_REPLACED")
        finally:
            os.close(probe)
        info = os.stat("experiment.lock", dir_fd=self._directory_fd, follow_symlinks=False)
        if not _same_inode(info, os.fstat(self._lock_fd)):
            raise durable.UnsafePath("U0_LOCK_REPLACED")
        durable._regular(self._lock_fd, self._directory_fd, "experiment.lock")

    def stop(self, reason):
        # Never replace or repair stop evidence, even if it is torn.
        self.ensure_current()
        if durable._read(self._directory_fd, "stop.json") is None:
            self._write_once("stop.json", durable._json({
                "schema_version": "U0StopV1", "experiment_id": EXPERIMENT_ID,
                "stopped_at": durable._now(), "reason": reason,
                "retry_allowance": 0, "allowance_reclaimed_usd": "0",
            }))


class Experiment:
    """A fixed 20-request manifest under a caller-owned, stable governed root.

    Use one context per attempted transport. Its reservation is durable before
    yielding and its exclusive lock remains held through terminal/STOP sealing.
    The caller must always use the same root; this class has no home/env fallback.
    """

    def __init__(self, governed_root, manifest: dict):
        self.root = Path(governed_root)
        if not self.root.is_absolute() or ".." in self.root.parts:
            raise durable.UnsafePath("U0_ABSOLUTE_GOVERNED_ROOT_REQUIRED")
        self.directory = self.root / EXPERIMENT_ID
        self._manifest = durable._object(durable._json(manifest))
        self_hash = self._manifest.pop("prepared_manifest_sha256", None)
        self.manifest_sha256 = durable._sha(durable._json(self._manifest))
        if self_hash is not None and self_hash != self.manifest_sha256:
            raise durable.BindingMismatch("U0_MANIFEST_HASH_MISMATCH")
        entries = self._manifest.get("entries")
        if (self._manifest.get("schema_version") != "U0PreparedManifestV1"
                or self._manifest.get("experiment_id") != EXPERIMENT_ID
                or self._manifest.get("source_manifest_sha256") != _SOURCE_MANIFEST
                or self._manifest.get("source_readiness_sha256") != _SOURCE_READINESS
                or not isinstance(entries, list) or len(entries) != 20):
            raise durable.BindingMismatch("U0_EXACT_MANIFEST_REQUIRED")
        policy_hash = self._manifest.get("route_policy_sha256")
        if not _digest(policy_hash):
            raise durable.BindingMismatch("U0_ROUTE_POLICY_HASH_INVALID")
        self._entries = {}
        hashes, counts = set(), {"R1": 0, "R2": 0}
        for entry in entries:
            if (not isinstance(entry, dict)
                    or any(not isinstance(entry.get(key), str) or not entry[key]
                           for key in _ENTRY_FIELDS)
                    or entry["role"] not in _MODELS
                    or entry["model"] != _MODELS[entry["role"]]
                    or entry["endpoint"] != _ENDPOINTS[entry["role"]]
                    or not _digest(entry["request_sha256"])
                    or entry["request_sha256"] == _OLD_SPECIMEN):
                raise durable.BindingMismatch("U0_INVALID_MANIFEST_ENTRY")
            pair = (entry["role"], entry["case_id"])
            if pair in self._entries or entry["request_sha256"] in hashes:
                raise durable.BindingMismatch("U0_DUPLICATE_MANIFEST_ENTRY")
            self._entries[pair] = entry
            hashes.add(entry["request_sha256"])
            counts[entry["role"]] += 1
        if counts != {"R1": 10, "R2": 10}:
            raise durable.BindingMismatch("U0_ROLE_LIMIT")
        cases = sorted({entry["case_id"] for entry in entries})
        self._order = [(role, case_id) for case_id in cases for role in ("R1", "R2")]
        if len(cases) != 10 or list(self._entries) != self._order:
            raise durable.BindingMismatch("U0_CASE_ROLE_ORDER_REQUIRED")
        self._manifest_record = {"schema_version": "U0LedgerManifestV1",
                                 "prepared_manifest_sha256": self.manifest_sha256,
                                 "manifest": self._manifest}

    def _binding(self, value, request_bytes=None):
        value = durable._object(durable._json(value))
        entry = self._entries.get((value.get("role"), value.get("case_id")))
        if (entry is None or any(value.get(key) != entry[key] for key in _ENTRY_FIELDS)
                or not isinstance(value.get("profile_home"), str) or not value["profile_home"]):
            raise durable.BindingMismatch("U0_REQUEST_NOT_ALLOWLISTED")
        policy_hash = self._manifest.get("route_policy_sha256")
        if value.get("route_policy_sha256") != policy_hash:
            raise durable.BindingMismatch("U0_ROUTE_POLICY_CHANGED")
        expected = value["request_sha256"]
        if request_bytes is not None and (
                not isinstance(request_bytes, bytes) or durable._sha(request_bytes) != expected):
            raise durable.BindingMismatch("U0_WIRE_HASH_MISMATCH")
        if value.get("operation_id", expected) != expected:
            raise durable.BindingMismatch("U0_OPERATION_ID_CHANGED")
        value["operation_id"] = expected
        return value

    def _child_terminal(self, scope, index, binding, *, terminal_inspection=False):
        name = f"attempt-{index:02d}"
        fd = _child_fd(scope._directory_fd, name)
        try:
            required = {"operation.lock", "binding.json", "request.bin", "reservation.json",
                        "response.redacted.bin", "terminal.json"}
            names = set(os.listdir(fd))
            if not required.issubset(names):
                raise durable.EvidenceCorrupt("U0_CHILD_INCOMPLETE")
            for item in names:
                if item not in required and re.fullmatch(
                        r"\.(response\.redacted\.bin|terminal\.json)\.[0-9a-f]{32}\.tmp", item) is None:
                    raise durable.EvidenceCorrupt("U0_UNKNOWN_CHILD_EVIDENCE")
                check = durable._open_file(fd, item, os.O_RDONLY)
                os.close(check)
            reservation = _read_object(fd, "reservation.json")
            terminal = _read_object(fd, "terminal.json")
            request = durable._read(fd, "request.bin")
            response = durable._read(fd, "response.redacted.bin")
            if (_read_object(fd, "binding.json") != binding
                    or reservation.get("schema_version") != "JevReservationV1"
                    or reservation.get("binding") != binding
                    or reservation.get("attempts_consumed") != 1
                    or reservation.get("retry_allowance") != 0
                    or durable._sha(request) != binding["request_sha256"]
                    or terminal.get("schema_version") != "JevOperationTerminalV1"
                    or terminal.get("binding") != binding
                    or terminal.get("attempts_consumed") != 1
                    or terminal.get("retry_allowance") != 0
                    or terminal.get("retained_response_sha256") != durable._sha(response)
                    or terminal.get("retained_response_bytes") != len(response)
                    or type(terminal.get("u0_passed")) is not bool
                    or (not terminal_inspection and terminal.get("u0_passed") is not True)
                    or terminal.get("reserved_usd") != str(ROLE_RESERVES_USD[binding["role"]])):
                raise durable.EvidenceCorrupt("U0_CHILD_EVIDENCE_MISMATCH")
            return durable._sha(durable._read(fd, "terminal.json"))
        finally:
            os.close(fd)

    def _history(self, scope, *, terminal_inspection=False):
        fd = scope._directory_fd
        names = set(os.listdir(fd))
        # Check every existing regular-file candidate, not just files later used.
        for name in names:
            if re.fullmatch(r"attempt-[0-9]{2}", name):
                child = _child_fd(fd, name)
                os.close(child)
            else:
                check = durable._open_file(fd, name, os.O_RDONLY)
                os.close(check)
        if "stop.json" in names and not terminal_inspection:
            raise ExperimentStopped("U0_STOP_PERSISTED")
        if terminal_inspection:
            if "manifest.json" not in names:
                raise durable.EvidenceCorrupt("U0_MANIFEST_MISSING")
            if "stop.json" in names:
                _read_object(fd, "stop.json")
        if "manifest.json" not in names:
            if names != {"experiment.lock"}:
                raise durable.EvidenceCorrupt("U0_MANIFEST_MISSING")
            scope._write_once("manifest.json", durable._json(self._manifest_record))
            names.add("manifest.json")
        if _read_object(fd, "manifest.json") != self._manifest_record:
            raise durable.BindingMismatch("U0_MANIFEST_CHANGED")
        count = sum(re.fullmatch(r"reservation-[0-9]{2}\.json", name) is not None for name in names)
        expected = {"experiment.lock", "manifest.json"}
        for index in range(1, count + 1):
            expected.update({f"reservation-{index:02d}.json", f"terminal-{index:02d}.json",
                             f"attempt-{index:02d}"})
            record = _read_object(fd, f"reservation-{index:02d}.json")
            if record.get("binding", {}).get("execution_mode") == "U0_APPROVED_OPERATOR_P1":
                if f"wire-{index:02d}.json" in names:
                    expected.add(f"wire-{index:02d}.json")
                else:
                    child = _child_fd(fd, f"attempt-{index:02d}")
                    try: missing_wire_terminal = _read_object(child, "terminal.json")
                    finally: os.close(child)
                    if (not terminal_inspection or missing_wire_terminal.get("u0_passed") is not False
                            or missing_wire_terminal.get("attempts_observed") not in (0, None)):
                        raise durable.EvidenceCorrupt("U0_PHYSICAL_DISPATCH_EVIDENCE_MISSING")
        if terminal_inspection and "stop.json" in names:
            expected.add("stop.json")
        if count > 20 or names != expected:
            raise durable.EvidenceCorrupt("U0_INCOMPLETE_OR_UNKNOWN_HISTORY")
        counts, spent, used, hashes, homes = {"R1": 0, "R2": 0}, Decimal("0"), set(), set(), {}
        for index in range(1, count + 1):
            record = _read_object(fd, f"reservation-{index:02d}.json")
            value = self._binding(record.get("binding"))
            role, case_id = value["role"], value["case_id"]
            pair = (role, case_id)
            if pair != self._order[index - 1]:
                raise durable.EvidenceCorrupt("U0_HISTORY_ORDER_MISMATCH")
            if pair in used or value["request_sha256"] in hashes:
                raise durable.EvidenceCorrupt("U0_DUPLICATE_HISTORY")
            if role in homes and value["profile_home"] != homes[role]:
                raise durable.BindingMismatch("U0_PROFILE_CHANGED")
            used.add(pair)
            hashes.add(value["request_sha256"])
            homes[role] = value["profile_home"]
            counts[role] += 1
            spent += ROLE_RESERVES_USD[role]
            if (record.get("schema_version") != "U0ReservationV1"
                    or record.get("manifest_sha256") != self.manifest_sha256
                    or record.get("attempts_consumed") != index
                    or record.get("role_attempts_consumed") != counts
                    or record.get("reserved_usd") != str(ROLE_RESERVES_USD[role])
                    or record.get("cumulative_reserved_usd") != str(spent)
                    or record.get("retry_allowance") != 0
                    or counts[role] > 10 or spent > PROPOSED_BUDGET_USD):
                raise durable.EvidenceCorrupt("U0_ACCOUNTING_MISMATCH")
            terminal = _read_object(fd, f"terminal-{index:02d}.json")
            if (terminal.get("schema_version") != "U0AttemptTerminalV1"
                    or terminal.get("reservation_sha256") != durable._sha(durable._json(record))
                    or type(terminal.get("passed")) is not bool
                    or (not terminal_inspection and terminal.get("passed") is not True)
                    or terminal.get("operation_terminal_sha256") != self._child_terminal(scope, index, value, terminal_inspection=terminal_inspection)):
                raise durable.EvidenceCorrupt("U0_TERMINAL_MISMATCH")
            if value.get("execution_mode") == "U0_APPROVED_OPERATOR_P1" and f"wire-{index:02d}.json" in names:
                wire = _read_object(fd, f"wire-{index:02d}.json")
                if (wire.get("binding") != value or wire.get("index") != index
                        or wire.get("physical_attempt_allowance_consumed") != 1):
                    raise durable.EvidenceCorrupt("U0_PHYSICAL_DISPATCH_BINDING_MISMATCH")
            if terminal_inspection and terminal["passed"] is False:
                if index != count or "stop.json" not in names:
                    raise durable.EvidenceCorrupt("U0_FIRST_FAILURE_STOP_MISMATCH")
        return counts, spent, used, hashes, homes

    @contextmanager
    def next_slot(self, binding: dict, request_bytes: bytes):
        """Reserve before yielding; never reuse failed, interrupted or unknown work."""
        with _Scope(self.root) as scope:
            slot = None
            try:
                counts, spent, used, hashes, homes = self._history(scope)
                value = self._binding(binding, request_bytes)
                role = value["role"]
                pair = (role, value["case_id"])
                if pair in used or value["request_sha256"] in hashes:
                    raise ExperimentStopped("U0_ATTEMPT_ALREADY_CONSUMED")
                if role in homes and value["profile_home"] != homes[role]:
                    raise durable.BindingMismatch("U0_PROFILE_CHANGED")
                if sum(counts.values()) >= 20 or counts[role] >= 10:
                    raise ExperimentStopped("U0_ATTEMPT_LIMIT")
                if pair != self._order[sum(counts.values())]:
                    raise durable.BindingMismatch("U0_DISPATCH_ORDER_MISMATCH")
                if not isinstance(request_bytes, bytes):
                    raise durable.BindingMismatch("U0_WIRE_BYTES_REQUIRED")
                reserve = ROLE_RESERVES_USD[role]
                if spent + reserve > PROPOSED_BUDGET_USD:
                    raise ExperimentStopped("U0_BUDGET_LIMIT")
                index = sum(counts.values()) + 1
                counts[role] += 1
                record = {"schema_version": "U0ReservationV1", "reserved_at": durable._now(),
                          "manifest_sha256": self.manifest_sha256, "binding": value,
                          "attempts_consumed": index, "role_attempts_consumed": counts,
                          "reserved_usd": str(reserve), "cumulative_reserved_usd": str(spent + reserve),
                          "retry_allowance": 0}
                scope.ensure_current()
                scope._write_once(f"reservation-{index:02d}.json", durable._json(record))
                with durable.Operation(self.directory / f"attempt-{index:02d}", value, request_bytes) as op:
                    op.reserve()
                    slot = _Attempt(scope, op, index, record)
                    slot.ensure_current()
                    yield slot
                    if not slot.completed:
                        scope.stop("U0_CONSUMED_WITHOUT_TERMINAL")
            except BaseException:
                # Persist the first failure before releasing the experiment lock.
                # If IO itself fails, the incomplete reservation remains consumed.
                scope.stop("U0_FAILURE_OR_UNKNOWN")
                raise
            finally:
                if slot is not None:
                    slot.active = False


def claim_physical_attempt(reservation_path):
    """Consume the existing reserved slot once at its physical wire boundary.

    The operator parent must hold both original ledger/operation locks. This
    marker neither reserves another allowance nor grants authority. Torn writes
    are consumed; a second worker cannot reuse even an unsealed reservation.
    """
    path = Path(reservation_path)
    child = durable._open_directory(path.parent)
    parent = durable._open_directory(path.parent.parent)
    locks = []
    try:
        for directory, name in ((parent, 'experiment.lock'), (child, 'operation.lock')):
            lock = durable._open_file(directory, name, os.O_RDWR)
            locks.append(lock)
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                fcntl.flock(lock, fcntl.LOCK_UN)
                raise durable.BindingMismatch('U0_OPERATOR_LOCK_NOT_HELD')
        record = _read_object(child, 'reservation.json')
        count = len([name for name in os.listdir(parent) if re.fullmatch(r'reservation-[0-9]{2}\.json', name)])
        outer = _read_object(parent, f'reservation-{count:02d}.json')
        if (path.parent.name != f'attempt-{count:02d}'
                or outer['binding'] != record['binding']
                or record['binding'].get('execution_mode') != 'U0_APPROVED_OPERATOR_P1'
                or outer['attempts_consumed'] != count
                or durable._read(child, 'terminal.json') is not None
                or durable._read(parent, 'stop.json') is not None):
            raise durable.BindingMismatch('U0_PHYSICAL_DISPATCH_NOT_CURRENT')
        marker = {'schema_version':'U0PhysicalAttemptV1','index':count,
            'entered_at':durable._now(),'binding':record['binding'],
            'physical_attempt_allowance_consumed':1,'retry_allowance':0}
        owner = _Scope(path.parent.parent.parent)
        owner._directory_fd = parent
        owner._write_once(f'wire-{count:02d}.json', durable._json(marker))
        return durable._sha(durable._json(marker))
    finally:
        for lock in locks: os.close(lock)
        os.close(child); os.close(parent)


def inspect_terminal_history(root, manifest):
    """Reuse exact ledger IO/history checks for sealed GREEN or first-STOP replay."""
    experiment = Experiment(root, manifest)
    if not experiment.directory.is_dir() or not (experiment.directory/'manifest.json').is_file():
        raise durable.EvidenceCorrupt('U0_TERMINAL_HISTORY_REQUIRED')
    with _Scope(Path(root), read_only=True) as scope:
        history = experiment._history(scope, terminal_inspection=True)
        files = {}
        for name in os.listdir(scope._directory_fd):
            if re.fullmatch(r'attempt-[0-9]{2}', name):
                child = _child_fd(scope._directory_fd, name)
                try:
                    for filename in ('binding.json','request.bin','reservation.json','response.redacted.bin','terminal.json'):
                        files[name+'/'+filename] = durable._read(child, filename)
                finally: os.close(child)
            elif name != 'experiment.lock':
                files[name] = durable._read(scope._directory_fd, name)
        scope.ensure_current()
        return {'history':history,'files':files}


class _Attempt:
    def __init__(self, scope, operation, index, record):
        self.scope, self.operation, self.index, self.record = scope, operation, index, record
        self.completed = False
        self._completion_attempted = False
        self.active = True

    def ensure_current(self):
        self.scope.ensure_current()
        fd = _child_fd(self.scope._directory_fd, f"attempt-{self.index:02d}")
        try:
            if not _same_inode(os.fstat(fd), os.fstat(self.operation._directory_fd)):
                raise durable.UnsafePath("U0_CHILD_REPLACED")
            info = os.stat("operation.lock", dir_fd=fd, follow_symlinks=False)
            if not _same_inode(info, os.fstat(self.operation._lock_fd)):
                raise durable.UnsafePath("U0_CHILD_LOCK_REPLACED")
            durable._regular(self.operation._lock_fd, fd, "operation.lock")
        finally:
            os.close(fd)

    def complete(self, *, passed: bool | None = None, receipt: dict, response_bytes: bytes):
        """Caller supplies every integrity/transport/accounting/conformance verdict.

        Missing/error outcomes must never be passed as True. Costs may be unknown;
        the full fixed reservation remains charged for every outcome, even success.
        """
        if not self.active:
            raise durable.OperationConsumed("U0_ATTEMPT_CLOSED")
        try:
            if self._completion_attempted:
                raise durable.OperationConsumed("U0_TERMINAL_ALREADY_ATTEMPTED")
            self._completion_attempted = True
            if type(passed) is not bool:
                raise TypeError("U0_BOOLEAN_VERDICT_REQUIRED")
            self.ensure_current()
            if not passed:
                self.scope.stop("U0_CALLER_REPORTED_FAILURE")
            result = durable._object(durable._json(receipt))
            result.update(u0_passed=passed, reserved_usd=self.record["reserved_usd"])
            terminal = self.operation.complete(result, response_bytes)
            self.ensure_current()
            self.scope._write_once(f"terminal-{self.index:02d}.json", durable._json({
                "schema_version": "U0AttemptTerminalV1", "sealed_at": durable._now(),
                "reservation_sha256": durable._sha(durable._json(self.record)),
                "operation_terminal_sha256": durable._sha(
                    durable._read(self.operation._directory_fd, "terminal.json")),
                "passed": passed,
            }))
            self.completed = True
            return terminal
        except BaseException:
            self.scope.stop("U0_COMPLETION_FAILURE_OR_UNKNOWN")
            raise
