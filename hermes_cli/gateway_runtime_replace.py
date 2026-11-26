"""Replace a gateway left running by an older Hermes on this install's own home.

After an upgrade the old daemon keeps answering the home's control socket, but its ``identify`` has
no ``runtime_protocol`` (it predates the session-authority runtime). Every client of this version
ensures through :func:`hermes_cli.gateway_runtime.ensure_gateway_runtime` (the TUI bootstrap,
Desktop's ``hermes gateway ensure``, ``hermes chat``), so ensure replaces that owner once, exactly as
``hermes gateway stop`` would stop it (a planned-stop marker in the owner's home, then SIGTERM; a
bounded wait; a start-time-guarded force kill), and then starts the current one the normal way.

Refused, never stopped:
- a gateway that speaks a NEWER runtime protocol than this client (a downgrade must not kill it);
- a gateway of another install: its home resolves outside this install's root (a profile symlinked
  into another HERMES root), or the process serves another home than the socket that answered;
- anything whose ``identify`` cannot be pinned to one live process (pid/start time disagree, our
  own process, not a gateway).
"""
from __future__ import annotations

import os
import time
import sys
from pathlib import Path
from typing import Any

from hermes_platform.host.windows_package import package_source as _windows_package_source

# Graceful drain before the guarded force kill; an idle old gateway exits in well under a second.
_GRACEFUL_STOP_SECONDS = 10.0
_FORCED_STOP_SECONDS = 5.0

# One line each, short enough for the TUI's activity row after its "gateway incompatible: " prefix.
REFUSAL_DETAIL = {
    "owner_newer": "a newer Hermes's gateway owns this profile; update Hermes or run: hermes gateway stop",
    "foreign_install": "another Hermes install's gateway owns this profile; stop it there: hermes gateway stop",
    "owner_unverified": "an older Hermes gateway owns this profile and was not verified; run: hermes gateway stop",
    "owner_did_not_exit": "an older Hermes gateway owns this profile and did not stop; run: hermes gateway stop",
    "owner_unreachable": "an older Hermes gateway owns this profile; run: hermes gateway stop",
}


def _install_root() -> Path:
    from hermes_constants import get_default_hermes_root
    return Path(get_default_hermes_root()).resolve()


def _identify(home: Path, timeout: float) -> tuple[Path, dict] | None:
    """``(control_home, identify payload)`` of the process answering for *home*: its own socket, or
    the default multiplexer's for a served named profile (the same order discovery uses)."""
    from hermes_cli.gateway_runtime_discovery import query_identify
    from hermes_cli.gateway_runtime_multiplex import multiplexer_root_for
    for control in (home, multiplexer_root_for(home)):
        if control is None:
            continue
        try:
            return control, query_identify(control, timeout=timeout)
        except (FileNotFoundError, ConnectionRefusedError):
            continue
    return None


def _same(a: Any, b: Path) -> bool:
    try:
        return os.path.normcase(str(Path(a).resolve())) == os.path.normcase(str(b.resolve()))
    except (OSError, TypeError, ValueError):
        return False


def _within_install(home: Path, root: Path) -> bool:
    home = home.resolve()
    return home == root or (home.parent == root / "profiles")


def _verified_owner(control: Path, payload: dict, *, require_install_root: bool = True) -> tuple[int, Any] | str:
    from gateway.status import _GATEWAY_KIND, _pid_exists, get_process_start_time, start_time_fingerprints_match
    protocol = payload.get("runtime_protocol")
    if type(protocol) is int and protocol > 1:
        return "owner_newer"
    if ((require_install_root and not _within_install(control, _install_root()))
            or not _same(payload.get("hermes_home"), control)):
        return "foreign_install"
    pid, recorded = payload.get("pid"), payload.get("start_time")
    if (payload.get("kind") != _GATEWAY_KIND or type(pid) is not int or pid <= 1
            or pid in {os.getpid(), os.getppid()} or recorded is None or not _pid_exists(pid)):
        return "owner_unverified"
    current = get_process_start_time(pid)
    try:
        if current is None or not start_time_fingerprints_match(recorded, current):
            return "owner_unverified"
    except (TypeError, ValueError):
        return "owner_unverified"
    return pid, current


def _native_process_source(pid: int) -> Path:
    import psutil
    import subprocess
    from gateway.runtime_bootstrap_windows import _process_sid
    from gateway.status import looks_like_gateway_runtime_command_line
    if _process_sid(pid) != _process_sid(os.getpid()):
        raise PermissionError('Native Gateway user mismatch')
    try:
        process = psutil.Process(pid)
        if not looks_like_gateway_runtime_command_line(subprocess.list2cmdline(process.cmdline())):
            raise ValueError('Native Gateway process role mismatch')
        return Path(process.exe()).resolve()
    except psutil.Error:
        raise OSError('Native Gateway process identity unavailable') from None


def native_package_owner(control: Path, payload: dict) -> str | None:
    """Compatibility of the verified live image, not a version/port self-claim."""
    try:
        client = _windows_package_source(sys.executable, registered=True)
        if client is None:
            return None  # Existing source/dev discovery is unchanged.
        verified = _verified_owner(control, payload, require_install_root=False)
        if isinstance(verified, str):
            return verified
        pid, birth = verified
        from gateway.status import get_process_start_time, start_time_fingerprints_match
        # Windows creation time comes from the kernel; macOS clock-drift slack
        # must not admit a recycled Windows PID even a second after its owner.
        if not start_time_fingerprints_match(payload['start_time'], birth, tolerance=0):
            return 'owner_unverified'
        image = _native_process_source(pid)
        current_birth = get_process_start_time(pid)
        if current_birth is None or not start_time_fingerprints_match(birth, current_birth, tolerance=0):
            return 'owner_unverified'
        owner = _windows_package_source(image, registered=False)
        if (owner is None or owner[0].parent != client[0].parent
                or owner[1] != client[1] or owner[3] != client[3]):
            return 'foreign_install'
        if owner[0] == client[0]:
            return 'same'
        if not _within_install(control, _install_root()):
            return 'foreign_install'  # Reuse never grants permission to replace another home.
        return 'older' if owner[2] < client[2] else 'owner_newer'
    except (OSError, ValueError, TypeError):
        return 'owner_unverified'


def _write_planned_stop(control: Path, pid: int, start_time: Any) -> None:
    """The marker ``hermes gateway stop`` writes, in the OWNER's home (it reads its own), so the old
    gateway exits as a planned stop: clean exit, ``stopped`` persisted, no supervisor revival."""
    from gateway.status import _PLANNED_STOP_MARKER_FILENAME, _utc_now_iso, _write_marker
    _write_marker(control / _PLANNED_STOP_MARKER_FILENAME, {
        "target_pid": pid, "target_start_time": start_time,
        "stopper_pid": os.getpid(), "written_at": _utc_now_iso(),
    })


def _wait_gone(pid: int, seconds: float) -> bool:
    from gateway.status import _pid_exists
    end = time.monotonic() + max(seconds, 0.0)
    while _pid_exists(pid):
        if time.monotonic() >= end:
            return False
        time.sleep(0.1)
    return True


def replace_incompatible_owner(home: Path, *, deadline: float, for_package_update: bool = False,
                               desktop_instance: str | None = None) -> str | None:
    """Stop the verified gateway answering for *home*; None once it is gone, else a refusal code
    (see :data:`REFUSAL_DETAIL`). Never starts anything: the caller's ensure loop does that."""
    from gateway.status import terminate_pid
    budget = deadline - time.monotonic()
    if budget <= 0:
        return "owner_unreachable"
    while True:
        try:
            found = _identify(home, timeout=min(2.0, budget))
        except (OSError, ValueError, TypeError):  # timeout, refused/denied peer, malformed reply
            return "owner_unreachable"
        if found is not None:
            break
        if desktop_instance is None:
            return None
        # An ensure deadline can leave a daemon booting with a reservation but
        # without a control endpoint. Wait for its identity before deciding.
        from hermes_cli.gateway_runtime_discovery import missing_owner_state
        from hermes_cli.gateway_runtime_multiplex import multiplexer_root_for
        states = [missing_owner_state(home)]
        if (host := multiplexer_root_for(home)) is not None:
            states.append(missing_owner_state(host))
        if all(state == "absent" for state in states):
            return None
        if "inaccessible" in states or budget <= _FORCED_STOP_SECONDS + 1:
            return "owner_unreachable"
        time.sleep(0.1)
        budget = deadline - time.monotonic()
    control, payload = found
    if desktop_instance is not None and (
        payload.get("instance_id") != desktop_instance or payload.get("supervisor") != "none"
    ):
        return None
    verdict = _verified_owner(control, payload)
    if isinstance(verdict, str):
        return verdict
    pid, start_time = verdict
    if desktop_instance is not None:
        from gateway.status import start_time_fingerprints_match
        if not start_time_fingerprints_match(payload["start_time"], start_time, tolerance=0):
            return "owner_unverified"
    native = native_package_owner(control, payload)
    if native == 'same' and not for_package_update and desktop_instance is None:
        return None  # A new owner won the discovery/stop race; do not stop it.
    if for_package_update and native is None:
        return 'owner_unverified'  # This explicit drain never stops an unpackaged owner.
    if native not in (None, 'older', 'same'):
        return native
    _write_planned_stop(control, pid, start_time)
    graceful = min(_GRACEFUL_STOP_SECONDS, max(deadline - time.monotonic() - _FORCED_STOP_SECONDS, 1.0))
    try:
        if os.name != "nt":
            # Windows has no SIGTERM for an asyncio loop: the marker watcher IS its graceful stop.
            terminate_pid(pid, expected_start_time=start_time)
        if _wait_gone(pid, graceful):
            return None
        terminate_pid(pid, force=True, expected_start_time=start_time)
    except ProcessLookupError:
        return None
    except (PermissionError, OSError):
        return "owner_did_not_exit" if not _wait_gone(pid, 0) else None
    return None if _wait_gone(pid, _FORCED_STOP_SECONDS) else "owner_did_not_exit"
