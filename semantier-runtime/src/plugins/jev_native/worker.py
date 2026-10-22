"""Supervised transport worker; exact operator specimen or socket-blocked offline fixture."""
import base64
import ctypes
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time

def main():
    deadline = float(sys.argv[1])
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return 124
    if sys.platform != "linux":
        return 125
    parent = os.getppid()
    # Independent kill on parent loss and total wall-clock expiry. No worker children.
    if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        return 125
    if parent != int(sys.argv[2]) or os.getppid() != parent:
        return 125
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.setitimer(signal.ITIMER_REAL, remaining)
    def blocked(*args, **kwargs):
        raise RuntimeError("NETWORK_FORBIDDEN")
    socket_calls = (socket.socket.connect, socket.socket.connect_ex,
                    socket.create_connection, socket.getaddrinfo)
    socket.socket.connect = blocked
    socket.socket.connect_ex = blocked
    socket.create_connection = blocked
    socket.getaddrinfo = blocked
    entered_native = False
    try:
        incoming = sys.stdin.buffer.read(8 * 1024 * 1024 + 1)
        if len(incoming) > 8 * 1024 * 1024:
            raise ValueError("IPC_TOO_LARGE")
        message = json.loads(incoming)
        package = Path(__file__).resolve().parent
        allowed_sources = {"worker.py", "native.py", "service.py", "host.py", "durable.py", "__init__.py", "plugin.yaml", "specimen.py"}
        if (message.get("execution_mode") == "OPERATOR_SPECIMEN_ONCE"
                and set(message["candidate_sources"]) != allowed_sources):
            raise ValueError("SOURCE_SET_INVALID")
        for name, expected in message["candidate_sources"].items():
            if name not in allowed_sources:
                raise ValueError("SOURCE_SET_INVALID")
            if hashlib.sha256((package/name).read_bytes()).hexdigest() != expected:
                raise ValueError("CANDIDATE_SOURCE_DRIFT")
        config = Path(message["profile_home"]) / "config.yaml"
        if config.is_symlink() or hashlib.sha256(config.read_bytes()).hexdigest() != message["config_file_sha256"]:
            raise ValueError("PROFILE_CONFIG_DRIFT")
        from native import run_native, run_specimen_native, SPECIMEN_BYTES, SPECIMEN_SHA256
        raw = base64.b64decode(message["request_b64"], validate=True)
        if message.get("execution_mode") == "OPERATOR_SPECIMEN_ONCE":
            if (len(raw) != SPECIMEN_BYTES or hashlib.sha256(raw).hexdigest() != SPECIMEN_SHA256
                    or message.get("fixture") is not None):
                raise ValueError("SPECIMEN_REQUEST_MISMATCH")
            socket.socket.connect, socket.socket.connect_ex, socket.create_connection, socket.getaddrinfo = socket_calls
            entered_native = True
            result = run_specimen_native(raw, message["credential"], deadline)
        else:
            if not isinstance(message.get("fixture"), dict):
                raise PermissionError("PAID_EXPERIMENT_AUTHORITY_UNAVAILABLE")
            entered_native = True
            result = run_native(raw, message["credential"], deadline, message["fixture"])
        sys.stdout.write(json.dumps(result, ensure_ascii=True, allow_nan=False))
        sys.stdout.flush()
        return 0
    except Exception:
        sys.stdout.write(json.dumps({"outcome": "WORKER_FAILURE_UNKNOWN" if entered_native else "PREDISPATCH_BLOCKED", "error_category": "worker_prerequisite",
                                     "attempts_observed": None if entered_native else 0, "response_available": False,
                                     "response_complete": False, "http_status": None,
                                     "request_id": None, "actual_model": None, "usage": None,
                                     "response_b64": ""}))
        sys.stdout.flush()
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
