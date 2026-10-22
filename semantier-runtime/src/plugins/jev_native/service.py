"""Disabled candidate service: real host gates and durable supervision, external HTTP offline."""
from __future__ import annotations
import base64
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from .host import resolve_route
from .durable import Operation
from .native import prepare_request, redact_bytes, safe_provider_value

PACKAGE = Path(__file__).resolve().parent

def _sources():
    names = ("worker.py", "native.py", "service.py", "host.py", "durable.py", "__init__.py", "plugin.yaml", "specimen.py")
    return {name: hashlib.sha256((PACKAGE/name).read_bytes()).hexdigest() for name in names}

def execute_live(*args, **kwargs):
    """Plugin enablement never confers paid experiment authority."""
    raise PermissionError("PAID_EXPERIMENT_AUTHORITY_UNAVAILABLE")

def execute_offline(ctx, request_bytes, operation_id=None, *, receipt_root=None, deadline_seconds,
                    fixture, expected_profile_home=None):
    return _execute(ctx, request_bytes, operation_id, receipt_root=receipt_root,
                    deadline_seconds=deadline_seconds, fixture=fixture,
                    expected_profile_home=expected_profile_home)


def _execute(ctx, request_bytes, operation_id=None, *, receipt_root=None, deadline_seconds,
             fixture, expected_profile_home=None, _specimen=None, _started=None):
    started = time.monotonic() if _started is None else _started
    if sys.platform != "linux":
        raise RuntimeError("QUALIFIED_PLATFORM_LINUX_REQUIRED")
    if type(deadline_seconds) not in (int, float) or not math.isfinite(deadline_seconds) or not 0 < deadline_seconds <= 120:
        raise ValueError("DEADLINE_INVALID")
    deadline = started + deadline_seconds
    if _specimen is None and not isinstance(fixture, dict):
        raise PermissionError("PAID_EXPERIMENT_AUTHORITY_UNAVAILABLE")
    if _specimen is not None:
        from .specimen import verify_invocation, verify_composition, _OperatorInvocation
        if type(_specimen) is not _OperatorInvocation:
            raise PermissionError("IN_MEMORY_OPERATOR_INVOCATION_REQUIRED")
        _specimen = _specimen.record
        verify_invocation(request_bytes, _specimen, deadline_seconds)
        verify_composition()
        if fixture is not None:
            raise ValueError("LIVE_FIXTURE_FORBIDDEN")
    route = resolve_route(ctx, expected_profile_home=expected_profile_home)
    prepared = prepare_request(request_bytes)
    from agent.secret_scope import current_secret_scope, get_secret, build_profile_secret_scope
    scope = current_secret_scope()
    if scope is None or "TYPESAFE_API_KEY" not in scope:
        raise PermissionError("PROFILE_SECRET_SCOPE_REQUIRED")
    credential_home = Path(route.profile_home if _specimen is None else _specimen["credential_profile_home"])
    expected_scope = build_profile_secret_scope(credential_home)
    if not expected_scope.get("TYPESAFE_API_KEY") or expected_scope.get("TYPESAFE_API_KEY") != scope.get("TYPESAFE_API_KEY"):
        raise PermissionError("PROFILE_SECRET_SCOPE_MISMATCH")
    credential = get_secret("TYPESAFE_API_KEY")
    if type(credential) is not str or re.fullmatch(r"[!-~]{1,8192}", credential) is None:
        raise PermissionError("PROFILE_CREDENTIAL_UNAVAILABLE")
    # The caller cannot manufacture a second allowance with a fresh ID or root.
    stable_id = prepared.sha256
    if _specimen is None:
        root = ctx.state.data_dir / "offline-operations"
    else:
        from .specimen import operation_parent
        root = operation_parent()
    if operation_id is not None and operation_id != stable_id:
        raise ValueError("OPERATION_ID_MUST_MATCH_REQUEST")
    if receipt_root is not None and Path(receipt_root).absolute() != root.absolute():
        raise ValueError("RECEIPT_ROOT_IS_HOST_OWNED")
    home = Path(route.profile_home) if _specimen is None else root.parent
    relative = root.relative_to(home)
    current = home
    for component in relative.parts:
        parent = current
        current = parent / component
        if current.is_symlink():
            raise ValueError("UNSAFE_RECEIPT_ROOT")
        current.mkdir(mode=0o700, exist_ok=True)
        if current.stat().st_uid != os.geteuid() or current.stat().st_mode & 0o022:
            raise ValueError("UNSAFE_RECEIPT_ROOT")
        directory_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    operation_id = stable_id
    # Validate host policy again after preparation/secret resolution.
    if resolve_route(ctx, expected_profile_home=route.profile_home) != route:
        raise PermissionError("HOST_POLICY_DRIFT")
    if time.monotonic() >= deadline:
        raise TimeoutError("PREDISPATCH_DEADLINE")
    route_dict = asdict(route)
    binding = {"operation_id": operation_id, "profile_home": route.profile_home,
               "request_sha256": prepared.sha256, "model": prepared.model,
               "endpoint": prepared.endpoint, "host_route": route_dict,
               "candidate_sources": _sources(), "execution_mode": "OFFLINE_ONLY",
               "paid_execution_authority": False}
    if _specimen is not None:
        binding.update(execution_mode="OPERATOR_SPECIMEN_ONCE",
                       operator_confirmation=_specimen,
                       paid_execution_authority="DIRECT_OPERATOR_COMMAND_NOT_RUNTIME_AUTHORITY")
    with Operation(root / operation_id, binding, prepared.body) as operation:
        if time.monotonic() >= deadline:
            raise TimeoutError("PREDISPATCH_DEADLINE")
        operation.reserve()
        result = {"outcome": "INTERRUPTED_UNKNOWN", "attempts_observed": None,
                  "http_status": None, "request_id": None, "actual_model": None,
                  "usage": None, "response_available": False, "response_complete": False,
                  "raw_response_sha256": None, "retries": 0, "redirects": 0,
                  "fallbacks": 0, "attempt_allowance_consumed": 1,
                  "provider_side_cancellation": "NOT_ESTABLISHED"}
        proc = None
        interrupted = None
        response_bytes = b""
        try:
            if time.monotonic() >= deadline:
                result.update(outcome="PREDISPATCH_DEADLINE", attempts_observed=0)
            else:
                message = {"request_b64": base64.b64encode(prepared.body).decode(),
                           "credential": credential, "fixture": fixture,
                           "profile_home": route.profile_home,
                           "config_file_sha256": route.config_file_sha256,
                           "candidate_sources": binding["candidate_sources"],
                           "execution_mode": binding["execution_mode"]}
                raw = json.dumps(message, ensure_ascii=False, allow_nan=False).encode()
                if len(raw) > 8 * 1024 * 1024:
                    raise ValueError("IPC_TOO_LARGE")
                # A clean environment carries no inherited credential values.
                env = {"PATH": os.defpath, "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1",
                       "HERMES_HOME": route.profile_home}
                proc = subprocess.Popen([sys.executable, "-B", str(PACKAGE/"worker.py"),
                                         repr(deadline), str(os.getpid())],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, env=env, start_new_session=True)
                result["worker_pid"] = proc.pid
                stdout, _ = proc.communicate(raw, timeout=max(0.001, deadline-time.monotonic()))
                if proc.returncode in (-signal.SIGALRM, -signal.SIGKILL) or time.monotonic() >= deadline:
                    result.update(outcome="DEADLINE_EXCEEDED", attempts_observed=None)
                elif proc.returncode not in (0, 2):
                    result.update(outcome="WORKER_INTERRUPTED", attempts_observed=None)
                elif len(stdout) > 4 * 1024 * 1024:
                    result.update(outcome="INVALID_WORKER_RECEIPT", attempts_observed=None)
                else:
                    envelope = json.loads(stdout)
                    response_bytes = base64.b64decode(envelope.pop("response_b64"), validate=True)
                    if len(response_bytes) > 2*1024*1024:
                        raise ValueError("RESPONSE_TOO_LARGE")
                    result.update(envelope)
        except subprocess.TimeoutExpired:
            result.update(outcome="DEADLINE_EXCEEDED", attempts_observed=None)
        except (KeyboardInterrupt, SystemExit) as error:
            interrupted = error
            result.update(outcome="CANCELLED_UNKNOWN", attempts_observed=None)
        except Exception:
            result.update(outcome="LOCAL_EXECUTION_ERROR", attempts_observed=None)
        finally:
            if proc is not None:
                if proc.poll() is None:
                    proc.kill()
                proc.wait()
                for stream in (proc.stdin, proc.stdout, proc.stderr):
                    if stream is not None:
                        stream.close()
                result["worker_reaped"] = True
                result["worker_returncode"] = proc.returncode
            else:
                result["worker_reaped"] = True
            result["elapsed_ms"] = round((time.monotonic()-started)*1000)
            result["deadline_seconds"] = deadline_seconds
            # The guard applies again before any retained error/metadata publication.
            response_bytes = redact_bytes(response_bytes, credential)
            for name in ("actual_model", "usage", "answers", "request_id"):
                if name in result:
                    result[name] = safe_provider_value(result[name], credential)
        if _specimen is not None:
            from .specimen import record_cost
            record_cost(result)
        terminal = operation.complete(result, response_bytes)
        if interrupted is not None:
            raise interrupted
        return terminal
