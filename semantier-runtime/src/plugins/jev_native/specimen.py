"""One fixed public specimen, invoked by its local operator, never a runtime tool.

The confirmation record is provenance. It is not a Semantier authority artifact.
No API key is read by dry_run; the user-run final path uses the host scope loader.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

MAIN = Path("/home/chris/repo/semantier-runtime")
PACKAGE = Path(__file__).resolve().parent
REQUEST_SHA256 = "e160b92c55bdf0c16eb17668a2233fdb21aa3fe794b02c08afa7348627eac19d"
REQUEST = MAIN / "dlm-research/.worktrees/ttpc-materialization-v3-preflight/studies/jev-input-capacity-specimen-v1/request.freeze.json"
PHRASE = "SEND ONE PUBLIC TTPC-V3-0034 REQUEST"
DEADLINE = 60

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def operation_parent():
    # Fixed across invocation directories, temporary homes and credential profiles.
    return MAIN / "docs/research/logs/hermes-jev-live-specimen-v1"

def validate_request(raw):
    if type(raw) is not bytes or len(raw) != 24077 or sha(raw) != REQUEST_SHA256:
        raise ValueError("FROZEN_SPECIMEN_MISMATCH")
    from .native import prepare_request
    prepare_request(raw)
    return raw

def request_bytes():
    return validate_request(REQUEST.read_bytes())

def verify_composition():
    frozen = json.loads((PACKAGE / "OPERATOR_FREEZE.json").read_bytes())
    names = {"worker.py", "native.py", "service.py", "host.py", "durable.py",
             "__init__.py", "plugin.yaml", "specimen.py", "upstream/client.py", "operator-disclosure.txt"}
    if set(frozen.get("source_sha256", {})) != names:
        raise ValueError("OPERATOR_SOURCE_SET_INVALID")
    for name, expected in frozen["source_sha256"].items():
        if sha((PACKAGE / name).read_bytes()) != expected:
            raise ValueError("OPERATOR_SOURCE_DRIFT")
    if frozen.get("request_sha256") != REQUEST_SHA256:
        raise ValueError("OPERATOR_REQUEST_PIN_DRIFT")
    return sha((PACKAGE / "OPERATOR_FREEZE.json").read_bytes())

@dataclass(frozen=True)
class _OperatorInvocation:
    record: dict


def confirmation(home):
    return {
        "experiment_id": "hermes-jev-live-specimen-v1",
        "confirmation_source": "DIRECT_LOCAL_OPERATOR",
        "credential_profile_home": home,
        "request_sha256": REQUEST_SHA256,
        "attempt_limit": 1,
        "api_metered_allowance_usd": "0.01",
        "account_standard_tariff_no_autorefill": True,
        "confirmed_at": datetime.now(timezone.utc).isoformat(),
    }

def verify_invocation(raw, record, deadline_seconds):
    validate_request(raw)
    if not isinstance(record, dict):
        raise PermissionError("DIRECT_OPERATOR_CONFIRMATION_REQUIRED")
    home = record.get("credential_profile_home")
    if not isinstance(home, str) or not Path(home).is_absolute() or str(Path(home).resolve()) != home:
        raise ValueError("EXPLICIT_CREDENTIAL_PROFILE_REQUIRED")
    expected = confirmation(home)
    expected["confirmed_at"] = record.get("confirmed_at")
    if record != expected or type(record.get("attempt_limit")) is not int:
        raise PermissionError("EXACT_OPERATOR_SCOPE_REQUIRED")
    try:
        stamp = datetime.fromisoformat(record["confirmed_at"])
        age = (datetime.now(timezone.utc) - stamp).total_seconds()
        if stamp.utcoffset() is None or not 0 <= age <= 300:
            raise ValueError()
    except (ValueError, TypeError):
        raise PermissionError("OPERATOR_CONFIRMATION_EXPIRED") from None
    if deadline_seconds != DEADLINE:
        raise ValueError("SPECIMEN_DEADLINE_FIXED")

def record_cost(result):
    usage = result.get("usage")
    count = usage.get("input_tokens") if isinstance(usage, dict) else None
    amount = Decimal(count) * Decimal("0.042") / Decimal(1000000) if type(count) is int and count >= 0 else None
    result.update(computed_api_metered_usd=str(amount) if amount is not None else None,
                  tariff_usd_per_million_input="0.042", output_tariff_usd="0",
                  api_metered_allowance_usd="0.01", billing_guarantee=False,
                  invoice_verified=False, account_settings_verified_by="OPERATOR_AT_CONFIRMATION",
                  excluded=["account_taxes", "FX_settlement_fees"],
                  automatic_successor=False)
    if amount is not None and amount > Decimal("0.01"):
        result["outcome"] = "METERED_ALLOWANCE_EXCEEDED"

@contextmanager
def temporary_context():
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from hermes_cli.plugins import get_plugin_manager, PluginContext
    with tempfile.TemporaryDirectory(prefix="jev-specimen-home-") as temp:
        root = Path(temp)
        home = root / "home"; home.mkdir()
        bundle = root / "bundle"; bundle.mkdir()
        (bundle / "jev_native").symlink_to(PACKAGE, target_is_directory=True)
        entry = {"settings": {"connection": {"backend": "typesafe", "model": "jev-1.13.0"}},
                 "llm": {"allow_provider_override": True, "allowed_providers": ["typesafe"],
                         "allow_model_override": True, "allowed_models": ["jev-1.13.0"]}}
        (home / "config.yaml").write_text(json.dumps({"plugins": {
            "enabled": ["jev_native"], "disabled": [], "entries": {"jev_native": entry}}}))
        changes = {"HERMES_HOME": str(home), "HERMES_BUNDLED_PLUGINS": str(bundle),
                   "HERMES_ENABLE_PROJECT_PLUGINS": "0", "HERMES_SAFE_MODE": "0",
                   "HERMES_IGNORE_USER_CONFIG": "0"}
        before = {key: os.environ.get(key) for key in changes}
        os.environ.update(changes)
        marker = set_hermes_home_override(home)
        try:
            manager = get_plugin_manager(); manager.discover_and_load()
            loaded = manager._plugins.get("jev_native")
            if loaded is None or Path(loaded.manifest.path).resolve() != PACKAGE:
                raise PermissionError("LOADED_SPECIMEN_PLUGIN_MISMATCH")
            ctx = PluginContext(loaded.manifest, manager)
            from .host import resolve_route
            resolve_route(ctx, expected_profile_home=home)
            yield ctx
        finally:
            reset_hermes_home_override(marker)
            for key, value in before.items():
                if value is None: os.environ.pop(key, None)
                else: os.environ[key] = value

def dry_run():
    frozen_sha = verify_composition()
    raw = request_bytes()
    with temporary_context() as ctx:
        from .host import resolve_route
        route = resolve_route(ctx)
    consumed = (operation_parent() / REQUEST_SHA256).exists()
    return {"disposition": "ATTEMPT_DIRECTORY_EXISTS_STOP" if consumed else "LOCAL_READY_OPERATOR_INPUTS_PENDING",
            "paid_authority": False, "credential_checked": False, "request_bytes": len(raw),
            "request_sha256": REQUEST_SHA256, "source_freeze_sha256": frozen_sha,
            "provider": route.provider, "model": route.model, "endpoint": route.endpoint,
            "attempt_limit": 1, "deadline_seconds": DEADLINE, "operation_parent": str(operation_parent()),
            "billing_guarantee": False,
            "missing": ["user_selected_existing_credential_profile", "current_account_standard_tariff_and_no_autorefill",
                        "direct_operator_data_and_spending_confirmation"]}


SAFE_ERROR_CODES = frozenset({
    "EXPLICIT_CREDENTIAL_PROFILE_REQUIRED", "CREDENTIAL_PROFILE_EMPTY",
    "CREDENTIAL_PROFILE_NOT_ABSOLUTE", "CREDENTIAL_PROFILE_NOT_DIRECTORY",
    "CREDENTIAL_PROFILE_SYMLINK", "USER_TERMINAL_REQUIRED", "ACCOUNT_CONFIRMATION_REQUIRED",
    "ATTEMPT_DIRECTORY_EXISTS_STOP", "OPERATOR_DECLINED", "OPERATOR_SOURCE_SET_INVALID",
    "OPERATOR_SOURCE_DRIFT", "OPERATOR_REQUEST_PIN_DRIFT", "FROZEN_SPECIMEN_MISMATCH",
    "PROFILE_CREDENTIAL_UNAVAILABLE", "PROFILE_SECRET_SCOPE_REQUIRED",
    "PROFILE_SECRET_SCOPE_MISMATCH", "HOST_SOURCE_MISMATCH", "HOST_SOURCE_UNAVAILABLE",
    "HOST_PROFILE_MISMATCH", "HOST_PLUGIN_NOT_LOADED", "HOST_PLUGIN_DISABLED",
    "HOST_PROFILE_CONFIG_MISSING", "HOST_PROFILE_CONFIG_EMPTY", "HOST_CONNECTION_INVALID",
    "HOST_ROUTE_DENIED", "HOST_CONFIG_CHANGED", "HOST_POLICY_DRIFT",
    "LOADED_SPECIMEN_PLUGIN_MISMATCH", "PREDISPATCH_DEADLINE",
})

def safe_error_code(error):
    # Never display arbitrary exception text: config/provider errors can contain secrets.
    if error.args and type(error.args[0]) is str and error.args[0] in SAFE_ERROR_CODES:
        return error.args[0]
    if type(error).__name__ in {"OperationConsumed", "OperationBusy", "BindingMismatch", "EvidenceCorrupt"}:
        return "ATTEMPT_UNAVAILABLE_STOP"
    if isinstance(error, FileNotFoundError):
        return "LOCAL_PREREQUISITE_PATH_MISSING"
    if isinstance(error, PermissionError):
        return "LOCAL_PREREQUISITE_PERMISSION_DENIED"
    if isinstance(error, OSError):
        return "LOCAL_PREREQUISITE_IO_FAILURE"
    return "LOCAL_PREREQUISITE_INVALID"

def profile_diagnostic(value):
    result = {"argument_provided": value is not None, "argument_nonempty": False,
              "absolute": False, "directory_exists": False, "is_symlink": False,
              "resolved_directory": None, "env_file_exists_metadata_only": False,
              "credential_checked": False}
    if value is None:
        result["check_code"] = "EXPLICIT_CREDENTIAL_PROFILE_REQUIRED"
        return result
    value = str(value)
    result["argument_nonempty"] = bool(value.strip())
    if not value.strip():
        result["check_code"] = "CREDENTIAL_PROFILE_EMPTY"
        return result
    home = Path(value)
    result["absolute"] = home.is_absolute()
    if not home.is_absolute():
        result["check_code"] = "CREDENTIAL_PROFILE_NOT_ABSOLUTE"
        return result
    result["directory_exists"] = home.is_dir()
    result["is_symlink"] = home.is_symlink()
    if home.is_symlink():
        result["check_code"] = "CREDENTIAL_PROFILE_SYMLINK"
    elif not home.is_dir():
        result["check_code"] = "CREDENTIAL_PROFILE_NOT_DIRECTORY"
    else:
        result.update(check_code="PROFILE_DIRECTORY_OK", resolved_directory=str(home.resolve()),
                      env_file_exists_metadata_only=(home / ".env").is_file())
    return result

def diagnose(value=None, account_confirmed=False):
    report = {"dispatch_performed": False, "credential_checked": False,
              "stdin_is_terminal": sys.stdin.isatty(),
              "account_confirmation_flag": bool(account_confirmed),
              "profile": profile_diagnostic(value)}
    attempt = operation_parent() / REQUEST_SHA256
    report["attempt_path_exists"] = attempt.exists() or attempt.is_symlink()
    report["reservation_file_exists"] = (attempt / "reservation.json").exists()
    report["terminal_file_exists"] = (attempt / "terminal.json").exists()
    try:
        report["local_checks"] = dry_run()["disposition"]
    except Exception as error:
        report["local_checks"] = safe_error_code(error)
    return report


def confirm_operator(home, account_confirmed):
    profile = profile_diagnostic(home)
    if profile["check_code"] != "PROFILE_DIRECTORY_OK":
        raise ValueError(profile["check_code"])
    home = Path(profile["resolved_directory"])
    if not sys.stdin.isatty():
        raise PermissionError("USER_TERMINAL_REQUIRED")
    if not account_confirmed:
        raise PermissionError("ACCOUNT_CONFIRMATION_REQUIRED")
    print((PACKAGE / "operator-disclosure.txt").read_text().format(
        request_sha256=REQUEST_SHA256, credential_home=str(home.resolve()),
        receipt_root=str(operation_parent())))
    if input("To authorize and dispatch, type " + PHRASE + ": ") != PHRASE:
        raise PermissionError("OPERATOR_DECLINED")
    return _OperatorInvocation(confirmation(str(home.resolve())))

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="User-run only; terminal confirmation required")
    mode.add_argument("--diagnose", action="store_true", help="Metadata and check codes only; no key reads or sends")
    parser.add_argument("--credential-profile-home")
    parser.add_argument("--confirm-standard-tariff-no-autorefill", action="store_true")
    args = parser.parse_args(argv)
    if args.diagnose:
        print(json.dumps(diagnose(args.credential_profile_home, args.confirm_standard_tariff_no_autorefill), sort_keys=True)); return 0
    if not args.execute:
        print(json.dumps(dry_run(), sort_keys=True)); return 0
    verify_composition(); raw = request_bytes()
    if (operation_parent() / REQUEST_SHA256).exists():
        raise PermissionError("ATTEMPT_DIRECTORY_EXISTS_STOP")
    if args.credential_profile_home is None:
        raise PermissionError("EXPLICIT_CREDENTIAL_PROFILE_REQUIRED")
    record = confirm_operator(args.credential_profile_home, args.confirm_standard_tariff_no_autorefill)
    started = time.monotonic()
    with temporary_context() as ctx:
        from agent.secret_scope import build_profile_secret_scope, set_secret_scope, reset_secret_scope
        from .service import _execute
        scope = build_profile_secret_scope(Path(record.record["credential_profile_home"]))
        if not scope.get("TYPESAFE_API_KEY"):
            raise PermissionError("PROFILE_CREDENTIAL_UNAVAILABLE")
        # Scope is ephemeral; no .env, global environment or credential file is written.
        marker = set_secret_scope({"TYPESAFE_API_KEY": scope["TYPESAFE_API_KEY"]})
        try:
            result = _execute(ctx, raw, deadline_seconds=DEADLINE, fixture=None, _specimen=record, _started=started)
        finally:
            reset_secret_scope(marker)
    print(json.dumps({key: result.get(key) for key in
                      ("outcome", "attempts_observed", "http_status", "actual_model", "usage", "computed_api_metered_usd")}, sort_keys=True))
    return 0 if result.get("outcome") == "VALID_RESPONSE" else 2

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print("BLOCKED: " + safe_error_code(error) + "; use --diagnose (no key reads or sends).", file=sys.stderr)
        raise SystemExit(2)
