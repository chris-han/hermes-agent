"""Machine-readable ensure boundary. No bootstrap credentials leave this command."""
from dataclasses import asdict
import json
import sys
from contextlib import redirect_stdout


def ensure_exit_code(result) -> int:
    if result.reason_code == "deadline":
        return 5
    if result.reason_code in {"authorization", "profile_mismatch"}:
        return 4
    return {"ready": 0, "incompatible": 3, "draining": 6,
            "inaccessible": 7, "conflict": 7, "starting": 5, "absent": 5}[result.state]


def cmd_gateway_ensure(args) -> None:
    from hermes_constants import get_hermes_home
    from hermes_cli.gateway_runtime import ensure_gateway_runtime

    try:
        # Legacy imported utilities can print; protocol stdout belongs only to us.
        with redirect_stdout(sys.stderr):
            result = ensure_gateway_runtime(get_hermes_home(), timeout=float(args.timeout))
        payload = asdict(result)
        if result.endpoint:
            payload["endpoint"]["capabilities"] = sorted(result.endpoint.capabilities)
        # The code THIS client (and any gateway it starts) runs, resolved exactly like the owner's
        # ``endpoint.code_sha``: a client that sees them differ is attached to a gateway that
        # outlived ``hermes update`` and must restart it rather than re-attach.
        from hermes_cli.version_info import get_code_identity
        payload["client_code_sha"] = get_code_identity().get("sha")
        code = ensure_exit_code(result)
    except (ValueError, TypeError):
        payload, code = {"state": "inaccessible", "reason_code": "invalid_invocation", "endpoint": None}, 2
    print(json.dumps(payload, separators=(",", ":"), sort_keys=True))
    raise SystemExit(code)


def cmd_gateway_package_stop(args) -> None:
    """Explicit native handoff: guarded drain plus read-only absence proof, no start."""
    import time
    from hermes_constants import get_hermes_home
    from hermes_cli.gateway_runtime import discover_gateway_endpoint
    from hermes_cli.gateway_runtime_replace import replace_incompatible_owner, REFUSAL_DETAIL
    if getattr(args, 'all', False) or getattr(args, 'system', False):
        print('Native package shutdown cannot request an all-profile/system sweep.')
        raise SystemExit(2)
    home = get_hermes_home()
    deadline = time.monotonic() + 30
    refusal = replace_incompatible_owner(home, deadline=deadline, for_package_update=True)
    if refusal is None:
        while time.monotonic() < deadline:
            observed = discover_gateway_endpoint(home, timeout=min(2, deadline - time.monotonic()))
            if observed.state == 'absent':
                print('Native package Gateway shutdown confirmed.')
                return
            if observed.state not in {'starting', 'draining'}:
                break
            time.sleep(0.05)
        refusal = 'owner_did_not_exit'
    print('Native package Gateway shutdown refused: ' + REFUSAL_DETAIL.get(refusal, 'owner unverified'))
    raise SystemExit(2)
