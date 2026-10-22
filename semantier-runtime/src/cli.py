import json, re, sys
from type_system import load_type, type_check
from authority import authority_commit
from intent import build_intent
from approval import ApprovalStore
from status import get_status
from events import ExecutionEventStore
from replay import replay
from eos.benchmark_cli import run_results_bundle

IDEMPOTENCY_KEY_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


def normalize_idempotency_key(raw_key: str) -> str:
    key = raw_key.strip()
    if not IDEMPOTENCY_KEY_PATTERN.fullmatch(key):
        raise SystemExit("Invalid idempotency key: expected 64 hexadecimal characters")
    return key.lower()

def approve():
    if len(sys.argv) < 4:
        raise SystemExit("Usage: semantier approve <idempotency_key> <approver>")

    key = normalize_idempotency_key(sys.argv[2])
    approver = sys.argv[3]

    store = ApprovalStore()
    store.add_approval(key, approver)

    print(f"APPROVED: {key} by {approver}")

def status():
    if len(sys.argv) < 3:
        raise SystemExit("Usage: semantier status <idempotency_key>")

    key = normalize_idempotency_key(sys.argv[2])
    s = get_status(key)

    print(f"STATUS: {s['status']}")
    print(f"APPROVALS: {s['approvals']}/{s['required']}")
    print(f"APPROVERS: {', '.join(s['approvers']) or '-'}")

def list_events():
    store = ExecutionEventStore()
    events = store.list_events()

    for e in events:
        print(f"{e['occurred_at']} | {e['event_type']} | {e['payload']}")

def replay_cmd():
    state = replay()

    print("REPLAY RESULT")
    print("----------------")
    print(f"Committed decisions: {len(state['committed_keys'])}")

    for k in state["committed_keys"]:
        print(f"- {k}")


def benchmark_cmd():
    args = sys.argv[2:]
    if len(args) < 4 or len(args) % 2 != 0:
        raise SystemExit(
            "Usage: semantier benchmark (--cases <cases.json> | --v2-cases-dir <cases_dir>) "
            "[--baseline <baseline.json>] [--baseline-secondary <baseline_secondary.json>] "
            "--output <results.json>"
        )

    values = {}
    for i in range(0, len(args), 2):
        values[args[i]] = args[i + 1]

    has_cases = "--cases" in values
    has_v2_cases = "--v2-cases-dir" in values
    if has_cases == has_v2_cases:
        raise SystemExit(
            "Usage error: provide exactly one of --cases <cases.json> or --v2-cases-dir <cases_dir>"
        )
    if "--output" not in values:
        raise SystemExit("Usage error: --output <results.json> is required")

    run_results_bundle(
        case_path=values.get("--cases"),
        v2_cases_dir=values.get("--v2-cases-dir"),
        baseline_path=values.get("--baseline"),
        output_path=values["--output"],
        secondary_baseline_path=values.get("--baseline-secondary"),
    )
    print(f"WROTE: {values['--output']}")

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "approve":
        approve()
        return 0
    elif len(sys.argv) > 1 and sys.argv[1] == "status":
        status()
        return 0
    elif len(sys.argv) > 1 and sys.argv[1] == "events":
        list_events()
        return 0
    elif len(sys.argv) > 1 and sys.argv[1] == "replay":
        replay_cmd()
        return 0    
    elif len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        benchmark_cmd()
        return 0

    if len(sys.argv) < 3:
        raise SystemExit(
            "Usage:\n"
            "  semantier <phi.json> <semantic_type.yaml>\n"
            "  semantier approve <idempotency_key> <approver>"
        )

    phi_path = sys.argv[1]
    type_path = sys.argv[2]

    with open(phi_path) as f:
        phi = json.load(f)

    semantic_type = load_type(type_path)

    context = phi.get("context", {})

    valid = type_check(context, phi, semantic_type)
    
    intent = build_intent(phi)
    
    print(f"IDEMPOTENCY_KEY: {intent.idempotency_key()}")
    
    result = authority_commit(intent, valid)

    print(result)
    return 0

if __name__ == "__main__":
    sys.exit(main())
