from typing import Dict, Any, List
from datetime import datetime, timezone
import hashlib

# --- Execution Trace Validation ---

def validate_trace(program: Dict[str, Any], trace: Dict[str, Any]) -> Dict[str, Any]:
    steps = program.get("steps", [])
    trace_steps = trace.get("steps", [])

    if len(steps) != len(trace_steps):
        return {"status": "REJECT", "reason": "trace_length_mismatch"}

    for p, t in zip(steps, trace_steps):
        if p.get("tool") != t.get("tool"):
            return {"status": "REJECT", "reason": "tool_mismatch"}

    return {"status": "ALLOW"}

# --- Effect Validation ---

def _normalize_effect(e: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "org_id": e.get("org_id"),
        "type": e.get("type"),
        "entity": e.get("entity"),
        "amount": float(e.get("amount", 0)),
        "currency": e.get("currency"),
        "ref": e.get("ref"),
        "timestamp": e.get("timestamp")
    }


def validate_effects(intended: List[Dict[str, Any]], actual: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(intended) != len(actual):
        return {"status": "REJECT", "reason": "effect_length_mismatch"}

    norm_intended = [_normalize_effect(e) for e in intended]
    norm_actual = [_normalize_effect(e) for e in actual]

    for i, a in zip(norm_intended, norm_actual):
        if i["org_id"] != a["org_id"]:
            return {"status": "REJECT", "reason": "org_mismatch"}
        if i["type"] != a["type"]:
            return {"status": "REJECT", "reason": "effect_type_mismatch"}
        if i["entity"] != a["entity"]:
            return {"status": "REJECT", "reason": "entity_mismatch"}
        if abs(i["amount"] - a["amount"]) > 1e-6:
            return {"status": "REJECT", "reason": "amount_mismatch"}
        if i["currency"] != a["currency"]:
            return {"status": "REJECT", "reason": "currency_mismatch"}

    return {"status": "ALLOW"}

# --- Economic Invariants ---

def validate_conservation(org_id: str, effects: List[Dict[str, Any]]) -> Dict[str, Any]:
    scoped = [e for e in effects if e.get("org_id") == org_id]
    total = sum(float(e.get("amount", 0)) for e in scoped)
    if abs(total) > 1e-6:
        return {"status": "REJECT", "reason": "not_conserved"}
    return {"status": "ALLOW"}


def validate_duality(org_id: str, effects: List[Dict[str, Any]]) -> Dict[str, Any]:
    scoped = [e for e in effects if e.get("org_id") == org_id]
    refs = [e.get("ref") for e in scoped if e.get("ref")]
    if refs and len(set(refs)) != 1:
        return {"status": "REJECT", "reason": "linkage_mismatch"}
    return {"status": "ALLOW"}


def validate_rea_linkage(org_id: str, effects: List[Dict[str, Any]]) -> Dict[str, Any]:
    # REA linkage currently follows the same reference consistency rule.
    return validate_duality(org_id, effects)


def validate_temporal(org_id: str, effects: List[Dict[str, Any]]) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    scoped = [e for e in effects if e.get("org_id") == org_id]
    for e in scoped:
        ts = e.get("timestamp")
        if ts:
            try:
                t = datetime.fromisoformat(ts)
                if t > now:
                    return {"status": "REJECT", "reason": "future_timestamp"}
            except Exception:
                return {"status": "REJECT", "reason": "invalid_timestamp"}
    return {"status": "ALLOW"}

# --- Intent Identity ---

def compute_intent_id(action: Dict[str, Any], phi: Dict[str, Any]) -> str:
    """I_t = H(a_t, φ_t) — deterministic identity for an execution intent."""
    import json
    raw = json.dumps({"action": action, "phi": phi}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()

# --- Idempotency ---

def compute_event_hash(org_id: str, program: Dict[str, Any], effects: List[Dict[str, Any]], context_hash: str = "", ontology_version: str = "", policy_version: str = "") -> str:
    raw = (
        str(org_id)
        + str(program)
        + str(effects)
        + str(context_hash)
        + str(ontology_version)
        + str(policy_version)
    )
    return hashlib.sha256(raw.encode()).hexdigest()

class _InMemoryIdemStore:
    """Lightweight in-process idempotency store (used when no explicit store is provided)."""

    def __init__(self):
        self._seen: set = set()

    def seen(self, org_id: str, hash_val: str) -> bool:
        return (org_id, hash_val) in self._seen

    def commit(self, org_id: str, hash_val: str) -> None:
        self._seen.add((org_id, hash_val))


_default_idem_store = _InMemoryIdemStore()

# --- Sandbox ---

class Sandbox:
    def __init__(self, org_id: str, allowed_tools: List[str]):
        self.org_id = org_id
        self.allowed_tools = set(allowed_tools)

    def execute(self, program: Dict[str, Any]) -> Dict[str, Any]:
        trace = {"steps": []}
        effects = []

        for step in program.get("steps", []):
            tool = step.get("tool")

            if tool not in self.allowed_tools:
                raise Exception("Sandbox violation: tool not allowed")

            trace["steps"].append({"tool": tool})

            effects.append({
                "org_id": self.org_id,
                "type": "ledger_update",
                "entity": step.get("inputs", {}).get("entity"),
                "amount": step.get("inputs", {}).get("amount", 0),
                "currency": step.get("inputs", {}).get("currency"),
                "ref": step.get("inputs", {}).get("ref"),
                "timestamp": step.get("inputs", {}).get("timestamp")
            })

        return {"trace": trace, "effects": effects}

# --- Combined ---

def validate_post_execution(org_id: str, program, trace, intended_effects, actual_effects, idem_store=None, context_hash: str = "", ontology_version: str = "", policy_version: str = ""):
    store = idem_store if idem_store is not None else _default_idem_store

    t = validate_trace(program, trace)
    if t["status"] != "ALLOW":
        return t

    e = validate_effects(intended_effects, actual_effects)
    if e["status"] != "ALLOW":
        return e

    de = validate_conservation(org_id, actual_effects)
    if de["status"] != "ALLOW":
        return de

    lk = validate_rea_linkage(org_id, actual_effects)
    if lk["status"] != "ALLOW":
        return lk

    tm = validate_temporal(org_id, actual_effects)
    if tm["status"] != "ALLOW":
        return tm

    event_hash = compute_event_hash(
        org_id,
        program,
        actual_effects,
        context_hash,
        ontology_version,
        policy_version
    )

    if store.seen(org_id, event_hash):
        return {"status": "REJECT", "reason": "duplicate_event"}

    store.commit(org_id, event_hash)

    return {"status": "ALLOW", "event_hash": event_hash}
