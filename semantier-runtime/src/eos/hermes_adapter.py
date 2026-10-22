from typing import Dict, Any
from eos.validation_service import validate_v5
from eos.projection_versioning import get_projection


def validate_execution(program: Dict[str, Any], policy: Dict[str, Any]):
    rules = policy.get("rules", {})

    max_steps = rules.get("max_steps", 10)
    allowed_tools = set(rules.get("allowed_tools", []))
    denied_tools = set(rules.get("denied_tools", []))
    max_runtime = rules.get("max_runtime_ms", 2000)
    allow_external_write = rules.get("side_effects", {}).get("external_write", False)

    steps = program.get("steps", [])

    # Step count enforcement
    if len(steps) > max_steps:
        return {"status": "REJECT", "reason": "too_many_steps"}

    # Tool enforcement
    for step in steps:
        tool = step.get("tool")

        if tool in denied_tools:
            return {"status": "REJECT", "reason": "tool_denied"}

        if allowed_tools and tool not in allowed_tools:
            return {"status": "REJECT", "reason": "tool_not_allowed"}

    # Runtime enforcement
    runtime = program.get("runtime_ms")
    if runtime is not None and runtime > max_runtime:
        return {"status": "REJECT", "reason": "runtime_exceeded"}

    # Side effect enforcement
    if program.get("external_write") and not allow_external_write:
        return {"status": "REJECT", "reason": "external_write_blocked"}

    # Network enforcement
    network_mode = rules.get("side_effects", {}).get("network_access", "restricted")
    if program.get("network_access") == "open" and network_mode != "open":
        return {"status": "REJECT", "reason": "network_blocked"}

    # Determinism enforcement
    determinism = rules.get("determinism", {})
    if determinism.get("require_seed") and not program.get("seed"):
        return {"status": "REJECT", "reason": "missing_seed"}

    if determinism.get("forbid_time_now") and program.get("uses_time_now"):
        return {"status": "REJECT", "reason": "nondeterministic_time"}

    return {"status": "ALLOW"}


def validate_hermes_output(J_t: Dict[str, Any], pi_t: Dict[str, Any], org_id: str = "") -> str:
    """Unified ALLOW/REJECT gate per §7.3.

    Loads the pinned ontology/projection versions from J_t.pins, runs both
    semantic and execution validation, and returns a single verdict.
    """
    pins = J_t.get("pins", {})
    ontology_version = pins.get("ontology_version", "")
    projection_version = pins.get("projection_version")

    # Load pinned projection for exec policy; fall back to empty policy if not found.
    proj = get_projection(org_id, ontology_version, projection_version) if org_id else None
    exec_policy = (proj or {}).get("policy", {})

    data_result = validate_v5(J_t)
    exec_result = validate_execution(pi_t, exec_policy)

    data_ok = data_result.get("status") == "ALLOW"
    exec_ok = exec_result.get("status") == "ALLOW"
    return "ALLOW" if (data_ok and exec_ok) else "REJECT"


def attach_skill_pins(J_t: Dict[str, Any], program: Dict[str, Any]):
    skills = {}

    for step in program.get("steps", []):
        skill = step.get("skill_ref")
        if skill:
            skills[skill["skill_id"]] = skill["version"]

    J_t.setdefault("pins", {})["skills"] = skills

    return J_t
