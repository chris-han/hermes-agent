import uuid
from eos.projection_versioning import get_projection
from eos.execution_guard import compute_intent_id
from eos.validation_service import validate_v5, default_v5_artifacts
from eos.resolution_engine import run_resolution


def execute_with_governance(org_id, J_t, pi_t, semantic_projection_input, ontology, sandbox, idem_store, post_validator,
                            projection_version=None, event_store=None):
    # --- Step 1: load pinned projection (MUST NOT recompute) ---
    if projection_version is None:
        projection_version = (J_t.get("pins") or {}).get("projection_version")

    ontology_version = (J_t.get("pins") or {}).get("ontology_version") or ontology.get("version")

    proj = get_projection(org_id, ontology_version, projection_version)
    if proj is None:
        return {"status": "REJECT", "reason": "projection_not_found",
                "ontology_version": ontology_version, "projection_version": projection_version}

    constraint_layer = proj.get("policy", {})

    # --- Step 2: v5 validation + deterministic resolution ---
    artifacts = default_v5_artifacts()
    v = validate_v5(J_t, (J_t.get("pins") or {}).get("C_v", "rb_v1"), artifacts=artifacts)
    if v["status"] != "ALLOW":
        return {"status": "REJECT", "reason": "validation_failed", "validation": v}

    resolution = run_resolution(
        claims=(J_t.get("claims") or []),
        constraint_results=v.get("constraint_results", []),
        precedence_graph=artifacts["PrecedenceGraph_v"],
        resolution_rules=artifacts["ResolutionRules_v"],
        scope="execution",
        pins=(J_t.get("pins") or {}),
    )
    if resolution.get("execution_decision") != "ALLOW":
        return {"status": "REJECT", "reason": "resolution_blocked", "resolution": resolution}

    # --- Step 3: idempotency check via I_t = H(a_t, φ_t) ---
    intent_id = compute_intent_id(pi_t, J_t)
    if idem_store is not None and idem_store.seen(org_id, intent_id):
        return {"status": "REJECT", "reason": "duplicate_intent", "intent_id": intent_id}

    # --- Step 3.5: pre-execution lineage event (G5) ---
    # Append the intent to the append-only log *before* sandbox execution so
    # the chain is auditable even if the sandbox crashes mid-flight.
    if event_store is not None:
        event_store.append_event({
            "event_id": str(uuid.uuid4()),
            "org_id": org_id,
            "ontology_version": ontology_version,
            "projection_version": projection_version,
            "policy": constraint_layer,
            "justification": J_t,
            "execution": {"intent_id": intent_id, "status": "PENDING", "program": pi_t},
            "effects": [],
            "intent_id": intent_id,
        })

    pi_t["_validated"] = True
    pi_t["_intent_id"] = intent_id

    result = sandbox.execute(pi_t)

    # --- Step 4: commit idempotency record ---
    if idem_store is not None:
        idem_store.commit(org_id, intent_id)

    post = post_validator(org_id, ontology, pi_t, result)

    return post
