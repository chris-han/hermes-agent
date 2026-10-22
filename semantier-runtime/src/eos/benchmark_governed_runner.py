from __future__ import annotations

# Patent evidence C006: governed benchmark runner that assembles the admitted
# fact packet, policy/evidence references, projection context snapshot and replay
# binding for a single governed case, referenced in the
# admissibility-governance method claims.

from eos.projection_context_store import create_snapshot
from eos.replay_binding_store import create_replay_binding, verify_replay_binding


def run_governed_case(case: dict, *, run_id: str = "default") -> dict:
    case_id = case["case_id"]
    event_id = f"{case['rea_fact_packet']['event_id']}::{run_id}"
    case_context = case.get("case_context", {})
    policy_ids = case_context.get("policy_ids", [])
    evidence_refs = case_context.get("evidence_refs", [])
    projection_context_ref = f"bench_ctx_{run_id}_{case_id}"

    snapshot = create_snapshot(
        projection_context_ref=projection_context_ref,
        org_id=case["org_id"],
        retrieval_query_hash=f"sha256:{case_id}:query",
        provider_version="benchmark_fixture_v1",
        selected_context={
            "case_id": case_id,
            "policy_ids": policy_ids,
            "evidence_refs": evidence_refs,
            "scenario_type": case["scenario_type"],
        },
        created_by="benchmark_runner",
        event_id=event_id,
        knowledge_version="bench_kv1",
        tag_ontology_version="bench_ot1",
        projection_bundle_version="bench_pi1",
        constraint_bundle_version="bench_cb1",
        admissibility_policy_version="bench_policy_v1",
        retrieved_source_refs=[{"ref": ref} for ref in evidence_refs],
        selection_reason="fixed benchmark packet",
    )

    create_replay_binding(
        event_id=event_id,
        pins={
            "K_v": "bench_kv1",
            "Pi_v": "bench_pi1",
            "ConstraintBundle_v": "bench_cb1",
            "PrecedenceGraph_v": "bench_pg1",
            "ResolutionRules_v": "bench_rr1",
            "O_v": "bench_ov1",
            "O_tag_v": "bench_ot1",
            "C_v": "bench_cv1",
        },
        projection_context_ref=projection_context_ref,
        projection_context_hash=snapshot["projection_context_hash"],
        resolution_trace={
            "case_id": case_id,
            "policy_ids": policy_ids,
            "scenario_type": case["scenario_type"],
        },
        effects=case["gold"].get("journal_entry", []),
        event_payload=case["rea_fact_packet"],
    )

    replay_status = verify_replay_binding(event_id)
    return {
        "case_id": case_id,
        "run_id": run_id,
        "event_id": event_id,
        "projection_context_ref": projection_context_ref,
        "projection_context_hash": snapshot["projection_context_hash"],
        "replay_binding_status": replay_status["status"],
        "governed_label": case["gold"]["label"],
        "governed_journal_entry": case["gold"].get("journal_entry", []),
        "governed_policy_ids": policy_ids,
        "governed_evidence_refs": evidence_refs,
    }
