from storage import hosted_postgres as hosted
# --- Runtime Integration for Contextual RL ---

from eos.validation_service import validate_v5, default_v5_artifacts
from eos.explainability_surface import build_explainability_surface
from eos.replay_resolution import replay_resolution
from eos.resolution_engine import run_resolution
from eos.replay_binding_store import verify_replay_binding
from eos.db import get_conn, now_iso, insert_json, init_db
import json, uuid

# Ensure all tables (including policy_stats) exist on import
init_db()


def compute_online_reward(active, shadow):
    reward = 0
    if active == shadow:
        reward += 0.2
    if active != "ALLOW" and shadow == "ALLOW":
        reward += 0.3
    if active == "ALLOW" and shadow != "ALLOW":
        reward -= 0.5
    return reward


def update_policy_stats(context, policy, reward):
    context_key = json.dumps(context, sort_keys=True)

    with get_conn() as conn:
        row = conn.execute(
            hosted.sql_query("SELECT * FROM policy_stats WHERE context_key=? AND policy=?", 'SELECT * FROM policy_stats WHERE context_key=%s AND policy=%s'),
            (context_key, policy)
        ).fetchone()

        if row:
            conn.execute(
                hosted.sql_query("UPDATE policy_stats SET count=?, reward=? WHERE id=?", 'UPDATE policy_stats SET count=%s, reward=%s WHERE id=%s'),
                (row["count"] + 1, row["reward"] + reward, row["id"])
            )
        else:
            conn.execute(
                hosted.sql_query("INSERT INTO policy_stats VALUES (?, ?, ?, ?, ?, ?)", 'INSERT INTO policy_stats VALUES (%s, %s, %s, %s, %s, %s)'),
                (
                    f"stat_{uuid.uuid4().hex}",
                    context_key,
                    policy,
                    1,
                    reward,
                    now_iso()
                )
            )


def load_policy_stats():
    stats = {}
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM policy_stats").fetchall()

    for r in rows:
        key = (r["context_key"], r["policy"])
        stats[key] = {
            "count": r["count"],
            "reward": r["reward"]
        }
    return stats


# --- NEW: RL policy proposal ---

def generate_candidate_policies(base_policy: dict):
    candidates = []
    for k, v in base_policy.items():
        if isinstance(v, bool):
            candidates.append({**base_policy, k: not v})
    return candidates


def evaluate_policy_candidate(policy_version: str, candidate_policy: dict, sample_justifications: list):
    total = len(sample_justifications)
    if total == 0:
        return {"reward": 0, "metrics": {}}

    drifted = 0
    accepted = 0
    risk = 0.0

    for J in sample_justifications:
        hist = validate_v5(J, policy_version).get("status")
        new = validate_v5(J, candidate_policy).get("status")

        if new == "ALLOW":
            accepted += 1

        if hist != new:
            drifted += 1
            if hist == "ALLOW" and new != "ALLOW":
                risk += 0.8
            elif hist != "ALLOW" and new == "ALLOW":
                risk += 0.3
            else:
                risk += 0.6

    drift_rate = drifted / total
    acceptance_rate = accepted / total
    avg_risk = risk / total

    reward = (-1.0 * drift_rate) + (0.5 * acceptance_rate) - (0.8 * avg_risk)

    return {
        "reward": reward,
        "metrics": {
            "drift_rate": drift_rate,
            "acceptance_rate": acceptance_rate,
            "risk": avg_risk
        }
    }


# simple callable function (no router wiring yet)
def rl_generate_policy_proposals(base_policy: dict, sample_justifications: list, base_version: str = "rb_v1"):
    candidates = generate_candidate_policies(base_policy)

    evaluated = []
    for c in candidates:
        result = evaluate_policy_candidate(base_version, c, sample_justifications)
        evaluated.append({
            "policy": c,
            "reward": result["reward"],
            "metrics": result["metrics"]
        })

    evaluated = sorted(evaluated, key=lambda x: x["reward"], reverse=True)

    if evaluated:
        best = evaluated[0]
        insert_json("policy_proposals", {
            "proposal_id": f"prop_{uuid.uuid4().hex}",
            "rule_change": json.dumps(best["policy"]),
            "source": "rl_auto",
            "confidence": best["reward"],
            "status": "candidate",
            "created_at": now_iso()
        })

    return evaluated[:5]


# --- Context extraction and runtime selection helpers ---
def extract_context(justification: dict) -> dict:
    bindings = justification.get("bindings", {}) or {}
    amount = bindings.get("amount")

    # normalize amount into buckets
    try:
        amt = float(amount)
    except Exception:
        amt = None

    if amt is None:
        amount_bucket = "unknown"
    elif amt <= 0:
        amount_bucket = "low"
    elif amt < 100:
        amount_bucket = "low"
    elif amt < 10000:
        amount_bucket = "mid"
    else:
        amount_bucket = "high"

    ctx = {
        "amount": amount,
        "amount_bucket": amount_bucket,
        "counterparty": bindings.get("counterparty"),
        "workflow": (justification.get("workflow") or {}).get("workflow"),
    }
    return ctx


def select_policy(context: dict, policies: list[str]) -> str:
    # deterministic selection based on context fingerprint
    if not policies:
        raise ValueError("no policies provided")

    key = json.dumps(context, sort_keys=True)
    idx = abs(hash(key)) % len(policies)
    return policies[idx]


def run_with_contextual_policy(policy_version: str, justification: dict, available_policies: list) -> dict:
    """Run a validation with a selected active policy and evaluate shadow policies.

    - extracts a small context from `justification`
    - selects an active policy from `available_policies`
    - validates active and shadow policies, updates policy stats
    - returns a summary dict with `active_policy`, `active_result`, and `context`
    """
    ctx = extract_context(justification)

    active_policy = select_policy(ctx, available_policies) if available_policies else policy_version

    # perform active validation (map policy id to policy bundle key if necessary)
    active_result = validate_v5(justification, active_policy)

    # evaluate shadows and update stats
    shadow_results = []
    for p in (available_policies or []):
        if p == active_policy:
            continue
        shadow_result = validate_v5(justification, p)
        shadow_results.append({"policy": p, **shadow_result})
        # determine status strings
        a_status = active_result.get("status")
        s_status = shadow_result.get("status")
        reward = compute_online_reward(a_status, s_status)
        try:
            update_policy_stats(ctx, p, reward)
        except Exception:
            # ignore persistence errors during test runs
            pass

    return {
        "context": ctx,
        "active_policy": active_policy,
        "active_result": active_result,
        "shadow_results": shadow_results,
    }


# ---------------------------------------------------------------------------
# EOS route handlers — mounted by apps/combined.py onto the canonical app
# ---------------------------------------------------------------------------

try:
    from fastapi import APIRouter, Body, HTTPException

    router = APIRouter()

    @router.post("/signals")
    def create_signal(body: dict):
        signal_id = f"sig_{uuid.uuid4().hex}"
        insert_json("signals", {
            "signal_id": signal_id,
            "signal_type": body.get("type", "generic"),
            "payload_json": json.dumps(body.get("payload", {}), sort_keys=True),
            "created_at": now_iso(),
        })
        return {"signal_id": signal_id}

    @router.post("/run/{signal_id}")
    def run_signal(signal_id: str):
        with get_conn() as conn:
            row = conn.execute(
                hosted.sql_query("SELECT payload_json FROM signals WHERE signal_id=?", 'SELECT payload_json FROM signals WHERE signal_id=%s'), (signal_id,)
            ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="signal not found")

        payload = json.loads(row["payload_json"])
        amount_str = payload.get("amount", "0")
        currency = payload.get("currency", "USD")

        J = {
            "pins": {"O_v": "v2026-05-02", "C_v": "rb_v2"},
            "semantic_claim": {
                "resources": [
                    {"delta": float(amount_str), "currency": currency},
                    {"delta": -float(amount_str), "currency": currency},
                ],
                "duality": {"settles": signal_id},
            },
            "bindings": {
                "amount": amount_str,
                "currency": currency,
                "counterparty": payload.get("counterparty", "unknown"),
            },
            "claims": [
                {
                    "id": f"claim:{signal_id}",
                    "key": "factual_evidence.signal",
                    "value": True,
                    "authority_domain": "factual_evidence",
                    "tag_authority_level": "descriptive",
                    "semantic_tier": "T0",
                    "source_type": "system",
                    "status": "ACTIVE",
                }
            ],
        }

        artifacts = default_v5_artifacts()
        validation_result = validate_v5(J, "rb_v2", artifacts=artifacts)
        resolution_result = run_resolution(
            claims=J.get("claims", []),
            constraint_results=validation_result.get("constraint_results", []),
            precedence_graph=artifacts["PrecedenceGraph_v"],
            resolution_rules=artifacts["ResolutionRules_v"],
            scope="execution",
            pins=J.get("pins", {}),
        )

        workflow_id = f"wf_{uuid.uuid4().hex}"
        rea_id = f"rea_{uuid.uuid4().hex}"
        justification_id = f"jst_{uuid.uuid4().hex}"
        validation_id = f"val_{uuid.uuid4().hex}"
        journal_id = f"jrn_{uuid.uuid4().hex}"

        insert_json(
            "workflows",
            {
                "workflow_id": workflow_id,
                "signal_id": signal_id,
                "workflow_type": "api_run_signal",
                "workflow_json": json.dumps({"signal_id": signal_id}, sort_keys=True),
                "status": "COMMITTED" if validation_result["status"] == "ALLOW" else "VALIDATED",
                "created_at": now_iso(),
            },
        )
        insert_json(
            "rea_claims",
            {
                "rea_id": rea_id,
                "workflow_id": workflow_id,
                "claim_json": json.dumps(J.get("semantic_claim", {}), sort_keys=True),
                "claim_hash": f"hash_{uuid.uuid4().hex}",
                "created_at": now_iso(),
            },
        )
        stored_justification = dict(J)
        stored_justification["governance_artifacts"] = artifacts
        insert_json(
            "justifications",
            {
                "justification_id": justification_id,
                "workflow_id": workflow_id,
                "rea_id": rea_id,
                "justification_json": json.dumps(stored_justification, sort_keys=True),
                "content_hash": f"content_{uuid.uuid4().hex}",
                "created_at": now_iso(),
            },
        )
        insert_json(
            "validation_results",
            {
                "validation_id": validation_id,
                "justification_id": justification_id,
                "status": validation_result["status"],
                "result_json": json.dumps(
                    {
                        **validation_result,
                        "resolution_result": resolution_result,
                        "pins": J.get("pins", {}),
                    },
                    sort_keys=True,
                ),
                "policy_bundle_version": "rb_v2",
                "created_at": now_iso(),
            },
        )

        insert_json("journal_entries", {
            "journal_id": journal_id,
            "justification_id": justification_id,
            "je_json": json.dumps({"justification": J}, sort_keys=True),
            "status": validation_result["status"],
            "posted_at": now_iso() if validation_result["status"] == "ALLOW" else None,
        })
        return {
            "journal_id": journal_id,
            "status": validation_result["status"],
            "justification_id": justification_id,
            "resolution_decision": resolution_result.get("final_decision"),
        }

    # --- Admin / Audit endpoints ---
    from eos.sqlite_event_store import SemantierEventStore
    import eos.signer as signer_mod

    @router.post("/admin/signing-key")
    def set_signing_key(
        body: dict = Body(..., examples={"default": {"summary": "Set signing key", "value": {"key": "my-secret-key"}}})
    ):
        """Set the HMAC signing key for event signing (admin).

        Example body: {"key": "my-secret-key"}
        """
        key = body.get("key")
        if not key:
            raise HTTPException(status_code=400, detail="missing key")
        signer_mod.set_signing_key(key.encode("utf-8"))
        return {"status": "ok"}

    @router.post("/merkle/batches")
    def create_merkle_batch(
        body: dict = Body(
            ...,
            examples={"default": {"summary": "Create Merkle batch", "value": {"event_ids": ["e1", "e2"]}}},
        )
    ):
        """Create a Merkle batch from a list of event IDs.

        Example body: {"event_ids": ["e1", "e2"]}
        """
        batch_id = body.get("batch_id") or f"batch_{uuid.uuid4().hex}"
        event_ids = body.get("event_ids")
        if not event_ids or not isinstance(event_ids, list):
            raise HTTPException(status_code=400, detail="event_ids must be a non-empty list")

        with SemantierEventStore() as store:
            try:
                store.create_merkle_batch(batch_id, event_ids)
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
            return {"batch_id": batch_id, "root": store.get_merkle_root(batch_id)}

    @router.get("/merkle/batches/{batch_id}/root")
    def merkle_root(batch_id: str):
        with SemantierEventStore() as store:
            root = store.get_merkle_root(batch_id)
        if root is None:
            raise HTTPException(status_code=404, detail="batch not found")
        return {"batch_id": batch_id, "root": root}

    @router.get("/merkle/{batch_id}/proofs/{event_id}")
    def merkle_proof(batch_id: str, event_id: str):
        with SemantierEventStore() as store:
            proof = store.get_merkle_proof(event_id, batch_id)
        if proof is None:
            raise HTTPException(status_code=404, detail="proof not found")
        return {"batch_id": batch_id, "event_id": event_id, "proof": proof}

    @router.get("/replay/{journal_id}")
    def replay_journal(journal_id: str):
        with get_conn() as conn:
            row = conn.execute(
                hosted.sql_query("SELECT je_json, status FROM journal_entries WHERE journal_id=?", 'SELECT je_json, status FROM journal_entries WHERE journal_id=%s'),
                (journal_id,),
            ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="journal not found")

        je_data = json.loads(row["je_json"])
        J = je_data.get("justification", {})
        original_status = row["status"]

        replay_result = validate_v5(J, "rb_v2")
        replay_status = replay_result["status"]

        return {
            "original": {"status": original_status},
            "replay": {"status": replay_status},
            "diff": {"status_changed": original_status != replay_status},
        }

    @router.get("/replay-resolution/{justification_id}")
    def replay_resolution_endpoint(justification_id: str):
        replay = replay_resolution(justification_id)
        if replay.get("status") == "NOT_FOUND":
            raise HTTPException(status_code=404, detail=replay.get("reason"))
        if replay.get("status") == "FAIL_CLOSED":
            raise HTTPException(status_code=409, detail=replay.get("reason"))
        return replay

    @router.get("/replay-binding/{event_id}")
    def replay_binding_endpoint(event_id: str):
        result = verify_replay_binding(event_id)
        if result.get("status") == "NOT_FOUND":
            raise HTTPException(status_code=404, detail=result.get("reason"))
        if result.get("status") == "FAIL_CLOSED":
            raise HTTPException(status_code=409, detail=result.get("reason"))
        return result

    @router.get("/explainability/{event_id}")
    def explainability_surface_endpoint(event_id: str):
        result = build_explainability_surface(event_id)
        if result.get("status") == "NOT_FOUND":
            raise HTTPException(status_code=404, detail=result.get("reason"))
        if result.get("status") == "INVALID_EXPLANATION_SURFACE":
            raise HTTPException(status_code=409, detail=result.get("reason", "invalid_explanation_surface"))
        return result

except ImportError:
    # FastAPI not installed — router unavailable
    router = None  # type: ignore
