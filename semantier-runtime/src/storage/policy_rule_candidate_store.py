from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from eos.artifact_hashing import canonical_json, content_hash as compute_content_hash
from eos.db import init_db
from runtime_paths import sqlite_db_path
from storage.hosted_postgres import enabled,connect_authority,require_tables,sql_query,begin_write
from storage.transaction_scope import transaction_scope,StoreMutationLock

CandidateState = Literal[
    "DRAFT",
    "PROPOSED",
    "NEEDS_REVIEW",
    "APPROVED",
    "ACTIVATED",
    "SUPERSEDED",
    "REJECTED",
]
ActorType = Literal["ai", "plugin", "human", "migration", "system"]

_VALID_STATES = {
    "DRAFT",
    "PROPOSED",
    "NEEDS_REVIEW",
    "APPROVED",
    "ACTIVATED",
    "SUPERSEDED",
    "REJECTED",
}
_NON_AUTHORITY_ACTORS = {"ai", "plugin"}
_NON_AUTHORITY_INITIAL_STATES = {"DRAFT", "PROPOSED"}
_TRANSITIONS: dict[str, set[str]] = {
    "DRAFT": {"PROPOSED", "REJECTED"},
    "PROPOSED": {"NEEDS_REVIEW", "REJECTED"},
    "NEEDS_REVIEW": {"APPROVED", "REJECTED"},
    "APPROVED": {"ACTIVATED", "SUPERSEDED", "REJECTED"},
    "ACTIVATED": {"SUPERSEDED"},
    "SUPERSEDED": set(),
    "REJECTED": set(),
}


def _db_path() -> Path:
    return Path(os.environ.get("SEMANTIER_EOS_DB_PATH", str(sqlite_db_path("eos.db", ".semantier/eos.db"))))


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _as_json(value: Any) -> str:
    return canonical_json(value if value is not None else {})


def _loads_json(value: str) -> Any:
    return json.loads(value) if value else None


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    hydrated = dict(row)
    for field in (
        "source_anchor_refs_json",
        "applicability_scope_json",
        "human_edits_json",
        "approval_evidence_json",
        "test_evidence_json",
        "activation_refs_json",
    ):
        hydrated[field.removesuffix("_json")] = _loads_json(hydrated[field])
    return hydrated


def _assert_ascii(value: str, field: str) -> str:
    normalized = (value or "").strip()
    if not normalized:
        raise ValueError(f"{field} is required")
    try:
        normalized.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{field} must be an ASCII-stable identifier") from exc
    return normalized


def _assert_state(state: str) -> str:
    normalized = _assert_ascii(state, "candidate_state")
    if normalized not in _VALID_STATES:
        raise ValueError(f"candidate_state must be one of {sorted(_VALID_STATES)}")
    return normalized


class PolicyRuleCandidateStore:
    def __init__(self, db_path: str | Path | None = None) -> None:
        path = Path(db_path) if db_path else _db_path()
        if enabled():
            self._conn=connect_authority("eos",path)
            require_tables(self._conn,'policy_rule_candidate_events','policy_rule_candidates')
            self._conn.commit()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            prior_db_path = os.environ.get("SEMANTIER_EOS_DB_PATH")
            os.environ["SEMANTIER_EOS_DB_PATH"] = str(path)
            try:
                init_db()
            finally:
                if prior_db_path is None:
                    os.environ.pop("SEMANTIER_EOS_DB_PATH", None)
                else:
                    os.environ["SEMANTIER_EOS_DB_PATH"] = prior_db_path
            self._conn = sqlite3.connect(str(path), check_same_thread=False, timeout=10.0)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._lock = StoreMutationLock(self._conn)

    def create_candidate(
        self,
        *,
        rule_candidate_id: str | None = None,
        organization_id: str | None = None,
        proposed_workspace_scope: str | None = None,
        proposed_workspace_id: str | None = None,
        authority_domain: str | None = None,
        semantic_tier: str | None = None,
        jurisdiction: str | None = None,
        candidate_idempotency_key: str | None = None,
        rule_family: str,
        candidate_state: CandidateState = "DRAFT",
        source_anchor_refs: list[str] | None = None,
        applicability_scope: dict[str, Any] | None = None,
        extracted_rationale: str,
        draft_rule_text: str,
        severity: str | None = None,
        confidence: float | None = None,
        uncertainty_notes: str | None = None,
        human_edits: list[dict[str, Any]] | None = None,
        approval_evidence: dict[str, Any] | None = None,
        test_evidence: dict[str, Any] | None = None,
        supersedes_candidate_id: str | None = None,
        activation_refs: list[str] | None = None,
        created_by: str,
        created_by_actor_type: ActorType,
    ) -> dict[str, Any]:
        state = _assert_state(candidate_state)
        actor_type = _assert_ascii(created_by_actor_type, "created_by_actor_type")
        if actor_type in _NON_AUTHORITY_ACTORS and state not in _NON_AUTHORITY_INITIAL_STATES:
            raise ValueError("AI or plugin actors can create only DRAFT or PROPOSED candidates")
        candidate_id = _assert_ascii(rule_candidate_id or f"prc_{uuid4().hex}", "rule_candidate_id")
        organization_id = _assert_ascii(organization_id, "organization_id") if organization_id else None
        proposed_workspace_scope = (
            _assert_ascii(proposed_workspace_scope, "proposed_workspace_scope")
            if proposed_workspace_scope
            else None
        )
        proposed_workspace_id = (
            _assert_ascii(proposed_workspace_id, "proposed_workspace_id")
            if proposed_workspace_id
            else None
        )
        authority_domain = _assert_ascii(authority_domain, "authority_domain") if authority_domain else None
        semantic_tier = _assert_ascii(semantic_tier, "semantic_tier") if semantic_tier else None
        jurisdiction = _assert_ascii(jurisdiction, "jurisdiction") if jurisdiction else None
        candidate_idempotency_key = (
            _assert_ascii(candidate_idempotency_key, "candidate_idempotency_key")
            if candidate_idempotency_key
            else None
        )
        family = _assert_ascii(rule_family, "rule_family")
        actor = _assert_ascii(created_by, "created_by")
        now = _now_iso()
        payload = {
            "rule_candidate_id": candidate_id,
            "organization_id": organization_id,
            "proposed_workspace_scope": proposed_workspace_scope,
            "proposed_workspace_id": proposed_workspace_id,
            "authority_domain": authority_domain,
            "semantic_tier": semantic_tier,
            "jurisdiction": jurisdiction,
            "rule_family": family,
            "candidate_state": state,
            "source_anchor_refs": source_anchor_refs or [],
            "applicability_scope": applicability_scope or {},
            "extracted_rationale": extracted_rationale,
            "draft_rule_text": draft_rule_text,
            "severity": severity,
            "confidence": confidence,
            "uncertainty_notes": uncertainty_notes,
            "human_edits": human_edits or [],
            "approval_evidence": approval_evidence or {},
            "test_evidence": test_evidence or {},
            "supersedes_candidate_id": supersedes_candidate_id,
            "activation_refs": activation_refs or [],
            "candidate_idempotency_key": candidate_idempotency_key,
            "created_by": actor,
            "created_by_actor_type": actor_type,
        }
        candidate_hash = compute_content_hash(canonical_json(payload))
        with self._lock:
            self._conn.execute(
                sql_query("""
                INSERT INTO policy_rule_candidates (
                  rule_candidate_id, organization_id, proposed_workspace_scope,
                  proposed_workspace_id, authority_domain, semantic_tier,
                  jurisdiction, rule_family, candidate_state,
                  source_anchor_refs_json, applicability_scope_json,
                  extracted_rationale, draft_rule_text, severity, confidence,
                  uncertainty_notes, human_edits_json, approval_evidence_json,
                  test_evidence_json, supersedes_candidate_id, activation_refs_json,
                  candidate_idempotency_key, created_by, created_by_actor_type,
                  created_at, updated_at, content_hash
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, '\n                INSERT INTO policy_rule_candidates (\n                  rule_candidate_id, organization_id, proposed_workspace_scope,\n                  proposed_workspace_id, authority_domain, semantic_tier,\n                  jurisdiction, rule_family, candidate_state,\n                  source_anchor_refs_json, applicability_scope_json,\n                  extracted_rationale, draft_rule_text, severity, confidence,\n                  uncertainty_notes, human_edits_json, approval_evidence_json,\n                  test_evidence_json, supersedes_candidate_id, activation_refs_json,\n                  candidate_idempotency_key, created_by, created_by_actor_type,\n                  created_at, updated_at, content_hash\n                )\n                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)\n                '),
                (
                    candidate_id,
                    organization_id,
                    proposed_workspace_scope,
                    proposed_workspace_id,
                    authority_domain,
                    semantic_tier,
                    jurisdiction,
                    family,
                    state,
                    _as_json(source_anchor_refs or []),
                    _as_json(applicability_scope or {}),
                    extracted_rationale,
                    draft_rule_text,
                    severity,
                    confidence,
                    uncertainty_notes,
                    _as_json(human_edits or []),
                    _as_json(approval_evidence or {}),
                    _as_json(test_evidence or {}),
                    supersedes_candidate_id,
                    _as_json(activation_refs or []),
                    candidate_idempotency_key,
                    actor,
                    actor_type,
                    now,
                    now,
                    candidate_hash,
                ),
            )
            self._record_event(
                rule_candidate_id=candidate_id,
                event_type="CREATED",
                actor=actor,
                actor_authority=None,
                from_state=None,
                to_state=state,
                event_payload={"created_by_actor_type": actor_type},
                created_at=now,
            )
            self._conn.commit()
            return self.get_candidate(candidate_id)  # type: ignore[return-value]

    def propose(self, rule_candidate_id: str, *, actor: str) -> dict[str, Any]:
        return self.transition_candidate(
            rule_candidate_id,
            to_state="PROPOSED",
            actor=actor,
            event_type="PROPOSED",
        )

    def request_review(self, rule_candidate_id: str, *, actor: str) -> dict[str, Any]:
        return self.transition_candidate(
            rule_candidate_id,
            to_state="NEEDS_REVIEW",
            actor=actor,
            event_type="REVIEW_REQUESTED",
        )

    def approve(
        self,
        rule_candidate_id: str,
        *,
        actor: str,
        actor_authority: str,
        approval_evidence: dict[str, Any],
        test_evidence: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.transition_candidate(
            rule_candidate_id,
            to_state="APPROVED",
            actor=actor,
            actor_authority=actor_authority,
            event_type="APPROVED",
            event_payload={
                "approval_evidence": approval_evidence,
                "test_evidence": test_evidence or {},
            },
            updates={
                "approval_evidence_json": _as_json(approval_evidence),
                "test_evidence_json": _as_json(test_evidence or {}),
            },
        )

    def record_reviewer_edit(
        self,
        rule_candidate_id: str,
        *,
        actor: str,
        actor_authority: str,
        human_edits: list[dict[str, Any]],
        review_comments: str | None = None,
    ) -> dict[str, Any]:
        candidate_id = _assert_ascii(rule_candidate_id, "rule_candidate_id")
        actor = _assert_ascii(actor, "actor")
        now = _now_iso()
        with self._lock:
            row = self._conn.execute(
                sql_query("SELECT * FROM policy_rule_candidates WHERE rule_candidate_id = ?", 'SELECT * FROM policy_rule_candidates WHERE rule_candidate_id = %s'),
                (candidate_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"rule candidate not found: {candidate_id}")
            state = row["candidate_state"]
            if state not in {"PROPOSED", "NEEDS_REVIEW"}:
                raise ValueError(f"reviewer edits are not permitted for {state}")
            candidate = _row_to_dict(row)
            assert candidate is not None
            recomputed_idempotency_key = compute_content_hash(
                canonical_json(
                    {
                        "prior_candidate_idempotency_key": candidate.get(
                            "candidate_idempotency_key"
                        ),
                        "rule_candidate_id": candidate_id,
                        "human_edits": human_edits,
                        "review_comments": review_comments,
                    }
                )
            )
            self._conn.execute(
                sql_query("""
                UPDATE policy_rule_candidates
                SET human_edits_json = ?, candidate_idempotency_key = ?, updated_at = ?
                WHERE rule_candidate_id = ?
                """, '\n                UPDATE policy_rule_candidates\n                SET human_edits_json = %s, candidate_idempotency_key = %s, updated_at = %s\n                WHERE rule_candidate_id = %s\n                '),
                (_as_json(human_edits), recomputed_idempotency_key, now, candidate_id),
            )
            self._record_event(
                rule_candidate_id=candidate_id,
                event_type="REVIEWER_EDITED",
                actor=actor,
                actor_authority=actor_authority,
                from_state=state,
                to_state=state,
                event_payload={
                    "human_edits": human_edits,
                    "review_comments": review_comments,
                    "candidate_idempotency_key": recomputed_idempotency_key,
                },
                created_at=now,
            )
            self._conn.commit()
            return self.get_candidate(candidate_id)  # type: ignore[return-value]

    def reject(
        self,
        rule_candidate_id: str,
        *,
        actor: str,
        actor_authority: str,
        reason: str,
    ) -> dict[str, Any]:
        return self.transition_candidate(
            rule_candidate_id,
            to_state="REJECTED",
            actor=actor,
            actor_authority=actor_authority,
            event_type="REJECTED",
            event_payload={"reason": reason},
        )

    def mark_activated(
        self,
        rule_candidate_id: str,
        *,
        actor: str,
        actor_authority: str,
        activation_ref: str,
    ) -> dict[str, Any]:
        return self.transition_candidate(
            rule_candidate_id,
            to_state="ACTIVATED",
            actor=actor,
            actor_authority=actor_authority,
            event_type="ACTIVATED",
            event_payload={"activation_ref": activation_ref},
            updates={"activation_refs_json": _as_json([activation_ref])},
        )

    def supersede(
        self,
        rule_candidate_id: str,
        *,
        actor: str,
        actor_authority: str,
        superseded_by: str,
    ) -> dict[str, Any]:
        return self.transition_candidate(
            rule_candidate_id,
            to_state="SUPERSEDED",
            actor=actor,
            actor_authority=actor_authority,
            event_type="SUPERSEDED",
            event_payload={"superseded_by": superseded_by},
        )

    def transition_candidate(
        self,
        rule_candidate_id: str,
        *,
        to_state: CandidateState,
        actor: str,
        event_type: str,
        actor_authority: str | None = None,
        event_payload: dict[str, Any] | None = None,
        updates: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        candidate_id = _assert_ascii(rule_candidate_id, "rule_candidate_id")
        state = _assert_state(to_state)
        actor = _assert_ascii(actor, "actor")
        now = _now_iso()
        with self._lock:
            row = self._conn.execute(
                sql_query("SELECT * FROM policy_rule_candidates WHERE rule_candidate_id = ?", 'SELECT * FROM policy_rule_candidates WHERE rule_candidate_id = %s'),
                (candidate_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"rule candidate not found: {candidate_id}")
            from_state = row["candidate_state"]
            if state not in _TRANSITIONS.get(from_state, set()):
                raise ValueError(f"illegal candidate transition: {from_state} -> {state}")
            set_clauses = [sql_query("candidate_state = ?", 'candidate_state = %s'), sql_query("updated_at = ?", 'updated_at = %s')]
            values: list[Any] = [state, now]
            for column, value in (updates or {}).items():
                set_clauses.append(sql_query(f"{column} = ?", f"{column} = %s"))
                values.append(value)
            values.append(candidate_id)
            self._conn.execute(
                sql_query(f"UPDATE policy_rule_candidates SET {', '.join(set_clauses)} WHERE rule_candidate_id = ?", f"UPDATE policy_rule_candidates SET {', '.join(set_clauses)} WHERE rule_candidate_id = %s"),
                values,
            )
            self._record_event(
                rule_candidate_id=candidate_id,
                event_type=event_type,
                actor=actor,
                actor_authority=actor_authority,
                from_state=from_state,
                to_state=state,
                event_payload=event_payload or {},
                created_at=now,
            )
            self._conn.commit()
            return self.get_candidate(candidate_id)  # type: ignore[return-value]

    def get_candidate(self, rule_candidate_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            sql_query("SELECT * FROM policy_rule_candidates WHERE rule_candidate_id = ?", 'SELECT * FROM policy_rule_candidates WHERE rule_candidate_id = %s'),
            (_assert_ascii(rule_candidate_id, "rule_candidate_id"),),
        ).fetchone()
        return _row_to_dict(row)

    def get_candidate_by_idempotency_key(
        self,
        candidate_idempotency_key: str,
    ) -> dict[str, Any] | None:
        row = self._conn.execute(
            sql_query("SELECT * FROM policy_rule_candidates WHERE candidate_idempotency_key = ?", 'SELECT * FROM policy_rule_candidates WHERE candidate_idempotency_key = %s'),
            (_assert_ascii(candidate_idempotency_key, "candidate_idempotency_key"),),
        ).fetchone()
        return _row_to_dict(row)

    def list_events(self, rule_candidate_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            sql_query("""
            SELECT * FROM policy_rule_candidate_events
            WHERE rule_candidate_id = ?
            ORDER BY created_at, candidate_event_id
            """, '\n            SELECT * FROM policy_rule_candidate_events\n            WHERE rule_candidate_id = %s\n            ORDER BY created_at, candidate_event_id\n            '),
            (_assert_ascii(rule_candidate_id, "rule_candidate_id"),),
        ).fetchall()
        return [dict(row) for row in rows]

    def list_candidates(
        self,
        *,
        rule_family: str | None = None,
        candidate_state: CandidateState | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        values: list[Any] = []
        if rule_family is not None:
            clauses.append(sql_query("rule_family = ?", 'rule_family = %s'))
            values.append(_assert_ascii(rule_family, "rule_family"))
        if candidate_state is not None:
            clauses.append(sql_query("candidate_state = ?", 'candidate_state = %s'))
            values.append(_assert_state(candidate_state))
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        rows = self._conn.execute(
            f"SELECT * FROM policy_rule_candidates{where} ORDER BY updated_at DESC",
            values,
        ).fetchall()
        return [_row_to_dict(row) for row in rows if row is not None]  # type: ignore[list-item]

    def _record_event(
        self,
        *,
        rule_candidate_id: str,
        event_type: str,
        actor: str,
        actor_authority: str | None,
        from_state: str | None,
        to_state: str,
        event_payload: dict[str, Any],
        created_at: str,
    ) -> str:
        prior = self._conn.execute(
            sql_query("""
            SELECT event_hash FROM policy_rule_candidate_events
            WHERE rule_candidate_id = ?
            ORDER BY created_at DESC, candidate_event_id DESC
            LIMIT 1
            """, '\n            SELECT event_hash FROM policy_rule_candidate_events\n            WHERE rule_candidate_id = %s\n            ORDER BY created_at DESC, candidate_event_id DESC\n            LIMIT 1\n            '),
            (rule_candidate_id,),
        ).fetchone()
        prior_hash = prior["event_hash"] if prior else None
        event_id = f"prce_{uuid4().hex}"
        payload = {
            "rule_candidate_id": rule_candidate_id,
            "event_type": event_type,
            "actor": actor,
            "actor_authority": actor_authority,
            "from_state": from_state,
            "to_state": to_state,
            "event_payload": event_payload,
            "prior_event_hash": prior_hash,
            "created_at": created_at,
        }
        event_hash = compute_content_hash(canonical_json(payload))
        self._conn.execute(
            sql_query("""
            INSERT INTO policy_rule_candidate_events (
              candidate_event_id, rule_candidate_id, event_type, actor,
              actor_authority, from_state, to_state, event_json,
              prior_event_hash, event_hash, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, '\n            INSERT INTO policy_rule_candidate_events (\n              candidate_event_id, rule_candidate_id, event_type, actor,\n              actor_authority, from_state, to_state, event_json,\n              prior_event_hash, event_hash, created_at\n            )\n            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)\n            '),
            (
                event_id,
                rule_candidate_id,
                event_type,
                actor,
                actor_authority,
                from_state,
                to_state,
                _as_json(event_payload),
                prior_hash,
                event_hash,
                created_at,
            ),
        )
        return event_hash
