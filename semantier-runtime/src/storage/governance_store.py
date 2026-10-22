"""
Governance promotion store — customised Holographic pattern.

Manages policy_bundle and ontology_entry candidates through a deterministic
state machine: PENDING → APPROVED → ACTIVE | REJECTED.

Mirrors the Holographic MemoryStore design:
  - Local SQLite (WAL mode, thread-safe RLock)
  - Structured retrieval via category + status columns (no LLM in read path)
  - content_hash for idempotency (same pattern as UnifiedIdempotencyStore)
  - replay_result stores the replay gate output as JSON before approval

Reads the same database as eos/db.py (default: <runtime-root>/eos.db).
The policy_candidates table is created by eos/db.init_db().
"""

from __future__ import annotations

import json
from contextlib import contextmanager
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal, Optional
from uuid import uuid4

from eos.artifact_hashing import canonical_json, content_hash
from runtime_paths import sqlite_db_path
from storage.hosted_postgres import enabled,connect_authority,require_tables,sql_query,begin_write
from storage.transaction_scope import transaction_scope,StoreMutationLock
from eos.t5_drift_audit import audit_t5_candidate_jurisdiction

_DB_PATH = sqlite_db_path("eos.db", ".semantier/eos.db")
_VALID_CATEGORIES = ("policy_bundle", "ontology_entry", "coa_version", "investment_radar_universe", "investment_radar_scoring_profile", "learned_procedure", "organization_learning_contribution")
_VALID_STATUSES = ("PENDING", "APPROVED", "REJECTED", "ACTIVE")

CandidateStatus = Literal["PENDING", "APPROVED", "REJECTED", "ACTIVE"]
CandidateCategory = Literal["policy_bundle", "ontology_entry", "coa_version", "investment_radar_universe", "investment_radar_scoring_profile", "learned_procedure", "organization_learning_contribution"]
CandidateSemanticTier = Literal["T1", "T2", "T3", "T4", "T5", "T6"]

_VALID_SEMANTIC_TIERS = ("T1", "T2", "T3", "T4", "T5", "T6")
_CATEGORY_DEFAULT_TIERS: dict[CandidateCategory, CandidateSemanticTier] = {
    "policy_bundle": "T5",
    "ontology_entry": "T1",
    "coa_version": "T3",
    "investment_radar_universe": "T5",
    "investment_radar_scoring_profile": "T5",
    "learned_procedure": "T5",
    "organization_learning_contribution": "T5",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _db_path() -> Path:
    return Path(os.environ.get("SEMANTIER_EOS_DB_PATH", str(_DB_PATH)))


def _bundle_version(bundle: dict) -> str:
    bundle_id = bundle.get("bundle_id") or bundle.get("name") or "bundle"
    version = bundle.get("version")
    if version is None:
        return str(bundle_id)
    return f"{bundle_id}:{version}"


def _loads_governed_content(raw: str) -> dict:
    try:
        loaded = json.loads(raw)
        if isinstance(loaded, dict):
            return loaded
    except Exception:
        pass
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(raw)
        if isinstance(loaded, dict):
            return loaded
    except Exception:
        pass
    raise ValueError("Unsupported governed content format; expected JSON/YAML object")


def _default_semantic_tier(category: CandidateCategory) -> CandidateSemanticTier:
    return _CATEGORY_DEFAULT_TIERS[category]


def _validate_t5_governance_metadata(
    *,
    semantic_tier: str,
    scoped_owner: Optional[str],
    rollback_window_hours: Optional[int],
    rollback_strategy: Optional[str],
) -> None:
    if semantic_tier != "T5":
        return
    if not scoped_owner:
        raise ValueError("T5 candidates require scoped_owner")
    if rollback_window_hours is None or rollback_window_hours <= 0 or rollback_window_hours > 24:
        raise ValueError("T5 candidates require rollback_window_hours between 1 and 24")
    if not rollback_strategy:
        raise ValueError("T5 candidates require rollback_strategy")


class GovernanceStore:
    """SQLite-backed governance candidate store.

    Follows the Holographic MemoryStore pattern:
    - One connection per instance, WAL mode, threading.RLock
    - Structured retrieval by category/status — no LLM in any code path
    - content_hash deduplication (SHA-256 of canonical YAML)

    State machine transitions::

        PENDING  →  APPROVED  (record_approval)
        PENDING  →  REJECTED  (record_rejection)
        APPROVED →  ACTIVE    (activate)
        ACTIVE   →  *         (superseded implicitly when a newer one is activated)

    Only one candidate per name can be active at a time. Older active
    candidates with the same name are demoted back to APPROVED when a newer
    one is activated.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        path = Path(db_path) if db_path else _db_path()
        self._db_path = path
        if enabled():
            self._conn=connect_authority("eos",path)
            require_tables(self._conn,'policy_candidates','policy_candidate_events','coa_versions','projection_bundles','coa_projection_bundles','coa_entries','coa_mapping_rules')
            self._conn.commit()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(path), check_same_thread=False, timeout=10.0)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._lock = StoreMutationLock(self._conn)

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def organization_contribution_request(self, *, organization_id, actor, request_id):
        with self._lock:
            row = self._conn.execute(sql_query("SELECT * FROM policy_candidates WHERE category=? "
                "AND scoped_owner=? AND proposed_by=? AND json_extract(content_yaml,'$.request_id')=?", "SELECT * FROM policy_candidates WHERE category=%s AND scoped_owner=%s AND proposed_by=%s AND (CASE WHEN category='organization_learning_contribution' THEN content_yaml::jsonb #>> '{request_id}' ELSE NULL END)=%s"),
                ('organization_learning_contribution', organization_id, actor, request_id)).fetchone()
            return dict(row) if row else None

    def publish_organization_contribution(self, *, body, verify_authority):
        """Bind request and immutable snapshot atomically across store connections."""
        with self._lock:
            begin_write(self._conn)
            try:
                verify_authority()
                prior = self.organization_contribution_request(organization_id=body['destination_organization_id'],
                    actor=body['submitter'], request_id=body['request_id'])
                if prior:
                    if json.loads(prior['content_yaml'])['command_hash'] != body['command_hash']:
                        raise ValueError('REQUEST_CONFLICT')
                    self._conn.commit()
                    return prior
                digest = content_hash(body)
                return self.propose(name=f"organization-contribution:{body['destination_organization_id']}:{digest}",
                    category='organization_learning_contribution', scoped_owner=body['destination_organization_id'],
                    content_yaml=canonical_json(body), content_hash=digest, proposed_by=body['submitter'],
                    rollback_window_hours=24, rollback_strategy='Retain snapshot and create a separately reviewed successor')
            except BaseException:
                self._conn.rollback()
                raise

    def propose(
        self,
        name: str,
        category: CandidateCategory,
        content_yaml: str,
        content_hash: str,
        proposed_by: str,
        semantic_tier: CandidateSemanticTier | None = None,
        scoped_owner: str | None = None,
        rollback_window_hours: int | None = None,
        rollback_strategy: str | None = None,
        notes: str | None = None,
        jurisdiction_review: dict | None = None,
    ) -> dict:
        """Insert a new PENDING candidate.

        Idempotent on content_hash — returns the existing row if the same
        content has already been proposed (regardless of its current status).

        Raises ValueError if name already exists with different content.
        """
        if category not in _VALID_CATEGORIES:
            raise ValueError(f"category must be one of {_VALID_CATEGORIES}")
        semantic_tier = semantic_tier or _default_semantic_tier(category)
        if category in ("learned_procedure", "organization_learning_contribution") and semantic_tier != "T5":
            raise ValueError(f"{category} must remain scoped T5")
        if semantic_tier not in _VALID_SEMANTIC_TIERS:
            raise ValueError(f"semantic_tier must be one of {_VALID_SEMANTIC_TIERS}")
        _validate_t5_governance_metadata(
            semantic_tier=semantic_tier,
            scoped_owner=scoped_owner,
            rollback_window_hours=rollback_window_hours,
            rollback_strategy=rollback_strategy,
        )

        with self._lock:
            existing = self._conn.execute(
                sql_query("SELECT * FROM policy_candidates WHERE content_hash = ?", 'SELECT * FROM policy_candidates WHERE content_hash = %s'),
                (content_hash,),
            ).fetchone()
            if existing:
                if (category == "learned_procedure" or existing["category"] == "learned_procedure") and (
                    existing["category"], existing["name"], existing["scoped_owner"]
                ) != (category, name, scoped_owner):
                    raise ValueError("LEARNED_PROCEDURE_SCOPE_MISMATCH")
                return dict(existing)

            name_row = self._conn.execute(
                sql_query("SELECT candidate_id, status FROM policy_candidates"
                " WHERE name = ? ORDER BY created_at DESC LIMIT 1", 'SELECT candidate_id, status FROM policy_candidates WHERE name = %s ORDER BY created_at DESC LIMIT 1'),
                (name,),
            ).fetchone()
            if category != "learned_procedure" and name_row and name_row["status"] in ("PENDING", "APPROVED"):
                raise ValueError(
                    f"Candidate '{name}' already exists with status={name_row['status']}. "
                    "Reject or activate it before proposing a replacement."
                )

            candidate_id = f"cand_{uuid4().hex}"
            now = _now_iso()
            review_status = None
            jurisdiction_reason = None
            if jurisdiction_review is not None:
                review_status = jurisdiction_review.get("status") or (
                    "OK" if jurisdiction_review.get("fast_track_allowed") else "REVIEW_REQUIRED"
                )
                jurisdiction_reason = jurisdiction_review.get("reason") or jurisdiction_review.get("route_family")
            self._conn.execute(
                sql_query("""
                INSERT INTO policy_candidates
                  (candidate_id, name, category, semantic_tier, content_yaml, content_hash,
                   status, proposed_by, scoped_owner, rollback_window_hours, rollback_strategy, created_at, notes,
                   review_status, jurisdiction_reason)
                VALUES (?, ?, ?, ?, ?, ?, 'PENDING', ?, ?, ?, ?, ?, ?, ?, ?)
                """, "\n                INSERT INTO policy_candidates\n                  (candidate_id, name, category, semantic_tier, content_yaml, content_hash,\n                   status, proposed_by, scoped_owner, rollback_window_hours, rollback_strategy, created_at, notes,\n                   review_status, jurisdiction_reason)\n                VALUES (%s, %s, %s, %s, %s, %s, 'PENDING', %s, %s, %s, %s, %s, %s, %s, %s)\n                "),
                (candidate_id, name, category, semantic_tier, content_yaml, content_hash,
                 proposed_by, scoped_owner, rollback_window_hours, rollback_strategy, now, notes,
                 review_status, jurisdiction_reason),
            )
            self._record_event(
                candidate_id=candidate_id,
                event_type="PROPOSED",
                actor=proposed_by,
                prior_version=None,
                new_version="PENDING",
                event_payload={
                    "name": name,
                    "category": category,
                    "semantic_tier": semantic_tier,
                    "content_hash": content_hash,
                    "proposed_by": proposed_by,
                    "scoped_owner": scoped_owner,
                    "rollback_window_hours": rollback_window_hours,
                    "rollback_strategy": rollback_strategy,
                    "notes": notes,
                    "jurisdiction_review": jurisdiction_review,
                },
                created_at=now,
            )
            self._conn.commit()
            return self._get_by_id(candidate_id)  # type: ignore[return-value]

    def record_replay_result(
        self,
        candidate_id: str,
        regressions: list[dict],
        tested_claims: int,
        tested_at: str | None = None,
    ) -> dict:
        """Attach the replay gate result to a PENDING candidate.

        regressions: list of {rea_id, old_status, new_status} dicts.
        If regressions is non-empty the candidate should be rejected by the
        human reviewer; this method only records — it does not auto-reject.
        """
        result_json = json.dumps(
            {
                "regressions": regressions,
                "tested_claims": tested_claims,
                "tested_at": tested_at or _now_iso(),
            },
            sort_keys=True,
        )
        with self._lock:
            self._assert_status(candidate_id, "PENDING")
            self._conn.execute(
                sql_query("UPDATE policy_candidates SET replay_result = ? WHERE candidate_id = ?", 'UPDATE policy_candidates SET replay_result = %s WHERE candidate_id = %s'),
                (result_json, candidate_id),
            )
            self._record_event(
                candidate_id=candidate_id,
                event_type="REPLAY_RECORDED",
                actor="system",
                prior_version="PENDING",
                new_version="PENDING",
                event_payload=json.loads(result_json),
            )
            self._conn.commit()
            return self._get_by_id(candidate_id)  # type: ignore[return-value]

    def record_approval(
        self,
        candidate_id: str,
        approved_by: str,
        *,
        actor_roles: Optional[Iterable[str]] = None,
    ) -> dict:
        """Transition PENDING → APPROVED.

        Requires replay_result to be present (replay gate must have run).
        """
        with self._lock:
            row = self._assert_status(candidate_id, "PENDING")
            if not row["replay_result"]:
                raise ValueError(
                    f"Candidate {candidate_id} has no replay_result. "
                    "Run the replay gate before approving."
                )
            now = _now_iso()
            self._conn.execute(
                sql_query("""
                UPDATE policy_candidates
                SET status = 'APPROVED', approved_by = ?, approved_at = ?
                WHERE candidate_id = ?
                """, "\n                UPDATE policy_candidates\n                SET status = 'APPROVED', approved_by = %s, approved_at = %s\n                WHERE candidate_id = %s\n                "),
                (approved_by, now, candidate_id),
            )
            self._record_event(
                candidate_id=candidate_id,
                event_type="APPROVED",
                actor=approved_by,
                prior_version="PENDING",
                new_version="APPROVED",
                event_payload={"approved_by": approved_by, "approved_at": now},
                created_at=now,
            )
            self._conn.commit()
            return self._get_by_id(candidate_id)  # type: ignore[return-value]

    def decide_organization_contribution(self, *, candidate_id, expected_hash, approved,
                                        actor, expected_baseline, verify_authority):
        """Exact immutable reviewer decision with organization-baseline CAS under lock."""
        with self._lock:
            begin_write(self._conn)
            try:
                verify_authority()
                row = self._get_by_id(candidate_id)
                if (not row or row['category'] != 'organization_learning_contribution'
                        or row['content_hash'] != expected_hash):
                    raise ValueError('EXACT_CONTRIBUTION_MISMATCH')
                desired = 'APPROVED' if approved else 'REJECTED'
                if row['status'] == desired:
                    self._conn.commit()
                    return row
                if row['status'] != 'PENDING':
                    raise ValueError('IMMUTABLE_ORGANIZATION_DECISION')
                body = json.loads(row['content_yaml'])
                if body['organization_baseline'] != expected_baseline:
                    raise ValueError('STALE_ORGANIZATION_BASELINE')
                baselines = [dict(item) for item in self._conn.execute(
                    sql_query("SELECT * FROM policy_candidates WHERE category=? AND scoped_owner=? AND status='APPROVED'", "SELECT * FROM policy_candidates WHERE category=%s AND scoped_owner=%s AND status='APPROVED'"),
                    ('organization_learning_contribution', row['scoped_owner'])).fetchall()
                    if json.loads(item['content_yaml'])['skill_id'] == body['skill_id']]
                latest = max(baselines, key=lambda item: (item['approved_at'] or '', item['candidate_id'])) if baselines else None
                actual = {'ref': latest['candidate_id'], 'hash': latest['content_hash']} if latest else None
                if actual != expected_baseline:
                    raise ValueError('STALE_ORGANIZATION_BASELINE')
                now = _now_iso()
                self._conn.execute(sql_query('UPDATE policy_candidates SET status=?,approved_by=?,approved_at=? WHERE candidate_id=?', 'UPDATE policy_candidates SET status=%s,approved_by=%s,approved_at=%s WHERE candidate_id=%s'),
                    (desired, actor if approved else None, now if approved else None, candidate_id))
                self._record_event(candidate_id=candidate_id, event_type=desired, actor=actor,
                    prior_version='PENDING', new_version=desired,
                    event_payload={'snapshot_hash': expected_hash, 'organization_baseline': expected_baseline,
                        'activation': 'not_activated', 'distribution': 'unsupported'}, created_at=now)
                self._conn.commit()
                return self._get_by_id(candidate_id)
            except BaseException:
                self._conn.rollback()
                raise

    def record_rejection(self, candidate_id: str, rejected_by: str, reason: str = "") -> dict:
        """Transition PENDING → REJECTED."""
        with self._lock:
            self._assert_status(candidate_id, "PENDING")
            note = f"Rejected by {rejected_by}. {reason}".strip()
            self._conn.execute(
                sql_query("""
                UPDATE policy_candidates
                SET status = 'REJECTED', notes = ?
                WHERE candidate_id = ?
                """, "\n                UPDATE policy_candidates\n                SET status = 'REJECTED', notes = %s\n                WHERE candidate_id = %s\n                "),
                (note, candidate_id),
            )
            self._record_event(
                candidate_id=candidate_id,
                event_type="REJECTED",
                actor=rejected_by,
                prior_version="PENDING",
                new_version="REJECTED",
                event_payload={"rejected_by": rejected_by, "reason": reason, "notes": note},
            )
            self._conn.commit()
            return self._get_by_id(candidate_id)  # type: ignore[return-value]

    def activate(
        self,
        candidate_id: str,
        *,
        actor_roles: Optional[Iterable[str]] = None,
        actor: str | None = None,
    ) -> dict:
        """Transition APPROVED → ACTIVE.

        Any previously ACTIVE candidate with the same name is demoted to
        APPROVED before the requested candidate becomes ACTIVE.
        """
        with self._lock:
            row = self._assert_status(candidate_id, "APPROVED")
            if row["category"] == "organization_learning_contribution":
                raise ValueError("ORGANIZATION_DISTRIBUTION_UNSUPPORTED")
            superseded_rows = self._conn.execute(
                sql_query("SELECT candidate_id FROM policy_candidates WHERE name=? AND status='ACTIVE' AND candidate_id != ?", "SELECT candidate_id FROM policy_candidates WHERE name=%s AND status='ACTIVE' AND candidate_id != %s"),
                (row["name"], candidate_id),
            ).fetchall()
            self._conn.execute(
                sql_query("""
                UPDATE policy_candidates
                SET status = 'APPROVED'
                WHERE name = ? AND status = 'ACTIVE' AND candidate_id != ?
                """, "\n                UPDATE policy_candidates\n                SET status = 'APPROVED'\n                WHERE name = %s AND status = 'ACTIVE' AND candidate_id != %s\n                "),
                (row["name"], candidate_id),
            )
            for superseded in superseded_rows:
                self._record_event(
                    candidate_id=superseded["candidate_id"],
                    event_type="SUPERSEDED",
                    actor="system",
                    prior_version="ACTIVE",
                    new_version="APPROVED",
                    event_payload={"superseded_by": candidate_id},
                )
            self._conn.execute(
                sql_query("UPDATE policy_candidates SET status = 'ACTIVE' WHERE candidate_id = ?", "UPDATE policy_candidates SET status = 'ACTIVE' WHERE candidate_id = %s"),
                (candidate_id,),
            )
            self._record_event(
                candidate_id=candidate_id,
                event_type="ACTIVATED",
                actor=actor or "system",
                prior_version="APPROVED",
                new_version="ACTIVE",
                event_payload={"name": row["name"], "category": row["category"]},
            )
            if row["category"] == "coa_version":
                self._sync_coa_bundle_tables(row)
            if not getattr(self, "_procedure_transition_depth", 0):
                self._conn.commit()
            return self._get_by_id(candidate_id)  # type: ignore[return-value]

    @contextmanager
    def learned_procedure_transition(self, *, name: str, expected_version: dict | None = None):
        """Serialize scoped activation/publication across SQLite connections.

        The optional exact predecessor is compared under the database write
        lock. Existing candidate artifacts and activation events remain the
        canonical state; this introduces no separate head store.
        """
        with self._lock:
            begin_write(self._conn)
            try:
                rows = self._conn.execute(
                    sql_query("SELECT * FROM policy_candidates WHERE name=? AND category='learned_procedure' AND status='ACTIVE'", "SELECT * FROM policy_candidates WHERE name=%s AND category='learned_procedure' AND status='ACTIVE'"),
                    (name,),
                ).fetchall()
                if len(rows) > 1:
                    raise ValueError("LEARNED_PROCEDURE_INTEGRITY")
                active = dict(rows[0]) if rows else None
                if expected_version is not None and active is not None:
                    actual = json.loads(active["content_yaml"])["version"]
                    if actual != expected_version:
                        raise ValueError("STALE_PREDECESSOR")
                self._procedure_transition_depth = 1
                yield active
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise
            finally:
                self._procedure_transition_depth = 0

    def fast_track_activate(
        self,
        candidate_id: str,
        *,
        actor: str,
        actor_roles: Optional[Iterable[str]] = None,
        notes: str = "",
    ) -> dict:
        """Transition PENDING -> ACTIVE for explicitly allowed scoped T5 paths only."""
        with self._lock:
            row = self._assert_status(candidate_id, "PENDING")
            if row["category"] == "learned_procedure":
                raise ValueError("learned_procedure requires separate approval and activation")
            if row["semantic_tier"] != "T5":
                raise ValueError("fast_track_activate_allowed_only_for_T5")
            _validate_t5_governance_metadata(
                semantic_tier=row["semantic_tier"],
                scoped_owner=row["scoped_owner"],
                rollback_window_hours=row["rollback_window_hours"],
                rollback_strategy=row["rollback_strategy"],
            )
            now = _now_iso()
            self._conn.execute(
                sql_query("""
                UPDATE policy_candidates
                SET status = 'ACTIVE', approved_by = ?, approved_at = ?, notes = ?
                WHERE candidate_id = ?
                """, "\n                UPDATE policy_candidates\n                SET status = 'ACTIVE', approved_by = %s, approved_at = %s, notes = %s\n                WHERE candidate_id = %s\n                "),
                (actor, now, notes or row["notes"], candidate_id),
            )
            self._record_event(
                candidate_id=candidate_id,
                event_type="ACTIVATED",
                actor=actor,
                prior_version="PENDING",
                new_version="ACTIVE",
                event_payload={
                    "name": row["name"],
                    "category": row["category"],
                    "semantic_tier": row["semantic_tier"],
                    "scoped_owner": row["scoped_owner"],
                    "rollback_window_hours": row["rollback_window_hours"],
                    "rollback_strategy": row["rollback_strategy"],
                    "fast_track": True,
                },
                created_at=now,
            )
            self._conn.commit()
            return self._get_by_id(candidate_id)  # type: ignore[return-value]

    def mark_legacy_t5_candidates_for_review(self) -> int:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM policy_candidates
                WHERE semantic_tier = 'T5' AND status = 'ACTIVE'
                  AND COALESCE(imported_legacy, 0) = 0
                """
            ).fetchall()
            updated = 0
            for row in rows:
                audit = audit_t5_candidate_jurisdiction(dict(row))
                self._conn.execute(
                    sql_query("""
                    UPDATE policy_candidates
                    SET imported_legacy = 1, review_status = ?, jurisdiction_reason = ?
                    WHERE candidate_id = ?
                    """, '\n                    UPDATE policy_candidates\n                    SET imported_legacy = 1, review_status = %s, jurisdiction_reason = %s\n                    WHERE candidate_id = %s\n                    '),
                    (audit["status"], audit["reason"], row["candidate_id"]),
                )
                updated += 1
            self._conn.commit()
            return updated

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def get_active(self, category: CandidateCategory) -> Optional[dict]:
        """Return the most recently activated candidate for a category, or None."""
        with self._lock:
            row = self._conn.execute(
                sql_query("""
                SELECT * FROM policy_candidates
                WHERE category = ? AND status = 'ACTIVE'
                ORDER BY approved_at DESC
                LIMIT 1
                """, "\n                SELECT * FROM policy_candidates\n                WHERE category = %s AND status = 'ACTIVE'\n                ORDER BY approved_at DESC\n                LIMIT 1\n                "),
                (category,),
            ).fetchone()
            return dict(row) if row else None

    def list_candidates(
        self,
        category: CandidateCategory | None = None,
        status: CandidateStatus | None = None,
        semantic_tier: CandidateSemanticTier | None = None,
    ) -> list[dict]:
        """Return candidates filtered by optional category and/or status."""
        clauses: list[str] = []
        params: list[str] = []
        if category is not None:
            clauses.append(sql_query("category = ?", 'category = %s'))
            params.append(category)
        if status is not None:
            clauses.append(sql_query("status = ?", 'status = %s'))
            params.append(status)
        if semantic_tier is not None:
            clauses.append(sql_query("semantic_tier = ?", 'semantic_tier = %s'))
            params.append(semantic_tier)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM policy_candidates {where} ORDER BY created_at DESC",
                params,
            ).fetchall()
            return [dict(r) for r in rows]

    def get_by_id(self, candidate_id: str) -> Optional[dict]:
        """Retrieve a candidate by its ID."""
        with self._lock:
            return self._get_by_id(candidate_id)

    def get_by_name(self, name: str) -> Optional[dict]:
        """Retrieve the most recent candidate with this name."""
        with self._lock:
            row = self._conn.execute(
                sql_query("SELECT * FROM policy_candidates WHERE name = ? ORDER BY created_at DESC LIMIT 1", 'SELECT * FROM policy_candidates WHERE name = %s ORDER BY created_at DESC LIMIT 1'),
                (name,),
            ).fetchone()
            return dict(row) if row else None

    def learning_graph_events(self, *, candidate, owner_ref, as_of, limit):
        """Exact owner-persisted candidate binding, bounded before receipt decoding."""
        with self._lock:
            rows = self._conn.execute(sql_query(
                "SELECT e.* FROM policy_candidate_events e JOIN policy_candidates c ON c.candidate_id=e.candidate_id WHERE c.category='learned_procedure' AND c.scoped_owner=? AND json_extract(c.content_yaml,'$.candidate.artifact_ref')=? AND json_extract(c.content_yaml,'$.candidate.artifact_hash')=? AND (substr(e.created_at,1,19)||'.'||CASE WHEN instr(e.created_at,'.')=0 THEN '000000' ELSE substr(e.created_at,instr(e.created_at,'.')+1,6) END||'Z')<=? ORDER BY e.created_at,e.candidate_event_id LIMIT ?",
                "SELECT e.* FROM policy_candidate_events e JOIN policy_candidates c ON c.candidate_id=e.candidate_id WHERE c.category='learned_procedure' AND c.scoped_owner=%s AND c.content_yaml::jsonb->'candidate'->>'artifact_ref'=%s AND c.content_yaml::jsonb->'candidate'->>'artifact_hash'=%s AND e.created_at::timestamptz<=%s::timestamptz ORDER BY e.created_at,e.candidate_event_id LIMIT %s"),
                (owner_ref,candidate.artifact_ref,candidate.artifact_hash,as_of,min(201,max(1,limit)))).fetchall()
            return [dict(row) for row in rows]

    def list_candidate_events(self, candidate_id: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                sql_query("""
                SELECT * FROM policy_candidate_events
                WHERE candidate_id = ?
                ORDER BY created_at ASC, candidate_event_id ASC
                """, '\n                SELECT * FROM policy_candidate_events\n                WHERE candidate_id = %s\n                ORDER BY created_at ASC, candidate_event_id ASC\n                '),
                (candidate_id,),
            ).fetchall()
            return [
                {
                    **dict(row),
                    "event": json.loads(row["event_json"]),
                }
                for row in rows
            ]

    def record_candidate_event(
        self,
        *,
        candidate_id: str,
        event_type: str,
        actor: str,
        event_payload: dict,
    ) -> None:
        """Append an event without changing the frozen candidate row."""
        with self._lock:
            row = self._assert_status(candidate_id, "PENDING")
            now = _now_iso()
            self._record_event(
                candidate_id=candidate_id,
                event_type=event_type,
                actor=actor,
                prior_version=row["status"],
                new_version=row["status"],
                event_payload=event_payload,
                created_at=now,
            )
            self._conn.commit()

    def record_learned_procedure_publication(self, *, candidate_id: str, actor: str, receipt: dict, versions: list[dict]) -> None:
        """Attest exact projection only under the already-held transition lock."""
        with self._lock:
            if not getattr(self, '_procedure_transition_depth', 0):
                raise ValueError('LEARNED_PROCEDURE_TRANSITION_REQUIRED')
            row = self._assert_status(candidate_id, 'ACTIVE')
            if row['category'] != 'learned_procedure' or row['scoped_owner'] != actor:
                raise ValueError('LEARNED_PROCEDURE_SCOPE_MISMATCH')
            payload = json.loads(row['content_yaml'])
            expected = list(payload['member_versions'].values()) if payload.get('member_versions') else [payload['version']]
            key = lambda pin: (pin['artifact_ref'], pin['artifact_hash'])
            if sorted(expected, key=key) != sorted(versions, key=key):
                raise ValueError('EXACT_PUBLICATION_VERSION_MISMATCH')
            event_payload = {'receipt': receipt, 'versions': versions}
            if any(e['event_type'] == 'PROCEDURE_PUBLISHED' and e['event']['payload'] == event_payload for e in self.list_candidate_events(candidate_id)):
                return
            self._record_event(candidate_id=candidate_id, event_type='PROCEDURE_PUBLISHED', actor=actor,
                prior_version='ACTIVE', new_version='ACTIVE', event_payload=event_payload)

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    def get_active_coa_bundle(self, org_id: str = "global") -> Optional[dict]:
        with self._lock:
            row = self._conn.execute(
                sql_query("""
                SELECT * FROM coa_projection_bundles
                WHERE org_id = ? AND status = 'ACTIVE'
                ORDER BY activated_at DESC, created_at DESC
                LIMIT 1
                """, "\n                SELECT * FROM coa_projection_bundles\n                WHERE org_id = %s AND status = 'ACTIVE'\n                ORDER BY activated_at DESC, created_at DESC\n                LIMIT 1\n                "),
                (org_id,),
            ).fetchone()
            return self._assemble_coa_bundle(dict(row)) if row else None

    def get_coa_bundle(self, bundle_version: str, org_id: str = "global") -> Optional[dict]:
        with self._lock:
            row = self._conn.execute(
                sql_query("""
                SELECT * FROM coa_projection_bundles
                WHERE bundle_version = ? AND org_id = ?
                LIMIT 1
                """, '\n                SELECT * FROM coa_projection_bundles\n                WHERE bundle_version = %s AND org_id = %s\n                LIMIT 1\n                '),
                (bundle_version, org_id),
            ).fetchone()
            return self._assemble_coa_bundle(dict(row)) if row else None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_by_id(self, candidate_id: str) -> Optional[dict]:
        row = self._conn.execute(
            sql_query("SELECT * FROM policy_candidates WHERE candidate_id = ?", 'SELECT * FROM policy_candidates WHERE candidate_id = %s'),
            (candidate_id,),
        ).fetchone()
        return dict(row) if row else None

    def _assert_status(self, candidate_id: str, expected: str) -> sqlite3.Row:
        row = self._conn.execute(
            sql_query("SELECT * FROM policy_candidates WHERE candidate_id = ?", 'SELECT * FROM policy_candidates WHERE candidate_id = %s'),
            (candidate_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"No candidate with id={candidate_id}")
        if row["status"] != expected:
            raise ValueError(
                f"Candidate {candidate_id} has status={row['status']}, expected {expected}."
            )
        return row

    def _record_event(
        self,
        *,
        candidate_id: str,
        event_type: str,
        actor: str,
        prior_version: str | None,
        new_version: str | None,
        event_payload: dict,
        created_at: str | None = None,
    ) -> None:
        event_id = f"cand_evt_{uuid4().hex}"
        occurred_at = created_at or _now_iso()
        envelope = {
            "event_id": event_id,
            "event_type": event_type,
            "occurred_at": occurred_at,
            "actor": actor,
            "source_refs": [],
            "prior_version": prior_version,
            "new_version": new_version,
            "justification_ref": None,
            "payload": event_payload,
        }
        envelope["content_hash"] = content_hash(envelope)
        self._conn.execute(
            sql_query("""
            INSERT INTO policy_candidate_events
              (candidate_event_id, candidate_id, event_type, event_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """, '\n            INSERT INTO policy_candidate_events\n              (candidate_event_id, candidate_id, event_type, event_json, created_at)\n            VALUES (%s, %s, %s, %s, %s)\n            '),
            (
                event_id,
                candidate_id,
                event_type,
                canonical_json(envelope),
                occurred_at,
            ),
        )

    def _sync_coa_bundle_tables(self, row: sqlite3.Row) -> None:
        bundle = _loads_governed_content(row["content_yaml"])
        org_id = bundle.get("org_id", "global")
        bundle_version = _bundle_version(bundle)
        created_at = row["created_at"] or _now_iso()
        activated_at = row["approved_at"] or _now_iso()
        source_refs = canonical_json(bundle.get("source_refs", []))

        self._conn.execute(
            sql_query("""
            INSERT OR IGNORE INTO coa_versions
              (coa_version_id, org_id, bundle_version, version_label, content_json,
               content_hash, status, created_at, activated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, '\n            INSERT INTO coa_versions\n              (coa_version_id, org_id, bundle_version, version_label, content_json,\n               content_hash, status, created_at, activated_at)\n            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)\n             ON CONFLICT DO NOTHING'),
            (
                f"coa_v_{bundle_version}",
                org_id,
                bundle_version,
                bundle_version,
                canonical_json(bundle),
                row["content_hash"],
                "ACTIVE",
                created_at,
                activated_at,
            ),
        )

        projection_bundle_version = bundle.get("projection_bundle_version", bundle_version)
        projection_bundle_payload = {
            "bundle_version": projection_bundle_version,
            "mapping_rules": bundle.get("mapping_rules", []),
            "source_refs": bundle.get("source_refs", []),
        }
        self._conn.execute(
            sql_query("""
            INSERT OR IGNORE INTO projection_bundles
              (projection_bundle_id, org_id, bundle_version, coa_version_id, content_json,
               content_hash, status, created_at, activated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, '\n            INSERT INTO projection_bundles\n              (projection_bundle_id, org_id, bundle_version, coa_version_id, content_json,\n               content_hash, status, created_at, activated_at)\n            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)\n             ON CONFLICT DO NOTHING'),
            (
                f"proj_bundle_{projection_bundle_version}",
                org_id,
                projection_bundle_version,
                f"coa_v_{bundle_version}",
                canonical_json(projection_bundle_payload),
                content_hash(projection_bundle_payload),
                "ACTIVE",
                created_at,
                activated_at,
            ),
        )

        self._conn.execute(
            sql_query("UPDATE coa_projection_bundles SET status = 'DEPRECATED' WHERE org_id = ? AND status = 'ACTIVE'", "UPDATE coa_projection_bundles SET status = 'DEPRECATED' WHERE org_id = %s AND status = 'ACTIVE'"),
            (org_id,),
        )
        self._conn.execute(
            sql_query("""
            INSERT OR REPLACE INTO coa_projection_bundles
              (bundle_version, org_id, bundle_id, jurisdiction, accounting_standard,
               status, content_yaml, content_hash, source_refs, created_at, activated_at)
            VALUES (?, ?, ?, ?, ?, 'ACTIVE', ?, ?, ?, ?, ?)
            """, "\n            INSERT INTO coa_projection_bundles\n              (bundle_version, org_id, bundle_id, jurisdiction, accounting_standard,\n               status, content_yaml, content_hash, source_refs, created_at, activated_at)\n            VALUES (%s, %s, %s, %s, %s, 'ACTIVE', %s, %s, %s, %s, %s)\n             ON CONFLICT(bundle_version) DO UPDATE SET org_id=excluded.org_id,bundle_id=excluded.bundle_id,jurisdiction=excluded.jurisdiction,accounting_standard=excluded.accounting_standard,status=excluded.status,content_yaml=excluded.content_yaml,content_hash=excluded.content_hash,source_refs=excluded.source_refs,created_at=excluded.created_at,activated_at=excluded.activated_at"),
            (
                bundle_version,
                org_id,
                bundle.get("bundle_id", row["name"]),
                bundle.get("jurisdiction", "UNKNOWN"),
                bundle.get("accounting_standard", "UNKNOWN"),
                canonical_json(bundle),
                row["content_hash"],
                source_refs,
                created_at,
                activated_at,
            ),
        )

        for entry in bundle.get("entries", []) or []:
            entry_hash = content_hash(entry)
            self._conn.execute(
                sql_query("""
                INSERT OR IGNORE INTO coa_entries
                  (coa_entry_id, org_id, bundle_version, account_code, account_name,
                   account_class, normal_balance, parent_code, auxiliary_dimensions,
                   authority_domain, tag_authority_level, semantic_tier, source_refs,
                   status, content_hash, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, '\n                INSERT INTO coa_entries\n                  (coa_entry_id, org_id, bundle_version, account_code, account_name,\n                   account_class, normal_balance, parent_code, auxiliary_dimensions,\n                   authority_domain, tag_authority_level, semantic_tier, source_refs,\n                   status, content_hash, created_at)\n                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)\n                 ON CONFLICT DO NOTHING'),
                (
                    entry.get("coa_entry_id") or f"coa_{bundle_version}_{entry['account_code']}",
                    org_id,
                    bundle_version,
                    entry["account_code"],
                    entry.get("account_name", entry["account_code"]),
                    entry.get("account_class", "expense"),
                    entry.get("normal_balance", "debit"),
                    entry.get("parent_code"),
                    canonical_json(entry.get("auxiliary_dimensions", [])),
                    entry.get("authority_domain", "accounting"),
                    entry.get("tag_authority_level", "accounting"),
                    entry.get("semantic_tier", "T3"),
                    canonical_json(entry.get("source_refs", [])),
                    entry.get("governance_state", "ACTIVE"),
                    entry_hash,
                    created_at,
                ),
            )

        for rule in bundle.get("mapping_rules", []) or []:
            rule_hash = content_hash(rule)
            self._conn.execute(
                sql_query("""
                INSERT OR IGNORE INTO coa_mapping_rules
                  (rule_id, org_id, projection_bundle_version, coa_bundle_version, rule_yaml,
                   content_hash, status, source_refs, created_at, activated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, '\n                INSERT INTO coa_mapping_rules\n                  (rule_id, org_id, projection_bundle_version, coa_bundle_version, rule_yaml,\n                   content_hash, status, source_refs, created_at, activated_at)\n                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)\n                 ON CONFLICT DO NOTHING'),
                (
                    f"{projection_bundle_version}:{rule['rule_id']}",
                    org_id,
                    projection_bundle_version,
                    bundle_version,
                    canonical_json(rule),
                    rule_hash,
                    rule.get("governance_state", "ACTIVE"),
                    canonical_json(rule.get("source_refs", [])),
                    created_at,
                    activated_at,
                ),
            )

    def _assemble_coa_bundle(self, row: dict) -> dict:
        bundle = _loads_governed_content(row["content_yaml"])
        org_id = row["org_id"]
        bundle_version = row["bundle_version"]
        entry_rows = self._conn.execute(
            sql_query("""
            SELECT * FROM coa_entries
            WHERE org_id = ? AND bundle_version = ? AND status = 'ACTIVE'
            ORDER BY account_code
            """, "\n            SELECT * FROM coa_entries\n            WHERE org_id = %s AND bundle_version = %s AND status = 'ACTIVE'\n            ORDER BY account_code\n            "),
            (org_id, bundle_version),
        ).fetchall()
        if entry_rows:
            bundle["entries"] = [
                {
                    **dict(entry),
                    "auxiliary_dimensions": json.loads(entry["auxiliary_dimensions"]),
                    "source_refs": json.loads(entry["source_refs"]),
                    "governance_state": entry["status"],
                }
                for entry in entry_rows
            ]
        rule_rows = self._conn.execute(
            sql_query("""
            SELECT * FROM coa_mapping_rules
            WHERE org_id = ? AND coa_bundle_version = ? AND status = 'ACTIVE'
            ORDER BY rule_id
            """, "\n            SELECT * FROM coa_mapping_rules\n            WHERE org_id = %s AND coa_bundle_version = %s AND status = 'ACTIVE'\n            ORDER BY rule_id\n            "),
            (org_id, bundle_version),
        ).fetchall()
        if rule_rows:
            bundle["mapping_rules"] = [
                {
                    **json.loads(rule["rule_yaml"]),
                    "governance_state": rule["status"],
                }
                for rule in rule_rows
            ]
        bundle["bundle_version"] = bundle_version
        return bundle


# ---------------------------------------------------------------------------
# Module-level singleton (mirrors get_draft_store pattern)
# ---------------------------------------------------------------------------

_store: GovernanceStore | None = None
_store_lock = threading.Lock()


def get_governance_store() -> GovernanceStore:
    """Return the process-level GovernanceStore singleton."""
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = GovernanceStore()
    return _store
