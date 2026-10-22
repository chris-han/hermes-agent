from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from storage.hosted_postgres import enabled, HostedStorageUnavailable

from .models import (
    CognitiveItem, DecisionContext, DecisionStep, DecisionStepConflictError,
    EvidenceRecord, LearningCandidate, LearningProcess, ItemType, Scope, Status,
)


class CognitiveContextStore:
    """Authoritative persistence boundary for the small cognitive contract."""

    def __init__(self, db_path: str | Path) -> None:
        if enabled():
            raise HostedStorageUnavailable('Cognitive context evaluation is outside the selected hosted stores; explicit migration is required')
        self.db_path = str(db_path)
        self.ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS cognitive_items (
              id TEXT PRIMARY KEY, type TEXT NOT NULL, scope TEXT NOT NULL, status TEXT NOT NULL,
              source_refs_json TEXT NOT NULL, content_json TEXT NOT NULL, workspace_id TEXT NOT NULL,
              organization_id TEXT NOT NULL, user_id TEXT, confidence REAL, created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_cognitive_items_scope ON cognitive_items(organization_id, workspace_id, status, type);
            CREATE TABLE IF NOT EXISTS cognitive_evidence (
              id TEXT PRIMARY KEY, workspace_id TEXT NOT NULL, organization_id TEXT NOT NULL,
              session_id TEXT NOT NULL, episode_id TEXT NOT NULL, turn_id TEXT NOT NULL,
              kind TEXT NOT NULL, content_json TEXT NOT NULL, created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_cognitive_evidence_session ON cognitive_evidence(organization_id, workspace_id, session_id, created_at);
            CREATE TABLE IF NOT EXISTS cognitive_contexts (
              context_id TEXT PRIMARY KEY, request_ref TEXT NOT NULL, evidence_refs_json TEXT NOT NULL,
              knowledge_refs_json TEXT NOT NULL, rule_refs_json TEXT NOT NULL, conflicts_json TEXT NOT NULL,
              context_hash TEXT NOT NULL, workspace_id TEXT NOT NULL, organization_id TEXT NOT NULL, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS cognitive_learning_candidates (
              id TEXT PRIMARY KEY, proposed_type TEXT NOT NULL, evidence_refs_json TEXT NOT NULL,
              hypothesis TEXT NOT NULL, confidence REAL NOT NULL, proposed_change_json TEXT NOT NULL,
              workspace_id TEXT NOT NULL, organization_id TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS cognitive_decision_steps (
              step_id TEXT PRIMARY KEY, step_key TEXT NOT NULL UNIQUE,
              organization_id TEXT NOT NULL, workspace_id TEXT NOT NULL,
              session_id TEXT NOT NULL, turn_id TEXT, request_ref TEXT NOT NULL,
              decision_context_ref TEXT NOT NULL, checkpoint_ref TEXT,
              trajectory_ref TEXT NOT NULL, action_refs_json TEXT NOT NULL,
              tool_refs_json TEXT NOT NULL, outcome_ref TEXT NOT NULL,
              event_kind TEXT NOT NULL, logical_step_id TEXT,
              execution_attempt_ref TEXT, execution_attempt_hash TEXT,
              attempt_index INTEGER NOT NULL DEFAULT 0,
              lane TEXT NOT NULL DEFAULT 'TASK', created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_cognitive_decision_steps_scope
              ON cognitive_decision_steps(organization_id, workspace_id, session_id);
            CREATE TABLE IF NOT EXISTS cognitive_derived_projections (
              projection_ref TEXT PRIMARY KEY, projection_hash TEXT NOT NULL,
              projection_kind TEXT NOT NULL, organization_id TEXT NOT NULL,
              workspace_id TEXT NOT NULL, payload_json TEXT NOT NULL,
              recorded_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_cognitive_derived_projections_scope
              ON cognitive_derived_projections(organization_id, workspace_id, projection_kind);
            CREATE TABLE IF NOT EXISTS cognitive_learning_processes (
              learning_evidence_key TEXT PRIMARY KEY,
              organization_id TEXT NOT NULL, workspace_id TEXT NOT NULL,
              step_id TEXT NOT NULL, event_kind TEXT NOT NULL, outcome_ref TEXT NOT NULL,
              stage TEXT NOT NULL CHECK(stage IN ('started','evidence_captured','proposal_captured','candidate_created','closed')),
              evidence_id TEXT, proposal_schema_version INTEGER, proposal_json TEXT,
              candidate_id TEXT,
              terminal_status TEXT CHECK(terminal_status IS NULL OR terminal_status IN ('active','pending','rejected','ignored')),
              item_id TEXT, last_error_code TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_cognitive_learning_processes_scope
              ON cognitive_learning_processes(organization_id, workspace_id, step_id);
            """)
            step_columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(cognitive_decision_steps)").fetchall()
            }
            migrations = (
                ("logical_step_id", "TEXT"),
                ("execution_attempt_ref", "TEXT"),
                ("execution_attempt_hash", "TEXT"),
                ("attempt_index", "INTEGER NOT NULL DEFAULT 0"),
                ("lane", "TEXT NOT NULL DEFAULT 'TASK'"),
            )
            for name, declaration in migrations:
                if name not in step_columns:
                    conn.execute(f"ALTER TABLE cognitive_decision_steps ADD COLUMN {name} {declaration}")

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def save_evidence(self, record: EvidenceRecord) -> EvidenceRecord:
        with self._connect() as conn:
            conn.execute("INSERT INTO cognitive_evidence VALUES (?,?,?,?,?,?,?,?,?)",
                         (record.id, record.workspace_id, record.organization_id, record.session_id,
                          record.episode_id, record.turn_id, record.kind, self._json(record.content), record.created_at))
        return record

    def get_evidence(self, evidence_id: str, *, workspace_id: str, organization_id: str) -> EvidenceRecord:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM cognitive_evidence WHERE id=? AND workspace_id=? AND organization_id=?",
                (evidence_id, workspace_id, organization_id),
            ).fetchone()
        if row is None:
            raise KeyError(evidence_id)
        return EvidenceRecord(row["id"], row["workspace_id"], row["organization_id"], row["session_id"],
                              row["episode_id"], row["turn_id"], row["kind"], json.loads(row["content_json"]), row["created_at"])

    def save_or_get_evidence(self, record: EvidenceRecord) -> EvidenceRecord:
        with self._connect() as conn:
            conn.execute("INSERT OR IGNORE INTO cognitive_evidence VALUES (?,?,?,?,?,?,?,?,?)",
                         (record.id, record.workspace_id, record.organization_id, record.session_id,
                          record.episode_id, record.turn_id, record.kind, self._json(record.content), record.created_at))
        return self.get_evidence(record.id, workspace_id=record.workspace_id, organization_id=record.organization_id)

    def save_item(self, item: CognitiveItem) -> CognitiveItem:
        with self._connect() as conn:
            conn.execute("INSERT INTO cognitive_items VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                         (item.id, item.type.value, item.scope.value, item.status.value, self._json(item.source_refs),
                          self._json(item.content), item.workspace_id, item.organization_id, item.user_id,
                          item.confidence, item.created_at))
        return item

    def save_or_get_item(self, item: CognitiveItem) -> CognitiveItem:
        with self._connect() as conn:
            conn.execute("INSERT OR IGNORE INTO cognitive_items VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                         (item.id, item.type.value, item.scope.value, item.status.value, self._json(item.source_refs),
                          self._json(item.content), item.workspace_id, item.organization_id, item.user_id,
                          item.confidence, item.created_at))
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM cognitive_items WHERE id=? AND workspace_id=? AND organization_id=?",
                               (item.id, item.workspace_id, item.organization_id)).fetchone()
        if row is None:
            raise KeyError(item.id)
        return self._item(row)

    def update_item_status(self, item_id: str, status: Status, *, workspace_id: str, organization_id: str) -> None:
        with self._connect() as conn:
            changed = conn.execute("UPDATE cognitive_items SET status=? WHERE id=? AND workspace_id=? AND organization_id=?",
                                   (status.value, item_id, workspace_id, organization_id)).rowcount
        if not changed:
            raise KeyError(item_id)

    def _item(self, row: sqlite3.Row) -> CognitiveItem:
        return CognitiveItem(row["id"], ItemType(row["type"]), Scope(row["scope"]), Status(row["status"]),
                             json.loads(row["source_refs_json"]), json.loads(row["content_json"]), row["workspace_id"],
                             row["organization_id"], row["user_id"], row["confidence"], row["created_at"])

    def list_items(self, *, workspace_id: str, organization_id: str, type: ItemType | None = None,
                   user_id: str | None = None, active_only: bool = True) -> list[CognitiveItem]:
        clauses = ["workspace_id=?", "organization_id=?"]
        params: list[Any] = [workspace_id, organization_id]
        if type is not None:
            clauses.append("type=?")
            params.append(type.value)
        if active_only:
            clauses.append("status=?")
            params.append(Status.ACTIVE.value)
        if user_id is not None:
            clauses.append("(scope != 'user' OR user_id=?)")
            params.append(user_id)
        with self._connect() as conn:
            rows = conn.execute(f"SELECT * FROM cognitive_items WHERE {' AND '.join(clauses)} ORDER BY created_at, id", params).fetchall()
        return [self._item(row) for row in rows]

    def save_context(self, context: DecisionContext) -> DecisionContext:
        with self._connect() as conn:
            conn.execute("INSERT INTO cognitive_contexts VALUES (?,?,?,?,?,?,?,?,?,?)",
                         (context.context_id, context.request_ref, self._json(context.evidence_refs), self._json(context.knowledge_refs),
                          self._json(context.rule_refs), self._json(context.conflicts), context.context_hash,
                          context.workspace_id, context.organization_id, context.created_at))
        return context

    def get_context(self, context_id: str, *, workspace_id: str, organization_id: str) -> DecisionContext:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM cognitive_contexts WHERE context_id=? AND workspace_id=? AND organization_id=?",
                               (context_id, workspace_id, organization_id)).fetchone()
        if row is None:
            raise KeyError(context_id)
        return DecisionContext(row["context_id"], row["request_ref"], json.loads(row["evidence_refs_json"]),
                               json.loads(row["knowledge_refs_json"]), json.loads(row["rule_refs_json"]),
                               json.loads(row["conflicts_json"]), row["context_hash"], row["workspace_id"],
                               row["organization_id"], row["created_at"])

    @staticmethod
    def _derived_projection_identity(projection: Any) -> tuple[str, str, str]:
        if hasattr(projection, "snapshot_ref") and hasattr(projection, "snapshot_hash"):
            return str(projection.snapshot_ref), str(projection.snapshot_hash), str(projection.schema_version)
        if hasattr(projection, "projection_ref") and hasattr(projection, "projection_hash"):
            return str(projection.projection_ref), str(projection.projection_hash), str(projection.schema_version)
        if hasattr(projection, "gate_ref") and hasattr(projection, "gate_hash"):
            return str(projection.gate_ref), str(projection.gate_hash), str(projection.schema_version)
        if hasattr(projection, "receipt_ref") and hasattr(projection, "receipt_hash"):
            return str(projection.receipt_ref), str(projection.receipt_hash), str(projection.schema_version)
        if hasattr(projection, "episode_ref") and hasattr(projection, "episode_hash"):
            return str(projection.episode_ref), str(projection.episode_hash), str(projection.schema_version)
        if hasattr(projection, "attribution_ref") and hasattr(projection, "attribution_hash"):
            return str(projection.attribution_ref), str(projection.attribution_hash), str(projection.schema_version)
        if hasattr(projection, "hypothesis_ref") and hasattr(projection, "hypothesis_hash"):
            return str(projection.hypothesis_ref), str(projection.hypothesis_hash), str(projection.schema_version)
        if hasattr(projection, "manifest_ref") and hasattr(projection, "manifest_hash"):
            return str(projection.manifest_ref), str(projection.manifest_hash), str(projection.schema_version)
        if hasattr(projection, "binding_ref") and hasattr(projection, "binding_hash"):
            return str(projection.binding_ref), str(projection.binding_hash), str(projection.schema_version)
        if hasattr(projection, "resolution_ref") and hasattr(projection, "resolution_hash"):
            return str(projection.resolution_ref), str(projection.resolution_hash), str(projection.schema_version)
        if hasattr(projection, "window_ref") and hasattr(projection, "window_hash"):
            return str(projection.window_ref), str(projection.window_hash), str(projection.schema_version)
        if (
            getattr(projection, "schema_version", None)
            in {
                "semantier.institutional_adaptation_ingress.v1",
                "semantier.institutional_divergence_detection.v1",
            }
            and hasattr(projection, "artifact_ref")
            and hasattr(projection, "artifact_hash")
        ):
            return str(projection.artifact_ref), str(projection.artifact_hash), str(projection.schema_version)
        if hasattr(projection, "value_ref") and hasattr(projection, "value_hash"):
            return (
                str(projection.value_ref),
                str(projection.value_hash),
                str(getattr(projection, "schema_version", None) or getattr(projection, "VALUE_KIND", "derived_projection.v1")),
            )
        raise TypeError("unsupported derived projection receipt")

    def save_or_get_derived_projection(self, projection: Any, *, require_new: bool = False) -> Any:
        projection_ref, projection_hash, projection_kind = self._derived_projection_identity(projection)
        organization_id = str(projection.organization_id)
        workspace_id = str(projection.workspace_id)
        payload_json = self._json(projection.model_dump(mode="json"))
        recorded_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        with self._connect() as conn:
            inserted = conn.execute(
                """INSERT OR IGNORE INTO cognitive_derived_projections
                   (projection_ref,projection_hash,projection_kind,organization_id,workspace_id,payload_json,recorded_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (projection_ref, projection_hash, projection_kind, organization_id, workspace_id, payload_json, recorded_at),
            )
            if require_new and inserted.rowcount != 1:
                raise ValueError("DERIVED_PROJECTION_ALREADY_EXISTS")
            row = conn.execute(
                "SELECT * FROM cognitive_derived_projections WHERE projection_ref=? AND organization_id=? AND workspace_id=?",
                (projection_ref, organization_id, workspace_id),
            ).fetchone()
        if row is None:
            raise KeyError(projection_ref)
        if row["projection_hash"] != projection_hash or row["payload_json"] != payload_json:
            raise ValueError("DERIVED_PROJECTION_CONTENT_CONFLICT")
        return projection

    def get_derived_projection(self, projection_ref: str, *, organization_id: str, workspace_id: str,
                               contract_type: Any) -> Any:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM cognitive_derived_projections WHERE projection_ref=? AND organization_id=? AND workspace_id=?",
                (projection_ref, organization_id, workspace_id),
            ).fetchone()
        if row is None:
            raise KeyError(projection_ref)
        projection = contract_type.model_validate(json.loads(row["payload_json"]))
        resolved_ref, resolved_hash, resolved_kind = self._derived_projection_identity(projection)
        if (
            resolved_ref != row["projection_ref"]
            or resolved_hash != row["projection_hash"]
            or resolved_kind != row["projection_kind"]
        ):
            raise ValueError("DERIVED_PROJECTION_STORED_IDENTITY_MISMATCH")
        return projection

    def list_derived_projections(self, *, organization_id: str, workspace_id: str,
                                 projection_kind: str, contract_type: Any) -> tuple[Any, ...]:
        """Reload typed derived projections from one exact tenant/workspace scope."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM cognitive_derived_projections
                   WHERE organization_id=? AND workspace_id=? AND projection_kind=?
                   ORDER BY projection_ref""",
                (organization_id, workspace_id, projection_kind),
            ).fetchall()
        projections: list[Any] = []
        for row in rows:
            projection = contract_type.model_validate(json.loads(row["payload_json"]))
            resolved_ref, resolved_hash, resolved_kind = self._derived_projection_identity(projection)
            if (
                (str(projection.organization_id), str(projection.workspace_id)) != (organization_id, workspace_id)
                or resolved_ref != row["projection_ref"]
                or resolved_hash != row["projection_hash"]
                or resolved_kind != row["projection_kind"]
            ):
                raise ValueError("DERIVED_PROJECTION_STORED_IDENTITY_MISMATCH")
            projections.append(projection)
        return tuple(projections)

    def save_candidate(self, candidate: LearningCandidate) -> LearningCandidate:
        with self._connect() as conn:
            conn.execute("INSERT INTO cognitive_learning_candidates VALUES (?,?,?,?,?,?,?,?,?,?)",
                         (candidate.id, candidate.proposed_type.value, self._json(candidate.evidence_refs), candidate.hypothesis,
                          candidate.confidence, self._json(candidate.proposed_change), candidate.workspace_id,
                          candidate.organization_id, candidate.status.value, candidate.created_at))
        return candidate

    def get_candidate(self, candidate_id: str, *, workspace_id: str, organization_id: str) -> LearningCandidate:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM cognitive_learning_candidates WHERE id=? AND workspace_id=? AND organization_id=?",
                               (candidate_id, workspace_id, organization_id)).fetchone()
        if row is None:
            raise KeyError(candidate_id)
        return LearningCandidate(row["id"], ItemType(row["proposed_type"]), json.loads(row["evidence_refs_json"]),
                                 row["hypothesis"], row["confidence"], json.loads(row["proposed_change_json"]),
                                 row["workspace_id"], row["organization_id"], Status(row["status"]), row["created_at"])

    def save_or_get_candidate(self, candidate: LearningCandidate) -> LearningCandidate:
        with self._connect() as conn:
            conn.execute("INSERT OR IGNORE INTO cognitive_learning_candidates VALUES (?,?,?,?,?,?,?,?,?,?)",
                         (candidate.id, candidate.proposed_type.value, self._json(candidate.evidence_refs), candidate.hypothesis,
                          candidate.confidence, self._json(candidate.proposed_change), candidate.workspace_id,
                          candidate.organization_id, candidate.status.value, candidate.created_at))
        return self.get_candidate(candidate.id, workspace_id=candidate.workspace_id, organization_id=candidate.organization_id)

    def evidence_refs_exist(self, refs: list[str], *, workspace_id: str, organization_id: str) -> bool:
        if not refs:
            return False
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id FROM cognitive_evidence WHERE workspace_id=? AND organization_id=? AND id IN (%s)"
                % ",".join("?" for _ in refs),
                [workspace_id, organization_id, *refs],
            ).fetchall()
        return {row["id"] for row in rows} == set(refs)

    def update_candidate_status(self, candidate_id: str, status: Status, *, workspace_id: str, organization_id: str) -> None:
        with self._connect() as conn:
            changed = conn.execute("UPDATE cognitive_learning_candidates SET status=? WHERE id=? AND workspace_id=? AND organization_id=?",
                                   (status.value, candidate_id, workspace_id, organization_id)).rowcount
        if not changed:
            raise KeyError(candidate_id)

    @staticmethod
    def _step_payload(step: DecisionStep) -> tuple[Any, ...]:
        return (step.organization_id, step.workspace_id, step.session_id, step.turn_id, step.request_ref,
                step.decision_context_ref, step.checkpoint_ref, step.trajectory_ref, tuple(step.action_refs),
                tuple(step.tool_refs), step.outcome_ref, step.event_kind, step.logical_step_id,
                step.execution_attempt_ref, step.execution_attempt_hash, step.attempt_index, step.lane)

    def _step_from_row(self, row: sqlite3.Row) -> DecisionStep:
        return DecisionStep(row["step_id"], row["step_key"], row["organization_id"], row["workspace_id"],
                            row["session_id"], row["turn_id"], row["request_ref"], row["decision_context_ref"],
                            row["checkpoint_ref"], row["trajectory_ref"], tuple(json.loads(row["action_refs_json"])),
                            tuple(json.loads(row["tool_refs_json"])), row["outcome_ref"], row["event_kind"],
                            row["logical_step_id"], row["execution_attempt_ref"], row["execution_attempt_hash"],
                            int(row["attempt_index"] or 0), row["lane"] or "TASK", row["created_at"])

    def save_or_get_decision_step(self, step: DecisionStep) -> DecisionStep:
        try:
            with self._connect() as conn:
                conn.execute(
                    """INSERT INTO cognitive_decision_steps (
                       step_id,step_key,organization_id,workspace_id,session_id,turn_id,request_ref,
                       decision_context_ref,checkpoint_ref,trajectory_ref,action_refs_json,tool_refs_json,
                       outcome_ref,event_kind,logical_step_id,execution_attempt_ref,execution_attempt_hash,
                       attempt_index,lane,created_at
                       ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (step.step_id, step.step_key, step.organization_id, step.workspace_id, step.session_id,
                     step.turn_id, step.request_ref, step.decision_context_ref, step.checkpoint_ref,
                     step.trajectory_ref, self._json(list(step.action_refs)), self._json(list(step.tool_refs)),
                     step.outcome_ref, step.event_kind, step.logical_step_id, step.execution_attempt_ref,
                     step.execution_attempt_hash, step.attempt_index, step.lane, step.created_at),
                )
        except sqlite3.IntegrityError:
            with self._connect() as conn:
                row = conn.execute("SELECT * FROM cognitive_decision_steps WHERE step_key=?", (step.step_key,)).fetchone()
            if row is None:
                raise
            existing = self._step_from_row(row)
            if self._step_payload(existing) != self._step_payload(step):
                raise DecisionStepConflictError(step.step_key)
            return existing
        return step

    def get_decision_step(self, step_id: str, *, organization_id: str, workspace_id: str) -> DecisionStep:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM cognitive_decision_steps WHERE step_id=? AND organization_id=? AND workspace_id=?",
                               (step_id, organization_id, workspace_id)).fetchone()
        if row is None:
            raise KeyError(step_id)
        return self._step_from_row(row)

    def get_decision_step_by_key(self, step_key: str, *, organization_id: str, workspace_id: str) -> DecisionStep:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM cognitive_decision_steps WHERE step_key=? AND organization_id=? AND workspace_id=?",
                (step_key, organization_id, workspace_id),
            ).fetchone()
        if row is None:
            raise KeyError(step_key)
        return self._step_from_row(row)

    @staticmethod
    def _process_from_row(row: sqlite3.Row) -> LearningProcess:
        return LearningProcess(
            row["learning_evidence_key"], row["organization_id"], row["workspace_id"], row["step_id"],
            row["event_kind"], row["outcome_ref"], row["stage"], row["evidence_id"],
            row["proposal_schema_version"], row["proposal_json"], row["candidate_id"], row["terminal_status"],
            row["item_id"], row["last_error_code"], row["created_at"], row["updated_at"],
        )

    def get_learning_process(self, key: str, *, organization_id: str, workspace_id: str) -> LearningProcess:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM cognitive_learning_processes WHERE learning_evidence_key=? AND organization_id=? AND workspace_id=?",
                               (key, organization_id, workspace_id)).fetchone()
        if row is None:
            raise KeyError(key)
        return self._process_from_row(row)

    def reserve_learning_process(self, *, key: str, organization_id: str, workspace_id: str,
                                 step_id: str, event_kind: str, outcome_ref: str) -> LearningProcess:
        now = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        with self._connect() as conn:
            conn.execute("INSERT OR IGNORE INTO cognitive_learning_processes "
                         "(learning_evidence_key,organization_id,workspace_id,step_id,event_kind,outcome_ref,stage,created_at,updated_at) "
                         "VALUES (?,?,?,?,?,?, 'started', ?,?)",
                         (key, organization_id, workspace_id, step_id, event_kind, outcome_ref, now, now))
        return self.get_learning_process(key, organization_id=organization_id, workspace_id=workspace_id)

    def update_learning_process(self, key: str, *, organization_id: str, workspace_id: str,
                                stage: str, evidence_id: str | None = None, proposal_schema_version: int | None = None,
                                proposal_json: str | None = None, candidate_id: str | None = None,
                                terminal_status: str | None = None, item_id: str | None = None,
                                last_error_code: str | None = None) -> LearningProcess:
        current = self.get_learning_process(key, organization_id=organization_id, workspace_id=workspace_id)
        order = {"started": 0, "evidence_captured": 1, "proposal_captured": 2, "candidate_created": 3, "closed": 4}
        if order[stage] < order[current.stage]:
            return current
        now = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        with self._connect() as conn:
            conn.execute("""UPDATE cognitive_learning_processes SET stage=?, evidence_id=COALESCE(?, evidence_id),
                         proposal_schema_version=COALESCE(?, proposal_schema_version), proposal_json=COALESCE(?, proposal_json),
                         candidate_id=COALESCE(?, candidate_id), terminal_status=COALESCE(?, terminal_status),
                         item_id=COALESCE(?, item_id), last_error_code=COALESCE(?, last_error_code), updated_at=?
                         WHERE learning_evidence_key=? AND organization_id=? AND workspace_id=?""",
                         (stage, evidence_id, proposal_schema_version, proposal_json, candidate_id, terminal_status,
                          item_id, last_error_code, now, key, organization_id, workspace_id))
        return self.get_learning_process(key, organization_id=organization_id, workspace_id=workspace_id)

    def count_learning_rows(self, key: str, *, workspace_id: str, organization_id: str) -> dict[str, int]:
        with self._connect() as conn:
            row = conn.execute("SELECT evidence_id,candidate_id,item_id FROM cognitive_learning_processes WHERE learning_evidence_key=? AND workspace_id=? AND organization_id=?",
                               (key, workspace_id, organization_id)).fetchone()
        if row is None:
            return {"process": 0, "evidence": 0, "candidate": 0, "item": 0}
        return {"process": 1, "evidence": int(row["evidence_id"] is not None),
                "candidate": int(row["candidate_id"] is not None), "item": int(row["item_id"] is not None)}
