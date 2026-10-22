"""Graph schema discovery Brief and decision persistence."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from storage.hosted_postgres import enabled,connect_authority,mapped_store,require_tables,sql_query,begin_write,integrity_errors
from eos.artifact_hashing import canonical_json, content_hash
from services.canonical_skill_registry import (
    resolve_historical_ref,
    select_for_new_run,
)


BRIEF_SCHEMA_VERSION = "graph_schema_discovery_brief.v1"
DECISION_SCHEMA_VERSION = "semantic_discovery_decision.v1"
LLM_PROPOSAL_EVIDENCE_SCHEMA_VERSION = "llm_proposal_evidence.v1"
RUN_SCHEMA_VERSION = "graph_schema_discovery_run.v1"
SAMPLE_RESULT_SCHEMA_VERSION = "corpus_graph_schema_sample_record.v1"
FEEDBACK_RETEST_SCHEMA_VERSION = "corpus_graph_schema_feedback_retest.v1"
DISCOVERY_LIFECYCLE = ("DISCOVERED", "PROPOSED", "HUMAN_REVIEWED", "ACCEPTED")
HumanReviewState = Literal["unreviewed", "accepted", "edited", "rejected", "unresolved"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_utc_iso(value: str, *, label: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} must be timezone-aware UTC ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"{label} must be timezone-aware UTC ISO-8601")
    return parsed.astimezone(timezone.utc).isoformat()


def _require_safe_artifact_id(value: str, *, label: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a safe artifact id")
    path = Path(value)
    if (
        not value.strip()
        or path.is_absolute()
        or "/" in value
        or "\\" in value
        or ".." in path.parts
    ):
        raise ValueError(f"{label} must be a safe artifact id")


def _dumps(value: Any) -> str:
    return canonical_json(value if value is not None else {})


def _loads(value: str) -> Any:
    return json.loads(value) if value else None


@dataclass(frozen=True)
class GraphSchemaDiscoveryRun:
    schema_version: str
    run_id: str
    experiment_id: str
    workspace_id: str
    lineage_id: str
    source_ref: str
    source_hash: str
    skill_ref: str
    lifecycle_state: str
    created_by: str
    created_at: str

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class GraphSchemaDiscoveryStore:
    """SQLite-backed governed record for W2 discovery contracts."""

    _process_locks: dict[Path, threading.RLock] = {}
    _process_locks_guard = threading.Lock()

    def __init__(self, workspace_root: str | Path, *, workspace_id: str) -> None:
        _require_safe_artifact_id(workspace_id, label="workspace_id")
        self.workspace_root = Path(workspace_root).resolve()
        self.workspace_id = workspace_id
        self.root = self.workspace_root / ".semantier" / "graph-schema-discovery"
        if not enabled():self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "discovery.sqlite3"
        self._initialize()

    @property
    def _lock(self) -> threading.RLock:
        with self._process_locks_guard:
            return self._process_locks.setdefault(self.db_path, threading.RLock())

    def _connect(self) -> sqlite3.Connection:
        if enabled():
            binding=mapped_store('standalone',self.db_path)
            if binding.get('workspace_id',self.workspace_id)!=self.workspace_id:
                raise ValueError('foreign workspace auxiliary binding')
            return connect_authority('standalone',self.db_path)
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        if enabled():
            with self._connect() as connection:require_tables(connection,'graph_schema_discovery_runs','semantic_discovery_decisions','graph_schema_discovery_briefs','llm_proposal_evidence','corpus_graph_schema_sample_results','corpus_graph_schema_feedback_retests','lineage_retirement_events')
            return
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS graph_schema_discovery_runs (
                    run_id TEXT PRIMARY KEY,
                    schema_version TEXT NOT NULL,
                    experiment_id TEXT NOT NULL,
                    workspace_id TEXT NOT NULL,
                    source_ref TEXT NOT NULL,
                    source_hash TEXT NOT NULL,
                    skill_ref TEXT NOT NULL,
                    lifecycle_state TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            run_columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(graph_schema_discovery_runs)"
                ).fetchall()
            }
            if "lineage_id" not in run_columns:
                connection.execute(
                    "ALTER TABLE graph_schema_discovery_runs ADD COLUMN lineage_id TEXT"
                )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_graph_schema_discovery_runs_lineage_id "
                "ON graph_schema_discovery_runs(lineage_id)"
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS semantic_discovery_decisions (
                    decision_id TEXT PRIMARY KEY,
                    schema_version TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    workspace_id TEXT NOT NULL,
                    discovery_type TEXT NOT NULL,
                    subject_json TEXT NOT NULL,
                    proposed_semantics_json TEXT NOT NULL,
                    rationale_json TEXT NOT NULL,
                    alternatives_considered_json TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    semantica_signals_json TEXT NOT NULL,
                    human_review_json TEXT NOT NULL,
                    provenance_json TEXT NOT NULL,
                    learning_eligibility_json TEXT NOT NULL,
                    decision_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES graph_schema_discovery_runs(run_id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS graph_schema_discovery_briefs (
                    brief_id TEXT PRIMARY KEY,
                    schema_version TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    experiment_id TEXT NOT NULL,
                    workspace_id TEXT NOT NULL,
                    source_ref TEXT NOT NULL,
                    source_hash TEXT NOT NULL,
                    skill_ref TEXT NOT NULL,
                    semantic_roles_json TEXT NOT NULL,
                    semantic_dimensions_json TEXT NOT NULL,
                    hierarchy_edges_json TEXT NOT NULL,
                    inheritance_policies_json TEXT NOT NULL,
                    competency_questions_json TEXT NOT NULL,
                    semantic_discovery_decision_refs_json TEXT NOT NULL,
                    brief_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES graph_schema_discovery_runs(run_id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_proposal_evidence (
                    evidence_id TEXT PRIMARY KEY,
                    schema_version TEXT NOT NULL,
                    run_id TEXT,
                    workspace_id TEXT NOT NULL,
                    semantica_commit TEXT NOT NULL,
                    skill_major_version TEXT NOT NULL,
                    provider_contract_version TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    status TEXT NOT NULL,
                    proposal_ref TEXT,
                    proposal_hash TEXT,
                    diagnostics_json TEXT NOT NULL,
                    observed_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS corpus_graph_schema_sample_results (
                    sample_id TEXT PRIMARY KEY,
                    schema_version TEXT NOT NULL,
                    workspace_id TEXT NOT NULL,
                    decision_id TEXT NOT NULL,
                    decision_hash TEXT NOT NULL,
                    authoritative_candidate_id TEXT NOT NULL,
                    authoritative_candidate_hash TEXT NOT NULL,
                    source_ref TEXT NOT NULL,
                    source_hash TEXT NOT NULL,
                    profile_id TEXT NOT NULL,
                    engine_ref TEXT NOT NULL,
                    execution_contract_hash TEXT NOT NULL,
                    result_hash TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(decision_id) REFERENCES semantic_discovery_decisions(decision_id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS corpus_graph_schema_feedback_retests (
                    comparison_id TEXT PRIMARY KEY,
                    schema_version TEXT NOT NULL,
                    workspace_id TEXT NOT NULL,
                    baseline_decision_id TEXT NOT NULL,
                    baseline_decision_hash TEXT NOT NULL,
                    baseline_sample_id TEXT NOT NULL,
                    correction_decision_id TEXT NOT NULL,
                    correction_decision_hash TEXT NOT NULL,
                    correction_sample_id TEXT NOT NULL,
                    source_ref TEXT NOT NULL,
                    source_hash TEXT NOT NULL,
                    profile_id TEXT NOT NULL,
                    original_candidate_hash TEXT NOT NULL,
                    actor_id TEXT NOT NULL,
                    requested_relation_type TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    comparison_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(baseline_decision_id) REFERENCES semantic_discovery_decisions(decision_id),
                    FOREIGN KEY(correction_decision_id) REFERENCES semantic_discovery_decisions(decision_id),
                    FOREIGN KEY(baseline_sample_id) REFERENCES corpus_graph_schema_sample_results(sample_id),
                    FOREIGN KEY(correction_sample_id) REFERENCES corpus_graph_schema_sample_results(sample_id)
                )
                """
            )
            for table in (
                "graph_schema_discovery_runs",
                "semantic_discovery_decisions",
                "graph_schema_discovery_briefs",
                "llm_proposal_evidence",
            ):
                columns = {
                    row["name"]
                    for row in connection.execute(
                        f"PRAGMA table_info({table})"
                    ).fetchall()
                }
                if "lineage_id" not in columns:
                    connection.execute(
                        f"ALTER TABLE {table} ADD COLUMN lineage_id TEXT"
                    )
                connection.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_{table}_lineage_id "
                    f"ON {table}(lineage_id)"
                )
                columns = {
                    row["name"]
                    for row in connection.execute(
                        f"PRAGMA table_info({table})"
                    ).fetchall()
                }
                if "retirement_state" not in columns:
                    connection.execute(
                        f"ALTER TABLE {table} ADD COLUMN retirement_state "
                        "TEXT NOT NULL DEFAULT 'active'"
                    )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS lineage_retirement_events (
                    event_id TEXT PRIMARY KEY,
                    operation_id TEXT NOT NULL,
                    lineage_id TEXT NOT NULL,
                    actor_id TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    previous_state TEXT NOT NULL,
                    next_state TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(operation_id, lineage_id, next_state)
                )"""
            )

    def persist_feedback_retest(
        self,
        *,
        baseline_decision_id: str,
        correction_decision_id: str,
        baseline_sample_id: str,
        correction_sample_id: str,
        actor_id: str,
        requested_relation_type: str,
    ) -> dict[str, Any]:
        """Append one bounded before/after correction comparison."""
        self._require(
            actor_id=actor_id, requested_relation_type=requested_relation_type
        )
        if baseline_decision_id == correction_decision_id:
            raise ValueError("feedback retest decisions must be distinct")
        baseline = self.get_decision(baseline_decision_id)
        correction = self.get_decision(correction_decision_id)
        if (baseline.get("human_review") or {}).get("state") != "accepted":
            raise ValueError("feedback retest baseline must be accepted")
        correction_review = correction.get("human_review") or {}
        if correction_review.get("state") != "edited":
            raise ValueError("feedback retest correction must be edited")
        baseline_original = baseline["proposed_semantics"]["machine_readable"][
            "candidate"
        ]
        correction_original = correction["proposed_semantics"]["machine_readable"][
            "candidate"
        ]
        edited = correction_review.get("edited_payload")
        if not isinstance(edited, dict):
            raise ValueError("feedback retest correction has no edited projection")
        baseline_run = self.get_run(baseline["run_id"])
        correction_run = self.get_run(correction["run_id"])
        baseline_profile = baseline_original.get("source_structure_profile") or {}
        correction_profile = correction_original.get("source_structure_profile") or {}
        shared_pins = (
            baseline_run.get("source_ref"),
            baseline_run.get("source_hash"),
            baseline_profile.get("profile_id"),
            baseline_original.get("candidate_hash"),
        )
        if shared_pins != (
            correction_run.get("source_ref"),
            correction_run.get("source_hash"),
            correction_profile.get("profile_id"),
            correction_original.get("candidate_hash"),
        ):
            raise ValueError("feedback retest source pins do not match")
        for run, profile in (
            (baseline_run, baseline_profile),
            (correction_run, correction_profile),
        ):
            if run.get("source_ref") != profile.get("source_ref") or run.get(
                "source_hash"
            ) != profile.get("document_hash"):
                raise ValueError(
                    "feedback retest source pins do not match governed candidates"
                )
        before = self.get_sample_result(baseline_sample_id)
        after = self.get_sample_result(correction_sample_id)
        if before.get("decision_id") != baseline_decision_id:
            raise ValueError("baseline sample pin does not match baseline decision")
        if after.get("decision_id") != correction_decision_id:
            raise ValueError("correction sample pin does not match correction decision")
        if (
            before.get("decision_hash") != baseline.get("decision_hash")
            or after.get("decision_hash") != correction.get("decision_hash")
            or before.get("source_ref") != shared_pins[0]
            or after.get("source_ref") != shared_pins[0]
            or before.get("source_hash") != shared_pins[1]
            or after.get("source_hash") != shared_pins[1]
            or before.get("profile_id") != shared_pins[2]
            or after.get("profile_id") != shared_pins[2]
            or before.get("authoritative_candidate_hash") != shared_pins[3]
        ):
            raise ValueError(
                "feedback retest sample pins do not match governed decisions"
            )
        if after.get("authoritative_candidate_id") != edited.get(
            "candidate_id"
        ) or after.get("authoritative_candidate_hash") != edited.get("candidate_hash"):
            raise ValueError("after result is not bound to the edited projection")

        def observations(
            sample: dict[str, Any],
        ) -> tuple[list[dict[str, Any]], set[str]]:
            events = sample["result"].get("events") or []
            relations = [
                relation
                for event in events
                for relation in (event.get("relations") or [])
                if isinstance(relation, dict)
            ]
            evidence = {
                str(ref)
                for event in events
                for ref in (event.get("evidence_refs") or [])
            }
            return relations, evidence

        before_relations, before_evidence = observations(before)
        after_relations, after_evidence = observations(after)
        before_events = before["result"].get("events") or []
        after_events = after["result"].get("events") or []
        metrics = {
            "requested_relation_present": {
                "before": any(
                    item.get("type_id") == requested_relation_type
                    for item in before_relations
                ),
                "after": any(
                    item.get("type_id") == requested_relation_type
                    for item in after_relations
                ),
            },
            "retained_sample_count": len(
                {item.get("event_id") for item in before_events}
                & {item.get("event_id") for item in after_events}
            ),
            "retained_evidence_count": len(before_evidence & after_evidence),
            "before_sample_hash": before["result_hash"],
            "after_sample_hash": after["result_hash"],
        }
        if metrics["requested_relation_present"] != {"before": False, "after": True}:
            raise ValueError(
                "feedback retest must prove requested relation absent before and present after"
            )
        evidence = {
            "baseline_evidence_refs": sorted(before_evidence),
            "correction_evidence_refs": sorted(after_evidence),
        }
        immutable = {
            "schema_version": FEEDBACK_RETEST_SCHEMA_VERSION,
            "workspace_id": self.workspace_id,
            "baseline_decision_id": baseline_decision_id,
            "baseline_decision_hash": baseline["decision_hash"],
            "baseline_sample_id": baseline_sample_id,
            "correction_decision_id": correction_decision_id,
            "correction_decision_hash": correction["decision_hash"],
            "correction_sample_id": correction_sample_id,
            "source_ref": shared_pins[0],
            "source_hash": shared_pins[1],
            "profile_id": shared_pins[2],
            "original_candidate_hash": shared_pins[3],
            "actor_id": actor_id,
            "requested_relation_type": requested_relation_type,
            "metrics": metrics,
            "evidence": evidence,
        }
        comparison_hash = f"sha256:{content_hash(immutable)}"
        comparison_id = (
            f"cgsfr_{content_hash({'comparison_hash': comparison_hash})[:24]}"
        )
        record = {
            "comparison_id": comparison_id,
            **immutable,
            "comparison_hash": comparison_hash,
            "created_at": _utc_now(),
        }
        with self._lock, self._connect() as connection:
            if enabled():begin_write(connection)
            connection.execute(
                sql_query("""INSERT OR IGNORE INTO corpus_graph_schema_feedback_retests VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )""",'INSERT INTO corpus_graph_schema_feedback_retests VALUES (\n                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s\n                ) ON CONFLICT DO NOTHING'),
                (
                    comparison_id,
                    record["schema_version"],
                    self.workspace_id,
                    baseline_decision_id,
                    baseline["decision_hash"],
                    baseline_sample_id,
                    correction_decision_id,
                    correction["decision_hash"],
                    correction_sample_id,
                    shared_pins[0],
                    shared_pins[1],
                    shared_pins[2],
                    shared_pins[3],
                    actor_id,
                    requested_relation_type,
                    _dumps(metrics),
                    _dumps(evidence),
                    comparison_hash,
                    record["created_at"],
                ),
            )
        return self.get_feedback_retest(comparison_id)

    def get_feedback_retest(self, comparison_id: str) -> dict[str, Any]:
        _require_safe_artifact_id(comparison_id, label="comparison_id")
        with self._connect() as connection:
            row = connection.execute(
                sql_query("SELECT * FROM corpus_graph_schema_feedback_retests WHERE comparison_id = ? AND workspace_id = ?",'SELECT * FROM corpus_graph_schema_feedback_retests WHERE comparison_id = %s AND workspace_id = %s'),
                (comparison_id, self.workspace_id),
            ).fetchone()
        if row is None:
            raise KeyError(
                f"corpus graph schema feedback retest not found: {comparison_id}"
            )
        record = dict(row)
        record["metrics"] = _loads(record.pop("metrics_json"))
        record["evidence"] = _loads(record.pop("evidence_json"))
        _require_utc_iso(record["created_at"], label="created_at")
        immutable = {
            key: record[key]
            for key in (
                "schema_version",
                "workspace_id",
                "baseline_decision_id",
                "baseline_decision_hash",
                "baseline_sample_id",
                "correction_decision_id",
                "correction_decision_hash",
                "correction_sample_id",
                "source_ref",
                "source_hash",
                "profile_id",
                "original_candidate_hash",
                "actor_id",
                "requested_relation_type",
                "metrics",
                "evidence",
            )
        }
        if record["comparison_hash"] != f"sha256:{content_hash(immutable)}":
            raise ValueError("persisted feedback retest comparison is tampered")
        if (
            record["comparison_id"]
            != f"cgsfr_{content_hash({'comparison_hash': record['comparison_hash']})[:24]}"
        ):
            raise ValueError("persisted feedback retest identity is tampered")
        return record

    def persist_sample_result(
        self,
        *,
        decision_id: str,
        decision_hash: str,
        authoritative_candidate_id: str,
        authoritative_candidate_hash: str,
        source_ref: str,
        source_hash: str,
        profile_id: str,
        engine_ref: str,
        execution_contract_hash: str,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """Append an immutable, fully pinned sample result (idempotently)."""
        decision = self.get_decision(decision_id)
        run = self.get_run(decision["run_id"])
        if decision.get("decision_hash") != decision_hash:
            raise ValueError("sample decision hash does not match governed decision")
        if run.get("source_ref") != source_ref or run.get("source_hash") != source_hash:
            raise ValueError("sample source pins do not match governed run")
        original = decision["proposed_semantics"]["machine_readable"]["candidate"]
        review = decision.get("human_review") or {}
        if review.get("state") not in {"accepted", "edited"}:
            raise ValueError("sample execution requires accepted or edited review")
        authoritative = (
            review.get("edited_payload") if review["state"] == "edited" else original
        )
        if not isinstance(authoritative, dict):
            raise ValueError("authoritative sample candidate is missing")
        if (
            authoritative.get("candidate_id") != authoritative_candidate_id
            or authoritative.get("candidate_hash") != authoritative_candidate_hash
        ):
            raise ValueError("sample candidate pins do not match authoritative review")
        profile = original.get("source_structure_profile") or {}
        if profile.get("profile_id") != profile_id:
            raise ValueError("sample profile pin does not match governed candidate")
        if (
            not isinstance(result, dict)
            or result.get("result_hash")
            != f"sha256:{content_hash({key: value for key, value in result.items() if key != 'result_hash'})}"
        ):
            raise ValueError("sample result hash is missing or inconsistent")
        pins = {
            "workspace_id": self.workspace_id,
            "decision_id": decision_id,
            "decision_hash": decision_hash,
            "authoritative_candidate_id": authoritative_candidate_id,
            "authoritative_candidate_hash": authoritative_candidate_hash,
            "source_ref": source_ref,
            "source_hash": source_hash,
            "profile_id": profile_id,
            "engine_ref": engine_ref,
            "execution_contract_hash": execution_contract_hash,
            "result_hash": result["result_hash"],
        }
        sample_id = f"cgss_{content_hash(pins)[:24]}"
        record = {
            "sample_id": sample_id,
            "schema_version": SAMPLE_RESULT_SCHEMA_VERSION,
            **pins,
            "result": json.loads(canonical_json(result)),
            "created_at": _utc_now(),
        }
        with self._lock, self._connect() as connection:
            if enabled():begin_write(connection)
            connection.execute(
                sql_query("""
                INSERT OR IGNORE INTO corpus_graph_schema_sample_results VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,'\n                INSERT INTO corpus_graph_schema_sample_results VALUES (\n                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s\n                )\n                 ON CONFLICT DO NOTHING'),
                (
                    sample_id,
                    record["schema_version"],
                    self.workspace_id,
                    decision_id,
                    decision_hash,
                    authoritative_candidate_id,
                    authoritative_candidate_hash,
                    source_ref,
                    source_hash,
                    profile_id,
                    engine_ref,
                    execution_contract_hash,
                    result["result_hash"],
                    _dumps(record["result"]),
                    record["created_at"],
                ),
            )
        return self.get_sample_result(sample_id)

    def get_sample_result(self, sample_id: str) -> dict[str, Any]:
        _require_safe_artifact_id(sample_id, label="sample_id")
        with self._connect() as connection:
            row = connection.execute(
                sql_query("SELECT * FROM corpus_graph_schema_sample_results WHERE sample_id = ? AND workspace_id = ?",'SELECT * FROM corpus_graph_schema_sample_results WHERE sample_id = %s AND workspace_id = %s'),
                (sample_id, self.workspace_id),
            ).fetchone()
        if row is None:
            raise KeyError(f"corpus graph schema sample not found: {sample_id}")
        payload = dict(row)
        payload["result"] = _loads(payload.pop("result_json"))
        _require_utc_iso(payload["created_at"], label="created_at")
        result_body = {
            key: value
            for key, value in payload["result"].items()
            if key != "result_hash"
        }
        if (
            payload["result_hash"] != payload["result"].get("result_hash")
            or payload["result_hash"] != f"sha256:{content_hash(result_body)}"
        ):
            raise ValueError("persisted sample result is tampered")
        pins = {
            key: payload[key]
            for key in (
                "workspace_id",
                "decision_id",
                "decision_hash",
                "authoritative_candidate_id",
                "authoritative_candidate_hash",
                "source_ref",
                "source_hash",
                "profile_id",
                "engine_ref",
                "execution_contract_hash",
                "result_hash",
            )
        }
        if payload["sample_id"] != f"cgss_{content_hash(pins)[:24]}":
            raise ValueError("persisted sample identity is tampered")
        return payload

    def create_run(
        self,
        *,
        lineage_id: str,
        experiment_id: str,
        source_ref: str,
        source_hash: str,
        skill_ref: str,
        created_by: str,
        run_id: str | None = None,
    ) -> GraphSchemaDiscoveryRun:
        selected = select_for_new_run(skill_ref=skill_ref)
        if selected is None:
            raise ValueError("new run requires a selectable canonical skill")
        return self._persist_run(
            lineage_id=lineage_id,
            experiment_id=experiment_id,
            source_ref=source_ref,
            source_hash=source_hash,
            skill_ref=selected.skill_ref,
            created_by=created_by,
            run_id=run_id,
        )

    def import_historical_run(
        self,
        *,
        lineage_id: str,
        experiment_id: str,
        source_ref: str,
        source_hash: str,
        skill_ref: str,
        imported_by: str,
    ) -> GraphSchemaDiscoveryRun:
        historical = resolve_historical_ref(skill_ref)
        return self._persist_run(
            lineage_id=lineage_id,
            experiment_id=experiment_id,
            source_ref=source_ref,
            source_hash=source_hash,
            skill_ref=historical.skill_ref,
            created_by=imported_by,
        )

    def _persist_run(
        self,
        *,
        lineage_id: str,
        experiment_id: str,
        source_ref: str,
        source_hash: str,
        skill_ref: str,
        created_by: str,
        run_id: str | None = None,
    ) -> GraphSchemaDiscoveryRun:
        if not isinstance(lineage_id, str) or not lineage_id.strip():
            raise ValueError("lineage_id is required")
        self._require(
            lineage_id=lineage_id,
            experiment_id=experiment_id,
            source_ref=source_ref,
            source_hash=source_hash,
            skill_ref=skill_ref,
            created_by=created_by,
        )
        if run_id is not None:
            _require_safe_artifact_id(run_id, label="run_id")
            if not run_id.startswith("gsdr_"):
                raise ValueError("run_id must be a graph schema discovery id")
        run = GraphSchemaDiscoveryRun(
            schema_version=RUN_SCHEMA_VERSION,
            run_id=run_id or f"gsdr_{uuid.uuid4().hex}",
            experiment_id=experiment_id,
            workspace_id=self.workspace_id,
            lineage_id=lineage_id,
            source_ref=source_ref,
            source_hash=source_hash,
            skill_ref=skill_ref,
            lifecycle_state="DISCOVERED",
            created_by=created_by,
            created_at=_utc_now(),
        )
        with self._lock, self._connect() as connection:
            if enabled():begin_write(connection)
            statement = (
                sql_query("""INSERT OR IGNORE INTO graph_schema_discovery_runs (
                    run_id, schema_version, experiment_id, workspace_id, source_ref,
                    source_hash, skill_ref, lifecycle_state, created_by, created_at,
                    lineage_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",'INSERT INTO graph_schema_discovery_runs (\n                    run_id, schema_version, experiment_id, workspace_id, source_ref,\n                    source_hash, skill_ref, lifecycle_state, created_by, created_at,\n                    lineage_id\n                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING')
                if run_id is not None
                else sql_query("""INSERT INTO graph_schema_discovery_runs (
                    run_id, schema_version, experiment_id, workspace_id, source_ref,
                    source_hash, skill_ref, lifecycle_state, created_by, created_at,
                    lineage_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",'INSERT INTO graph_schema_discovery_runs (\n                    run_id, schema_version, experiment_id, workspace_id, source_ref,\n                    source_hash, skill_ref, lifecycle_state, created_by, created_at,\n                    lineage_id\n                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)')
            )
            connection.execute(
                statement,
                (
                    run.run_id,
                    run.schema_version,
                    run.experiment_id,
                    run.workspace_id,
                    run.source_ref,
                    run.source_hash,
                    run.skill_ref,
                    run.lifecycle_state,
                    run.created_by,
                    run.created_at,
                    run.lineage_id,
                ),
            )
            if run_id is not None:
                row = connection.execute(
                    sql_query("SELECT * FROM graph_schema_discovery_runs "
                    "WHERE run_id = ? AND workspace_id = ?",'SELECT * FROM graph_schema_discovery_runs WHERE run_id = %s AND workspace_id = %s'),
                    (run_id, self.workspace_id),
                ).fetchone()
                if row is None:
                    raise ValueError("GRAPH_SCHEMA_DISCOVERY_RUN_PERSISTENCE_FAILED")
                stored = dict(row)
                expected = run.as_dict()
                immutable_fields = {
                    "schema_version",
                    "run_id",
                    "experiment_id",
                    "workspace_id",
                    "lineage_id",
                    "source_ref",
                    "source_hash",
                    "skill_ref",
                }
                if (
                    stored.get("retirement_state", "active") != "active"
                    or any(
                        stored[field] != expected[field]
                        for field in immutable_fields
                    )
                ):
                    raise ValueError("GRAPH_SCHEMA_DISCOVERY_RUN_ID_CONFLICT")
                return GraphSchemaDiscoveryRun(
                    **{field: stored[field] for field in expected}
                )
        return run

    def create_decision(
        self,
        *,
        run_id: str,
        discovery_type: str,
        subject: dict[str, Any],
        proposed_semantics: dict[str, Any],
        rationale: dict[str, Any],
        alternatives_considered: list[dict[str, Any]] | None = None,
        confidence: float,
        semantica_signals: dict[str, Any] | None = None,
        provenance: dict[str, Any],
        decision_id: str | None = None,
        idempotent: bool = False,
    ) -> dict[str, Any]:
        run = self.get_run(run_id)
        if idempotent and decision_id is None:
            raise ValueError("idempotent decision creation requires decision_id")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        evidence_refs = rationale.get("evidence_refs") or []
        structural_refs = rationale.get("structural_evidence_refs") or []
        if not evidence_refs and not structural_refs:
            raise ValueError(
                "semantic discovery decision rationale must cite evidence refs"
            )
        decision = {
            "schema_version": DECISION_SCHEMA_VERSION,
            "decision_id": decision_id or f"gsdd_{uuid.uuid4().hex}",
            "run_id": run_id,
            "lineage_id": run["lineage_id"],
            "discovery_type": discovery_type,
            "subject": subject,
            "proposed_semantics": proposed_semantics,
            "rationale": rationale,
            "alternatives_considered": alternatives_considered or [],
            "confidence": confidence,
            "semantica_signals": semantica_signals
            or {
                "llm_proposal_ref": None,
                "deterministic_proposal_ref": None,
                "graph_validation_refs": [],
            },
            "human_review": {
                "state": "unreviewed",
                "actor_id": None,
                "reviewed_at": None,
                "note": None,
            },
            "provenance": {
                **provenance,
                "source_ref": provenance.get("source_ref", run["source_ref"]),
                "source_hash": provenance.get("source_hash", run["source_hash"]),
                "created_at": provenance.get("created_at", _utc_now()),
            },
            "learning_eligibility": {
                "status": "excluded",
                "reason": "not human reviewed",
            },
            "created_at": _utc_now(),
        }
        decision["decision_hash"] = f"sha256:{content_hash(decision)}"
        with self._lock, self._connect() as connection:
            if enabled():begin_write(connection)
            statement = (
                sql_query("""
                INSERT OR IGNORE INTO semantic_discovery_decisions (
                    decision_id, schema_version, run_id, workspace_id,
                    discovery_type, subject_json, proposed_semantics_json,
                    rationale_json, alternatives_considered_json, confidence,
                    semantica_signals_json, human_review_json, provenance_json,
                    learning_eligibility_json, decision_hash, created_at, lineage_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,'\n                INSERT INTO semantic_discovery_decisions (\n                    decision_id, schema_version, run_id, workspace_id,\n                    discovery_type, subject_json, proposed_semantics_json,\n                    rationale_json, alternatives_considered_json, confidence,\n                    semantica_signals_json, human_review_json, provenance_json,\n                    learning_eligibility_json, decision_hash, created_at, lineage_id\n                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)\n                 ON CONFLICT DO NOTHING')
                if idempotent
                else sql_query("""
                INSERT INTO semantic_discovery_decisions (
                    decision_id, schema_version, run_id, workspace_id,
                    discovery_type, subject_json, proposed_semantics_json,
                    rationale_json, alternatives_considered_json, confidence,
                    semantica_signals_json, human_review_json, provenance_json,
                    learning_eligibility_json, decision_hash, created_at, lineage_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,'\n                INSERT INTO semantic_discovery_decisions (\n                    decision_id, schema_version, run_id, workspace_id,\n                    discovery_type, subject_json, proposed_semantics_json,\n                    rationale_json, alternatives_considered_json, confidence,\n                    semantica_signals_json, human_review_json, provenance_json,\n                    learning_eligibility_json, decision_hash, created_at, lineage_id\n                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)\n                ')
            )
            connection.execute(
                statement,
                (
                    decision["decision_id"],
                    decision["schema_version"],
                    run_id,
                    self.workspace_id,
                    discovery_type,
                    _dumps(subject),
                    _dumps(proposed_semantics),
                    _dumps(rationale),
                    _dumps(decision["alternatives_considered"]),
                    confidence,
                    _dumps(decision["semantica_signals"]),
                    _dumps(decision["human_review"]),
                    _dumps(decision["provenance"]),
                    _dumps(decision["learning_eligibility"]),
                    decision["decision_hash"],
                    decision["created_at"],
                    decision["lineage_id"],
                ),
            )
            connection.execute(
                sql_query("""
                UPDATE graph_schema_discovery_runs
                SET lifecycle_state = 'PROPOSED'
                WHERE run_id = ? AND workspace_id = ?
                ""","\n                UPDATE graph_schema_discovery_runs\n                SET lifecycle_state = 'PROPOSED'\n                WHERE run_id = %s AND workspace_id = %s\n                "),
                (run_id, self.workspace_id),
            )
            if idempotent:
                row = connection.execute(
                    sql_query("SELECT * FROM semantic_discovery_decisions "
                    "WHERE decision_id = ? AND workspace_id = ?",'SELECT * FROM semantic_discovery_decisions WHERE decision_id = %s AND workspace_id = %s'),
                    (decision_id, self.workspace_id),
                ).fetchone()
                if row is None:
                    raise ValueError("SEMANTIC_DISCOVERY_DECISION_PERSISTENCE_FAILED")
                stored = self._hydrate_decision(row)
                immutable_fields = {
                    "schema_version",
                    "decision_id",
                    "run_id",
                    "lineage_id",
                    "discovery_type",
                    "subject",
                    "proposed_semantics",
                    "rationale",
                    "alternatives_considered",
                    "confidence",
                    "semantica_signals",
                }
                if (
                    stored.get("retirement_state", "active") != "active"
                    or any(
                        stored[field] != decision[field]
                        for field in immutable_fields
                    )
                ):
                    raise ValueError("SEMANTIC_DISCOVERY_DECISION_ID_CONFLICT")
                stored_provenance = dict(stored["provenance"])
                expected_provenance = dict(decision["provenance"])
                stored_provenance.pop("created_at", None)
                expected_provenance.pop("created_at", None)
                if stored_provenance != expected_provenance:
                    raise ValueError("SEMANTIC_DISCOVERY_DECISION_ID_CONFLICT")
                return stored
        return decision

    def record_human_review(
        self,
        decision_id: str,
        *,
        state: HumanReviewState,
        actor_id: str,
        note: str | None = None,
        edited_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if state not in {"accepted", "edited", "rejected", "unresolved"}:
            raise ValueError(
                "human review state must be accepted, edited, rejected, or unresolved"
            )
        if not isinstance(actor_id, str) or not actor_id.strip():
            raise ValueError("actor_id is required")
        if note is not None and not isinstance(note, str):
            raise ValueError("human review note must be a string")
        if state == "edited" and not isinstance(edited_payload, dict):
            raise ValueError("edited review requires an edited payload object")
        if state != "edited" and edited_payload is not None:
            raise ValueError("edited payload is only valid for edited review")
        decision = self.get_decision(decision_id)
        review = {
            "state": state,
            "actor_id": actor_id,
            "reviewed_at": _utc_now(),
            "note": note,
        }
        if edited_payload is not None:
            bound_payload = json.loads(canonical_json(edited_payload))
            payload_hash = f"sha256:{content_hash(bound_payload)}"
            review.update(
                {
                    "edited_payload": bound_payload,
                    "edited_payload_hash": payload_hash,
                    "edited_payload_binding": f"sha256:{content_hash({'decision_hash': decision['decision_hash'], 'edited_payload_hash': payload_hash})}",
                }
            )
        learning = {
            "status": "positive_example"
            if state in {"accepted", "edited"}
            else "negative_example"
            if state == "rejected"
            else "excluded",
            "reason": f"human review state: {state}",
        }
        with self._lock, self._connect() as connection:
            if enabled():begin_write(connection)
            connection.execute(
                sql_query("""
                UPDATE semantic_discovery_decisions
                SET human_review_json = ?, learning_eligibility_json = ?
                WHERE decision_id = ? AND workspace_id = ?
                """,'\n                UPDATE semantic_discovery_decisions\n                SET human_review_json = %s, learning_eligibility_json = %s\n                WHERE decision_id = %s AND workspace_id = %s\n                '),
                (_dumps(review), _dumps(learning), decision_id, self.workspace_id),
            )
            connection.execute(
                sql_query("""
                UPDATE graph_schema_discovery_runs
                SET lifecycle_state = ?
                WHERE run_id = ? AND workspace_id = ?
                """,'\n                UPDATE graph_schema_discovery_runs\n                SET lifecycle_state = %s\n                WHERE run_id = %s AND workspace_id = %s\n                '),
                (
                    "ACCEPTED" if state in {"accepted", "edited"} else "HUMAN_REVIEWED",
                    decision["run_id"],
                    self.workspace_id,
                ),
            )
        return self.get_decision(decision_id)

    def create_brief(
        self,
        *,
        run_id: str,
        semantic_roles: list[dict[str, Any]],
        semantic_dimensions: list[dict[str, Any]],
        hierarchy_edges: list[dict[str, Any]] | None = None,
        inheritance_policies: list[dict[str, Any]] | None = None,
        competency_questions: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        run = self.get_run(run_id)
        decision_refs = [
            decision["decision_id"] for decision in self.list_decisions(run_id)
        ]
        brief = {
            "schema_version": BRIEF_SCHEMA_VERSION,
            "brief_id": f"gsdb_{uuid.uuid4().hex}",
            "run_id": run_id,
            "lineage_id": run["lineage_id"],
            "experiment_id": run["experiment_id"],
            "workspace_id": self.workspace_id,
            "source_ref": run["source_ref"],
            "source_hash": run["source_hash"],
            "skill_ref": run["skill_ref"],
            "semantic_roles": semantic_roles,
            "semantic_dimensions": semantic_dimensions,
            "hierarchy_edges": hierarchy_edges or [],
            "inheritance_policies": inheritance_policies or [],
            "competency_questions": competency_questions or [],
            "semantic_discovery_decision_refs": decision_refs,
            "created_at": _utc_now(),
        }
        brief["brief_hash"] = f"sha256:{content_hash(brief)}"
        with self._lock, self._connect() as connection:
            if enabled():begin_write(connection)
            connection.execute(
                sql_query("""
                INSERT INTO graph_schema_discovery_briefs (
                    brief_id, schema_version, run_id, experiment_id, workspace_id,
                    source_ref, source_hash, skill_ref, semantic_roles_json,
                    semantic_dimensions_json, hierarchy_edges_json,
                    inheritance_policies_json, competency_questions_json,
                    semantic_discovery_decision_refs_json, brief_hash, created_at,
                    lineage_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,'\n                INSERT INTO graph_schema_discovery_briefs (\n                    brief_id, schema_version, run_id, experiment_id, workspace_id,\n                    source_ref, source_hash, skill_ref, semantic_roles_json,\n                    semantic_dimensions_json, hierarchy_edges_json,\n                    inheritance_policies_json, competency_questions_json,\n                    semantic_discovery_decision_refs_json, brief_hash, created_at,\n                    lineage_id\n                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)\n                '),
                (
                    brief["brief_id"],
                    brief["schema_version"],
                    run_id,
                    brief["experiment_id"],
                    self.workspace_id,
                    brief["source_ref"],
                    brief["source_hash"],
                    brief["skill_ref"],
                    _dumps(semantic_roles),
                    _dumps(semantic_dimensions),
                    _dumps(brief["hierarchy_edges"]),
                    _dumps(brief["inheritance_policies"]),
                    _dumps(brief["competency_questions"]),
                    _dumps(decision_refs),
                    brief["brief_hash"],
                    brief["created_at"],
                    brief["lineage_id"],
                ),
            )
        return brief

    def record_llm_proposal_evidence(
        self,
        *,
        semantica_commit: str,
        skill_major_version: str,
        provider_contract_version: str,
        provider: str,
        model: str,
        status: Literal["passed", "failed", "unavailable", "disagreement"],
        run_id: str | None = None,
        proposal_ref: str | None = None,
        proposal_hash: str | None = None,
        diagnostics: dict[str, Any] | None = None,
        observed_at: str | None = None,
        evidence_id: str | None = None,
    ) -> dict[str, Any]:
        self._require(
            semantica_commit=semantica_commit,
            skill_major_version=skill_major_version,
            provider_contract_version=provider_contract_version,
            provider=provider,
            model=model,
            status=status,
        )
        if status not in {"passed", "failed", "unavailable", "disagreement"}:
            raise ValueError("LLM evidence status is invalid")
        if status == "passed" and (not proposal_ref or not proposal_hash):
            raise ValueError(
                "passed LLM evidence requires proposal_ref and proposal_hash"
            )
        lineage_id = None
        if run_id is not None:
            _require_safe_artifact_id(run_id, label="run_id")
            lineage_id = self.get_run(run_id)["lineage_id"]
        if evidence_id is not None:
            _require_safe_artifact_id(evidence_id, label="evidence_id")
        if diagnostics is not None and not isinstance(diagnostics, dict):
            raise ValueError("LLM evidence diagnostics must be an object")
        normalized_observed_at = (
            _require_utc_iso(observed_at, label="observed_at")
            if observed_at is not None
            else _utc_now()
        )
        evidence = {
            "schema_version": LLM_PROPOSAL_EVIDENCE_SCHEMA_VERSION,
            "evidence_id": evidence_id or f"llmpe_{uuid.uuid4().hex}",
            "run_id": run_id,
            "lineage_id": lineage_id,
            "workspace_id": self.workspace_id,
            "semantica_commit": semantica_commit,
            "skill_major_version": skill_major_version,
            "provider_contract_version": provider_contract_version,
            "provider": provider,
            "model": model,
            "status": status,
            "proposal_ref": proposal_ref,
            "proposal_hash": proposal_hash,
            "diagnostics": diagnostics or {},
            "observed_at": normalized_observed_at,
        }
        with self._lock, self._connect() as connection:
            if enabled():begin_write(connection)
            connection.execute(
                sql_query("""
                INSERT INTO llm_proposal_evidence (
                    evidence_id, schema_version, run_id, workspace_id,
                    semantica_commit, skill_major_version,
                    provider_contract_version, provider, model, status,
                    proposal_ref, proposal_hash, diagnostics_json, observed_at,
                    lineage_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,'\n                INSERT INTO llm_proposal_evidence (\n                    evidence_id, schema_version, run_id, workspace_id,\n                    semantica_commit, skill_major_version,\n                    provider_contract_version, provider, model, status,\n                    proposal_ref, proposal_hash, diagnostics_json, observed_at,\n                    lineage_id\n                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)\n                '),
                (
                    evidence["evidence_id"],
                    evidence["schema_version"],
                    evidence["run_id"],
                    self.workspace_id,
                    semantica_commit,
                    skill_major_version,
                    provider_contract_version,
                    provider,
                    model,
                    status,
                    proposal_ref,
                    proposal_hash,
                    _dumps(evidence["diagnostics"]),
                    evidence["observed_at"],
                    evidence["lineage_id"],
                ),
            )
        return evidence

    def latest_qualifying_llm_evidence(
        self,
        *,
        semantica_commit: str,
        skill_major_version: str,
        provider_contract_version: str,
        now: str | None = None,
        max_age_days: int = 30,
        require_reviewed: bool = False,
    ) -> dict[str, Any] | None:
        if (
            not isinstance(max_age_days, int)
            or isinstance(max_age_days, bool)
            or max_age_days <= 0
        ):
            raise ValueError("max_age_days must be positive")
        reference = datetime.fromisoformat(
            _require_utc_iso(now, label="now") if now is not None else _utc_now()
        )
        with self._connect() as connection:
            rows = connection.execute(
                sql_query("""
                SELECT * FROM llm_proposal_evidence
                WHERE workspace_id = ?
                  AND semantica_commit = ?
                  AND skill_major_version = ?
                  AND provider_contract_version = ?
                  AND status = 'passed'
                ORDER BY observed_at DESC, evidence_id DESC
                ""","\n                SELECT * FROM llm_proposal_evidence\n                WHERE workspace_id = %s\n                  AND semantica_commit = %s\n                  AND skill_major_version = %s\n                  AND provider_contract_version = %s\n                  AND status = 'passed'\n                ORDER BY observed_at DESC, evidence_id DESC\n                "),
                (
                    self.workspace_id,
                    semantica_commit,
                    skill_major_version,
                    provider_contract_version,
                ),
            ).fetchall()
        for row in rows:
            evidence = self._hydrate_llm_evidence(row)
            observed = datetime.fromisoformat(
                evidence["observed_at"].replace("Z", "+00:00")
            )
            if observed.tzinfo is None:
                continue
            if (reference - observed).days <= max_age_days:
                if require_reviewed and not self._llm_evidence_is_reviewed(evidence):
                    continue
                return evidence
        return None

    def record_llm_proposal_evidence_review(
        self,
        evidence_id: str,
        *,
        state: Literal["accepted", "rejected"],
        actor_id: str,
        note: str | None = None,
    ) -> dict[str, Any]:
        self._require(evidence_id=evidence_id, actor_id=actor_id, state=state)
        _require_safe_artifact_id(evidence_id, label="evidence_id")
        if state not in {"accepted", "rejected"}:
            raise ValueError("LLM evidence review state is invalid")
        if note is not None and not isinstance(note, str):
            raise ValueError("LLM evidence review note must be a string")

        evidence = self.get_llm_proposal_evidence(evidence_id)
        if state == "accepted" and (
            evidence.get("status") != "passed"
            or not evidence.get("proposal_ref")
            or not evidence.get("proposal_hash")
        ):
            raise ValueError(
                "accepted LLM evidence review requires passed proposal evidence"
            )
        diagnostics = dict(evidence.get("diagnostics") or {})
        review: dict[str, Any] = {
            "state": state,
            "actor_id": actor_id,
            "reviewed_at": _utc_now(),
        }
        if note is not None:
            review["note"] = note
        diagnostics["review"] = review

        with self._lock, self._connect() as connection:
            if enabled():begin_write(connection)
            connection.execute(
                sql_query("""
                UPDATE llm_proposal_evidence
                SET diagnostics_json = ?
                WHERE evidence_id = ? AND workspace_id = ?
                """,'\n                UPDATE llm_proposal_evidence\n                SET diagnostics_json = %s\n                WHERE evidence_id = %s AND workspace_id = %s\n                '),
                (_dumps(diagnostics), evidence_id, self.workspace_id),
            )
        return self.get_llm_proposal_evidence(evidence_id)

    @staticmethod
    def _llm_evidence_is_reviewed(evidence: dict[str, Any]) -> bool:
        diagnostics = evidence.get("diagnostics") or {}
        if not isinstance(diagnostics, dict):
            return False
        review = diagnostics.get("review") or {}
        if not isinstance(review, dict):
            return False
        if review.get("state") != "accepted":
            return False
        actor_id = review.get("actor_id")
        reviewed_at = review.get("reviewed_at")
        if not isinstance(actor_id, str) or not actor_id.strip():
            return False
        if not isinstance(reviewed_at, str):
            return False
        try:
            _require_utc_iso(reviewed_at, label="reviewed_at")
        except ValueError:
            return False
        return True

    def get_run(self, run_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                sql_query("SELECT * FROM graph_schema_discovery_runs WHERE run_id = ? AND workspace_id = ?",'SELECT * FROM graph_schema_discovery_runs WHERE run_id = %s AND workspace_id = %s'),
                (run_id, self.workspace_id),
            ).fetchone()
        if row is None:
            raise KeyError(f"graph schema discovery run not found: {run_id}")
        return dict(row)

    def get_decision(self, decision_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                sql_query("SELECT * FROM semantic_discovery_decisions WHERE decision_id = ? AND workspace_id = ?",'SELECT * FROM semantic_discovery_decisions WHERE decision_id = %s AND workspace_id = %s'),
                (decision_id, self.workspace_id),
            ).fetchone()
        if row is None:
            raise KeyError(f"semantic discovery decision not found: {decision_id}")
        return self._hydrate_decision(row)

    def list_decisions(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                sql_query("""
                SELECT * FROM semantic_discovery_decisions
                WHERE run_id = ? AND workspace_id = ?
                ORDER BY created_at, decision_id
                """,'\n                SELECT * FROM semantic_discovery_decisions\n                WHERE run_id = %s AND workspace_id = %s\n                ORDER BY created_at, decision_id\n                '),
                (run_id, self.workspace_id),
            ).fetchall()
        return [self._hydrate_decision(row) for row in rows]

    def list_decisions_by_type(
        self, discovery_type: str, *, limit: int = 20
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        with self._connect() as connection:
            rows = connection.execute(
                sql_query("""
                SELECT * FROM semantic_discovery_decisions
                WHERE discovery_type = ? AND workspace_id = ?
                  AND retirement_state = 'active'
                  AND lineage_id IS NOT NULL AND TRIM(lineage_id) != ''
                ORDER BY created_at DESC, decision_id DESC
                LIMIT ?
                ""","\n                SELECT * FROM semantic_discovery_decisions\n                WHERE discovery_type = %s AND workspace_id = %s\n                  AND retirement_state = 'active'\n                  AND lineage_id IS NOT NULL AND TRIM(lineage_id) != ''\n                ORDER BY created_at DESC, decision_id DESC\n                LIMIT %s\n                "),
                (discovery_type, self.workspace_id, limit),
            ).fetchall()
        return [self._hydrate_decision(row) for row in rows]

    def get_brief(self, brief_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                sql_query("SELECT * FROM graph_schema_discovery_briefs WHERE brief_id = ? AND workspace_id = ?",'SELECT * FROM graph_schema_discovery_briefs WHERE brief_id = %s AND workspace_id = %s'),
                (brief_id, self.workspace_id),
            ).fetchone()
        if row is None:
            raise KeyError(f"graph schema discovery brief not found: {brief_id}")
        payload = dict(row)
        for field in (
            "semantic_roles_json",
            "semantic_dimensions_json",
            "hierarchy_edges_json",
            "inheritance_policies_json",
            "competency_questions_json",
            "semantic_discovery_decision_refs_json",
        ):
            payload[field.removesuffix("_json")] = _loads(payload.pop(field))
        return payload

    def get_llm_proposal_evidence(self, evidence_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                sql_query("SELECT * FROM llm_proposal_evidence WHERE evidence_id = ? AND workspace_id = ?",'SELECT * FROM llm_proposal_evidence WHERE evidence_id = %s AND workspace_id = %s'),
                (evidence_id, self.workspace_id),
            ).fetchone()
        if row is None:
            raise KeyError(f"LLM proposal evidence not found: {evidence_id}")
        return self._hydrate_llm_evidence(row)

    def _hydrate_decision(self, row: sqlite3.Row) -> dict[str, Any]:
        payload = dict(row)
        for field in (
            "subject_json",
            "proposed_semantics_json",
            "rationale_json",
            "alternatives_considered_json",
            "semantica_signals_json",
            "human_review_json",
            "provenance_json",
            "learning_eligibility_json",
        ):
            payload[field.removesuffix("_json")] = _loads(payload.pop(field))
        review = payload["human_review"]
        edited_fields = {
            "edited_payload",
            "edited_payload_hash",
            "edited_payload_binding",
        }
        if review.get("state") == "edited" and edited_fields.intersection(review):
            edited_payload = review.get("edited_payload")
            edited_payload_hash = review.get("edited_payload_hash")
            expected_hash = (
                f"sha256:{content_hash(edited_payload)}"
                if isinstance(edited_payload, dict)
                else None
            )
            expected_binding = f"sha256:{content_hash({'decision_hash': payload['decision_hash'], 'edited_payload_hash': edited_payload_hash})}"
            if (
                edited_payload_hash != expected_hash
                or review.get("edited_payload_binding") != expected_binding
            ):
                raise ValueError(
                    "edited review payload or binding is missing or tampered"
                )
        return payload

    def _hydrate_llm_evidence(self, row: sqlite3.Row) -> dict[str, Any]:
        payload = dict(row)
        payload["diagnostics"] = _loads(payload.pop("diagnostics_json"))
        return payload

    def _require(self, **values: str) -> None:
        missing = sorted(
            name
            for name, value in values.items()
            if not isinstance(value, str) or not value.strip()
        )
        if missing:
            raise ValueError(f"graph schema discovery record missing fields: {missing}")
