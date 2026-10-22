"""Graph delta and learning-loop storage for graph schema MVL work."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import yaml

from storage.hosted_postgres import enabled,connect_authority,mapped_store,require_tables,sql_query,begin_write,integrity_errors
from eos.artifact_hashing import canonical_json, content_hash
from services.canonical_skill_registry import (
    CANONICAL_JOURNEY_REF,
    CANONICAL_JOURNEY_NAME,
    SkillRegistryError,
    resolve_historical_ref,
    select_for_new_run,
)


GRAPH_DELTA_CANDIDATE_SCHEMA_VERSION = "graph_delta_candidate.v1"
LEARNING_CANDIDATE_SCHEMA_VERSION = "learning_candidate.v1"
SKILL_PATCH_PROPOSAL_SCHEMA_VERSION = "skill_patch_proposal.v1"
EFFECTIVE_SKILL_VERSION_SCHEMA_VERSION = "effective_skill_version.v1"
DeltaReviewState = Literal[
    "proposed", "human_accepted", "human_edited", "human_rejected"
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _loads(value: str) -> Any:
    return json.loads(value) if value else None


def _reviewed_result_hash(semantic_version: str, overlay_content: str) -> str:
    return f"sha256:{content_hash({'semantic_version': semantic_version, 'overlay_content': overlay_content})}"


def _validate_overlay_frontmatter(
    *, semantic_version: str, overlay_content: str
) -> None:
    if not overlay_content.startswith("---\n") or "\n---\n" not in overlay_content[4:]:
        raise ValueError("reviewed overlay frontmatter is missing")
    raw_frontmatter = overlay_content[4:].split("\n---\n", 1)[0]
    parsed = yaml.safe_load(raw_frontmatter) or {}
    if not isinstance(parsed, dict):
        raise ValueError("reviewed overlay frontmatter is invalid")
    if parsed.get("name") != CANONICAL_JOURNEY_NAME:
        raise ValueError(
            "reviewed overlay frontmatter name must be knowledge-graph-building"
        )
    if str(parsed.get("version")) != semantic_version:
        raise ValueError(
            "reviewed overlay frontmatter version must match semantic identity"
        )


@dataclass(frozen=True)
class GraphDeltaCandidate:
    schema_version: str
    delta_id: str
    experiment_id: str
    base_graph_version_ref: str
    base_ontology_version_ref: str | None
    feedback_refs: list[str]
    attribution_refs: list[str]
    proposed_node_changes: list[dict[str, Any]]
    proposed_edge_changes: list[dict[str, Any]]
    proposed_inheritance_policy_changes: list[dict[str, Any]]
    proposed_profile_changes: list[dict[str, Any]]
    rationale_summary: str
    review_state: DeltaReviewState
    reviewed_by: str | None
    reviewed_at: str | None
    delta_hash: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "delta_id": self.delta_id,
            "experiment_id": self.experiment_id,
            "base_graph_version_ref": self.base_graph_version_ref,
            "base_ontology_version_ref": self.base_ontology_version_ref,
            "feedback_refs": self.feedback_refs,
            "attribution_refs": self.attribution_refs,
            "proposed_node_changes": self.proposed_node_changes,
            "proposed_edge_changes": self.proposed_edge_changes,
            "proposed_inheritance_policy_changes": self.proposed_inheritance_policy_changes,
            "proposed_profile_changes": self.proposed_profile_changes,
            "rationale_summary": self.rationale_summary,
            "review_state": self.review_state,
            "reviewed_by": self.reviewed_by,
            "reviewed_at": self.reviewed_at,
            "delta_hash": self.delta_hash,
        }


class GraphSchemaLearningStore:
    """Append-only graph delta and skill-learning records."""

    _process_locks: dict[Path, threading.RLock] = {}
    _process_locks_guard = threading.Lock()

    def __init__(self, workspace_root: str | Path, *, workspace_id: str) -> None:
        if not workspace_id:
            raise ValueError("workspace_id is required")
        self.workspace_root = Path(workspace_root).resolve()
        self.workspace_id = workspace_id
        self.root = self.workspace_root / ".semantier" / "graph-schema-learning"
        if not enabled():self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "learning.sqlite3"
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
            with self._connect() as connection:require_tables(connection,'graph_delta_candidates','learning_candidates','skill_patch_proposals','effective_skill_versions')
            return
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS graph_delta_candidates (
                    delta_id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    schema_version TEXT NOT NULL,
                    experiment_id TEXT NOT NULL,
                    base_graph_version_ref TEXT NOT NULL,
                    base_ontology_version_ref TEXT,
                    feedback_refs_json TEXT NOT NULL,
                    attribution_refs_json TEXT NOT NULL,
                    proposed_node_changes_json TEXT NOT NULL,
                    proposed_edge_changes_json TEXT NOT NULL,
                    proposed_inheritance_policy_changes_json TEXT NOT NULL,
                    proposed_profile_changes_json TEXT NOT NULL,
                    rationale_summary TEXT NOT NULL,
                    review_state TEXT NOT NULL CHECK (review_state IN ('proposed', 'human_accepted', 'human_edited', 'human_rejected')),
                    reviewed_by TEXT,
                    reviewed_at TEXT,
                    delta_hash TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS learning_candidates (
                    candidate_id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    schema_version TEXT NOT NULL,
                    source_delta_ref TEXT NOT NULL,
                    evidence_refs_json TEXT NOT NULL,
                    hypothesis TEXT NOT NULL,
                    promoted_to_skill_review INTEGER NOT NULL,
                    promoted_by TEXT,
                    promoted_at TEXT,
                    candidate_hash TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS skill_patch_proposals (
                    proposal_id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    schema_version TEXT NOT NULL,
                    target_skill TEXT NOT NULL,
                    base_skill_ref TEXT NOT NULL,
                    learning_candidate_ref TEXT NOT NULL,
                    evidence_refs_json TEXT NOT NULL,
                    patch_unified_diff TEXT NOT NULL,
                    proposed_by TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    review_state TEXT NOT NULL CHECK (review_state IN ('proposed', 'human_accepted', 'human_edited', 'human_rejected')),
                    reviewed_by TEXT,
                    reviewed_at TEXT,
                    reviewed_semantic_version TEXT,
                    reviewed_overlay_content TEXT,
                    reviewed_result_hash TEXT,
                    proposal_hash TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS effective_skill_versions (
                    effective_id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    schema_version TEXT NOT NULL,
                    skill_name TEXT NOT NULL,
                    skill_ref TEXT NOT NULL,
                    semantic_version TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    source_kind TEXT NOT NULL CHECK (source_kind IN ('application_release', 'governed_workspace_overlay')),
                    source_ref TEXT NOT NULL,
                    activated_by TEXT NOT NULL,
                    activated_at TEXT NOT NULL,
                    supersedes_skill_ref TEXT,
                    rollback_to_skill_ref TEXT,
                    activation_key TEXT,
                    effective_hash TEXT NOT NULL
                )
                """
            )
            proposal_columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(skill_patch_proposals)"
                )
            }
            for column in (
                "reviewed_semantic_version",
                "reviewed_overlay_content",
                "reviewed_result_hash",
            ):
                if column not in proposal_columns:
                    connection.execute(
                        f"ALTER TABLE skill_patch_proposals ADD COLUMN {column} TEXT"
                    )
            effective_columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(effective_skill_versions)"
                )
            }
            if "activation_key" not in effective_columns:
                connection.execute(
                    "ALTER TABLE effective_skill_versions ADD COLUMN activation_key TEXT"
                )
            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_effective_skill_activation_key
                ON effective_skill_versions(workspace_id, activation_key)
                WHERE activation_key IS NOT NULL
                """
            )

    def create_graph_delta_candidate(
        self,
        *,
        experiment_id: str,
        base_graph_version_ref: str,
        feedback_refs: list[str],
        attribution_refs: list[str],
        proposed_node_changes: list[dict[str, Any]],
        proposed_edge_changes: list[dict[str, Any]] | None = None,
        proposed_inheritance_policy_changes: list[dict[str, Any]] | None = None,
        proposed_profile_changes: list[dict[str, Any]] | None = None,
        rationale_summary: str,
        base_ontology_version_ref: str | None = None,
    ) -> GraphDeltaCandidate:
        if not experiment_id or not base_graph_version_ref or not rationale_summary:
            raise ValueError(
                "experiment_id, base_graph_version_ref, and rationale_summary are required"
            )
        if not feedback_refs or not attribution_refs:
            raise ValueError("graph delta requires feedback_refs and attribution_refs")
        if (
            not proposed_node_changes
            and not proposed_edge_changes
            and not proposed_profile_changes
        ):
            raise ValueError("graph delta requires at least one proposed change")
        payload = {
            "schema_version": GRAPH_DELTA_CANDIDATE_SCHEMA_VERSION,
            "delta_id": f"gdc_{uuid.uuid4().hex}",
            "experiment_id": experiment_id,
            "base_graph_version_ref": base_graph_version_ref,
            "base_ontology_version_ref": base_ontology_version_ref,
            "feedback_refs": feedback_refs,
            "attribution_refs": attribution_refs,
            "proposed_node_changes": proposed_node_changes,
            "proposed_edge_changes": proposed_edge_changes or [],
            "proposed_inheritance_policy_changes": proposed_inheritance_policy_changes
            or [],
            "proposed_profile_changes": proposed_profile_changes or [],
            "rationale_summary": rationale_summary,
            "review_state": "proposed",
            "reviewed_by": None,
            "reviewed_at": None,
        }
        payload["delta_hash"] = f"sha256:{content_hash(payload)}"
        delta = GraphDeltaCandidate(**payload)
        with self._lock, self._connect() as connection:
            if enabled():begin_write(connection)
            connection.execute(
                sql_query("""
                INSERT INTO graph_delta_candidates VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,'\n                INSERT INTO graph_delta_candidates VALUES (\n                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s\n                )\n                '),
                (
                    delta.delta_id,
                    self.workspace_id,
                    delta.schema_version,
                    delta.experiment_id,
                    delta.base_graph_version_ref,
                    delta.base_ontology_version_ref,
                    canonical_json(delta.feedback_refs),
                    canonical_json(delta.attribution_refs),
                    canonical_json(delta.proposed_node_changes),
                    canonical_json(delta.proposed_edge_changes),
                    canonical_json(delta.proposed_inheritance_policy_changes),
                    canonical_json(delta.proposed_profile_changes),
                    delta.rationale_summary,
                    delta.review_state,
                    delta.reviewed_by,
                    delta.reviewed_at,
                    delta.delta_hash,
                ),
            )
        return delta

    def record_graph_delta_review(
        self,
        delta_id: str,
        *,
        review_state: DeltaReviewState,
        reviewed_by: str,
    ) -> GraphDeltaCandidate:
        if review_state not in {"human_accepted", "human_edited", "human_rejected"}:
            raise ValueError(
                "delta review must be human_accepted, human_edited, or human_rejected"
            )
        if not reviewed_by:
            raise ValueError("reviewed_by is required")
        with self._lock, self._connect() as connection:
            if enabled():begin_write(connection)
            if (
                connection.execute(
                    sql_query("SELECT 1 FROM graph_delta_candidates WHERE workspace_id = ? AND delta_id = ?",'SELECT 1 FROM graph_delta_candidates WHERE workspace_id = %s AND delta_id = %s'),
                    (self.workspace_id, delta_id),
                ).fetchone()
                is None
            ):
                raise KeyError(f"graph delta candidate not found: {delta_id}")
            connection.execute(
                sql_query("""
                UPDATE graph_delta_candidates
                SET review_state = ?, reviewed_by = ?, reviewed_at = ?
                WHERE workspace_id = ? AND delta_id = ?
                """,'\n                UPDATE graph_delta_candidates\n                SET review_state = %s, reviewed_by = %s, reviewed_at = %s\n                WHERE workspace_id = %s AND delta_id = %s\n                '),
                (review_state, reviewed_by, _utc_now(), self.workspace_id, delta_id),
            )
        return self.get_graph_delta_candidate(delta_id)

    def get_graph_delta_candidate(self, delta_id: str) -> GraphDeltaCandidate:
        with self._connect() as connection:
            row = connection.execute(
                sql_query("SELECT * FROM graph_delta_candidates WHERE workspace_id = ? AND delta_id = ?",'SELECT * FROM graph_delta_candidates WHERE workspace_id = %s AND delta_id = %s'),
                (self.workspace_id, delta_id),
            ).fetchone()
        if row is None:
            raise KeyError(f"graph delta candidate not found: {delta_id}")
        return GraphDeltaCandidate(
            schema_version=row["schema_version"],
            delta_id=row["delta_id"],
            experiment_id=row["experiment_id"],
            base_graph_version_ref=row["base_graph_version_ref"],
            base_ontology_version_ref=row["base_ontology_version_ref"],
            feedback_refs=_loads(row["feedback_refs_json"]),
            attribution_refs=_loads(row["attribution_refs_json"]),
            proposed_node_changes=_loads(row["proposed_node_changes_json"]),
            proposed_edge_changes=_loads(row["proposed_edge_changes_json"]),
            proposed_inheritance_policy_changes=_loads(
                row["proposed_inheritance_policy_changes_json"]
            ),
            proposed_profile_changes=_loads(row["proposed_profile_changes_json"]),
            rationale_summary=row["rationale_summary"],
            review_state=row["review_state"],
            reviewed_by=row["reviewed_by"],
            reviewed_at=row["reviewed_at"],
            delta_hash=row["delta_hash"],
        )

    def create_learning_candidate(
        self,
        *,
        source_delta_ref: str,
        evidence_refs: list[str],
        hypothesis: str,
    ) -> dict[str, Any]:
        if not source_delta_ref or not evidence_refs or not hypothesis:
            raise ValueError(
                "learning candidate requires source_delta_ref, evidence_refs, and hypothesis"
            )
        payload = {
            "schema_version": LEARNING_CANDIDATE_SCHEMA_VERSION,
            "candidate_id": f"lc_{uuid.uuid4().hex}",
            "source_delta_ref": source_delta_ref,
            "evidence_refs": evidence_refs,
            "hypothesis": hypothesis,
            "promoted_to_skill_review": False,
            "promoted_by": None,
            "promoted_at": None,
        }
        payload["candidate_hash"] = f"sha256:{content_hash(payload)}"
        with self._lock, self._connect() as connection:
            if enabled():begin_write(connection)
            connection.execute(
                sql_query("""
                INSERT INTO learning_candidates VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,'\n                INSERT INTO learning_candidates VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)\n                '),
                (
                    payload["candidate_id"],
                    self.workspace_id,
                    payload["schema_version"],
                    payload["source_delta_ref"],
                    canonical_json(payload["evidence_refs"]),
                    payload["hypothesis"],
                    0,
                    None,
                    None,
                    payload["candidate_hash"],
                ),
            )
        return payload

    def promote_learning_candidate(
        self, candidate_id: str, *, promoted_by: str
    ) -> dict[str, Any]:
        if not promoted_by:
            raise ValueError("promoted_by is required")
        with self._lock, self._connect() as connection:
            if enabled():begin_write(connection)
            if (
                connection.execute(
                    sql_query("SELECT 1 FROM learning_candidates WHERE workspace_id = ? AND candidate_id = ?",'SELECT 1 FROM learning_candidates WHERE workspace_id = %s AND candidate_id = %s'),
                    (self.workspace_id, candidate_id),
                ).fetchone()
                is None
            ):
                raise KeyError(f"learning candidate not found: {candidate_id}")
            connection.execute(
                sql_query("""
                UPDATE learning_candidates
                SET promoted_to_skill_review = 1, promoted_by = ?, promoted_at = ?
                WHERE workspace_id = ? AND candidate_id = ?
                """,'\n                UPDATE learning_candidates\n                SET promoted_to_skill_review = 1, promoted_by = %s, promoted_at = %s\n                WHERE workspace_id = %s AND candidate_id = %s\n                '),
                (promoted_by, _utc_now(), self.workspace_id, candidate_id),
            )
        return self.get_learning_candidate(candidate_id)

    def get_learning_candidate(self, candidate_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                sql_query("SELECT * FROM learning_candidates WHERE workspace_id = ? AND candidate_id = ?",'SELECT * FROM learning_candidates WHERE workspace_id = %s AND candidate_id = %s'),
                (self.workspace_id, candidate_id),
            ).fetchone()
        if row is None:
            raise KeyError(f"learning candidate not found: {candidate_id}")
        return {
            "schema_version": row["schema_version"],
            "candidate_id": row["candidate_id"],
            "source_delta_ref": row["source_delta_ref"],
            "evidence_refs": _loads(row["evidence_refs_json"]),
            "hypothesis": row["hypothesis"],
            "promoted_to_skill_review": bool(row["promoted_to_skill_review"]),
            "promoted_by": row["promoted_by"],
            "promoted_at": row["promoted_at"],
            "candidate_hash": row["candidate_hash"],
        }

    def create_skill_patch_proposal(
        self,
        *,
        target_skill: str,
        base_skill_ref: str,
        learning_candidate_ref: str,
        evidence_refs: list[str],
        patch_unified_diff: str,
        proposed_by: str,
    ) -> dict[str, Any]:
        candidate = self.get_learning_candidate(learning_candidate_ref)
        if not candidate["promoted_to_skill_review"]:
            raise ValueError(
                "learning candidate must be human-promoted to skill review"
            )
        self._validate_new_learning_base(
            skill_name=target_skill,
            skill_ref=base_skill_ref,
        )
        if not evidence_refs or not patch_unified_diff or not proposed_by:
            raise ValueError(
                "skill patch proposal requires evidence, diff, and proposer"
            )
        payload = {
            "schema_version": SKILL_PATCH_PROPOSAL_SCHEMA_VERSION,
            "proposal_id": f"spp_{uuid.uuid4().hex}",
            "target_skill": target_skill,
            "base_skill_ref": base_skill_ref,
            "learning_candidate_ref": learning_candidate_ref,
            "evidence_refs": evidence_refs,
            "patch_unified_diff": patch_unified_diff,
            "proposed_by": proposed_by,
            "created_at": _utc_now(),
            "review_state": "proposed",
            "reviewed_by": None,
            "reviewed_at": None,
        }
        payload["proposal_hash"] = f"sha256:{content_hash(payload)}"
        with self._lock, self._connect() as connection:
            if enabled():begin_write(connection)
            connection.execute(
                sql_query("""
                INSERT INTO skill_patch_proposals (
                    proposal_id, workspace_id, schema_version, target_skill,
                    base_skill_ref, learning_candidate_ref, evidence_refs_json,
                    patch_unified_diff, proposed_by, created_at, review_state,
                    reviewed_by, reviewed_at, proposal_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,'\n                INSERT INTO skill_patch_proposals (\n                    proposal_id, workspace_id, schema_version, target_skill,\n                    base_skill_ref, learning_candidate_ref, evidence_refs_json,\n                    patch_unified_diff, proposed_by, created_at, review_state,\n                    reviewed_by, reviewed_at, proposal_hash\n                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)\n                '),
                (
                    payload["proposal_id"],
                    self.workspace_id,
                    payload["schema_version"],
                    payload["target_skill"],
                    payload["base_skill_ref"],
                    payload["learning_candidate_ref"],
                    canonical_json(payload["evidence_refs"]),
                    payload["patch_unified_diff"],
                    payload["proposed_by"],
                    payload["created_at"],
                    payload["review_state"],
                    payload["reviewed_by"],
                    payload["reviewed_at"],
                    payload["proposal_hash"],
                ),
            )
        return payload

    def _validate_new_learning_base(self, *, skill_name: str, skill_ref: str) -> None:
        try:
            registry_record = select_for_new_run(skill_ref=skill_ref)
        except SkillRegistryError as exc:
            if (
                self.resolve_governed_overlay_base(
                    skill_name=skill_name,
                    skill_ref=skill_ref,
                )
                is not None
            ):
                return
            raise ValueError(
                f"base skill ref is not eligible for new learning: {skill_ref}"
            ) from exc
        if registry_record is None or registry_record.name != skill_name:
            raise ValueError(
                f"base skill ref is not eligible for new learning: {skill_ref}"
            )

    def record_skill_patch_review(
        self,
        proposal_id: str,
        *,
        review_state: str,
        reviewed_by: str,
        reviewed_semantic_version: str | None = None,
        reviewed_overlay_content: str | None = None,
    ) -> dict[str, Any]:
        if review_state not in {"human_accepted", "human_edited", "human_rejected"}:
            raise ValueError(
                "skill patch review must be human_accepted, human_edited, or human_rejected"
            )
        if not reviewed_by:
            raise ValueError("reviewed_by is required")
        reviewed_result_hash: str | None = None
        if review_state in {"human_accepted", "human_edited"}:
            if not reviewed_semantic_version or not reviewed_overlay_content:
                raise ValueError(
                    "accepted or edited review requires the exact reviewed result"
                )
            _validate_overlay_frontmatter(
                semantic_version=reviewed_semantic_version,
                overlay_content=reviewed_overlay_content,
            )
            reviewed_result_hash = _reviewed_result_hash(
                reviewed_semantic_version,
                reviewed_overlay_content,
            )
        elif (
            reviewed_semantic_version is not None
            or reviewed_overlay_content is not None
        ):
            raise ValueError("rejected review cannot pin an activation result")
        with self._lock, self._connect() as connection:
            begin_write(connection)
            row = connection.execute(
                sql_query("SELECT * FROM skill_patch_proposals WHERE workspace_id = ? AND proposal_id = ?",'SELECT * FROM skill_patch_proposals WHERE workspace_id = %s AND proposal_id = %s'),
                (self.workspace_id, proposal_id),
            ).fetchone()
            if row is None:
                raise KeyError(f"skill patch proposal not found: {proposal_id}")
            self._validate_skill_patch_proposal_integrity(
                self._skill_patch_proposal_from_row(row)
            )
            source_ref = (
                Path(".semantier")
                / "skill-overlays"
                / row["target_skill"]
                / f"{proposal_id}.SKILL.md"
            ).as_posix()
            activated = connection.execute(
                sql_query("""
                SELECT 1 FROM effective_skill_versions
                WHERE workspace_id = ? AND source_kind = 'governed_workspace_overlay'
                  AND (activation_key = ? OR source_ref = ?)
                LIMIT 1
                ""","\n                SELECT 1 FROM effective_skill_versions\n                WHERE workspace_id = %s AND source_kind = 'governed_workspace_overlay'\n                  AND (activation_key = %s OR source_ref = %s)\n                LIMIT 1\n                "),
                (self.workspace_id, proposal_id, source_ref),
            ).fetchone()
            if activated is not None:
                raise ValueError(
                    "activated skill patch proposal is immutable; "
                    "create a new successor proposal with base_skill_ref set "
                    "to the activated overlay"
                )
            connection.execute(
                sql_query("""
                UPDATE skill_patch_proposals SET
                    review_state = ?, reviewed_by = ?, reviewed_at = ?,
                    reviewed_semantic_version = ?, reviewed_overlay_content = ?,
                    reviewed_result_hash = ?
                WHERE workspace_id = ? AND proposal_id = ?
                """,'\n                UPDATE skill_patch_proposals SET\n                    review_state = %s, reviewed_by = %s, reviewed_at = %s,\n                    reviewed_semantic_version = %s, reviewed_overlay_content = %s,\n                    reviewed_result_hash = %s\n                WHERE workspace_id = %s AND proposal_id = %s\n                '),
                (
                    review_state,
                    reviewed_by,
                    _utc_now(),
                    reviewed_semantic_version,
                    reviewed_overlay_content,
                    reviewed_result_hash,
                    self.workspace_id,
                    proposal_id,
                ),
            )
        return self.get_skill_patch_proposal(proposal_id)

    def get_skill_patch_proposal(self, proposal_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                sql_query("SELECT * FROM skill_patch_proposals WHERE workspace_id = ? AND proposal_id = ?",'SELECT * FROM skill_patch_proposals WHERE workspace_id = %s AND proposal_id = %s'),
                (self.workspace_id, proposal_id),
            ).fetchone()
        if row is None:
            raise KeyError(f"skill patch proposal not found: {proposal_id}")
        return self._skill_patch_proposal_from_row(row)

    @staticmethod
    def _skill_patch_proposal_from_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "schema_version": row["schema_version"],
            "proposal_id": row["proposal_id"],
            "target_skill": row["target_skill"],
            "base_skill_ref": row["base_skill_ref"],
            "learning_candidate_ref": row["learning_candidate_ref"],
            "evidence_refs": _loads(row["evidence_refs_json"]),
            "patch_unified_diff": row["patch_unified_diff"],
            "proposed_by": row["proposed_by"],
            "created_at": row["created_at"],
            "review_state": row["review_state"],
            "reviewed_by": row["reviewed_by"],
            "reviewed_at": row["reviewed_at"],
            "reviewed_semantic_version": row["reviewed_semantic_version"],
            "reviewed_overlay_content": row["reviewed_overlay_content"],
            "reviewed_result_hash": row["reviewed_result_hash"],
            "proposal_hash": row["proposal_hash"],
        }

    def _validate_skill_patch_proposal_integrity(
        self, proposal: dict[str, Any]
    ) -> None:
        original_payload = {
            "schema_version": proposal["schema_version"],
            "proposal_id": proposal["proposal_id"],
            "target_skill": proposal["target_skill"],
            "base_skill_ref": proposal["base_skill_ref"],
            "learning_candidate_ref": proposal["learning_candidate_ref"],
            "evidence_refs": proposal["evidence_refs"],
            "patch_unified_diff": proposal["patch_unified_diff"],
            "proposed_by": proposal["proposed_by"],
            "created_at": proposal["created_at"],
            "review_state": "proposed",
            "reviewed_by": None,
            "reviewed_at": None,
        }
        expected_hash = f"sha256:{content_hash(original_payload)}"
        if proposal["proposal_hash"] != expected_hash:
            raise ValueError("skill patch proposal integrity check failed")

    def activate_reviewed_skill_patch(
        self,
        *,
        proposal_id: str,
        activated_by: str,
    ) -> dict[str, Any]:
        if not activated_by:
            raise ValueError("activated_by is required")
        with self._lock, self._connect() as connection:
            begin_write(connection)
            row = connection.execute(
                sql_query("SELECT * FROM skill_patch_proposals WHERE workspace_id = ? AND proposal_id = ?",'SELECT * FROM skill_patch_proposals WHERE workspace_id = %s AND proposal_id = %s'),
                (self.workspace_id, proposal_id),
            ).fetchone()
            if row is None:
                raise KeyError(f"skill patch proposal not found: {proposal_id}")
            proposal = self._skill_patch_proposal_from_row(row)
            if proposal["target_skill"] != CANONICAL_JOURNEY_NAME:
                raise ValueError(
                    "skill patch proposal target is not the canonical journey"
                )
            self._validate_skill_patch_proposal_integrity(proposal)
            if proposal["review_state"] not in {"human_accepted", "human_edited"}:
                raise ValueError(
                    "only human-accepted or human-edited skill patches may be activated"
                )
            semantic_version = proposal["reviewed_semantic_version"]
            overlay_content = proposal["reviewed_overlay_content"]
            reviewed_result_hash = proposal["reviewed_result_hash"]
            if not semantic_version or not overlay_content or not reviewed_result_hash:
                raise ValueError(
                    "skill patch activation lacks an exact reviewed result"
                )
            if reviewed_result_hash != _reviewed_result_hash(
                semantic_version, overlay_content
            ):
                raise ValueError("skill patch exact reviewed result integrity failed")
            _validate_overlay_frontmatter(
                semantic_version=semantic_version,
                overlay_content=overlay_content,
            )
            self._validate_new_learning_base(
                skill_name=proposal["target_skill"],
                skill_ref=proposal["base_skill_ref"],
            )
            skill_ref = (
                f"{proposal['target_skill']}@{semantic_version}+overlay.{proposal_id}"
            )
            source_ref = (
                Path(".semantier")
                / "skill-overlays"
                / proposal["target_skill"]
                / f"{proposal_id}.SKILL.md"
            ).as_posix()
            overlay_hash = f"sha256:{content_hash(overlay_content)}"
            existing = self._effective_overlay_by_source_ref(
                source_ref,
                connection=connection,
            )
            if existing is not None:
                self._validate_governed_overlay_metadata(existing)
                return existing
            prior = connection.execute(
                sql_query("""
                SELECT * FROM effective_skill_versions
                WHERE workspace_id = ? AND skill_name = ?
                ORDER BY activated_at DESC, effective_id DESC LIMIT 1
                """,'\n                SELECT * FROM effective_skill_versions\n                WHERE workspace_id = %s AND skill_name = %s\n                ORDER BY activated_at DESC, effective_id DESC LIMIT 1\n                '),
                (self.workspace_id, proposal["target_skill"]),
            ).fetchone()
            expected_base = prior["skill_ref"] if prior else CANONICAL_JOURNEY_REF
            if proposal["base_skill_ref"] != expected_base:
                raise ValueError(
                    "skill patch proposal base transition does not match the effective skill"
                )
            overlay_path = self.workspace_root / source_ref
            overlay_path.parent.mkdir(parents=True, exist_ok=True)
            if overlay_path.exists():
                raise ValueError(
                    "partial overlay artifact exists without activation row"
                )
            temporary_path = overlay_path.with_name(
                f".{proposal_id}.{uuid.uuid4().hex}.tmp"
            )
            try:
                temporary_path.write_text(overlay_content, encoding="utf-8")
                effective = self._persist_effective_skill_version(
                    skill_name=proposal["target_skill"],
                    skill_ref=skill_ref,
                    semantic_version=semantic_version,
                    content_hash_value=overlay_hash,
                    source_kind="governed_workspace_overlay",
                    source_ref=source_ref,
                    activated_by=activated_by,
                    supersedes_skill_ref=proposal["base_skill_ref"],
                    rollback_to_skill_ref=proposal["base_skill_ref"],
                    activation_key=proposal_id,
                    connection=connection,
                )
                temporary_path.replace(overlay_path)
                return effective
            finally:
                if temporary_path.exists():
                    temporary_path.unlink()

    def create_effective_skill_version(
        self,
        *,
        skill_name: str,
        skill_ref: str,
        semantic_version: str,
        content_hash_value: str,
        source_kind: str,
        source_ref: str,
        activated_by: str,
        supersedes_skill_ref: str | None,
        rollback_to_skill_ref: str | None,
    ) -> dict[str, Any]:
        if source_kind == "governed_workspace_overlay":
            raise ValueError("governed overlays require proposal-bound activation")
        if source_kind != "application_release":
            raise ValueError("unsupported effective skill source_kind")
        record = {
            "skill_name": skill_name,
            "skill_ref": skill_ref,
            "semantic_version": semantic_version,
            "content_hash": content_hash_value,
            "source_kind": source_kind,
            "source_ref": source_ref,
        }
        self._validate_application_release_metadata(record)
        for linked_ref in (supersedes_skill_ref, rollback_to_skill_ref):
            if linked_ref is not None:
                self._validate_new_learning_base(
                    skill_name=skill_name,
                    skill_ref=linked_ref,
                )
        return self._persist_effective_skill_version(
            skill_name=skill_name,
            skill_ref=skill_ref,
            semantic_version=semantic_version,
            content_hash_value=content_hash_value,
            source_kind=source_kind,
            source_ref=source_ref,
            activated_by=activated_by,
            supersedes_skill_ref=supersedes_skill_ref,
            rollback_to_skill_ref=rollback_to_skill_ref,
        )

    def import_historical_effective_skill_version(
        self,
        *,
        skill_ref: str,
        imported_by: str,
    ) -> dict[str, Any]:
        """Import one exact archived release for compatibility reads only."""
        if not imported_by:
            raise ValueError("imported_by is required")
        historical = resolve_historical_ref(skill_ref)
        return self._persist_effective_skill_version(
            skill_name=historical.name,
            skill_ref=historical.skill_ref,
            semantic_version=historical.skill_ref.split("@", 1)[1],
            content_hash_value=historical.skill_md_sha256,
            source_kind="application_release",
            source_ref=historical.source_ref,
            activated_by=imported_by,
            supersedes_skill_ref=None,
            rollback_to_skill_ref=None,
        )

    def _validate_application_release_metadata(self, record: dict[str, Any]) -> None:
        try:
            registered = select_for_new_run(skill_ref=record["skill_ref"])
        except SkillRegistryError as exc:
            raise ValueError(
                f"effective application release is not selectable: {record['skill_ref']}"
            ) from exc
        if registered is None or registered.name != record["skill_name"]:
            raise ValueError("effective application release identity is invalid")
        expected = {
            "semantic_version": registered.skill_ref.split("@", 1)[1],
            "content_hash": registered.skill_md_sha256,
            "source_ref": registered.source_ref,
        }
        if any(record[field] != value for field, value in expected.items()):
            raise ValueError(
                "effective application release metadata must match the registry"
            )

    def _validate_governed_overlay_metadata(self, record: dict[str, Any]) -> None:
        skill_name = record["skill_name"]
        if skill_name != CANONICAL_JOURNEY_NAME:
            raise ValueError("governed overlay must target the canonical journey")
        expected_prefix = f"{skill_name}@{record['semantic_version']}+overlay."
        if (
            not record["skill_ref"].startswith(expected_prefix)
            or record["skill_ref"] == expected_prefix
        ):
            raise ValueError("governed overlay identity metadata is invalid")
        proposal_id = record["skill_ref"][len(expected_prefix) :]
        try:
            proposal = self.get_skill_patch_proposal(proposal_id)
        except KeyError as exc:
            raise ValueError(
                "governed overlay is not bound to a skill patch proposal"
            ) from exc
        if proposal["target_skill"] != skill_name:
            raise ValueError("governed overlay proposal target does not match")
        self._validate_skill_patch_proposal_integrity(proposal)
        if proposal["review_state"] not in {"human_accepted", "human_edited"}:
            raise ValueError("governed overlay proposal is not human-reviewed")
        reviewed_semantic_version = proposal["reviewed_semantic_version"]
        reviewed_overlay_content = proposal["reviewed_overlay_content"]
        reviewed_result_hash = proposal["reviewed_result_hash"]
        if (
            not reviewed_semantic_version
            or not reviewed_overlay_content
            or not reviewed_result_hash
        ):
            raise ValueError("governed overlay lacks an exact reviewed result")
        if reviewed_result_hash != _reviewed_result_hash(
            reviewed_semantic_version,
            reviewed_overlay_content,
        ):
            raise ValueError("governed overlay reviewed result integrity failed")
        _validate_overlay_frontmatter(
            semantic_version=reviewed_semantic_version,
            overlay_content=reviewed_overlay_content,
        )
        if record["semantic_version"] != reviewed_semantic_version:
            raise ValueError("governed overlay semantic identity is not reviewed")
        if record["content_hash"] != f"sha256:{content_hash(reviewed_overlay_content)}":
            raise ValueError(
                "governed overlay content is not the exact reviewed result"
            )
        original_transition = (
            record.get("supersedes_skill_ref") == proposal["base_skill_ref"]
            and record.get("rollback_to_skill_ref") == proposal["base_skill_ref"]
        )
        if not original_transition:
            if record.get("rollback_to_skill_ref") is not None:
                raise ValueError(
                    "governed overlay proposal base transition does not match"
                )
            with self._connect() as connection:
                original = connection.execute(
                    sql_query("""
                    SELECT 1 FROM effective_skill_versions
                    WHERE workspace_id = ? AND skill_name = ? AND skill_ref = ?
                      AND semantic_version = ? AND content_hash = ?
                      AND source_kind = 'governed_workspace_overlay'
                      AND source_ref = ? AND supersedes_skill_ref = ?
                      AND rollback_to_skill_ref = ?
                    LIMIT 1
                    ""","\n                    SELECT 1 FROM effective_skill_versions\n                    WHERE workspace_id = %s AND skill_name = %s AND skill_ref = %s\n                      AND semantic_version = %s AND content_hash = %s\n                      AND source_kind = 'governed_workspace_overlay'\n                      AND source_ref = %s AND supersedes_skill_ref = %s\n                      AND rollback_to_skill_ref = %s\n                    LIMIT 1\n                    "),
                    (
                        self.workspace_id,
                        skill_name,
                        record["skill_ref"],
                        record["semantic_version"],
                        record["content_hash"],
                        record["source_ref"],
                        proposal["base_skill_ref"],
                        proposal["base_skill_ref"],
                    ),
                ).fetchone()
            if original is None:
                raise ValueError(
                    "governed overlay rollback lacks a proposal-bound activation"
                )
        source_ref = Path(record["source_ref"])
        if source_ref.is_absolute():
            raise ValueError("governed overlay source_ref must be workspace-relative")
        source_path = (self.workspace_root / source_ref).resolve()
        overlay_root = (
            self.workspace_root / ".semantier" / "skill-overlays" / skill_name
        ).resolve()
        try:
            source_path.relative_to(overlay_root)
        except ValueError as exc:
            raise ValueError(
                "governed overlay source_ref is outside its governed root"
            ) from exc
        if not source_path.is_file():
            raise ValueError("governed overlay source_ref does not exist")
        normalized_ref = source_path.relative_to(self.workspace_root).as_posix()
        if normalized_ref != source_ref.as_posix():
            raise ValueError("governed overlay source_ref metadata is not canonical")
        expected_source_ref = (
            Path(".semantier")
            / "skill-overlays"
            / skill_name
            / f"{proposal_id}.SKILL.md"
        ).as_posix()
        if source_ref.as_posix() != expected_source_ref:
            raise ValueError("governed overlay source_ref is not proposal-owned")
        actual_hash = f"sha256:{content_hash(source_path.read_text(encoding='utf-8'))}"
        if record["content_hash"] != actual_hash:
            raise ValueError("governed overlay content hash metadata does not match")

    def resolve_governed_overlay_base(
        self, *, skill_name: str, skill_ref: str
    ) -> dict[str, Any] | None:
        record = self.effective_skill_version_by_ref(
            skill_name=skill_name,
            skill_ref=skill_ref,
        )
        if record is None or record["source_kind"] != "governed_workspace_overlay":
            return None
        self._validate_governed_overlay_metadata(record)
        return record

    def _effective_overlay_by_source_ref(
        self,
        source_ref: str,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> dict[str, Any] | None:
        owns_connection = connection is None
        active_connection = connection or self._connect()
        try:
            row = active_connection.execute(
                sql_query("""
                SELECT * FROM effective_skill_versions
                WHERE workspace_id = ? AND source_kind = 'governed_workspace_overlay'
                  AND source_ref = ?
                ORDER BY activated_at DESC, effective_id DESC LIMIT 1
                ""","\n                SELECT * FROM effective_skill_versions\n                WHERE workspace_id = %s AND source_kind = 'governed_workspace_overlay'\n                  AND source_ref = %s\n                ORDER BY activated_at DESC, effective_id DESC LIMIT 1\n                "),
                (self.workspace_id, source_ref),
            ).fetchone()
        finally:
            if owns_connection:
                active_connection.close()
        if row is None:
            return None
        return self._effective_skill_version_from_row(row)

    def rollback_effective_skill_version(
        self,
        *,
        skill_name: str,
        activated_by: str,
    ) -> dict[str, Any]:
        if not activated_by:
            raise ValueError("activated_by is required")
        current = self.latest_effective_skill_version(skill_name=skill_name)
        if current is None or not current.get("rollback_to_skill_ref"):
            raise ValueError("no rollback target is available")
        if current["source_kind"] == "governed_workspace_overlay":
            self._validate_governed_overlay_metadata(current)
        else:
            self._validate_application_release_metadata(current)
        rollback_ref = current["rollback_to_skill_ref"]
        target = self.effective_skill_version_by_ref(
            skill_name=skill_name,
            skill_ref=rollback_ref,
        )
        if target is None:
            try:
                registered = select_for_new_run(skill_ref=rollback_ref)
            except SkillRegistryError as exc:
                raise ValueError(
                    "rollback target is not eligible for a new effective write"
                ) from exc
            if registered is None or registered.name != skill_name:
                raise ValueError("rollback target is not a registered skill artifact")
            target = {
                "skill_name": registered.name,
                "skill_ref": registered.skill_ref,
                "semantic_version": registered.skill_ref.split("@", 1)[1],
                "content_hash": registered.skill_md_sha256,
                "source_kind": "application_release",
                "source_ref": registered.source_ref,
            }
        elif target["source_kind"] == "governed_workspace_overlay":
            self._validate_governed_overlay_metadata(target)
        else:
            self._validate_application_release_metadata(target)
        return self._persist_effective_skill_version(
            skill_name=target["skill_name"],
            skill_ref=target["skill_ref"],
            semantic_version=target["semantic_version"],
            content_hash_value=target["content_hash"],
            source_kind=target["source_kind"],
            source_ref=target["source_ref"],
            activated_by=activated_by,
            supersedes_skill_ref=current["skill_ref"],
            rollback_to_skill_ref=None,
        )

    def _persist_effective_skill_version(
        self,
        *,
        skill_name: str,
        skill_ref: str,
        semantic_version: str,
        content_hash_value: str,
        source_kind: str,
        source_ref: str,
        activated_by: str,
        supersedes_skill_ref: str | None,
        rollback_to_skill_ref: str | None,
        activation_key: str | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> dict[str, Any]:
        if connection is None:
            with self._lock, self._connect() as owned_connection:
                return self._persist_effective_skill_version(
                    skill_name=skill_name,
                    skill_ref=skill_ref,
                    semantic_version=semantic_version,
                    content_hash_value=content_hash_value,
                    source_kind=source_kind,
                    source_ref=source_ref,
                    activated_by=activated_by,
                    supersedes_skill_ref=supersedes_skill_ref,
                    rollback_to_skill_ref=rollback_to_skill_ref,
                    activation_key=activation_key,
                    connection=owned_connection,
                )
        payload = {
            "schema_version": EFFECTIVE_SKILL_VERSION_SCHEMA_VERSION,
            "effective_id": f"esv_{uuid.uuid4().hex}",
            "skill_name": skill_name,
            "skill_ref": skill_ref,
            "semantic_version": semantic_version,
            "content_hash": content_hash_value,
            "source_kind": source_kind,
            "source_ref": source_ref,
            "activated_by": activated_by,
            "activated_at": _utc_now(),
            "supersedes_skill_ref": supersedes_skill_ref,
            "rollback_to_skill_ref": rollback_to_skill_ref,
            "activation_key": activation_key,
        }
        payload["effective_hash"] = f"sha256:{content_hash(payload)}"
        owns_connection = connection is None
        active_connection = connection or self._connect()
        try:
            if owns_connection and enabled():begin_write(active_connection)
            active_connection.execute(
                sql_query("""
                INSERT INTO effective_skill_versions (
                    effective_id, workspace_id, schema_version, skill_name,
                    skill_ref, semantic_version, content_hash, source_kind,
                    source_ref, activated_by, activated_at, supersedes_skill_ref,
                    rollback_to_skill_ref, activation_key, effective_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,'\n                INSERT INTO effective_skill_versions (\n                    effective_id, workspace_id, schema_version, skill_name,\n                    skill_ref, semantic_version, content_hash, source_kind,\n                    source_ref, activated_by, activated_at, supersedes_skill_ref,\n                    rollback_to_skill_ref, activation_key, effective_hash\n                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)\n                '),
                (
                    payload["effective_id"],
                    self.workspace_id,
                    payload["schema_version"],
                    payload["skill_name"],
                    payload["skill_ref"],
                    payload["semantic_version"],
                    payload["content_hash"],
                    payload["source_kind"],
                    payload["source_ref"],
                    payload["activated_by"],
                    payload["activated_at"],
                    payload["supersedes_skill_ref"],
                    payload["rollback_to_skill_ref"],
                    payload["activation_key"],
                    payload["effective_hash"],
                ),
            )
            if owns_connection:
                active_connection.commit()
        finally:
            if owns_connection:
                active_connection.close()
        return payload

    def latest_effective_skill_version(
        self, *, skill_name: str = "graph-schema-discovery"
    ) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                sql_query("""
                SELECT * FROM effective_skill_versions
                WHERE workspace_id = ? AND skill_name = ?
                ORDER BY activated_at DESC, effective_id DESC LIMIT 1
                """,'\n                SELECT * FROM effective_skill_versions\n                WHERE workspace_id = %s AND skill_name = %s\n                ORDER BY activated_at DESC, effective_id DESC LIMIT 1\n                '),
                (self.workspace_id, skill_name),
            ).fetchone()
        if row is None:
            return None
        return self._effective_skill_version_from_row(row)

    @staticmethod
    def _effective_skill_version_from_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "schema_version": row["schema_version"],
            "effective_id": row["effective_id"],
            "skill_name": row["skill_name"],
            "skill_ref": row["skill_ref"],
            "semantic_version": row["semantic_version"],
            "content_hash": row["content_hash"],
            "source_kind": row["source_kind"],
            "source_ref": row["source_ref"],
            "activated_by": row["activated_by"],
            "activated_at": row["activated_at"],
            "supersedes_skill_ref": row["supersedes_skill_ref"],
            "rollback_to_skill_ref": row["rollback_to_skill_ref"],
            "activation_key": row["activation_key"],
            "effective_hash": row["effective_hash"],
        }

    def effective_skill_version_by_ref(
        self, *, skill_name: str, skill_ref: str
    ) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                sql_query("""
                SELECT * FROM effective_skill_versions
                WHERE workspace_id = ? AND skill_name = ? AND skill_ref = ?
                ORDER BY activated_at DESC, effective_id DESC LIMIT 1
                """,'\n                SELECT * FROM effective_skill_versions\n                WHERE workspace_id = %s AND skill_name = %s AND skill_ref = %s\n                ORDER BY activated_at DESC, effective_id DESC LIMIT 1\n                '),
                (self.workspace_id, skill_name, skill_ref),
            ).fetchone()
        if row is None:
            return None
        return self._effective_skill_version_from_row(row)
