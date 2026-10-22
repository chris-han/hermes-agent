"""W6 migration helpers for candidate-only reference concept runs.

The migration surface is append-only: it computes suspect candidates and
comparison artifacts without mutating accepted/activated releases.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any
from uuid import uuid4

from contracts.knowledge_feature_pack import content_hash
from storage.knowledge_extraction_store import KnowledgeExtractionStore


_DELIMITER_TOKENS = ("、", "，", ",", "；", ";")


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _ensure_migration_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS contextgraph_candidate_run_migration_assessments (
          assessment_id TEXT PRIMARY KEY,
          old_extraction_run_id TEXT NOT NULL,
          new_extraction_run_id TEXT,
          organization_id TEXT NOT NULL,
          workspace_id TEXT,
          assessment_json TEXT NOT NULL,
          content_hash TEXT NOT NULL,
          created_at TEXT NOT NULL,
          FOREIGN KEY(old_extraction_run_id) REFERENCES knowledge_builder_extraction_runs(extraction_run_id),
          FOREIGN KEY(new_extraction_run_id) REFERENCES knowledge_builder_extraction_runs(extraction_run_id)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_contextgraph_candidate_run_migration_old
          ON contextgraph_candidate_run_migration_assessments(old_extraction_run_id, created_at)
        """
    )


def detect_suspect_reference_concepts(
    *,
    db_path: str | Path,
    extraction_run_id: str,
    advisory_label_code_points: int = 40,
) -> list[dict[str, Any]]:
    """Detect candidate concepts likely produced by over-broad boundaries."""

    store = KnowledgeExtractionStore(db_path)
    reference = store.list_reference_concepts(extraction_run_id=extraction_run_id)
    suspects: list[dict[str, Any]] = []
    for concept in reference["concepts"]:
        label = str(concept.get("canonical_label") or "")
        reasons: list[str] = []
        delimiter_hits = sum(label.count(token) for token in _DELIMITER_TOKENS)
        if delimiter_hits > 0:
            reasons.append("multi_delimiter")
        if len(label) >= advisory_label_code_points:
            reasons.append("advisory_label_length")
        if not reasons:
            continue
        suspects.append(
            {
                "concept_id": concept["concept_id"],
                "canonical_label": label,
                "reasons": reasons,
                "delimiter_hits": delimiter_hits,
                "label_code_points": len(label),
                "grounding_state": concept.get("grounding_state"),
            }
        )
    return suspects


def build_candidate_run_comparison(
    *, db_path: str | Path, old_extraction_run_id: str, new_extraction_run_id: str
) -> dict[str, Any]:
    """Build deterministic old/new comparison artifact for migration review."""

    store = KnowledgeExtractionStore(db_path)
    old_reference = store.list_reference_concepts(extraction_run_id=old_extraction_run_id)
    new_reference = store.list_reference_concepts(extraction_run_id=new_extraction_run_id)
    old_labels = {
        str(item.get("canonical_label") or ""): str(item["concept_id"])
        for item in old_reference["concepts"]
    }
    new_labels = {
        str(item.get("canonical_label") or ""): str(item["concept_id"])
        for item in new_reference["concepts"]
    }

    exact_matches = sorted(label for label in old_labels if label in new_labels)
    unmatched_old = sorted(label for label in old_labels if label not in new_labels)
    unmatched_new = sorted(label for label in new_labels if label not in old_labels)

    old_suspects = detect_suspect_reference_concepts(
        db_path=db_path,
        extraction_run_id=old_extraction_run_id,
    )
    new_suspects = detect_suspect_reference_concepts(
        db_path=db_path,
        extraction_run_id=new_extraction_run_id,
    )

    return {
        "schema_version": "contextgraph_candidate_run_migration_comparison.v1",
        "old_extraction_run_id": old_extraction_run_id,
        "new_extraction_run_id": new_extraction_run_id,
        "old_concept_count": len(old_reference["concepts"]),
        "new_concept_count": len(new_reference["concepts"]),
        "old_suspect_count": len(old_suspects),
        "new_suspect_count": len(new_suspects),
        "exact_label_matches": exact_matches,
        "unmatched_old_labels": unmatched_old,
        "unmatched_new_labels": unmatched_new,
        "old_suspects": old_suspects,
        "new_suspects": new_suspects,
    }


def record_candidate_run_migration_assessment(
    *,
    db_path: str | Path,
    old_extraction_run_id: str,
    new_extraction_run_id: str,
    comparison_artifact: dict[str, Any],
    assessment_reason: str,
) -> dict[str, Any]:
    """Persist an append-only migration assessment record."""
    from storage.hosted_postgres import enabled, HostedStorageUnavailable
    if enabled():
        raise HostedStorageUnavailable('offline candidate-run reassessment is local-only')

    store = KnowledgeExtractionStore(db_path)
    old_run = store.get_extraction_run(old_extraction_run_id)
    new_run = store.get_extraction_run(new_extraction_run_id)
    if old_run is None or new_run is None:
        raise ValueError("migration assessment requires existing old/new extraction runs")
    if old_run["organization_id"] != new_run["organization_id"]:
        raise ValueError("old/new extraction runs must belong to one organization")
    if old_run.get("workspace_id") != new_run.get("workspace_id"):
        raise ValueError("old/new extraction runs must belong to one workspace")

    payload = {
        "schema_version": "contextgraph_candidate_run_migration_assessment.v1",
        "assessment_id": f"migration_assessment_{uuid4().hex}",
        "old_extraction_run_id": old_extraction_run_id,
        "new_extraction_run_id": new_extraction_run_id,
        "organization_id": old_run["organization_id"],
        "workspace_id": old_run.get("workspace_id"),
        "assessment_reason": assessment_reason,
        "comparison_artifact": comparison_artifact,
        "created_at": _now(),
    }
    digest = content_hash(payload)

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        _ensure_migration_schema(conn)
        conn.execute(
            """
            INSERT INTO contextgraph_candidate_run_migration_assessments (
              assessment_id, old_extraction_run_id, new_extraction_run_id,
              organization_id, workspace_id, assessment_json, content_hash, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["assessment_id"],
                old_extraction_run_id,
                new_extraction_run_id,
                old_run["organization_id"],
                old_run.get("workspace_id"),
                json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                digest,
                payload["created_at"],
            ),
        )
        conn.commit()

    return {**payload, "content_hash": digest}


def list_candidate_run_migration_assessments(
    *, db_path: str | Path, old_extraction_run_id: str
) -> list[dict[str, Any]]:
    """List append-only migration assessments for one old extraction run."""
    from storage.hosted_postgres import enabled, HostedStorageUnavailable
    if enabled():
        raise HostedStorageUnavailable('offline candidate-run reassessment is local-only')

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        _ensure_migration_schema(conn)
        rows = conn.execute(
            """
            SELECT assessment_json, content_hash
            FROM contextgraph_candidate_run_migration_assessments
            WHERE old_extraction_run_id=?
            ORDER BY created_at, assessment_id
            """,
            (old_extraction_run_id,),
        ).fetchall()
    assessments: list[dict[str, Any]] = []
    for row in rows:
        payload = json.loads(row["assessment_json"])
        payload["content_hash"] = row["content_hash"]
        assessments.append(payload)
    return assessments
