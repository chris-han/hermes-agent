"""Frozen runtime detection snapshots for MVL replay binding checks."""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from storage.hosted_postgres import enabled,connect_authority,mapped_store,require_tables,sql_query,begin_write,integrity_errors
from eos.artifact_hashing import canonical_json, content_hash


RUNTIME_DETECTION_SNAPSHOT_SCHEMA_VERSION = "runtime_detection_snapshot.v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _loads(value: str) -> Any:
    return json.loads(value) if value else None


@dataclass(frozen=True)
class RuntimeDetectionSnapshot:
    schema_version: str
    run_id: str
    experiment_id: str
    input_source_ref: str
    input_source_hash: str
    profile_ref: str
    profile_version: str
    graph_version_ref: str
    graph_checksum: str
    ontology_version_ref: str | None
    ontology_checksum: str | None
    skill_ref: str
    semantica_commit: str
    matcher_version: str
    findings_hash: str
    finding_refs: list[str]
    findings: list[dict[str, Any]]
    created_at: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "input_source_ref": self.input_source_ref,
            "input_source_hash": self.input_source_hash,
            "profile_ref": self.profile_ref,
            "profile_version": self.profile_version,
            "graph_version_ref": self.graph_version_ref,
            "graph_checksum": self.graph_checksum,
            "ontology_version_ref": self.ontology_version_ref,
            "ontology_checksum": self.ontology_checksum,
            "skill_ref": self.skill_ref,
            "semantica_commit": self.semantica_commit,
            "matcher_version": self.matcher_version,
            "findings_hash": self.findings_hash,
            "finding_refs": self.finding_refs,
            "findings": self.findings,
            "created_at": self.created_at,
        }


class RuntimeDetectionSnapshotStore:
    """Append-only frozen run/finding envelope storage."""

    _process_locks: dict[Path, threading.RLock] = {}
    _process_locks_guard = threading.Lock()

    def __init__(self, workspace_root: str | Path, *, workspace_id: str) -> None:
        if not workspace_id:
            raise ValueError("workspace_id is required")
        self.workspace_root = Path(workspace_root).resolve()
        self.workspace_id = workspace_id
        self.root = self.workspace_root / ".semantier" / "runtime-detection"
        if not enabled():self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "snapshots.sqlite3"
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
            with self._connect() as connection:require_tables(connection,'runtime_detection_snapshots')
            return
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS runtime_detection_snapshots (
                    run_id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    schema_version TEXT NOT NULL,
                    experiment_id TEXT NOT NULL,
                    input_source_ref TEXT NOT NULL,
                    input_source_hash TEXT NOT NULL,
                    profile_ref TEXT NOT NULL,
                    profile_version TEXT NOT NULL,
                    graph_version_ref TEXT NOT NULL,
                    graph_checksum TEXT NOT NULL,
                    ontology_version_ref TEXT,
                    ontology_checksum TEXT,
                    skill_ref TEXT NOT NULL,
                    semantica_commit TEXT NOT NULL,
                    matcher_version TEXT NOT NULL,
                    findings_hash TEXT NOT NULL,
                    finding_refs_json TEXT NOT NULL,
                    findings_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    def create_snapshot(
        self,
        *,
        run_id: str,
        experiment_id: str,
        input_source_ref: str,
        input_source_hash: str,
        profile_ref: str,
        profile_version: str,
        graph_version_ref: str,
        graph_checksum: str,
        skill_ref: str,
        semantica_commit: str,
        matcher_version: str,
        findings: list[dict[str, Any]],
        ontology_version_ref: str | None = None,
        ontology_checksum: str | None = None,
    ) -> RuntimeDetectionSnapshot:
        required = {
            "run_id": run_id,
            "experiment_id": experiment_id,
            "input_source_ref": input_source_ref,
            "input_source_hash": input_source_hash,
            "profile_ref": profile_ref,
            "profile_version": profile_version,
            "graph_version_ref": graph_version_ref,
            "graph_checksum": graph_checksum,
            "skill_ref": skill_ref,
            "semantica_commit": semantica_commit,
            "matcher_version": matcher_version,
        }
        missing = sorted(name for name, value in required.items() if not value)
        if missing:
            raise ValueError(f"runtime detection snapshot missing fields: {missing}")
        finding_refs = [str(finding["match_id"]) for finding in findings]
        frozen_findings = json.loads(canonical_json(findings))
        snapshot = RuntimeDetectionSnapshot(
            schema_version=RUNTIME_DETECTION_SNAPSHOT_SCHEMA_VERSION,
            run_id=run_id,
            experiment_id=experiment_id,
            input_source_ref=input_source_ref,
            input_source_hash=input_source_hash,
            profile_ref=profile_ref,
            profile_version=profile_version,
            graph_version_ref=graph_version_ref,
            graph_checksum=graph_checksum,
            ontology_version_ref=ontology_version_ref,
            ontology_checksum=ontology_checksum,
            skill_ref=skill_ref,
            semantica_commit=semantica_commit,
            matcher_version=matcher_version,
            findings_hash=f"sha256:{content_hash(frozen_findings)}",
            finding_refs=finding_refs,
            findings=frozen_findings,
            created_at=_utc_now(),
        )
        with self._lock, self._connect() as connection:
            if enabled():begin_write(connection)
            connection.execute(
                sql_query("""
                INSERT INTO runtime_detection_snapshots VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,'\n                INSERT INTO runtime_detection_snapshots VALUES (\n                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s\n                )\n                '),
                (
                    snapshot.run_id,
                    self.workspace_id,
                    snapshot.schema_version,
                    snapshot.experiment_id,
                    snapshot.input_source_ref,
                    snapshot.input_source_hash,
                    snapshot.profile_ref,
                    snapshot.profile_version,
                    snapshot.graph_version_ref,
                    snapshot.graph_checksum,
                    snapshot.ontology_version_ref,
                    snapshot.ontology_checksum,
                    snapshot.skill_ref,
                    snapshot.semantica_commit,
                    snapshot.matcher_version,
                    snapshot.findings_hash,
                    canonical_json(snapshot.finding_refs),
                    canonical_json(snapshot.findings),
                    snapshot.created_at,
                ),
            )
        return snapshot

    def get_snapshot(self, run_id: str) -> RuntimeDetectionSnapshot:
        with self._connect() as connection:
            row = connection.execute(
                sql_query("""
                SELECT * FROM runtime_detection_snapshots
                WHERE workspace_id = ? AND run_id = ?
                """,'\n                SELECT * FROM runtime_detection_snapshots\n                WHERE workspace_id = %s AND run_id = %s\n                '),
                (self.workspace_id, run_id),
            ).fetchone()
        if row is None:
            raise KeyError(f"runtime detection snapshot not found: {run_id}")
        findings = _loads(row["findings_json"])
        expected_hash = f"sha256:{content_hash(findings)}"
        if expected_hash != row["findings_hash"]:
            raise ValueError("runtime detection snapshot findings hash mismatch")
        return RuntimeDetectionSnapshot(
            schema_version=row["schema_version"],
            run_id=row["run_id"],
            experiment_id=row["experiment_id"],
            input_source_ref=row["input_source_ref"],
            input_source_hash=row["input_source_hash"],
            profile_ref=row["profile_ref"],
            profile_version=row["profile_version"],
            graph_version_ref=row["graph_version_ref"],
            graph_checksum=row["graph_checksum"],
            ontology_version_ref=row["ontology_version_ref"],
            ontology_checksum=row["ontology_checksum"],
            skill_ref=row["skill_ref"],
            semantica_commit=row["semantica_commit"],
            matcher_version=row["matcher_version"],
            findings_hash=row["findings_hash"],
            finding_refs=_loads(row["finding_refs_json"]),
            findings=findings,
            created_at=row["created_at"],
        )

