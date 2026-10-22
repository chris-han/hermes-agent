"""Append-only canonical persistence for evaluation-control-plane records."""

from __future__ import annotations

from storage import hosted_postgres as hosted

import json
import sqlite3
import threading
from dataclasses import is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eos.evaluation_control_plane import (
    EvaluationEnvironmentManifest,
    EvaluationSubjectManifestV2,
)


class EvaluationControlPlaneIntegrityError(ValueError):
    pass


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _record(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_record"):
        return dict(value.to_record())
    if isinstance(value, dict):
        return dict(value)
    if is_dataclass(value):
        raise TypeError("canonical dataclasses must expose to_record()")
    raise TypeError(f"unsupported canonical record: {type(value)!r}")


def _aware_iso(value: str, field: str) -> str:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise EvaluationControlPlaneIntegrityError(f"{field} must be timezone-aware")
    return value


class EvaluationControlPlaneStore:
    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        if hosted.enabled():
            # This owner is used by the global evaluation API and by the
            # governed workspace ontology publisher. Resolve the approved owner
            # before connecting; a database outage never selects another store.
            try:
                hosted.mapped_store('eos', self._db_path)
                owner = 'eos'
            except hosted.HostedStorageUnavailable:
                hosted.mapped_store('standalone', self._db_path)
                owner = 'standalone'
            self._conn = hosted.connect_authority(owner, self._db_path)
        else:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._init_schema()

    def close(self) -> None:
        self._conn.close()

    def _init_schema(self) -> None:
        if hosted.enabled():
            hosted.require_tables(self._conn, *('evaluation_control_plane_subject_manifests', 'evaluation_control_plane_environment_manifests', 'evaluation_control_plane_runs', 'evaluation_control_plane_evidence', 'evaluation_control_plane_waivers'))
            self._conn.commit()
            return
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS evaluation_control_plane_subject_manifests (
                  content_hash TEXT PRIMARY KEY,
                  subject_id TEXT NOT NULL,
                  schema_version TEXT NOT NULL,
                  record_json TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_ecp_subject_id
                  ON evaluation_control_plane_subject_manifests(subject_id);

                CREATE TABLE IF NOT EXISTS evaluation_control_plane_environment_manifests (
                  environment_manifest_ref TEXT PRIMARY KEY,
                  content_hash TEXT NOT NULL UNIQUE,
                  runtime_commit TEXT NOT NULL,
                  environment_fingerprint TEXT NOT NULL,
                  schema_version TEXT NOT NULL,
                  record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS evaluation_control_plane_runs (
                  run_id TEXT PRIMARY KEY,
                  benchmark_family_ref TEXT NOT NULL,
                  scenario_ref TEXT NOT NULL,
                  subject_manifest_ref TEXT NOT NULL,
                  environment_manifest_ref TEXT NOT NULL,
                  evaluator_bundle_ref TEXT NOT NULL,
                  validity TEXT NOT NULL,
                  started_at TEXT NOT NULL,
                  completed_at TEXT NULL,
                  retry_of_run_ref TEXT NULL,
                  parent_orchestration_ref TEXT NULL,
                  schema_version TEXT NOT NULL,
                  content_hash TEXT NOT NULL UNIQUE,
                  record_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_ecp_runs_parent
                  ON evaluation_control_plane_runs(parent_orchestration_ref);

                CREATE TABLE IF NOT EXISTS evaluation_control_plane_evidence (
                  evidence_id TEXT PRIMARY KEY,
                  run_ref TEXT NOT NULL UNIQUE,
                  benchmark_family TEXT NOT NULL,
                  certification_result TEXT NOT NULL,
                  signed_result_ref TEXT NOT NULL UNIQUE,
                  schema_version TEXT NOT NULL,
                  content_hash TEXT NOT NULL UNIQUE,
                  record_json TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS evaluation_control_plane_waivers (
                  waiver_id TEXT PRIMARY KEY,
                  scope_ref TEXT NOT NULL,
                  signed_waiver_ref TEXT NOT NULL UNIQUE,
                  expires_at TEXT NULL,
                  content_hash TEXT NOT NULL UNIQUE,
                  record_json TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );
                """
            )
            self._conn.commit()

    def put_subject_manifest(self, record: Any) -> dict[str, Any]:
        payload = _record(record)
        with self._transaction() as connection:
            self._put_subject(connection, payload)
        return self.get_subject_manifest(payload["content_hash"])

    def put_run(self, record: Any) -> dict[str, Any]:
        payload = _record(record)
        with self._transaction() as connection:
            self._put_run(connection, payload)
        return self.get_run(payload["run_id"])

    def put_environment_manifest(self, record: Any) -> dict[str, Any]:
        payload = EvaluationEnvironmentManifest.from_record(_record(record)).to_record()
        with self._transaction() as connection:
            self._put_environment(connection, payload)
        return self.get_environment_manifest(payload["environment_manifest_ref"])

    def put_evidence(self, record: Any) -> dict[str, Any]:
        payload = _record(record)
        with self._transaction() as connection:
            self._put_evidence(connection, payload)
        return self.get_evidence(payload["evidence_id"])

    def put_waiver(self, record: Any) -> dict[str, Any]:
        payload = _record(record)
        created_at = payload.get("approved_at")
        if not created_at:
            raise EvaluationControlPlaneIntegrityError("waiver approved_at is required")
        _aware_iso(str(created_at), "approved_at")
        expires_at = payload.get("expires_at")
        if expires_at is not None:
            _aware_iso(str(expires_at), "expires_at")
        with self._transaction() as connection:
            self._insert_immutable(
                connection,
                table="evaluation_control_plane_waivers",
                id_column="waiver_id",
                identity=str(payload["waiver_id"]),
                content_hash=str(payload["content_hash"]),
                sql=hosted.sql_query("""
                    INSERT INTO evaluation_control_plane_waivers
                      (waiver_id, scope_ref, signed_waiver_ref, expires_at,
                       content_hash, record_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, '\n                    INSERT INTO evaluation_control_plane_waivers\n                      (waiver_id, scope_ref, signed_waiver_ref, expires_at,\n                       content_hash, record_json, created_at)\n                    VALUES (%s, %s, %s, %s, %s, %s, %s)\n                '),
                values=(
                    payload["waiver_id"],
                    payload["scope_ref"],
                    payload["signed_waiver_ref"],
                    expires_at,
                    payload["content_hash"],
                    _json(payload),
                    created_at,
                ),
            )
        return self.get_waiver(str(payload["waiver_id"]))

    def persist_completed_evaluation(
        self,
        subject: Any,
        run: Any,
        evidence: Any,
        environment: Any | None = None,
    ) -> dict[str, dict[str, Any]]:
        subject_payload = _record(subject)
        run_payload = _record(run)
        evidence_payload = _record(evidence)
        environment_payload = (
            EvaluationEnvironmentManifest.from_record(_record(environment)).to_record()
            if environment is not None
            else None
        )
        if not run_payload.get("completed_at"):
            raise EvaluationControlPlaneIntegrityError("completed EvaluationRun requires completed_at")
        if evidence_payload.get("run_ref") != run_payload.get("run_id"):
            raise EvaluationControlPlaneIntegrityError("evidence run_ref does not match run_id")
        if run_payload.get("subject_manifest_ref") not in {
            subject_payload.get("content_hash"),
            subject_payload.get("subject_id"),
        }:
            raise EvaluationControlPlaneIntegrityError(
                "run subject_manifest_ref does not resolve to supplied subject"
            )
        if environment_payload is not None:
            if run_payload.get("environment_manifest_ref") != environment_payload.get(
                "environment_manifest_ref"
            ):
                raise EvaluationControlPlaneIntegrityError(
                    "run environment_manifest_ref does not resolve to supplied environment"
                )
            if subject_payload.get("schema_version") != "evaluation_subject_manifest.v2":
                raise EvaluationControlPlaneIntegrityError(
                    "persisted environment manifest requires subject manifest v2"
                )
            if (
                subject_payload.get("runtime_commit"),
                subject_payload.get("environment_manifest_ref"),
                subject_payload.get("environment_fingerprint"),
            ) != (
                environment_payload.get("runtime_commit"),
                environment_payload.get("environment_manifest_ref"),
                environment_payload.get("environment_fingerprint"),
            ):
                raise EvaluationControlPlaneIntegrityError(
                    "subject and environment execution pins differ"
                )
        with self._transaction() as connection:
            if environment_payload is not None:
                self._put_environment(connection, environment_payload)
            self._put_subject(connection, subject_payload)
            self._put_run(connection, run_payload)
            self._put_evidence(connection, evidence_payload)
        result = {
            "subject": self.get_subject_manifest(str(subject_payload["content_hash"])),
            "run": self.get_run(str(run_payload["run_id"])),
            "evidence": self.get_evidence(str(evidence_payload["evidence_id"])),
        }
        if environment_payload is not None:
            result["environment"] = self.get_environment_manifest(
                str(environment_payload["environment_manifest_ref"])
            )
        return result

    def get_run(self, run_id: str) -> dict[str, Any]:
        return self._get_one("evaluation_control_plane_runs", "run_id", run_id)

    def get_environment_manifest(self, environment_ref: str) -> dict[str, Any]:
        row = self._conn.execute(
            hosted.sql_query("""SELECT record_json FROM evaluation_control_plane_environment_manifests
            WHERE environment_manifest_ref = ? OR content_hash = ?""", 'SELECT record_json FROM evaluation_control_plane_environment_manifests\n            WHERE environment_manifest_ref = %s OR content_hash = %s'),
            (environment_ref, environment_ref),
        ).fetchone()
        if row is None:
            raise KeyError(environment_ref)
        payload = json.loads(row["record_json"])
        try:
            return EvaluationEnvironmentManifest.from_record(payload).to_record()
        except (TypeError, ValueError, KeyError) as exc:
            raise EvaluationControlPlaneIntegrityError(str(exc)) from exc

    def get_evidence(self, evidence_id: str) -> dict[str, Any]:
        return self._get_one("evaluation_control_plane_evidence", "evidence_id", evidence_id)

    def get_subject_manifest(self, subject_ref: str) -> dict[str, Any]:
        row = self._conn.execute(
            hosted.sql_query("""
            SELECT schema_version,record_json FROM evaluation_control_plane_subject_manifests
            WHERE content_hash = ? OR subject_id = ?
            ORDER BY created_at DESC, content_hash DESC LIMIT 1
            """, '\n            SELECT schema_version,record_json FROM evaluation_control_plane_subject_manifests\n            WHERE content_hash = %s OR subject_id = %s\n            ORDER BY created_at DESC, content_hash DESC LIMIT 1\n            '),
            (subject_ref, subject_ref),
        ).fetchone()
        if row is None:
            raise KeyError(subject_ref)
        payload = json.loads(row["record_json"])
        if row["schema_version"] == "evaluation_subject_manifest.v2":
            try:
                return EvaluationSubjectManifestV2.from_record(payload).to_record()
            except (TypeError, ValueError, KeyError) as exc:
                raise EvaluationControlPlaneIntegrityError(str(exc)) from exc
        return payload

    def get_waiver(self, waiver_ref: str) -> dict[str, Any]:
        row = self._conn.execute(
            hosted.sql_query("""
            SELECT record_json FROM evaluation_control_plane_waivers
            WHERE waiver_id = ? OR signed_waiver_ref = ?
            """, '\n            SELECT record_json FROM evaluation_control_plane_waivers\n            WHERE waiver_id = %s OR signed_waiver_ref = %s\n            '),
            (waiver_ref, waiver_ref),
        ).fetchone()
        if row is None:
            raise KeyError(waiver_ref)
        return json.loads(row["record_json"])

    def _get_one(self, table: str, column: str, identity: str) -> dict[str, Any]:
        row = self._conn.execute(
            hosted.sql_query(f"SELECT record_json FROM {table} WHERE {column} = ?", f'SELECT record_json FROM {table} WHERE {column} = %s'), (identity,)
        ).fetchone()
        if row is None:
            raise KeyError(identity)
        return json.loads(row["record_json"])

    def _put_subject(self, connection: sqlite3.Connection, payload: dict[str, Any]) -> None:
        if payload.get("schema_version") == "evaluation_subject_manifest.v2":
            try:
                payload = EvaluationSubjectManifestV2.from_record(payload).to_record()
            except (TypeError, ValueError, KeyError) as exc:
                raise EvaluationControlPlaneIntegrityError(str(exc)) from exc
        created_at = payload.get("created_at") or payload.get("generated_at")
        if created_at is None:
            created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        _aware_iso(str(created_at), "subject created_at")
        self._insert_immutable(
            connection,
            table="evaluation_control_plane_subject_manifests",
            id_column="content_hash",
            identity=str(payload["content_hash"]),
            content_hash=str(payload["content_hash"]),
            sql=hosted.sql_query("""
                INSERT INTO evaluation_control_plane_subject_manifests
                  (content_hash, subject_id, schema_version, record_json, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, '\n                INSERT INTO evaluation_control_plane_subject_manifests\n                  (content_hash, subject_id, schema_version, record_json, created_at)\n                VALUES (%s, %s, %s, %s, %s)\n            '),
            values=(
                payload["content_hash"],
                payload["subject_id"],
                payload["schema_version"],
                _json(payload),
                created_at,
            ),
        )

    def _put_environment(
        self, connection: sqlite3.Connection, payload: dict[str, Any]
    ) -> None:
        try:
            payload = EvaluationEnvironmentManifest.from_record(payload).to_record()
        except (TypeError, ValueError, KeyError) as exc:
            raise EvaluationControlPlaneIntegrityError(str(exc)) from exc
        expected = payload["content_hash"]
        expected_ref = payload["environment_manifest_ref"]
        self._insert_immutable(
            connection,
            table="evaluation_control_plane_environment_manifests",
            id_column="environment_manifest_ref",
            identity=expected_ref,
            content_hash=expected,
            sql=hosted.sql_query("""
                INSERT INTO evaluation_control_plane_environment_manifests
                  (environment_manifest_ref, content_hash, runtime_commit,
                   environment_fingerprint, schema_version, record_json)
                VALUES (?, ?, ?, ?, ?, ?)
            """, '\n                INSERT INTO evaluation_control_plane_environment_manifests\n                  (environment_manifest_ref, content_hash, runtime_commit,\n                   environment_fingerprint, schema_version, record_json)\n                VALUES (%s, %s, %s, %s, %s, %s)\n            '),
            values=(
                expected_ref,
                expected,
                payload["runtime_commit"],
                payload["environment_fingerprint"],
                payload["schema_version"],
                _json(payload),
            ),
        )

    def _put_run(self, connection: sqlite3.Connection, payload: dict[str, Any]) -> None:
        _aware_iso(str(payload["started_at"]), "started_at")
        if payload.get("completed_at") is not None:
            _aware_iso(str(payload["completed_at"]), "completed_at")
        self._insert_immutable(
            connection,
            table="evaluation_control_plane_runs",
            id_column="run_id",
            identity=str(payload["run_id"]),
            content_hash=str(payload["content_hash"]),
            sql=hosted.sql_query("""
                INSERT INTO evaluation_control_plane_runs
                  (run_id, benchmark_family_ref, scenario_ref, subject_manifest_ref,
                   environment_manifest_ref, evaluator_bundle_ref, validity,
                   started_at, completed_at, retry_of_run_ref, parent_orchestration_ref,
                   schema_version, content_hash, record_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, '\n                INSERT INTO evaluation_control_plane_runs\n                  (run_id, benchmark_family_ref, scenario_ref, subject_manifest_ref,\n                   environment_manifest_ref, evaluator_bundle_ref, validity,\n                   started_at, completed_at, retry_of_run_ref, parent_orchestration_ref,\n                   schema_version, content_hash, record_json)\n                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)\n            '),
            values=(
                payload["run_id"],
                payload["benchmark_family_ref"],
                payload["scenario_ref"],
                payload["subject_manifest_ref"],
                payload["environment_manifest_ref"],
                payload["evaluator_bundle_ref"],
                payload["validity"],
                payload["started_at"],
                payload.get("completed_at"),
                payload.get("retry_of_run_ref"),
                payload.get("parent_orchestration_ref"),
                payload["schema_version"],
                payload["content_hash"],
                _json(payload),
            ),
        )

    def _put_evidence(self, connection: sqlite3.Connection, payload: dict[str, Any]) -> None:
        run = connection.execute(
            hosted.sql_query("SELECT 1 FROM evaluation_control_plane_runs WHERE run_id = ?", 'SELECT 1 FROM evaluation_control_plane_runs WHERE run_id = %s'),
            (payload["run_ref"],),
        ).fetchone()
        if run is None:
            raise EvaluationControlPlaneIntegrityError("evidence run_ref is not persisted")
        created_at = payload.get("created_at") or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        _aware_iso(str(created_at), "evidence created_at")
        self._insert_immutable(
            connection,
            table="evaluation_control_plane_evidence",
            id_column="evidence_id",
            identity=str(payload["evidence_id"]),
            content_hash=str(payload["content_hash"]),
            sql=hosted.sql_query("""
                INSERT INTO evaluation_control_plane_evidence
                  (evidence_id, run_ref, benchmark_family, certification_result,
                   signed_result_ref, schema_version, content_hash, record_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, '\n                INSERT INTO evaluation_control_plane_evidence\n                  (evidence_id, run_ref, benchmark_family, certification_result,\n                   signed_result_ref, schema_version, content_hash, record_json, created_at)\n                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)\n            '),
            values=(
                payload["evidence_id"],
                payload["run_ref"],
                payload["benchmark_family"],
                payload["certification_result"],
                payload["signed_result_ref"],
                payload["schema_version"],
                payload["content_hash"],
                _json(payload),
                created_at,
            ),
        )

    def _insert_immutable(
        self,
        connection: sqlite3.Connection,
        *,
        table: str,
        id_column: str,
        identity: str,
        content_hash: str,
        sql: str,
        values: tuple[Any, ...],
    ) -> None:
        row = connection.execute(
            hosted.sql_query(f"SELECT content_hash, record_json FROM {table} WHERE {id_column} = ?", f'SELECT content_hash, record_json FROM {table} WHERE {id_column} = %s'),
            (identity,),
        ).fetchone()
        if row is not None:
            if row["content_hash"] != content_hash:
                raise EvaluationControlPlaneIntegrityError(
                    f"immutable {table} identity has different content"
                )
            return
        try:
            connection.execute(sql, values)
        except hosted.integrity_errors() as exc:
            raise EvaluationControlPlaneIntegrityError(str(exc)) from exc

    class _Transaction:
        def __init__(self, store: "EvaluationControlPlaneStore") -> None:
            self.store = store

        def __enter__(self) -> sqlite3.Connection:
            self.store._lock.acquire()
            try:
                hosted.begin_write(self.store._conn)
            except BaseException:
                self.store._lock.release()
                raise
            return self.store._conn

        def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
            try:
                if exc_type is None:
                    self.store._conn.commit()
                else:
                    self.store._conn.rollback()
            finally:
                self.store._lock.release()

    def _transaction(self) -> "_Transaction":
        return self._Transaction(self)
