"""Disposable eos.db projection for knowledge benchmark observability."""

from __future__ import annotations

from storage import hosted_postgres as hosted
from storage.transaction_scope import StoreMutationLock

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Mapping


class BenchmarkReadModelIntegrityError(ValueError):
    pass


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class BenchmarkReadModelStore:
    schema_version = "benchmark_read_model_schema.v1"
    _scope_locks: dict[tuple[str, str | None], threading.Lock] = {}
    _scope_locks_guard = threading.Lock()

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        if hosted.enabled():
            self._conn = hosted.connect_authority('eos', self._db_path)
        else:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        self._lock = StoreMutationLock(self._conn)
        self._init_schema()

    def close(self) -> None:
        self._conn.close()

    def _init_schema(self) -> None:
        if hosted.enabled():
            hosted.require_tables(self._conn, *('benchmark_read_model_schema_metadata', 'benchmark_read_model_orchestrations', 'benchmark_read_model_layer_runs', 'benchmark_read_model_case_results', 'benchmark_read_model_challenge_slices', 'benchmark_read_model_artifact_refs'))
            self._conn.commit()
            return
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS benchmark_read_model_schema_metadata (
                  schema_version TEXT PRIMARY KEY,
                  initialized_at TEXT NOT NULL
                );
                INSERT OR IGNORE INTO benchmark_read_model_schema_metadata
                  (schema_version, initialized_at)
                VALUES ('benchmark_read_model_schema.v1', strftime('%Y-%m-%dT%H:%M:%fZ','now'));

                CREATE TABLE IF NOT EXISTS benchmark_read_model_orchestrations (
                  benchmark_orchestration_id TEXT PRIMARY KEY,
                  schema_version TEXT NOT NULL,
                  organization_id TEXT NOT NULL,
                  workspace_id TEXT NULL,
                  benchmark_family_ref TEXT NOT NULL,
                  profile_id TEXT NOT NULL,
                  profile_version TEXT NOT NULL,
                  requested_layers_json TEXT NOT NULL,
                  execution_mode TEXT NOT NULL,
                  operational_status TEXT NOT NULL,
                  warnings_json TEXT NOT NULL,
                  child_run_refs_json TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  completed_at TEXT NULL,
                  stale_reason_code TEXT NULL,
                  content_hash TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_brm_orch_org_created
                  ON benchmark_read_model_orchestrations(organization_id, created_at DESC, benchmark_orchestration_id DESC);
                CREATE INDEX IF NOT EXISTS idx_brm_orch_workspace_created
                  ON benchmark_read_model_orchestrations(organization_id, workspace_id, created_at DESC, benchmark_orchestration_id DESC);
                CREATE INDEX IF NOT EXISTS idx_brm_orch_status
                  ON benchmark_read_model_orchestrations(organization_id, operational_status, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_brm_orch_profile
                  ON benchmark_read_model_orchestrations(organization_id, profile_id, profile_version, created_at DESC);

                CREATE TABLE IF NOT EXISTS benchmark_read_model_layer_runs (
                  evaluation_run_id TEXT PRIMARY KEY,
                  schema_version TEXT NOT NULL,
                  benchmark_orchestration_id TEXT NOT NULL,
                  organization_id TEXT NOT NULL,
                  workspace_id TEXT NULL,
                  layer TEXT NOT NULL,
                  provider_id TEXT NULL,
                  profile_id TEXT NOT NULL,
                  profile_version TEXT NOT NULL,
                  execution_mode TEXT NOT NULL,
                  operational_status TEXT NOT NULL,
                  evaluation_evidence_id TEXT NULL,
                  run_validity TEXT NULL,
                  certification_result TEXT NULL,
                  benchmark_family_ref TEXT NOT NULL,
                  scenario_ref TEXT NOT NULL,
                  subject_manifest_ref TEXT NOT NULL,
                  environment_manifest_ref TEXT NOT NULL,
                  evaluator_bundle_ref TEXT NOT NULL,
                  reference_release_id TEXT NULL,
                  reference_release_hash TEXT NULL,
                  created_at TEXT NOT NULL,
                  completed_at TEXT NULL,
                  summary_json TEXT NOT NULL,
                  warnings_json TEXT NOT NULL,
                  stale_reason_code TEXT NULL,
                  content_hash TEXT NOT NULL,
                  UNIQUE (benchmark_orchestration_id, layer, evaluation_run_id)
                );
                CREATE INDEX IF NOT EXISTS idx_brm_layer_parent ON benchmark_read_model_layer_runs(benchmark_orchestration_id, layer);
                CREATE INDEX IF NOT EXISTS idx_brm_layer_cert ON benchmark_read_model_layer_runs(organization_id, certification_result, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_brm_layer_provider ON benchmark_read_model_layer_runs(organization_id, layer, provider_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_brm_layer_status ON benchmark_read_model_layer_runs(organization_id, execution_mode, operational_status, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_brm_layer_reference ON benchmark_read_model_layer_runs(organization_id, reference_release_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS benchmark_read_model_case_results (
                  evaluation_run_id TEXT NOT NULL, case_id TEXT NOT NULL,
                  organization_id TEXT NOT NULL, workspace_id TEXT NULL,
                  layer TEXT NOT NULL, provider_id TEXT NULL, status TEXT NOT NULL,
                  challenge_tags_json TEXT NOT NULL, metric_contributions_json TEXT NOT NULL,
                  source_anchor_refs_json TEXT NOT NULL, assertion_refs_json TEXT NOT NULL,
                  artifact_refs_json TEXT NOT NULL, result_json TEXT NOT NULL,
                  stale_reason_code TEXT NULL, content_hash TEXT NOT NULL,
                  PRIMARY KEY (evaluation_run_id, case_id)
                );
                CREATE INDEX IF NOT EXISTS idx_brm_case_org_status ON benchmark_read_model_case_results(organization_id, layer, status, case_id);
                CREATE INDEX IF NOT EXISTS idx_brm_case_provider ON benchmark_read_model_case_results(organization_id, provider_id, status, case_id);

                CREATE TABLE IF NOT EXISTS benchmark_read_model_challenge_slices (
                  evaluation_run_id TEXT NOT NULL, challenge_tag TEXT NOT NULL,
                  organization_id TEXT NOT NULL, workspace_id TEXT NULL, layer TEXT NOT NULL,
                  base_case_set_hash TEXT NOT NULL, base_case_count INTEGER NOT NULL,
                  included_case_count INTEGER NOT NULL, excluded_case_count INTEGER NOT NULL,
                  excluded_case_refs_json TEXT NOT NULL, denominator_policy TEXT NOT NULL,
                  metrics_json TEXT NOT NULL, artifact_refs_json TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  PRIMARY KEY (evaluation_run_id, challenge_tag)
                );

                CREATE TABLE IF NOT EXISTS benchmark_read_model_artifact_refs (
                  evaluation_run_id TEXT NOT NULL, artifact_ref TEXT NOT NULL,
                  organization_id TEXT NOT NULL, workspace_id TEXT NULL,
                  artifact_kind TEXT NOT NULL, media_type TEXT NOT NULL,
                  artifact_content_hash TEXT NOT NULL, availability TEXT NOT NULL,
                  stale_reason_code TEXT NULL,
                  PRIMARY KEY (evaluation_run_id, artifact_ref)
                );
                """
            )
            self._conn.commit()

    def put_orchestration(self, record: Mapping[str, Any]) -> dict[str, Any]:
        row = dict(record)
        values = (
            row["benchmark_orchestration_id"], row["schema_version"], row["organization_id"],
            row.get("workspace_id"), row["benchmark_family_ref"], row["profile_id"],
            row["profile_version"], _json(row["requested_layers"]), row["execution_mode"],
            row["operational_status"], _json(row.get("warnings", [])),
            _json(row.get("child_run_refs", [])), row["created_at"], row.get("completed_at"),
            row.get("stale_reason_code"), row["content_hash"],
        )
        self._insert_immutable(
            "benchmark_read_model_orchestrations",
            "benchmark_orchestration_id",
            row["benchmark_orchestration_id"],
            row["content_hash"],
            hosted.sql_query("""INSERT INTO benchmark_read_model_orchestrations VALUES
            (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", 'INSERT INTO benchmark_read_model_orchestrations VALUES\n            (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)'),
            values,
        )
        return self.get_orchestration(row["benchmark_orchestration_id"], row["organization_id"], row.get("workspace_id"))

    def put_layer_run(self, record: Mapping[str, Any]) -> dict[str, Any]:
        row = dict(record)
        values = (
            row["evaluation_run_id"], row["schema_version"], row["benchmark_orchestration_id"],
            row["organization_id"], row.get("workspace_id"), row["layer"], row.get("provider_id"),
            row["profile_id"], row["profile_version"], row["execution_mode"],
            row["operational_status"], row.get("evaluation_evidence_id"), row.get("run_validity"),
            row.get("certification_result"), row["benchmark_family_ref"], row["scenario_ref"],
            row["subject_manifest_ref"], row["environment_manifest_ref"], row["evaluator_bundle_ref"],
            row.get("reference_release_id"), row.get("reference_release_hash"), row["created_at"],
            row.get("completed_at"), _json(row.get("summary", {})), _json(row.get("warnings", [])),
            row.get("stale_reason_code"), row["content_hash"],
        )
        self._insert_immutable(
            "benchmark_read_model_layer_runs", "evaluation_run_id", row["evaluation_run_id"],
            row["content_hash"],
            hosted.sql_query("""INSERT INTO benchmark_read_model_layer_runs VALUES
            (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", 'INSERT INTO benchmark_read_model_layer_runs VALUES\n            (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)'),
            values,
        )
        return self.get_layer_run(row["evaluation_run_id"], row["organization_id"], row.get("workspace_id"))

    def put_case_result(self, record: Mapping[str, Any]) -> None:
        with self._lock:
            row = dict(record)
            identity = (row["evaluation_run_id"], row["case_id"])
            existing = self._conn.execute(
                hosted.sql_query("SELECT content_hash FROM benchmark_read_model_case_results WHERE evaluation_run_id=? AND case_id=?", 'SELECT content_hash FROM benchmark_read_model_case_results WHERE evaluation_run_id=%s AND case_id=%s'),
                identity,
            ).fetchone()
            if existing:
                if existing["content_hash"] != row["content_hash"]:
                    raise BenchmarkReadModelIntegrityError("immutable case result has different content")
                return
            self._conn.execute(
                hosted.sql_query("""INSERT INTO benchmark_read_model_case_results VALUES
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", 'INSERT INTO benchmark_read_model_case_results VALUES\n            (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)'),
                (
                    *identity, row["organization_id"], row.get("workspace_id"), row["layer"],
                    row.get("provider_id"), row["status"], _json(row.get("challenge_tags", [])),
                    _json(row.get("metric_contributions", [])), _json(row.get("source_anchor_refs", [])),
                    _json(row.get("assertion_refs", [])), _json(row.get("artifact_refs", [])),
                    _json(row), row.get("stale_reason_code"), row["content_hash"],
                ),
            )
            self._conn.commit()

    def get_orchestration(self, identity: str, organization_id: str, workspace_id: str | None) -> dict[str, Any]:
        row = self._owned(
            "benchmark_read_model_orchestrations", "benchmark_orchestration_id",
            identity, organization_id, workspace_id,
        )
        return self._decode_orchestration(row)

    def list_orchestrations(
        self, organization_id: str, workspace_id: str | None, *, limit: int = 50, cursor: str | None = None
    ) -> tuple[list[dict[str, Any]], str | None]:
        if not 1 <= limit <= 200:
            raise ValueError("limit must be between 1 and 200")
        params: list[Any] = [organization_id, workspace_id]
        clause = ""
        if cursor:
            try:
                created_at, identity = json.loads(bytes.fromhex(cursor).decode())
            except Exception as exc:
                raise ValueError("invalid cursor") from exc
            clause = hosted.sql_query("AND (created_at < ? OR (created_at = ? AND benchmark_orchestration_id < ?))", 'AND (created_at < %s OR (created_at = %s AND benchmark_orchestration_id < %s))')
            params.extend((created_at, created_at, identity))
        params.append(limit + 1)
        rows = self._conn.execute(
            hosted.sql_query(f"""SELECT * FROM benchmark_read_model_orchestrations
            WHERE organization_id=? AND workspace_id IS ?
            {clause}
            ORDER BY created_at DESC, benchmark_orchestration_id DESC LIMIT ?""", f'SELECT * FROM benchmark_read_model_orchestrations\n            WHERE organization_id=%s AND workspace_id IS NOT DISTINCT FROM %s\n            {clause}\n            ORDER BY created_at DESC, benchmark_orchestration_id DESC LIMIT %s'),
            params,
        ).fetchall()
        page = rows[:limit]
        next_cursor = None
        if len(rows) > limit and page:
            next_cursor = json.dumps([page[-1]["created_at"], page[-1]["benchmark_orchestration_id"]]).encode().hex()
        return [self._decode_orchestration(row) for row in page], next_cursor

    def list_layers(self, orchestration_id: str, organization_id: str, workspace_id: str | None) -> list[dict[str, Any]]:
        self.get_orchestration(orchestration_id, organization_id, workspace_id)
        rows = self._conn.execute(
            hosted.sql_query("""SELECT * FROM benchmark_read_model_layer_runs
            WHERE benchmark_orchestration_id=? AND organization_id=? AND workspace_id IS ?
            ORDER BY layer, evaluation_run_id""", 'SELECT * FROM benchmark_read_model_layer_runs\n            WHERE benchmark_orchestration_id=%s AND organization_id=%s AND workspace_id IS NOT DISTINCT FROM %s\n            ORDER BY layer, evaluation_run_id'),
            (orchestration_id, organization_id, workspace_id),
        ).fetchall()
        return [self._decode_layer(row) for row in rows]

    def get_layer_run(self, identity: str, organization_id: str, workspace_id: str | None) -> dict[str, Any]:
        return self._decode_layer(self._owned(
            "benchmark_read_model_layer_runs", "evaluation_run_id", identity, organization_id, workspace_id
        ))

    def list_cases(
        self, run_id: str, organization_id: str, workspace_id: str | None, *, limit: int = 50, cursor: str | None = None
    ) -> tuple[list[dict[str, Any]], str | None]:
        self.get_layer_run(run_id, organization_id, workspace_id)
        if not 1 <= limit <= 200:
            raise ValueError("limit must be between 1 and 200")
        params: list[Any] = [run_id]
        clause = ""
        if cursor:
            try:
                last = bytes.fromhex(cursor).decode()
            except Exception as exc:
                raise ValueError("invalid cursor") from exc
            clause = hosted.sql_query("AND case_id > ?", 'AND case_id > %s')
            params.append(last)
        params.append(limit + 1)
        rows = self._conn.execute(
            hosted.sql_query(f"""SELECT result_json FROM benchmark_read_model_case_results
            WHERE evaluation_run_id=? {clause} ORDER BY case_id LIMIT ?""", f'SELECT result_json FROM benchmark_read_model_case_results\n            WHERE evaluation_run_id=%s {clause} ORDER BY case_id LIMIT %s'), params
        ).fetchall()
        values = [json.loads(row["result_json"]) for row in rows[:limit]]
        next_cursor = values[-1]["case_id"].encode().hex() if len(rows) > limit and values else None
        return values, next_cursor

    def list_challenge_slices(self, run_id: str, organization_id: str, workspace_id: str | None) -> list[dict[str, Any]]:
        self.get_layer_run(run_id, organization_id, workspace_id)
        return [
            {
                **dict(row),
                "excluded_case_refs": json.loads(row["excluded_case_refs_json"]),
                "metrics": json.loads(row["metrics_json"]),
                "artifact_refs": json.loads(row["artifact_refs_json"]),
            }
            for row in self._conn.execute(
                hosted.sql_query("SELECT * FROM benchmark_read_model_challenge_slices WHERE evaluation_run_id=? ORDER BY challenge_tag", 'SELECT * FROM benchmark_read_model_challenge_slices WHERE evaluation_run_id=%s ORDER BY challenge_tag'),
                (run_id,),
            ).fetchall()
        ]

    def list_artifacts(self, run_id: str, organization_id: str, workspace_id: str | None) -> list[dict[str, Any]]:
        self.get_layer_run(run_id, organization_id, workspace_id)
        return [
            dict(row)
            for row in self._conn.execute(
                hosted.sql_query("SELECT * FROM benchmark_read_model_artifact_refs WHERE evaluation_run_id=? ORDER BY artifact_ref", 'SELECT * FROM benchmark_read_model_artifact_refs WHERE evaluation_run_id=%s ORDER BY artifact_ref'),
                (run_id,),
            ).fetchall()
        ]

    def rebuild_from_artifacts(
        self, organization_id: str, workspace_id: str | None = None, *, records: Iterable[Mapping[str, Any]] = ()
    ) -> dict[str, Any]:
        scope = (organization_id, workspace_id)
        with self._scope_locks_guard:
            lock = self._scope_locks.setdefault(scope, threading.Lock())
        if not lock.acquire(blocking=False):
            raise RuntimeError("benchmark read-model rebuild already active for scope")
        try:
            with self._lock:
                hosted.begin_write(self._conn)
                self._conn.execute(
                    hosted.sql_query("""UPDATE benchmark_read_model_orchestrations SET stale_reason_code='ORPHANED_FROM_REBUILD'
                    WHERE organization_id=? AND workspace_id IS ?""", "UPDATE benchmark_read_model_orchestrations SET stale_reason_code='ORPHANED_FROM_REBUILD'\n                    WHERE organization_id=%s AND workspace_id IS NOT DISTINCT FROM %s"), scope
                )
                count = 0
                for record in records:
                    if record["kind"] == "artifact_unavailable":
                        self._conn.execute(
                            hosted.sql_query("""UPDATE benchmark_read_model_artifact_refs
                            SET availability='unavailable', stale_reason_code=?
                            WHERE evaluation_run_id=? AND artifact_ref=? AND organization_id=? AND workspace_id IS ?""", "UPDATE benchmark_read_model_artifact_refs\n                            SET availability='unavailable', stale_reason_code=%s\n                            WHERE evaluation_run_id=%s AND artifact_ref=%s AND organization_id=%s AND workspace_id IS NOT DISTINCT FROM %s"),
                            (
                                record.get("stale_reason_code", "ARTIFACT_UNAVAILABLE"),
                                record["evaluation_run_id"], record["artifact_ref"], *scope,
                            ),
                        )
                        count += 1
                self._conn.commit()
            return {"schema_version": self.schema_version, "scope": scope, "processed": count}
        except Exception:
            self._conn.rollback()
            raise
        finally:
            lock.release()

    def _owned(self, table: str, column: str, identity: str, organization_id: str, workspace_id: str | None) -> sqlite3.Row:
        row = self._conn.execute(
            hosted.sql_query(f"SELECT * FROM {table} WHERE {column}=? AND organization_id=? AND workspace_id IS ?", f'SELECT * FROM {table} WHERE {column}=%s AND organization_id=%s AND workspace_id IS NOT DISTINCT FROM %s'),
            (identity, organization_id, workspace_id),
        ).fetchone()
        if row is None:
            raise KeyError(identity)
        return row

    def _insert_immutable(self, table: str, id_column: str, identity: str, digest: str, sql: str, values: tuple[Any, ...]) -> None:
        with self._lock:
            row = self._conn.execute(
                hosted.sql_query(f"SELECT content_hash FROM {table} WHERE {id_column}=?", f'SELECT content_hash FROM {table} WHERE {id_column}=%s'), (identity,)
            ).fetchone()
            if row:
                if row["content_hash"] != digest:
                    raise BenchmarkReadModelIntegrityError(f"immutable {table} identity has different content")
                return
            self._conn.execute(sql, values)
            self._conn.commit()

    @staticmethod
    def _decode_orchestration(row: sqlite3.Row) -> dict[str, Any]:
        value = dict(row)
        value["requested_layers"] = json.loads(value.pop("requested_layers_json"))
        value["warnings"] = json.loads(value.pop("warnings_json"))
        value["child_run_refs"] = json.loads(value.pop("child_run_refs_json"))
        return value

    @staticmethod
    def _decode_layer(row: sqlite3.Row) -> dict[str, Any]:
        value = dict(row)
        value["summary"] = json.loads(value.pop("summary_json"))
        value["warnings"] = json.loads(value.pop("warnings_json"))
        return value
