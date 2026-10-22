from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from eos.artifact_hashing import canonical_json, content_hash
from eos.db import init_db, now_iso
from storage import hosted_postgres as hosted
from eos.replay_binding_store import verify_replay_binding
from runtime_paths import sqlite_db_path


SENSITIVE_DETECTION_TYPE = "sensitive_labeling_detection.v1"
SENSITIVE_REPORT_TYPE = "sensitive_labeling_report.v1"
PROJECTION_EXECUTION_TYPE = "projection_execution.v1"
INVESTMENT_RADAR_CANDIDATE_TYPE = "investment_radar_candidate.v1"

_SPECIFIC_TABLES = {
    SENSITIVE_DETECTION_TYPE: "sensitive_labeling_detection_bindings",
    SENSITIVE_REPORT_TYPE: "sensitive_labeling_report_bindings",
    INVESTMENT_RADAR_CANDIDATE_TYPE: "investment_radar_candidate_bindings",
}
_SUPPORTED_BINDING_TYPES = {
    PROJECTION_EXECUTION_TYPE,
    SENSITIVE_DETECTION_TYPE,
    SENSITIVE_REPORT_TYPE,
    INVESTMENT_RADAR_CANDIDATE_TYPE,
}


def _db_path() -> Path:
    return Path(
        os.environ.get(
            "SEMANTIER_EOS_DB_PATH",
            str(sqlite_db_path("eos.db", ".semantier/eos.db")),
        )
    )


def _loads_json(value: str) -> Any:
    return json.loads(value) if value else None


def _envelope_hash(envelope: dict[str, Any]) -> str:
    return content_hash(
        {
            "replay_binding_id": envelope["replay_binding_id"],
            "binding_type": envelope["binding_type"],
            "binding_schema_version": envelope["binding_schema_version"],
            "subject_type": envelope["subject_type"],
            "subject_id": envelope["subject_id"],
            "specific_binding_content_hash": envelope["specific_binding_content_hash"],
            "organization_id": envelope["organization_id"],
            "workspace_id": envelope.get("workspace_id"),
        }
    )


class GenericReplayEnvelopeStore:
    def __init__(self, db_path: str | Path | None = None) -> None:
        path = Path(db_path) if db_path else _db_path()
        if not hosted.enabled():
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
        if hosted.enabled():
            self._conn = hosted.connect_authority('eos', path, autocommit=True)
        else:
            self._conn = sqlite3.connect(str(path), check_same_thread=False, timeout=10.0)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._lock = threading.RLock()

    def create_typed_replay_binding(
        self,
        *,
        replay_binding_id: str,
        binding_type: str,
        binding_schema_version: str,
        subject_type: str,
        subject_id: str,
        organization_id: str,
        workspace_id: str | None,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if binding_type not in _SPECIFIC_TABLES:
            raise ValueError("UNSUPPORTED_REPLAY_BINDING_TYPE")
        payload_hash = content_hash(payload)
        envelope = {
            "replay_binding_id": replay_binding_id,
            "binding_type": binding_type,
            "binding_schema_version": binding_schema_version,
            "subject_type": subject_type,
            "subject_id": subject_id,
            "specific_binding_content_hash": payload_hash,
            "organization_id": organization_id,
            "workspace_id": workspace_id,
        }
        envelope["envelope_hash"] = _envelope_hash(envelope)
        created_at = now_iso()
        table_name = _SPECIFIC_TABLES[binding_type]

        with self._lock, (self._conn.transaction() if hosted.is_postgres(self._conn) else nullcontext()):
            if hosted.is_postgres(self._conn):
                self._conn.execute("SELECT pg_advisory_xact_lock(hashtext(current_schema()),hashtext('replay-envelope'))")
            existing = self._conn.execute(
                hosted.sql_query("SELECT * FROM replay_envelopes WHERE replay_binding_id = ?", 'SELECT * FROM replay_envelopes WHERE replay_binding_id = %s'),
                (replay_binding_id,),
            ).fetchone()
            if existing:
                verification = self.verify_typed_replay_binding(
                    replay_binding_id=replay_binding_id,
                    expected_binding_type=binding_type,
                    expected_organization_id=organization_id,
                    expected_workspace_id=workspace_id,
                )
                if verification.get("status") != "OK":
                    raise ValueError(
                        f"existing replay envelope failed verification: {verification.get('reason')}"
                    )
                existing_payload = verification.get("payload")
                if content_hash(existing_payload) != payload_hash:
                    raise ValueError(
                        f"replay envelope already exists for {replay_binding_id} with different payload"
                    )
                return verification

            if not hosted.is_postgres(self._conn):
                self._conn.execute("BEGIN")
            try:
                self._conn.execute(
                    hosted.sql_query("""
                    INSERT INTO replay_envelopes (
                      replay_binding_id, binding_type, binding_schema_version,
                      subject_type, subject_id, specific_binding_content_hash,
                      envelope_hash, organization_id, workspace_id, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, """
                    INSERT INTO replay_envelopes (
                      replay_binding_id, binding_type, binding_schema_version,
                      subject_type, subject_id, specific_binding_content_hash,
                      envelope_hash, organization_id, workspace_id, created_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """),
                    (
                        replay_binding_id,
                        binding_type,
                        binding_schema_version,
                        subject_type,
                        subject_id,
                        payload_hash,
                        envelope["envelope_hash"],
                        organization_id,
                        workspace_id,
                        created_at,
                    ),
                )
                self._conn.execute(
                    hosted.sql_query(f"""
                    INSERT INTO {table_name} (
                      replay_binding_id, binding_type, organization_id, workspace_id,
                      binding_payload_json, binding_payload_hash, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, f'\n                    INSERT INTO {table_name} (\n                      replay_binding_id, binding_type, organization_id, workspace_id,\n                      binding_payload_json, binding_payload_hash, created_at\n                    )\n                    VALUES (%s, %s, %s, %s, %s, %s, %s)\n                    '),
                    (
                        replay_binding_id,
                        binding_type,
                        organization_id,
                        workspace_id,
                        canonical_json(payload),
                        payload_hash,
                        created_at,
                    ),
                )
                if not hosted.is_postgres(self._conn):
                    self._conn.commit()
            except Exception:
                if not hosted.is_postgres(self._conn):
                    self._conn.rollback()
                raise

        return self.verify_typed_replay_binding(
            replay_binding_id=replay_binding_id,
            expected_binding_type=binding_type,
            expected_organization_id=organization_id,
            expected_workspace_id=workspace_id,
        )

    def verify_typed_replay_binding(
        self,
        *,
        replay_binding_id: str,
        expected_binding_type: str | None = None,
        expected_organization_id: str | None = None,
        expected_workspace_id: str | None = None,
    ) -> dict[str, Any]:
        if expected_binding_type is not None and expected_binding_type not in _SUPPORTED_BINDING_TYPES:
            return {"status": "FAIL_CLOSED", "reason": "UNSUPPORTED_REPLAY_BINDING_TYPE"}
        envelope_row = self._conn.execute(
            hosted.sql_query("SELECT * FROM replay_envelopes WHERE replay_binding_id = ?", 'SELECT * FROM replay_envelopes WHERE replay_binding_id = %s'),
            (replay_binding_id,),
        ).fetchone()
        if envelope_row is None:
            if expected_binding_type in {None, PROJECTION_EXECUTION_TYPE}:
                legacy = verify_replay_binding(replay_binding_id)
                return {
                    "status": legacy.get("status"),
                    "binding_type": PROJECTION_EXECUTION_TYPE,
                    "legacy_projection_replay": legacy,
                }
            return {"status": "NOT_FOUND", "reason": "replay_envelope_not_found"}

        envelope = dict(envelope_row)
        if expected_binding_type and envelope["binding_type"] != expected_binding_type:
            return {
                "status": "FAIL_CLOSED",
                "reason": "replay_binding_type_mismatch",
                "actual": envelope["binding_type"],
                "expected": expected_binding_type,
            }
        if (
            expected_organization_id is not None
            and envelope["organization_id"] != expected_organization_id
        ):
            return {"status": "FAIL_CLOSED", "reason": "replay_binding_org_scope_mismatch"}
        if (
            expected_workspace_id is not None
            and envelope["workspace_id"] != expected_workspace_id
        ):
            return {
                "status": "FAIL_CLOSED",
                "reason": "replay_binding_workspace_scope_mismatch",
            }

        table_rows = []
        for binding_type, table_name in _SPECIFIC_TABLES.items():
            row = self._conn.execute(
                hosted.sql_query(f"SELECT * FROM {table_name} WHERE replay_binding_id = ?", f'SELECT * FROM {table_name} WHERE replay_binding_id = %s'),
                (replay_binding_id,),
            ).fetchone()
            if row is not None:
                table_rows.append((binding_type, table_name, dict(row)))
        if len(table_rows) != 1:
            return {
                "status": "FAIL_CLOSED",
                "reason": "replay_specific_binding_count_mismatch",
                "specific_binding_count": len(table_rows),
            }
        binding_type, _table_name, specific = table_rows[0]
        if binding_type != envelope["binding_type"] or specific["binding_type"] != envelope["binding_type"]:
            return {"status": "FAIL_CLOSED", "reason": "replay_specific_binding_type_mismatch"}
        if specific["organization_id"] != envelope["organization_id"]:
            return {"status": "FAIL_CLOSED", "reason": "replay_specific_org_scope_mismatch"}
        if specific["workspace_id"] != envelope["workspace_id"]:
            return {
                "status": "FAIL_CLOSED",
                "reason": "replay_specific_workspace_scope_mismatch",
            }
        payload = _loads_json(specific["binding_payload_json"])
        payload_hash = content_hash(payload)
        if payload_hash != specific["binding_payload_hash"]:
            return {"status": "FAIL_CLOSED", "reason": "replay_specific_payload_hash_mismatch"}
        if payload_hash != envelope["specific_binding_content_hash"]:
            return {
                "status": "FAIL_CLOSED",
                "reason": "replay_envelope_specific_hash_mismatch",
            }
        expected_envelope_hash = _envelope_hash(envelope)
        if expected_envelope_hash != envelope["envelope_hash"]:
            return {"status": "FAIL_CLOSED", "reason": "replay_envelope_hash_mismatch"}

        return {
            "status": "OK",
            "replay_binding_id": replay_binding_id,
            "binding_type": envelope["binding_type"],
            "binding_schema_version": envelope["binding_schema_version"],
            "subject_type": envelope["subject_type"],
            "subject_id": envelope["subject_id"],
            "organization_id": envelope["organization_id"],
            "workspace_id": envelope["workspace_id"],
            "specific_binding_content_hash": envelope["specific_binding_content_hash"],
            "envelope_hash": envelope["envelope_hash"],
            "payload": payload,
        }


def create_typed_replay_binding(**kwargs: Any) -> dict[str, Any]:
    return GenericReplayEnvelopeStore().create_typed_replay_binding(**kwargs)


def verify_typed_replay_binding(**kwargs: Any) -> dict[str, Any]:
    return GenericReplayEnvelopeStore().verify_typed_replay_binding(**kwargs)
