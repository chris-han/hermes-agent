"""Versioned canonical-list runtime profile storage."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from storage.hosted_postgres import enabled,connect_authority,mapped_store,require_tables,sql_query,begin_write,integrity_errors
from eos.artifact_hashing import canonical_json, content_hash


CANONICAL_LIST_PROFILE_SCHEMA_VERSION = "canonical_list_runtime_profile.v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _loads(value: str) -> Any:
    return json.loads(value) if value else None


@dataclass(frozen=True)
class CanonicalListProfileVersion:
    schema_version: str
    profile_ref: str
    version: str
    content_hash: str
    role_mapping: dict[str, Any]
    runtime_policy: dict[str, Any]
    created_by: str
    created_at: str
    supersedes_version: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile_ref": self.profile_ref,
            "version": self.version,
            "content_hash": self.content_hash,
            "role_mapping": self.role_mapping,
            "runtime_policy": self.runtime_policy,
            "created_by": self.created_by,
            "created_at": self.created_at,
            "supersedes_version": self.supersedes_version,
        }


class CanonicalListProfileStore:
    """Append-only profile version records scoped to a workspace."""

    _process_locks: dict[Path, threading.RLock] = {}
    _process_locks_guard = threading.Lock()

    def __init__(self, workspace_root: str | Path, *, workspace_id: str) -> None:
        if not workspace_id:
            raise ValueError("workspace_id is required")
        self.workspace_root = Path(workspace_root).resolve()
        self.workspace_id = workspace_id
        self.root = self.workspace_root / ".semantier" / "canonical-list"
        if not enabled():self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "profiles.sqlite3"
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
            with self._connect() as connection:require_tables(connection,'canonical_list_profile_versions')
            return
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS canonical_list_profile_versions (
                    profile_ref TEXT NOT NULL,
                    version TEXT NOT NULL,
                    workspace_id TEXT NOT NULL,
                    schema_version TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    role_mapping_json TEXT NOT NULL,
                    runtime_policy_json TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    supersedes_version TEXT,
                    PRIMARY KEY (workspace_id, profile_ref, version)
                )
                """
            )

    def create_profile_version(
        self,
        *,
        profile_ref: str,
        role_mapping: dict[str, Any],
        runtime_policy: dict[str, Any] | None,
        created_by: str,
        version: str | None = None,
        supersedes_version: str | None = None,
    ) -> CanonicalListProfileVersion:
        if not profile_ref or not created_by:
            raise ValueError("profile_ref and created_by are required")
        self._validate_role_mapping(role_mapping)
        policy = {
            "unknown_selector_behavior": "retain_candidate",
            "reference_example_promotion": "explicit_human_review_only",
            **(runtime_policy or {}),
        }
        if policy["unknown_selector_behavior"] != "retain_candidate":
            raise ValueError("MVL profile must retain unknown-selector candidates")
        with self._lock, self._connect() as connection:
            if enabled():begin_write(connection)
            if version is None:
                version = self._next_version(profile_ref,connection)
            payload = {
                "schema_version": CANONICAL_LIST_PROFILE_SCHEMA_VERSION,
                "profile_ref": profile_ref,
                "version": version,
                "role_mapping": role_mapping,
                "runtime_policy": policy,
                "created_by": created_by,
                "supersedes_version": supersedes_version,
            }
            profile = CanonicalListProfileVersion(
                schema_version=CANONICAL_LIST_PROFILE_SCHEMA_VERSION,
                profile_ref=profile_ref,
                version=version,
                content_hash=f"sha256:{content_hash(payload)}",
                role_mapping=role_mapping,
                runtime_policy=policy,
                created_by=created_by,
                created_at=_utc_now(),
                supersedes_version=supersedes_version,
            )
            connection.execute(
                sql_query("""
                INSERT INTO canonical_list_profile_versions VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,'\n                INSERT INTO canonical_list_profile_versions VALUES (\n                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s\n                )\n                '),
                (
                    profile.profile_ref,
                    profile.version,
                    self.workspace_id,
                    profile.schema_version,
                    profile.content_hash,
                    canonical_json(profile.role_mapping),
                    canonical_json(profile.runtime_policy),
                    profile.created_by,
                    profile.created_at,
                    profile.supersedes_version,
                ),
            )
        return profile

    def get_profile_version(self, *, profile_ref: str, version: str) -> CanonicalListProfileVersion:
        with self._connect() as connection:
            row = connection.execute(
                sql_query("""
                SELECT * FROM canonical_list_profile_versions
                WHERE workspace_id = ? AND profile_ref = ? AND version = ?
                """,'\n                SELECT * FROM canonical_list_profile_versions\n                WHERE workspace_id = %s AND profile_ref = %s AND version = %s\n                '),
                (self.workspace_id, profile_ref, version),
            ).fetchone()
        if row is None:
            raise KeyError(f"canonical-list profile version not found: {profile_ref}@{version}")
        return self._hydrate(row)

    def _next_version(self, profile_ref: str, connection) -> str:
        row = connection.execute(
            sql_query("""
            SELECT COUNT(*) AS count FROM canonical_list_profile_versions
            WHERE workspace_id = ? AND profile_ref = ?
            """,'\n                SELECT COUNT(*) AS count FROM canonical_list_profile_versions\n                WHERE workspace_id = %s AND profile_ref = %s\n                '),
            (self.workspace_id, profile_ref),
        ).fetchone()
        return str(int(row["count"]) + 1)

    def _hydrate(self, row: sqlite3.Row) -> CanonicalListProfileVersion:
        return CanonicalListProfileVersion(
            schema_version=row["schema_version"],
            profile_ref=row["profile_ref"],
            version=row["version"],
            content_hash=row["content_hash"],
            role_mapping=_loads(row["role_mapping_json"]),
            runtime_policy=_loads(row["runtime_policy_json"]),
            created_by=row["created_by"],
            created_at=row["created_at"],
            supersedes_version=row["supersedes_version"],
        )

    def _validate_role_mapping(self, role_mapping: dict[str, Any]) -> None:
        required = {"canonical_concept", "classification", "runtime_selector", "output_suggestion", "reference_example"}
        missing = sorted(required - set(role_mapping))
        if missing:
            raise ValueError(f"canonical-list profile role_mapping missing roles: {missing}")
        concept = role_mapping.get("canonical_concept") or {}
        if not concept.get("source_fields") or concept.get("required") is not True:
            raise ValueError("canonical_concept source_fields and required=true are required")
        reference = role_mapping.get("reference_example") or {}
        if reference.get("canonical_by_default") is not False:
            raise ValueError("ReferenceExample must not be canonical by default")

