"""Governed append-only persistence for GraphIntentBrief.v1 artifacts."""

from __future__ import annotations

import base64
import binascii
from contextlib import AbstractContextManager, closing, contextmanager
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import errno
import json
import os
from pathlib import Path
import re
import sqlite3
from storage.hosted_postgres import enabled,connect_authority,require_tables,sql_query,begin_write,in_transaction
import stat
import threading
import time
from typing import Any, Iterator
from uuid import uuid4

from contracts.canonical_source_ir import CanonicalSourceDocument, source_block_identity
from contracts.canonical_source_ir import content_hash as canonical_ir_content_hash
from contracts.evidence_location import EvidenceRef, SourceIdentity
from contracts.graph_intent_brief import (
    GraphIntentBrief,
    GraphIntentReview,
    GraphIntentSourceBinding,
)
from contracts.runtime_context_reconstruction import CapturedJsonV1
from eos.artifact_hashing import canonical_json, content_hash
from semantier.knowledge_feature_packs.canonical_source_ir_adapter import (
    AmbiguousSourceMapping,
    resolve_unique_mapping,
)
from semantier.knowledge_feature_packs.evidence_selector_builder import (
    evidence_ref_for_block,
)
from storage.knowledge_builder_store import KnowledgeBuilderStore


_ARTIFACT_NAME = "graph_intent_brief.v1.json"
_BINDING_NAME = "source_binding_evidence.v1.json"
_MATERIAL_FIELDS = {
    "frame_mapping",
    "document_function",
    "represented_or_governed_subject",
    "primary_relationship",
    "preferred_representation",
    "authority_boundary",
    "material_dimensions",
    "source_role_interpretation",
    "unresolved_questions",
}
_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_SHA256_REF = re.compile(r"^sha256:[0-9a-f]{64}$")
_GRAPH_INTENT_REF = re.compile(r"^graph-intent-[0-9a-f]{24}$")
_MANIFEST_DB_NAME = "graph_intent_publication_manifest.sqlite3"
_MANIFEST_SCHEMA_VERSION = "graph_intent_publication_manifest.v1"
_PUBLICATION_KIND_W1 = "graph_intent_brief_lifecycle"
_PUBLICATION_KIND_W2 = "graph_intent_discovery_proposal"
_PUBLICATION_KINDS = {_PUBLICATION_KIND_W1, _PUBLICATION_KIND_W2}
_ARTIFACT_ROLES = {
    "graph_intent_brief",
    "source_binding_evidence",
    "successor_link",
    "partial_assessment",
    "proposal_evidence",
}
_STAGED_TEMP_NAME = re.compile(
    r"^\..+\.tx-(?P<pid>[0-9]+)-(?P<tx>[0-9a-f]{32})\.[0-9a-f]{32}\.tmp$"
)


def _encoded(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)


def _captured_json(payload: Any) -> CapturedJsonV1:
    encoded = canonical_json(payload)
    return CapturedJsonV1(
        media_type="application/json",
        canonical_json=encoded,
        bytes_hash=f"sha256:{sha256(encoded.encode('utf-8')).hexdigest()}",
    )


class _PinnedManifestConnection(sqlite3.Connection):
    """SQLite connection that keeps the securely opened manifest DB fd alive."""

    _manifest_fd: int | None = None

    def close(self) -> None:
        descriptor = self._manifest_fd
        self._manifest_fd = None
        try:
            super().close()
        finally:
            if descriptor is not None:
                os.close(descriptor)


class GraphIntentPublicationTransaction:
    """SQLite-coordinated committed-manifest transaction for graph intent files."""

    _local = threading.local()

    def __init__(self, artifact_root: str | Path) -> None:
        self.artifact_root = Path(artifact_root).resolve()
        self.db_path = self.manifest_db_path(self.artifact_root)
        self.transaction_id = uuid4().hex
        self._pending_artifacts: list[dict[str, Any]] = []
        self._pending_by_role: dict[str, dict[str, Any]] = {}
        self._manifest: dict[str, str | None] | None = None
        self._depth = 0
        self._active: GraphIntentPublicationTransaction | None = None
        self._rollback_only = False
        self._rollback_reason: BaseException | None = None

    def __enter__(self) -> GraphIntentPublicationTransaction:
        current = self.current(self.artifact_root)
        if current is not None:
            current._depth += 1
            self._active = current
            return current
        if enabled():
            # Validate the governed home/schema before any staged artifact IO.
            with closing(self._connect_manifest_db(self.artifact_root)):pass
        self._depth = 1
        self._active = self
        if not enabled():self._recover_stale_staged_temps(self.artifact_root)
        self._transactions()[self.artifact_root.as_posix()] = self
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> bool:
        active = self._active
        if active is None:
            return False
        active._depth -= 1
        if active._depth > 0:
            if exc_type is not None:
                active._mark_rollback_only(exc)
            return False
        self._transactions().pop(active.artifact_root.as_posix(), None)
        pending_error: BaseException | None = None
        try:
            if exc_type is not None or active._rollback_only:
                active._cleanup_staged_temps()
                if exc_type is None:
                    pending_error = RuntimeError(
                        "graph intent publication transaction marked rollback-only"
                    )
                    if active._rollback_reason is not None:
                        pending_error.__cause__ = active._rollback_reason
            else:
                try:
                    active._commit_manifest_transaction()
                except BaseException:
                    active._cleanup_staged_temps()
                    raise
        finally:
            active._cleanup_staged_temps()
        if pending_error is not None:
            raise pending_error
        return False

    @classmethod
    def current(
        cls, artifact_root: str | Path
    ) -> GraphIntentPublicationTransaction | None:
        key = Path(artifact_root).resolve().as_posix()
        return cls._transactions().get(key)

    @classmethod
    def current_for_path(cls, path: Path) -> GraphIntentPublicationTransaction | None:
        resolved = path.resolve(strict=False)
        for transaction in cls._transactions().values():
            if resolved == transaction.artifact_root or resolved.is_relative_to(
                transaction.artifact_root
            ):
                return transaction
        return None

    def require_no_conflict(self, path: Path, encoded: str, error: str) -> None:
        path = self._governed_artifact_path(path)
        if path.exists() and path.read_text(encoding="utf-8") != encoded:
            raise ValueError(error)

    def publish(
        self, path: Path, encoded: str, error: str, *, role: str | None = None
    ) -> bool:
        path = self._governed_artifact_path(path)
        artifact_hash = f"sha256:{sha256(encoded.encode('utf-8')).hexdigest()}"
        role = role or self._infer_artifact_role(path)
        self._validate_artifact_role(role)
        if path.exists():
            if f"sha256:{sha256(path.read_bytes()).hexdigest()}" != artifact_hash:
                raise ValueError(error)
            self._record_artifact(
                role=role,
                path=path,
                encoded=encoded,
                artifact_hash=artifact_hash,
                temp_path=None,
                error=error,
            )
            return False
        self._ensure_parent_directory(path)
        temp = path.with_name(
            f".{path.name}.tx-{os.getpid()}-{self.transaction_id}.{uuid4().hex}.tmp"
        )
        temp_created = False
        try:
            temp.write_text(encoded, encoding="utf-8")
            temp_created = True
            self._fsync_file(temp)
            self._fsync_directory(temp.parent)
            self._record_artifact(
                role=role,
                path=path,
                encoded=encoded,
                artifact_hash=artifact_hash,
                temp_path=temp,
                error=error,
            )
            return True
        except BaseException:
            if temp_created or temp.exists():
                try:
                    temp.unlink()
                except FileNotFoundError:
                    pass
                self._fsync_directory(temp.parent)
            raise

    @classmethod
    def publish_unlocked(cls, path: Path, encoded: str, error: str) -> bool:
        transaction = cls(cls._infer_artifact_root(path))
        with transaction as active:
            return active.publish(path, encoded, error)

    def record_graph_intent(
        self, brief: GraphIntentBrief, *, source_identity: SourceIdentity
    ) -> None:
        manifest = {
            "publication_kind": self._publication_kind_for_brief(brief),
            "organization_id": brief.source_binding.organization_id,
            "workspace_id": brief.source_binding.workspace_id,
            "lineage_id": brief.source_binding.lineage_id,
            "graph_intent_ref": brief.graph_intent_id,
            "graph_intent_hash": brief.graph_intent_hash,
            "source_identity_ref": source_identity.source_identity_ref,
            "predecessor_graph_intent_ref": brief.predecessor_graph_intent_ref,
        }
        if self._manifest is not None and self._manifest != manifest:
            raise ValueError(
                "graph intent transaction cannot publish multiple manifests"
            )
        self._manifest = manifest

    @staticmethod
    def _publication_kind_for_brief(brief: GraphIntentBrief) -> str:
        proposal_refs = [
            ref
            for ref in brief.evidence_refs
            if ref.startswith("evidence:graph-intent-proposal:")
        ]
        if proposal_refs:
            return _PUBLICATION_KIND_W2
        return _PUBLICATION_KIND_W1

    @classmethod
    def _transactions(cls) -> dict[str, GraphIntentPublicationTransaction]:
        transactions = getattr(cls._local, "transactions", None)
        if transactions is None:
            transactions = {}
            cls._local.transactions = transactions
        return transactions

    @staticmethod
    def manifest_db_path(artifact_root: Path) -> Path:
        return artifact_root / _MANIFEST_DB_NAME

    @staticmethod
    def _infer_artifact_root(path: Path) -> Path:
        resolved = path.resolve(strict=False)
        for candidate in (resolved, *resolved.parents):
            if candidate.name == "artifacts":
                return candidate
        return resolved.parent

    @classmethod
    def _ensure_parent_directory(cls, path: Path) -> None:
        parent = path.parent
        missing: list[Path] = []
        cursor = parent
        while not cursor.exists():
            missing.append(cursor)
            cursor = cursor.parent
        parent.mkdir(parents=True, exist_ok=True)
        for directory in reversed(missing):
            cls._fsync_directory(directory.parent)
            cls._fsync_directory(directory)

    def _mark_rollback_only(self, reason: BaseException | None) -> None:
        self._rollback_only = True
        if reason is not None and self._rollback_reason is None:
            self._rollback_reason = reason

    def _governed_artifact_path(self, path: Path) -> Path:
        resolved = path.resolve(strict=False)
        if not self._path_is_relative_to(resolved, self.artifact_root):
            raise ValueError("graph intent publication path escapes governed root")
        return resolved

    def _record_artifact(
        self,
        *,
        role: str,
        path: Path,
        encoded: str,
        artifact_hash: str,
        temp_path: Path | None,
        error: str,
    ) -> None:
        relative_path = path.relative_to(self.artifact_root).as_posix()
        existing = self._pending_by_role.get(role)
        artifact = {
            "role": role,
            "path": path,
            "relative_path": relative_path,
            "encoded": encoded,
            "artifact_hash": artifact_hash,
            "temp_path": temp_path,
            "error": error,
        }
        if existing is not None:
            if (
                existing["relative_path"] != relative_path
                or existing["artifact_hash"] != artifact_hash
            ):
                raise ValueError("duplicate graph intent artifact role in transaction")
            return
        self._pending_by_role[role] = artifact
        self._pending_artifacts.append(artifact)

    def _commit_manifest_transaction(self) -> None:
        if not self._pending_artifacts:
            return
        if self._manifest is None:
            raise ValueError("graph intent publication manifest metadata missing")
        manifest_payload = self._manifest_payload()
        connection = self._connect_manifest_db(self.artifact_root)
        began = False
        try:
            begin_write(connection)
            began = True
            try:
                existing = self._existing_manifest(connection)
                if existing is not None:
                    self._validate_existing_manifest(
                        connection, existing, manifest_payload
                    )
                else:
                    decoded_artifacts = self._pending_decoded_artifacts()
                    self._validate_manifest_bundle(
                        manifest_payload,
                        decoded_by_role=decoded_artifacts,
                    )
                    self._validate_manifest_authority_bindings(
                        manifest_payload,
                        decoded_artifacts,
                    )
                    self._validate_manifest_predecessor_authority(
                        connection,
                        manifest_payload,
                        decoded_artifacts,
                        artifact_root=self.artifact_root,
                    )
                    self._publish_final_paths(connection)
                    self._insert_committed_manifest(connection, manifest_payload)
            except BaseException:
                connection.rollback()
                raise
            connection.commit()
        except BaseException:
            if began and in_transaction(connection):
                connection.rollback()
            raise
        finally:
            connection.close()

    @classmethod
    def _connect_manifest_db(cls, artifact_root: Path) -> sqlite3.Connection:
        if enabled():
            connection=connect_authority('standalone',cls.manifest_db_path(artifact_root),autocommit=True)
            try:
                require_tables(connection,'graph_intent_publication_manifests','graph_intent_publication_artifacts')
                cls._recover_stale_staged_temps(artifact_root)
            except BaseException:
                connection.close()
                raise
            return connection
        cls._recover_stale_staged_temps(artifact_root)
        _db_path, descriptor = cls._open_manifest_db_fd(artifact_root)
        connection: _PinnedManifestConnection | None = None
        descriptor_attached = False
        proc_fd = Path("/proc/self/fd")
        if not proc_fd.is_dir():
            os.close(descriptor)
            raise ValueError("graph intent manifest db fd-pinned sqlite unavailable")
        try:
            connection = sqlite3.connect(
                f"file:/proc/self/fd/{descriptor}?mode=rw",
                timeout=0.2,
                isolation_level=None,
                uri=True,
                factory=_PinnedManifestConnection,
            )
            connection._manifest_fd = descriptor
            descriptor_attached = True
            connection.row_factory = sqlite3.Row
            cls._validate_pinned_manifest_db(
                db_path=_db_path,
                descriptor=descriptor,
                artifact_root=artifact_root,
            )
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA busy_timeout = 200")
            for attempt in range(20):
                try:
                    connection.execute("PRAGMA journal_mode = WAL")
                    cls._ensure_manifest_schema(
                        connection,
                        artifact_root=artifact_root,
                    )
                    break
                except sqlite3.OperationalError as exc:
                    if "locked" not in str(exc).lower() or attempt == 19:
                        raise
                    time.sleep(0.05)
        except BaseException:
            if connection is not None:
                connection.close()
            elif not descriptor_attached:
                os.close(descriptor)
            raise
        return connection

    @classmethod
    def _open_manifest_db_fd(cls, artifact_root: Path) -> tuple[Path, int]:
        artifact_root = artifact_root.resolve(strict=False)
        db_path = cls.manifest_db_path(artifact_root)
        if db_path.resolve(strict=False) != db_path:
            raise ValueError("graph intent manifest db path escapes governed root")
        cls._ensure_parent_directory(db_path)
        if db_path.is_symlink():
            raise ValueError("graph intent manifest db path is unsafe")
        if not cls._path_is_relative_to(db_path, artifact_root):
            raise ValueError("graph intent manifest db path escapes governed root")
        if not artifact_root.is_dir() or artifact_root.is_symlink():
            raise ValueError("graph intent manifest db path is unsafe")
        cls._validate_manifest_parent_chain(artifact_root, db_path.parent)
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(db_path, os.O_RDWR | nofollow)
        except FileNotFoundError:
            try:
                descriptor = os.open(
                    db_path,
                    os.O_RDWR | os.O_CREAT | os.O_EXCL | nofollow,
                    0o600,
                )
            except FileExistsError:
                try:
                    descriptor = os.open(db_path, os.O_RDWR | nofollow)
                except OSError as exc:
                    raise ValueError("graph intent manifest db path is unsafe") from exc
            except OSError as exc:
                raise ValueError("graph intent manifest db path is unsafe") from exc
            else:
                cls._fsync_directory(db_path.parent)
        except OSError as exc:
            raise ValueError("graph intent manifest db path is unsafe") from exc
        try:
            status = os.fstat(descriptor)
            if not stat.S_ISREG(status.st_mode):
                raise ValueError("graph intent manifest db path is unsafe")
            if hasattr(os, "geteuid") and status.st_uid != os.geteuid():
                raise ValueError("graph intent manifest db owner mismatch")
            if stat.S_IMODE(status.st_mode) != 0o600:
                os.fchmod(descriptor, 0o600)
                cls._fsync_directory(db_path.parent)
            fd_resolved = Path(f"/proc/self/fd/{descriptor}").resolve(strict=True)
            if not cls._path_is_relative_to(fd_resolved, artifact_root):
                raise ValueError("graph intent manifest db path escapes governed root")
            return db_path, descriptor
        except BaseException:
            os.close(descriptor)
            raise

    @classmethod
    def _validate_pinned_manifest_db(
        cls,
        *,
        db_path: Path,
        descriptor: int,
        artifact_root: Path,
    ) -> None:
        artifact_root = artifact_root.resolve(strict=False)
        try:
            fd_status = os.fstat(descriptor)
            fd_resolved = Path(f"/proc/self/fd/{descriptor}").resolve(strict=True)
        except OSError as exc:
            raise ValueError("graph intent manifest db path is unsafe") from exc
        if not stat.S_ISREG(fd_status.st_mode):
            raise ValueError("graph intent manifest db path is unsafe")
        if hasattr(os, "geteuid") and fd_status.st_uid != os.geteuid():
            raise ValueError("graph intent manifest db owner mismatch")
        fd_mode = stat.S_IMODE(fd_status.st_mode)
        if fd_mode != 0o600:
            raise ValueError("graph intent manifest db mode mismatch")
        if not cls._path_is_relative_to(fd_resolved, artifact_root):
            raise ValueError("graph intent manifest db path escapes governed root")
        if db_path.is_symlink():
            raise ValueError("graph intent manifest db path is unsafe")
        try:
            path_status = os.lstat(db_path)
        except OSError as exc:
            raise ValueError("graph intent manifest db path is unsafe") from exc
        if not stat.S_ISREG(path_status.st_mode):
            raise ValueError("graph intent manifest db path is unsafe")
        path_resolved = db_path.resolve(strict=True)
        if not cls._path_is_relative_to(path_resolved, artifact_root):
            raise ValueError("graph intent manifest db path escapes governed root")
        path_mode = stat.S_IMODE(path_status.st_mode)
        expected_identity = (
            fd_status.st_dev,
            fd_status.st_ino,
            fd_status.st_uid,
            fd_mode,
        )
        actual_identity = (
            path_status.st_dev,
            path_status.st_ino,
            path_status.st_uid,
            path_mode,
        )
        if actual_identity != expected_identity:
            raise ValueError("graph intent manifest db path changed during open")

    @classmethod
    def _manifest_db_identity(
        cls, db_path: Path, artifact_root: Path
    ) -> tuple[int, int, int, int, str]:
        if db_path.is_symlink():
            raise ValueError("graph intent manifest db path is unsafe")
        try:
            status = os.lstat(db_path)
        except OSError as exc:
            raise ValueError("graph intent manifest db path is unsafe") from exc
        if not stat.S_ISREG(status.st_mode):
            raise ValueError("graph intent manifest db path is unsafe")
        resolved = db_path.resolve(strict=True)
        if not cls._path_is_relative_to(resolved, artifact_root):
            raise ValueError("graph intent manifest db path escapes governed root")
        mode = stat.S_IMODE(status.st_mode)
        if hasattr(os, "geteuid") and status.st_uid != os.geteuid():
            raise ValueError("graph intent manifest db owner mismatch")
        if mode != 0o600:
            raise ValueError("graph intent manifest db mode mismatch")
        return (
            status.st_dev,
            status.st_ino,
            status.st_uid,
            mode,
            resolved.as_posix(),
        )

    @classmethod
    def _validate_manifest_db_identity(
        cls,
        db_path: Path,
        expected: tuple[int, int, int, int, str],
        artifact_root: Path,
    ) -> None:
        actual = cls._manifest_db_identity(db_path, artifact_root)
        if actual != expected:
            raise ValueError("graph intent manifest db path changed during open")

    @classmethod
    def _validate_manifest_parent_chain(cls, root: Path, parent: Path) -> None:
        root = root.resolve(strict=False)
        parent = parent.resolve(strict=False)
        if not cls._path_is_relative_to(parent, root):
            raise ValueError("graph intent manifest db path escapes governed root")
        current = root
        for part in parent.relative_to(root).parts:
            current = current / part
            if current.is_symlink() or not current.is_dir():
                raise ValueError("graph intent manifest db path is unsafe")

    @classmethod
    def _recover_stale_staged_temps(cls, artifact_root: Path) -> None:
        artifact_root = artifact_root.resolve(strict=False)
        if not artifact_root.exists() or not artifact_root.is_dir():
            return
        for temp in artifact_root.rglob("*.tmp"):
            if temp.is_symlink():
                continue
            match = _STAGED_TEMP_NAME.fullmatch(temp.name)
            if match is None:
                continue
            try:
                pid = int(match.group("pid"))
            except ValueError:
                continue
            if cls._pid_is_alive(pid):
                continue
            if not cls._path_is_relative_to(temp.resolve(strict=False), artifact_root):
                continue
            try:
                temp.unlink()
            except FileNotFoundError:
                continue
            else:
                cls._fsync_directory(temp.parent)

    @staticmethod
    def _pid_is_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except OSError as exc:
            return exc.errno == errno.EPERM
        return True

    @classmethod
    def _ensure_manifest_schema(
        cls,
        connection: sqlite3.Connection,
        *,
        artifact_root: Path,
    ) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS graph_intent_publication_manifests (
                manifest_id TEXT PRIMARY KEY,
                schema_version TEXT NOT NULL,
                publication_kind TEXT NOT NULL,
                organization_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                lineage_id TEXT NOT NULL,
                graph_intent_ref TEXT NOT NULL,
                graph_intent_hash TEXT NOT NULL,
                source_identity_ref TEXT NOT NULL,
                predecessor_graph_intent_ref TEXT,
                manifest_hash TEXT NOT NULL UNIQUE,
                manifest_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE (organization_id, workspace_id, graph_intent_ref),
                UNIQUE (organization_id, workspace_id, graph_intent_hash)
            );
            CREATE TABLE IF NOT EXISTS graph_intent_publication_artifacts (
                manifest_id TEXT NOT NULL,
                organization_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                graph_intent_ref TEXT NOT NULL,
                artifact_role TEXT NOT NULL,
                artifact_path TEXT NOT NULL,
                artifact_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (manifest_id, artifact_role),
                FOREIGN KEY (manifest_id)
                    REFERENCES graph_intent_publication_manifests(manifest_id)
                    ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS graph_intent_publication_lineage_idx
                ON graph_intent_publication_manifests (
                    organization_id, workspace_id, lineage_id
                );
            CREATE INDEX IF NOT EXISTS graph_intent_publication_source_idx
                ON graph_intent_publication_manifests (
                    organization_id, workspace_id, source_identity_ref
                );
            CREATE INDEX IF NOT EXISTS graph_intent_publication_predecessor_idx
                ON graph_intent_publication_manifests (
                    organization_id, workspace_id, predecessor_graph_intent_ref
                );
            """
        )
        columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(graph_intent_publication_manifests)"
            )
        }
        if "publication_kind" not in columns:
            connection.execute(
                "ALTER TABLE graph_intent_publication_manifests "
                "ADD COLUMN publication_kind TEXT"
            )
        cls._migrate_global_artifact_path_uniqueness(connection)
        cls._migrate_previous_schema_manifests(connection, artifact_root)

    @classmethod
    def _migrate_global_artifact_path_uniqueness(
        cls,
        connection: sqlite3.Connection,
    ) -> None:
        if not cls._artifact_table_has_global_path_unique(connection):
            return
        begin_write(connection)
        try:
            conflicts = connection.execute(
                """SELECT artifact_path
                FROM graph_intent_publication_artifacts
                GROUP BY artifact_path
                HAVING COUNT(DISTINCT artifact_hash) > 1
                LIMIT 1"""
            ).fetchone()
            if conflicts is not None:
                raise ValueError("committed graph intent artifact path hash mismatch")
            connection.execute(
                """ALTER TABLE graph_intent_publication_artifacts
                RENAME TO graph_intent_publication_artifacts_legacy_unique"""
            )
            connection.execute(
                """CREATE TABLE graph_intent_publication_artifacts (
                    manifest_id TEXT NOT NULL,
                    organization_id TEXT NOT NULL,
                    workspace_id TEXT NOT NULL,
                    graph_intent_ref TEXT NOT NULL,
                    artifact_role TEXT NOT NULL,
                    artifact_path TEXT NOT NULL,
                    artifact_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (manifest_id, artifact_role),
                    FOREIGN KEY (manifest_id)
                        REFERENCES graph_intent_publication_manifests(manifest_id)
                        ON DELETE CASCADE
                )"""
            )
            connection.execute(
                """INSERT INTO graph_intent_publication_artifacts (
                    manifest_id, organization_id, workspace_id, graph_intent_ref,
                    artifact_role, artifact_path, artifact_hash, created_at
                )
                SELECT manifest_id, organization_id, workspace_id, graph_intent_ref,
                    artifact_role, artifact_path, artifact_hash, created_at
                FROM graph_intent_publication_artifacts_legacy_unique"""
            )
            connection.execute(
                "DROP TABLE graph_intent_publication_artifacts_legacy_unique"
            )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise

    @classmethod
    def _artifact_table_has_global_path_unique(
        cls,
        connection: sqlite3.Connection,
    ) -> bool:
        for index in connection.execute(
            "PRAGMA index_list(graph_intent_publication_artifacts)"
        ):
            if not index["unique"]:
                continue
            identifier = cls._sqlite_identifier(index["name"])
            columns = [
                row["name"]
                for row in connection.execute(f"PRAGMA index_info({identifier})")
            ]
            if columns == ["artifact_path"]:
                return True
        return False

    @staticmethod
    def _sqlite_identifier(value: str) -> str:
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value) is None:
            raise ValueError("graph intent manifest db schema identifier is invalid")
        return value

    @classmethod
    def _migrate_previous_schema_manifests(
        cls,
        connection: sqlite3.Connection,
        artifact_root: Path,
    ) -> None:
        rows = connection.execute(
            """SELECT * FROM graph_intent_publication_manifests
            WHERE publication_kind IS NULL OR publication_kind = ''
            ORDER BY organization_id, workspace_id, lineage_id,
                graph_intent_ref, manifest_id"""
        ).fetchall()
        if not rows:
            return
        remaining = {row["manifest_id"]: row for row in rows}
        begin_write(connection)
        try:
            while remaining:
                progressed = False
                for manifest_id, row in sorted(
                    list(remaining.items()),
                    key=lambda item: cls._previous_schema_migration_sort_key(item[1]),
                ):
                    if not cls._previous_schema_predecessor_ready(
                        connection,
                        row,
                    ):
                        continue
                    payload = cls._typed_payload_from_previous_schema(
                        connection,
                        row,
                        artifact_root,
                    )
                    connection.execute(
                        sql_query("""UPDATE graph_intent_publication_manifests
                        SET publication_kind = ?, manifest_json = ?,
                            manifest_hash = ?
                        WHERE manifest_id = ?""",'UPDATE graph_intent_publication_manifests\n                        SET publication_kind = %s, manifest_json = %s,\n                            manifest_hash = %s\n                        WHERE manifest_id = %s'),
                        (
                            payload["publication_kind"],
                            json.dumps(
                                payload,
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                            f"sha256:{content_hash(payload)}",
                            row["manifest_id"],
                        ),
                    )
                    remaining.pop(manifest_id)
                    progressed = True
                if not progressed:
                    raise ValueError(
                        "legacy graph intent publication manifest incomplete"
                    )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise

    @classmethod
    def _previous_schema_migration_sort_key(
        cls,
        row: sqlite3.Row,
    ) -> tuple[str, str, str, str, str]:
        return (
            str(row["organization_id"]),
            str(row["workspace_id"]),
            str(row["lineage_id"]),
            str(row["graph_intent_ref"]),
            str(row["manifest_id"]),
        )

    @classmethod
    def _previous_schema_predecessor_ready(
        cls,
        connection: sqlite3.Connection,
        row: sqlite3.Row,
    ) -> bool:
        predecessor_ref = row["predecessor_graph_intent_ref"]
        if predecessor_ref is None:
            return True
        predecessor = connection.execute(
            sql_query("""SELECT * FROM graph_intent_publication_manifests
            WHERE organization_id = ? AND workspace_id = ?
              AND graph_intent_ref = ?""",'SELECT * FROM graph_intent_publication_manifests\n            WHERE organization_id = %s AND workspace_id = %s\n              AND graph_intent_ref = %s'),
            (
                row["organization_id"],
                row["workspace_id"],
                predecessor_ref,
            ),
        ).fetchone()
        if predecessor is None:
            raise ValueError("legacy graph intent publication manifest incomplete")
        if predecessor["lineage_id"] != row["lineage_id"]:
            raise ValueError("legacy graph intent publication manifest incomplete")
        return not (
            predecessor["publication_kind"] is None
            or predecessor["publication_kind"] == ""
        )

    @classmethod
    def _typed_payload_from_previous_schema(
        cls,
        connection: sqlite3.Connection,
        row: sqlite3.Row,
        artifact_root: Path,
    ) -> dict[str, Any]:
        try:
            payload = json.loads(row["manifest_json"])
        except json.JSONDecodeError as exc:
            raise ValueError(
                "legacy graph intent publication manifest incomplete"
            ) from exc
        if payload.get("publication_kind") is not None:
            raise ValueError("legacy graph intent publication manifest incomplete")
        if row["manifest_hash"] != f"sha256:{content_hash(payload)}":
            raise ValueError("legacy graph intent publication manifest incomplete")
        for field in (
            "schema_version",
            "organization_id",
            "workspace_id",
            "lineage_id",
            "graph_intent_ref",
            "graph_intent_hash",
            "source_identity_ref",
            "predecessor_graph_intent_ref",
        ):
            if payload.get(field) != row[field]:
                raise ValueError("legacy graph intent publication manifest incomplete")
        rows = connection.execute(
            sql_query("""SELECT * FROM graph_intent_publication_artifacts
            WHERE manifest_id = ?
            ORDER BY artifact_role, artifact_path""",'SELECT * FROM graph_intent_publication_artifacts\n            WHERE manifest_id = %s\n            ORDER BY artifact_role, artifact_path'),
            (row["manifest_id"],),
        ).fetchall()
        row_artifacts = [
            {
                "artifact_role": artifact_row["artifact_role"],
                "artifact_path": artifact_row["artifact_path"],
                "artifact_hash": artifact_row["artifact_hash"],
            }
            for artifact_row in rows
        ]
        try:
            payload_artifacts = sorted(
                cls._manifest_artifacts_by_role(payload).values(),
                key=lambda item: (item["artifact_role"], item["artifact_path"]),
            )
        except ValueError as exc:
            raise ValueError(
                "legacy graph intent publication manifest incomplete"
            ) from exc
        if row_artifacts != payload_artifacts:
            raise ValueError("legacy graph intent publication manifest incomplete")
        decoded_by_role: dict[str, Any] = {}
        for artifact_row in rows:
            if (
                artifact_row["organization_id"] != row["organization_id"]
                or artifact_row["workspace_id"] != row["workspace_id"]
                or artifact_row["graph_intent_ref"] != row["graph_intent_ref"]
            ):
                raise ValueError("legacy graph intent publication manifest incomplete")
            try:
                path = cls._verify_committed_artifact_row(
                    artifact_root,
                    artifact_row,
                )
                decoded_by_role[artifact_row["artifact_role"]] = json.loads(
                    path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                raise ValueError(
                    "legacy graph intent publication manifest incomplete"
                ) from exc
        brief_payload = decoded_by_role.get("graph_intent_brief")
        if not isinstance(brief_payload, dict):
            raise ValueError("legacy graph intent publication manifest incomplete")
        refs = brief_payload.get("evidence_refs", [])
        if not isinstance(refs, list):
            raise ValueError("legacy graph intent publication manifest incomplete")
        proposal_refs = [
            ref
            for ref in refs
            if isinstance(ref, str)
            and ref.startswith("evidence:graph-intent-proposal:")
        ]
        payload["publication_kind"] = (
            _PUBLICATION_KIND_W2 if proposal_refs else _PUBLICATION_KIND_W1
        )
        try:
            cls._validate_manifest_bundle(payload, decoded_by_role=decoded_by_role)
            cls._validate_manifest_authority_bindings(payload, decoded_by_role)
            cls._validate_manifest_predecessor_authority(
                connection,
                payload,
                decoded_by_role,
                artifact_root=artifact_root,
            )
        except ValueError as exc:
            raise ValueError(
                "legacy graph intent publication manifest incomplete"
            ) from exc
        return payload

    def _publish_final_paths(self, connection: sqlite3.Connection) -> None:
        del connection  # SQLite transaction is the authority; paths are verified here.
        for artifact in self._pending_artifacts:
            path: Path = artifact["path"]
            expected_hash = str(artifact["artifact_hash"])
            if path.exists():
                actual_hash = f"sha256:{sha256(path.read_bytes()).hexdigest()}"
                if actual_hash != expected_hash:
                    raise ValueError(str(artifact["error"]))
                continue
            temp_path: Path | None = artifact["temp_path"]
            if temp_path is None or not temp_path.is_file():
                raise ValueError("graph intent staged artifact missing")
            try:
                os.link(temp_path, path)
            except FileExistsError:
                actual_hash = f"sha256:{sha256(path.read_bytes()).hexdigest()}"
                if actual_hash != expected_hash:
                    raise ValueError(str(artifact["error"]))
            self._fsync_directory(path.parent)

    def _insert_committed_manifest(
        self,
        connection: sqlite3.Connection,
        manifest_payload: dict[str, Any],
    ) -> None:
        assert self._manifest is not None
        manifest_hash = f"sha256:{content_hash(manifest_payload)}"
        manifest_id = f"graph-intent-publication-{manifest_hash[7:31]}"
        created_at = datetime.now(timezone.utc).isoformat()
        connection.execute(
            sql_query("""INSERT INTO graph_intent_publication_manifests (
                manifest_id, schema_version, publication_kind, organization_id,
                workspace_id, lineage_id, graph_intent_ref, graph_intent_hash,
                source_identity_ref, predecessor_graph_intent_ref,
                manifest_hash, manifest_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",'INSERT INTO graph_intent_publication_manifests (\n                manifest_id, schema_version, publication_kind, organization_id,\n                workspace_id, lineage_id, graph_intent_ref, graph_intent_hash,\n                source_identity_ref, predecessor_graph_intent_ref,\n                manifest_hash, manifest_json, created_at\n            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)'),
            (
                manifest_id,
                _MANIFEST_SCHEMA_VERSION,
                self._manifest["publication_kind"],
                self._manifest["organization_id"],
                self._manifest["workspace_id"],
                self._manifest["lineage_id"],
                self._manifest["graph_intent_ref"],
                self._manifest["graph_intent_hash"],
                self._manifest["source_identity_ref"],
                self._manifest["predecessor_graph_intent_ref"],
                manifest_hash,
                json.dumps(
                    manifest_payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                created_at,
            ),
        )
        for artifact in manifest_payload["artifacts"]:
            existing_hashes = {
                row["artifact_hash"]
                for row in connection.execute(
                    sql_query("""SELECT DISTINCT artifact_hash
                    FROM graph_intent_publication_artifacts
                    WHERE artifact_path = ?""",'SELECT DISTINCT artifact_hash\n                    FROM graph_intent_publication_artifacts\n                    WHERE artifact_path = %s'),
                    (artifact["artifact_path"],),
                )
            }
            if existing_hashes and existing_hashes != {artifact["artifact_hash"]}:
                raise ValueError("committed graph intent artifact path hash mismatch")
            connection.execute(
                sql_query("""INSERT INTO graph_intent_publication_artifacts (
                    manifest_id, organization_id, workspace_id, graph_intent_ref,
                    artifact_role, artifact_path, artifact_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",'INSERT INTO graph_intent_publication_artifacts (\n                    manifest_id, organization_id, workspace_id, graph_intent_ref,\n                    artifact_role, artifact_path, artifact_hash, created_at\n                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)'),
                (
                    manifest_id,
                    self._manifest["organization_id"],
                    self._manifest["workspace_id"],
                    self._manifest["graph_intent_ref"],
                    artifact["artifact_role"],
                    artifact["artifact_path"],
                    artifact["artifact_hash"],
                    created_at,
                ),
            )

    def _existing_manifest(self, connection: sqlite3.Connection) -> sqlite3.Row | None:
        assert self._manifest is not None
        rows = connection.execute(
            sql_query("""SELECT * FROM graph_intent_publication_manifests
            WHERE organization_id = ? AND workspace_id = ?
              AND (graph_intent_ref = ? OR graph_intent_hash = ?)
            ORDER BY created_at, manifest_id""",'SELECT * FROM graph_intent_publication_manifests\n            WHERE organization_id = %s AND workspace_id = %s\n              AND (graph_intent_ref = %s OR graph_intent_hash = %s)\n            ORDER BY created_at, manifest_id'),
            (
                self._manifest["organization_id"],
                self._manifest["workspace_id"],
                self._manifest["graph_intent_ref"],
                self._manifest["graph_intent_hash"],
            ),
        ).fetchall()
        if not rows:
            return None
        manifest_ids = {row["manifest_id"] for row in rows}
        if len(manifest_ids) != 1:
            raise ValueError("graph intent publication manifest conflict")
        return rows[0]

    def _validate_existing_manifest(
        self,
        connection: sqlite3.Connection,
        existing: sqlite3.Row,
        expected: dict[str, Any],
    ) -> None:
        assert self._manifest is not None
        existing_payload = self._validate_committed_manifest_integrity(
            connection, existing, artifact_root=self.artifact_root
        )
        if existing_payload != expected:
            raise ValueError("graph intent publication manifest conflict")

    def _manifest_payload(self) -> dict[str, Any]:
        assert self._manifest is not None
        artifacts = [
            {
                "artifact_role": str(artifact["role"]),
                "artifact_path": str(artifact["relative_path"]),
                "artifact_hash": str(artifact["artifact_hash"]),
            }
            for artifact in sorted(
                self._pending_artifacts,
                key=lambda item: (str(item["role"]), str(item["relative_path"])),
            )
        ]
        return {
            "schema_version": _MANIFEST_SCHEMA_VERSION,
            **self._manifest,
            "artifacts": artifacts,
        }

    def _pending_decoded_artifacts(self) -> dict[str, Any]:
        decoded: dict[str, Any] = {}
        for artifact in self._pending_artifacts:
            role = str(artifact["role"])
            try:
                decoded[role] = json.loads(str(artifact["encoded"]))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "graph intent publication artifact is invalid JSON"
                ) from exc
        return decoded

    @classmethod
    def _validate_committed_manifest_integrity(
        cls,
        connection: sqlite3.Connection,
        manifest: sqlite3.Row,
        *,
        artifact_root: Path,
        _visited_manifest_ids: set[str] | None = None,
    ) -> dict[str, Any]:
        visited_manifest_ids = _visited_manifest_ids or set()
        manifest_id = manifest["manifest_id"]
        if manifest_id in visited_manifest_ids:
            raise ValueError("graph intent predecessor publication cycle")
        visited_manifest_ids.add(manifest_id)
        if "publication_kind" not in manifest.keys():
            raise ValueError("graph intent publication manifest kind missing")
        try:
            payload = json.loads(manifest["manifest_json"])
        except json.JSONDecodeError as exc:
            raise ValueError(
                "graph intent publication manifest JSON is invalid"
            ) from exc
        expected_hash = f"sha256:{content_hash(payload)}"
        if manifest["manifest_hash"] != expected_hash:
            raise ValueError("graph intent publication manifest hash mismatch")
        comparisons = {
            "schema_version": _MANIFEST_SCHEMA_VERSION,
            "publication_kind": manifest["publication_kind"],
            "organization_id": manifest["organization_id"],
            "workspace_id": manifest["workspace_id"],
            "lineage_id": manifest["lineage_id"],
            "graph_intent_ref": manifest["graph_intent_ref"],
            "graph_intent_hash": manifest["graph_intent_hash"],
            "source_identity_ref": manifest["source_identity_ref"],
            "predecessor_graph_intent_ref": manifest["predecessor_graph_intent_ref"],
        }
        for field, value in comparisons.items():
            if payload.get(field) != value:
                raise ValueError("graph intent publication manifest JSON mismatch")
        rows = connection.execute(
            sql_query("""SELECT * FROM graph_intent_publication_artifacts
            WHERE manifest_id = ?
            ORDER BY artifact_role, artifact_path""",'SELECT * FROM graph_intent_publication_artifacts\n            WHERE manifest_id = %s\n            ORDER BY artifact_role, artifact_path'),
            (manifest["manifest_id"],),
        ).fetchall()
        row_artifacts = [
            {
                "artifact_role": row["artifact_role"],
                "artifact_path": row["artifact_path"],
                "artifact_hash": row["artifact_hash"],
            }
            for row in rows
        ]
        payload_artifacts = sorted(
            cls._manifest_artifacts_by_role(payload).values(),
            key=lambda item: (item["artifact_role"], item["artifact_path"]),
        )
        if row_artifacts != payload_artifacts:
            raise ValueError("graph intent publication manifest artifact rows mismatch")
        for row in rows:
            if (
                row["organization_id"] != manifest["organization_id"]
                or row["workspace_id"] != manifest["workspace_id"]
                or row["graph_intent_ref"] != manifest["graph_intent_ref"]
            ):
                raise ValueError(
                    "graph intent publication manifest artifact rows mismatch"
                )
            shared_hashes = {
                item["artifact_hash"]
                for item in connection.execute(
                    sql_query("""SELECT DISTINCT artifact_hash
                    FROM graph_intent_publication_artifacts
                    WHERE artifact_path = ?""",'SELECT DISTINCT artifact_hash\n                    FROM graph_intent_publication_artifacts\n                    WHERE artifact_path = %s'),
                    (row["artifact_path"],),
                )
            }
            if shared_hashes != {row["artifact_hash"]}:
                raise ValueError("committed graph intent artifact path hash mismatch")
        decoded_by_role: dict[str, Any] = {}
        for row in rows:
            path = cls._verify_committed_artifact_row(artifact_root, row)
            try:
                decoded_by_role[row["artifact_role"]] = json.loads(
                    path.read_text(encoding="utf-8")
                )
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "committed graph intent artifact is invalid JSON"
                ) from exc
        cls._validate_manifest_bundle(payload, decoded_by_role=decoded_by_role)
        cls._validate_manifest_authority_bindings(payload, decoded_by_role)
        cls._validate_manifest_predecessor_authority(
            connection,
            payload,
            decoded_by_role,
            artifact_root=artifact_root,
            _visited_manifest_ids=visited_manifest_ids,
        )
        return payload

    @classmethod
    def _validate_manifest_bundle(
        cls,
        payload: dict[str, Any],
        *,
        decoded_by_role: dict[str, Any] | None = None,
    ) -> None:
        decoded_by_role = decoded_by_role or {}
        kind = payload.get("publication_kind")
        if payload.get("schema_version") != _MANIFEST_SCHEMA_VERSION:
            raise ValueError("graph intent publication manifest schema mismatch")
        if kind not in _PUBLICATION_KINDS:
            raise ValueError("graph intent publication manifest kind is invalid")
        for field in (
            "organization_id",
            "workspace_id",
            "lineage_id",
            "graph_intent_ref",
            "graph_intent_hash",
            "source_identity_ref",
        ):
            if not isinstance(payload.get(field), str) or not payload[field]:
                raise ValueError("graph intent publication manifest field is invalid")
        if _GRAPH_INTENT_REF.fullmatch(payload["graph_intent_ref"]) is None:
            raise ValueError("graph intent publication manifest ref is invalid")
        if _SHA256_REF.fullmatch(payload["graph_intent_hash"]) is None:
            raise ValueError("graph intent publication manifest hash is invalid")
        artifacts = cls._manifest_artifacts_by_role(payload)
        expected_paths = cls._expected_artifact_paths(payload, decoded_by_role)
        if set(artifacts) != set(expected_paths):
            raise ValueError("required graph intent publication roles mismatch")
        for role, expected_path in expected_paths.items():
            artifact = artifacts[role]
            if artifact["artifact_path"] != expected_path:
                raise ValueError(
                    "graph intent publication manifest canonical path mismatch"
                )
            if _SHA256_REF.fullmatch(artifact["artifact_hash"]) is None:
                raise ValueError(
                    "graph intent publication manifest artifact hash invalid"
                )

    @classmethod
    def _manifest_artifacts_by_role(
        cls, payload: dict[str, Any]
    ) -> dict[str, dict[str, str]]:
        raw_artifacts = payload.get("artifacts")
        if not isinstance(raw_artifacts, list):
            raise ValueError("graph intent publication manifest artifacts invalid")
        artifacts: dict[str, dict[str, str]] = {}
        for item in raw_artifacts:
            if not isinstance(item, dict):
                raise ValueError("graph intent publication manifest artifacts invalid")
            role = item.get("artifact_role")
            artifact_path = item.get("artifact_path")
            artifact_hash = item.get("artifact_hash")
            if (
                not isinstance(role, str)
                or not isinstance(artifact_path, str)
                or not isinstance(artifact_hash, str)
            ):
                raise ValueError("graph intent publication manifest artifacts invalid")
            cls._validate_artifact_role(role)
            if role in artifacts:
                raise ValueError("duplicate graph intent artifact role in manifest")
            artifacts[role] = {
                "artifact_role": role,
                "artifact_path": artifact_path,
                "artifact_hash": artifact_hash,
            }
        return artifacts

    @classmethod
    def _expected_artifact_paths(
        cls,
        payload: dict[str, Any],
        decoded_by_role: dict[str, Any],
    ) -> dict[str, str]:
        digest = payload["graph_intent_hash"].removeprefix("sha256:")
        graph_intent_ref = payload["graph_intent_ref"]
        predecessor_ref = payload.get("predecessor_graph_intent_ref")
        base = f"graph-intent-briefs/{digest}"
        expected = {
            "graph_intent_brief": f"{base}/{_ARTIFACT_NAME}",
            "source_binding_evidence": f"{base}/{_BINDING_NAME}",
        }
        if predecessor_ref is not None:
            if (
                not isinstance(predecessor_ref, str)
                or _GRAPH_INTENT_REF.fullmatch(predecessor_ref) is None
            ):
                raise ValueError("graph intent publication predecessor ref is invalid")
            expected["successor_link"] = (
                f"graph-intent-briefs/successor-links/{predecessor_ref}/"
                f"{graph_intent_ref}.json"
            )
        brief_payload = decoded_by_role.get("graph_intent_brief")
        proposal_refs: list[str] = []
        if isinstance(brief_payload, dict):
            raw_refs = brief_payload.get("evidence_refs", [])
            if not isinstance(raw_refs, list):
                raise ValueError("graph intent publication brief evidence refs invalid")
            proposal_refs = [
                ref
                for ref in raw_refs
                if isinstance(ref, str)
                and ref.startswith("evidence:graph-intent-proposal:")
            ]
        if payload["publication_kind"] == _PUBLICATION_KIND_W1:
            if proposal_refs:
                raise ValueError("graph intent publication kind mismatch")
            return expected
        if len(proposal_refs) != 1:
            raise ValueError(
                "graph intent must pin exactly one proposal evidence artifact"
            )
        proposal_digest = proposal_refs[0].rsplit(":", 1)[1]
        if re.fullmatch(r"[0-9a-f]{64}", proposal_digest) is None:
            raise ValueError("graph intent proposal evidence ref is malformed")
        expected["proposal_evidence"] = (
            f"graph-intent-proposals/{proposal_digest}/"
            "graph_intent_proposal_evidence.v1.json"
        )
        proposal_payload = decoded_by_role.get("proposal_evidence")
        if isinstance(proposal_payload, dict):
            partial_pin = proposal_payload.get("partial_assessment")
            if partial_pin is not None:
                if (
                    not isinstance(partial_pin, dict)
                    or not isinstance(partial_pin.get("artifact_hash"), str)
                    or not isinstance(partial_pin.get("artifact_ref"), str)
                    or _SHA256_REF.fullmatch(partial_pin["artifact_hash"]) is None
                ):
                    raise ValueError("graph intent partial assessment pin is malformed")
                partial_digest = partial_pin["artifact_hash"].removeprefix("sha256:")
                expected_ref = (
                    "artifacts/graph-intent-partial-assessments/"
                    f"{partial_digest}/graph_intent_partial_assessment.v1.json"
                )
                if partial_pin["artifact_ref"] != expected_ref:
                    raise ValueError(
                        "graph intent partial assessment artifact ref mismatch"
                    )
                expected["partial_assessment"] = expected_ref.removeprefix("artifacts/")
        return expected

    @classmethod
    def _validate_manifest_authority_bindings(
        cls,
        payload: dict[str, Any],
        decoded_by_role: dict[str, Any],
    ) -> None:
        brief = decoded_by_role.get("graph_intent_brief")
        binding = decoded_by_role.get("source_binding_evidence")
        if not isinstance(brief, dict) or not isinstance(binding, dict):
            raise ValueError("graph intent publication manifest authority mismatch")
        source_binding = brief.get("source_binding")
        if not isinstance(source_binding, dict):
            raise ValueError("graph intent publication manifest authority mismatch")
        if (
            brief.get("graph_intent_id") != payload["graph_intent_ref"]
            or brief.get("graph_intent_hash") != payload["graph_intent_hash"]
            or source_binding.get("organization_id") != payload["organization_id"]
            or source_binding.get("workspace_id") != payload["workspace_id"]
            or source_binding.get("lineage_id") != payload["lineage_id"]
            or source_binding.get("source_identity_ref")
            != payload["source_identity_ref"]
        ):
            raise ValueError("graph intent publication manifest authority mismatch")
        if (
            binding.get("organization_id") != payload["organization_id"]
            or binding.get("workspace_id") != payload["workspace_id"]
            or binding.get("lineage_id") != payload["lineage_id"]
            or binding.get("graph_intent_ref") != payload["graph_intent_ref"]
            or binding.get("graph_intent_hash") != payload["graph_intent_hash"]
            or binding.get("canonical_source_ir_ref")
            != source_binding.get("canonical_source_ir_ref")
            or binding.get("canonical_source_ir_hash")
            != source_binding.get("canonical_source_ir_hash")
        ):
            raise ValueError("graph intent publication manifest authority mismatch")
        identity = binding.get("source_identity")
        if (
            not isinstance(identity, dict)
            or identity.get("sourceIdentityRef") != payload["source_identity_ref"]
            or identity.get("sourceHash") != source_binding.get("source_hash")
            or identity.get("organizationId") != payload["organization_id"]
            or identity.get("workspaceId") != payload["workspace_id"]
        ):
            raise ValueError("graph intent publication manifest authority mismatch")
        successor = decoded_by_role.get("successor_link")
        predecessor_ref = payload.get("predecessor_graph_intent_ref")
        if predecessor_ref is None:
            if successor is not None:
                raise ValueError("graph intent publication manifest authority mismatch")
        elif (
            not isinstance(successor, dict)
            or successor.get("predecessor_graph_intent_ref") != predecessor_ref
            or successor.get("successor_graph_intent_ref")
            != payload["graph_intent_ref"]
            or successor.get("successor_graph_intent_hash")
            != payload["graph_intent_hash"]
        ):
            raise ValueError("graph intent publication manifest authority mismatch")
        proposal = decoded_by_role.get("proposal_evidence")
        if payload["publication_kind"] == _PUBLICATION_KIND_W1:
            if proposal is not None or "partial_assessment" in decoded_by_role:
                raise ValueError("graph intent publication manifest authority mismatch")
            return
        if not isinstance(proposal, dict):
            raise ValueError("graph intent publication manifest authority mismatch")
        pins = proposal.get("source_pins")
        if (
            not isinstance(pins, dict)
            or pins.get("source_identity_ref") != payload["source_identity_ref"]
            or pins.get("source_hash") != source_binding.get("source_hash")
            or pins.get("canonical_source_ir_ref")
            != source_binding.get("canonical_source_ir_ref")
            or pins.get("canonical_source_ir_hash")
            != source_binding.get("canonical_source_ir_hash")
        ):
            raise ValueError("graph intent publication manifest authority mismatch")
        assessment = decoded_by_role.get("partial_assessment")
        partial_pin = proposal.get("partial_assessment")
        if partial_pin is None:
            if assessment is not None:
                raise ValueError("graph intent publication manifest authority mismatch")
            return
        if not isinstance(assessment, dict):
            raise ValueError("graph intent publication manifest authority mismatch")
        assessment_source_pins = assessment.get("source_pins")
        if not isinstance(assessment_source_pins, dict):
            raise ValueError("graph intent publication manifest authority mismatch")
        if (
            assessment_source_pins.get("source_identity_ref")
            != payload["source_identity_ref"]
        ):
            raise ValueError("graph intent partial assessment source identity mismatch")
        if assessment_source_pins.get("source_hash") != source_binding.get(
            "source_hash"
        ):
            raise ValueError("graph intent partial assessment source hash mismatch")

    @classmethod
    def _validate_manifest_predecessor_authority(
        cls,
        connection: sqlite3.Connection,
        payload: dict[str, Any],
        decoded_by_role: dict[str, Any],
        *,
        artifact_root: Path,
        _visited_manifest_ids: set[str] | None = None,
    ) -> None:
        brief = decoded_by_role.get("graph_intent_brief")
        successor = decoded_by_role.get("successor_link")
        if not isinstance(brief, dict):
            raise ValueError("graph intent predecessor authority mismatch")
        predecessor_ref = payload.get("predecessor_graph_intent_ref")
        if brief.get("predecessor_graph_intent_ref") != predecessor_ref:
            raise ValueError("graph intent predecessor authority mismatch")
        if predecessor_ref is None:
            return
        if predecessor_ref == payload["graph_intent_ref"]:
            raise ValueError("graph intent predecessor authority mismatch")
        if not isinstance(successor, dict):
            raise ValueError("graph intent predecessor authority mismatch")
        predecessor = connection.execute(
            sql_query("""SELECT * FROM graph_intent_publication_manifests
            WHERE organization_id = ? AND workspace_id = ?
              AND graph_intent_ref = ?""",'SELECT * FROM graph_intent_publication_manifests\n            WHERE organization_id = %s AND workspace_id = %s\n              AND graph_intent_ref = %s'),
            (
                payload["organization_id"],
                payload["workspace_id"],
                predecessor_ref,
            ),
        ).fetchone()
        if predecessor is None:
            raise ValueError("graph intent predecessor publication missing")
        predecessor_payload = cls._validate_committed_manifest_integrity(
            connection,
            predecessor,
            artifact_root=artifact_root,
            _visited_manifest_ids=_visited_manifest_ids,
        )
        if predecessor_payload["lineage_id"] != payload[
            "lineage_id"
        ] or predecessor_payload["graph_intent_hash"] != successor.get(
            "predecessor_graph_intent_hash"
        ):
            raise ValueError("graph intent predecessor authority mismatch")

    def _cleanup_staged_temps(self) -> None:
        for artifact in self._pending_artifacts:
            temp_path: Path | None = artifact.get("temp_path")
            if temp_path is None:
                continue
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass
            else:
                self._fsync_directory(temp_path.parent)

    @classmethod
    def _verify_committed_artifact_row(
        cls, artifact_root: Path, row: sqlite3.Row
    ) -> Path:
        path = (artifact_root / row["artifact_path"]).resolve(strict=False)
        if not cls._path_is_relative_to(path, artifact_root):
            raise ValueError("committed graph intent artifact escapes governed root")
        if not path.is_file():
            raise ValueError("committed graph intent artifact missing")
        actual_hash = f"sha256:{sha256(path.read_bytes()).hexdigest()}"
        if actual_hash != row["artifact_hash"]:
            raise ValueError("committed graph intent artifact hash mismatch")
        return path

    @staticmethod
    def _infer_artifact_role(path: Path) -> str:
        if path.name == _ARTIFACT_NAME:
            return "graph_intent_brief"
        if path.name == _BINDING_NAME:
            return "source_binding_evidence"
        if path.name == "graph_intent_partial_assessment.v1.json":
            return "partial_assessment"
        if path.name == "graph_intent_proposal_evidence.v1.json":
            return "proposal_evidence"
        if "successor-links" in path.parts:
            return "successor_link"
        raise ValueError("graph intent artifact role is unknown")

    @staticmethod
    def _validate_artifact_role(role: str) -> None:
        if role not in _ARTIFACT_ROLES:
            raise ValueError("graph intent artifact role is invalid")

    @staticmethod
    def _path_is_relative_to(path: Path, root: Path) -> bool:
        resolved_path = path.resolve(strict=False)
        resolved_root = root.resolve(strict=False)
        return resolved_path == resolved_root or resolved_path.is_relative_to(
            resolved_root
        )

    @staticmethod
    def _fsync_file(path: Path) -> None:
        with path.open("rb") as handle:
            os.fsync(handle.fileno())

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        try:
            descriptor = os.open(path, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


class GraphIntentBriefStore:
    """Workspace-scoped artifact store that verifies governed source pins."""

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        organization_id: str,
        workspace_id: str,
        source_registry: KnowledgeBuilderStore,
    ) -> None:
        for field, value in (
            ("organization_id", organization_id),
            ("workspace_id", workspace_id),
        ):
            if (
                not value.isascii()
                or _SAFE_SEGMENT.fullmatch(value) is None
                or value in {".", ".."}
            ):
                raise ValueError(
                    f"{field} must be a safe single-segment ASCII identifier"
                )
        self.organization_id = organization_id
        self.workspace_id = workspace_id
        base = Path(workspace_root).resolve()
        self.workspace_path = (
            base
            if base.name == workspace_id and base.parent.name == "workspaces"
            else base / "workspaces" / workspace_id
        )
        self.artifact_root = (self.workspace_path / "artifacts").resolve()
        self.root = (self.artifact_root / "graph-intent-briefs").resolve()
        self.manifest_db_path = GraphIntentPublicationTransaction.manifest_db_path(
            self.artifact_root
        )
        if self.artifact_root not in self.root.parents:
            raise ValueError("graph intent root escapes governed workspace artifacts")
        if base.name != workspace_id or base.parent.name != "workspaces":
            workspaces_root = (base / "workspaces").resolve()
            if workspaces_root not in self.workspace_path.parents:
                raise ValueError("workspace path escapes governed workspace root")
        self.source_registry = source_registry

    def create(
        self,
        brief: GraphIntentBrief,
        *,
        source_identity: SourceIdentity,
    ) -> GraphIntentBrief:
        """Create a proposed root or a proposed source-superseding successor."""

        predecessor = self._validate_predecessor(brief)
        superseded_source_ref = source_identity.supersedes_source_identity_ref
        if predecessor is None:
            if superseded_source_ref is not None:
                raise ValueError("superseding SourceIdentity requires predecessor")
            if brief.review.state != "proposed":
                raise ValueError("new graph intent must start proposed")
        else:
            if superseded_source_ref != predecessor.source_binding.source_identity_ref:
                raise ValueError(
                    "successor SourceIdentity does not supersede predecessor source identity"
                )
            if brief.review.state != "proposed":
                raise ValueError("source successor must restart proposed")
            if predecessor.review.state not in {"accepted", "edited"}:
                raise ValueError(
                    "source successor predecessor must be compiler-eligible"
                )
        return self._persist(brief, source_identity=source_identity)

    def _persist(
        self,
        brief: GraphIntentBrief,
        *,
        source_identity: SourceIdentity,
    ) -> GraphIntentBrief:
        """Persist a verified immutable contract and its replay evidence."""

        self._validate_scope(brief)
        canonical = self._validate_source_binding(brief, source_identity)
        predecessor = self._validate_predecessor(brief)
        if brief.frame_mapping:
            brief.frame_mapping.validate_source(brief.source_binding, canonical)
            mapping = brief.frame_mapping.as_dict()
            if predecessor is None:
                raise ValueError("frame mapping requires native review predecessor")
            proposal = self.get(mapping["proposal_ref"], graph_intent_hash=mapping["proposal_hash"])
            brief.frame_mapping.validate_review(proposal, canonical)

        digest = brief.graph_intent_hash.removeprefix("sha256:")
        artifact_dir = self.root / digest
        target = artifact_dir / _ARTIFACT_NAME
        binding_target = artifact_dir / _BINDING_NAME
        binding_payload = self._binding_evidence(
            brief, source_identity=source_identity, canonical=canonical
        )
        link_target = (
            self._successor_link_path(
                brief.predecessor_graph_intent_ref, brief.graph_intent_id
            )
            if predecessor is not None
            else None
        )
        link_payload = (
            self._successor_link_payload(predecessor, brief)
            if predecessor is not None
            else None
        )
        brief_encoded = _encoded(brief.as_dict())
        binding_encoded = _encoded(binding_payload)
        publications = [
            (
                target,
                brief_encoded,
                "GRAPH_INTENT_ARTIFACT_CONFLICT",
                "graph_intent_brief",
            ),
            (
                binding_target,
                binding_encoded,
                "GRAPH_INTENT_SOURCE_BINDING_CONFLICT",
                "source_binding_evidence",
            ),
        ]
        if link_target is not None and link_payload is not None:
            publications.append(
                (
                    link_target,
                    _encoded(link_payload),
                    "GRAPH_INTENT_SUCCESSOR_LINK_CONFLICT",
                    "successor_link",
                )
            )
        publications.extend(self._carried_w2_evidence_publications(brief, predecessor))
        with GraphIntentPublicationTransaction(self.artifact_root) as transaction:
            for path, encoded, error, _role in publications:
                transaction.require_no_conflict(path, encoded, error)
            for path, encoded, error, role in publications:
                transaction.publish(path, encoded, error, role=role)
            transaction.record_graph_intent(brief, source_identity=source_identity)
        return brief

    def get(
        self,
        graph_intent_ref: str,
        *,
        graph_intent_hash: str | None = None,
    ) -> GraphIntentBrief:
        return self._get_unlocked(
            graph_intent_ref,
            graph_intent_hash=graph_intent_hash,
        )

    def _get_unlocked(
        self,
        graph_intent_ref: str,
        *,
        graph_intent_hash: str | None = None,
    ) -> GraphIntentBrief:
        if graph_intent_hash is not None:
            self._validate_hash_ref(graph_intent_hash)
        with closing(self._manifest_connection()) as connection:
            manifest = self._manifest_for_ref(
                connection,
                graph_intent_ref,
                graph_intent_hash=graph_intent_hash,
            )
            if manifest is None:
                raise ValueError(f"graph intent not found: {graph_intent_ref}")
            return self._load_brief_from_manifest(connection, manifest)

    def _load_brief_from_manifest(
        self, connection: sqlite3.Connection, manifest: sqlite3.Row
    ) -> GraphIntentBrief:
        brief_payload = json.loads(
            self.read_committed_artifact(
                manifest["graph_intent_ref"],
                "graph_intent_brief",
                connection=connection,
            )
        )
        brief = GraphIntentBrief.from_dict(brief_payload)
        self._validate_scope(brief)
        if brief.graph_intent_hash != manifest["graph_intent_hash"]:
            raise ValueError("graph_intent_hash mismatch")
        evidence = self._load_binding_evidence_from_text(
            self.read_committed_artifact(
                brief.graph_intent_id,
                "source_binding_evidence",
                connection=connection,
            ),
            brief,
        )
        identity = SourceIdentity.model_validate(evidence["source_identity"])
        canonical = self._validate_source_binding(brief, identity)
        if brief.frame_mapping:
            brief.frame_mapping.validate_source(brief.source_binding, canonical)
            mapping = brief.frame_mapping.as_dict()
            if mapping["proposal_ref"] == brief.graph_intent_id:
                raise ValueError("frame mapping requires original review proposal")
            proposal = self.get(mapping["proposal_ref"], graph_intent_hash=mapping["proposal_hash"])
            brief.frame_mapping.validate_review(proposal, canonical)
        return brief

    def _manifest_for_ref(
        self,
        connection: sqlite3.Connection,
        graph_intent_ref: str,
        *,
        graph_intent_hash: str | None = None,
    ) -> sqlite3.Row | None:
        if graph_intent_hash is None:
            return connection.execute(
                sql_query("""SELECT * FROM graph_intent_publication_manifests
                WHERE organization_id = ? AND workspace_id = ?
                  AND graph_intent_ref = ?""",'SELECT * FROM graph_intent_publication_manifests\n                WHERE organization_id = %s AND workspace_id = %s\n                  AND graph_intent_ref = %s'),
                (self.organization_id, self.workspace_id, graph_intent_ref),
            ).fetchone()
        return connection.execute(
            sql_query("""SELECT * FROM graph_intent_publication_manifests
            WHERE organization_id = ? AND workspace_id = ?
              AND graph_intent_ref = ? AND graph_intent_hash = ?""",'SELECT * FROM graph_intent_publication_manifests\n            WHERE organization_id = %s AND workspace_id = %s\n              AND graph_intent_ref = %s AND graph_intent_hash = %s'),
            (
                self.organization_id,
                self.workspace_id,
                graph_intent_ref,
                graph_intent_hash,
            ),
        ).fetchone()

    def _manifest_for_hash(
        self, connection: sqlite3.Connection, graph_intent_hash: str
    ) -> sqlite3.Row | None:
        return connection.execute(
            sql_query("""SELECT * FROM graph_intent_publication_manifests
            WHERE organization_id = ? AND workspace_id = ?
              AND graph_intent_hash = ?""",'SELECT * FROM graph_intent_publication_manifests\n            WHERE organization_id = %s AND workspace_id = %s\n              AND graph_intent_hash = %s'),
            (self.organization_id, self.workspace_id, graph_intent_hash),
        ).fetchone()

    def _manifest_connection(self) -> sqlite3.Connection:
        connection = GraphIntentPublicationTransaction._connect_manifest_db(
            self.artifact_root
        )
        return connection

    def read_committed_artifact(
        self,
        graph_intent_ref: str,
        artifact_role: str,
        *,
        expected_path: Path | None = None,
        expected_kind: str | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> str:
        GraphIntentPublicationTransaction._validate_artifact_role(artifact_role)
        close_connection = connection is None
        if connection is None:
            connection = self._manifest_connection()
        try:
            manifest = self._manifest_for_ref(connection, graph_intent_ref)
            if manifest is None:
                raise ValueError(f"graph intent not found: {graph_intent_ref}")
            manifest_payload = GraphIntentPublicationTransaction._validate_committed_manifest_integrity(
                connection,
                manifest,
                artifact_root=self.artifact_root,
            )
            if (
                expected_kind is not None
                and manifest_payload["publication_kind"] != expected_kind
            ):
                raise ValueError("graph intent publication manifest kind mismatch")
            row = connection.execute(
                sql_query("""SELECT * FROM graph_intent_publication_artifacts
                WHERE manifest_id = ? AND artifact_role = ?""",'SELECT * FROM graph_intent_publication_artifacts\n                WHERE manifest_id = %s AND artifact_role = %s'),
                (manifest["manifest_id"], artifact_role),
            ).fetchone()
            if row is None:
                raise ValueError("committed graph intent artifact missing")
            path = GraphIntentPublicationTransaction._verify_committed_artifact_row(
                self.artifact_root, row
            )
            if expected_path is not None and path != expected_path.resolve():
                raise ValueError("committed graph intent artifact path mismatch")
            return path.read_text(encoding="utf-8")
        finally:
            if close_connection and connection is not None:
                connection.close()

    def read_committed_artifact_by_hash(
        self,
        graph_intent_ref: str,
        artifact_role: str,
        artifact_hash: str,
        *,
        expected_path: Path | None = None,
        expected_kind: str | None = None,
    ) -> str:
        GraphIntentPublicationTransaction._validate_artifact_role(artifact_role)
        with closing(self._manifest_connection()) as connection:
            manifest = self._manifest_for_ref(connection, graph_intent_ref)
            if manifest is None:
                raise ValueError(f"graph intent not found: {graph_intent_ref}")
            manifest_payload = GraphIntentPublicationTransaction._validate_committed_manifest_integrity(
                connection,
                manifest,
                artifact_root=self.artifact_root,
            )
            if (
                expected_kind is not None
                and manifest_payload["publication_kind"] != expected_kind
            ):
                raise ValueError("graph intent publication manifest kind mismatch")
            row = connection.execute(
                sql_query("""SELECT * FROM graph_intent_publication_artifacts
                WHERE manifest_id = ? AND organization_id = ? AND workspace_id = ?
                  AND graph_intent_ref = ? AND artifact_role = ?
                  AND artifact_hash = ?""",'SELECT * FROM graph_intent_publication_artifacts\n                WHERE manifest_id = %s AND organization_id = %s AND workspace_id = %s\n                  AND graph_intent_ref = %s AND artifact_role = %s\n                  AND artifact_hash = %s'),
                (
                    manifest["manifest_id"],
                    self.organization_id,
                    self.workspace_id,
                    graph_intent_ref,
                    artifact_role,
                    artifact_hash,
                ),
            ).fetchone()
            if row is None:
                raise ValueError("committed graph intent artifact missing")
            path = GraphIntentPublicationTransaction._verify_committed_artifact_row(
                self.artifact_root, row
            )
            if expected_path is not None and path != expected_path.resolve():
                raise ValueError("committed graph intent artifact path mismatch")
            return path.read_text(encoding="utf-8")

    def get_by_hash(self, graph_intent_hash: str) -> GraphIntentBrief:
        self._validate_hash_ref(graph_intent_hash)
        with closing(self._manifest_connection()) as connection:
            manifest = self._manifest_for_hash(connection, graph_intent_hash)
            if manifest is None:
                raise ValueError(f"graph intent not found: {graph_intent_hash}")
            return self._load_brief_from_manifest(connection, manifest)

    def list(
        self,
        *,
        lineage_id: str | None = None,
        source_identity_ref: str | None = None,
    ) -> list[GraphIntentBrief]:
        return self._list_unlocked(
            lineage_id=lineage_id,
            source_identity_ref=source_identity_ref,
        )

    @contextmanager
    def iter_advisory_capture_briefs(
        self,
        *,
        lineage_id: str | None = None,
        source_identity_ref: str | None = None,
    ) -> AbstractContextManager[Iterator[GraphIntentBrief]]:
        """Stream scoped committed capture witnesses in stable manifest order."""

        query = (
            sql_query("SELECT * FROM graph_intent_publication_manifests "
            "WHERE organization_id = ? AND workspace_id = ?",'SELECT * FROM graph_intent_publication_manifests WHERE organization_id = %s AND workspace_id = %s')
        )
        params: list[str] = [self.organization_id, self.workspace_id]
        if lineage_id is not None:
            query += sql_query(" AND lineage_id = ?",' AND lineage_id = %s')
            params.append(lineage_id)
        if source_identity_ref is not None:
            query += sql_query(" AND source_identity_ref = ?",' AND source_identity_ref = %s')
            params.append(source_identity_ref)
        query += " ORDER BY created_at, graph_intent_ref"
        connection = self._manifest_connection()
        cursor = None
        try:
            cursor = connection.execute(query, params)

            def values() -> Iterator[GraphIntentBrief]:
                for manifest in cursor:
                    yield self._load_brief_from_manifest(connection, manifest)

            yield values()
        finally:
            try:
                if cursor is not None:
                    cursor.close()
            finally:
                connection.close()

    def _list_unlocked(
        self,
        *,
        lineage_id: str | None = None,
        source_identity_ref: str | None = None,
    ) -> list[GraphIntentBrief]:
        briefs: list[GraphIntentBrief] = []
        query = (
            sql_query("SELECT * FROM graph_intent_publication_manifests "
            "WHERE organization_id = ? AND workspace_id = ?",'SELECT * FROM graph_intent_publication_manifests WHERE organization_id = %s AND workspace_id = %s')
        )
        params: list[str] = [self.organization_id, self.workspace_id]
        if lineage_id is not None:
            query += sql_query(" AND lineage_id = ?",' AND lineage_id = %s')
            params.append(lineage_id)
        if source_identity_ref is not None:
            query += sql_query(" AND source_identity_ref = ?",' AND source_identity_ref = %s')
            params.append(source_identity_ref)
        query += " ORDER BY created_at, graph_intent_ref"
        with closing(self._manifest_connection()) as connection:
            for manifest in connection.execute(query, params):
                briefs.append(self._load_brief_from_manifest(connection, manifest))

        by_ref = {item.graph_intent_id: item for item in briefs}

        def depth(item: GraphIntentBrief) -> int:
            count = 0
            seen: set[str] = set()
            ref = item.predecessor_graph_intent_ref
            while ref in by_ref and ref not in seen:
                seen.add(ref)
                count += 1
                ref = by_ref[ref].predecessor_graph_intent_ref
            return count

        return sorted(briefs, key=lambda item: (depth(item), item.graph_intent_id))

    def import_legacy_filesystem_records(self) -> int:
        """Explicitly import complete legacy graph-intent filesystem bundles."""

        if not self.root.is_dir():
            return 0
        bundles: dict[str, dict[str, Any]] = {}
        for artifact_path in sorted(self.root.glob(f"*/{_ARTIFACT_NAME}")):
            try:
                payload = json.loads(artifact_path.read_text(encoding="utf-8"))
                brief = GraphIntentBrief.from_dict(payload)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                raise ValueError("invalid legacy graph intent bundle") from exc
            if artifact_path.parent.name != brief.graph_intent_hash.removeprefix(
                "sha256:"
            ):
                raise ValueError("invalid legacy graph intent bundle")
            binding_path = artifact_path.parent / _BINDING_NAME
            if not binding_path.is_file():
                raise ValueError("incomplete legacy graph intent bundle")
            binding = self._load_binding_evidence(artifact_path.parent, brief)
            identity = SourceIdentity.model_validate(binding["source_identity"])
            canonical = self._validate_source_binding(brief, identity)
            expected_binding = self._binding_evidence(
                brief, source_identity=identity, canonical=canonical
            )
            if binding != expected_binding:
                raise ValueError("invalid legacy graph intent bundle")
            publications = [
                (
                    artifact_path,
                    _encoded(brief.as_dict()),
                    "GRAPH_INTENT_ARTIFACT_CONFLICT",
                    "graph_intent_brief",
                ),
                (
                    binding_path,
                    _encoded(expected_binding),
                    "GRAPH_INTENT_SOURCE_BINDING_CONFLICT",
                    "source_binding_evidence",
                ),
            ]
            publications.extend(self._legacy_w2_publications(brief))
            bundles[brief.graph_intent_id] = {
                "brief": brief,
                "identity": identity,
                "publications": publications,
            }

        for item in bundles.values():
            brief = item["brief"]
            predecessor_ref = brief.predecessor_graph_intent_ref
            if predecessor_ref is None:
                continue
            predecessor_item = bundles.get(predecessor_ref)
            if predecessor_item is None:
                try:
                    predecessor = self.get(predecessor_ref)
                except ValueError as exc:
                    raise ValueError("incomplete legacy graph intent bundle") from exc
            else:
                predecessor = predecessor_item["brief"]
            link_path = self._successor_link_path(
                predecessor.graph_intent_id, brief.graph_intent_id
            )
            if not link_path.is_file():
                raise ValueError("incomplete legacy graph intent bundle")
            expected_link = self._successor_link_payload(predecessor, brief)
            try:
                actual_link = json.loads(link_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError("invalid legacy graph intent bundle") from exc
            if actual_link != expected_link:
                raise ValueError("invalid legacy graph intent bundle")
            item["publications"].append(
                (
                    link_path,
                    _encoded(expected_link),
                    "GRAPH_INTENT_SUCCESSOR_LINK_CONFLICT",
                    "successor_link",
                )
            )

        imported = 0
        for item in sorted(
            bundles.values(),
            key=lambda bundle: (
                bundle["brief"].predecessor_graph_intent_ref is not None,
                bundle["brief"].graph_intent_id,
            ),
        ):
            brief = item["brief"]
            with closing(self._manifest_connection()) as connection:
                already_committed = self._manifest_for_ref(
                    connection, brief.graph_intent_id
                )
            if already_committed is not None:
                continue
            with GraphIntentPublicationTransaction(self.artifact_root) as transaction:
                for path, encoded, error, role in item["publications"]:
                    transaction.publish(path, encoded, error, role=role)
                transaction.record_graph_intent(brief, source_identity=item["identity"])
            imported += 1
        return imported

    def _legacy_w2_publications(
        self, brief: GraphIntentBrief
    ) -> list[tuple[Path, str, str, str]]:
        proposal_refs = [
            ref
            for ref in brief.evidence_refs
            if ref.startswith("evidence:graph-intent-proposal:")
        ]
        if not proposal_refs:
            return []
        if len(proposal_refs) != 1:
            raise ValueError("invalid legacy graph intent bundle")
        proposal_digest = proposal_refs[0].rsplit(":", 1)[1]
        if re.fullmatch(r"[0-9a-f]{64}", proposal_digest) is None:
            raise ValueError("invalid legacy graph intent bundle")
        proposal_path = (
            self.artifact_root
            / "graph-intent-proposals"
            / proposal_digest
            / "graph_intent_proposal_evidence.v1.json"
        )
        if not proposal_path.is_file():
            raise ValueError("incomplete legacy graph intent bundle")
        try:
            proposal_encoded = proposal_path.read_text(encoding="utf-8")
            proposal = json.loads(proposal_encoded)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("invalid legacy graph intent bundle") from exc
        if content_hash(proposal) != proposal_digest:
            raise ValueError("invalid legacy graph intent bundle")
        publications: list[tuple[Path, str, str, str]] = [
            (
                proposal_path,
                proposal_encoded,
                "GRAPH_INTENT_PROPOSAL_EVIDENCE_CONFLICT",
                "proposal_evidence",
            )
        ]
        partial_pin = proposal.get("partial_assessment")
        if partial_pin is None:
            return publications
        if (
            not isinstance(partial_pin, dict)
            or not isinstance(partial_pin.get("artifact_hash"), str)
            or not isinstance(partial_pin.get("artifact_ref"), str)
            or not isinstance(partial_pin.get("evidence_ref"), str)
            or _SHA256_REF.fullmatch(partial_pin["artifact_hash"]) is None
        ):
            raise ValueError("invalid legacy graph intent bundle")
        partial_digest = partial_pin["artifact_hash"].removeprefix("sha256:")
        expected_ref = (
            "artifacts/graph-intent-partial-assessments/"
            f"{partial_digest}/graph_intent_partial_assessment.v1.json"
        )
        expected_evidence_ref = (
            "evidence:graph-intent-partial-assessment:" + partial_digest
        )
        if (
            partial_pin["artifact_ref"] != expected_ref
            or partial_pin["evidence_ref"] != expected_evidence_ref
        ):
            raise ValueError("invalid legacy graph intent bundle")
        assessment_path = (self.workspace_path / partial_pin["artifact_ref"]).resolve()
        if not assessment_path.is_relative_to(self.artifact_root):
            raise ValueError("invalid legacy graph intent bundle")
        if not assessment_path.is_file():
            raise ValueError("incomplete legacy graph intent bundle")
        try:
            assessment_encoded = assessment_path.read_text(encoding="utf-8")
            assessment = json.loads(assessment_encoded)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("invalid legacy graph intent bundle") from exc
        if f"sha256:{content_hash(assessment)}" != partial_pin["artifact_hash"]:
            raise ValueError("invalid legacy graph intent bundle")
        publications.append(
            (
                assessment_path,
                assessment_encoded,
                "GRAPH_INTENT_PARTIAL_ASSESSMENT_CONFLICT",
                "partial_assessment",
            )
        )
        return publications

    def _carried_w2_evidence_publications(
        self,
        brief: GraphIntentBrief,
        predecessor: GraphIntentBrief | None,
    ) -> list[tuple[Path, str, str, str]]:
        if (
            GraphIntentPublicationTransaction._publication_kind_for_brief(brief)
            != _PUBLICATION_KIND_W2
        ):
            return []
        if predecessor is None:
            return []
        if self._proposal_evidence_refs(brief) != self._proposal_evidence_refs(
            predecessor
        ):
            raise ValueError("graph intent review evidence refs mismatch")
        publications: list[tuple[Path, str, str, str]] = []
        with closing(self._manifest_connection()) as connection:
            manifest = self._manifest_for_ref(
                connection,
                predecessor.graph_intent_id,
                graph_intent_hash=predecessor.graph_intent_hash,
            )
            if manifest is None:
                raise ValueError("graph intent predecessor publication missing")
            manifest_payload = GraphIntentPublicationTransaction._validate_committed_manifest_integrity(
                connection,
                manifest,
                artifact_root=self.artifact_root,
            )
            if manifest_payload["publication_kind"] != _PUBLICATION_KIND_W2:
                raise ValueError("graph intent predecessor publication kind mismatch")
            artifacts = {
                item["artifact_role"]: item for item in manifest_payload["artifacts"]
            }
            for role, error in (
                ("proposal_evidence", "GRAPH_INTENT_PROPOSAL_EVIDENCE_CONFLICT"),
                ("partial_assessment", "GRAPH_INTENT_PARTIAL_ASSESSMENT_CONFLICT"),
            ):
                artifact = artifacts.get(role)
                if artifact is None:
                    continue
                path = (self.artifact_root / artifact["artifact_path"]).resolve()
                encoded = self.read_committed_artifact(
                    predecessor.graph_intent_id,
                    role,
                    expected_path=path,
                    expected_kind=_PUBLICATION_KIND_W2,
                    connection=connection,
                )
                publications.append((path, encoded, error, role))
        return publications

    @staticmethod
    def _proposal_evidence_refs(brief: GraphIntentBrief) -> list[str]:
        return sorted(
            ref
            for ref in brief.evidence_refs
            if ref.startswith("evidence:graph-intent-proposal:")
        )

    def review(
        self,
        graph_intent_ref: str,
        *,
        review: GraphIntentReview,
        source_identity: SourceIdentity,
        changes: dict[str, Any] | None = None,
    ) -> GraphIntentBrief:
        predecessor = self.get(graph_intent_ref)
        if predecessor.review.state == "rejected":
            raise ValueError("rejected graph intent is terminal")
        if source_identity.source_identity_ref != (
            predecessor.source_binding.source_identity_ref
        ):
            raise ValueError("review cannot change source identity")
        updates = dict(changes or {})
        forbidden = {
            "schema_version",
            "graph_intent_id",
            "graph_intent_hash",
            "source_binding",
            "review",
            "predecessor_graph_intent_ref",
            "successor_graph_intent_refs",
        }
        if forbidden.intersection(updates):
            raise ValueError("review cannot mutate source binding or identity fields")
        material_changes = _MATERIAL_FIELDS.intersection(updates)
        if material_changes and review.state != "edited":
            raise ValueError(
                "material graph intent changes require edited review state"
            )
        if review.state == predecessor.review.state and not updates:
            raise ValueError("review transition is a no-op")
        if review.state == "proposed":
            raise ValueError("review transition cannot return to proposed")
        if review.state == "edited" and not material_changes:
            raise ValueError("edited review requires a material graph intent change")
        if updates and not material_changes:
            raise ValueError("review changes must be material graph intent fields")
        payload = deepcopy(predecessor.as_dict())
        payload.update(updates)
        payload["review"] = review.as_dict()
        payload["predecessor_graph_intent_ref"] = predecessor.graph_intent_id
        payload["successor_graph_intent_refs"] = []
        successor = GraphIntentBrief.create(payload)
        if review.state == "edited" and not any(
            successor.as_dict().get(field) != predecessor.as_dict().get(field)
            for field in material_changes
        ):
            raise ValueError("edited review requires a material graph intent change")
        return self._persist(successor, source_identity=source_identity)

    def successors(self, predecessor_graph_intent_ref: str) -> list[str]:
        return self._successors_unlocked(predecessor_graph_intent_ref)

    def _successors_unlocked(self, predecessor_graph_intent_ref: str) -> list[str]:
        if _GRAPH_INTENT_REF.fullmatch(predecessor_graph_intent_ref) is None:
            raise ValueError(
                "predecessor_graph_intent_ref must be an exact graph intent ref"
            )
        result: list[str] = []
        with closing(self._manifest_connection()) as connection:
            manifests = connection.execute(
                sql_query("""SELECT * FROM graph_intent_publication_manifests
                WHERE organization_id = ? AND workspace_id = ?
                  AND predecessor_graph_intent_ref = ?
                ORDER BY created_at, graph_intent_ref""",'SELECT * FROM graph_intent_publication_manifests\n                WHERE organization_id = %s AND workspace_id = %s\n                  AND predecessor_graph_intent_ref = %s\n                ORDER BY created_at, graph_intent_ref'),
                (
                    self.organization_id,
                    self.workspace_id,
                    predecessor_graph_intent_ref,
                ),
            ).fetchall()
            for manifest in manifests:
                link_payload = json.loads(
                    self.read_committed_artifact(
                        manifest["graph_intent_ref"],
                        "successor_link",
                        connection=connection,
                    )
                )
                if (
                    link_payload.get("predecessor_graph_intent_ref")
                    != predecessor_graph_intent_ref
                ):
                    raise ValueError("invalid graph intent successor link")
                successor = self._load_brief_from_manifest(connection, manifest)
                if (
                    link_payload.get("successor_graph_intent_ref")
                    != successor.graph_intent_id
                    or link_payload.get("successor_graph_intent_hash")
                    != successor.graph_intent_hash
                ):
                    raise ValueError("invalid graph intent successor link")
                result.append(successor.graph_intent_id)
        return sorted(result)

    def is_superseded_for_new_compilation(self, graph_intent_ref: str) -> bool:
        brief = self.get(graph_intent_ref)
        bound_ref = brief.source_binding.source_identity_ref
        lineage_id = brief.source_binding.lineage_id
        identities = self.source_registry.list_source_identities(
            organization_id=self.organization_id,
            workspace_id=self.workspace_id,
            lineage_id=lineage_id,
        )
        return any(
            identity.supersedes_source_identity_ref == bound_ref
            for identity in identities
        )

    def require_compiler_eligible(self, graph_intent_ref: str) -> GraphIntentBrief:
        brief = self.get(graph_intent_ref)
        if not brief.compiler_eligible:
            raise ValueError("graph intent review state is not compiler-eligible")
        if self.is_superseded_for_new_compilation(graph_intent_ref):
            raise ValueError("graph intent is superseded for new compilation")
        return brief

    def _validate_scope(self, brief: GraphIntentBrief) -> None:
        binding = brief.source_binding
        if binding.organization_id != self.organization_id:
            raise ValueError("graph intent organization mismatch")
        if binding.workspace_id != self.workspace_id:
            raise ValueError("graph intent workspace mismatch")

    def _validate_source_binding(
        self,
        brief: GraphIntentBrief,
        source_identity: SourceIdentity,
    ) -> CanonicalSourceDocument:
        binding = brief.source_binding
        if source_identity.organization_id != self.organization_id:
            raise ValueError("source identity organization mismatch")
        if source_identity.workspace_id != self.workspace_id:
            raise ValueError("source identity workspace mismatch")
        if source_identity.created_at.utcoffset() != timedelta(0):
            raise ValueError("source identity created_at must be UTC")
        comparisons = {
            "source_identity_ref": source_identity.source_identity_ref,
            "source_hash": source_identity.source_hash,
            "source_version": source_identity.source_version,
        }
        for field, actual in comparisons.items():
            if getattr(binding, field) != actual:
                raise ValueError(f"source binding {field} mismatch")

        registered = self.source_registry.get_source_identity(
            source_identity_ref=source_identity.source_identity_ref,
            organization_id=self.organization_id,
            workspace_id=self.workspace_id,
            lineage_id=binding.lineage_id,
        )
        if registered is None:
            raise ValueError("governed SourceIdentity not found")
        supplied_payload = source_identity.model_dump(
            mode="json", by_alias=True, exclude_none=False
        )
        registered_payload = registered.model_dump(
            mode="json", by_alias=True, exclude_none=False
        )
        if (
            binding.source_version is None
            and source_identity.source_version is None
            and registered.source_version is not None
            and KnowledgeBuilderStore.permits_historical_null_source_version(
                self.source_registry,
                supplied_identity=source_identity,
                registered_identity=registered,
                lineage_id=binding.lineage_id,
            )
        ):
            # Historical governed W1 lineages predate derived source-record
            # versions. Their exact source identity/hash remains authoritative;
            # normalize only that optional legacy field for replay comparison.
            registered_payload["sourceVersion"] = None
        if supplied_payload != registered_payload:
            raise ValueError("source identity conflict with governed registry")

        path = self._resolve_canonical_source_ir(binding.canonical_source_ir_ref)
        payload = json.loads(path.read_text(encoding="utf-8"))
        canonical = CanonicalSourceDocument.model_validate(payload)
        if canonical.source_identity_ref != source_identity.source_identity_ref:
            raise ValueError("canonical source IR source_identity_ref mismatch")
        locators = [
            locator
            for mapping in canonical.source_map.mappings
            for locator in mapping.source_locators
        ]
        if not locators or any(
            locator.source_hash != source_identity.source_hash
            or locator.source_identity_ref != source_identity.source_identity_ref
            for locator in locators
        ):
            raise ValueError(
                "CanonicalSourceIR locator source_hash or source identity mismatch"
            )
        actual_hash = canonical_ir_content_hash(
            canonical.model_dump(mode="json", by_alias=True, exclude_none=False)
        )
        if actual_hash != binding.canonical_source_ir_hash:
            raise ValueError("canonical_source_ir_hash mismatch")
        path_digest = path.parent.name
        if f"sha256:{path_digest}" != binding.canonical_source_ir_hash:
            raise ValueError("canonical_source_ir_ref content hash mismatch")
        return canonical

    def resolve_evidence_source(
        self, *, source_identity_ref: str, canonical_source_ir_ref: str,
        canonical_source_ir_hash: str,
    ) -> tuple[SourceIdentity, CanonicalSourceDocument]:
        """Read an exact published source binding; caller pins never grant authority."""
        lineage_id, identity = self.source_registry.resolve_source_identity_lineage(
            source_identity_ref=source_identity_ref,
            organization_id=self.organization_id, workspace_id=self.workspace_id,
        )
        bindings = [
            brief.source_binding
            for brief in self.list(lineage_id=lineage_id, source_identity_ref=source_identity_ref)
            if brief.source_binding.canonical_source_ir_ref == canonical_source_ir_ref
            and brief.source_binding.canonical_source_ir_hash == canonical_source_ir_hash
        ]
        if not bindings:
            raise ValueError("Evidence source has no matching governed canonical source binding")
        binding = next((item for item in bindings if item.source_version == identity.source_version), bindings[0])
        if binding.source_version is None and identity.source_version is not None:
            historical = identity.model_copy(update={"source_version": None})
            if KnowledgeBuilderStore.permits_historical_null_source_version(
                self.source_registry, supplied_identity=historical,
                registered_identity=identity, lineage_id=lineage_id,
            ):
                identity = historical
        return identity, self.verify_source_binding(binding, source_identity=identity)

    def verify_source_binding(
        self,
        binding: GraphIntentSourceBinding,
        *,
        source_identity: SourceIdentity,
    ) -> CanonicalSourceDocument:
        """Verify proposal pins through the same authority boundary used on create."""

        class _BoundIntent:
            source_binding = binding

        bound = _BoundIntent()
        self._validate_scope(bound)  # type: ignore[arg-type]
        return self._validate_source_binding(bound, source_identity)  # type: ignore[arg-type]

    def capture_advisory_publication(
        self,
        *,
        graph_intent_ref: str,
        graph_intent_hash: str,
    ) -> CapturedJsonV1:
        """Export one exact supported W1 publication after live owner checks."""

        if _GRAPH_INTENT_REF.fullmatch(graph_intent_ref) is None:
            raise ValueError("graph_intent_ref must be an exact graph intent ref")
        self._validate_hash_ref(graph_intent_hash)
        with closing(self._manifest_connection()) as connection:
            manifest = self._manifest_for_ref(
                connection,
                graph_intent_ref,
                graph_intent_hash=graph_intent_hash,
            )
            if manifest is None:
                raise ValueError(f"graph intent not found: {graph_intent_ref}")
            manifest_payload = GraphIntentPublicationTransaction._validate_committed_manifest_integrity(
                connection,
                manifest,
                artifact_root=self.artifact_root,
            )
            manifest_capture = _captured_json(manifest_payload)
            if manifest_capture.canonical_json != manifest["manifest_json"]:
                raise ValueError("graph intent publication manifest bytes mismatch")
            brief = self._load_brief_from_manifest(connection, manifest)
            self._validate_advisory_publication_lineage(
                connection,
                manifest_payload=manifest_payload,
                brief=brief,
            )

            artifacts_by_role = {
                row["artifact_role"]: row
                for row in connection.execute(
                    sql_query("""SELECT * FROM graph_intent_publication_artifacts
                    WHERE manifest_id = ?
                    ORDER BY artifact_role, artifact_path""",'SELECT * FROM graph_intent_publication_artifacts\n                    WHERE manifest_id = %s\n                    ORDER BY artifact_role, artifact_path'),
                    (manifest["manifest_id"],),
                ).fetchall()
            }
            captures: list[dict[str, Any]] = []
            for item in manifest_payload["artifacts"]:
                role = item["artifact_role"]
                row = artifacts_by_role.get(role)
                if row is None or row["artifact_hash"] != item["artifact_hash"]:
                    raise ValueError(
                        "graph intent publication manifest artifact rows mismatch"
                    )
                path = GraphIntentPublicationTransaction._verify_committed_artifact_row(
                    self.artifact_root,
                    row,
                )
                raw = path.read_bytes()
                try:
                    decoded = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        "committed graph intent artifact is invalid JSON"
                    ) from exc
                captures.append(
                    {
                        "role": role,
                        "payload": _captured_json(decoded).model_dump(mode="json"),
                        "bytes_base64": base64.b64encode(raw).decode("ascii"),
                    }
                )

        return _captured_json(
            {
                "schema_version": ("advisory_graph_intent_publication_capture.v1"),
                "manifest": manifest_capture.model_dump(mode="json"),
                "artifacts": captures,
            }
        )

    def _validate_advisory_publication_lineage(
        self,
        connection: sqlite3.Connection,
        *,
        manifest_payload: dict[str, Any],
        brief: GraphIntentBrief,
    ) -> None:
        if manifest_payload["publication_kind"] != _PUBLICATION_KIND_W1:
            raise ValueError("advisory publication requires a W1 graph intent")
        predecessor_ref = brief.predecessor_graph_intent_ref
        if predecessor_ref is None:
            if brief.review.state != "proposed":
                raise ValueError(
                    "advisory root publication must have proposed review state"
                )
            return
        if brief.review.state != "accepted":
            raise ValueError(
                "advisory successor publication must have accepted review state"
            )
        predecessor_manifest = self._manifest_for_ref(connection, predecessor_ref)
        if predecessor_manifest is None:
            raise ValueError("graph intent predecessor publication missing")
        predecessor_payload = (
            GraphIntentPublicationTransaction._validate_committed_manifest_integrity(
                connection,
                predecessor_manifest,
                artifact_root=self.artifact_root,
            )
        )
        predecessor = self._load_brief_from_manifest(
            connection,
            predecessor_manifest,
        )
        if (
            predecessor_payload["publication_kind"] != _PUBLICATION_KIND_W1
            or predecessor.predecessor_graph_intent_ref is not None
            or predecessor.review.state != "proposed"
            or predecessor.source_binding != brief.source_binding
        ):
            raise ValueError("advisory accepted review predecessor mismatch")
        predecessor_semantics = predecessor.as_dict()
        successor_semantics = brief.as_dict()
        for payload in (predecessor_semantics, successor_semantics):
            for field in (
                "graph_intent_id",
                "graph_intent_hash",
                "review",
                "predecessor_graph_intent_ref",
                "successor_graph_intent_refs",
            ):
                payload.pop(field)
        if predecessor_semantics != successor_semantics:
            raise ValueError("advisory accepted review changes semantic content")
        link = json.loads(
            self.read_committed_artifact(
                brief.graph_intent_id,
                "successor_link",
                connection=connection,
            )
        )
        if link != self._successor_link_payload(predecessor, brief):
            raise ValueError("advisory accepted review successor link mismatch")
        deeper = connection.execute(
            sql_query("""SELECT 1 FROM graph_intent_publication_manifests
            WHERE organization_id = ? AND workspace_id = ?
              AND predecessor_graph_intent_ref = ?
            LIMIT 1""",'SELECT 1 FROM graph_intent_publication_manifests\n            WHERE organization_id = %s AND workspace_id = %s\n              AND predecessor_graph_intent_ref = %s\n            LIMIT 1'),
            (self.organization_id, self.workspace_id, brief.graph_intent_id),
        ).fetchone()
        if deeper is not None:
            raise ValueError(
                "advisory publication does not support deeper review lineage"
            )

    def capture_advisory_source(
        self,
        *,
        binding: GraphIntentSourceBinding,
        source_identity: SourceIdentity,
    ) -> CapturedJsonV1:
        """Export exact governed source members without granting new authority."""

        canonical = self.verify_source_binding(
            binding,
            source_identity=source_identity,
        )
        source_path = self.source_registry._verify_source_artifact(source_identity)
        source_bytes = source_path.read_bytes()
        if f"sha256:{sha256(source_bytes).hexdigest()}" != binding.source_hash:
            raise ValueError("governed source artifact hash mismatch")

        blocks_by_id = {block.block_id: block for block in canonical.blocks}
        if not blocks_by_id or len(blocks_by_id) != len(canonical.blocks):
            raise ValueError("canonical source IR block membership mismatch")
        mapped_ids = [
            block_id
            for mapping in canonical.source_map.mappings
            for block_id in mapping.block_ids
        ]
        if len(mapped_ids) != len(set(mapped_ids)) or set(mapped_ids) != set(
            blocks_by_id
        ):
            raise ValueError("canonical source IR mapping membership mismatch")

        evidence: list[dict[str, Any]] = []
        for block in canonical.blocks:
            if block.content_hash != canonical_ir_content_hash(block.content):
                raise ValueError("canonical source IR block content hash mismatch")
            try:
                locator = resolve_unique_mapping(canonical, block.block_id)
            except AmbiguousSourceMapping as exc:
                raise ValueError("canonical source IR block locator mismatch") from exc
            if (
                locator.source_identity_ref != source_identity.source_identity_ref
                or locator.source_hash != source_identity.source_hash
                or locator.media_type != source_identity.media_type
            ):
                raise ValueError("canonical source IR block locator mismatch")
            if block.block_id != source_block_identity(
                source_identity.source_hash,
                locator,
                block.block_type,
            ):
                raise ValueError("canonical source IR block identity mismatch")
            evidence.append(
                evidence_ref_for_block(
                    canonical,
                    block,
                    source_identity=source_identity,
                    created_at=source_identity.created_at,
                ).model_dump(mode="json", by_alias=True, exclude_none=False)
            )

        return _captured_json(
            {
                "schema_version": "advisory_source_capture.v1",
                "identity": source_identity.model_dump(
                    mode="json", by_alias=True, exclude_none=False
                ),
                "canonical_source_ir": canonical.model_dump(
                    mode="json", by_alias=True, exclude_none=False
                ),
                "source_bytes_base64": base64.b64encode(source_bytes).decode("ascii"),
                "evidence": evidence,
            }
        )

    def capture_advisory_evidence(
        self,
        *,
        evidence_ref: str,
        source: CapturedJsonV1,
    ) -> CapturedJsonV1:
        """Return one exact source member after live owner revalidation."""

        payload, identity, canonical, source_bytes = (
            self._validate_advisory_source_capture(source)
        )
        self._require_live_advisory_source(
            identity=identity,
            canonical=canonical,
            source_bytes=source_bytes,
        )
        matches = [
            item for item in payload["evidence"] if item["evidenceRef"] == evidence_ref
        ]
        if len(matches) != 1:
            raise ValueError("advisory evidence member must resolve exactly once")
        return _captured_json(matches[0])

    def _validate_advisory_source_capture(
        self,
        source: CapturedJsonV1,
    ) -> tuple[dict[str, Any], SourceIdentity, CanonicalSourceDocument, bytes]:
        try:
            payload = json.loads(source.canonical_json)
        except (
            json.JSONDecodeError
        ) as exc:  # CapturedJsonV1 normally rejects this first.
            raise ValueError("invalid advisory source capture JSON") from exc
        expected_fields = {
            "schema_version",
            "identity",
            "canonical_source_ir",
            "source_bytes_base64",
            "evidence",
        }
        if not isinstance(payload, dict) or set(payload) != expected_fields:
            raise ValueError("invalid advisory source capture shape")
        if payload["schema_version"] != "advisory_source_capture.v1":
            raise ValueError("invalid advisory source capture schema")

        identity = SourceIdentity.model_validate(payload["identity"])
        identity_payload = identity.model_dump(
            mode="json", by_alias=True, exclude_none=False
        )
        if payload["identity"] != identity_payload:
            raise ValueError("advisory source identity encoding mismatch")
        if (
            identity.organization_id != self.organization_id
            or identity.workspace_id != self.workspace_id
            or identity.created_at.utcoffset() != timedelta(0)
        ):
            raise ValueError("advisory source scope mismatch")

        canonical = CanonicalSourceDocument.model_validate(
            payload["canonical_source_ir"]
        )
        canonical_payload = canonical.model_dump(
            mode="json", by_alias=True, exclude_none=False
        )
        if payload["canonical_source_ir"] != canonical_payload:
            raise ValueError("advisory canonical source IR encoding mismatch")

        encoded_source = payload["source_bytes_base64"]
        if not isinstance(encoded_source, str):
            raise ValueError("advisory source bytes must be strict base64")
        try:
            source_bytes = base64.b64decode(encoded_source, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("advisory source bytes must be strict base64") from exc
        if base64.b64encode(source_bytes).decode("ascii") != encoded_source:
            raise ValueError("advisory source bytes must be canonical base64")
        if f"sha256:{sha256(source_bytes).hexdigest()}" != identity.source_hash:
            raise ValueError("advisory source bytes hash mismatch")

        raw_evidence = payload["evidence"]
        if not isinstance(raw_evidence, list):
            raise ValueError("advisory source evidence membership mismatch")
        expected_evidence: list[dict[str, Any]] = []
        block_ids: set[str] = set()
        mapped_ids = [
            block_id
            for mapping in canonical.source_map.mappings
            for block_id in mapping.block_ids
        ]
        if len(mapped_ids) != len(set(mapped_ids)):
            raise ValueError("canonical source IR mapping membership mismatch")
        for block in canonical.blocks:
            if block.block_id in block_ids:
                raise ValueError("canonical source IR block membership mismatch")
            block_ids.add(block.block_id)
            if block.content_hash != canonical_ir_content_hash(block.content):
                raise ValueError("canonical source IR block content hash mismatch")
            try:
                locator = resolve_unique_mapping(canonical, block.block_id)
            except AmbiguousSourceMapping as exc:
                raise ValueError("canonical source IR block locator mismatch") from exc
            if (
                locator.source_identity_ref != identity.source_identity_ref
                or locator.source_hash != identity.source_hash
                or locator.media_type != identity.media_type
                or block.block_id
                != source_block_identity(
                    identity.source_hash, locator, block.block_type
                )
            ):
                raise ValueError("canonical source IR block identity mismatch")
            expected_evidence.append(
                evidence_ref_for_block(
                    canonical,
                    block,
                    source_identity=identity,
                    created_at=identity.created_at,
                ).model_dump(mode="json", by_alias=True, exclude_none=False)
            )
        if not block_ids or set(mapped_ids) != block_ids:
            raise ValueError("canonical source IR mapping membership mismatch")

        evidence = [
            EvidenceRef.model_validate(item).model_dump(
                mode="json", by_alias=True, exclude_none=False
            )
            for item in raw_evidence
        ]
        if evidence != expected_evidence:
            raise ValueError("advisory source evidence membership mismatch")
        return payload, identity, canonical, source_bytes

    def _require_live_advisory_source(
        self,
        *,
        identity: SourceIdentity,
        canonical: CanonicalSourceDocument,
        source_bytes: bytes,
    ) -> None:
        canonical_payload = canonical.model_dump(
            mode="json", by_alias=True, exclude_none=False
        )
        canonical_hash = canonical_ir_content_hash(canonical_payload)
        matching_bindings: set[tuple[str, ...]] = set()
        with closing(self._manifest_connection()) as connection:
            manifests = connection.execute(
                sql_query("""SELECT * FROM graph_intent_publication_manifests
                WHERE organization_id = ? AND workspace_id = ?
                  AND source_identity_ref = ?
                ORDER BY created_at, graph_intent_ref""",'SELECT * FROM graph_intent_publication_manifests\n                WHERE organization_id = %s AND workspace_id = %s\n                  AND source_identity_ref = %s\n                ORDER BY created_at, graph_intent_ref'),
                (
                    self.organization_id,
                    self.workspace_id,
                    identity.source_identity_ref,
                ),
            ).fetchall()
            for manifest in manifests:
                brief = self._load_brief_from_manifest(connection, manifest)
                binding = brief.source_binding
                if (
                    binding.source_hash != identity.source_hash
                    or binding.source_version != identity.source_version
                    or binding.canonical_source_ir_hash != canonical_hash
                ):
                    continue
                live_canonical = self.verify_source_binding(
                    binding,
                    source_identity=identity,
                )
                if (
                    live_canonical.model_dump(
                        mode="json", by_alias=True, exclude_none=False
                    )
                    != canonical_payload
                ):
                    continue
                matching_bindings.add(
                    (
                        binding.organization_id,
                        binding.workspace_id,
                        binding.lineage_id,
                        binding.source_identity_ref,
                        binding.source_hash,
                        binding.source_version or "",
                        binding.canonical_source_ir_ref,
                        binding.canonical_source_ir_hash,
                    )
                )
        if len(matching_bindings) != 1:
            raise ValueError("advisory source has no unique live owner binding")
        source_path = self.source_registry._verify_source_artifact(identity)
        if source_path.read_bytes() != source_bytes:
            raise ValueError("advisory source bytes differ from live owner")

    @staticmethod
    def _validate_hash_ref(value: str) -> None:
        if _SHA256_REF.fullmatch(value) is None:
            raise ValueError("graph_intent_hash must be an exact sha256 ref")

    def _resolve_canonical_source_ir(self, artifact_ref: str) -> Path:
        candidate = Path(artifact_ref)
        if candidate.is_absolute():
            resolved = candidate.resolve()
        else:
            relative = candidate.as_posix()
            if relative.startswith("artifacts/"):
                relative = relative.removeprefix("artifacts/")
            resolved = (self.artifact_root / relative).resolve()
        if (
            resolved != self.artifact_root
            and self.artifact_root not in resolved.parents
        ):
            raise ValueError(
                "canonical_source_ir_ref must resolve inside the governed workspace artifact root"
            )
        if not resolved.is_file():
            raise ValueError(f"canonical source IR artifact not found: {artifact_ref}")
        return resolved

    def _validate_predecessor(self, brief: GraphIntentBrief) -> GraphIntentBrief | None:
        ref = brief.predecessor_graph_intent_ref
        if ref is None:
            return None
        predecessor = self.get(ref)
        if predecessor.source_binding.lineage_id != brief.source_binding.lineage_id:
            raise ValueError("successor graph intent lineage mismatch")
        return predecessor

    def _binding_evidence(
        self,
        brief: GraphIntentBrief,
        *,
        source_identity: SourceIdentity,
        canonical: CanonicalSourceDocument,
    ) -> dict[str, Any]:
        identity_payload = source_identity.model_dump(
            mode="json", by_alias=True, exclude_none=False
        )
        canonical_payload = canonical.model_dump(
            mode="json", by_alias=True, exclude_none=False
        )
        return {
            "schema_version": "graph_intent_source_binding_evidence.v1",
            "organization_id": self.organization_id,
            "workspace_id": self.workspace_id,
            "lineage_id": brief.source_binding.lineage_id,
            "graph_intent_ref": brief.graph_intent_id,
            "graph_intent_hash": brief.graph_intent_hash,
            "source_identity": identity_payload,
            "source_identity_hash": f"sha256:{content_hash(identity_payload)}",
            "canonical_source_ir_ref": brief.source_binding.canonical_source_ir_ref,
            "canonical_source_ir_hash": f"sha256:{content_hash(canonical_payload)}",
        }

    def _load_binding_evidence(
        self, artifact_dir: Path, brief: GraphIntentBrief
    ) -> dict[str, Any]:
        path = artifact_dir / _BINDING_NAME
        if not path.is_file():
            raise ValueError("graph intent source binding evidence not found")
        return self._load_binding_evidence_from_text(
            path.read_text(encoding="utf-8"), brief
        )

    def _load_binding_evidence_from_text(
        self, encoded: str, brief: GraphIntentBrief
    ) -> dict[str, Any]:
        payload = json.loads(encoded)
        if payload.get("schema_version") != "graph_intent_source_binding_evidence.v1":
            raise ValueError("invalid graph intent source binding evidence")
        if payload.get("graph_intent_ref") != brief.graph_intent_id:
            raise ValueError("graph intent source binding ref mismatch")
        if payload.get("graph_intent_hash") != brief.graph_intent_hash:
            raise ValueError("graph intent source binding hash mismatch")
        identity_payload = payload.get("source_identity")
        if not isinstance(identity_payload, dict):
            raise ValueError("invalid source identity evidence")
        expected_identity_hash = f"sha256:{content_hash(identity_payload)}"
        if payload.get("source_identity_hash") != expected_identity_hash:
            raise ValueError("source identity evidence hash mismatch")
        if payload.get("organization_id") != self.organization_id:
            raise ValueError("graph intent organization mismatch")
        if payload.get("workspace_id") != self.workspace_id:
            raise ValueError("graph intent workspace mismatch")
        if payload.get("lineage_id") != brief.source_binding.lineage_id:
            raise ValueError("graph intent lineage evidence mismatch")
        if (
            payload.get("canonical_source_ir_ref")
            != brief.source_binding.canonical_source_ir_ref
        ):
            raise ValueError("canonical_source_ir_ref evidence mismatch")
        if (
            payload.get("canonical_source_ir_hash")
            != brief.source_binding.canonical_source_ir_hash
        ):
            raise ValueError("canonical_source_ir_hash evidence mismatch")
        return payload

    def _reject_source_identity_conflict(
        self, brief: GraphIntentBrief, source_identity: SourceIdentity
    ) -> None:
        if not self.root.is_dir():
            return
        supplied = source_identity.model_dump(
            mode="json", by_alias=True, exclude_none=False
        )
        for path in self.root.glob(f"*/{_BINDING_NAME}"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            existing = payload.get("source_identity")
            if not isinstance(existing, dict):
                raise ValueError("invalid source identity evidence")
            if existing.get("sourceIdentityRef") != source_identity.source_identity_ref:
                continue
            if existing != supplied:
                raise ValueError("source identity conflict")
            if (
                payload.get("lineage_id") == brief.source_binding.lineage_id
                and payload.get("canonical_source_ir_ref")
                != brief.source_binding.canonical_source_ir_ref
            ):
                raise ValueError("source identity canonical IR conflict")

    def _successor_link_path(self, predecessor_ref: str, successor_ref: str) -> Path:
        return self.root / "successor-links" / predecessor_ref / f"{successor_ref}.json"

    @staticmethod
    def _successor_link_payload(
        predecessor: GraphIntentBrief, successor: GraphIntentBrief
    ) -> dict[str, Any]:
        return {
            "schema_version": "graph_intent_successor_link.v1",
            "predecessor_graph_intent_ref": predecessor.graph_intent_id,
            "predecessor_graph_intent_hash": predecessor.graph_intent_hash,
            "successor_graph_intent_ref": successor.graph_intent_id,
            "successor_graph_intent_hash": successor.graph_intent_hash,
        }

    @staticmethod
    def _require_no_conflict(path: Path, encoded: str, error: str) -> None:
        if path.exists() and path.read_text(encoding="utf-8") != encoded:
            raise ValueError(error)

    @classmethod
    def _write_once(
        cls,
        path: Path,
        encoded: str,
        error: str = "GRAPH_INTENT_ARTIFACT_CONFLICT",
    ) -> bool:
        transaction = GraphIntentPublicationTransaction.current_for_path(path)
        if transaction is not None:
            return transaction.publish(path, encoded, error)
        return GraphIntentPublicationTransaction.publish_unlocked(path, encoded, error)


__all__ = ["GraphIntentBriefStore", "GraphIntentPublicationTransaction"]
