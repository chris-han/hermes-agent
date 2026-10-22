from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, BinaryIO, Protocol
from pathlib import Path
from uuid import uuid4

from eos.artifact_hashing import canonical_json, content_hash
from eos.db import init_db,get_conn
from storage.hosted_postgres import enabled,sql_query
from contracts.tender_sensitive_term_detection_modeling_core import GovernedArtifactDTO, RuntimeContextDTO


class GovernedArtifactService(Protocol):
    def resolve_artifact(
        self,
        ref: str,
        organization_id: str,
        workspace_id: str | None,
    ) -> GovernedArtifactDTO: ...

    def create_derivative_artifact(
        self,
        *,
        parent_ref: str,
        media_type: str,
        bytes_or_stream: bytes | BinaryIO,
        metadata: dict[str, Any],
        context: RuntimeContextDTO,
    ) -> GovernedArtifactDTO: ...

    def verify_artifact_hash(
        self,
        ref: str,
        expected_hash: str,
        context: RuntimeContextDTO,
    ) -> bool: ...

    def open_artifact_for_read(self, ref: str, context: RuntimeContextDTO) -> BinaryIO: ...


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + sha256(data).hexdigest()


def _publish_immutable_bytes(path: Path, data: bytes) -> None:
    """Link a flushed complete file into its final name without overwriting."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix="." + path.name + ".", delete=False) as output:
            temporary = Path(output.name)
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError("DERIVATIVE_CONFLICT") from None
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class StoreBackedGovernedArtifactService:
    def __init__(self, db_path: str | Path | None = None, *, trusted_session: Any | None = None) -> None:
        self._db_path = Path(db_path) if db_path is not None else None
        self._trusted_session = trusted_session

    def resolve_tender_run_source(self, *, ctx: Any, run_id: str) -> Path:
        """Resolve only the original bytes pinned by a stored, scoped run."""
        from storage.tender_detection_run_store import TenderDetectionRunStore

        run = TenderDetectionRunStore(self._db_path).get_run(run_id)
        if run is None or str(run['organization_id']) != str(ctx.organization_id):
            raise ValueError('tender run not found')
        if str(run.get('workspace_id') or '') != str(ctx.workspace_id or ''):
            raise ValueError('tender run workspace mismatch')
        metadata = run.get('project_metadata') or {}
        session_id = metadata.get('target_session_id')
        if not session_id:
            raise ValueError('stored tender target session required')
        root = self._session_root(ctx, session_id)
        artifact = self._load_normalized_artifact(root, metadata.get('target_normalized_document_artifact_ref'))
        if artifact.get('artifact_hash') != metadata.get('target_normalized_document_artifact_hash'):
            raise ValueError('stored tender target artifact hash mismatch')
        source = self._resolve_session_ref(root, artifact.get('source_artifact_ref'))
        if not source.is_file() or _sha256_bytes(source.read_bytes()) != metadata.get('target_source_hash'):
            raise ValueError('tender target source hash mismatch')
        return source

    def _active_db_path(self) -> Path:
        if self._db_path is not None:
            return self._db_path
        from runtime_paths import sqlite_db_path

        return sqlite_db_path("eos.db", ".semantier/eos.db")

    def _session_root(self, ctx: Any, target_session_id: str | None = None) -> Path:
        if self._trusted_session is not None:
            if target_session_id is not None and target_session_id != self._trusted_session.session_id:
                raise ValueError("SESSION_CLAIM_MISMATCH")
            return self._trusted_session.session_root.resolve()
        if target_session_id is not None:
            workspace_home = getattr(ctx, 'hermes_home', None)
            if not workspace_home or not target_session_id:
                raise ValueError('governed workspace and target session required for artifact output')
            from runtime_paths import workspace_session_runs_root_from_hermes_home
            return workspace_session_runs_root_from_hermes_home(Path(workspace_home), target_session_id).parent
        session_id = os.environ.get("HERMES_SESSION_ID") or os.environ.get("SESSION_ID") or ""
        workspace_home = os.environ.get("HERMES_HOME") or os.environ.get("SESSION_HERMES_HOME") or ""
        if not workspace_home:
            workspace_home = str(getattr(ctx, "hermes_home", "") or "")
        if not workspace_home:
            raise ValueError("governed workspace home required for artifact output")
        if not session_id:
            raise ValueError("session binding required for governed artifact access")
        from runtime_paths import workspace_session_runs_root_from_hermes_home

        return workspace_session_runs_root_from_hermes_home(Path(workspace_home), session_id).parent

    def _resolve_session_ref(self, session_root: Path, ref: str) -> Path:
        raw = str(ref or "").strip()
        if not raw:
            raise ValueError("artifact reference required")
        if Path(raw).is_absolute() or raw.startswith(("http://", "https://", "file://")):
            raise ValueError("artifact reference must be governed session-relative")
        path = (session_root / raw).resolve()
        try:
            path.relative_to(session_root.resolve())
        except ValueError as exc:
            raise ValueError("artifact reference must remain inside the governed session") from exc
        return path

    def _load_normalized_artifact(self, session_root: Path, ref: str) -> dict[str, Any]:
        path = self._resolve_session_ref(session_root, ref)
        if not path.is_file():
            raise ValueError("normalized document artifact not found")
        artifact = json.loads(path.read_text(encoding="utf-8"))
        expected_hash = artifact.get("artifact_hash")
        actual_hash = "sha256:" + sha256(
            canonical_json({key: value for key, value in artifact.items() if key != "artifact_hash"}).encode(
                "utf-8"
            )
        ).hexdigest()
        if expected_hash != actual_hash:
            raise ValueError("normalized document artifact hash mismatch")
        document = artifact.get("document")
        if not isinstance(document, dict) or document.get("schema_version") != "canonical_document.v2":
            raise ValueError("canonical_document.v2 is required for labeled DOCX output")
        return artifact

    def resolve_artifact(
        self,
        ref: str,
        organization_id: str,
        workspace_id: str | None,
    ) -> GovernedArtifactDTO:
        context = RuntimeContextDTO(
            organization_id=organization_id,
            workspace_id=workspace_id,
            actor_id="artifact_resolver",
            actor_role="system",
        )
        session_root = self._session_root(context)
        path = self._resolve_session_ref(session_root, ref)
        if not path.is_file():
            raise ValueError("governed artifact not found")
        artifact_hash = _sha256_bytes(path.read_bytes())
        manifest_path = self._resolve_session_ref(session_root, f"{ref}.manifest.json")
        metadata: dict[str, Any] = {}
        media_type: str | None = None
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(manifest, dict):
                if manifest.get("content_hash") and manifest["content_hash"] != artifact_hash:
                    raise ValueError("governed artifact manifest hash mismatch")
                media_type = str(manifest.get("media_type") or "") or None
                if isinstance(manifest.get("metadata"), dict):
                    metadata = dict(manifest["metadata"])
        return GovernedArtifactDTO(
            artifact_ref=ref,
            content_hash=artifact_hash,
            media_type=media_type,
            metadata=metadata,
        )

    def verify_artifact_hash(
        self,
        ref: str,
        expected_hash: str,
        context: RuntimeContextDTO,
    ) -> bool:
        session_root = self._session_root(context)
        path = self._resolve_session_ref(session_root, ref)
        if not path.is_file():
            return False
        actual = _sha256_bytes(path.read_bytes())
        return actual.removeprefix("sha256:") == str(expected_hash).strip().removeprefix("sha256:")

    def open_artifact_for_read(self, ref: str, context: RuntimeContextDTO) -> BinaryIO:
        session_root = self._session_root(context)
        path = self._resolve_session_ref(session_root, ref)
        if not path.is_file():
            raise ValueError("governed artifact not found")
        return path.open("rb")

    def create_derivative_artifact(
        self,
        *,
        parent_ref: str,
        media_type: str,
        bytes_or_stream: bytes | BinaryIO,
        metadata: dict[str, Any],
        context: RuntimeContextDTO,
    ) -> GovernedArtifactDTO:
        if isinstance(bytes_or_stream, bytes):
            artifact_bytes = bytes_or_stream
        else:
            artifact_bytes = bytes_or_stream.read()
        if not isinstance(artifact_bytes, bytes):
            raise ValueError("artifact bytes required")
        session_root = self._session_root(context)
        artifact_hash = _sha256_bytes(artifact_bytes)
        artifact_id = str(metadata.get("artifact_id") or f"artifact_{uuid4().hex}")
        safe_artifact_id = "".join(ch if ch.isascii() and (ch.isalnum() or ch in {"_", "-"}) else "_" for ch in artifact_id)
        if not safe_artifact_id:
            raise ValueError("artifact_id must contain an ASCII-stable identifier")
        output_ref = str(metadata.get("artifact_ref") or f"artifacts/governed/{safe_artifact_id}.json")
        output_path = self._resolve_session_ref(session_root, output_ref)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_ref == parent_ref:
            raise ValueError("ORIGINAL_OVERWRITE_REJECTED")
        _publish_immutable_bytes(output_path, artifact_bytes)
        manifest = {
            "artifact_ref": output_ref,
            "content_hash": artifact_hash,
            "media_type": media_type,
            "parent_ref": parent_ref,
            "metadata": metadata,
            "created_at": _now_iso(),
        }
        manifest_ref = f"{output_ref}.manifest.json"
        manifest_path = self._resolve_session_ref(session_root, manifest_ref)
        if manifest_path.exists():
            existing=json.loads(manifest_path.read_text(encoding="utf-8"))
            if any(existing.get(key)!=manifest.get(key) for key in ("artifact_ref","content_hash","media_type","parent_ref","metadata")):
                raise ValueError("DERIVATIVE_CONFLICT")
        else:
            try:
                _publish_immutable_bytes(manifest_path, canonical_json(manifest).encode("utf-8"))
            except ValueError:
                existing=json.loads(manifest_path.read_text(encoding="utf-8"))
                if any(existing.get(key)!=manifest.get(key) for key in ("artifact_ref","content_hash","media_type","parent_ref","metadata")):
                    raise ValueError("DERIVATIVE_CONFLICT") from None
        return GovernedArtifactDTO(
            artifact_ref=output_ref,
            content_hash=artifact_hash,
            media_type=media_type,
            metadata={**metadata, "parent_ref": parent_ref, "manifest_ref": manifest_ref},
        )

    def create_labeled_docx_derivative(
        self,
        *,
        ctx: Any,
        run: dict[str, Any],
        created_by: str,
        comment_writer_version: str,
        writer: Any,
    ) -> dict[str, Any]:
        run_id = str(run.get("run_id") or "")
        if not run_id:
            raise ValueError("detection run id required")
        path = self._active_db_path()
        init_db(path)
        with get_conn(path) as conn:
            if not enabled():
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA foreign_keys=ON")
            persisted_run = conn.execute(
                sql_query("""
                SELECT organization_id, workspace_id
                FROM tender_detection_runs
                WHERE run_id = ?
                """,'\n                SELECT organization_id, workspace_id\n                FROM tender_detection_runs\n                WHERE run_id = %s\n                '),
                (run_id,),
            ).fetchone()
        if persisted_run is None:
            raise ValueError(f"detection run not found: {run_id}")
        if str(persisted_run["organization_id"]) != str(ctx.organization_id):
            raise ValueError("detection run organization mismatch")
        persisted_workspace_id = persisted_run["workspace_id"]
        if persisted_workspace_id is not None and str(persisted_workspace_id) != str(ctx.workspace_id):
            raise ValueError("detection run workspace mismatch")

        project_metadata = run.get("project_metadata") or {}
        normalized_ref = project_metadata.get("target_normalized_document_artifact_ref")
        if not normalized_ref:
            raise ValueError("target normalized document artifact ref required")
        session_root = self._session_root(ctx, project_metadata.get('target_session_id'))
        artifact = self._load_normalized_artifact(session_root, str(normalized_ref))
        expected_hash = project_metadata.get('target_normalized_document_artifact_hash')
        if expected_hash and artifact.get('artifact_hash') != expected_hash:
            raise ValueError('stored tender target artifact hash mismatch')
        source_ref = str(artifact.get("source_artifact_ref") or "")
        source_docx_path = self._resolve_session_ref(session_root, source_ref)
        if not source_docx_path.is_file():
            raise ValueError("source DOCX artifact not found")
        parent_bytes = source_docx_path.read_bytes()
        parent_hash = _sha256_bytes(parent_bytes)
        output_ref = f"artifacts/labeled-docx/labeled-{run['run_id']}.docx"
        output_path = self._resolve_session_ref(session_root, output_ref)
        manifest_ref = f"artifacts/labeled-docx/labeled-{run['run_id']}.manifest.json"
        manifest_path = self._resolve_session_ref(session_root, manifest_ref)
        finding_set_hash = content_hash(
            [
                {
                    "finding_id": finding["finding_id"],
                    "target_anchor_ref": finding.get("target_anchor_ref"),
                    "start_offset": finding.get("start_offset"),
                    "end_offset": finding.get("end_offset"),
                    "content_hash": finding["content_hash"],
                }
                for finding in run.get("findings", [])
            ]
        )
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                raise ValueError("labeled DOCX derivative manifest invalid")
            expected = {
                "run_id": run["run_id"],
                "parent_target_artifact_ref": source_ref,
                "parent_target_hash": parent_hash,
                "labeled_docx_artifact_ref": output_ref,
                "comment_writer_version": comment_writer_version,
                "finding_set_hash": finding_set_hash,
            }
            for key, value in expected.items():
                if manifest.get(key) != value:
                    raise ValueError("labeled DOCX derivative idempotency conflict")
            if not output_path.is_file():
                raise ValueError("labeled DOCX derivative artifact missing")
            if manifest.get("labeled_docx_hash") != _sha256_bytes(output_path.read_bytes()):
                raise ValueError("labeled DOCX derivative artifact hash mismatch")
            manifest["manifest_artifact_ref"] = manifest_ref
            manifest["idempotency_status"] = "reused"
            return manifest
        labeled_bytes, projection_result = writer(
            source_docx_bytes=parent_bytes,
            canonical_document=artifact["document"],
            findings=list(run.get("findings", [])),
            created_by=created_by,
        )
        if isinstance(projection_result, dict):
            comments_written = int(projection_result.get("comments_written") or 0)
            comment_projection_manifest = projection_result
        else:
            comments_written = int(projection_result)
            comment_projection_manifest = {
                "comment_projection_schema_version": "legacy_comment_count.v1",
                "comments_written": comments_written,
                "unsupported_findings": [],
                "comment_ranges": [],
            }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(labeled_bytes)
        labeled_hash = _sha256_bytes(labeled_bytes)
        generated_at = _now_iso()
        row = {
            "labeled_docx_derivative_id": f"sldocx_{uuid4().hex}",
            "run_id": run["run_id"],
            "parent_target_artifact_ref": source_ref,
            "parent_target_hash": parent_hash,
            "labeled_docx_artifact_ref": output_ref,
            "labeled_docx_hash": labeled_hash,
            "comment_writer_version": comment_writer_version,
            "finding_set_hash": finding_set_hash,
            "generated_at": generated_at,
        }
        row["content_hash"] = content_hash(row)
        with get_conn(path) as conn:
            if not enabled():conn.execute("PRAGMA foreign_keys=ON")
            conn.execute(
                sql_query("""
                INSERT INTO sensitive_labeling_labeled_docx_derivatives (
                  labeled_docx_derivative_id, run_id, parent_target_artifact_ref,
                  parent_target_hash, labeled_docx_artifact_ref, labeled_docx_hash,
                  comment_writer_version, finding_set_hash, generated_at, content_hash
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,'\n                INSERT INTO sensitive_labeling_labeled_docx_derivatives (\n                  labeled_docx_derivative_id, run_id, parent_target_artifact_ref,\n                  parent_target_hash, labeled_docx_artifact_ref, labeled_docx_hash,\n                  comment_writer_version, finding_set_hash, generated_at, content_hash\n                )\n                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)\n                '),
                (
                    row["labeled_docx_derivative_id"],
                    row["run_id"],
                    row["parent_target_artifact_ref"],
                    row["parent_target_hash"],
                    row["labeled_docx_artifact_ref"],
                    row["labeled_docx_hash"],
                    row["comment_writer_version"],
                    row["finding_set_hash"],
                    row["generated_at"],
                    row["content_hash"],
                ),
            )
        manifest = {
            **row,
            "comments_written": comments_written,
            "comment_projection_manifest": comment_projection_manifest,
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(canonical_json(manifest), encoding="utf-8")
        manifest["manifest_artifact_ref"] = manifest_ref
        manifest["idempotency_status"] = "created"
        return manifest

    def get_labeled_docx_derivative(
        self,
        artifact_id: str,
        context: RuntimeContextDTO,
    ) -> dict[str, Any] | None:
        path = self._active_db_path()
        init_db(path)
        with get_conn(path) as conn:
            if not enabled():conn.row_factory = sqlite3.Row
            row = conn.execute(
                sql_query("""
                SELECT d.*
                FROM sensitive_labeling_labeled_docx_derivatives d
                JOIN tender_detection_runs r ON r.run_id = d.run_id
                WHERE d.labeled_docx_derivative_id = ?
                   OR d.labeled_docx_artifact_ref = ?
                """,'\n                SELECT d.*\n                FROM sensitive_labeling_labeled_docx_derivatives d\n                JOIN tender_detection_runs r ON r.run_id = d.run_id\n                WHERE d.labeled_docx_derivative_id = %s\n                   OR d.labeled_docx_artifact_ref = %s\n                '),
                (artifact_id, artifact_id),
            ).fetchone()
            if row is None:
                return None
            run = conn.execute(
                sql_query("SELECT organization_id, workspace_id FROM tender_detection_runs WHERE run_id = ?",'SELECT organization_id, workspace_id FROM tender_detection_runs WHERE run_id = %s'),
                (row["run_id"],),
            ).fetchone()
        if run is None or run["organization_id"] != context.organization_id:
            return None
        if context.workspace_id and (run["workspace_id"] or None) not in {None, context.workspace_id}:
            return None
        return dict(row)
