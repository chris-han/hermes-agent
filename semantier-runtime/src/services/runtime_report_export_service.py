"""Deterministic runtime report rendering and content-addressed publication."""

from __future__ import annotations

from storage.hosted_postgres import sql_query,begin_write,in_transaction

from datetime import datetime, timezone
import html
import json
import os
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from contracts.runtime_inspection_report import RuntimeInspectionReportV1, RuntimeReportReplayInputsV1
from contracts.ontology_building import ArtifactPinV1
from contracts.semantic_kernel import WorkspaceScope
from contracts.runtime_report_export import (
    ExportBytesV1,
    PinnedRenderProfileV1,
    RuntimeAccessProjection,
    RuntimeReportExportReceiptV1,
)
from eos.artifact_hashing import content_hash
from storage.ontology_build_store import OntologyBuildStore
from storage.runtime_inspection_report_store import RuntimeInspectionReportStore


def server_runtime_export_profile(
    *,
    report: RuntimeInspectionReportV1,
    format: str,
    access_projection: RuntimeAccessProjection,
) -> PinnedRenderProfileV1:
    """Return the deterministic server profile accepted for Workbench export."""
    if format not in ("JSON", "HTML"):
        raise ValueError("UNSUPPORTED_FORMAT")
    if access_projection not in ("MEMBER", "AUDIT"):
        raise ValueError("INVALID_ACCESS_PROJECTION")
    redaction_basis = {
        "schema_version": "runtime_export_redaction_policy.v1",
        "access_projection": access_projection,
    }
    redaction_digest = content_hash(redaction_basis)
    suffix = "json" if format == "JSON" else "html"
    return PinnedRenderProfileV1.create(
        format=format,
        media_type="application/json" if format == "JSON" else "text/html",
        renderer=report.renderer_pin,
        template=report.template_pin,
        environment=report.environment_pin,
        redaction_policy=ArtifactPinV1(
            artifact_kind="runtime_export_redaction_policy",
            artifact_ref=f"runtime-export-redaction:{redaction_digest}",
            artifact_hash=f"sha256:{redaction_digest}",
        ),
        canonical_filename=f"runtime-report-{report.report_hash.removeprefix('sha256:')[:12]}.{suffix}",
        archive_timestamp=report.subject.recorded_at,
        options={"access_projection": access_projection},
    )


class RuntimeReportExportService:
    @staticmethod
    def render(
        *,
        report: RuntimeInspectionReportV1,
        replay_inputs: RuntimeReportReplayInputsV1,
        render_profile: PinnedRenderProfileV1,
        access_projection: RuntimeAccessProjection,
    ) -> ExportBytesV1:
        if report.replay_access.replay_pin != replay_inputs.replay_pin:
            raise ValueError("PIN_MISMATCH")
        if access_projection not in ("MEMBER", "AUDIT"):
            raise ValueError("INVALID_ACCESS_PROJECTION")
        if render_profile.format not in ("JSON", "HTML"):
            raise ValueError("UNSUPPORTED_FORMAT")
        payload = report.model_dump(mode="json")
        if access_projection == "MEMBER":
            payload["history"].pop("handoff_pins", None)
            payload["history"].pop("learning_pins", None)
            payload["history"].pop("attribution_pins", None)
        if render_profile.format == "JSON":
            content = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
            media_type = "application/json"
        else:
            title = html.escape(report.subject.document_title)
            status = html.escape(report.summary.status)
            content = (
                "<!doctype html><html><head><meta charset=\"utf-8\"><title>"
                + title
                + "</title></head><body><h1>"
                + title
                + "</h1><p data-status=\""
                + status
                + "\">"
                + status
                + "</p><pre>"
                + html.escape(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
                + "</pre></body></html>"
            ).encode()
            media_type = "text/html"
        digest = sha256(content).hexdigest()
        return ExportBytesV1(
            format=render_profile.format,
            media_type=media_type,
            canonical_filename=render_profile.canonical_filename,
            content=content,
            content_hash=f"sha256:{digest}",
            size_bytes=len(content),
            render_profile=ArtifactPinV1(
                artifact_kind="pinned_render_profile",
                artifact_ref=render_profile.profile_ref,
                artifact_hash=render_profile.profile_hash,
            ),
            access_projection=access_projection,
            source_timestamps=(report.subject.business_effective_at, report.subject.recorded_at),
        )

    @staticmethod
    def publish(
        *,
        build_store: OntologyBuildStore,
        workspace_root: Path,
        scope: WorkspaceScope,
        report_pin: ArtifactPinV1,
        replay_inputs_pin: ArtifactPinV1,
        render_profile: PinnedRenderProfileV1,
        access_projection: RuntimeAccessProjection,
        request_id: str,
        request_hash: str,
        actor_ref: str,
        expected_export_head_pin: ArtifactPinV1 | None = None,
    ) -> RuntimeReportExportReceiptV1:
        """Render and publish one immutable export; identical retries reuse the receipt."""
        if not isinstance(build_store, OntologyBuildStore):
            raise TypeError("runtime export requires the native workspace store")
        scope = WorkspaceScope.model_validate(scope.model_dump(mode="json"))
        if expected_export_head_pin is not None:
            raise ValueError("STALE_EXPORT_HEAD")
        if not request_id or not actor_ref or not request_hash.startswith("sha256:"):
            raise ValueError("INVALID_STATE")

        report_store = RuntimeInspectionReportStore(build_store)
        report = report_store.get(scope=scope, report_pin=report_pin)
        replay = report_store.get_replay_inputs(scope=scope, replay_pin=replay_inputs_pin)
        expected_profile = server_runtime_export_profile(
            report=report,
            format=render_profile.format,
            access_projection=access_projection,
        )
        if render_profile != expected_profile:
            raise ValueError("PIN_MISMATCH")
        rendered = RuntimeReportExportService.render(
            report=report,
            replay_inputs=replay,
            render_profile=render_profile,
            access_projection=access_projection,
        )

        connection = build_store._connection
        key = (
            scope.organization_id,
            scope.workspace_id,
            actor_ref,
            "EXPORT_RUNTIME_REPORT",
            request_id,
        )
        row = connection.execute(
            sql_query("SELECT request_hash,receipt_ref,receipt_hash FROM workspace_operation_request_index "
            "WHERE organization_id=? AND workspace_id=? AND actor_ref=? AND operation=? AND request_id=?",'SELECT request_hash,receipt_ref,receipt_hash FROM workspace_operation_request_index WHERE organization_id=%s AND workspace_id=%s AND actor_ref=%s AND operation=%s AND request_id=%s'),
            key,
        ).fetchone()
        if row is not None:
            if row[0] != request_hash:
                raise ValueError("REQUEST_CONFLICT")
            existing = build_store.get(
                organization_id=scope.organization_id,
                workspace_id=scope.workspace_id,
                artifact_ref=row[1],
                contract_type=RuntimeReportExportReceiptV1,
            )
            if existing is None or existing.receipt_hash != row[2]:
                raise ValueError("INCOMPLETE_CLOSURE")
            return existing

        root = Path(workspace_root).resolve()
        digest = rendered.content_hash.removeprefix("sha256:")
        suffix = ".json" if rendered.format == "JSON" else ".html"
        relative = Path("artifacts") / "runtime-exports" / "sha256" / digest[:2] / f"{digest}{suffix}"
        target = (root / relative).resolve()
        if root not in target.parents:
            raise ValueError("FORBIDDEN_SCOPE")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.is_symlink() or not target.is_file() or target.read_bytes() != rendered.content:
                raise ValueError("PIN_MISMATCH")
        else:
            temporary = target.parent / f".{target.name}.{uuid4().hex}.tmp"
            try:
                with open(temporary, "xb") as stream:
                    stream.write(rendered.content)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, target)
            finally:
                if temporary.exists():
                    temporary.unlink()

        published = ArtifactPinV1(
            artifact_kind="published_artifact",
            artifact_ref=relative.as_posix(),
            artifact_hash=rendered.content_hash,
        )
        now = datetime.now(timezone.utc)
        receipt = RuntimeReportExportReceiptV1.create(
            scope=scope,
            report=report_pin,
            replay_inputs=replay_inputs_pin,
            format=rendered.format,
            access_projection=access_projection,
            render_profile=rendered.render_profile,
            content_hash=rendered.content_hash,
            size_bytes=rendered.size_bytes,
            media_type=rendered.media_type,
            stage_ref=f"runtime-export-stage:{sha256(request_hash.encode()).hexdigest()}",
            published_artifact=published,
            request_id=request_id,
            request_hash=request_hash,
            state="COMPLETE",
            predecessor_intent=None,
            actor_ref=actor_ref,
            recorded_at=now,
            completed_at=now,
            abandoned_at=None,
        )

        def validate(current):
            row = current.execute(
                sql_query("SELECT request_hash,receipt_ref,receipt_hash FROM workspace_operation_request_index "
                "WHERE organization_id=? AND workspace_id=? AND actor_ref=? AND operation=? AND request_id=?",'SELECT request_hash,receipt_ref,receipt_hash FROM workspace_operation_request_index WHERE organization_id=%s AND workspace_id=%s AND actor_ref=%s AND operation=%s AND request_id=%s'),
                key,
            ).fetchone()
            if row is None:
                return True
            if row[0] != request_hash:
                raise ValueError("REQUEST_CONFLICT")
            return False

        def mutate(current):
            current.execute(
                sql_query("INSERT INTO workspace_operation_request_index "
                "(organization_id,workspace_id,actor_ref,operation,request_id,request_hash,receipt_ref,receipt_hash) "
                "VALUES (?,?,?,?,?,?,?,?)",'INSERT INTO workspace_operation_request_index (organization_id,workspace_id,actor_ref,operation,request_id,request_hash,receipt_ref,receipt_hash) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)'),
                (*key, request_hash, receipt.receipt_ref, receipt.receipt_hash),
            )

        inserted = build_store.atomic_write(
            artifacts=(receipt,),
            validate_before_insert=validate,
            mutate_indexes=mutate,
        )
        if inserted is None:
            row = connection.execute(
                sql_query("SELECT request_hash,receipt_ref,receipt_hash FROM workspace_operation_request_index "
                "WHERE organization_id=? AND workspace_id=? AND actor_ref=? AND operation=? AND request_id=?",'SELECT request_hash,receipt_ref,receipt_hash FROM workspace_operation_request_index WHERE organization_id=%s AND workspace_id=%s AND actor_ref=%s AND operation=%s AND request_id=%s'),
                key,
            ).fetchone()
            if row is None or row[0] != request_hash:
                raise ValueError("INCOMPLETE_CLOSURE")
            existing = build_store.get(
                organization_id=scope.organization_id,
                workspace_id=scope.workspace_id,
                artifact_ref=row[1],
                contract_type=RuntimeReportExportReceiptV1,
            )
            if existing is None or existing.receipt_hash != row[2]:
                raise ValueError("INCOMPLETE_CLOSURE")
            return existing
        return receipt


__all__ = ["RuntimeReportExportService", "server_runtime_export_profile"]
