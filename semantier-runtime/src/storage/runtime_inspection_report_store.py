"""Atomic persistence for immutable runtime inspection reports and replay inputs."""

from __future__ import annotations

from storage.hosted_postgres import sql_query,begin_write,in_transaction,enabled,require_tables

from datetime import datetime, timezone

from contracts.ontology_building import ArtifactPinV1
from contracts.runtime_inspection_report import (
    RuntimeInspectionReportV1,
    RuntimeReportReceiptV1,
    RuntimeReportReplayInputsV1,
)
from contracts.semantic_kernel import WorkspaceScope
from storage.ontology_build_store import OntologyBuildStore


class RuntimeInspectionReportStore:
    def __init__(self, build_store: OntologyBuildStore):
        if not isinstance(build_store, OntologyBuildStore):
            raise TypeError("runtime reports require the native workspace store")
        self._store = build_store
        require_tables(self._store._connection,'runtime_report_heads') if enabled() else self._store._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS runtime_report_heads (
                organization_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                lineage_ref TEXT NOT NULL,
                revision INTEGER NOT NULL,
                report_ref TEXT NOT NULL,
                report_hash TEXT NOT NULL,
                PRIMARY KEY (organization_id, workspace_id, lineage_ref)
            )
            """
        )
        self._store._connection.commit()

    @staticmethod
    def _scope(scope: WorkspaceScope) -> WorkspaceScope:
        return WorkspaceScope.model_validate(scope.model_dump(mode="json"))

    @staticmethod
    def _same_pin(left: ArtifactPinV1, right: ArtifactPinV1 | None) -> bool:
        return right is not None and left == right

    def _receipt_for_request(self, scope: WorkspaceScope, actor_ref: str, request_id: str, request_hash: str):
        row = self._store._connection.execute(
            sql_query("SELECT request_hash,receipt_ref,receipt_hash FROM workspace_operation_request_index "
            "WHERE organization_id=? AND workspace_id=? AND actor_ref=? AND operation=? AND request_id=?",'SELECT request_hash,receipt_ref,receipt_hash FROM workspace_operation_request_index WHERE organization_id=%s AND workspace_id=%s AND actor_ref=%s AND operation=%s AND request_id=%s'),
            (scope.organization_id, scope.workspace_id, actor_ref, "PUBLISH_RUNTIME_REPORT", request_id),
        ).fetchone()
        if row is None:
            return None
        if row[0] != request_hash:
            raise ValueError("REQUEST_CONFLICT")
        return self._load(scope, ArtifactPinV1(
            artifact_kind="runtime_report_receipt",
            artifact_ref=row[1], artifact_hash=row[2],
        ), RuntimeReportReceiptV1)

    def put_report_with_replay(
        self,
        *,
        scope: WorkspaceScope,
        report: RuntimeInspectionReportV1,
        replay_inputs: RuntimeReportReplayInputsV1,
        expected_predecessor: ArtifactPinV1 | None,
        request_id: str,
        request_hash: str,
        actor_ref: str,
    ) -> RuntimeReportReceiptV1:
        scope = self._scope(scope)
        report = RuntimeInspectionReportV1.model_validate(report.model_dump(mode="json"))
        replay_inputs = RuntimeReportReplayInputsV1.model_validate(replay_inputs.model_dump(mode="json"))
        if (report.organization_id, report.workspace_id) != (scope.organization_id, scope.workspace_id):
            raise ValueError("FORBIDDEN_SCOPE")
        if report.replay_access.replay_pin != replay_inputs.replay_pin:
            raise ValueError("PIN_MISMATCH")
        if report.history.predecessor_report_pin != expected_predecessor:
            raise ValueError("PREDECESSOR_MISMATCH")
        existing = self._receipt_for_request(scope, actor_ref, request_id, request_hash)
        if existing is not None:
            return existing
        if not request_id or not actor_ref or not request_hash.startswith("sha256:"):
            raise ValueError("INVALID_STATE")

        connection = self._store._connection
        lineage_ref = report.report_ref
        revision = 0
        receipt = RuntimeReportReceiptV1.create(
            organization_id=scope.organization_id,
            workspace_id=scope.workspace_id,
            request_id=request_id,
            request_hash=request_hash,
            report_pin=report.report_pin,
            replay_pin=replay_inputs.replay_pin,
            predecessor_pin=expected_predecessor,
            actor_ref=actor_ref,
            recorded_at=datetime.now(timezone.utc),
        )

        def validate_before_insert(current):
            nonlocal lineage_ref, revision
            if current is not connection:
                raise ValueError("INVALID_STATE")
            existing = self._receipt_for_request(scope, actor_ref, request_id, request_hash)
            if existing is not None:
                return False
            if expected_predecessor is not None:
                row = current.execute(
                    sql_query("SELECT lineage_ref,revision,report_ref,report_hash FROM runtime_report_heads "
                    "WHERE organization_id=? AND workspace_id=? AND report_ref=? AND report_hash=?",'SELECT lineage_ref,revision,report_ref,report_hash FROM runtime_report_heads WHERE organization_id=%s AND workspace_id=%s AND report_ref=%s AND report_hash=%s'),
                    (scope.organization_id, scope.workspace_id,
                     expected_predecessor.artifact_ref, expected_predecessor.artifact_hash),
                ).fetchone()
                if row is None:
                    raise ValueError("STALE_REVISION")
                lineage_ref, revision = row[0], int(row[1]) + 1
            else:
                row = current.execute(
                    sql_query("SELECT lineage_ref,report_ref,report_hash FROM runtime_report_heads "
                    "WHERE organization_id=? AND workspace_id=? AND lineage_ref=?",'SELECT lineage_ref,report_ref,report_hash FROM runtime_report_heads WHERE organization_id=%s AND workspace_id=%s AND lineage_ref=%s'),
                    (scope.organization_id, scope.workspace_id, lineage_ref),
                ).fetchone()
                if row is not None and tuple(row[1:]) != (report.report_ref, report.report_hash):
                    raise ValueError("STALE_REVISION")
            return True

        def mutate_indexes(current):
            current.execute(
                sql_query("INSERT INTO workspace_operation_request_index "
                "(organization_id,workspace_id,actor_ref,operation,request_id,request_hash,receipt_ref,receipt_hash) "
                "VALUES (?,?,?,?,?,?,?,?)",'INSERT INTO workspace_operation_request_index (organization_id,workspace_id,actor_ref,operation,request_id,request_hash,receipt_ref,receipt_hash) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)'),
                (scope.organization_id, scope.workspace_id, actor_ref, "PUBLISH_RUNTIME_REPORT",
                 request_id, request_hash, receipt.receipt_ref, receipt.receipt_hash),
            )
            existing_head = current.execute(
                sql_query("SELECT report_ref,report_hash FROM runtime_report_heads "
                "WHERE organization_id=? AND workspace_id=? AND lineage_ref=?",'SELECT report_ref,report_hash FROM runtime_report_heads WHERE organization_id=%s AND workspace_id=%s AND lineage_ref=%s'),
                (scope.organization_id, scope.workspace_id, lineage_ref),
            ).fetchone()
            if existing_head is None:
                current.execute(
                    sql_query("INSERT INTO runtime_report_heads "
                    "(organization_id,workspace_id,lineage_ref,revision,report_ref,report_hash) VALUES (?,?,?,?,?,?)",'INSERT INTO runtime_report_heads (organization_id,workspace_id,lineage_ref,revision,report_ref,report_hash) VALUES (%s,%s,%s,%s,%s,%s)'),
                    (scope.organization_id, scope.workspace_id, lineage_ref, revision,
                     report.report_ref, report.report_hash),
                )
            elif expected_predecessor is not None:
                if tuple(existing_head) != (
                    expected_predecessor.artifact_ref,
                    expected_predecessor.artifact_hash,
                ):
                    raise ValueError("STALE_REVISION")
                updated = current.execute(
                    sql_query("UPDATE runtime_report_heads SET revision=?,report_ref=?,report_hash=? "
                    "WHERE organization_id=? AND workspace_id=? AND lineage_ref=? "
                    "AND revision=? AND report_ref=? AND report_hash=?",'UPDATE runtime_report_heads SET revision=%s,report_ref=%s,report_hash=%s WHERE organization_id=%s AND workspace_id=%s AND lineage_ref=%s AND revision=%s AND report_ref=%s AND report_hash=%s'),
                    (
                        revision, report.report_ref, report.report_hash,
                        scope.organization_id, scope.workspace_id, lineage_ref,
                        revision - 1, expected_predecessor.artifact_ref,
                        expected_predecessor.artifact_hash,
                    ),
                )
                if updated.rowcount != 1:
                    raise ValueError("STALE_REVISION")
            elif tuple(existing_head) != (report.report_ref, report.report_hash):
                raise ValueError("STALE_REVISION")

        inserted = self._store.atomic_write(
            artifacts=(replay_inputs, report, receipt),
            validate_before_insert=validate_before_insert,
            mutate_indexes=mutate_indexes,
        )
        if inserted is None:
            retry = self._receipt_for_request(scope, actor_ref, request_id, request_hash)
            if retry is None:
                raise ValueError("INCOMPLETE_CLOSURE")
            return retry
        return receipt

    def _load(self, scope: WorkspaceScope, pin: ArtifactPinV1, model):
        expected_kind = {
            RuntimeInspectionReportV1: "runtime_inspection_report",
            RuntimeReportReplayInputsV1: "runtime_report_replay_inputs",
            RuntimeReportReceiptV1: "runtime_report_receipt",
        }[model]
        if pin.artifact_kind != expected_kind:
            raise ValueError("PIN_MISMATCH")
        raw = self._store.get_raw(
            organization_id=scope.organization_id,
            workspace_id=scope.workspace_id,
            artifact_ref=pin.artifact_ref,
        )
        if raw is None:
            raise ValueError("PIN_NOT_FOUND")
        if raw["artifact_hash"] != pin.artifact_hash:
            raise ValueError("PIN_MISMATCH")
        value = model.model_validate(raw["payload"])
        ref_field = {RuntimeInspectionReportV1: "report_ref", RuntimeReportReplayInputsV1: "replay_ref", RuntimeReportReceiptV1: "receipt_ref"}[model]
        if getattr(value, ref_field) != pin.artifact_ref:
            raise ValueError("PIN_MISMATCH")
        return value

    def get(self, *, scope: WorkspaceScope, report_pin: ArtifactPinV1) -> RuntimeInspectionReportV1:
        return self._load(self._scope(scope), report_pin, RuntimeInspectionReportV1)

    def get_replay_inputs(self, *, scope: WorkspaceScope, replay_pin: ArtifactPinV1) -> RuntimeReportReplayInputsV1:
        return self._load(self._scope(scope), replay_pin, RuntimeReportReplayInputsV1)


__all__ = ["RuntimeInspectionReportStore"]
