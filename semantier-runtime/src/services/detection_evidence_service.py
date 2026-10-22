from __future__ import annotations

from typing import Any, Protocol

from eos.artifact_hashing import canonical_json, content_hash
from contracts.tender_sensitive_term_detection_modeling_core import (
    DetectionEvidenceDTO,
    DetectionRunCreateDTO,
    DetectionRunDTO,
    FindingSetDTO,
    OccurrenceFindingDTO,
    RuntimeConflictDTO,
    RuntimeConflictSetDTO,
    RuntimeContextDTO,
)


class DetectionEvidenceService(Protocol):
    def create_detection_run_idempotent(
        self,
        request: DetectionRunCreateDTO,
        context: RuntimeContextDTO,
    ) -> DetectionRunDTO: ...

    def persist_occurrence_findings(
        self,
        run_id: str,
        findings: list[OccurrenceFindingDTO],
        context: RuntimeContextDTO,
    ) -> FindingSetDTO: ...

    def persist_runtime_conflicts(
        self,
        run_id: str,
        conflicts: list[RuntimeConflictDTO],
        context: RuntimeContextDTO,
    ) -> RuntimeConflictSetDTO: ...

    def finalize_detection_run(
        self,
        run_id: str,
        finding_set_hash: str,
        conflict_set_hash: str,
        status: str,
        context: RuntimeContextDTO,
    ) -> DetectionRunDTO: ...

    def get_detection_evidence(
        self,
        run_id: str,
        context: RuntimeContextDTO,
    ) -> DetectionEvidenceDTO: ...


class StoreBackedDetectionEvidenceService:
    def __init__(self, run_store: Any | None = None) -> None:
        if run_store is None:
            from storage.tender_detection_run_store import TenderDetectionRunStore

            run_store = TenderDetectionRunStore()
        self._run_store = run_store

    @property
    def authoritative_db_path(self):
        """Public composition path for services sharing this authority store."""

        db_path = getattr(self._run_store, "db_path", None)
        if db_path is None:
            raise RuntimeError("detection run store does not expose its authoritative db path")
        return db_path

    def create_legacy_detection_run(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return self._run_store.create_run(**kwargs)

    def create_detection_run_idempotent(
        self,
        request: DetectionRunCreateDTO,
        context: RuntimeContextDTO,
    ) -> DetectionRunDTO:
        request_hash = content_hash(
            {
                "request": request.__dict__,
                "context": context.__dict__,
            }
        )
        metadata = dict(request.metadata)
        run = self._run_store.create_pending_run_idempotent(
            run_id=metadata.get("run_id"),
            tender_document_id=request.tender_document_id,
            document_text=str(metadata["document_text"]),
            organization_id=context.organization_id,
            workspace_id=context.workspace_id,
            parent_run_id=metadata.get("parent_run_id"),
            project_metadata={
                key: value
                for key, value in metadata.items()
                if key
                not in {
                    "document_text",
                    "parent_run_id",
                    "review_purpose",
                    "caller_role",
                    "requested_rule_families",
                    "activation_resolver_policy_version",
                    "ignored_local_rules",
                }
            },
            review_purpose=str(metadata.get("review_purpose", "pre_release_check")),
            caller_role=str(metadata.get("caller_role", context.actor_role or "member")),
            requested_rule_families=list(metadata.get("requested_rule_families", [])),
            activation_set_snapshot_id=request.activation_set_snapshot_id,
            activation_resolver_policy_version=str(
                metadata["activation_resolver_policy_version"]
            ),
            resolved_activation_set_hash=request.resolved_activation_set_hash,
            ignored_local_rules=list(metadata.get("ignored_local_rules", [])),
            idempotency_key=request.idempotency_key,
            request_hash=request_hash,
        )
        return DetectionRunDTO(
            run_id=str(run["run_id"]),
            status=str(run.get("run_status") or "PENDING").lower(),
            content_hash=str(run.get("content_hash") or ""),
            metadata={
                "idempotency_key": request.idempotency_key,
                "request_hash": request_hash,
                "run": run,
            },
        )

    def persist_occurrence_findings(
        self,
        run_id: str,
        findings: list[OccurrenceFindingDTO],
        context: RuntimeContextDTO,
    ) -> FindingSetDTO:
        self._get_scoped_run(run_id, context)
        finding_set_hash = content_hash([finding.__dict__ for finding in findings])
        self._run_store.persist_findings(
            run_id=run_id,
            findings=[
                self._finding_dto_to_store_payload(finding)
                for finding in findings
            ],
            finding_set_hash=finding_set_hash,
        )
        return FindingSetDTO(
            run_id=run_id,
            finding_count=len(findings),
            finding_set_hash=finding_set_hash,
        )

    def persist_runtime_conflicts(
        self,
        run_id: str,
        conflicts: list[RuntimeConflictDTO],
        context: RuntimeContextDTO,
    ) -> RuntimeConflictSetDTO:
        self._get_scoped_run(run_id, context)
        conflict_set_hash = content_hash([conflict.__dict__ for conflict in conflicts])
        self._run_store.persist_runtime_conflicts(
            run_id=run_id,
            runtime_conflicts=[
                self._conflict_dto_to_store_payload(conflict)
                for conflict in conflicts
            ],
            runtime_conflict_set_hash=conflict_set_hash,
        )
        return RuntimeConflictSetDTO(
            run_id=run_id,
            conflict_count=len(conflicts),
            conflict_set_hash=conflict_set_hash,
        )

    def finalize_detection_run(
        self,
        run_id: str,
        finding_set_hash: str,
        conflict_set_hash: str,
        status: str,
        context: RuntimeContextDTO,
    ) -> DetectionRunDTO:
        self._get_scoped_run(run_id, context)
        run = self._run_store.finalize_run(
            run_id=run_id,
            finding_set_hash=finding_set_hash,
            runtime_conflict_set_hash=conflict_set_hash,
            status=status,
        )
        return DetectionRunDTO(
            run_id=run_id,
            status=status,
            content_hash=str(run.get("content_hash") or ""),
            metadata={"run": run},
        )

    def _get_scoped_run(
        self,
        run_id: str,
        context: RuntimeContextDTO,
    ) -> dict[str, Any]:
        run = self._run_store.get_run(run_id)
        if run is None:
            raise ValueError("DETECTION_RUN_NOT_FOUND")
        if run.get("organization_id") != context.organization_id:
            raise ValueError("DETECTION_RUN_NOT_FOUND")
        if (run.get("workspace_id") or None) != (context.workspace_id or None):
            raise ValueError("DETECTION_RUN_NOT_FOUND")
        return run

    @staticmethod
    def _finding_dto_to_store_payload(finding: OccurrenceFindingDTO) -> dict[str, Any]:
        metadata = dict(finding.metadata)
        original = dict(metadata.pop("store_payload", {}) or {})
        original.update(
            {
                "finding_id": finding.finding_id,
                "triggered_rule_version_id": finding.rule_version_id,
                "target_anchor_ref": finding.target_anchor_ref,
                "start_offset": finding.start_offset,
                "end_offset": finding.end_offset,
                "occurrence_index": finding.occurrence_index,
                "target_element_hash": finding.element_hash,
            }
        )
        return original

    @staticmethod
    def _conflict_dto_to_store_payload(conflict: RuntimeConflictDTO) -> dict[str, Any]:
        metadata = dict(conflict.metadata)
        original = dict(metadata.pop("store_payload", {}) or {})
        original.update(
            {
                "conflict_id": conflict.conflict_id,
                "conflict_type": conflict.conflict_type,
                "target_anchor_ref": conflict.target_anchor_ref,
                "rule_version_ids": list(conflict.rule_version_ids),
                "start_offset": conflict.start_offset,
                "end_offset": conflict.end_offset,
                "content_hash": conflict.content_hash,
            }
        )
        return original

    def get_detection_evidence(
        self,
        run_id: str,
        context: RuntimeContextDTO,
    ) -> DetectionEvidenceDTO:
        run = self._get_scoped_run(run_id, context)

        findings = tuple(
            OccurrenceFindingDTO(
                finding_id=str(finding["finding_id"]),
                rule_version_id=str(
                    finding.get("winning_rule_version_ref")
                    or finding.get("triggered_rule_version_id")
                    or ""
                ),
                target_anchor_ref=str(finding.get("target_anchor_ref") or ""),
                start_offset=int(finding.get("start_offset") or 0),
                end_offset=int(finding.get("end_offset") or 0),
                occurrence_index=int(finding.get("occurrence_index") or 0),
                element_hash=str(finding.get("target_element_hash") or ""),
                content_hash=str(finding.get("content_hash") or ""),
                metadata={
                    "issue_type": finding.get("issue_type"),
                    "matched_text": finding.get("matched_text"),
                    "source_anchor_refs": finding.get("source_anchor_refs") or [],
                    "triggered_rule_version_refs": finding.get("triggered_rule_version_refs") or [],
                    "conflicting_rule_version_refs": finding.get("conflicting_rule_version_refs") or [],
                    "conflict_resolution_ref": finding.get("conflict_resolution_ref"),
                    "escalation_flag": bool(finding.get("escalation_flag")),
                },
            )
            for finding in run.get("findings", [])
        )
        runtime_conflicts_payload = run.get("runtime_conflicts") or []
        runtime_conflicts = tuple(
            RuntimeConflictDTO(
                conflict_id=str(
                    conflict.get("conflict_id")
                    or content_hash(canonical_json(conflict))
                ),
                conflict_type=str(conflict.get("conflict_type") or ""),
                target_anchor_ref=str(conflict.get("target_anchor_ref") or ""),
                rule_version_ids=tuple(str(item) for item in conflict.get("rule_version_ids") or ()),
                start_offset=int(conflict.get("start_offset") or 0),
                end_offset=int(conflict.get("end_offset") or 0),
                content_hash=str(conflict.get("content_hash") or content_hash(canonical_json(conflict))),
                metadata={key: value for key, value in (conflict.get("conflict_payload") or conflict).items() if key not in {
                    "conflict_id",
                    "conflict_type",
                    "target_anchor_ref",
                    "rule_version_ids",
                    "start_offset",
                    "end_offset",
                    "content_hash",
                }},
            )
            for conflict in runtime_conflicts_payload
            if isinstance(conflict, dict)
        )
        return DetectionEvidenceDTO(
            run=DetectionRunDTO(
                run_id=str(run["run_id"]),
                status="created",
                content_hash=str(run.get("content_hash") or ""),
                metadata={
                    "tender_document_id": run.get("tender_document_id"),
                    "source_document_hash": run.get("source_document_hash"),
                    "activation_set_snapshot_id": run.get("activation_set_snapshot_id"),
                    "resolved_activation_set_hash": run.get("resolved_activation_set_hash"),
                    "detection_replay_binding": run.get("detection_replay_binding"),
                    "project_metadata": run.get("project_metadata") or {},
                },
            ),
            findings=findings,
            runtime_conflicts=runtime_conflicts,
        )
