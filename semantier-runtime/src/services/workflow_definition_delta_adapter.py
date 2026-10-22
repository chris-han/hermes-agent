"""Read-only delta adapter over owner-produced WorkflowDefinition artifacts.

The adapter consumes hashes/deltas produced by the canonical TypeScript workflow
owner. It does not reimplement WorkflowDefinitionIR hashing, BPMN compilation,
or workflow semantic diffing.
"""

from __future__ import annotations

from typing import Any

from contracts.governed_delta import (
    ArtifactVersionPinV1,
    DeltaLens,
    DeltaLensResultV1,
    DeltaLensStatus,
    DeltaMateriality,
    DeltaOperationKind,
    DeltaOperationV1,
    DeltaTargetType,
)
from eos.artifact_hashing import content_hash
from services.governed_delta_service import GovernedDeltaError, GovernedDeltaService


_STRUCTURAL_KINDS = {
    "ADD_ACTIVITY",
    "REMOVE_ACTIVITY",
    "ADD_SEQUENCE_FLOW",
    "REMOVE_SEQUENCE_FLOW",
    "CHANGE_SEQUENCE_FLOW",
    "ADD_MESSAGE_FLOW",
    "REMOVE_MESSAGE_FLOW",
    "CHANGE_MESSAGE_FLOW",
    "ADD_PARTICIPANT",
    "REMOVE_PARTICIPANT",
    "ADD_RESPONSIBILITY_PARTITION",
    "REMOVE_RESPONSIBILITY_PARTITION",
    "WRAP_SUBPROCESS",
    "UNWRAP_SUBPROCESS",
}

_SEMANTIC_KINDS = {
    "CHANGE_ACTIVITY_KIND",
    "CHANGE_ACTIVITY_CONFIG",
    "CHANGE_RESPONSIBILITY",
    "CHANGE_ACTOR_KIND",
    "CHANGE_GATEWAY_KIND",
    "CHANGE_EVENT_KIND",
}

_AUTHORITY_KEYS = {
    "authorityRef",
    "authorityClassRef",
}


def _require_sha256(value: Any, field: str) -> str:
    raw = str(value or "").strip()
    if len(raw) == 71 and raw.startswith("sha256:") and all(
        ch in "0123456789abcdef" for ch in raw[7:]
    ):
        return raw
    raise GovernedDeltaError(f"INVALID_{field.upper()}")


def _snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    required = {
        "schemaVersion",
        "workflowId",
        "workflowVersion",
        "ownerRef",
        "topology",
        "nodes",
        "edges",
        "contentHash",
    }
    missing = sorted(required - set(snapshot))
    if missing:
        raise GovernedDeltaError("WORKFLOW_SNAPSHOT_FIELDS_MISSING:" + ",".join(missing))
    if snapshot["schemaVersion"] != "workflow_definition.v1":
        raise GovernedDeltaError("INVALID_WORKFLOW_SCHEMA")
    if snapshot["topology"] != "dag":
        raise GovernedDeltaError("INVALID_WORKFLOW_TOPOLOGY")
    _require_sha256(snapshot["contentHash"], "workflow_content_hash")
    for field in ("workflowId", "workflowVersion", "ownerRef"):
        value = str(snapshot[field] or "")
        if not value or not value.isascii():
            raise GovernedDeltaError(f"INVALID_WORKFLOW_{field.upper()}")
    return snapshot


def _authority_refs(value: Any) -> set[str]:
    refs: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key in _AUTHORITY_KEYS and isinstance(child, str) and child:
                refs.add(child)
            refs.update(_authority_refs(child))
    elif isinstance(value, list):
        for child in value:
            refs.update(_authority_refs(child))
    return refs


def _op_kind(delta_kind: str) -> DeltaOperationKind:
    if delta_kind.startswith("ADD_"):
        return DeltaOperationKind.ADD
    if delta_kind.startswith("REMOVE_"):
        return DeltaOperationKind.REMOVE
    if delta_kind in {"WRAP_SUBPROCESS", "UNWRAP_SUBPROCESS"}:
        return DeltaOperationKind.MODIFY
    return DeltaOperationKind.MODIFY


def _delta_operation(
    *,
    delta: dict[str, Any],
    lens: DeltaLens,
    materiality: DeltaMateriality = DeltaMateriality.MATERIAL,
) -> DeltaOperationV1:
    delta_id = str(delta.get("deltaId") or "")
    subject = str(delta.get("subjectRef") or "")
    kind = str(delta.get("kind") or "")
    if not delta_id or not delta_id.isascii():
        raise GovernedDeltaError("INVALID_WORKFLOW_DELTA_ID")
    if not subject or not subject.isascii():
        raise GovernedDeltaError("INVALID_WORKFLOW_DELTA_SUBJECT")
    before = delta.get("before")
    after = delta.get("after")
    changed_fields = tuple(
        sorted(
            {
                str(key)
                for value in (before, after)
                if isinstance(value, dict)
                for key in value
            }
        )
    )
    return DeltaOperationV1(
        operation_id=f"workflow:{lens.value.lower()}:{delta_id}",
        operation_kind=_op_kind(kind),
        lens=lens,
        subject_ref=subject,
        before_ref=(
            f"before:sha256:{content_hash(before)}" if before is not None else None
        ),
        after_ref=f"after:sha256:{content_hash(after)}" if after is not None else None,
        changed_fields=changed_fields,
        evidence_refs=(),
        materiality=materiality,
    )


class WorkflowDefinitionDeltaAdapter:
    """Map owner-produced workflow snapshots/diffs into shared delta lenses."""

    target_type = DeltaTargetType.WORKFLOW

    def normalize_version(
        self,
        source: dict[str, Any],
    ) -> tuple[ArtifactVersionPinV1, dict[str, Any]]:
        snapshot = _snapshot(source)
        pin = ArtifactVersionPinV1(
            artifact_type="workflow_definition",
            artifact_id=str(snapshot["workflowId"]),
            version_id=str(snapshot["workflowVersion"]),
            content_hash=str(snapshot["contentHash"]),
            owner_ref=str(snapshot["ownerRef"]),
        )
        return pin, snapshot

    def validate_pin(
        self,
        snapshot: dict[str, Any],
        pin: ArtifactVersionPinV1,
    ) -> None:
        expected, _ = self.normalize_version(snapshot)
        if pin != expected:
            raise GovernedDeltaError("WORKFLOW_OWNER_PIN_MISMATCH")

    def compute_lenses(
        self,
        *,
        predecessor: dict[str, Any],
        successor: dict[str, Any],
        semantic_deltas: list[dict[str, Any]],
        layout_diff: dict[str, Any] | None = None,
        simulation_evidence: list[dict[str, Any]] | None = None,
        running_instance_version: str | None = None,
    ) -> dict[DeltaLens, DeltaLensResultV1]:
        before_pin, before = self.normalize_version(predecessor)
        after_pin, after = self.normalize_version(successor)
        if before_pin.artifact_id != after_pin.artifact_id:
            raise GovernedDeltaError("WORKFLOW_ID_MISMATCH")

        structural_ops: list[DeltaOperationV1] = []
        semantic_ops: list[DeltaOperationV1] = []
        authority_ops: list[DeltaOperationV1] = []

        for delta in semantic_deltas:
            kind = str(delta.get("kind") or "")
            if kind in _STRUCTURAL_KINDS:
                structural_ops.append(_delta_operation(delta=delta, lens=DeltaLens.STRUCTURAL))
            if kind in _SEMANTIC_KINDS or kind in _STRUCTURAL_KINDS:
                semantic_ops.append(_delta_operation(delta=delta, lens=DeltaLens.SEMANTIC))

            before_authority = _authority_refs(delta.get("before"))
            after_authority = _authority_refs(delta.get("after"))
            if before_authority != after_authority:
                authority_ops.append(
                    _delta_operation(delta=delta, lens=DeltaLens.AUTHORITY)
                )

        basis_refs = tuple(
            sorted(
                (
                    f"workflow-snapshot:{before_pin.version_id}:{before_pin.content_hash}",
                    f"workflow-snapshot:{after_pin.version_id}:{after_pin.content_hash}",
                )
            )
        )

        behavior_evidence = simulation_evidence or []
        behavior_ops: tuple[DeltaOperationV1, ...] = ()
        behavior_status = DeltaLensStatus.UNAVAILABLE
        behavior_basis: tuple[str, ...] = ()
        if behavior_evidence:
            receipt_refs = tuple(
                sorted(
                    {
                        str(item.get("receiptRef") or "")
                        for item in behavior_evidence
                        if str(item.get("receiptRef") or "")
                    }
                )
            )
            if not receipt_refs:
                raise GovernedDeltaError("INVALID_WORKFLOW_SIMULATION_EVIDENCE")
            behavior_status = DeltaLensStatus.AVAILABLE
            behavior_basis = receipt_refs
            behavior_ops = (
                DeltaOperationV1(
                    operation_id=(
                        "workflow:behavior:"
                        + content_hash([item for item in behavior_evidence])[:24]
                    ),
                    operation_kind=DeltaOperationKind.MODIFY,
                    lens=DeltaLens.BEHAVIOR,
                    subject_ref=str(after["workflowId"]),
                    changed_fields=("simulationEvidence",),
                    evidence_refs=receipt_refs,
                    materiality=DeltaMateriality.UNKNOWN,
                ),
            )

        layout = layout_diff or {}
        moved = tuple(sorted(str(item) for item in layout.get("nodesMoved", [])))
        viewport_changed = bool(layout.get("viewportChanged", False))
        presentation_ops: tuple[DeltaOperationV1, ...] = ()
        if moved or viewport_changed:
            presentation_ops = (
                DeltaOperationV1(
                    operation_id=(
                        "workflow:presentation:"
                        + content_hash(
                            {"nodesMoved": moved, "viewportChanged": viewport_changed}
                        )[:24]
                    ),
                    operation_kind=DeltaOperationKind.MOVE,
                    lens=DeltaLens.PRESENTATION,
                    subject_ref=str(after["workflowId"]),
                    changed_fields=tuple(
                        sorted(
                            (
                                ("nodesMoved",) if moved else ()
                            )
                            + (("viewportChanged",) if viewport_changed else ())
                        )
                    ),
                    materiality=DeltaMateriality.NON_MATERIAL,
                ),
            )

        evidence_refs = tuple(
            sorted(
                {
                    str(item.get("receiptRef") or "")
                    for item in behavior_evidence
                    if str(item.get("receiptRef") or "")
                }
            )
        )

        binding_from = running_instance_version or before_pin.version_id
        binding_ops: tuple[DeltaOperationV1, ...] = ()
        if binding_from != after_pin.version_id:
            binding_ops = (
                DeltaOperationV1(
                    operation_id=(
                        "workflow:binding:"
                        + content_hash(
                            {
                                "from": binding_from,
                                "to": after_pin.version_id,
                                "workflow": after_pin.artifact_id,
                            }
                        )[:24]
                    ),
                    operation_kind=DeltaOperationKind.REBIND,
                    lens=DeltaLens.EXECUTION_BINDING,
                    subject_ref=f"workflow-binding:{after_pin.artifact_id}",
                    before_ref=f"workflow-version:{binding_from}",
                    after_ref=f"workflow-version:{after_pin.version_id}",
                    changed_fields=("workflowVersion",),
                    materiality=DeltaMateriality.MATERIAL,
                ),
            )

        return {
            DeltaLens.STRUCTURAL: DeltaLensResultV1(
                lens=DeltaLens.STRUCTURAL,
                status=DeltaLensStatus.AVAILABLE,
                operations=tuple(sorted(structural_ops, key=lambda item: item.operation_id)),
                basis_refs=basis_refs,
            ),
            DeltaLens.SEMANTIC: DeltaLensResultV1(
                lens=DeltaLens.SEMANTIC,
                status=DeltaLensStatus.AVAILABLE,
                operations=tuple(sorted(semantic_ops, key=lambda item: item.operation_id)),
                basis_refs=basis_refs,
            ),
            DeltaLens.AUTHORITY: DeltaLensResultV1(
                lens=DeltaLens.AUTHORITY,
                status=(
                    DeltaLensStatus.AVAILABLE
                    if authority_ops
                    else DeltaLensStatus.UNAVAILABLE
                ),
                operations=tuple(sorted(authority_ops, key=lambda item: item.operation_id)),
                basis_refs=basis_refs if authority_ops else (),
            ),
            DeltaLens.BEHAVIOR: DeltaLensResultV1(
                lens=DeltaLens.BEHAVIOR,
                status=behavior_status,
                operations=behavior_ops,
                basis_refs=behavior_basis,
            ),
            DeltaLens.EVIDENCE: DeltaLensResultV1(
                lens=DeltaLens.EVIDENCE,
                status=DeltaLensStatus.AVAILABLE,
                operations=(),
                basis_refs=evidence_refs,
            ),
            DeltaLens.EXECUTION_BINDING: DeltaLensResultV1(
                lens=DeltaLens.EXECUTION_BINDING,
                status=DeltaLensStatus.AVAILABLE,
                operations=binding_ops,
                basis_refs=basis_refs,
            ),
            DeltaLens.PRESENTATION: DeltaLensResultV1(
                lens=DeltaLens.PRESENTATION,
                status=DeltaLensStatus.AVAILABLE,
                operations=presentation_ops,
                basis_refs=(),
            ),
        }

    def build_envelope(
        self,
        *,
        delta_id: str,
        predecessor: dict[str, Any],
        successor: dict[str, Any],
        semantic_deltas: list[dict[str, Any]],
        layout_diff: dict[str, Any] | None = None,
        simulation_evidence: list[dict[str, Any]] | None = None,
        running_instance_version: str | None = None,
    ):
        predecessor_pin, _ = self.normalize_version(predecessor)
        successor_pin, _ = self.normalize_version(successor)
        return GovernedDeltaService.build_envelope(
            delta_id=delta_id,
            target_type=self.target_type,
            predecessor=predecessor_pin,
            successor_candidate=successor_pin,
            lens_results=self.compute_lenses(
                predecessor=predecessor,
                successor=successor,
                semantic_deltas=semantic_deltas,
                layout_diff=layout_diff,
                simulation_evidence=simulation_evidence,
                running_instance_version=running_instance_version,
            ),
        )


__all__ = ["WorkflowDefinitionDeltaAdapter"]
