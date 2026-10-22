"""Pure deterministic projection of an already verified runtime closure."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import platform
import sys

from contracts.ontology_building import ArtifactPinV1
from eos.artifact_hashing import content_hash
from contracts.runtime_inspection_report import (
    RuntimeInspectionReportV1,
    RuntimeReportFindingV1,
    RuntimeReportHistoryV1,
    RuntimeReportReplayInputsV1,
    RuntimeReportReviewV1,
    RuntimeReportSubjectV1,
    RuntimeReportSummaryV1,
    RuntimeReportUnresolvedV1,
    VerifiedRuntimeReportInputsV1,
)


def _pin_key(pin: ArtifactPinV1) -> tuple[str, str, str]:
    return pin.artifact_kind, pin.artifact_ref, pin.artifact_hash


def _sorted_pins(pins: tuple[ArtifactPinV1, ...]) -> tuple[ArtifactPinV1, ...]:
    return tuple(sorted(pins, key=_pin_key))


def server_runtime_report_profile_pins() -> tuple[ArtifactPinV1, ArtifactPinV1, ArtifactPinV1, ArtifactPinV1]:
    """Return the exact local report-profile identities advertised to Workbench2."""
    projector_hash = "sha256:" + sha256(Path(__file__).read_bytes()).hexdigest()
    descriptors = (
        ("runtime_report_renderer", "runtime-report-renderer:structured:v1", {"renderer": "structured", "version": 1}),
        ("runtime_report_template", "runtime-report-template:inspection:v1", {"template": "inspection", "version": 1}),
        (
            "runtime_report_environment",
            "runtime-report-environment:local:v1",
            {
                "python": platform.python_version(),
                "implementation": platform.python_implementation(),
                "system": platform.system(),
                "machine": platform.machine(),
                "byteorder": sys.byteorder,
            },
        ),
    )
    return (
        ArtifactPinV1(
            artifact_kind="runtime_report_projector",
            artifact_ref="runtime-inspection-report-projector:v1",
            artifact_hash=projector_hash,
        ),
        *tuple(
            ArtifactPinV1(
                artifact_kind=kind,
                artifact_ref=ref,
                artifact_hash="sha256:" + content_hash(payload),
            )
            for kind, ref, payload in descriptors
        ),
    )


class RuntimeInspectionReportService:
    """Build reports without clocks, persistence, policy resolution or evaluation."""

    @staticmethod
    def project(
        *,
        inputs: VerifiedRuntimeReportInputsV1,
        projector_pin: ArtifactPinV1,
        renderer_pin: ArtifactPinV1,
        template_pin: ArtifactPinV1,
        environment_pin: ArtifactPinV1,
        predecessor_pin: ArtifactPinV1 | None = None,
    ) -> tuple[RuntimeInspectionReportV1, RuntimeReportReplayInputsV1]:
        predecessor = predecessor_pin or inputs.predecessor_report_pin
        if inputs.disposition_pins and predecessor is None:
            raise ValueError("changed disposition requires an explicit predecessor report")

        review = RuntimeReportReviewV1(
            status=inputs.review_status,
            disposition_pins=_sorted_pins(inputs.disposition_pins),
            final_user_decision=inputs.final_user_decision,
            review_receipt_pin=inputs.review_receipt_pin,
            rationale=inputs.review_rationale,
            remediation_provenance=inputs.remediation_provenance,
            remediation_artifact_pin=inputs.remediation_artifact_pin,
        )
        unresolved = tuple(
            RuntimeReportUnresolvedV1(
                result_pin=pin,
                finding_pin=None,
                missing_evidence_types=(),
                missing_inputs=("verified_resolution_basis",),
                competing_interpretations=("The retained result has competing interpretations",),
                indeterminate_rule_pins=(),
                absent_rule_reason="No applicable rule interpretation was retained",
                resolution_requirements=inputs.resolution_requirements,
            )
            for pin in inputs.unknown_results
        )
        finding_pins = tuple(item.finding_pin for item in inputs.findings if item.finding_pin)
        result_states = [item.outcome for item in inputs.findings]
        unknown_count = len(inputs.unknown_results) + result_states.count("UNKNOWN")
        summary = RuntimeReportSummaryV1(
            status=inputs.status,
            ordered_result_pins=inputs.result_pins,
            ordered_finding_pins=finding_pins,
            evaluated_competency_pins=_sorted_pins(inputs.evaluated_competency_pins),
            material_count=len(inputs.findings),
            resolved_count=max(0, len(inputs.findings) - unknown_count),
            unresolved_count=len(unresolved),
            unknown_count=unknown_count,
            review_pending_count=0 if inputs.review_status != "NEEDS_REVIEW" else 1,
            denominator_definition="one count per retained runtime result; UNKNOWN remains machine-visible",
        )
        subject = RuntimeReportSubjectV1(
            target_pin=inputs.target_pin,
            source_pin=inputs.source_pin,
            document_title=inputs.document_title,
            run_pin=inputs.run_pin,
            root_run_pin=inputs.root_run_pin,
            parent_run_pin=inputs.parent_run_pin,
            epoch_pin=inputs.epoch_pin,
            certified_plan_pin=inputs.certified_plan_pin,
            provider_identity_pin=inputs.provider_identity_pin,
            graph_pin=inputs.graph_pin,
            release_pin=inputs.release_pin,
            normative_ir_pin=inputs.normative_ir_pin,
            resolver_pin=inputs.resolver_pin,
            rule_version_pins=_sorted_pins(inputs.rule_version_pins),
            evaluator_pin=inputs.evaluator_pin,
            context_pin=inputs.context_pin,
            world_pin=inputs.world_pin,
            transition_pin=inputs.transition_pin,
            business_effective_at=inputs.business_effective_at,
            recorded_at=inputs.recorded_at,
        )
        replay = RuntimeReportReplayInputsV1.create(
            organization_id=inputs.organization_id,
            workspace_id=inputs.workspace_id,
            source_pins=_sorted_pins((inputs.target_pin, inputs.source_pin)),
            closure_manifest=_sorted_pins(
                tuple(inputs.result_pins)
                + tuple(inputs.rule_version_pins)
                + tuple(inputs.learning_pins)
                + tuple(inputs.attribution_pins)
            ),
            renderer_pin=renderer_pin,
            template_pin=template_pin,
            environment_pin=environment_pin,
            access_policy_pin=None,
            supported_projections=("MEMBER", "AUDIT"),
        )
        history = RuntimeReportHistoryV1(
            modification_events=inputs.modification_events,
            predecessor_report_pin=predecessor,
            successor_link_receipt_pins=_sorted_pins(inputs.successor_link_receipt_pins),
            root_run_pin=inputs.root_run_pin,
            parent_run_pin=inputs.parent_run_pin,
            learning_pins=_sorted_pins(inputs.learning_pins),
            attribution_pins=_sorted_pins(inputs.attribution_pins),
            handoff_pins=_sorted_pins(inputs.handoff_pins),
        )
        report = RuntimeInspectionReportV1.create(
            organization_id=inputs.organization_id,
            workspace_id=inputs.workspace_id,
            subject=subject,
            summary=summary,
            findings=inputs.findings,
            unresolved=unresolved,
            review=review,
            history=history,
            replay_access=replay,
            projector_pin=projector_pin,
            renderer_pin=renderer_pin,
            template_pin=template_pin,
            environment_pin=environment_pin,
        )
        return report, replay


__all__ = ["RuntimeInspectionReportService", "server_runtime_report_profile_pins"]
