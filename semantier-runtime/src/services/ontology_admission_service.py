"""Server-owned ontology admission readiness and atomic A1 admission."""

from __future__ import annotations

from datetime import datetime

from contracts.ontology_admission import (
    AdmitOntologyCommandV1,
    CertifiedSemanticExecutionPlanV1,
    OntologyAdmissionDecisionV1,
    OntologyReleaseBundleV1,
)
from contracts.ontology_building import ArtifactPinV1, OntologyBuildManifestV1, OntologyOperation
from contracts.ontology_quality import QualityEvaluationRunV1, QualityProfileV1
from eos.artifact_hashing import content_hash
from storage.ontology_build_store import OntologyBuildStore


class OntologyAdmissionService:
    def admit_persisted(
        self, *, store: OntologyBuildStore, organization_id: str, workspace_id: str,
        command: AdmitOntologyCommandV1, actor_authorized: bool, decided_at: datetime,
    ) -> tuple[OntologyAdmissionDecisionV1, OntologyReleaseBundleV1]:
        """Publish one exact admission and recover it after an interrupted response."""
        scope = dict(organization_id=organization_id, workspace_id=workspace_id)
        manifest = store.get(**scope, artifact_ref=command.build_ref, contract_type=OntologyBuildManifestV1)
        if manifest is None:
            raise KeyError("Ontology build not found")
        operation = dict(**scope, actor_ref=command.actor_ref, operation="ADMIT_ONTOLOGY",
            request_id=command.build_ref,
            request_hash="sha256:" + content_hash({"command": command.model_dump(mode="json"), "build_hash": manifest.build_hash}))
        prior = store.get_operation_result(**operation, artifact_kind="ontology_release_bundle")
        if prior is None:
            def child(kind, contract, hash_field):
                pins = [pin for pin in manifest.child_artifacts if pin.artifact_kind == kind]
                if len(pins) != 1:
                    raise ValueError("admission requires one pinned " + kind)
                artifact = store.get(**scope, artifact_ref=pins[0].artifact_ref, contract_type=contract)
                if artifact is None or getattr(artifact, hash_field) != pins[0].artifact_hash:
                    raise ValueError("admission child pin mismatch: " + kind)
                return artifact

            decision, bundle = self.admit(
                command=command, manifest=manifest,
                quality_evaluation=child("quality_evaluation_run", QualityEvaluationRunV1, "run_hash"),
                quality_profile=child("quality_profile", QualityProfileV1, "profile_hash"),
                execution_plan=child("certified_semantic_execution_plan", CertifiedSemanticExecutionPlanV1, "execution_plan_hash"),
                actor_authorized=actor_authorized, decided_at=decided_at, store=store,
            )
            # Existing managed admissions may predate the operation index.
            # Adopt only the exact deterministic result, preserving its time.
            existing_decision = store.get(**scope, artifact_ref=decision.decision_ref,
                contract_type=OntologyAdmissionDecisionV1)
            existing_bundle = store.get(**scope, artifact_ref=bundle.release_bundle_ref,
                contract_type=OntologyReleaseBundleV1)
            if existing_decision is not None or existing_bundle is not None:
                if (existing_decision is None or existing_bundle is None
                    or existing_decision.model_dump(exclude={"decided_at"}) != decision.model_dump(exclude={"decided_at"})
                    or existing_bundle.model_dump(exclude={"created_at"}) != bundle.model_dump(exclude={"created_at"})):
                    raise ValueError("admission existing result mismatch")
                decision, bundle = existing_decision, existing_bundle
            prior = store.put_many_with_operation_result(**operation, artifacts=(decision, bundle),
                result_pin=ArtifactPinV1(artifact_kind="ontology_release_bundle",
                    artifact_ref=bundle.release_bundle_ref, artifact_hash=bundle.release_bundle_hash))
        bundle = store.get(**scope, artifact_ref=prior.artifact_ref, contract_type=OntologyReleaseBundleV1)
        if bundle is None or bundle.release_bundle_hash != prior.artifact_hash:
            raise ValueError("admission release receipt mismatch")
        decision = store.get(**scope, artifact_ref=bundle.admission_decision_ref, contract_type=OntologyAdmissionDecisionV1)
        if decision is None or (
            decision.decision_hash != bundle.admission_decision_hash
            or (decision.build_ref, decision.build_hash) != (manifest.build_ref, manifest.build_hash)
            or (bundle.build_ref, bundle.build_hash) != (manifest.build_ref, manifest.build_hash)
            or decision.actor_ref != command.actor_ref
            or (decision.execution_plan_ref, decision.execution_plan_hash) != (bundle.execution_plan_ref, bundle.execution_plan_hash)
        ):
            raise ValueError("admission decision receipt mismatch")
        # A cached release still needs its retained successor authorization chain.
        profiles = [p for p in manifest.child_artifacts if p.artifact_kind == 'quality_profile']
        if len(profiles) != 1:
            raise ValueError('admission requires one pinned quality_profile')
        profile = store.get(**scope, artifact_ref=profiles[0].artifact_ref, contract_type=QualityProfileV1)
        if profile is None or profile.pin() != profiles[0]:
            raise ValueError('admission retained profile mismatch')
        if profile.applicability_review is not None:
            runs = [p for p in manifest.child_artifacts if p.artifact_kind == 'quality_evaluation_run']
            if len(runs) != 1:
                raise ValueError('admission requires one pinned quality_evaluation_run')
            run = store.get(**scope, artifact_ref=runs[0].artifact_ref, contract_type=QualityEvaluationRunV1)
            if run is None or run.pin() != runs[0]:
                raise ValueError('admission retained quality run mismatch')
            from services.ontology_quality_profile_service import QualityProfileService
            QualityProfileService.verify_consumer_basis(store=store, profile=profile, run=run)
        return decision, bundle

    def admit(
        self,
        *,
        command: AdmitOntologyCommandV1,
        manifest: OntologyBuildManifestV1,
        quality_evaluation: QualityEvaluationRunV1,
        quality_profile: QualityProfileV1,
        execution_plan: CertifiedSemanticExecutionPlanV1,
        actor_authorized: bool,
        decided_at: datetime,
        store=None,
    ) -> tuple[OntologyAdmissionDecisionV1, OntologyReleaseBundleV1]:
        if command.build_ref != manifest.build_ref:
            raise ValueError("admission command does not name the resolved build")
        if manifest.assurance_profile != execution_plan.assurance_profile:
            raise ValueError("admission assurance profile mismatch")
        quality_evaluation = QualityEvaluationRunV1.model_validate(quality_evaluation)
        quality_profile = QualityProfileV1.model_validate(quality_profile)
        from services.ontology_quality_profile_service import QualityProfileService
        QualityProfileService.verify_consumer_basis(store=store, profile=quality_profile, run=quality_evaluation)
        tenants = {
            (manifest.organization_id, manifest.workspace_id),
            (quality_evaluation.organization_id, quality_evaluation.workspace_id),
            (quality_profile.organization_id, quality_profile.workspace_id),
            (execution_plan.organization_id, execution_plan.workspace_id),
        }
        if len(tenants) != 1:
            raise PermissionError("admission inputs cross governed tenant scope")
        if (execution_plan.quality_evaluation != quality_evaluation.pin()
            or execution_plan.quality_profile != quality_profile.pin()
            or quality_evaluation.quality_profile != quality_profile.pin()
            or quality_evaluation.validation != ArtifactPinV1(artifact_kind="validation_contract",
                artifact_ref=execution_plan.validation_contract_ref, artifact_hash=execution_plan.validation_contract_hash)):
            raise ValueError("admission execution-plan quality pins do not match")
        if quality_evaluation.subject != ArtifactPinV1(artifact_kind="normative_semantic_candidate_set",
            artifact_ref=execution_plan.canonical_normative_ir_ref, artifact_hash=execution_plan.canonical_normative_ir_hash):
            raise ValueError("admission execution-plan candidate pins do not match quality evaluation")
        for pin in (quality_profile.pin(), quality_evaluation.pin()):
            if tuple(item for item in manifest.child_artifacts if item.artifact_kind == pin.artifact_kind) != (pin,):
                raise ValueError("admission manifest quality pins do not match")
        admit_gate = next(
            (gate for gate in manifest.gates if gate.operation is OntologyOperation.ADMIT),
            None,
        )
        if admit_gate is None or admit_gate.decision != "PROCEED" or admit_gate.blocker_refs:
            raise ValueError("admission gate is not ready")
        if quality_evaluation.disposition not in {"QUALIFIED", "QUALIFIED_WITH_WARNINGS"}:
            raise ValueError("admission requires qualified quality evaluation")
        if quality_evaluation.disposition == "QUALIFIED_WITH_WARNINGS" and not quality_profile.allow_warnings:
            raise ValueError("admission quality policy does not permit warnings")

        decision_basis = {
            "build_ref": manifest.build_ref,
            "build_hash": manifest.build_hash,
            "assurance_profile": manifest.assurance_profile,
            "assurance_policy_ref": execution_plan.assurance_policy_ref,
            "assurance_policy_hash": execution_plan.assurance_policy_hash,
            "quality_profile": quality_profile.pin().model_dump(mode="json"),
            "quality_evaluation": quality_evaluation.pin().model_dump(mode="json"),
            "execution_plan_ref": execution_plan.execution_plan_ref,
            "execution_plan_hash": execution_plan.execution_plan_hash,
            "actor_ref": command.actor_ref,
            "decision": "ADMIT",
        }
        decision_digest = content_hash(decision_basis)
        decision = OntologyAdmissionDecisionV1(
            decision_ref=f"ontology-admission:{decision_digest}",
            decision_hash=f"sha256:{decision_digest}",
            organization_id=manifest.organization_id,
            workspace_id=manifest.workspace_id,
            unsupported_scope=(),
            unresolved_items=(),
            prohibited_runtime_uses=(),
            decided_at=decided_at,
            **decision_basis,
        )
        bundle_basis = {
            "build_ref": manifest.build_ref,
            "build_hash": manifest.build_hash,
            "admission_decision_ref": decision.decision_ref,
            "admission_decision_hash": decision.decision_hash,
            "execution_plan_ref": execution_plan.execution_plan_ref,
            "execution_plan_hash": execution_plan.execution_plan_hash,
        }
        bundle_digest = content_hash(bundle_basis)
        bundle = OntologyReleaseBundleV1(
            release_bundle_ref=f"ontology-release:{bundle_digest}",
            release_bundle_hash=f"sha256:{bundle_digest}",
            organization_id=manifest.organization_id,
            workspace_id=manifest.workspace_id,
            governed_rule_artifact_refs=(),
            activation_state="NOT_ACTIVE",
            created_at=decided_at,
            **bundle_basis,
        )
        return decision, bundle


__all__ = ["OntologyAdmissionService"]
