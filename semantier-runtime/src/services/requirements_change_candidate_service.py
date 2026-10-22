"""Bounded requirements-domain consumer of ChangeCandidate v2 ordinary profiles."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from contracts.change_candidate_v2 import (
    AtomicChangeProfileV1,
    ChangeCandidateV2,
    ChangeSetMemberV1,
    ChangeSetProfileV1,
)
from contracts.governed_delta import (
    ArtifactVersionPinV1,
    DeltaEnvelopeV1,
    DeltaLens,
    DeltaLensResultV1,
    DeltaLensStatus,
    DeltaTargetType,
    ImpactProjectionV1,
)
from contracts.ontology_building import ArtifactPinV1
from contracts.release_activation_projection import (
    ActivationBindingProjectionV1,
    ActivationOwnerBindingV1,
    ReleaseSetManifestV1,
    ReleaseSetMemberV1,
)
from contracts.requirements_change import (
    RequirementActivationReceiptV1,
    RequirementAdmissionReceiptV1,
    RequirementBaselineV1,
    RequirementContextProjectionV1,
    RequirementQualificationReceiptV1,
)
from services.change_candidate_service import ChangeCandidateService
from services.governed_delta_service import GovernedDeltaService
from storage.ontology_build_store import OntologyBuildStore


@dataclass(frozen=True)
class RequirementPublicationResult:
    baseline: RequirementBaselineV1
    admission_receipt: RequirementAdmissionReceiptV1
    activation_receipt: RequirementActivationReceiptV1
    release_manifest: ReleaseSetManifestV1
    activation_projection: ActivationBindingProjectionV1


class RequirementsChangeCandidateService:
    """First bounded ordinary-change writer; owner authority remains requirements-native."""

    def __init__(
        self,
        *,
        store: OntologyBuildStore,
        organization_id: str,
        workspace_id: str,
        owner_ref: str,
        authority_domain_ref: str,
    ) -> None:
        self._store = store
        self.candidate_service = ChangeCandidateService(store=store)
        self._organization_id = organization_id
        self._workspace_id = workspace_id
        self._owner_ref = owner_ref
        self._authority_domain_ref = authority_domain_ref
        self._baselines: dict[tuple[str, str, str], RequirementBaselineV1] = {}
        self._qualifications: dict[tuple[str, str, str], RequirementQualificationReceiptV1] = {}
        self._impacts: dict[tuple[str, str, str], tuple[ImpactProjectionV1, ...]] = {}
        self._head: ArtifactPinV1 | None = None
        self._restore_owner_state()

    @staticmethod
    def _key(pin: ArtifactPinV1) -> tuple[str, str, str]:
        return (pin.artifact_kind, pin.artifact_ref, pin.artifact_hash)

    def _load_schema(self, schema_version: str, contract_type):
        values = []
        for indexed in self._store.list_artifact_pins(
            organization_id=self._organization_id,
            workspace_id=self._workspace_id,
            schema_version=schema_version,
        ):
            value = self._store.get(
                organization_id=self._organization_id,
                workspace_id=self._workspace_id,
                artifact_ref=indexed.artifact_ref,
                contract_type=contract_type,
            )
            if value is not None:
                values.append(value)
        return tuple(values)

    def _restore_owner_state(self) -> None:
        baselines = self._load_schema("requirement_baseline.v1", RequirementBaselineV1)
        for baseline in baselines:
            self._baselines[self._key(baseline.pin())] = baseline
        if baselines:
            predecessor_pins = {
                item.predecessor_baseline
                for item in baselines
                if item.predecessor_baseline is not None
            }
            leaves = tuple(item for item in baselines if item.pin() not in predecessor_pins)
            if len(leaves) != 1:
                raise ValueError("REQUIREMENT_BASELINE_HEAD_CONFLICT")
            self._head = leaves[0].pin()
        for receipt in self._load_schema(
            "requirement_change_qualification.v1",
            RequirementQualificationReceiptV1,
        ):
            self._qualifications[self._key(receipt.pin())] = receipt

    def seed_baseline(
        self,
        *,
        baseline_ref: str,
        member_versions: tuple[ArtifactVersionPinV1, ...],
        recorded_at: datetime,
    ) -> RequirementBaselineV1:
        if self._head is not None:
            raise ValueError("baseline head already initialized")
        baseline = RequirementBaselineV1.create(
            schema_version="requirement_baseline.v1",
            organization_id=self._organization_id,
            workspace_id=self._workspace_id,
            owner_ref=self._owner_ref,
            baseline_ref=baseline_ref,
            predecessor_baseline=None,
            member_versions=member_versions,
            recorded_at=recorded_at,
        )
        self._store.put(baseline)
        self._baselines[self._key(baseline.pin())] = baseline
        self._head = baseline.pin()
        return baseline

    def propose_atomic(
        self,
        *,
        delta: DeltaEnvelopeV1,
        rationale: str,
        basis_refs: tuple[ArtifactPinV1, ...],
        recorded_at: datetime,
    ) -> ChangeCandidateV2:
        candidate = ChangeCandidateV2.create(
            schema_version="semantier.change_candidate.v2",
            organization_id=self._organization_id,
            workspace_id=self._workspace_id,
            revision=0,
            predecessor=None,
            candidate_kind="ATOMIC_CHANGE",
            profile=AtomicChangeProfileV1(
                owner_ref=self._owner_ref,
                authority_domain_ref=self._authority_domain_ref,
                rationale=rationale,
                basis_refs=basis_refs,
                delta=delta,
            ),
            recorded_at=recorded_at,
        )
        self._store.put(candidate)
        return candidate

    def propose_change_set(
        self,
        *,
        deltas: tuple[DeltaEnvelopeV1, ...],
        rationale: str,
        basis_refs: tuple[ArtifactPinV1, ...],
        coherence_refs: tuple[ArtifactPinV1, ...],
        impact_projections: tuple[ImpactProjectionV1, ...],
        recorded_at: datetime,
    ) -> ChangeCandidateV2:
        members = tuple(
            ChangeSetMemberV1(member_id=delta.delta_id, delta=delta)
            for delta in deltas
        )
        candidate = ChangeCandidateV2.create(
            schema_version="semantier.change_candidate.v2",
            organization_id=self._organization_id,
            workspace_id=self._workspace_id,
            revision=0,
            predecessor=None,
            candidate_kind="CHANGE_SET",
            profile=ChangeSetProfileV1(
                owner_ref=self._owner_ref,
                authority_domain_ref=self._authority_domain_ref,
                rationale=rationale,
                basis_refs=basis_refs,
                coherence_refs=coherence_refs,
                members=members,
            ),
            recorded_at=recorded_at,
        )
        self._store.put(candidate)
        self._impacts[self._key(candidate.pin())] = tuple(impact_projections)
        return candidate

    def impact_projections(
        self, candidate: ArtifactPinV1
    ) -> tuple[ImpactProjectionV1, ...]:
        return self._impacts.get(self._key(candidate), ())

    def qualify(
        self,
        *,
        candidate: ArtifactPinV1,
        qualification_refs: tuple[ArtifactPinV1, ...],
        recorded_at: datetime,
    ) -> RequirementQualificationReceiptV1:
        value = self.candidate_service.get_versioned(pin=candidate)
        if not isinstance(value, ChangeCandidateV2) or value.candidate_kind not in {
            "ATOMIC_CHANGE",
            "CHANGE_SET",
        }:
            raise ValueError("requirements qualification requires ordinary ChangeCandidate v2")
        receipt = RequirementQualificationReceiptV1.create(
            schema_version="requirement_change_qualification.v1",
            organization_id=self._organization_id,
            workspace_id=self._workspace_id,
            owner_ref=self._owner_ref,
            candidate=candidate,
            qualification_refs=qualification_refs,
            disposition="QUALIFIED",
            recorded_at=recorded_at,
        )
        self._store.put(receipt)
        self._qualifications[self._key(receipt.pin())] = receipt
        return receipt

    def admit_and_publish(
        self,
        *,
        candidate: ArtifactPinV1,
        qualification: ArtifactPinV1,
        expected_head: ArtifactPinV1,
        successor_versions: tuple[ArtifactVersionPinV1, ...],
        baseline_ref: str,
        recorded_at: datetime,
        stale_disposition: str | None = None,
    ) -> RequirementPublicationResult | str:
        if self._head != expected_head:
            if stale_disposition not in {"REEVALUATE", "REBASE", "REFUSE"}:
                raise ValueError("STALE_BASELINE_HEAD")
            return stale_disposition
        qualification_value = self._qualifications.get(self._key(qualification))
        if qualification_value is None or qualification_value.candidate != candidate:
            raise ValueError("QUALIFICATION_MISMATCH")
        candidate_value = self.candidate_service.get_versioned(pin=candidate)
        if not isinstance(candidate_value, ChangeCandidateV2) or candidate_value.candidate_kind not in {
            "ATOMIC_CHANGE",
            "CHANGE_SET",
        }:
            raise ValueError("ordinary candidate required")
        baseline = RequirementBaselineV1.create(
            schema_version="requirement_baseline.v1",
            organization_id=self._organization_id,
            workspace_id=self._workspace_id,
            owner_ref=self._owner_ref,
            baseline_ref=baseline_ref,
            predecessor_baseline=expected_head,
            member_versions=successor_versions,
            recorded_at=recorded_at,
        )
        admission = RequirementAdmissionReceiptV1.create(
            schema_version="requirement_change_admission.v1",
            organization_id=self._organization_id,
            workspace_id=self._workspace_id,
            owner_ref=self._owner_ref,
            candidate=candidate,
            qualification=qualification,
            predecessor_baseline=expected_head,
            successor_baseline=baseline.pin(),
            disposition="ADMITTED",
            recorded_at=recorded_at,
        )
        activation = RequirementActivationReceiptV1.create(
            schema_version="requirement_baseline_activation.v1",
            organization_id=self._organization_id,
            workspace_id=self._workspace_id,
            owner_ref=self._owner_ref,
            baseline=baseline.pin(),
            admission=admission.pin(),
            effective_scope=baseline.pin(),
            recorded_at=recorded_at,
        )
        release = ReleaseSetManifestV1.create(
            schema_version="release_set_manifest.v1",
            organization_id=self._organization_id,
            workspace_id=self._workspace_id,
            release_set_ref=f"requirements-release:{baseline_ref}",
            parent_release_set=None,
            members=(
                ReleaseSetMemberV1(
                    owner_ref=self._owner_ref,
                    candidate=candidate,
                    admission_receipt=admission.pin(),
                    owner_release_receipt=baseline.pin(),
                    carried_forward_from=expected_head,
                ),
            ),
            compatibility_refs=(),
            effective_from=recorded_at,
            effective_to=None,
            published_at=recorded_at,
        )
        activation_projection = ActivationBindingProjectionV1.create(
            schema_version="activation_binding_projection.v1",
            organization_id=self._organization_id,
            workspace_id=self._workspace_id,
            release_set=release.pin(),
            owner_bindings=(
                ActivationOwnerBindingV1(
                    owner_ref=self._owner_ref,
                    activation_receipt=activation.pin(),
                    effective_scope=baseline.pin(),
                    runtime_binding_refs=(),
                    context_binding_refs=(),
                ),
            ),
            release_effective_from=recorded_at,
            release_effective_to=None,
            effective_from=recorded_at,
            effective_to=None,
            projected_at=recorded_at,
            predecessor_projection=None,
            projection_reason=None,
        )
        self._store.put_many(
            (baseline, admission, activation, release, activation_projection)
        )
        self._baselines[self._key(baseline.pin())] = baseline
        self._head = baseline.pin()
        return RequirementPublicationResult(
            baseline=baseline,
            admission_receipt=admission,
            activation_receipt=activation,
            release_manifest=release,
            activation_projection=activation_projection,
        )

    def current_baseline(self) -> RequirementBaselineV1:
        if self._head is None:
            raise ValueError("baseline head is not initialized")
        return self._baselines[self._key(self._head)]

    def get_baseline(self, pin: ArtifactPinV1) -> RequirementBaselineV1:
        try:
            return self._baselines[self._key(pin)]
        except KeyError as exc:
            raise ValueError("baseline not found") from exc

    def _get_persisted(self, pin: ArtifactPinV1, contract_type):
        value = self._store.get(
            organization_id=self._organization_id,
            workspace_id=self._workspace_id,
            artifact_ref=pin.artifact_ref,
            contract_type=contract_type,
        )
        if value is None or value.pin() != pin:
            raise ValueError("artifact not found or pin mismatch")
        return value

    def get_qualification(self, pin: ArtifactPinV1) -> RequirementQualificationReceiptV1:
        return self._get_persisted(pin, RequirementQualificationReceiptV1)

    def get_admission(self, pin: ArtifactPinV1) -> RequirementAdmissionReceiptV1:
        return self._get_persisted(pin, RequirementAdmissionReceiptV1)

    def get_activation(self, pin: ArtifactPinV1) -> RequirementActivationReceiptV1:
        return self._get_persisted(pin, RequirementActivationReceiptV1)

    def get_release_manifest(self, pin: ArtifactPinV1) -> ReleaseSetManifestV1:
        return self._get_persisted(pin, ReleaseSetManifestV1)

    def get_activation_projection(
        self, pin: ArtifactPinV1
    ) -> ActivationBindingProjectionV1:
        return self._get_persisted(pin, ActivationBindingProjectionV1)

    def rollback_to_baseline(
        self,
        *,
        target: ArtifactPinV1,
        expected_head: ArtifactPinV1,
        baseline_ref: str,
        rationale: str,
        recorded_at: datetime,
    ) -> RequirementPublicationResult:
        if self._head != expected_head:
            raise ValueError("STALE_BASELINE_HEAD")
        target_baseline = self.get_baseline(target)
        current = self.current_baseline()
        current_by_id = {
            (item.artifact_type, item.artifact_id): item for item in current.member_versions
        }
        deltas = []
        unavailable = {
            lens: DeltaLensResultV1(
                lens=lens,
                status=DeltaLensStatus.UNAVAILABLE,
                operations=(),
                basis_refs=("rollback-to-prior-owner-baseline",),
            )
            for lens in DeltaLens
        }
        for target_member in target_baseline.member_versions:
            predecessor = current_by_id.get(
                (target_member.artifact_type, target_member.artifact_id)
            )
            if predecessor is None or predecessor == target_member:
                continue
            deltas.append(
                GovernedDeltaService.build_envelope(
                    delta_id=f"rollback:{target_member.artifact_id}",
                    target_type=DeltaTargetType.OTHER,
                    predecessor=predecessor,
                    successor_candidate=target_member,
                    lens_results=unavailable,
                )
            )
        if not deltas:
            raise ValueError("ROLLBACK_HAS_NO_CHANGE")
        rollback_basis = (
            ArtifactPinV1(
                artifact_kind="requirement_baseline",
                artifact_ref=target.artifact_ref,
                artifact_hash=target.artifact_hash,
            ),
        )
        if len(deltas) == 1:
            candidate = self.propose_atomic(
                delta=deltas[0],
                rationale=rationale,
                basis_refs=rollback_basis,
                recorded_at=recorded_at,
            )
        else:
            candidate = self.propose_change_set(
                deltas=tuple(deltas),
                rationale=rationale,
                basis_refs=rollback_basis,
                coherence_refs=rollback_basis,
                impact_projections=(),
                recorded_at=recorded_at,
            )
        qualification = self.qualify(
            candidate=candidate.pin(),
            qualification_refs=rollback_basis,
            recorded_at=recorded_at,
        )
        previous_projection = self._latest_activation_projection()
        result = self.admit_and_publish(
            candidate=candidate.pin(),
            qualification=qualification.pin(),
            expected_head=expected_head,
            successor_versions=target_baseline.member_versions,
            baseline_ref=baseline_ref,
            recorded_at=recorded_at,
        )
        if isinstance(result, str):
            raise ValueError(result)
        if previous_projection is not None:
            replacement = result.activation_projection.model_copy(
                update={
                    "predecessor_projection": previous_projection.pin(),
                    "projection_reason": "ROLLBACK_TO_PRIOR_OWNER_NATIVE_RELEASE",
                }
            )
            replacement = ActivationBindingProjectionV1.create(
                **replacement.model_dump(
                    mode="python", exclude={"artifact_ref", "artifact_hash"}
                )
            )
            self._store.put(replacement)
            result = RequirementPublicationResult(
                baseline=result.baseline,
                admission_receipt=result.admission_receipt,
                activation_receipt=result.activation_receipt,
                release_manifest=result.release_manifest,
                activation_projection=replacement,
            )
        return result

    def _latest_activation_projection(self) -> ActivationBindingProjectionV1 | None:
        values = self._load_schema(
            "activation_binding_projection.v1",
            ActivationBindingProjectionV1,
        )
        if not values:
            return None
        return max(values, key=lambda item: item.projected_at)

    def resolve_future_context(self) -> RequirementContextProjectionV1:
        baseline = self.current_baseline()
        return RequirementContextProjectionV1.create(
            schema_version="requirement_context_projection.v1",
            organization_id=self._organization_id,
            workspace_id=self._workspace_id,
            baseline=baseline.pin(),
            member_versions=baseline.member_versions,
            projected_at=baseline.recorded_at,
        )


__all__ = ["RequirementPublicationResult", "RequirementsChangeCandidateService"]
