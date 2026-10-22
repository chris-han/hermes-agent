"""Stage-specific immutable binding over already-frozen experiment authorities."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contracts.artifact_pin import ArtifactPinV1
from contracts.ontology_quality import QualityEvaluationRunV1, QualityProfileV1
from contracts.ontology_paired_evaluation import OntologyPairedObligationSuiteV1, PairedEvaluatorBundleV1
from eos.artifact_hashing import content_hash

IntegrityStage = Literal["PRE_PILOT", "PILOT", "CAL_DEV", "CAL_HELDOUT"]
PRE_PILOT_AUTHORITY_KINDS = {
    "evaluator": "tender_decision_local_evaluator",
    "obligation_suite": "tender_decision_local_obligation_suite",
    "integrity_policy": "trajectory_integrity_policy",
    "runtime_contract": "tender_decision_runtime_contract",
    "scorer_contract": "tender_decision_scorer_contract",
    "collector_contract": "trajectory_collector_contract",
}


class PrePilotAuthorityArtifactV1(BaseModel):
    """Repository-owned freeze wrapper around an existing PRE_PILOT authority."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["pre_pilot_authority_artifact.v1"] = "pre_pilot_authority_artifact.v1"
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    authority_kind: str = Field(min_length=1)
    authority_ref: str = Field(min_length=1)
    authority_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    artifact_path: str = Field(min_length=1)
    payload: dict[str, Any]

    @model_validator(mode="after")
    def exact_content_identity(self) -> "PrePilotAuthorityArtifactV1":
        digest = "sha256:" + content_hash(self.payload)
        if self.authority_hash != digest or self.authority_ref != f"{self.authority_kind}:{digest[7:]}":
            raise ValueError("PRE_PILOT_AUTHORITY_CONTENT_MISMATCH")
        if self.payload.get("authority_kind") != self.authority_kind:
            raise ValueError("PRE_PILOT_AUTHORITY_KIND_MISMATCH")
        return self

    @property
    def snapshot_ref(self) -> str:
        return self.authority_ref

    @property
    def snapshot_hash(self) -> str:
        return self.authority_hash

    @property
    def source_file(self) -> Path:
        return Path(self.artifact_path)

    def pin(self) -> ArtifactPinV1:
        return ArtifactPinV1(artifact_kind=self.authority_kind,
                             artifact_ref=self.authority_ref,
                             artifact_hash=self.authority_hash)


class ExperimentIntegritySnapshotV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "experiment_integrity_snapshot.v1"
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    integrity_stage: IntegrityStage
    quality_profile: ArtifactPinV1
    evaluator: ArtifactPinV1
    obligation_suite: ArtifactPinV1
    integrity_policy: ArtifactPinV1
    runtime_contract: ArtifactPinV1
    scorer_contract: ArtifactPinV1
    collector_contract: ArtifactPinV1
    pilot_membership: ArtifactPinV1 | None = None
    producer_comparison_contract: ArtifactPinV1 | None = None
    ordered_producer_configs: tuple[ArtifactPinV1, ...] = ()
    producer_config: ArtifactPinV1 | None = None
    cal_dev_membership: ArtifactPinV1 | None = None
    cal_heldout_membership: ArtifactPinV1 | None = None
    calibration_protocol: ArtifactPinV1 | None = None
    operating_point: ArtifactPinV1 | None = None
    t4_freeze_receipt: ArtifactPinV1 | None = None
    authority_payloads: tuple[dict[str, Any], ...] = ()
    created_at: datetime
    snapshot_ref: str = Field(min_length=1)
    snapshot_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    @model_validator(mode="after")
    def exact_identity_and_stage_requirements(self) -> "ExperimentIntegritySnapshotV1":
        if self.created_at.tzinfo is None or self.created_at.utcoffset() != timedelta(0):
            raise ValueError("created_at must be timezone-aware UTC")
        for field, expected_kind in PRE_PILOT_AUTHORITY_KINDS.items():
            if getattr(self, field).artifact_kind != expected_kind:
                raise ValueError(f"{field} must pin {expected_kind}")
        if self.quality_profile.artifact_kind != "quality_profile":
            raise ValueError("quality_profile must pin the canonical quality profile")
        pins = self.bound_pins
        identities = [(pin.artifact_kind, pin.artifact_ref) for pin in pins]
        if len(identities) != len(set(identities)):
            raise ValueError("experiment integrity pins must have unique exact identities")
        if self.integrity_stage == "PILOT":
            _require(self.pilot_membership, "pilot_membership")
            _require(self.producer_comparison_contract, "producer_comparison_contract")
            if not self.ordered_producer_configs:
                raise ValueError("PILOT requires ordered producer configs")
        elif self.integrity_stage == "CAL_DEV":
            _require(self.cal_dev_membership, "cal_dev_membership")
            _require(self.calibration_protocol, "calibration_protocol")
            _require(self.producer_config, "producer_config")
        elif self.integrity_stage == "CAL_HELDOUT":
            _require(self.cal_heldout_membership, "cal_heldout_membership")
            _require(self.calibration_protocol, "calibration_protocol")
            _require(self.operating_point, "operating_point")
            _require(self.t4_freeze_receipt, "t4_freeze_receipt")
            _require(self.producer_config, "producer_config")
        _verify_typed_child(self.evaluator, self.authority_payloads, validate=False)
        _verify_typed_child(self.obligation_suite, self.authority_payloads, validate=False)
        expected = "sha256:" + content_hash(self.hash_payload())
        if self.snapshot_hash != expected or self.snapshot_ref != "experiment-integrity-snapshot:" + expected[7:]:
            raise ValueError("experiment integrity snapshot ref/hash does not bind exact content")
        return self

    @property
    def bound_pins(self) -> tuple[ArtifactPinV1, ...]:
        optional = (
            self.pilot_membership, self.producer_comparison_contract,
            self.cal_dev_membership, self.cal_heldout_membership,
            self.calibration_protocol, self.operating_point, self.t4_freeze_receipt,
            self.producer_config,
        )
        return (self.quality_profile, self.evaluator, self.obligation_suite, self.integrity_policy,
                self.runtime_contract, self.scorer_contract, self.collector_contract,
                *(pin for pin in optional if pin is not None), *self.ordered_producer_configs)

    def hash_payload(self) -> dict[str, Any]:
        value = self.model_dump(mode="json")
        value.pop("snapshot_ref", None)
        value.pop("snapshot_hash", None)
        return value

    def pin(self) -> ArtifactPinV1:
        return ArtifactPinV1(artifact_kind="experiment_integrity_snapshot",
                             artifact_ref=self.snapshot_ref, artifact_hash=self.snapshot_hash)

    def verify_resolved_authorities(self) -> bool:
        try:
            for pin in self.bound_pins:
                if pin.artifact_kind in _TYPED_CHILDREN:
                    _verify_typed_child(pin, self.authority_payloads)
                    contract, _, _ = _TYPED_CHILDREN[pin.artifact_kind]
                    row = next(row for row in self.authority_payloads
                               if (row.get("artifact_kind"), row.get("artifact_ref"), row.get("artifact_hash"))
                               == (pin.artifact_kind, pin.artifact_ref, pin.artifact_hash))
                    child = contract.model_validate(row["payload"])
                    if (child.organization_id, child.workspace_id) != (self.organization_id, self.workspace_id):
                        return False
                else:
                    rows = [row for row in self.authority_payloads
                            if (row.get("artifact_kind"), row.get("artifact_ref"), row.get("artifact_hash"))
                            == (pin.artifact_kind, pin.artifact_ref, pin.artifact_hash)
                            and isinstance(row.get("payload"), dict)]
                    if len(rows) != 1 or pin.artifact_hash != "sha256:" + content_hash(rows[0]["payload"]):
                        return False
            return True
        except (ValueError, TypeError):
            return False

    def require_quality_basis(self, *, evaluation: QualityEvaluationRunV1, decision_local_projection) -> None:
        """Qualify only the exact unperturbed graph used by the retained DLP."""
        from contracts.decision_local_projection import DecisionLocalProjectionV1
        from contracts.decision_local_projection_v2 import DecisionLocalProjectionV2
        evaluation = QualityEvaluationRunV1.model_validate(evaluation.model_dump(mode="json"))
        if isinstance(decision_local_projection, DecisionLocalProjectionV2):
            projection = DecisionLocalProjectionV2.model_validate(decision_local_projection.model_dump(mode="json")).legacy
        else:
            projection = DecisionLocalProjectionV1.model_validate(decision_local_projection.model_dump(mode="json"))
        scope = (self.organization_id, self.workspace_id)
        if scope != (evaluation.organization_id, evaluation.workspace_id) or scope != (projection.organization_id, projection.workspace_id):
            raise ValueError("experiment quality tenant mismatch")
        from services.ontology_quality_profile_service import tender_mvl0_quality_profile
        required = tender_mvl0_quality_profile(organization_id=self.organization_id, workspace_id=self.workspace_id)
        if evaluation.profile_definition.requirements != required.requirements:
            raise ValueError("experiment requires the full mandatory MVL0 profile")
        if evaluation.quality_profile != self.quality_profile:
            raise ValueError("experiment quality profile mismatch")
        if evaluation.subject != projection.context_graph:
            raise ValueError("experiment quality graph/context subject mismatch")
        if evaluation.disposition not in {"QUALIFIED", "QUALIFIED_WITH_WARNINGS"}:
            raise ValueError("experiment baseline must be qualified")
        if evaluation.disposition == "QUALIFIED_WITH_WARNINGS" and not evaluation.profile_definition.allow_warnings:
            raise ValueError("experiment frozen profile forbids warnings")

    def authorizes_stage(self, required_stage: IntegrityStage) -> bool:
        order = {"PRE_PILOT": 0, "PILOT": 1, "CAL_DEV": 2, "CAL_HELDOUT": 3}
        return order[self.integrity_stage] >= order[required_stage]

    @classmethod
    def create(cls, **values: Any) -> "ExperimentIntegritySnapshotV1":
        provisional = cls.model_construct(**{
            **values,
            "snapshot_ref": "experiment-integrity-snapshot:pending",
            "snapshot_hash": "sha256:" + "0" * 64,
        })
        digest = content_hash(provisional.hash_payload())
        return cls.model_validate({
            **values,
            "snapshot_ref": f"experiment-integrity-snapshot:{digest}",
            "snapshot_hash": f"sha256:{digest}",
        })


def _require(pin: ArtifactPinV1 | None, name: str) -> None:
    if pin is None:
        raise ValueError(f"{name} required for this experiment integrity stage")


def _verify_typed_child(pin: ArtifactPinV1, payloads: tuple[dict[str, Any], ...], *, validate: bool = True) -> None:
    model = _TYPED_CHILDREN.get(pin.artifact_kind)
    if model is None:
        return
    matches = [row for row in payloads if (row.get("artifact_kind"), row.get("artifact_ref"), row.get("artifact_hash")) == (pin.artifact_kind, pin.artifact_ref, pin.artifact_hash)]
    if len(matches) != 1 or not isinstance(matches[0].get("payload"), dict):
        raise ValueError("typed frozen authority payload missing")
    if not validate:
        return
    contract, ref_field, hash_field = model
    owner = contract.model_validate(matches[0]["payload"])
    if (getattr(owner, ref_field), getattr(owner, hash_field)) != (pin.artifact_ref, pin.artifact_hash):
        raise ValueError("typed frozen authority pin mismatch")


_TYPED_CHILDREN = {
        "quality_profile": (QualityProfileV1, "profile_ref", "profile_hash"),
        "paired_evaluator_bundle": (PairedEvaluatorBundleV1, "evaluator_ref", "evaluator_hash"),
        "ontology_paired_obligation_suite": (OntologyPairedObligationSuiteV1, "suite_ref", "suite_hash"),
}


__all__ = ["ExperimentIntegritySnapshotV1", "PrePilotAuthorityArtifactV1", "IntegrityStage"]
