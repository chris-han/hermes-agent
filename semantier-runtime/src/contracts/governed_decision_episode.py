"""Derived governed-decision episode envelope; telemetry/replay, never a kernel authority."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contracts.artifact_pin import ArtifactPinV1
from eos.artifact_hashing import content_hash

_SHA = r"^sha256:[0-9a-f]{64}$"


def _digest(value) -> str:
    return "sha256:" + content_hash(value)


class GovernedDecisionEpisodeV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["governed_decision_episode.v1"] = "governed_decision_episode.v1"
    episode_ref: str = Field(min_length=1)
    episode_hash: str = Field(pattern=_SHA)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)

    descriptive_basis: tuple[ArtifactPinV1, ...]
    normative_basis: tuple[ArtifactPinV1, ...]
    epistemic_basis: tuple[ArtifactPinV1, ...]

    decision_graph_projection: ArtifactPinV1
    projection_sufficiency_proof: ArtifactPinV1
    decision_evidence_view: ArtifactPinV1
    typed_decision_frontier: ArtifactPinV1 | None = None
    qualified_representation: ArtifactPinV1 | None = None
    serialized_model_input: ArtifactPinV1 | None = None
    model_provider: ArtifactPinV1 | None = None
    model_config_pin: ArtifactPinV1 | None = None
    model_prediction: ArtifactPinV1 | None = None
    model_evidence: tuple[ArtifactPinV1, ...] = ()
    deterministic_receipts: tuple[ArtifactPinV1, ...] = ()
    decision_verification: ArtifactPinV1
    decision_record: ArtifactPinV1
    acquisition_reasons: tuple[str, ...] = ()

    commitment_transition: ArtifactPinV1 | None = None
    responsibility_anchor: ArtifactPinV1 | None = None
    commitment_authority_basis_hash: str | None = None
    commitment_scope_ref: str | None = None
    commitment_effective_from: datetime | None = None
    commitment_effective_until: datetime | None = None

    observed_outcome: ArtifactPinV1 | None = None
    captured_at: datetime

    @model_validator(mode="after")
    def validate_boundaries(self):
        if self.decision_graph_projection.artifact_kind != "decision_graph_projection":
            raise ValueError("EPISODE_DGP_REQUIRED")
        if self.projection_sufficiency_proof.artifact_kind != "projection_sufficiency_proof":
            raise ValueError("EPISODE_SUFFICIENCY_PROOF_REQUIRED")
        if self.decision_evidence_view.artifact_kind != "decision_evidence_view":
            raise ValueError("EPISODE_EVIDENCE_VIEW_REQUIRED")
        if self.decision_verification.artifact_kind != "decision_verification":
            raise ValueError("EPISODE_DECISION_VERIFICATION_REQUIRED")
        if self.decision_record.artifact_kind != "decision_record":
            raise ValueError("EPISODE_DECISION_RECORD_REQUIRED")
        if self.typed_decision_frontier is not None and self.typed_decision_frontier.artifact_kind != "typed_decision_frontier":
            raise ValueError("EPISODE_FRONTIER_KIND")
        model_bindings = (
            self.model_provider,
            self.model_config_pin,
            self.model_prediction,
            self.serialized_model_input,
        )
        if any(item is not None for item in model_bindings) and any(item is None for item in model_bindings):
            raise ValueError("INCOMPLETE_MODEL_CAPTURE_BINDINGS")
        if self.model_prediction is not None:
            if self.model_prediction.artifact_kind != "decision_distribution":
                raise ValueError("MODEL_PREDICTION_MUST_BE_DISTRIBUTION")
            if self.model_prediction == self.decision_record:
                raise ValueError("MODEL_PREDICTION_CANNOT_BECOME_LABEL")
        if self.qualified_representation is not None and self.qualified_representation.artifact_kind != "qualified_decision_representation":
            raise ValueError("QUALIFIED_REPRESENTATION_KIND")
        if self.serialized_model_input is not None and self.serialized_model_input.artifact_kind != "typed_decision_model_input":
            raise ValueError("SERIALIZED_MODEL_INPUT_KIND")
        if len({(p.artifact_kind, p.artifact_ref, p.artifact_hash) for p in self.deterministic_receipts}) != len(self.deterministic_receipts):
            raise ValueError("DUPLICATE_DETERMINISTIC_RECEIPT")
        if any(not reason.strip() for reason in self.acquisition_reasons) or len(set(self.acquisition_reasons)) != len(self.acquisition_reasons):
            raise ValueError("INVALID_ACQUISITION_REASONS")
        bindings = (
            self.responsibility_anchor,
            self.commitment_authority_basis_hash,
            self.commitment_scope_ref,
            self.commitment_effective_from,
        )
        if self.commitment_transition is None:
            if any(item is not None for item in bindings) or self.commitment_effective_until is not None:
                raise ValueError("COMMITMENT_BINDINGS_WITHOUT_TRANSITION")
        else:
            if self.commitment_transition.artifact_kind != "commitment_transition":
                raise ValueError("EPISODE_COMMITMENT_TRANSITION_KIND")
            if any(item is None for item in bindings):
                raise ValueError("INCOMPLETE_COMMITMENT_BINDINGS")
            if self.responsibility_anchor.artifact_kind != "human_responsibility_context":
                raise ValueError("INCOMPLETE_COMMITMENT_BINDINGS")
            if self.commitment_authority_basis_hash is None or not self.commitment_authority_basis_hash.startswith("sha256:"):
                raise ValueError("INCOMPLETE_COMMITMENT_BINDINGS")
        digest = _digest(self.hash_payload())
        if (self.episode_ref, self.episode_hash) != (
            "governed-decision-episode:v1:" + digest[7:],
            digest,
        ):
            raise ValueError("governed decision episode identity mismatch")
        return self

    def hash_payload(self) -> dict:
        return self.model_dump(mode="json", exclude={"episode_ref", "episode_hash"})

    @classmethod
    def create(cls, **body):
        provisional = cls.model_construct(
            episode_ref="governed-decision-episode:v1:pending",
            episode_hash="sha256:" + "0" * 64,
            **body,
        )
        digest = _digest(provisional.hash_payload())
        return cls.model_validate({
            **body,
            "episode_ref": "governed-decision-episode:v1:" + digest[7:],
            "episode_hash": digest,
        })
