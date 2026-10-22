"""Executable P2 replay gate for persisted Tender decision-local projections.

The gate is verification-only. It creates no semantic authority and mutates no
WorkingContext, DecisionLocalProjection, governed rule, or activation state.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contracts.decision_local_projection import DecisionLocalProjectionV1
from contracts.observed_constraint_hypothesis import (
    EffectiveRestrictionRuntimeContextV1,
    RuntimeSemanticRetrievalSnapshotV1,
)
from contracts.ontology_building import ArtifactPinV1
from contracts.semantic_kernel import WorkingContext
from eos.artifact_hashing import content_hash
from services.decision_local_projection_service import DecisionLocalProjectionService
from services.effective_restriction_runtime_context_service import (
    EffectiveRestrictionRuntimeContextService,
)
from services.tender_decision_local_projection_service import (
    TenderDecisionLocalProjectionService,
)
from storage.governed_rule_store import GovernedRuleStore
from storage.ontology_build_store import OntologyBuildStore, ontology_build_store_path
from storage.runtime_semantic_hypothesis_artifact_store import (
    RuntimeSemanticHypothesisArtifactStore,
)

TENDER_DECISION_PROJECTION_REPLAY_GREEN = "TENDER_DECISION_PROJECTION_REPLAY_GREEN"
_REPLAY_CHECKS = (
    "PERSISTED_ARTIFACTS_VALID",
    "WORKING_CONTEXT_RECONSTRUCTED",
    "HISTORICAL_AUTHORITY_REPLAYED",
    "PROJECTION_IDENTITY_EQUAL",
)


class TenderDecisionProjectionReplayGateReceiptV1(BaseModel):
    """Content-addressed evidence that the P2 replay gate passed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["tender_decision_projection_replay_gate_receipt.v1"] = (
        "tender_decision_projection_replay_gate_receipt.v1"
    )
    gate_name: Literal["TENDER_DECISION_PROJECTION_REPLAY_GREEN"] = (
        TENDER_DECISION_PROJECTION_REPLAY_GREEN
    )
    status: Literal["GREEN"] = "GREEN"
    receipt_ref: str = Field(min_length=1)
    receipt_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    persisted_projection: ArtifactPinV1
    replayed_projection: ArtifactPinV1
    working_context: ArtifactPinV1
    retrieval_snapshot: ArtifactPinV1
    checks: tuple[str, ...]

    @model_validator(mode="after")
    def exact_identity(self) -> "TenderDecisionProjectionReplayGateReceiptV1":
        if self.checks != _REPLAY_CHECKS:
            raise ValueError("replay gate receipt checks are incomplete or reordered")
        if self.persisted_projection != self.replayed_projection:
            raise ValueError("replay gate receipt cannot report unequal projection identities")
        expected = "sha256:" + content_hash(self.hash_payload())
        if self.receipt_hash != expected or self.receipt_ref != "tender-decision-projection-replay:" + expected[7:]:
            raise ValueError("replay gate receipt identity does not bind exact content")
        return self

    def hash_payload(self) -> dict:
        payload = self.model_dump(mode="json")
        payload.pop("receipt_ref", None)
        payload.pop("receipt_hash", None)
        return payload

    @classmethod
    def create(cls, **values) -> "TenderDecisionProjectionReplayGateReceiptV1":
        draft = cls.model_construct(
            receipt_ref="pending",
            receipt_hash="sha256:" + "0" * 64,
            **values,
        )
        digest = "sha256:" + content_hash(draft.hash_payload())
        return cls.model_validate(
            {
                **values,
                "receipt_ref": "tender-decision-projection-replay:" + digest[7:],
                "receipt_hash": digest,
            }
        )


RuntimeContextResolver = Callable[
    [RuntimeSemanticRetrievalSnapshotV1], EffectiveRestrictionRuntimeContextV1
]
GovernedRuleResolver = Callable[[str], Mapping | None]


class TenderDecisionProjectionReplayGateService:
    """Re-open, reconstruct, and deterministically replay one persisted DLP."""

    def __init__(
        self,
        *,
        runtime_root: str | Path,
        workspace_root: str | Path,
        db_path: str | Path,
        build_store_path: str | Path | None = None,
        runtime_context_resolver: RuntimeContextResolver | None = None,
        governed_rule_resolver: GovernedRuleResolver | None = None,
    ) -> None:
        self._runtime_root = Path(runtime_root)
        self._workspace_root = Path(workspace_root)
        self._db_path = Path(db_path)
        self._build_store_path = (
            Path(build_store_path)
            if build_store_path is not None
            else ontology_build_store_path(self._workspace_root)
        )
        self._runtime_context_resolver = (
            runtime_context_resolver or self._resolve_historical_runtime_context
        )
        self._governed_rule_resolver = governed_rule_resolver

    def _resolve_historical_runtime_context(
        self, snapshot: RuntimeSemanticRetrievalSnapshotV1
    ) -> EffectiveRestrictionRuntimeContextV1:
        effective_at = datetime.fromisoformat(
            snapshot.effective_at.replace("Z", "+00:00")
        )
        return EffectiveRestrictionRuntimeContextService(
            self._db_path,
            clock=lambda: effective_at,
        ).resolve_current(
            organization_id=snapshot.organization_id,
            workspace_id=snapshot.workspace_id,
        )

    @staticmethod
    def _projection_pin(projection: DecisionLocalProjectionV1) -> ArtifactPinV1:
        return ArtifactPinV1(
            artifact_kind="decision_local_projection",
            artifact_ref=projection.projection_ref,
            artifact_hash=projection.projection_hash,
        )

    @staticmethod
    def _snapshot_pin(snapshot: RuntimeSemanticRetrievalSnapshotV1) -> ArtifactPinV1:
        return ArtifactPinV1(
            artifact_kind="runtime_semantic_retrieval_snapshot",
            artifact_ref=snapshot.snapshot_id,
            artifact_hash=snapshot.snapshot_hash,
        )

    @staticmethod
    def _reconstruct_working_context(stored: WorkingContext) -> WorkingContext:
        payload = stored.model_dump(
            mode="python", exclude={"value_ref", "value_hash"}
        )
        reconstructed = WorkingContext.create(**payload)
        if reconstructed != stored:
            raise ValueError("TENDER_DECISION_PROJECTION_REPLAY_WORKING_CONTEXT_MISMATCH")
        return reconstructed

    def verify(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        projection_ref: str,
        projection_hash: str,
    ) -> TenderDecisionProjectionReplayGateReceiptV1:
        # Fresh store instances are deliberate: the gate cannot succeed from
        # cached in-memory objects created by the original execution.
        artifact_store = RuntimeSemanticHypothesisArtifactStore(
            self._runtime_root,
            organization_id=organization_id,
            workspace_id=workspace_id,
        )
        try:
            persisted, stored_context, snapshot, source_region = artifact_store.get_decision_projection(
                projection_ref=projection_ref,
                projection_hash=projection_hash,
                include_source_region=True,
            )
        except Exception as exc:
            raise ValueError("TENDER_DECISION_PROJECTION_REPLAY_TAMPERED") from exc

        try:
            reconstructed_context = self._reconstruct_working_context(stored_context)
        except Exception as exc:
            if str(exc) == "TENDER_DECISION_PROJECTION_REPLAY_WORKING_CONTEXT_MISMATCH":
                raise
            raise ValueError("TENDER_DECISION_PROJECTION_REPLAY_TAMPERED") from exc

        canonical_patterns = tuple(
            pin
            for pin in persisted.deterministic_constraints
            if pin.artifact_kind == "canonical_pattern"
        )
        if len(canonical_patterns) != 1:
            raise ValueError("TENDER_DECISION_PROJECTION_REPLAY_TAMPERED")
        canonical_pattern = canonical_patterns[0]

        historical_runtime = self._runtime_context_resolver(snapshot)
        build_store = OntologyBuildStore(self._build_store_path)
        governed_rule_resolver = self._governed_rule_resolver
        if governed_rule_resolver is None:
            governed_rule_resolver = GovernedRuleStore(
                self._db_path
            ).get_rule_version
        projection_service = TenderDecisionLocalProjectionService(
            projection_service=DecisionLocalProjectionService(
                build_store=build_store
            ),
            governed_rule_resolver=governed_rule_resolver,
        )
        try:
            replayed = projection_service.compile_runtime_bound(
                working_context=reconstructed_context,
                decision_subject_ref=persisted.decision_subject_ref,
                runtime_context=historical_runtime,
                retrieval_snapshot=snapshot,
                canonical_pattern_ref=canonical_pattern.artifact_ref,
                source_region=source_region,
            )
        except ValueError as exc:
            if str(exc) == "TENDER_RUNTIME_AUTHORITY_MISMATCH":
                raise ValueError(
                    "TENDER_DECISION_PROJECTION_REPLAY_STALE_AUTHORITY"
                ) from exc
            raise

        if replayed != persisted:
            raise ValueError("TENDER_DECISION_PROJECTION_REPLAY_IDENTITY_MISMATCH")

        persisted_pin = self._projection_pin(persisted)
        return TenderDecisionProjectionReplayGateReceiptV1.create(
            organization_id=organization_id,
            workspace_id=workspace_id,
            persisted_projection=persisted_pin,
            replayed_projection=self._projection_pin(replayed),
            working_context=reconstructed_context.pin(),
            retrieval_snapshot=self._snapshot_pin(snapshot),
            checks=_REPLAY_CHECKS,
        )


__all__ = [
    "TENDER_DECISION_PROJECTION_REPLAY_GREEN",
    "TenderDecisionProjectionReplayGateReceiptV1",
    "TenderDecisionProjectionReplayGateService",
]
