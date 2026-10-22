"""Create a source-rebound GraphIntent successor behind neutral validation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from contracts.canonical_source_ir import CanonicalSourceDocument, content_hash
from contracts.evidence_location import SourceIdentity
from contracts.graph_intent_brief import GraphIntentBrief, GraphIntentReview
from contracts.source_structure_successor_reservation import (
    SourceStructureSuccessorReservationV1,
)
from semantier.knowledge_feature_packs.canonical_source_ir_adapter import (
    persist_canonical_source_ir,
)
from services.source_structure_semantic_validation_service import (
    SourceStructureSemanticValidationService,
    _SOURCE_STRUCTURE_VALIDATION_CAPABILITY,
    _block_content_inventory,
    _identity_for_binding,
)
from storage.graph_intent_brief_store import GraphIntentBriefStore
from storage.source_structure_successor_reservation_store import (
    SourceStructureSuccessorReservationStore,
    _SOURCE_STRUCTURE_SUCCESSOR_RESERVATION_AUTHORITY,
)


_SOURCE_STRUCTURE_SUCCESSOR_CAPABILITY = object()


class SourceStructureSuccessorService:
    def __init__(
        self,
        workspace_root: str | Path,
        *,
        organization_id: str,
        workspace_id: str,
        graph_intent_store: GraphIntentBriefStore,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.organization_id = organization_id
        self.workspace_id = workspace_id
        self.intents = graph_intent_store

    def create(
        self,
        *,
        predecessor_graph_intent_ref: str,
        predecessor_graph_intent_hash: str,
        corrected_canonical_source_ir: dict[str, Any],
        created_at: str,
        requesting_actor_ref: str,
        _authority: object | None = None,
    ) -> dict[str, Any]:
        if _authority is not _SOURCE_STRUCTURE_SUCCESSOR_CAPABILITY:
            raise PermissionError("source-structure successor creation is capability-sealed")
        predecessor = self.intents.get(
            predecessor_graph_intent_ref,
            graph_intent_hash=predecessor_graph_intent_hash,
        )
        if predecessor.review.state not in {"accepted", "edited"}:
            raise ValueError("source-structure predecessor must be compiler-eligible")
        binding = predecessor.source_binding
        predecessor_identity = self.intents.source_registry.get_source_identity(
            source_identity_ref=binding.source_identity_ref,
            organization_id=self.organization_id,
            workspace_id=self.workspace_id,
            lineage_id=binding.lineage_id,
        )
        if predecessor_identity is None:
            raise ValueError("source-structure predecessor SourceIdentity is missing")
        corrected = CanonicalSourceDocument.model_validate(corrected_canonical_source_ir)
        successor_ref = corrected.source_identity_ref
        if successor_ref == predecessor_identity.source_identity_ref:
            raise ValueError("corrected CanonicalSourceIR requires a new source identity")
        locators = [
            locator
            for mapping in corrected.source_map.mappings
            for locator in mapping.source_locators
        ]
        if not locators or any(
            locator.source_identity_ref != successor_ref
            or locator.source_hash != predecessor_identity.source_hash
            for locator in locators
        ):
            raise ValueError("corrected CanonicalSourceIR locator binding mismatch")
        corrected_hash = content_hash(
            corrected.model_dump(mode="json", by_alias=True, exclude_none=False)
        )
        if corrected_hash == binding.canonical_source_ir_hash:
            raise ValueError("source-structure successor requires a changed CanonicalSourceIR")
        predecessor_ir = self.intents.verify_source_binding(
            binding,
            source_identity=_identity_for_binding(predecessor_identity, binding),
        )
        predecessor_inventory = _block_content_inventory(predecessor_ir)
        corrected_inventory = _block_content_inventory(corrected)
        if corrected_inventory != predecessor_inventory:
            raise ValueError("source-structure correction changed block content inventory")
        corrected_ref = (
            "artifacts/canonical-source-ir/"
            f"{corrected_hash.removeprefix('sha256:')}/canonical_source_ir.v1.json"
        )
        reservation_candidate = SourceStructureSuccessorReservationV1.create(
            {
                "schema_version": "source_structure_successor_reservation.v1",
                "organization_id": self.organization_id,
                "workspace_id": self.workspace_id,
                "lineage_id": binding.lineage_id,
                "predecessor_graph_intent_ref": predecessor.graph_intent_id,
                "predecessor_graph_intent_hash": predecessor.graph_intent_hash,
                "predecessor_source_identity_ref": predecessor_identity.source_identity_ref,
                "successor_source_identity_ref": successor_ref,
                "corrected_canonical_source_ir_ref": corrected_ref,
                "corrected_canonical_source_ir_hash": corrected_hash,
                "block_content_inventory_hash": corrected_inventory,
                "requesting_actor_ref": requesting_actor_ref,
                "created_at": created_at,
            }
        )
        reservation = SourceStructureSuccessorReservationStore(
            self.workspace_root, workspace_id=self.workspace_id
        ).publish(
            reservation_candidate,
            _authority=_SOURCE_STRUCTURE_SUCCESSOR_RESERVATION_AUTHORITY,
        )
        created_at = reservation.created_at
        persisted_ref = persist_canonical_source_ir(
            corrected,
            self.workspace_root,
            organization_id=self.organization_id,
            workspace_id=self.workspace_id,
        )
        if persisted_ref != corrected_ref:
            raise ValueError("corrected CanonicalSourceIR ref derivation mismatch")
        successor_identity = self.intents.source_registry.get_source_identity(
            source_identity_ref=successor_ref,
            organization_id=self.organization_id,
            workspace_id=self.workspace_id,
            lineage_id=binding.lineage_id,
        )
        if successor_identity is None:
            successor_identity = SourceIdentity(
                schemaVersion="semantier.source_identity.v1",
                sourceIdentityRef=successor_ref,
                organizationId=self.organization_id,
                workspaceId=self.workspace_id,
                sourceHash=predecessor_identity.source_hash,
                mediaType=predecessor_identity.media_type,
                originalName=predecessor_identity.original_name,
                artifactRef=predecessor_identity.artifact_ref,
                sourceVersion=None,
                supersedesSourceIdentityRef=predecessor_identity.source_identity_ref,
                createdAt=datetime.fromisoformat(created_at.replace("Z", "+00:00")),
            )
            registered = self.intents.source_registry.register_source_identity(
                source_identity=successor_identity,
                lineage_id=binding.lineage_id,
            )
            successor_identity = successor_identity.model_copy(
                update={"source_version": registered["source_version"]}
            )
        elif (
            successor_identity.supersedes_source_identity_ref
            != predecessor_identity.source_identity_ref
            or successor_identity.source_hash != predecessor_identity.source_hash
        ):
            raise ValueError("existing source-structure successor identity conflicts")
        created_at = successor_identity.created_at.isoformat().replace("+00:00", "Z")
        payload = predecessor.as_dict()
        payload.update(
            {
                "source_binding": {
                    **binding.as_dict(),
                    "source_identity_ref": successor_identity.source_identity_ref,
                    "source_hash": successor_identity.source_hash,
                    "source_version": successor_identity.source_version,
                    "canonical_source_ir_ref": corrected_ref,
                    "canonical_source_ir_hash": corrected_hash,
                },
                "review": {
                    "state": "proposed",
                    "reviewed_by": "system",
                    "actor_ref": "system:source-structure-validation",
                    "operating_mode": "assisted",
                    "policy_ref": None,
                    "rationale_ref": "rationale:source-structure-correction",
                    "decision_evidence_refs": [],
                    "fallback_reason": None,
                },
                "predecessor_graph_intent_ref": predecessor.graph_intent_id,
                "successor_graph_intent_refs": [],
            }
        )
        proposed = self.intents.create(
            GraphIntentBrief.create(payload), source_identity=successor_identity
        )
        validation = SourceStructureSemanticValidationService(
            self.workspace_root,
            organization_id=self.organization_id,
            workspace_id=self.workspace_id,
            graph_intent_store=self.intents,
        ).validate(
            predecessor_graph_intent_ref=predecessor.graph_intent_id,
            predecessor_graph_intent_hash=predecessor.graph_intent_hash,
            successor_graph_intent_ref=proposed.graph_intent_id,
            successor_graph_intent_hash=proposed.graph_intent_hash,
            created_at=created_at,
            created_by=requesting_actor_ref,
            _authority=_SOURCE_STRUCTURE_VALIDATION_CAPABILITY,
        )
        accepted = self.intents.review(
            proposed.graph_intent_id,
            review=GraphIntentReview.from_dict(
                {
                    "state": "accepted",
                    "reviewed_by": "user",
                    "actor_ref": requesting_actor_ref,
                    "operating_mode": "assisted",
                    "policy_ref": None,
                    "rationale_ref": validation.validation_ref,
                    "decision_evidence_refs": [],
                    "fallback_reason": None,
                }
            ),
            source_identity=successor_identity,
        )
        return {
            "source_identity": successor_identity.model_dump(
                mode="json", by_alias=True, exclude_none=False
            ),
            "canonical_source_ir_ref": corrected_ref,
            "canonical_source_ir_hash": corrected_hash,
            "proposed_graph_intent": proposed.as_dict(),
            "semantic_validation": validation.model_dump(mode="json"),
            "accepted_graph_intent": accepted.as_dict(),
        }


__all__ = [
    "SourceStructureSuccessorService",
    "_SOURCE_STRUCTURE_SUCCESSOR_CAPABILITY",
]
