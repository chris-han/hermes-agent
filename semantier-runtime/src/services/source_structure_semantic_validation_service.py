"""Validate that a source-structure successor preserves source semantics."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from contracts.canonical_source_ir import CanonicalSourceDocument, content_hash
from eos.artifact_hashing import content_hash as artifact_content_hash
from storage.graph_intent_brief_store import GraphIntentBriefStore
from storage.source_structure_semantic_validation_store import (
    SourceStructureSemanticValidationStore,
    _SOURCE_STRUCTURE_VALIDATION_AUTHORITY,
)
from contracts.source_structure_semantic_validation import (
    SourceStructureSemanticValidationV1,
)


_SOURCE_STRUCTURE_VALIDATION_CAPABILITY = object()


def _source_identity_hash(identity) -> str:
    return "sha256:" + artifact_content_hash(
        identity.model_dump(mode="json", by_alias=True, exclude_none=False)
    )


def _block_content_inventory(document: CanonicalSourceDocument) -> str:
    counts: Counter[str] = Counter()
    for block in document.blocks:
        actual = content_hash(block.content)
        if actual != block.content_hash:
            raise ValueError("CanonicalSourceIR block content hash mismatch")
        counts[actual] += 1
    if not counts:
        raise ValueError("source-structure validation requires nonempty source content")
    return "sha256:" + artifact_content_hash(
        [{"content_hash": key, "count": counts[key]} for key in sorted(counts)]
    )


def _identity_for_binding(identity, binding):
    """Preserve the registry's bounded historical-null version alias on reads."""

    if binding.source_version is None and identity.source_version is not None:
        return identity.model_copy(update={"source_version": None})
    return identity


class SourceStructureSemanticValidationService:
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
        self.store = SourceStructureSemanticValidationStore(
            self.workspace_root, workspace_id=workspace_id
        )

    def validate(
        self,
        *,
        predecessor_graph_intent_ref: str,
        predecessor_graph_intent_hash: str,
        successor_graph_intent_ref: str,
        successor_graph_intent_hash: str,
        created_at: str,
        created_by: str,
        _authority: object | None = None,
    ) -> SourceStructureSemanticValidationV1:
        if _authority is not _SOURCE_STRUCTURE_VALIDATION_CAPABILITY:
            raise PermissionError("source-structure semantic validation is capability-sealed")
        predecessor = self.intents.get(
            predecessor_graph_intent_ref,
            graph_intent_hash=predecessor_graph_intent_hash,
        )
        successor = self.intents.get(
            successor_graph_intent_ref,
            graph_intent_hash=successor_graph_intent_hash,
        )
        if successor.review.state == "proposed":
            validated_successor = successor
            if successor.predecessor_graph_intent_ref != predecessor.graph_intent_id:
                raise ValueError("source-structure GraphIntent successor lineage mismatch")
        else:
            if successor.predecessor_graph_intent_ref is None:
                raise ValueError("source-structure GraphIntent successor lineage mismatch")
            proposed_successor = self.intents.get(successor.predecessor_graph_intent_ref)
            validated_successor = proposed_successor
            if (
                proposed_successor.predecessor_graph_intent_ref
                != predecessor.graph_intent_id
                or proposed_successor.review.state != "proposed"
                or successor.source_binding != proposed_successor.source_binding
            ):
                raise ValueError("source-structure GraphIntent successor lineage mismatch")
        if (
            predecessor.source_binding.organization_id != self.organization_id
            or predecessor.source_binding.workspace_id != self.workspace_id
            or successor.source_binding.organization_id != self.organization_id
            or successor.source_binding.workspace_id != self.workspace_id
            or successor.source_binding.lineage_id
            != predecessor.source_binding.lineage_id
        ):
            raise ValueError("source-structure GraphIntent tenant/lineage mismatch")
        semantic_fields = (
            "document_function",
            "represented_or_governed_subject",
            "primary_relationship",
            "preferred_representation",
            "material_dimensions",
            "source_role_interpretation",
            "authority_boundary",
            "competency_questions",
            "hard_negative_questions",
            "alternatives_considered",
            "assumptions",
            "unresolved_questions",
        )
        if any(
            getattr(predecessor, field) != getattr(successor, field)
            for field in semantic_fields
        ):
            raise ValueError("source-structure correction changed GraphIntent semantics")
        predecessor_identity = self.intents.source_registry.get_source_identity(
            source_identity_ref=predecessor.source_binding.source_identity_ref,
            organization_id=self.organization_id,
            workspace_id=self.workspace_id,
            lineage_id=predecessor.source_binding.lineage_id,
        )
        successor_identity = self.intents.source_registry.get_source_identity(
            source_identity_ref=successor.source_binding.source_identity_ref,
            organization_id=self.organization_id,
            workspace_id=self.workspace_id,
            lineage_id=successor.source_binding.lineage_id,
        )
        if predecessor_identity is None or successor_identity is None:
            raise ValueError("source-structure SourceIdentity authority is missing")
        if (
            successor_identity.supersedes_source_identity_ref
            != predecessor_identity.source_identity_ref
            or successor_identity.source_hash != predecessor_identity.source_hash
            or successor_identity.artifact_ref != predecessor_identity.artifact_ref
            or successor_identity.media_type != predecessor_identity.media_type
        ):
            raise ValueError("source-structure SourceIdentity successor mismatch")
        predecessor_ir = self.intents.verify_source_binding(
            predecessor.source_binding,
            source_identity=_identity_for_binding(
                predecessor_identity, predecessor.source_binding
            ),
        )
        successor_ir = self.intents.verify_source_binding(
            successor.source_binding,
            source_identity=_identity_for_binding(
                successor_identity, successor.source_binding
            ),
        )
        predecessor_inventory = _block_content_inventory(predecessor_ir)
        successor_inventory = _block_content_inventory(successor_ir)
        validation = SourceStructureSemanticValidationV1.create(
            {
                "schema_version": "source_structure_semantic_validation.v1",
                "organization_id": self.organization_id,
                "workspace_id": self.workspace_id,
                "predecessor_graph_intent_ref": predecessor.graph_intent_id,
                "predecessor_graph_intent_hash": predecessor.graph_intent_hash,
                "successor_graph_intent_ref": validated_successor.graph_intent_id,
                "successor_graph_intent_hash": validated_successor.graph_intent_hash,
                "predecessor_source_identity_ref": predecessor_identity.source_identity_ref,
                "predecessor_source_identity_hash": _source_identity_hash(
                    predecessor_identity
                ),
                "successor_source_identity_ref": successor_identity.source_identity_ref,
                "successor_source_identity_hash": _source_identity_hash(successor_identity),
                "predecessor_canonical_source_ir_ref": (
                    predecessor.source_binding.canonical_source_ir_ref
                ),
                "predecessor_canonical_source_ir_hash": (
                    predecessor.source_binding.canonical_source_ir_hash
                ),
                "successor_canonical_source_ir_ref": (
                    successor.source_binding.canonical_source_ir_ref
                ),
                "successor_canonical_source_ir_hash": (
                    successor.source_binding.canonical_source_ir_hash
                ),
                "validation_method": "block_content_identity.v1",
                "predecessor_block_content_inventory_hash": predecessor_inventory,
                "successor_block_content_inventory_hash": successor_inventory,
                "competency_question_refs": [
                    item.question_id for item in predecessor.competency_questions
                ],
                "hard_negative_question_refs": [
                    item.question_id for item in predecessor.hard_negative_questions
                ],
                "outcome": "passed",
                "created_at": created_at,
                "created_by": created_by,
            }
        )
        return self.store.publish(
            validation, _authority=_SOURCE_STRUCTURE_VALIDATION_AUTHORITY
        )


__all__ = [
    "SourceStructureSemanticValidationService",
    "_SOURCE_STRUCTURE_VALIDATION_CAPABILITY",
]
