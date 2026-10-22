"""Deterministic role-evidence-driven source semantic witness discovery."""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any, Mapping

from contracts.canonical_source_ir import CanonicalSourceDocument, SourceBlock
from contracts.evidence_location import SourceIdentity
from contracts.graph_intent_brief import GraphIntentBrief
from contracts.source_semantic_witness import (
    SemanticExpectation,
    SourceRoleInterpretationEvidenceV1,
    SourceSemanticWitnessV1,
    WitnessReviewRecord,
    WitnessSourceBinding,
)
from semantier.knowledge_feature_packs.canonical_source_ir_adapter import (
    resolve_unique_mapping,
)
from semantier.knowledge_feature_packs.evidence_resolver import resolve_evidence_ref
from semantier.knowledge_feature_packs.evidence_selector_builder import (
    build_evidence_selector,
    evidence_ref_for_block,
)


def _canonical_ir_hash(document: CanonicalSourceDocument) -> str:
    payload = document.model_dump(mode="json", by_alias=True, exclude_none=False)
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return "sha256:" + sha256(encoded).hexdigest()


class SourceSemanticWitnessService:
    """Discover witness cells from reviewed role evidence, without live inference."""

    def discover(
        self,
        *,
        document: CanonicalSourceDocument,
        source_identity: SourceIdentity,
        graph_intent: GraphIntentBrief,
        source_role_evidence: SourceRoleInterpretationEvidenceV1,
        actor_ref: str,
        review_set_ref: str,
        expectations_by_block_id: Mapping[str, SemanticExpectation] | None = None,
        witness_kind: str = "positive_exemplar",
        evidence_provenance_by_block_id: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> tuple[SourceSemanticWitnessV1, ...]:
        self._validate_inputs(
            document=document,
            source_identity=source_identity,
            graph_intent=graph_intent,
            source_role_evidence=source_role_evidence,
        )
        expectations = expectations_by_block_id or {}
        provenance_overrides = evidence_provenance_by_block_id or {}
        blocks_by_id = {block.block_id: block for block in document.blocks}
        reference_columns: set[tuple[int, int]] = set()
        direct_reference_blocks: set[str] = set()
        for binding in source_role_evidence.field_bindings:
            allowed_types = getattr(graph_intent.source_role_interpretation, binding.role_kind)
            if binding.canonical_type_id not in allowed_types:
                raise ValueError(
                    "source-role evidence canonical type is not present in GraphIntentBrief"
                )
            block_id = str(binding.header_evidence_ref.provenance.get("source_block_id") or "")
            block = blocks_by_id.get(block_id)
            if block is None:
                raise ValueError(
                    "source-role EvidenceRef does not resolve to a canonical block"
                )
            resolved = resolve_evidence_ref(
                binding.header_evidence_ref,
                document,
                artifact_ref=str(
                    binding.header_evidence_ref.provenance.get("artifact_ref")
                    or "artifact:canonical-source"
                ),
                selector_hash=binding.header_evidence_ref.selector_hash,
            )
            canonical_selector = build_evidence_selector(document, block)
            canonical_locator = resolve_unique_mapping(
                document, block.block_id
            ).model_dump(mode="json", by_alias=True)
            if (
                resolved.status != "exact"
                or resolved.matched_text != str(block.content or "")
                or binding.header_evidence_ref.selector != canonical_selector
                or binding.header_evidence_ref.provenance.get("source_locator")
                != canonical_locator
            ):
                raise ValueError(
                    "source-role EvidenceRef selector, locator, and provenance must "
                    "resolve to the same canonical block"
                )
            if block.block_type == "cell":
                if block.structure.get("rowOrdinal") != 1:
                    raise ValueError(
                        "source-role field binding must resolve to a table header cell"
                    )
            elif (
                block.block_type == "paragraph"
                and binding.role_kind == "reference_examples"
            ):
                direct_reference_blocks.add(block.block_id)
            else:
                raise ValueError(
                    "source-role binding must resolve to a table header cell or "
                    "an explicit reference-example paragraph"
                )
            if (
                binding.role_kind == "reference_examples"
                and block.block_type == "cell"
            ):
                reference_columns.add(
                    (
                        int(block.structure["tableOrdinal"]),
                        int(block.structure["columnOrdinal"]),
                    )
                )

        candidate_blocks = sorted(
            (
                block
                for block in document.blocks
                if str(block.content or "").strip()
                and (
                    block.block_id in direct_reference_blocks
                    or (
                        block.block_type == "cell"
                        and int(block.structure.get("rowOrdinal") or 0) > 1
                        and (
                            int(block.structure.get("tableOrdinal") or 0),
                            int(block.structure.get("columnOrdinal") or 0),
                        )
                        in reference_columns
                    )
                )
            ),
            key=lambda block: (
                0 if block.block_type == "cell" else 1,
                int(block.structure.get("tableOrdinal") or 0),
                int(
                    block.structure.get(
                        "rowOrdinal", block.structure.get("bodyOrdinal", 0)
                    )
                    or 0
                ),
                int(block.structure.get("columnOrdinal") or 0),
                block.block_id,
            ),
        )
        source_binding = WitnessSourceBinding(
            sourceIdentitySchemaVersion="semantier.source_identity.v1",
            sourceIdentityRef=source_identity.source_identity_ref,
            sourceHash=source_identity.source_hash,
            canonicalSourceIrRef=graph_intent.source_binding.canonical_source_ir_ref,
            canonicalSourceIrHash=graph_intent.source_binding.canonical_source_ir_hash,
        )
        result: list[SourceSemanticWitnessV1] = []
        for block in candidate_blocks:
            evidence = evidence_ref_for_block(
                document,
                block,
                source_identity=source_identity,
                evidence_role=(
                    "constructed_hard_negative"
                    if witness_kind == "hard_negative"
                    else "reference_example"
                ),
                created_at=source_identity.created_at,
            )
            provenance = {
                **evidence.provenance,
                "canonical_source_ir_ref": graph_intent.source_binding.canonical_source_ir_ref,
                "canonical_source_ir_hash": graph_intent.source_binding.canonical_source_ir_hash,
                **dict(provenance_overrides.get(block.block_id, {})),
            }
            evidence = evidence.model_copy(update={"provenance": provenance})
            expectation = expectations.get(block.block_id) or self._proposal_expectation(
                block, graph_intent
            )
            review_ref = f"review:{review_set_ref}:{block.block_id[7:31]}"
            review = WitnessReviewRecord.create(
                state="proposed",
                actor_ref=actor_ref,
                review_ref_seed=review_ref,
                decision_evidence_refs=(evidence,),
            )
            witness_seed = {
                "source_identity_ref": source_identity.source_identity_ref,
                "source_block_id": block.block_id,
                "witness_kind": witness_kind,
                "review_set_ref": review_set_ref,
            }
            witness_id = "witness:" + sha256(
                json.dumps(witness_seed, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()[:32]
            text = str(block.content)
            result.append(
                SourceSemanticWitnessV1.create(
                    witness_id=witness_id,
                    organization_id=source_identity.organization_id,
                    workspace_id=source_identity.workspace_id,
                    source_binding=source_binding,
                    witness_kind=witness_kind,
                    discovered_source_role="reference_examples",
                    evidence_refs=(evidence,),
                    source_anchor_refs=(block.block_id,),
                    exact_text_hash="sha256:" + sha256(text.encode("utf-8")).hexdigest(),
                    semantic_expectation=expectation,
                    review=review,
                )
            )
        return tuple(result)

    @staticmethod
    def _proposal_expectation(
        block: SourceBlock, graph_intent: GraphIntentBrief
    ) -> SemanticExpectation:
        preferred = graph_intent.preferred_representation.kind
        representation = preferred if preferred in {"binary_relation", "event", "frame"} else "frame"
        return SemanticExpectation(
            representationKind=representation,
            requiredDimensions=[item.type_id for item in graph_intent.material_dimensions],
            expectedStructure={
                "proposal_status": "requires_human_adjudication",
                "source_block_id": block.block_id,
                "proposed_subject_type": graph_intent.represented_or_governed_subject.type_id,
                "proposed_relationship_type": graph_intent.primary_relationship.type_id,
            },
            allowedVariants=[],
        )

    @staticmethod
    def _validate_inputs(
        *,
        document: CanonicalSourceDocument,
        source_identity: SourceIdentity,
        graph_intent: GraphIntentBrief,
        source_role_evidence: SourceRoleInterpretationEvidenceV1,
    ) -> None:
        if graph_intent.review.state not in {"proposed", "accepted", "edited"}:
            raise ValueError("rejected GraphIntentBrief cannot drive witness discovery")
        binding = graph_intent.source_binding
        expected = (
            source_identity.organization_id,
            source_identity.workspace_id,
            source_identity.source_identity_ref,
            source_identity.source_hash,
        )
        actual = (
            binding.organization_id,
            binding.workspace_id,
            binding.source_identity_ref,
            binding.source_hash,
        )
        if actual != expected:
            raise ValueError("GraphIntentBrief source scope/binding mismatch")
        if document.source_identity_ref != source_identity.source_identity_ref:
            raise ValueError("CanonicalSourceDocument SourceIdentity mismatch")
        locators = [
            locator
            for mapping in document.source_map.mappings
            for locator in mapping.source_locators
        ]
        if not locators or any(
            locator.source_identity_ref != source_identity.source_identity_ref
            or locator.source_hash != source_identity.source_hash
            for locator in locators
        ):
            raise ValueError("CanonicalSourceDocument frozen source binding mismatch")
        if _canonical_ir_hash(document) != binding.canonical_source_ir_hash:
            raise ValueError("CanonicalSourceDocument hash mismatch")
        if (
            source_role_evidence.organization_id != source_identity.organization_id
            or source_role_evidence.workspace_id != source_identity.workspace_id
            or source_role_evidence.source_binding.source_identity_ref
            != source_identity.source_identity_ref
            or source_role_evidence.source_binding.source_hash != source_identity.source_hash
            or source_role_evidence.source_binding.canonical_source_ir_ref
            != binding.canonical_source_ir_ref
            or source_role_evidence.source_binding.canonical_source_ir_hash
            != binding.canonical_source_ir_hash
        ):
            raise ValueError("source-role evidence scope/source binding mismatch")


__all__ = ["SourceSemanticWitnessService"]
