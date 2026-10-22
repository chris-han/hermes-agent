"""Derived, read-only Decision Basis projection for saved Tender findings.

This service does not create semantic authority. It projects only identities and
source occurrences already frozen into a saved inspection episode.
"""
from __future__ import annotations

from collections.abc import Mapping

from contracts.evidence_location import EvidenceRef
from contracts.canonical_source_ir import CanonicalSourceDocument
from contracts.ontology_building import ArtifactPinV1
from contracts.tender_decision_basis import (
    TenderDecisionBasisAffectedFindingV1,
    TenderDecisionBasisEdgeV1,
    TenderDecisionBasisNodeV1,
    TenderDecisionBasisSourceOccurrenceV1,
    TenderDecisionBasisV1,
)
from contracts.tender_journey import TenderJourneyInspectionEpisodeV1
from eos.artifact_hashing import content_hash
from semantier.knowledge_feature_packs.evidence_selector_builder import evidence_ref_pin
from semantier.knowledge_feature_packs.evidence_resolver import resolve_evidence_ref


class TenderDecisionBasisService:
    @staticmethod
    def _recorded_input_edge_ref(*, source_ref: str, finding_hash: str) -> str:
        return "recorded-input:" + content_hash({
            "source": source_ref,
            "target": "finding:" + finding_hash,
            "finding": finding_hash,
        })

    @staticmethod
    def _occurrence_ref(*, evidence: EvidenceRef, canonical_ir_pin: ArtifactPinV1) -> tuple[str | None, str | None]:
        source_block_ref = evidence.provenance.get("source_block_id")
        if not isinstance(source_block_ref, str) or not source_block_ref:
            return None, None
        pin = evidence_ref_pin(evidence)
        occurrence_ref = "evidence-occurrence:" + content_hash(
            {
                "source_identity_ref": evidence.source_identity_ref,
                "canonical_source_ir_pin": canonical_ir_pin.model_dump(mode="json"),
                "evidence_pin": pin.model_dump(mode="json"),
                "source_block_id": source_block_ref,
            }
        )
        return occurrence_ref, source_block_ref

    def project(
        self,
        *,
        episode: TenderJourneyInspectionEpisodeV1,
        finding_ref: str,
        finding_hash: str,
        source_names: Mapping[str, str] | None = None,
        source_documents: Mapping[str, CanonicalSourceDocument] | None = None,
        recorded_content: Mapping[str, dict] | None = None,
        provision_candidates: tuple[dict, ...] = (),
        candidate_context_status: str | None = None,
    ) -> TenderDecisionBasisV1:
        matches = tuple(
            finding
            for finding in episode.inspection_result.governed_findings
            if finding.finding_id == finding_ref
        )
        if len(matches) != 1:
            raise ValueError("FINDING_NOT_IN_EPISODE")
        finding = matches[0]
        if finding.finding_hash != finding_hash:
            raise ValueError("FINDING_PIN_MISMATCH")

        anchors_by_evidence: dict[str, list[tuple[object, object]]] = {}
        for region in episode.inspection_context.source_regions:
            for anchor in region.evidence:
                anchors_by_evidence.setdefault(
                    anchor.evidence_ref.evidence_ref, []
                ).append((region, anchor))

        occurrences = []
        gaps: list[str] = []
        for frozen in finding.evidence_refs:
            evidence_ref = str(frozen["evidenceRef"])
            bound = anchors_by_evidence.get(evidence_ref, [])
            if len(bound) != 1:
                gaps.append(
                    "SOURCE_EVIDENCE_AMBIGUOUS"
                    if len(bound) > 1
                    else "SOURCE_EVIDENCE_UNAVAILABLE"
                )
                evidence = EvidenceRef.model_validate(dict(frozen))
                exact_text = evidence.selector.exact_quote
                if not exact_text:
                    gaps.append("SOURCE_EXACT_TEXT_UNAVAILABLE")
                    continue
                occurrences.append(
                    TenderDecisionBasisSourceOccurrenceV1(
                        evidence_ref=evidence_ref,
                        occurrence_ref=None,
                        status="UNRESOLVED",
                        source_identity_ref=evidence.source_identity_ref,
                        document_name=(source_names or {}).get(evidence.source_identity_ref),
                        source_hash=finding.source_hash,
                        canonical_source_ir_pin=ArtifactPinV1(
                            artifact_kind="canonical_source_ir",
                            artifact_ref=episode.inspection_result.canonical_source_ir_ref,
                            artifact_hash=episode.inspection_result.canonical_source_ir_hash,
                        ),
                        source_region_ref="unresolved:" + evidence_ref,
                        source_block_ref=None,
                        media_type=None,
                        evidence_role=evidence.evidence_role,
                        selector_hash=evidence.selector_hash,
                        exact_text=exact_text,
                    )
                )
                continue

            region, anchor = bound[0]
            evidence = anchor.evidence_ref
            occurrence_ref, source_block_ref = self._occurrence_ref(
                evidence=evidence,
                canonical_ir_pin=region.source_binding.canonical_source_ir,
            )
            if occurrence_ref is None:
                gaps.append("SOURCE_OCCURRENCE_IDENTITY_UNAVAILABLE")
            exact_text = evidence.selector.exact_quote
            if not exact_text:
                gaps.append("SOURCE_EXACT_TEXT_UNAVAILABLE")
                continue
            location = None
            document = (source_documents or {}).get(region.source_binding.canonical_source_ir.artifact_ref)
            if document is not None:
                resolved = resolve_evidence_ref(
                    evidence, document,
                    artifact_ref=region.source_binding.canonical_source_ir.artifact_ref,
                    selector_hash=evidence.selector_hash,
                )
                if resolved.status in ("exact", "relocated_exact") and resolved.location.get("kind") in (
                    "docx_structural_range", "pdf_page_rect", "text_fallback"
                ):
                    location = resolved.location
            occurrences.append(
                TenderDecisionBasisSourceOccurrenceV1(
                    evidence_ref=evidence.evidence_ref,
                    occurrence_ref=occurrence_ref,
                    status="EXACT" if occurrence_ref is not None else "UNRESOLVED",
                    source_identity_ref=evidence.source_identity_ref,
                    document_name=(source_names or {}).get(evidence.source_identity_ref),
                    source_hash=region.source_binding.source_hash,
                    canonical_source_ir_pin=region.source_binding.canonical_source_ir,
                    source_region_ref=region.source_region_ref,
                    source_block_ref=source_block_ref,
                    media_type=anchor.locator.media_type,
                    evidence_role=evidence.evidence_role,
                    selector_hash=evidence.selector_hash,
                    exact_text=exact_text,
                    presentation_location=location,
                )
            )

        ordered_gaps = tuple(dict.fromkeys(gaps))
        finding_node = "finding:" + finding.finding_hash
        nodes = [TenderDecisionBasisNodeV1(
            node_ref=finding_node, layer="DECISION", label="Recorded finding",
        )]
        edges = []
        selected_evidence = {str(item["evidenceRef"]) for item in finding.evidence_refs}
        occurrences_by_ref = {item.evidence_ref: item for item in occurrences}
        for evidence_ref in sorted(selected_evidence):
            occurrence = occurrences_by_ref.get(evidence_ref)
            node_ref = "evidence:" + evidence_ref
            nodes.append(TenderDecisionBasisNodeV1(
                node_ref=node_ref, layer="EVIDENCE",
                label=occurrence.exact_text if occurrence else "Recorded evidence; exact excerpt unavailable",
            ))
            edges.append(TenderDecisionBasisEdgeV1(
                edge_ref=self._recorded_input_edge_ref(source_ref=node_ref, finding_hash=finding.finding_hash),
                source_ref=node_ref, target_ref=finding_node,
                basis_finding_hash=finding.finding_hash,
            ))
        pinned_inputs = (
            ("MEANING", "Recorded interpretation", finding.observed_hypothesis_ref, finding.observed_hypothesis_hash, "observed_constraint_hypothesis"),
            ("STRUCTURE", "Frame match", finding.frame_match_result_ref, finding.frame_match_result_hash, "frame_match_result"),
            ("STRUCTURE", "Comparison pattern", finding.canonical_pattern_ref, finding.canonical_pattern_hash, "canonical_pattern"),
            ("STRUCTURE", "Accepted graph release", finding.accepted_graph_release_ref, finding.accepted_graph_release_hash, "accepted_graph_release"),
            ("STRUCTURE", "Governed rule", finding.governed_rule_version_ref, finding.governed_rule_version_hash, "governed_rule_version"),
        )
        for layer, label, ref, digest, kind in pinned_inputs:
            pin = ArtifactPinV1(artifact_kind=kind, artifact_ref=ref, artifact_hash=digest)
            node_ref = "artifact:" + content_hash(pin.model_dump(mode="json"))
            nodes.append(TenderDecisionBasisNodeV1(
                node_ref=node_ref, layer=layer, label=label, artifact_pin=pin,
                recorded_content=(recorded_content or {}).get(ref),
            ))
            edges.append(TenderDecisionBasisEdgeV1(
                edge_ref=self._recorded_input_edge_ref(source_ref=node_ref, finding_hash=finding.finding_hash),
                source_ref=node_ref, target_ref=finding_node,
                basis_finding_hash=finding.finding_hash,
            ))

        declared = {
            (item.ref, item.hash)
            for item in episode.inspection_result.governed_finding_refs
        }
        actual = {
            (item.finding_id, item.finding_hash)
            for item in episode.inspection_result.governed_findings
        }
        impact_coverage = "COMPLETE" if declared == actual else "PARTIAL"
        affected = []
        for other in episode.inspection_result.governed_findings:
            if other.finding_id == finding.finding_id:
                continue
            shared = tuple(sorted(selected_evidence.intersection(
                str(item["evidenceRef"]) for item in other.evidence_refs
            )))
            if shared:
                affected.append(TenderDecisionBasisAffectedFindingV1(
                    finding_ref=other.finding_id,
                    finding_hash=other.finding_hash,
                    decision_status=other.decision_status,
                    rationale=other.rationale,
                    shared_evidence_refs=shared,
                    dependency_edge_refs=tuple(
                        self._recorded_input_edge_ref(
                            source_ref="evidence:" + evidence_ref,
                            finding_hash=other.finding_hash,
                        ) for evidence_ref in shared
                    ),
                ))
        # Optional proposal assertions must close on exact saved evidence; invalid
        # mappings never hide the recorded Finding or the candidate source.
        requirement_versions: dict[str, set[str]] = {}
        for candidate in provision_candidates:
            for requirement in (candidate.get("applicability_review") or {}).get(
                "requirements", []
            ):
                requirement_versions.setdefault(
                    requirement["requirement_ref"], set()
                ).add(requirement["requirement_hash"])
        validated_candidates = []
        for candidate in provision_candidates:
            review = candidate.get("applicability_review")
            if review is not None:
                valid = all(
                    len(requirement_versions[r["requirement_ref"]]) == 1
                    for r in review.get("requirements", [])
                ) and all(
                    any(
                        occurrence.status == "EXACT"
                        and (
                            observation["evidence_ref"],
                            observation["occurrence_ref"],
                            observation["source_hash"],
                            observation["selector_hash"],
                            observation["exact_text"],
                        )
                        == (
                            occurrence.evidence_ref,
                            occurrence.occurrence_ref,
                            occurrence.source_hash,
                            occurrence.selector_hash,
                            occurrence.exact_text,
                        )
                        for occurrence in occurrences
                    )
                    for requirement in review.get("requirements", [])
                    for observation in requirement.get("observations", [])
                )
                if not valid:
                    candidate = {
                        **candidate,
                        "applicability_review": None,
                        "review_context_status": "INVALID",
                    }
            validated_candidates.append(candidate)
        return TenderDecisionBasisV1(
            journey_id=episode.journey.journey_id,
            transition_id=episode.transition_id,
            inspection_result_ref=episode.inspection_result.result_ref,
            inspection_result_hash=episode.inspection_result.result_hash,
            finding_ref=finding.finding_id,
            finding_hash=finding.finding_hash,
            decision_status=finding.decision_status,
            rationale=finding.rationale,
            structural_match=finding.structural_match,
            semantic_relation=finding.semantic_relation,
            frame_match_state=finding.frame_match_state,
            hypothesis_pin=ArtifactPinV1(
                artifact_kind="observed_constraint_hypothesis",
                artifact_ref=finding.observed_hypothesis_ref,
                artifact_hash=finding.observed_hypothesis_hash,
            ),
            frame_match_pin=ArtifactPinV1(
                artifact_kind="frame_match_result",
                artifact_ref=finding.frame_match_result_ref,
                artifact_hash=finding.frame_match_result_hash,
            ),
            canonical_pattern_pin=ArtifactPinV1(
                artifact_kind="canonical_pattern",
                artifact_ref=finding.canonical_pattern_ref,
                artifact_hash=finding.canonical_pattern_hash,
            ),
            accepted_graph_release_pin=ArtifactPinV1(
                artifact_kind="accepted_graph_release",
                artifact_ref=finding.accepted_graph_release_ref,
                artifact_hash=finding.accepted_graph_release_hash,
            ),
            governed_rule_pin=ArtifactPinV1(
                artifact_kind="governed_rule_version",
                artifact_ref=finding.governed_rule_version_ref,
                artifact_hash=finding.governed_rule_version_hash,
            ),
            source_occurrences=tuple(occurrences),
            nodes=tuple(nodes),
            edges=tuple(edges),
            affected_findings=tuple(sorted(affected, key=lambda item: item.finding_ref)),
            impact_coverage=impact_coverage,
            completeness="COMPLETE" if not ordered_gaps else "PARTIAL",
            gaps=ordered_gaps,
            provision_candidates=tuple(validated_candidates),
            candidate_context_status=candidate_context_status or ("AVAILABLE" if provision_candidates else "UNAVAILABLE"),
        )


__all__ = ["TenderDecisionBasisService"]
