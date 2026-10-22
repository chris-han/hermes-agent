"""Deterministic and review-gated profiling of exact ontology snapshots."""

from __future__ import annotations

from datetime import datetime, timedelta
from hashlib import sha256
import re
from typing import Mapping

from pydantic import BaseModel
from rdflib import BNode, Graph, Literal, RDF, RDFS, URIRef
from rdflib.compare import to_canonical_graph
from rdflib.namespace import OWL, SH

from contracts.ontology_seeds import (
    CapabilityAssessmentV1, CapabilityName, CompetencyMetadataAssessmentV1,
    EvidenceBoundValueV1, IAAnalysisInputV1, IAAnalysisOutputV1,
    OntologyAuthorityV1, OntologyCapabilityProfileV1, OntologyCompetencyProfileV1,
    OntologyIdentityV1, OntologySemanticCoverageV1, OntologySnapshotIdentityV1,
    ProfileProvenanceV1, RepresentationCapabilitiesV1,
    RepresentationCapabilityClaimV1, SemanticCoverageAssessmentV1,
    StructuralEvidenceV1, StructuralTripleV1,
)
from eos.artifact_hashing import content_hash
from prompts.loader import load_repo_prompt

_CAPABILITY_NAMES: tuple[CapabilityName, ...] = (
    "conditional_constraint", "scoped_requirement", "exception_structure",
    "qualification_structure", "precedence_structure", "temporal_constraint",
    "entity_distinction",
)
_CAPABILITY_TERMS: dict[CapabilityName, frozenset[str]] = {
    "conditional_constraint": frozenset(("condition", "conditional", "when")),
    "scoped_requirement": frozenset(("scope", "scoped", "requirement")),
    "exception_structure": frozenset(("exception", "exemption")),
    "qualification_structure": frozenset(("qualification", "eligibility")),
    "precedence_structure": frozenset(("precedence", "priority", "overrides")),
    "temporal_constraint": frozenset(("temporal", "time", "date", "duration")),
    "entity_distinction": frozenset(("disjoint", "distinct", "different")),
}


def _rdf_term(value: object) -> str:
    if isinstance(value, Literal):
        return value.n3()
    if isinstance(value, (URIRef, BNode)):
        return str(value)
    return str(value)


def _canonical_graph(graph: Graph) -> Graph:
    canonical = Graph()
    for triple in to_canonical_graph(graph):
        canonical.add(triple)
    return canonical


def _inventory(graph: Graph) -> tuple[StructuralTripleV1, ...]:
    return tuple(sorted(
        (StructuralTripleV1(subject=_rdf_term(s), predicate=_rdf_term(p), object=_rdf_term(o)) for s, p, o in graph),
        key=lambda item: (item.subject, item.predicate, item.object),
    ))


def graph_content_hash(graph: Graph) -> str:
    material = [item.model_dump(mode="json") for item in _inventory(_canonical_graph(graph))]
    return "sha256:" + content_hash(material)


def _validated(model: BaseModel) -> BaseModel:
    """Re-run all content-binding validators over current serialized state."""
    return type(model).model_validate(model.model_dump(mode="json"))


def _evidence_ref(triple: StructuralTripleV1) -> str:
    return "structural-triple:" + content_hash(triple.model_dump(mode="json"))


class OntologyCapabilityProfileService:
    def __init__(self, *, generated_at: datetime) -> None:
        if generated_at.tzinfo is None or generated_at.utcoffset() != timedelta(0):
            raise ValueError("generated_at must be timezone-aware UTC")
        self._generated_at = generated_at

    def inspect(
        self, *, snapshot: OntologySnapshotIdentityV1, graph: Graph, source_bytes: bytes
    ) -> OntologyCapabilityProfileV1:
        snapshot = _validated(snapshot)  # type: ignore[assignment]
        actual_source_hash = "sha256:" + sha256(source_bytes).hexdigest()
        if actual_source_hash != snapshot.source_content_hash:
            raise ValueError("snapshot source content hash does not match exact source bytes")
        actual_graph_hash = graph_content_hash(graph)
        if actual_graph_hash != snapshot.graph_content_hash:
            raise ValueError("snapshot graph content hash does not match parsed graph")

        canonical = _canonical_graph(graph)
        inventory = _inventory(canonical)
        structural = self._structural_evidence(canonical, inventory)
        capabilities = {
            name: self._deterministic_claim(name, structural, inventory)
            for name in _CAPABILITY_NAMES
        }
        classes = tuple(item for item in inventory if item.predicate == str(RDF.type) and item.object in (str(OWL.Class), str(RDFS.Class)))
        properties = tuple(item for item in inventory if item.predicate == str(RDF.type) and item.object in (str(OWL.ObjectProperty), str(OWL.DatatypeProperty), str(RDF.Property)))
        shapes = tuple(item for item in inventory if item.predicate == str(RDF.type) and item.object == str(SH.NodeShape))
        return OntologyCapabilityProfileV1.create(
            organization_id=snapshot.organization_id, workspace_id=snapshot.workspace_id,
            ontology_ref=snapshot.ontology_ref, version=snapshot.version,
            source_content_hash=snapshot.source_content_hash, graph_content_hash=snapshot.graph_content_hash,
            revision=1, predecessor_profile_ref=None, predecessor_profile_hash=None,
            identity=OntologyIdentityV1(name=snapshot.name, domain_primary=snapshot.domain_primary,
                domain_secondary=snapshot.domain_secondary, provenance=snapshot.provenance, license=snapshot.license),
            structural_inventory=inventory,
            competency_profile=OntologyCompetencyProfileV1(questions=(), intended_tasks=()),
            semantic_coverage=OntologySemanticCoverageV1(
                definition_families=tuple(self._deterministic_value(value, evidence) for value, evidence in (
                    ("entity", classes), ("relationship", properties), ("constraint", shapes)) if evidence),
                concepts=tuple(self._deterministic_value(item.subject, (item,)) for item in classes),
                relation_families=tuple(self._deterministic_value(item.subject, (item,)) for item in properties),
            ),
            representation_capabilities=RepresentationCapabilitiesV1(**capabilities),
            authority=OntologyAuthorityV1(seed_class=snapshot.seed_class,
                authority_role=snapshot.authority_role, governance_state=snapshot.governance_state),
            profile_provenance=ProfileProvenanceV1(method="deterministic", state="proposed",
                generated_at=self._generated_at, changed_claims=()),
        )

    def analysis_input(self, *, profile: OntologyCapabilityProfileV1) -> IAAnalysisInputV1:
        profile = _validated(profile)  # type: ignore[assignment]
        prompt = load_repo_prompt("ontology_capability_profile/ia_analysis_v1.md",
                                  allowed_namespace="ontology_capability_profile")
        return IAAnalysisInputV1.create(
            organization_id=profile.organization_id, workspace_id=profile.workspace_id,
            ontology_ref=profile.ontology_ref, version=profile.version,
            source_content_hash=profile.source_content_hash, graph_content_hash=profile.graph_content_hash,
            base_profile_ref=profile.profile_ref, base_profile_hash=profile.profile_hash,
            structural_inventory=profile.structural_inventory, prompt_asset=prompt.relative_name,
            prompt_sha256="sha256:" + prompt.sha256,
        )

    def apply_ia_analysis(self, *, profile: OntologyCapabilityProfileV1,
                          analysis_input: IAAnalysisInputV1,
                          analysis: IAAnalysisOutputV1) -> OntologyCapabilityProfileV1:
        profile = _validated(profile)  # type: ignore[assignment]
        analysis_input = _validated(analysis_input)  # type: ignore[assignment]
        analysis = _validated(analysis)  # type: ignore[assignment]
        expected = (profile.organization_id, profile.workspace_id, profile.ontology_ref,
                    profile.version, profile.source_content_hash, profile.graph_content_hash,
                    profile.profile_ref, profile.profile_hash, profile.structural_inventory)
        supplied = (analysis_input.organization_id, analysis_input.workspace_id,
                    analysis_input.ontology_ref, analysis_input.version,
                    analysis_input.source_content_hash, analysis_input.graph_content_hash,
                    analysis_input.base_profile_ref, analysis_input.base_profile_hash,
                    analysis_input.structural_inventory)
        if supplied != expected or analysis.analysis_input_hash != analysis_input.analysis_input_hash:
            raise ValueError("IA analysis is not bound to the exact ontology profile input")

        claims = profile.representation_capabilities.model_dump()
        for name, assessment in analysis.assessments.items():
            current = getattr(profile.representation_capabilities, name)
            status = "partial" if assessment.status == "supported" and current.status != "supported" else assessment.status
            claims[name] = RepresentationCapabilityClaimV1(
                status=status, review_state="proposed", structural_evidence=current.structural_evidence,
                assessment_evidence_refs=assessment.evidence_refs,
                semantic_rationale=assessment.rationale, claim_method="hybrid")
        return self._successor(
            profile, capabilities=RepresentationCapabilitiesV1(**claims),
            competency=self._merge_competency(
                profile.competency_profile, analysis.competency, "proposed", "ia_skill"
            ),
            coverage=self._merge_coverage(
                profile.semantic_coverage, analysis.semantic_coverage, "proposed", "ia_skill"
            ),
            provenance=ProfileProvenanceV1(
                method="hybrid", state="proposed", generated_at=analysis.completed_at,
                prompt_asset=analysis_input.prompt_asset, prompt_sha256=analysis_input.prompt_sha256,
                analyst_ref=analysis.analyst_ref, analysis_input_hash=analysis_input.analysis_input_hash,
                analysis_output_hash=analysis.analysis_hash,
                analysis_input_envelope=analysis_input.model_dump(mode="json"),
                analysis_output_envelope=analysis.model_dump(mode="json"),
                changed_claims=tuple(sorted(analysis.assessments))),
        )

    def review(self, *, profile: OntologyCapabilityProfileV1,
               reviewer_identity_ref: str,
               assessments: Mapping[CapabilityName, CapabilityAssessmentV1],
               reviewed_at: datetime,
               competency: CompetencyMetadataAssessmentV1 | None = None,
               semantic_coverage: SemanticCoverageAssessmentV1 | None = None) -> OntologyCapabilityProfileV1:
        profile = _validated(profile)  # type: ignore[assignment]
        if not reviewer_identity_ref:
            raise ValueError("governed reviewer identity is required")
        claims = profile.representation_capabilities.model_dump()
        for name in _CAPABILITY_NAMES:
            current = getattr(profile.representation_capabilities, name)
            assessment = assessments.get(name)
            claims[name] = RepresentationCapabilityClaimV1(
                status=assessment.status if assessment else current.status,
                review_state="reviewed", structural_evidence=current.structural_evidence,
                assessment_evidence_refs=assessment.evidence_refs if assessment else current.assessment_evidence_refs,
                semantic_rationale=assessment.rationale if assessment else current.semantic_rationale,
                claim_method="human" if assessment else current.claim_method)
        reviewed_competency = self._merge_competency(
            self._mark_competency_reviewed(profile.competency_profile),
            competency, "reviewed", "human",
        )
        reviewed_coverage = self._merge_coverage(
            self._mark_coverage_reviewed(profile.semantic_coverage),
            semantic_coverage, "reviewed", "human",
        )
        return self._successor(
            profile, capabilities=RepresentationCapabilitiesV1(**claims),
            competency=reviewed_competency, coverage=reviewed_coverage,
            provenance=ProfileProvenanceV1(
                method="human", state="reviewed", generated_at=reviewed_at,
                prompt_asset=profile.profile_provenance.prompt_asset,
                prompt_sha256=profile.profile_provenance.prompt_sha256,
                analyst_ref=profile.profile_provenance.analyst_ref,
                analysis_input_hash=profile.profile_provenance.analysis_input_hash,
                analysis_output_hash=profile.profile_provenance.analysis_output_hash,
                based_on_analysis_output_hash=profile.profile_provenance.analysis_output_hash,
                analysis_input_envelope=profile.profile_provenance.analysis_input_envelope,
                analysis_output_envelope=profile.profile_provenance.analysis_output_envelope,
                reviewer_identity_ref=reviewer_identity_ref, reviewed_at=reviewed_at,
                changed_claims=_CAPABILITY_NAMES),
        )

    def _successor(self, profile: OntologyCapabilityProfileV1, *,
                   capabilities: RepresentationCapabilitiesV1,
                   competency: OntologyCompetencyProfileV1,
                   coverage: OntologySemanticCoverageV1,
                   provenance: ProfileProvenanceV1) -> OntologyCapabilityProfileV1:
        values = profile.model_dump(exclude={"schema_version", "profile_ref", "profile_hash"})
        values.update(revision=profile.revision + 1, predecessor_profile_ref=profile.profile_ref,
                      predecessor_profile_hash=profile.profile_hash,
                      representation_capabilities=capabilities, competency_profile=competency,
                      semantic_coverage=coverage, profile_provenance=provenance)
        return OntologyCapabilityProfileV1.create(**values)

    @staticmethod
    def _deterministic_value(value: str, triples: tuple[StructuralTripleV1, ...]) -> EvidenceBoundValueV1:
        return EvidenceBoundValueV1(value=value, evidence_refs=tuple(_evidence_ref(item) for item in triples),
                                    review_state="proposed", claim_method="deterministic")

    @staticmethod
    def _assessment_value(value: object, state: str, method: str) -> EvidenceBoundValueV1:
        return EvidenceBoundValueV1(value=value.value, evidence_refs=value.evidence_refs,
                                    review_state=state, claim_method=method)

    def _merge_competency(
        self,
        base: OntologyCompetencyProfileV1,
        value: CompetencyMetadataAssessmentV1 | None,
        state: str,
        method: str,
    ) -> OntologyCompetencyProfileV1:
        if value is None:
            return base
        return OntologyCompetencyProfileV1(**{
            name: getattr(base, name) if getattr(value, name) is None else tuple(
                self._assessment_value(item, state, method) for item in getattr(value, name)
            )
            for name in ("questions", "intended_tasks")
        })

    def _merge_coverage(
        self,
        base: OntologySemanticCoverageV1,
        value: SemanticCoverageAssessmentV1 | None,
        state: str,
        method: str,
    ) -> OntologySemanticCoverageV1:
        if value is None:
            return base
        return OntologySemanticCoverageV1(**{
            name: getattr(base, name) if getattr(value, name) is None else tuple(
                self._assessment_value(item, state, method) for item in getattr(value, name)
            )
            for name in ("definition_families", "concepts", "relation_families")
        })

    @staticmethod
    def _mark_competency_reviewed(value: OntologyCompetencyProfileV1) -> OntologyCompetencyProfileV1:
        return OntologyCompetencyProfileV1(**{
            name: tuple(item.model_copy(update={"review_state": "reviewed"}) for item in getattr(value, name))
            for name in ("questions", "intended_tasks")})

    @staticmethod
    def _mark_coverage_reviewed(value: OntologySemanticCoverageV1) -> OntologySemanticCoverageV1:
        return OntologySemanticCoverageV1(**{
            name: tuple(item.model_copy(update={"review_state": "reviewed"}) for item in getattr(value, name))
            for name in ("definition_families", "concepts", "relation_families")})

    @staticmethod
    def _structural_evidence(graph: Graph, inventory: tuple[StructuralTripleV1, ...]) -> tuple[StructuralEvidenceV1, ...]:
        by_subject: dict[str, list[StructuralTripleV1]] = {}
        for item in inventory:
            by_subject.setdefault(item.subject, []).append(item)
        evidence: list[StructuralEvidenceV1] = []
        for item in inventory:
            if item.predicate == str(RDF.type) and item.object in (str(OWL.Class), str(RDFS.Class)):
                evidence.append(StructuralEvidenceV1(evidence_kind="class_declaration", triples=(item,)))
            elif item.predicate == str(RDF.type) and item.object in (str(OWL.ObjectProperty), str(OWL.DatatypeProperty), str(RDF.Property)):
                related = tuple(x for x in by_subject[item.subject] if x.predicate in (str(RDF.type), str(RDFS.domain), str(RDFS.range)))
                evidence.append(StructuralEvidenceV1(evidence_kind="property_signature", triples=related))
            elif item.predicate == str(OWL.disjointWith):
                evidence.append(StructuralEvidenceV1(evidence_kind="owl_disjointness", triples=(item,)))
        for shape in graph.subjects(RDF.type, SH.NodeShape):
            shape_id = _rdf_term(shape)
            linked = [x for x in by_subject.get(shape_id, ())]
            for property_node in graph.objects(shape, SH.property):
                linked.extend(by_subject.get(_rdf_term(property_node), ()))
            if linked:
                evidence.append(StructuralEvidenceV1(evidence_kind="shacl_property_constraint",
                                                     triples=tuple(sorted(set(linked), key=lambda x: (x.subject, x.predicate, x.object)))))
        return tuple(sorted(evidence, key=lambda x: (x.evidence_kind, tuple((t.subject, t.predicate, t.object) for t in x.triples))))

    @staticmethod
    def _tokens(evidence: StructuralEvidenceV1) -> frozenset[str]:
        words: set[str] = set()
        for triple in evidence.triples:
            for value in (triple.subject, triple.predicate, triple.object):
                expanded = re.sub(r"([a-z])([A-Z])", r"\1 \2", value)
                words.update(part.lower() for part in re.split(r"[^A-Za-z0-9]+", expanded) if part)
        return frozenset(words)

    @classmethod
    def _deterministic_claim(cls, name: CapabilityName,
                             evidence: tuple[StructuralEvidenceV1, ...],
                             inventory: tuple[StructuralTripleV1, ...]) -> RepresentationCapabilityClaimV1:
        if name == "entity_distinction":
            matched = tuple(item for item in evidence if item.evidence_kind == "owl_disjointness")
        else:
            matched = tuple(item for item in evidence if cls._tokens(item) & _CAPABILITY_TERMS[name])
        if not matched:
            matched = (StructuralEvidenceV1(evidence_kind="graph_inspection", triples=(),
                       note=f"inspected {len(inventory)} canonical triples; no structural clue for {name}"),)
        return RepresentationCapabilityClaimV1(
            status="partial" if any(item.evidence_kind != "graph_inspection" for item in matched) else "unsupported",
            review_state="proposed", structural_evidence=matched, assessment_evidence_refs=(),
            semantic_rationale=None, claim_method="deterministic")


__all__ = ["OntologyCapabilityProfileService", "graph_content_hash"]
