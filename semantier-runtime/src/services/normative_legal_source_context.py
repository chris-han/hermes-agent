"""Retain exact reviewed legal sections for producer attention and offline replay."""
import json
from hashlib import sha256
from pathlib import Path

from contracts.normative_schema import SourceSearchCoverageManifestV1
from contracts.ontology_certification import NormativeAuthorityDependencySetV1
from eos.artifact_hashing import content_hash
from eos.legal_corpus_store import LegalCorpusStore
from services.ontology_certification_successor_service import OntologyCertificationSuccessorService


LEGAL_SOURCE_CONTEXT_HASH = sha256(Path(__file__).read_bytes()).hexdigest()


def _verify_context(context, scope, boundary):
    authority = NormativeAuthorityDependencySetV1.model_validate(context["authority"])
    coverage = SourceSearchCoverageManifestV1.model_validate(context["coverage"])
    if ((authority.organization_id, authority.workspace_id) != scope
        or (coverage.organization_id, coverage.workspace_id) != scope):
        raise ValueError("NORMATIVE_LEGAL_SOURCE_SCOPE_MISMATCH")
    if (authority.boundary_ref, authority.boundary_hash) != boundary:
        raise ValueError("NORMATIVE_LEGAL_SOURCE_BOUNDARY_MISMATCH")
    if (authority.search_coverage_ref, authority.search_coverage_hash) != (coverage.coverage_ref, coverage.coverage_hash):
        raise ValueError("NORMATIVE_LEGAL_SOURCE_COVERAGE_MISMATCH")
    if coverage.config_hash != OntologyCertificationSuccessorService._legal_coverage_config_hash(authority.dependencies):
        raise ValueError("NORMATIVE_LEGAL_SOURCE_COVERAGE_SELECTION_MISMATCH")
    expected = [(d, s) for d in authority.dependencies for s in d.sections]
    if len(expected) != len(context["sections"]):
        raise ValueError("NORMATIVE_LEGAL_SOURCE_SECTION_COUNT_MISMATCH")
    refs = []
    for (dependency, section), row in zip(expected, context["sections"]):
        pin = dict(source_id=dependency.source_id, version_id=dependency.version_id,
            artifact_id=dependency.artifact_id, artifact_hash=dependency.artifact_hash,
            **section.model_dump(mode="json"))
        if {key: row.get(key) for key in pin} != pin:
            raise ValueError("NORMATIVE_LEGAL_SOURCE_SECTION_PIN_MISMATCH")
        if content_hash({"text": row["text"]}) != section.text_hash:
            raise ValueError("NORMATIVE_LEGAL_SOURCE_TEXT_MISMATCH")
        refs.append("legal-section:" + section.section_id)
    if not set(refs).issubset(coverage.result_refs):
        raise ValueError("NORMATIVE_LEGAL_SOURCE_COVERAGE_MISSING_SECTIONS")
    if context.get("project_applicability") != "UNKNOWN":
        raise ValueError("NORMATIVE_LEGAL_SOURCE_PROJECT_APPLICABILITY_NOT_ESTABLISHED")
    return context


def retain_legal_source_context(*, store, scope, boundary, authority_pin, coverage_pin, legal_store):
    if authority_pin.artifact_kind != "normative_authority_dependency_set":
        raise ValueError("NORMATIVE_LEGAL_SOURCE_AUTHORITY_KIND_MISMATCH")
    if coverage_pin.artifact_kind != "source_search_coverage_manifest":
        raise ValueError("NORMATIVE_LEGAL_SOURCE_COVERAGE_KIND_MISMATCH")
    authority = store.get(**scope, artifact_ref=authority_pin.artifact_ref, contract_type=NormativeAuthorityDependencySetV1)
    coverage = store.get(**scope, artifact_ref=coverage_pin.artifact_ref, contract_type=SourceSearchCoverageManifestV1)
    if authority is None or authority.set_hash != authority_pin.artifact_hash:
        raise ValueError("NORMATIVE_LEGAL_SOURCE_AUTHORITY_PIN_MISMATCH")
    if coverage is None or coverage.coverage_hash != coverage_pin.artifact_hash:
        raise ValueError("NORMATIVE_LEGAL_SOURCE_COVERAGE_PIN_MISMATCH")
    # Reject stale review/boundary before reading legal rows or invoking a provider.
    context = dict(authority=authority.model_dump(mode="json"), coverage=coverage.model_dump(mode="json"),
        sections=[dict(source_id=d.source_id, version_id=d.version_id, artifact_id=d.artifact_id,
            artifact_hash=d.artifact_hash, **s.model_dump(mode="json"), text="")
            for d in authority.dependencies for s in d.sections], project_applicability="UNKNOWN")
    if (authority.search_coverage_ref, authority.search_coverage_hash) != (coverage.coverage_ref, coverage.coverage_hash):
        raise ValueError("NORMATIVE_LEGAL_SOURCE_COVERAGE_MISMATCH")
    if not isinstance(legal_store, LegalCorpusStore):
        raise TypeError("parent source input requires production LegalCorpusStore")
    owner = OntologyCertificationSuccessorService()
    owner.verify_authority_dependencies(authority=authority, legal_store=legal_store)
    rows = []
    for d in authority.dependencies:
        evidence = legal_store.resolve_certification_source_evidence(**scope,
            source_id=d.source_id, version_id=d.version_id, artifact_id=d.artifact_id,
            artifact_hash=d.artifact_hash, section_ids=tuple(s.section_id for s in d.sections))
        for s, raw in zip(d.sections, evidence["sections"]):
            text = json.loads(raw["anchor_json"]).get("retained_text")
            if not isinstance(text, str) or not text:
                raise ValueError("NORMATIVE_LEGAL_SOURCE_TEXT_REQUIRED")
            rows.append(dict(source_id=d.source_id, version_id=d.version_id, artifact_id=d.artifact_id,
                artifact_hash=d.artifact_hash, **s.model_dump(mode="json"), text=text))
    context["sections"] = rows
    return _verify_context(context, (scope["organization_id"], scope["workspace_id"]), boundary)


def legal_source_evidence(source):
    context = source.get("legal_source_context")
    if context is None:
        return []
    boundary = source["boundary"]
    _verify_context(context, (source["organization_id"], source["workspace_id"]),
        (boundary["boundary_ref"], boundary["boundary_hash"]))
    return [dict(evidenceRef="legal-section:" + row["section_id"],
        provenance=dict(source_block_id="legal-section:" + row["section_id"]))
        for row in context["sections"]]
