"""Pure retained verification for source-first build-family artifacts."""

from __future__ import annotations

import base64
from hashlib import sha256
import json
from typing import Any, Mapping

from contracts.canonical_source_ir import CanonicalSourceDocument
from contracts.evidence_location import SourceIdentity
from contracts.graph_intent_brief import GraphIntentBrief
from contracts.ontology_artifact_integrity import (
    ExploratoryGraphSnapshotIntegrityV1,
    verify_exploratory_snapshot_integrity,
)
from contracts.ontology_build_inputs import OntologyBuildInputEnvelopeV1
from contracts.ontology_building import (
    ArtifactPinV1,
    CanonicalEvidenceAdequacyAssessmentV1,
    CanonicalEvidenceBoundaryV1,
    ExploratoryGraphEdgeV1,
    ExploratoryGraphNodeV1,
    ExploratoryGraphSnapshotV1,
    OntologyBuildManifestV1,
    OntologyOperation,
)
from contracts.runtime_context_reconstruction import CapturedJsonV1, CapturedSourceV1
from eos.artifact_hashing import content_hash

from .common import (
    PinKey,
    _expect_dependencies,
    _expect_identity,
    _pin_key,
    _payload,
    _require_supplied_source,
    _strict_base64,
    _validate_closure,
)
from .source import _matching_source_member_capture
from .trial_build import (
    _verify_trial_projection_integrity,
    _verify_trial_projection_manifest,
    _verify_trial_projection_snapshot,
)


_SOURCE_FIRST_COMPILER_VERSION = "semantier_source_first_exploratory_v1"


_SOURCE_FIRST_PROFILE = "source_first_build.v1"


def _validate_source_first_origin_values(
    *,
    inputs: OntologyBuildInputEnvelopeV1,
    boundary: CanonicalEvidenceBoundaryV1,
    assessment: CanonicalEvidenceAdequacyAssessmentV1,
    snapshot: ExploratoryGraphSnapshotV1,
    integrity: ExploratoryGraphSnapshotIntegrityV1,
    manifest: OntologyBuildManifestV1,
    brief: GraphIntentBrief,
    identity: SourceIdentity,
    canonical: CanonicalSourceDocument,
    compiler_payload: Mapping[str, object],
    source_capture: CapturedJsonV1,
) -> None:
    """Validate the typed source-first origin used by live and retained owners."""

    expected_kinds = (
        "canonical_evidence_adequacy_assessment",
        "ontology_build_inputs",
        "canonical_evidence_boundary",
        "exploratory_graph_snapshot",
        "graph_intent_brief",
        "source_identity",
        "canonical_source_ir",
        "ontology_source_first_compiler",
    )
    children = {pin.artifact_kind: pin for pin in manifest.child_artifacts}
    identity_payload = identity.model_dump(
        mode="json", by_alias=True, exclude_none=False
    )
    canonical_payload = canonical.model_dump(
        mode="json", by_alias=True, exclude_none=False
    )
    expected_compiler = _source_first_compiler_payload(inputs)
    capture = source_capture.model_dump(mode="json")
    capture_payload = json.loads(source_capture.canonical_json)
    _validate_source_first_boundary_identity(boundary=boundary)
    _validate_source_first_boundary_relation(
        boundary=boundary, identity=identity, canonical=canonical
    )
    _validate_source_first_assessment_identity(assessment=assessment)
    _validate_source_first_assessment_relation(assessment=assessment, boundary=boundary)
    _validate_source_first_snapshot_shape(snapshot=snapshot)
    _validate_source_first_snapshot_binding(
        snapshot=snapshot, brief=brief, identity=identity, canonical=canonical
    )
    _validate_source_first_snapshot_projection(
        snapshot=snapshot, brief=brief, canonical=canonical
    )
    _validate_source_first_snapshot_integrity(snapshot=snapshot, integrity=integrity)
    if (
        tuple(children) != expected_kinds
        or manifest.assurance_profile != "A0_EXPLORATORY"
        or manifest.predecessor_build_ref is not None
        or manifest.predecessor_build_hash is not None
        or inputs.build_ref != manifest.build_ref
        or snapshot.build_ref != manifest.build_ref
        or brief.as_dict() != inputs.graph_intent
        or brief.review.state != "accepted"
        or not brief.compiler_eligible
        or inputs.source_identity != identity_payload
        or inputs.canonical_source_ir != canonical_payload
        or dict(compiler_payload) != expected_compiler
        or children["canonical_evidence_adequacy_assessment"]
        != ArtifactPinV1(
            artifact_kind="canonical_evidence_adequacy_assessment",
            artifact_ref=assessment.assessment_ref,
            artifact_hash=assessment.assessment_hash,
        )
        or children["ontology_build_inputs"]
        != ArtifactPinV1(
            artifact_kind="ontology_build_inputs",
            artifact_ref=inputs.envelope_ref,
            artifact_hash=inputs.envelope_hash,
        )
        or children["canonical_evidence_boundary"]
        != ArtifactPinV1(
            artifact_kind="canonical_evidence_boundary",
            artifact_ref=boundary.boundary_ref,
            artifact_hash=boundary.boundary_hash,
        )
        or children["exploratory_graph_snapshot"]
        != ArtifactPinV1(
            artifact_kind="exploratory_graph_snapshot",
            artifact_ref=snapshot.snapshot_ref,
            artifact_hash=snapshot.snapshot_hash,
        )
        or children["graph_intent_brief"]
        != ArtifactPinV1(
            artifact_kind="graph_intent_brief",
            artifact_ref=brief.graph_intent_id,
            artifact_hash=brief.graph_intent_hash,
        )
        or children["source_identity"].artifact_ref != identity.source_identity_ref
        or children["source_identity"].artifact_hash
        != "sha256:" + content_hash(identity_payload)
        or children["canonical_source_ir"]
        != ArtifactPinV1(
            artifact_kind="canonical_source_ir",
            artifact_ref=brief.source_binding.canonical_source_ir_ref,
            artifact_hash=brief.source_binding.canonical_source_ir_hash,
        )
        or children["ontology_source_first_compiler"].artifact_ref
        != inputs.compiler_version
        or children["ontology_source_first_compiler"].artifact_hash
        != "sha256:"
        + sha256(
            _strict_base64(inputs.compiler_bytes_base64, "source-first compiler bytes")
        ).hexdigest()
        or (boundary.organization_id, boundary.workspace_id)
        != (manifest.organization_id, manifest.workspace_id)
        or (assessment.organization_id, assessment.workspace_id)
        != (manifest.organization_id, manifest.workspace_id)
        or (snapshot.organization_id, snapshot.workspace_id)
        != (manifest.organization_id, manifest.workspace_id)
        or (identity.organization_id, identity.workspace_id)
        != (manifest.organization_id, manifest.workspace_id)
        or (assessment.boundary_ref, assessment.boundary_hash)
        != (boundary.boundary_ref, boundary.boundary_hash)
        or capture_payload.get("identity") != identity_payload
        or capture_payload.get("canonical_source_ir") != canonical_payload
        or capture_payload.get("source_bytes_base64") != inputs.source_bytes_base64
        or capture.get("bytes_hash")
        != "sha256:" + sha256(source_capture.canonical_json.encode("utf-8")).hexdigest()
    ):
        raise ValueError("source-first typed origin relationship mismatch")
    compiler_raw = _strict_base64(
        inputs.compiler_bytes_base64, "source-first compiler bytes"
    )
    compiler_pin = ArtifactPinV1(
        artifact_kind="ontology_source_first_compiler",
        artifact_ref=inputs.compiler_version,
        artifact_hash="sha256:" + sha256(compiler_raw).hexdigest(),
    )
    _validate_source_first_manifest_values(
        manifest=manifest,
        assessment=assessment,
        inputs=inputs,
        boundary=boundary,
        snapshot=snapshot,
        brief=brief,
        identity=identity,
        canonical=canonical,
        compiler_pin=compiler_pin,
    )


def _native_contract(
    *,
    source: CapturedSourceV1,
    closure: Mapping[PinKey, CapturedSourceV1],
    contract: type[Any],
    kind: str,
    ref_field: str,
    hash_field: str,
) -> tuple[CapturedSourceV1, Any]:
    source, payload = _payload(_require_supplied_source(source, closure))
    value = contract.model_validate(payload)
    if value.model_dump(mode="json") != payload:
        raise ValueError(f"retained {kind} encoding mismatch")
    _expect_identity(
        source,
        kind=kind,
        ref=getattr(value, ref_field),
        digest=getattr(value, hash_field),
    )
    return source, value


def _resolve_pin(
    closure: Mapping[PinKey, CapturedSourceV1],
    pin: ArtifactPinV1,
) -> CapturedSourceV1:
    resolved = closure.get(_pin_key(pin))
    if resolved is None:
        raise ValueError("retained native dependency is missing")
    return resolved


def _unique_ref(
    closure: Mapping[PinKey, CapturedSourceV1], *, kind: str, ref: str
) -> CapturedSourceV1:
    matches = [
        candidate
        for candidate in closure.values()
        if candidate.source.artifact_kind == kind
        and candidate.source.artifact_ref == ref
    ]
    if len(matches) != 1:
        raise ValueError(f"{kind} native ref must resolve exactly once")
    return matches[0]


def _model_payload(source: CapturedSourceV1, contract: type[Any], label: str) -> Any:
    source, payload = _payload(source)
    value = contract.model_validate(payload)
    by_alias = contract in {SourceIdentity, CanonicalSourceDocument}
    if value.model_dump(mode="json", by_alias=by_alias, exclude_none=False) != payload:
        raise ValueError(f"retained {label} encoding mismatch")
    return value


def _source_first_compiler_payload(
    inputs: OntologyBuildInputEnvelopeV1,
) -> dict[str, str]:
    raw = _strict_base64(inputs.compiler_bytes_base64, "source-first compiler bytes")
    if inputs.compiler_version != _SOURCE_FIRST_COMPILER_VERSION:
        raise ValueError("unsupported source-first compiler version")
    return {
        "compiler_version": inputs.compiler_version,
        "source_bytes_base64": base64.b64encode(raw).decode("ascii"),
    }


def _validate_source_first_boundary_identity(
    *, boundary: CanonicalEvidenceBoundaryV1
) -> None:
    basis = {
        "organization_id": boundary.organization_id,
        "workspace_id": boundary.workspace_id,
        "source_identity_ref": boundary.source_identity_ref,
        "source_hash": boundary.source_hash,
        "canonical_source_ir_ref": boundary.canonical_source_ir_ref,
        "canonical_source_ir_hash": boundary.canonical_source_ir_hash,
        "intended_domain": boundary.intended_domain,
        "intended_use": boundary.intended_use,
        "target_time": boundary.target_time.isoformat(),
        "jurisdiction": boundary.jurisdiction,
        "explicit_exclusions": boundary.explicit_exclusions,
    }
    digest = content_hash(basis)
    if (
        boundary.boundary_ref != f"canonical-boundary:{digest}"
        or boundary.boundary_hash != f"sha256:{digest}"
        or boundary.target_time != boundary.created_at
        or boundary.jurisdiction != "unresolved"
        or boundary.explicit_exclusions != ("certification", "admission", "activation")
    ):
        raise ValueError("canonical evidence boundary native producer mismatch")


def _validate_source_first_boundary_relation(
    *,
    boundary: CanonicalEvidenceBoundaryV1,
    identity: SourceIdentity,
    canonical: CanonicalSourceDocument,
) -> None:
    if (
        (identity.organization_id, identity.workspace_id)
        != (boundary.organization_id, boundary.workspace_id)
        or identity.source_hash != boundary.source_hash
        or canonical.source_identity_ref != boundary.source_identity_ref
    ):
        raise ValueError("canonical evidence boundary source binding mismatch")


def _validate_source_first_assessment_identity(
    *, assessment: CanonicalEvidenceAdequacyAssessmentV1
) -> None:
    basis = {
        "organization_id": assessment.organization_id,
        "workspace_id": assessment.workspace_id,
        "boundary_ref": assessment.boundary_ref,
        "boundary_hash": assessment.boundary_hash,
        "findings": [item.model_dump(mode="json") for item in assessment.findings],
    }
    digest = content_hash(basis)
    if (
        assessment.assessment_ref != f"evidence-adequacy:{digest}"
        or assessment.assessment_hash != f"sha256:{digest}"
    ):
        raise ValueError("canonical evidence adequacy native identity mismatch")


def _validate_source_first_assessment_relation(
    *,
    assessment: CanonicalEvidenceAdequacyAssessmentV1,
    boundary: CanonicalEvidenceBoundaryV1,
) -> None:
    if (assessment.organization_id, assessment.workspace_id) != (
        boundary.organization_id,
        boundary.workspace_id,
    ) or assessment.assessed_at != boundary.created_at:
        raise ValueError("canonical evidence adequacy boundary mismatch")
    expected_regions = ("effectivity", "exception", "qualification", "rule", "scope")
    findings = list(assessment.findings)
    authority = []
    if findings and findings[0].finding_id == "adequacy:unresolved-authority-role":
        authority = [findings.pop(0)]
    if authority:
        finding = authority[0]
        if finding.model_dump(mode="json") != {
            "finding_id": "adequacy:unresolved-authority-role",
            "finding_type": "unresolved_source_authority_role",
            "severity": "blocking",
            "message": "The canonical source authority role is unresolved.",
            "evidence_refs": [boundary.source_identity_ref],
            "search_evidence_refs": [boundary.canonical_source_ir_ref],
        }:
            raise ValueError("canonical evidence authority finding mismatch")
    if tuple(finding.finding_id for finding in findings) != tuple(
        f"adequacy:missing-definition:{region}" for region in expected_regions
    ):
        raise ValueError("canonical evidence critical-region findings mismatch")
    for finding, region in zip(findings, expected_regions, strict=True):
        if finding.model_dump(mode="json") != {
            "finding_id": f"adequacy:missing-definition:{region}",
            "finding_type": "missing_critical_definition_region",
            "severity": "blocking",
            "message": f"The canonical source has no evidenced {region} region.",
            "evidence_refs": [boundary.canonical_source_ir_ref],
            "search_evidence_refs": [boundary.canonical_source_ir_ref],
        }:
            raise ValueError("canonical evidence critical-region finding mismatch")


def _validate_source_first_manifest_values(
    *,
    manifest: OntologyBuildManifestV1,
    assessment: CanonicalEvidenceAdequacyAssessmentV1,
    inputs: OntologyBuildInputEnvelopeV1,
    boundary: CanonicalEvidenceBoundaryV1,
    snapshot: ExploratoryGraphSnapshotV1,
    brief: GraphIntentBrief,
    identity: SourceIdentity,
    canonical: CanonicalSourceDocument,
    compiler_pin: ArtifactPinV1,
) -> None:
    binding = brief.source_binding
    scope = (manifest.organization_id, manifest.workspace_id)
    if (
        not brief.compiler_eligible
        or brief.review.state != "accepted"
        or inputs.build_ref != manifest.build_ref
        or snapshot.build_ref != manifest.build_ref
        or scope != (inputs.organization_id, inputs.workspace_id)
        or scope != (boundary.organization_id, boundary.workspace_id)
        or scope != (assessment.organization_id, assessment.workspace_id)
        or scope != (snapshot.organization_id, snapshot.workspace_id)
        or scope != (identity.organization_id, identity.workspace_id)
        or boundary.source_identity_ref != binding.source_identity_ref
        or boundary.source_hash != binding.source_hash
        or boundary.canonical_source_ir_ref != binding.canonical_source_ir_ref
        or boundary.canonical_source_ir_hash != binding.canonical_source_ir_hash
        or boundary.intended_domain != brief.represented_or_governed_subject.description
        or boundary.intended_use != brief.document_function.description
        or snapshot.triggering_ref != brief.graph_intent_id
        or (assessment.boundary_ref, assessment.boundary_hash)
        != (boundary.boundary_ref, boundary.boundary_hash)
    ):
        raise ValueError("source-first manifest native relationship mismatch")
    expected_authority_finding = brief.authority_boundary.source_role not in {
        "governed_internal_policy",
        "authoritative_source",
    }
    has_authority_finding = any(
        finding.finding_id == "adequacy:unresolved-authority-role"
        for finding in assessment.findings
    )
    if expected_authority_finding != has_authority_finding:
        raise ValueError("source-first adequacy authority projection mismatch")
    expected_gates = tuple(
        assessment.gate_for(operation) for operation in OntologyOperation
    )
    if manifest.gates != expected_gates:
        raise ValueError("source-first manifest gate projection mismatch")
    if not (
        manifest.created_at
        == boundary.created_at
        == boundary.target_time
        == assessment.assessed_at
        == snapshot.created_at
    ):
        raise ValueError("source-first build historical timestamps mismatch")
    build_basis = {
        "source_binding": binding.as_dict(),
        "graph_intent_ref": brief.graph_intent_id,
        "graph_intent_hash": brief.graph_intent_hash,
        "compiler_version": inputs.compiler_version,
        "compiler_hash": compiler_pin.artifact_hash,
        "assurance_profile": manifest.assurance_profile,
    }
    if manifest.build_ref != f"ontology-build:{content_hash(build_basis)}":
        raise ValueError("source-first native build ref mismatch")
    manifest_basis = {
        "build_ref": manifest.build_ref,
        "organization_id": manifest.organization_id,
        "workspace_id": manifest.workspace_id,
        "assurance_profile": manifest.assurance_profile,
        "child_artifacts": [
            pin.model_dump(mode="json") for pin in manifest.child_artifacts
        ],
        "gates": [gate.model_dump(mode="json") for gate in manifest.gates],
    }
    if manifest.predecessor_build_ref is not None:
        manifest_basis.update(
            predecessor_build_ref=manifest.predecessor_build_ref,
            predecessor_build_hash=manifest.predecessor_build_hash,
        )
    manifest_digest = content_hash(manifest_basis)
    if manifest.build_hash != f"sha256:{manifest_digest}":
        raise ValueError("source-first manifest native hash mismatch")


def _validate_source_first_snapshot_shape(
    *, snapshot: ExploratoryGraphSnapshotV1
) -> None:
    if (
        snapshot.authority_state != "provisional"
        or snapshot.completeness_state != "incomplete"
        or snapshot.predecessor_snapshot_ref is not None
        or snapshot.predecessor_snapshot_hash is not None
        or snapshot.change_cause != "initial_exploration"
        or snapshot.triggering_ref is None
    ):
        raise ValueError("unsupported exploratory snapshot producer shape")


def _validate_source_first_snapshot_binding(
    *,
    snapshot: ExploratoryGraphSnapshotV1,
    brief: GraphIntentBrief,
    identity: SourceIdentity,
    canonical: CanonicalSourceDocument,
) -> None:
    binding = brief.source_binding
    if (
        (snapshot.organization_id, snapshot.workspace_id)
        != (binding.organization_id, binding.workspace_id)
        or (snapshot.source_identity_ref, snapshot.source_hash)
        != (binding.source_identity_ref, binding.source_hash)
        or (snapshot.canonical_source_ir_ref, snapshot.canonical_source_ir_hash)
        != (binding.canonical_source_ir_ref, binding.canonical_source_ir_hash)
        or identity.source_hash != snapshot.source_hash
        or canonical.source_identity_ref != snapshot.source_identity_ref
    ):
        raise ValueError("snapshot native source binding mismatch")


def _validate_source_first_snapshot_projection(
    *,
    snapshot: ExploratoryGraphSnapshotV1,
    brief: GraphIntentBrief,
    canonical: CanonicalSourceDocument,
) -> None:
    mapped = {
        block_id
        for mapping in canonical.source_map.mappings
        for block_id in mapping.block_ids
    }
    blocks = sorted(canonical.blocks, key=lambda block: (block.ordinal, block.block_id))
    if not blocks or any(block.block_id not in mapped for block in blocks):
        raise ValueError("snapshot canonical blocks require exact source-map evidence")
    evidence = {
        block.block_id: f"evidence:canonical-block:{block.block_id.removeprefix('sha256:')}"
        for block in blocks
    }
    known = set(evidence.values())
    nodes = [
        ExploratoryGraphNodeV1(
            node_id=f"source-block:{block.block_id}",
            node_kind="canonical_source_block",
            label=str(block.content) if block.content else block.block_type,
            evidence_refs=(evidence[block.block_id],),
        )
        for block in blocks
    ]
    gaps = ["gap:certification_not_started", "gap:normative_semantics_unverified"]
    for dimension in brief.material_dimensions:
        refs = tuple(ref for ref in dimension.evidence_refs if ref in known)
        if not refs:
            gaps.append(f"gap:dimension_evidence:{dimension.type_id}")
        else:
            nodes.append(
                ExploratoryGraphNodeV1(
                    node_id=f"intent-dimension:{dimension.type_id}",
                    node_kind="reviewed_intent_dimension",
                    label=dimension.description,
                    evidence_refs=refs,
                )
            )
    intent_evidence = tuple(ref for ref in brief.evidence_refs if ref in known)
    for question in brief.competency_questions:
        gaps.append(f"gap:competency:{question.question_id}")
        if intent_evidence:
            nodes.append(
                ExploratoryGraphNodeV1(
                    node_id=f"intent-competency:{question.question_id}",
                    node_kind="reviewed_competency_question",
                    label=question.text,
                    evidence_refs=intent_evidence,
                )
            )
    gaps.extend(
        f"gap:intent:{question.question_id}" for question in brief.unresolved_questions
    )
    source_node_by_evidence = {
        ref: f"source-block:{block_id}" for block_id, ref in evidence.items()
    }
    edges = tuple(
        ExploratoryGraphEdgeV1(
            edge_id=(
                "intent-evidence:"
                + content_hash({"node": node.node_id, "evidence": ref})
            ),
            source_node_id=node.node_id,
            target_node_id=source_node_by_evidence[ref],
            relation_type="evidenced_by",
            evidence_refs=(ref,),
        )
        for node in nodes
        if node.node_kind != "canonical_source_block"
        for ref in node.evidence_refs
    )
    if (
        snapshot.nodes != tuple(nodes)
        or snapshot.edges != edges
        or snapshot.unresolved_gap_ids != tuple(sorted(set(gaps)))
    ):
        raise ValueError("exploratory snapshot native projection mismatch")


def _validate_source_first_snapshot_integrity(
    *,
    snapshot: ExploratoryGraphSnapshotV1,
    integrity: ExploratoryGraphSnapshotIntegrityV1,
) -> None:
    if (
        integrity.producer_profile != _SOURCE_FIRST_PROFILE
        or integrity.producer_version != _SOURCE_FIRST_COMPILER_VERSION
    ):
        raise ValueError("unsupported source-first snapshot integrity producer")
    verify_exploratory_snapshot_integrity(snapshot=snapshot, integrity=integrity)


def verify_ontology_build_inputs_v1(
    *, source: CapturedSourceV1, closure: Mapping[PinKey, CapturedSourceV1]
) -> tuple[ArtifactPinV1, ...]:
    closure = _validate_closure(closure)
    source, inputs = _native_contract(
        source=source,
        closure=closure,
        contract=OntologyBuildInputEnvelopeV1,
        kind="ontology_build_inputs",
        ref_field="envelope_ref",
        hash_field="envelope_hash",
    )
    brief = GraphIntentBrief.from_dict(inputs.graph_intent)
    if brief.as_dict() != inputs.graph_intent or brief.review.state != "accepted":
        raise ValueError("build inputs require the exact accepted native GraphIntent")
    identity = SourceIdentity.model_validate(inputs.source_identity)
    identity_payload = identity.model_dump(
        mode="json", by_alias=True, exclude_none=False
    )
    canonical = CanonicalSourceDocument.model_validate(inputs.canonical_source_ir)
    canonical_payload = canonical.model_dump(
        mode="json", by_alias=True, exclude_none=False
    )
    if (
        identity_payload != inputs.source_identity
        or canonical_payload != inputs.canonical_source_ir
    ):
        raise ValueError("build inputs contain a non-native source archive encoding")

    graph_pin = ArtifactPinV1(
        artifact_kind="graph_intent_brief",
        artifact_ref=brief.graph_intent_id,
        artifact_hash=brief.graph_intent_hash,
    )
    identity_pin = ArtifactPinV1(
        artifact_kind="source_identity",
        artifact_ref=identity.source_identity_ref,
        artifact_hash="sha256:" + content_hash(identity_payload),
    )
    ir_pin = ArtifactPinV1(
        artifact_kind="canonical_source_ir",
        artifact_ref=brief.source_binding.canonical_source_ir_ref,
        artifact_hash=brief.source_binding.canonical_source_ir_hash,
    )
    compiler_payload = _source_first_compiler_payload(inputs)
    compiler_raw = _strict_base64(
        compiler_payload["source_bytes_base64"], "source-first compiler bytes"
    )
    compiler_pin = ArtifactPinV1(
        artifact_kind="ontology_source_first_compiler",
        artifact_ref=inputs.compiler_version,
        artifact_hash="sha256:" + sha256(compiler_raw).hexdigest(),
    )

    graph_source = _resolve_pin(closure, graph_pin)
    _graph_source, graph_payload = _payload(graph_source)
    identity_source = _resolve_pin(closure, identity_pin)
    _identity_source, retained_identity = _payload(identity_source)
    ir_source = _resolve_pin(closure, ir_pin)
    _ir_source, retained_ir = _payload(ir_source)
    compiler_source = _resolve_pin(closure, compiler_pin)
    _compiler_source, retained_compiler = _payload(compiler_source)
    if (
        graph_payload != inputs.graph_intent
        or retained_identity != inputs.source_identity
        or retained_ir != inputs.canonical_source_ir
        or retained_compiler != compiler_payload
    ):
        raise ValueError("build inputs do not equal their retained native members")

    source_bytes = _strict_base64(inputs.source_bytes_base64, "archived source bytes")
    owner, view = _matching_source_member_capture(
        closure=closure,
        predicate=lambda candidate: (
            candidate.identity_payload == identity_payload
            and candidate.payload["canonical_source_ir"] == canonical_payload
            and candidate.payload["source_bytes_base64"]
            == base64.b64encode(source_bytes).decode("ascii")
        ),
        label="build input source archive",
    )
    if (
        owner.source != identity_source.dependencies[0]
        or owner.source != ir_source.dependencies[0]
    ):
        raise ValueError("build input members do not share their exact source capture")
    if view.identity.source_hash != "sha256:" + sha256(source_bytes).hexdigest():
        raise ValueError("build input archived source bytes mismatch")
    return _expect_dependencies(
        source, (graph_pin, identity_pin, ir_pin, compiler_pin), closure
    )


def verify_ontology_source_first_compiler_v1(
    *, source: CapturedSourceV1, closure: Mapping[PinKey, CapturedSourceV1]
) -> tuple[ArtifactPinV1, ...]:
    closure = _validate_closure(closure)
    source, payload = _payload(_require_supplied_source(source, closure))
    if not isinstance(payload, dict) or set(payload) != {
        "compiler_version",
        "source_bytes_base64",
    }:
        raise ValueError("invalid source-first compiler archive member")
    if payload["compiler_version"] != _SOURCE_FIRST_COMPILER_VERSION:
        raise ValueError("unsupported source-first compiler version")
    raw = _strict_base64(payload["source_bytes_base64"], "source-first compiler bytes")
    _expect_identity(
        source,
        kind="ontology_source_first_compiler",
        ref=_SOURCE_FIRST_COMPILER_VERSION,
        digest="sha256:" + sha256(raw).hexdigest(),
    )
    return _expect_dependencies(source, (), closure)


def verify_canonical_evidence_boundary_v1(
    *, source: CapturedSourceV1, closure: Mapping[PinKey, CapturedSourceV1]
) -> tuple[ArtifactPinV1, ...]:
    closure = _validate_closure(closure)
    source, boundary = _native_contract(
        source=source,
        closure=closure,
        contract=CanonicalEvidenceBoundaryV1,
        kind="canonical_evidence_boundary",
        ref_field="boundary_ref",
        hash_field="boundary_hash",
    )
    _validate_source_first_boundary_identity(boundary=boundary)
    identity_source = _unique_ref(
        closure, kind="source_identity", ref=boundary.source_identity_ref
    )
    identity = _model_payload(identity_source, SourceIdentity, "source identity")
    ir_pin = ArtifactPinV1(
        artifact_kind="canonical_source_ir",
        artifact_ref=boundary.canonical_source_ir_ref,
        artifact_hash=boundary.canonical_source_ir_hash,
    )
    ir_source = _resolve_pin(closure, ir_pin)
    canonical = _model_payload(
        ir_source, CanonicalSourceDocument, "canonical source IR"
    )
    _validate_source_first_boundary_relation(
        boundary=boundary, identity=identity, canonical=canonical
    )
    return _expect_dependencies(source, (identity_source.source, ir_pin), closure)


def verify_canonical_evidence_adequacy_assessment_v1(
    *, source: CapturedSourceV1, closure: Mapping[PinKey, CapturedSourceV1]
) -> tuple[ArtifactPinV1, ...]:
    closure = _validate_closure(closure)
    source, assessment = _native_contract(
        source=source,
        closure=closure,
        contract=CanonicalEvidenceAdequacyAssessmentV1,
        kind="canonical_evidence_adequacy_assessment",
        ref_field="assessment_ref",
        hash_field="assessment_hash",
    )
    _validate_source_first_assessment_identity(assessment=assessment)
    boundary_pin = ArtifactPinV1(
        artifact_kind="canonical_evidence_boundary",
        artifact_ref=assessment.boundary_ref,
        artifact_hash=assessment.boundary_hash,
    )
    boundary_source = _resolve_pin(closure, boundary_pin)
    boundary = _model_payload(
        boundary_source, CanonicalEvidenceBoundaryV1, "canonical evidence boundary"
    )
    _validate_source_first_assessment_relation(assessment=assessment, boundary=boundary)

    dependencies = [boundary_pin]
    member_refs = {
        ref
        for finding in assessment.findings
        for ref in (*finding.evidence_refs, *finding.search_evidence_refs)
    }
    for ref in member_refs:
        if ref == boundary.source_identity_ref:
            member = _unique_ref(closure, kind="source_identity", ref=ref)
            identity = _model_payload(member, SourceIdentity, "source identity")
            if identity.source_hash != boundary.source_hash:
                raise ValueError("adequacy source identity member mismatch")
        elif ref == boundary.canonical_source_ir_ref:
            member = _resolve_pin(
                closure,
                ArtifactPinV1(
                    artifact_kind="canonical_source_ir",
                    artifact_ref=ref,
                    artifact_hash=boundary.canonical_source_ir_hash,
                ),
            )
        else:
            raise ValueError("adequacy finding cites an unsupported native member")
        dependencies.append(member.source)
    return _expect_dependencies(source, tuple(dependencies), closure)


def verify_exploratory_graph_snapshot_integrity_v1(
    *, source: CapturedSourceV1, closure: Mapping[PinKey, CapturedSourceV1]
) -> tuple[ArtifactPinV1, ...]:
    closure = _validate_closure(closure)
    source, integrity = _native_contract(
        source=source,
        closure=closure,
        contract=ExploratoryGraphSnapshotIntegrityV1,
        kind="exploratory_graph_snapshot_integrity",
        ref_field="integrity_ref",
        hash_field="integrity_hash",
    )
    if (
        integrity.producer_profile == "advisory_trial_projection.v1"
        and integrity.producer_version == "runtime_learning_trial_paired_build_v1"
    ):
        return _verify_trial_projection_integrity(
            source=source, integrity=integrity, closure=closure
        )
    if (
        integrity.producer_profile != _SOURCE_FIRST_PROFILE
        or integrity.producer_version != _SOURCE_FIRST_COMPILER_VERSION
    ):
        raise ValueError("unsupported source-first snapshot integrity producer")
    return _expect_dependencies(source, (), closure)


def verify_exploratory_graph_snapshot_v1(
    *, source: CapturedSourceV1, closure: Mapping[PinKey, CapturedSourceV1]
) -> tuple[ArtifactPinV1, ...]:
    closure = _validate_closure(closure)
    source, snapshot = _native_contract(
        source=source,
        closure=closure,
        contract=ExploratoryGraphSnapshotV1,
        kind="exploratory_graph_snapshot",
        ref_field="snapshot_ref",
        hash_field="snapshot_hash",
    )
    if snapshot.change_cause == "controlled_trial_projection":
        return _verify_trial_projection_snapshot(
            source=source, snapshot=snapshot, closure=closure
        )
    _validate_source_first_snapshot_shape(snapshot=snapshot)
    graph_source = _unique_ref(
        closure, kind="graph_intent_brief", ref=snapshot.triggering_ref
    )
    brief_source, brief_payload = _payload(graph_source)
    brief = GraphIntentBrief.from_dict(brief_payload)
    if brief.as_dict() != brief_payload or brief.review.state != "accepted":
        raise ValueError("snapshot triggering GraphIntent mismatch")
    identity_source = _unique_ref(
        closure, kind="source_identity", ref=snapshot.source_identity_ref
    )
    identity = _model_payload(identity_source, SourceIdentity, "source identity")
    ir_pin = ArtifactPinV1(
        artifact_kind="canonical_source_ir",
        artifact_ref=snapshot.canonical_source_ir_ref,
        artifact_hash=snapshot.canonical_source_ir_hash,
    )
    ir_source = _resolve_pin(closure, ir_pin)
    canonical = _model_payload(
        ir_source, CanonicalSourceDocument, "canonical source IR"
    )
    _validate_source_first_snapshot_binding(
        snapshot=snapshot, brief=brief, identity=identity, canonical=canonical
    )
    _matching_source_member_capture(
        closure=closure,
        predicate=lambda view: (
            view.identity_payload
            == identity.model_dump(mode="json", by_alias=True, exclude_none=False)
            and view.payload["canonical_source_ir"]
            == canonical.model_dump(mode="json", by_alias=True, exclude_none=False)
        ),
        label="snapshot source members",
    )

    _validate_source_first_snapshot_projection(
        snapshot=snapshot, brief=brief, canonical=canonical
    )

    integrity_matches = []
    for candidate in closure.values():
        if candidate.source.artifact_kind != "exploratory_graph_snapshot_integrity":
            continue
        integrity = _model_payload(
            candidate,
            ExploratoryGraphSnapshotIntegrityV1,
            "exploratory snapshot integrity",
        )
        if integrity.subject == source.source:
            integrity_matches.append((candidate, integrity))
    if len(integrity_matches) != 1:
        raise ValueError("snapshot must resolve one exact prospective integrity")
    integrity_source, integrity = integrity_matches[0]
    _validate_source_first_snapshot_integrity(snapshot=snapshot, integrity=integrity)
    return _expect_dependencies(
        source,
        (brief_source.source, identity_source.source, ir_pin, integrity_source.source),
        closure,
    )


def verify_ontology_build_manifest_v1(
    *, source: CapturedSourceV1, closure: Mapping[PinKey, CapturedSourceV1]
) -> tuple[ArtifactPinV1, ...]:
    closure = _validate_closure(closure)
    source, manifest = _native_contract(
        source=source,
        closure=closure,
        contract=OntologyBuildManifestV1,
        kind="ontology_build_manifest",
        ref_field="build_ref",
        hash_field="build_hash",
    )
    expected_kinds = (
        "canonical_evidence_adequacy_assessment",
        "ontology_build_inputs",
        "canonical_evidence_boundary",
        "exploratory_graph_snapshot",
        "graph_intent_brief",
        "source_identity",
        "canonical_source_ir",
        "ontology_source_first_compiler",
    )
    actual_kinds = tuple(pin.artifact_kind for pin in manifest.child_artifacts)
    projection_kinds = (
        "canonical_evidence_adequacy_assessment",
        "advisory_trial_build_inputs",
        "exploratory_graph_snapshot",
        "graph_intent_brief",
        "competency_contract",
        "normative_semantic_candidate_set",
        "validation_contract",
        "compiled_rule_plan",
    )
    if actual_kinds == projection_kinds:
        return _verify_trial_projection_manifest(
            source=source, manifest=manifest, closure=closure
        )
    if actual_kinds != expected_kinds:
        raise ValueError("source-first manifest child set or order mismatch")
    children = {
        pin.artifact_kind: _resolve_pin(closure, pin)
        for pin in manifest.child_artifacts
    }
    assessment = _model_payload(
        children["canonical_evidence_adequacy_assessment"],
        CanonicalEvidenceAdequacyAssessmentV1,
        "canonical evidence adequacy assessment",
    )
    inputs = _model_payload(
        children["ontology_build_inputs"], OntologyBuildInputEnvelopeV1, "build inputs"
    )
    boundary = _model_payload(
        children["canonical_evidence_boundary"],
        CanonicalEvidenceBoundaryV1,
        "canonical evidence boundary",
    )
    snapshot = _model_payload(
        children["exploratory_graph_snapshot"],
        ExploratoryGraphSnapshotV1,
        "exploratory graph snapshot",
    )
    _graph_source, graph_payload = _payload(children["graph_intent_brief"])
    brief = GraphIntentBrief.from_dict(graph_payload)
    identity = _model_payload(
        children["source_identity"], SourceIdentity, "source identity"
    )
    canonical = _model_payload(
        children["canonical_source_ir"], CanonicalSourceDocument, "canonical source IR"
    )
    _compiler_source, compiler_payload = _payload(
        children["ontology_source_first_compiler"]
    )
    expected_compiler_payload = _source_first_compiler_payload(inputs)
    compiler_raw = _strict_base64(
        expected_compiler_payload["source_bytes_base64"], "source-first compiler bytes"
    )
    compiler_pin = ArtifactPinV1(
        artifact_kind="ontology_source_first_compiler",
        artifact_ref=inputs.compiler_version,
        artifact_hash="sha256:" + sha256(compiler_raw).hexdigest(),
    )
    if (
        graph_payload != inputs.graph_intent
        or identity.model_dump(mode="json", by_alias=True, exclude_none=False)
        != inputs.source_identity
        or canonical.model_dump(mode="json", by_alias=True, exclude_none=False)
        != inputs.canonical_source_ir
        or compiler_payload != expected_compiler_payload
        or children["ontology_source_first_compiler"].source != compiler_pin
    ):
        raise ValueError("source-first manifest archive member mismatch")
    _validate_source_first_manifest_values(
        manifest=manifest,
        assessment=assessment,
        inputs=inputs,
        boundary=boundary,
        snapshot=snapshot,
        brief=brief,
        identity=identity,
        canonical=canonical,
        compiler_pin=compiler_pin,
    )
    dependencies = list(manifest.child_artifacts)
    if manifest.predecessor_build_ref is not None:
        predecessor = ArtifactPinV1(
            artifact_kind="ontology_build_manifest",
            artifact_ref=manifest.predecessor_build_ref,
            artifact_hash=manifest.predecessor_build_hash,
        )
        dependencies.append(predecessor)
    return _expect_dependencies(source, tuple(dependencies), closure)
