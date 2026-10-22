"""Real RDF/SHACL structural validation for normative semantic frames."""

from __future__ import annotations

from pyshacl import validate as shacl_validate
from rdflib import Graph, Literal, Namespace, RDF, URIRef

from contracts.normative_semantics import NormativeSemanticCandidateSetV1
from contracts.ontology_building import ArtifactPinV1
from contracts.ontology_quality import QualityFindingV1, QualityProfileV1, QualityRequirementResultV1
from eos.artifact_hashing import canonical_json, content_hash
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path


SEM = Namespace("https://semantier.example/ontology-building/")

_SHAPES = """
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix sem: <https://semantier.example/ontology-building/> .

sem:NormativeFrameShape a sh:NodeShape ;
  sh:targetClass sem:NormativeFrame ;
  sh:property [ sh:path sem:frameKind ; sh:minCount 1 ; sh:maxCount 1 ] ;
  sh:property [ sh:path sem:evidenceRef ; sh:minCount 1 ] .
"""


class ConstraintQualityEvaluator:
    """Execute real SHACL and retain canonical deterministic requirement evidence."""

    def evaluate(self, candidate: NormativeSemanticCandidateSetV1, *, profile: QualityProfileV1) -> QualityRequirementResultV1:
        if (candidate.organization_id, candidate.workspace_id) != (profile.organization_id, profile.workspace_id):
            raise PermissionError("constraint inputs cross tenant scope")
        requirement = next((item for item in profile.requirements if item.requirement_id == "shape_conformance"), None)
        if requirement is None or requirement.gate != "CONSTRAINT":
            raise ValueError("profile has no canonical constraint requirement")
        subject_pin = ArtifactPinV1(artifact_kind="normative_semantic_candidate_set",
            artifact_ref=candidate.candidate_set_ref, artifact_hash=candidate.candidate_set_hash)
        evaluator = ArtifactPinV1(artifact_kind="quality_evaluator", artifact_ref="constraint:pyshacl",
            artifact_hash="sha256:" + content_hash(dict(code_hash=sha256(Path(__file__).read_bytes()).hexdigest(),
                pyshacl_version=version("pyshacl"), rdflib_version=version("rdflib"), shapes=_SHAPES)))
        data = Graph()
        by_subject = {}
        for frame in candidate.frames:
            subject = URIRef(f"urn:semantier:frame:{frame.frame_id}")
            by_subject[subject] = frame
            data.add((subject, RDF.type, SEM.NormativeFrame))
            data.add((subject, SEM.frameKind, Literal(frame.frame_kind)))
            for evidence_ref in frame.evidence_refs:
                data.add((subject, SEM.evidenceRef, Literal(evidence_ref)))

        findings = []
        unavailable = None
        try:
            shapes = Graph().parse(data=_SHAPES, format="turtle")
            conforms, report_graph, _report_text = shacl_validate(
                data_graph=data, shacl_graph=shapes, inference="none", abort_on_first=False,
                allow_infos=False, allow_warnings=False)
            sh = Namespace("http://www.w3.org/ns/shacl#")
            # SHACL result nodes may be random blank-node IDs. Stable report
            # properties, not blank-node identifiers, determine content identity.
            for node in report_graph.subjects(RDF.type, sh.ValidationResult):
                focus = report_graph.value(node, sh.focusNode)
                frame = by_subject.get(focus)
                detail = dict(focus=str(focus), path=str(report_graph.value(node, sh.resultPath)),
                    constraint=str(report_graph.value(node, sh.sourceConstraintComponent)),
                    messages=sorted(str(message) for message in report_graph.objects(node, sh.resultMessage)))
                findings.append(QualityFindingV1.create(requirement_id=requirement.requirement_id,
                    gate="CONSTRAINT", severity="BLOCK", code="shape_violation",
                    message=canonical_json(detail), evidence_refs=tuple(sorted(set(frame.evidence_refs))) if frame else (),
                    subject_refs=(frame.frame_id,) if frame else (str(focus),)))
            if not conforms and not findings:
                findings.append(QualityFindingV1.create(requirement_id=requirement.requirement_id,
                    gate="CONSTRAINT", severity="BLOCK", code="shape_violation", message="SHACL rejected the candidate.",
                    evidence_refs=(), subject_refs=(candidate.candidate_set_ref,)))
        except Exception as exc:
            unavailable = "evaluator_error"
            findings = [QualityFindingV1.create(requirement_id=requirement.requirement_id,
                gate="CONSTRAINT", severity="BLOCK" if requirement.unavailable_status == "BLOCK" else "WARN",
                code=unavailable, message=f"SHACL evaluation failed: {type(exc).__name__}: {exc}",
                evidence_refs=(), subject_refs=(candidate.candidate_set_ref,))]
        findings = tuple(sorted({item.finding_ref: item for item in findings}.values(), key=lambda item: item.finding_ref))
        status = requirement.unavailable_status if unavailable else "BLOCK" if findings else "PASS"
        return QualityRequirementResultV1.create(requirement_id=requirement.requirement_id, gate="CONSTRAINT",
            quality_profile=profile.pin(), subject=subject_pin, evaluator=evaluator, status=status,
            findings=findings, unavailable_reason=unavailable)
