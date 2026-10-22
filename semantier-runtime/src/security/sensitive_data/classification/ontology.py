from __future__ import annotations

from dataclasses import dataclass

from security.sensitive_data.authority.bundle_loader import ActivatedSensitiveGovernanceBundle
from security.sensitive_data.detection.deterministic import DetectionFinding
from semantier.sensitive_governance_contracts.enums import FindingClass
from semantier.sensitive_governance_contracts.hashing import canonical_hash


@dataclass(frozen=True)
class BoundFinding:
    finding_id: str
    finding_class: FindingClass
    start: int
    end: int
    detector_id: str
    semantic_binding_ref: str
    ontology_artifact_id: str
    display_name_zh: str
    evidence_ref: str
    original_value_ref: str

    def to_public_dict(self) -> dict:
        return {
            "finding_id": self.finding_id,
            "finding_class": self.finding_class.value,
            "start": self.start,
            "end": self.end,
            "detector_id": self.detector_id,
            "semantic_binding_ref": self.semantic_binding_ref,
            "ontology_artifact_id": self.ontology_artifact_id,
            "display_name_zh": self.display_name_zh,
            "evidence_ref": self.evidence_ref,
            "original_value_ref": self.original_value_ref,
        }


def bind_findings_to_t1(
    findings: list[DetectionFinding],
    bundle: ActivatedSensitiveGovernanceBundle,
) -> list[BoundFinding]:
    ontology = bundle.ontology_artifact
    classes = ontology.get("classes")
    if not isinstance(classes, list):
        raise ValueError("T1 ontology classes are missing")
    class_labels = {
        str(entry.get("class_id")): str(entry.get("display_name_zh") or entry.get("class_id"))
        for entry in classes
        if isinstance(entry, dict)
    }

    bound: list[BoundFinding] = []
    for finding in findings:
        label = class_labels.get(finding.finding_class.value)
        if label is None:
            raise ValueError(f"finding class missing from T1 ontology: {finding.finding_class.value}")
        semantic_binding_ref = "sg_binding_" + canonical_hash(
            {
                "finding_id": finding.finding_id,
                "class": finding.finding_class.value,
                "ontology_artifact_id": ontology["artifact_id"],
            }
        )[:24]
        bound.append(
            BoundFinding(
                finding_id=finding.finding_id,
                finding_class=finding.finding_class,
                start=finding.start,
                end=finding.end,
                detector_id=finding.detector_id,
                semantic_binding_ref=semantic_binding_ref,
                ontology_artifact_id=str(ontology["artifact_id"]),
                display_name_zh=label,
                evidence_ref=finding.evidence_ref,
                original_value_ref=f"restricted://sensitive-governance/input/{finding.finding_id}",
            )
        )
    return bound
