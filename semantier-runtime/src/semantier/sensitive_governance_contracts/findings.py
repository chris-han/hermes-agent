from __future__ import annotations

from dataclasses import asdict, dataclass

from semantier.sensitive_governance_contracts.enums import FindingClass


@dataclass(frozen=True)
class InstitutionalClassification:
    classification_id: str
    finding_class: FindingClass
    tier: str
    display_name_zh: str
    confidence: float

    def to_dict(self) -> dict:
        result = asdict(self)
        result["finding_class"] = self.finding_class.value
        return result


@dataclass(frozen=True)
class SensitiveFinding:
    finding_id: str
    finding_class: FindingClass
    evidence_ref: str
    classification: InstitutionalClassification
    original_value_ref: str | None = None

    def to_dict(self) -> dict:
        result = asdict(self)
        result["finding_class"] = self.finding_class.value
        result["classification"] = self.classification.to_dict()
        return result
