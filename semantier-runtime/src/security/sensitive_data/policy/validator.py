from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from security.sensitive_data.detection.deterministic import SensitiveDataDetector
from security.sensitive_data.policy.evaluator import PolicyDecisionResult
from semantier.sensitive_governance_contracts.enums import FindingClass
from semantier.sensitive_governance_contracts.hashing import canonical_hash

CRITICAL_RESIDUAL_CLASSES = {
    FindingClass.CREDENTIAL_OR_SECRET,
    FindingClass.EXISTING_STATE_SECRET_MARKING,
    FindingClass.SUSPECTED_STATE_SECRET,
}


@dataclass(frozen=True)
class ValidationResult:
    validation_result_ref: str
    input_manifest_ref: str
    input_manifest_hash: str
    residual_scan_evidence_ref: str
    residual_scan_evidence_hash: str
    route: str
    reason_codes: list[str]
    validation_checks: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "validation_result_ref": self.validation_result_ref,
            "input_manifest_ref": self.input_manifest_ref,
            "input_manifest_hash": self.input_manifest_hash,
            "residual_scan_evidence_ref": self.residual_scan_evidence_ref,
            "residual_scan_evidence_hash": self.residual_scan_evidence_hash,
            "route": self.route,
            "reason_codes": self.reason_codes,
            "validation_checks": self.validation_checks,
            "validator_identity": "repo_sensitive_admissibility_validator_v1",
            "validator_contract_version": "sensitive_validation.v1",
        }


def validate_input_manifest(
    *,
    manifest: dict[str, Any],
    policy_decisions: list[PolicyDecisionResult],
    transformed_text: str,
    residual_detector: SensitiveDataDetector | None = None,
) -> ValidationResult:
    required = [
        "fact_context_hash",
        "semantic_binding_hash",
        "authority_resolution_hash",
        "enterprise_consent_hash",
        "authorization_snapshot_hash",
        "policy_decision_hash",
        "disclosure_justification_hash",
        "transformed_input_hash",
        "residual_scan_evidence_hash",
    ]
    missing = [key for key in required if not manifest.get(key)]
    authorization_invalid = manifest.get("authorization_valid") is not True
    residual = (residual_detector or SensitiveDataDetector()).detect(transformed_text)
    critical = [
        finding.finding_class.value
        for finding in residual
        if finding.finding_class in CRITICAL_RESIDUAL_CLASSES
    ]
    route = "PASS_AUTO"
    reason_codes: list[str] = []
    if missing:
        route = "FAIL_HARD"
        reason_codes.append("replay_input_manifest_incomplete")
    if authorization_invalid:
        route = "FAIL_HARD"
        reason_codes.extend(manifest.get("authorization_reason_codes") or ["authorization_snapshot_invalid"])
    if critical:
        route = "FAIL_HARD"
        reason_codes.append("critical_residual_sensitive_data_detected")
    if any(decision.institutional_route == "FAIL_HARD" for decision in policy_decisions):
        route = "FAIL_HARD"
        reason_codes.append("policy_fail_hard")
    elif route == "PASS_AUTO" and any(decision.institutional_route == "ESCALATE" for decision in policy_decisions):
        route = "ESCALATE"
        reason_codes.append("policy_escalation_required")

    manifest_hash = canonical_hash(manifest)
    residual_evidence = {
        "residual_count": len(residual),
        "critical_residual_classes": sorted(set(critical)),
    }
    return ValidationResult(
        validation_result_ref="sg_validation_" + canonical_hash({"manifest_hash": manifest_hash, "route": route})[:24],
        input_manifest_ref=str(manifest.get("input_manifest_ref") or "sg_input_manifest"),
        input_manifest_hash=manifest_hash,
        residual_scan_evidence_ref=str(manifest.get("residual_scan_evidence_ref") or "sg_residual_scan"),
        residual_scan_evidence_hash=canonical_hash(residual_evidence),
        route=route,
        reason_codes=reason_codes or ["validation_passed"],
        validation_checks=[
            "fact_sufficiency",
            "semantic_binding_valid",
            "authority_resolved",
            "enterprise_consent_valid",
            "authorization_valid",
            "policy_compatible",
            "residual_scan_complete",
            "replay_input_manifest_complete",
        ],
    )
