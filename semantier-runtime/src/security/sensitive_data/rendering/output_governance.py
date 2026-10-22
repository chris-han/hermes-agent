from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from security.sensitive_data.authority.bundle_loader import load_demo_bundle
from security.sensitive_data.classification.ontology import bind_findings_to_t1
from security.sensitive_data.detection.deterministic import SensitiveDataDetector
from security.sensitive_data.policy.evaluator import SensitivePolicyEvaluator
from semantier.sensitive_governance_contracts.enums import (
    DisclosureAction,
    DisplayAction,
    FindingClass,
    GovernedResponseSegmentKind,
)
from semantier.sensitive_governance_contracts.governed_response import (
    GovernedResponseSegment,
    build_response,
)
from semantier.sensitive_governance_contracts.hashing import canonical_hash
from semantier.sensitive_governance_contracts.pins import SensitiveGovernancePins
from semantier.sensitive_governance_contracts.registry import CURRENT_CONTRACT_VERSION


DEMO_BUNDLE_ID = "soe_sensitive_governance_demo_bundle_v1"
DISPLAY_PROFILE_VERSION = "soe_sensitive_display_profile_v1"
RAW_SIMULATED_AGENT_RESULT = (
    "已完成国企招标文件敏感信息治理检查。"
    "预算控制价1000万元。评标专家：张三 手机13800138000。"
    "API_KEY=sk-demo-secret 密级:秘密。"
)


@dataclass(frozen=True)
class OutputGovernanceResult:
    governed_response: dict[str, Any]
    authority_objects: list[dict[str, Any]]


def _display_action_for(
    *,
    finding_class: FindingClass,
    disclosure_action: DisclosureAction,
) -> DisplayAction:
    if finding_class in {
        FindingClass.CREDENTIAL_OR_SECRET,
        FindingClass.EXISTING_STATE_SECRET_MARKING,
    }:
        return DisplayAction.NEVER_RENDER
    if finding_class == FindingClass.TENDER_EXPERT_INFORMATION:
        return DisplayAction.REDACT
    if disclosure_action == DisclosureAction.TOKENIZE:
        return DisplayAction.TOKENIZE
    if disclosure_action == DisclosureAction.MASK:
        return DisplayAction.MASK
    if disclosure_action == DisclosureAction.REDACT:
        return DisplayAction.REDACT
    if disclosure_action == DisclosureAction.BLOCK:
        return DisplayAction.NEVER_RENDER
    return DisplayAction.SHOW


def _segment_kind_for(action: DisplayAction) -> GovernedResponseSegmentKind:
    if action == DisplayAction.NEVER_RENDER:
        return GovernedResponseSegmentKind.NEVER_RENDER_NOTICE
    if action == DisplayAction.REDACT:
        return GovernedResponseSegmentKind.REDACTION_NOTICE
    if action in {DisplayAction.MASK, DisplayAction.TOKENIZE}:
        return GovernedResponseSegmentKind.SENSITIVE_TEXT
    return GovernedResponseSegmentKind.TEXT


def _segment_text(action: DisplayAction, finding_class: FindingClass, index: int) -> str:
    if action == DisplayAction.TOKENIZE:
        return f"{finding_class.value} 已替换为 [{finding_class.value}_TOKEN_{index}]。"
    if action == DisplayAction.MASK:
        return f"{finding_class.value} 已遮蔽。"
    if action == DisplayAction.REDACT:
        return f"[已按 {finding_class.value} 规则删除]"
    if action == DisplayAction.NEVER_RENDER:
        return "[存在凭据或既有秘密标识，禁止渲染原文]"
    return "已完成国企招标文件敏感信息治理检查。"


def govern_simulated_agent_output(
    *,
    run_id: str,
    assessment_id: str,
    assessment_hash: str,
    execution_boundary: dict[str, Any],
) -> OutputGovernanceResult:
    bundle = load_demo_bundle()
    detector = SensitiveDataDetector()
    findings = bind_findings_to_t1(
        detector.detect(RAW_SIMULATED_AGENT_RESULT),
        bundle,
    )
    evaluator = SensitivePolicyEvaluator(bundle)
    policy_decisions = evaluator.decide(findings, destination="browser_display")
    decisions_by_finding = {
        decision.finding_id: decision for decision in policy_decisions
    }
    authority_refs = evaluator._authority_refs()
    display_decisions: list[dict[str, Any]] = []
    segments: list[GovernedResponseSegment] = [
        GovernedResponseSegment(
            segment_id="seg_public_summary",
            sequence=0,
            kind=GovernedResponseSegmentKind.TEXT,
            display_action=DisplayAction.SHOW,
            text="已完成国企招标文件敏感信息治理检查。",
            classification="PUBLIC_INFORMATION",
            authority_refs=authority_refs,
        )
    ]
    selected_findings = []
    seen_classes: set[FindingClass] = set()
    for finding in findings:
        if finding.finding_class in seen_classes:
            continue
        seen_classes.add(finding.finding_class)
        selected_findings.append(finding)
    for index, finding in enumerate(selected_findings, start=1):
        policy_decision = decisions_by_finding[finding.finding_id]
        display_action = _display_action_for(
            finding_class=finding.finding_class,
            disclosure_action=policy_decision.enforcement_action,
        )
        decision = {
            "display_decision_ref": "sg_display_decision_"
            + canonical_hash(
                {
                    "finding_id": finding.finding_id,
                    "display_action": display_action.value,
                    "response_index": index,
                }
            )[:24],
            "finding_id": finding.finding_id,
            "finding_class": finding.finding_class.value,
            "display_action": display_action.value,
            "institutional_route": policy_decision.institutional_route,
            "policy_decision_ref": policy_decision.policy_decision_ref,
            "authority_refs": policy_decision.authority_refs,
            "reason_codes": policy_decision.reason_codes,
        }
        display_decisions.append(decision)
        segments.append(
            GovernedResponseSegment(
                segment_id=f"seg_output_{finding.finding_class.value.lower()}",
                sequence=len(segments),
                kind=_segment_kind_for(display_action),
                display_action=display_action,
                text=_segment_text(display_action, finding.finding_class, index),
                classification=finding.finding_class.value,
                authority_refs=authority_refs,
                eligible_for_reveal=display_action in {
                    DisplayAction.MASK,
                    DisplayAction.TOKENIZE,
                },
                restricted_payload_ref=(
                    "restricted://sensitive-governance/output/"
                    + canonical_hash({"finding_id": finding.finding_id})[:24]
                    if display_action in {DisplayAction.MASK, DisplayAction.TOKENIZE}
                    else None
                ),
            )
        )
    response_id = f"sg_response_{uuid4().hex}"
    display_justification = {
        "display_justification_ref": "sg_display_justification_"
        + canonical_hash(display_decisions)[:24],
        "assessment_id": assessment_id,
        "display_profile_version": DISPLAY_PROFILE_VERSION,
        "display_decisions": display_decisions,
        "raw_output_hash": canonical_hash({"raw_agent_output": RAW_SIMULATED_AGENT_RESULT}),
    }
    display_admissibility = {
        "display_admissibility_ref": "sg_display_admissibility_"
        + canonical_hash(display_justification)[:24],
        "display_justification_ref": display_justification["display_justification_ref"],
        "route": "PASS_AUTO",
        "reason_codes": ["display_ast_validated"],
    }
    pins = SensitiveGovernancePins(
        contract_version=CURRENT_CONTRACT_VERSION,
        ontology_version="soe_sensitive_governance_t1_v1",
        authority_bundle_version="soe_sensitive_governance_t2_t3_v1",
        policy_bundle_version="soe_sensitive_governance_t4_v1",
        display_profile_version=DISPLAY_PROFILE_VERSION,
        fixture_manifest_hash=canonical_hash(
            {
                "bundle_id": DEMO_BUNDLE_ID,
                "assessment_id": assessment_id,
                "assessment_hash": assessment_hash,
                "display_justification_ref": display_justification[
                    "display_justification_ref"
                ],
            }
        ),
    )
    response = build_response(
        response_id=response_id,
        run_id=run_id,
        segments=segments,
        governance_summary={
            "bundle_id": DEMO_BUNDLE_ID,
            "assessment_id": assessment_id,
            "assessment_hash": assessment_hash,
            "segment_count": len(segments),
            "actions": sorted({decision["display_action"] for decision in display_decisions} | {"SHOW"}),
            "revealable_segment_count": sum(1 for segment in segments if segment.eligible_for_reveal),
            "execution_boundary": execution_boundary,
            "display_justification_ref": display_justification[
                "display_justification_ref"
            ],
            "display_admissibility_ref": display_admissibility[
                "display_admissibility_ref"
            ],
            "display_decision_count": len(display_decisions),
        },
        pins=pins,
    ).to_dict()
    serialized_response = str(response)
    residual = detector.detect(serialized_response)
    critical_residual = [
        finding.finding_class.value
        for finding in residual
        if finding.finding_class
        in {
            FindingClass.CREDENTIAL_OR_SECRET,
            FindingClass.EXISTING_STATE_SECRET_MARKING,
        }
    ]
    if critical_residual:
        raise ValueError("governed response contains critical residual sensitive data")
    return OutputGovernanceResult(
        governed_response=response,
        authority_objects=[
            {
                "object_type": "DisplayJustification",
                "object_ref": display_justification["display_justification_ref"],
                "payload": display_justification,
            },
            {
                "object_type": "DisplayAdmissibilityDecision",
                "object_ref": display_admissibility["display_admissibility_ref"],
                "payload": display_admissibility,
            },
        ],
    )
