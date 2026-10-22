from __future__ import annotations

from typing import Any
from uuid import uuid4

from security.sensitive_data.authority.bundle_loader import load_demo_bundle
from security.sensitive_data.policy.evaluator import SensitivePolicyEvaluator
from semantier.sensitive_governance_contracts.enums import DisplayAction, GovernedResponseSegmentKind
from semantier.sensitive_governance_contracts.governed_response import GovernedResponseSegment, build_response
from semantier.sensitive_governance_contracts.hashing import canonical_hash
from semantier.sensitive_governance_contracts.pins import SensitiveGovernancePins
from semantier.sensitive_governance_contracts.registry import CURRENT_CONTRACT_VERSION

DISPLAY_PROFILE_VERSION = "workshop_school_sensitive_display_profile_v1"


def _display_action(decision: dict[str, Any]) -> DisplayAction:
    action = str(decision.get("enforcement_action") or "SHOW")
    return {
        "MASK": DisplayAction.MASK,
        "TOKENIZE": DisplayAction.TOKENIZE,
        "REDACT": DisplayAction.REDACT,
        "BLOCK": DisplayAction.NEVER_RENDER,
    }.get(action, DisplayAction.SHOW)


def _kind(action: DisplayAction) -> GovernedResponseSegmentKind:
    if action == DisplayAction.NEVER_RENDER:
        return GovernedResponseSegmentKind.NEVER_RENDER_NOTICE
    if action == DisplayAction.REDACT:
        return GovernedResponseSegmentKind.REDACTION_NOTICE
    if action in {DisplayAction.MASK, DisplayAction.TOKENIZE}:
        return GovernedResponseSegmentKind.SENSITIVE_TEXT
    return GovernedResponseSegmentKind.TEXT


def govern_workshop_school_output(
    *,
    run_id: str,
    assessment_id: str,
    assessment_hash: str,
    purpose: str,
    source_evidence: dict[str, Any],
    findings: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    execution_boundary: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    bundle = load_demo_bundle()
    authority_refs = SensitivePolicyEvaluator(bundle)._authority_refs()
    source_ref = str(source_evidence.get("governed_upload_ref") or source_evidence.get("source_ref") or "inline")
    source_hash = str(source_evidence.get("content_hash") or "")
    summary = (
        f"已完成教学样本受控分析（{purpose}）。"
        f"来源 {source_ref}；发现 {len(findings)} 项候选。"
        "无法判断项仍需人工确认。"
    )
    segments: list[GovernedResponseSegment] = [
        GovernedResponseSegment(
            segment_id="seg_workshop_summary",
            sequence=0,
            kind=GovernedResponseSegmentKind.TEXT,
            display_action=DisplayAction.SHOW,
            text=summary,
            classification="PUBLIC_INFORMATION",
            authority_refs=authority_refs,
        )
    ]
    decisions_by_finding = {str(item.get("finding_id")): item for item in decisions}
    display_decisions: list[dict[str, Any]] = []
    citations: list[dict[str, Any]] = []
    seen_classes: set[str] = set()
    for finding in findings:
        finding_class = str(finding.get("finding_class") or "")
        if not finding_class or finding_class in seen_classes:
            continue
        seen_classes.add(finding_class)
        decision = decisions_by_finding.get(str(finding.get("finding_id")), {})
        action = _display_action(decision)
        evidence_ref = str(finding.get("evidence_ref") or "")
        citation = {"source_ref": source_ref, "content_hash": source_hash, "locator": evidence_ref}
        if source_evidence.get("parsed_spans") and evidence_ref.startswith("text_range:"):
            _, start, end = evidence_ref.split(":")
            citation["source_locations"] = [
                span for span in source_evidence["parsed_spans"]
                if span["start"] < int(end) and span["end"] > int(start)
            ]
            citation["source_hash"] = source_evidence["source_hash"]
            citation["parser_version"] = source_evidence["parser_version"]
        citations.append(citation)
        display_decisions.append(
            {
                "finding_id": finding.get("finding_id"),
                "finding_class": finding_class,
                "display_action": action.value,
                "evidence_ref": evidence_ref,
                "source_ref": source_ref,
            }
        )
        revealable = action in {DisplayAction.MASK, DisplayAction.TOKENIZE}
        segment_text = {
            DisplayAction.MASK: f"{finding_class} 已遮蔽。",
            DisplayAction.TOKENIZE: f"{finding_class} 已替换为受控令牌。",
            DisplayAction.REDACT: f"[已按 {finding_class} 规则删除]",
            DisplayAction.NEVER_RENDER: "[该片段禁止渲染原文]",
        }.get(action, f"{finding_class}：允许显示分类结论。")
        segments.append(
            GovernedResponseSegment(
                segment_id="seg_workshop_" + canonical_hash(
                    {"finding_id": finding.get("finding_id"), "purpose": purpose, "source_hash": source_hash}
                )[:20],
                sequence=len(segments),
                kind=_kind(action),
                display_action=action,
                text=segment_text,
                classification=finding_class,
                authority_refs=authority_refs,
                eligible_for_reveal=revealable,
                restricted_payload_ref=(str(finding.get("original_value_ref")) if revealable else None),
            )
        )

    justification = {
        "display_justification_ref": "sg_workshop_display_" + canonical_hash(
            {
                "assessment_id": assessment_id,
                "purpose": purpose,
                "source_ref": source_ref,
                "source_hash": source_hash,
                "display_decisions": display_decisions,
            }
        )[:24],
        "assessment_id": assessment_id,
        "purpose": purpose,
        "source_ref": source_ref,
        "source_content_hash": source_hash,
        "display_profile_version": DISPLAY_PROFILE_VERSION,
        "display_decisions": display_decisions,
        "source_citations": citations,
    }
    pins = SensitiveGovernancePins(
        contract_version=CURRENT_CONTRACT_VERSION,
        ontology_version="soe_sensitive_governance_t1_v1",
        authority_bundle_version="soe_sensitive_governance_t2_t3_v1",
        policy_bundle_version="soe_sensitive_governance_t4_v1",
        display_profile_version=DISPLAY_PROFILE_VERSION,
        fixture_manifest_hash=canonical_hash(
            {
                "assessment_id": assessment_id,
                "assessment_hash": assessment_hash,
                "purpose": purpose,
                "source_hash": source_hash,
                "justification_ref": justification["display_justification_ref"],
            }
        ),
    )
    response = build_response(
        response_id=f"sg_response_{uuid4().hex}",
        run_id=run_id,
        segments=segments,
        governance_summary={
            "bundle_id": bundle.bundle_id,
            "assessment_id": assessment_id,
            "assessment_hash": assessment_hash,
            "purpose": purpose,
            "source_ref": source_ref,
            "source_content_hash": source_hash,
            "source_citations": citations,
            "segment_count": len(segments),
            "revealable_segment_count": sum(1 for segment in segments if segment.eligible_for_reveal),
            "actions": sorted({item["display_action"] for item in display_decisions} | {"SHOW"}),
            "execution_boundary": execution_boundary,
            "display_justification_ref": justification["display_justification_ref"],
        },
        pins=pins,
    ).to_dict()
    return response, [{
        "object_type": "WorkshopSchoolDisplayJustification",
        "object_ref": justification["display_justification_ref"],
        "payload": justification,
    }]
