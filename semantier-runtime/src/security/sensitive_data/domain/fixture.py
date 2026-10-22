from __future__ import annotations

from uuid import uuid4

from semantier.sensitive_governance_contracts.enums import (
    DisplayAction,
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


def build_fixed_synthetic_governed_response(
    *,
    run_id: str,
    assessment_id: str,
    assessment_hash: str,
    execution_boundary: dict | None = None,
) -> dict:
    authority_refs = [
        {
            "tier": "T2",
            "artifact_id": "soe_tender_sensitive_information_authority_v1",
            "display_name_zh": "国企招投标敏感信息治理依据",
        },
        {
            "tier": "T4",
            "artifact_id": "soe_tender_demo_policy_v1",
            "display_name_zh": "演示策略：边界穿越与展示控制",
        },
    ]
    pins = SensitiveGovernancePins(
        contract_version=CURRENT_CONTRACT_VERSION,
        ontology_version="soe_sensitive_governance_t1_v1",
        authority_bundle_version="soe_sensitive_governance_t2_t3_v1",
        policy_bundle_version="soe_sensitive_governance_t4_v1",
        display_profile_version="soe_sensitive_display_profile_v1",
        fixture_manifest_hash=canonical_hash(
            {
                "bundle_id": DEMO_BUNDLE_ID,
                "assessment_id": assessment_id,
                "assessment_hash": assessment_hash,
            }
        ),
    )
    response_id = f"sg_response_{uuid4().hex}"
    response = build_response(
        response_id=response_id,
        run_id=run_id,
        segments=[
            GovernedResponseSegment(
                segment_id="seg_public_summary",
                sequence=0,
                kind=GovernedResponseSegmentKind.TEXT,
                display_action=DisplayAction.SHOW,
                text="已完成国企招标文件敏感信息治理检查。",
                classification="PUBLIC_INFORMATION",
                authority_refs=authority_refs,
            ),
            GovernedResponseSegment(
                segment_id="seg_budget_tokenized",
                sequence=1,
                kind=GovernedResponseSegmentKind.SENSITIVE_TEXT,
                display_action=DisplayAction.TOKENIZE,
                text="预算控制价已替换为 [TENDER_BUDGET_TOKEN_1]。",
                classification="TENDER_SENSITIVE_INFORMATION",
                authority_refs=authority_refs,
                eligible_for_reveal=True,
                restricted_payload_ref="restricted://sensitive-governance/demo/budget-token-1",
            ),
            GovernedResponseSegment(
                segment_id="seg_expert_redacted",
                sequence=2,
                kind=GovernedResponseSegmentKind.REDACTION_NOTICE,
                display_action=DisplayAction.REDACT,
                text="[已按评标专家身份信息规则删除]",
                classification="TENDER_EXPERT_INFORMATION",
                authority_refs=authority_refs,
            ),
            GovernedResponseSegment(
                segment_id="seg_secret_never_render",
                sequence=3,
                kind=GovernedResponseSegmentKind.NEVER_RENDER_NOTICE,
                display_action=DisplayAction.NEVER_RENDER,
                text="[存在凭据或既有秘密标识，禁止渲染原文]",
                classification="CREDENTIAL_OR_SECRET",
                authority_refs=authority_refs,
            ),
        ],
        governance_summary={
            "bundle_id": DEMO_BUNDLE_ID,
            "assessment_id": assessment_id,
            "assessment_hash": assessment_hash,
            "segment_count": 4,
            "actions": ["SHOW", "TOKENIZE", "REDACT", "NEVER_RENDER"],
            "revealable_segment_count": 1,
            "execution_boundary": execution_boundary or {},
        },
        pins=pins,
    )
    return response.to_dict()
