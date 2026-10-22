"""Deterministic occurrence-to-assertion gate for tender clauses.

Lexical occurrences are evidence candidates.  They become semantic assertions
only when the clause expresses the governed issue rather than an example,
information field, technical context, exception, or conditioned obligation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AssertionState(str, Enum):
    SUPPORTED = "supported"
    NOT_ASSERTED = "not_asserted"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class SemanticAssertionDecision:
    state: AssertionState
    reason_code: str
    preserved_condition: str | None = None


def classify_semantic_assertion(*, clause: str, term: str, issue_type: str) -> SemanticAssertionDecision:
    text = "".join(clause.split())

    if issue_type == "brand_designation":
        if ("如" in text and "等" in text) or "参见材料样板" in text:
            return _not_asserted("example_or_sample_context")
    elif issue_type == "financial_threshold":
        threshold_markers = ("不少于", "不低于", "至少", "达到", "超过", "亿元", "万元")
        if not any(marker in text for marker in threshold_markers):
            return _not_asserted("financial_evidence_without_threshold")
    elif issue_type == "geographic_discrimination":
        context_markers = ("本工程位于", "地址", "数据中心", "地方标准", "验收时")
        if any(marker in text for marker in context_markers):
            return _not_asserted("project_or_standard_location_context")
        if "情况" in text or "是否" in text:
            return _not_asserted("location_information_without_restriction")
    elif issue_type == "unbalanced_contract_risk":
        if "任何一方" in text or "甲乙" in text:
            return _not_asserted("bilateral_contract_clause")
    elif issue_type == "forced_lowest_price_award":
        if "不能作为中标" in text or "评标基准价" in text or "评分" in text:
            return _not_asserted("price_scoring_or_explicit_non_award")
    elif issue_type == "enterprise_scale_restriction":
        if "自评估" in text or "情况表" in text:
            return _not_asserted("information_collection_without_threshold")
    elif issue_type == "operating_history_threshold":
        if any(marker in text for marker in ("违法", "违规", "记录")):
            return _not_asserted("misconduct_lookback_not_operating_age")
    elif issue_type == "capital_scale_threshold":
        threshold_markers = ("不少于", "不低于", "至少", "达到", "超过")
        if not any(marker in text for marker in threshold_markers):
            return _not_asserted("capital_information_without_threshold")
    elif issue_type == "intellectual_property_requirement":
        if "或" in text and "授权" in text:
            return _not_asserted("ownership_or_authorization_allowed")
    elif issue_type == "fixed_bid_price_restriction":
        if any(marker in text for marker in ("咨询服务费", "合同", "乙方")):
            return _not_asserted("post_award_payment_model")
    elif issue_type == "unnecessary_certification_requirement":
        if "需经强制性认证的" in text:
            return SemanticAssertionDecision(
                state=AssertionState.NOT_ASSERTED,
                reason_code="conditioned_compulsory_certification",
                preserved_condition="需经强制性认证的",
            )
        if "建造师" in term and any(marker in text for marker in ("建筑工程", "项目总负责人")):
            return _not_asserted("role_relevant_certification_requires_domain_evidence")
    elif issue_type == "past_performance_requirement":
        if "可提交" in text and "核验" in text:
            return _not_asserted("optional_relevant_performance_evidence")
    elif issue_type in {"personnel_experience_threshold", "personnel_qualification_scoring"}:
        if "依法" in text and "法定资格" in text:
            return _not_asserted("legally_required_role_qualification")
    elif issue_type == "exclusionary_constraint":
        if "项目编号" in text or "电子归档" in text:
            return _not_asserted("document_identifier_constraint")
    elif issue_type == "supplier_identity_restriction":
        if "联合体投标的" in text and "应提交" in text:
            return _not_asserted("consortium_permitted_with_evidence")
    elif issue_type in {"biased_evaluation_criterion", "biased_personnel_scoring"}:
        if "统一公开标准" in text:
            return _not_asserted("open_relevant_evaluation_dimensions")

    return SemanticAssertionDecision(AssertionState.SUPPORTED, "explicit_issue_assertion")


def _not_asserted(reason_code: str) -> SemanticAssertionDecision:
    return SemanticAssertionDecision(AssertionState.NOT_ASSERTED, reason_code)
