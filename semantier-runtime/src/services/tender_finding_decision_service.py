"""Deterministic tender finding adjudication from pinned context."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from contracts.tender_sensitive_term_detection_modeling_core import (
    ConstraintFrameV01,
    FrameMatchResultDTO,
)
from contracts.frame_match_result_v2 import FrameMatchResultDTOV2
from contracts.governed_frame_finding_v1 import FrameFindingDecisionV2


DecisionStatus = Literal["candidate", "violation", "acceptable", "uncertain", "escalate"]


@dataclass(frozen=True)
class TenderFindingDecisionContext:
    schema_version: Literal["contextgraph_tender_finding_decision_context.v1"]
    finding_id: str
    concept_id: str
    meaning_match: bool
    relation_match: bool
    qualification_context: Literal[
        "requirement",
        "scoring",
        "description",
        "negation",
        "unknown",
    ]
    polarity: Literal["positive", "negative", "unclear"]
    modality: Literal["mandatory", "optional", "prohibited", "unknown"]
    rationale: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class FrameFindingDecision:
    status: DecisionStatus
    structural_match: bool
    semantic_relation: str
    semantic_profile_id: str
    rationale: str


class TenderFindingDecisionService:
    def decide(self, context: TenderFindingDecisionContext, *, escalation: bool = False) -> DecisionStatus:
        if escalation:
            return "escalate"
        if context.qualification_context == "negation" or context.polarity == "negative":
            return "acceptable" if context.meaning_match else "uncertain"
        if not context.meaning_match:
            return "acceptable"
        if not context.relation_match:
            return "uncertain"
        if context.qualification_context == "requirement" and context.modality == "mandatory":
            return "violation"
        if context.modality == "prohibited":
            return "violation"
        return "uncertain"

    def decide_frame_match(
        self,
        *,
        frame_match: FrameMatchResultDTO,
        observed_frame: ConstraintFrameV01,
    ) -> FrameFindingDecision:
        """Accept the matcher's explicit result without legacy lexical recomputation."""

        relation = frame_match.semantic_relation
        if relation == "NO_MATCH":
            return FrameFindingDecision(
                status="acceptable",
                structural_match=False,
                semantic_relation=relation,
                semantic_profile_id=frame_match.semantic_profile_id,
                rationale="Deterministic frame slot unification returned NO_MATCH.",
            )
        if relation not in {"LEXICAL_MATCH", "FRAME_EQUIVALENT", "FRAME_ENTAILS"}:
            raise ValueError(f"unsupported frame semantic relation: {relation}")
        return FrameFindingDecision(
            status="candidate",
            structural_match=True,
            semantic_relation=relation,
            semantic_profile_id=frame_match.semantic_profile_id,
            rationale=(
                "Accepted structural match; legal disposition remains a separate reviewer decision. "
                f"Observed modality={observed_frame.modality_ref}."
            ),
        )

    def decide_frame_match_v2(
        self, *, frame_match: FrameMatchResultDTOV2
    ) -> FrameFindingDecisionV2:
        """Map governed three-valued matching to a separate finding disposition."""

        if frame_match.frame_match_state == "UNKNOWN":
            return FrameFindingDecisionV2(
                status="uncertain",
                structural_match=False,
                semantic_relation=frame_match.semantic_relation,
                frame_match_state=frame_match.frame_match_state,
                match_result_ref=frame_match.result_id,
                match_result_hash=frame_match.result_hash,
                rationale=(
                    "Material semantic inputs remain unresolved; no Boolean or "
                    "violation conclusion is authorized."
                ),
            )
        if frame_match.frame_match_state == "NO_MATCH":
            return FrameFindingDecisionV2(
                status="acceptable",
                structural_match=False,
                semantic_relation=frame_match.semantic_relation,
                frame_match_state=frame_match.frame_match_state,
                match_result_ref=frame_match.result_id,
                match_result_hash=frame_match.result_hash,
                rationale="Resolved deterministic frame comparison returned NO_MATCH.",
            )
        return FrameFindingDecisionV2(
            status="candidate",
            structural_match=True,
            semantic_relation=frame_match.semantic_relation,
            frame_match_state=frame_match.frame_match_state,
            match_result_ref=frame_match.result_id,
            match_result_hash=frame_match.result_hash,
            rationale=(
                "Governed structural match is a candidate; legal disposition remains "
                "a separate applicability/reviewer decision."
            ),
        )

    def context_from_evidence(
        self,
        *,
        finding_id: str,
        concept_id: str,
        evidence_text: str,
        semantic_relation: str,
        semantic_reason: str,
        supported: bool,
    ) -> TenderFindingDecisionContext:
        compact = "".join(evidence_text.split())
        negated = any(marker in compact for marker in ("不要求", "无需", "不得要求", "不作为资格条件"))
        scoring = any(marker in compact for marker in ("评分", "加分", "得分", "评审因素"))
        mandatory = any(marker in compact for marker in ("必须", "须", "应", "要求", "资格条件", "具有"))
        optional = any(marker in compact for marker in ("可", "可以", "自愿", "鼓励"))
        if negated:
            qualification_context = "negation"
            polarity = "negative"
            modality = "unknown"
        elif scoring:
            qualification_context = "scoring"
            polarity = "positive"
            modality = "optional" if optional else "mandatory"
        elif mandatory:
            qualification_context = "requirement"
            polarity = "positive"
            modality = "mandatory"
        elif optional:
            qualification_context = "description"
            polarity = "positive"
            modality = "optional"
        else:
            qualification_context = "unknown"
            polarity = "unclear"
            modality = "unknown"
        relation_match = semantic_relation == "exact"
        return TenderFindingDecisionContext(
            schema_version="contextgraph_tender_finding_decision_context.v1",
            finding_id=finding_id,
            concept_id=concept_id,
            meaning_match=supported or negated,
            relation_match=relation_match,
            qualification_context=qualification_context,
            polarity=polarity,
            modality=modality,
            rationale=semantic_reason,
        )
