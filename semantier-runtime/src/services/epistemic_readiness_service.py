"""Pure ADR-0018 epistemic-readiness projection over existing quality evidence."""
from __future__ import annotations

from contracts.epistemic_readiness import EpistemicReadinessV1
from contracts.ontology_quality import QualityEvaluationRunV1, QualityProfileV1


class EpistemicReadinessService:
    @staticmethod
    def derive(
        *,
        quality_evaluation: QualityEvaluationRunV1 | None,
        quality_profile: QualityProfileV1 | None,
        structural_ready: bool,
        structural_blocked: bool = False,
        structural_reason_codes: tuple[str, ...] = (),
        repair_candidates: tuple[str, ...] = (),
        provider_failure: bool = False,
    ) -> EpistemicReadinessV1:
        evidence_refs: list[str] = []
        if quality_profile is not None:
            evidence_refs.append(quality_profile.profile_ref)
        if quality_evaluation is not None:
            evidence_refs.append(quality_evaluation.run_ref)
        evidence = tuple(sorted(set(evidence_refs)))

        if structural_blocked:
            return EpistemicReadinessV1(
                disposition="UNRESOLVED",
                structural_qualification="BLOCKED",
                reason_codes=tuple(sorted(set(structural_reason_codes or ("STRUCTURAL_BLOCKED",)))),
                evidence_refs=evidence,
            )
        if provider_failure:
            return EpistemicReadinessV1(
                disposition="UNRESOLVED",
                structural_qualification="UNRESOLVED",
                reason_codes=("PROVIDER_OR_TOOL_FAILURE",),
                evidence_refs=evidence,
                repair_candidates=tuple(sorted(set(repair_candidates))),
            )
        if repair_candidates:
            return EpistemicReadinessV1(
                disposition="REPAIRABLE",
                structural_qualification="QUALIFIED" if structural_ready else "UNRESOLVED",
                reason_codes=tuple(sorted(set(structural_reason_codes or ("REPAIR_AVAILABLE",)))),
                evidence_refs=evidence,
                repair_candidates=tuple(sorted(set(repair_candidates))),
            )
        if not structural_ready:
            return EpistemicReadinessV1(
                disposition="UNRESOLVED",
                structural_qualification="UNRESOLVED",
                reason_codes=tuple(sorted(set(structural_reason_codes or ("STRUCTURAL_QUALIFICATION_UNRESOLVED",)))),
                evidence_refs=evidence,
            )
        if quality_evaluation is None or quality_profile is None:
            return EpistemicReadinessV1(
                disposition="UNRESOLVED",
                structural_qualification="QUALIFIED",
                reason_codes=("QUALITY_EVIDENCE_MISSING",),
                evidence_refs=evidence,
            )
        if quality_evaluation.disposition == "QUALIFIED":
            return EpistemicReadinessV1(
                disposition="AUTO_QUALIFIED",
                structural_qualification="QUALIFIED",
                reason_codes=("QUALITY_QUALIFIED",),
                evidence_refs=evidence,
            )
        if quality_evaluation.disposition == "QUALIFIED_WITH_WARNINGS":
            if quality_profile.allow_warnings:
                return EpistemicReadinessV1(
                    disposition="AUTO_QUALIFIED",
                    structural_qualification="QUALIFIED",
                    reason_codes=("QUALITY_WARNINGS_PERMITTED",),
                    evidence_refs=evidence,
                )
            return EpistemicReadinessV1(
                disposition="UNRESOLVED",
                structural_qualification="QUALIFIED",
                reason_codes=("QUALITY_WARNINGS_NOT_PERMITTED",),
                evidence_refs=evidence,
            )
        if quality_evaluation.disposition == "NOT_EVALUABLE":
            return EpistemicReadinessV1(
                disposition="UNRESOLVED",
                structural_qualification="QUALIFIED",
                reason_codes=("QUALITY_NOT_EVALUABLE",),
                evidence_refs=evidence,
            )
        return EpistemicReadinessV1(
            disposition="UNRESOLVED",
            structural_qualification="QUALIFIED",
            reason_codes=("QUALITY_NOT_QUALIFIED",),
            evidence_refs=evidence,
        )


__all__ = ["EpistemicReadinessService"]
