from __future__ import annotations

from datetime import datetime
from typing import Any

from eos.artifact_hashing import content_hash

from .contracts import (
    ArchitectureAssessment,
    ArchitectureLearningClassification,
    ArchitectureMetricSnapshot,
    ArchitectureThresholdProfile,
    AssessmentOutcome,
    ChangeClass,
    INSTITUTIONAL_BOUNDARIES,
    MonitorValidationError,
    TargetLoop,
)

DEFAULT_EVALUATOR_VERSION = "architecture-evolution-monitor.evaluator.v1"
DEFAULT_POLICY_VERSION = "architecture-evolution-monitor.promotion-policy.v1"


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _scope_precedence(profile: ArchitectureThresholdProfile, snapshot: ArchitectureMetricSnapshot) -> int:
    scope = profile.scope
    score = 0
    if scope.get("organization_id") == snapshot.organization_id:
        score += 4
    elif scope.get("organization_id") not in (None, "*"):
        return -1
    if scope.get("environment") == snapshot.environment:
        score += 2
    elif scope.get("environment") not in (None, "*"):
        return -1
    if scope.get("architecture_profile") == snapshot.architecture_profile:
        score += 1
    elif scope.get("architecture_profile") not in (None, "*"):
        return -1
    return score


def select_threshold_profile(
    snapshot: ArchitectureMetricSnapshot,
    profiles: list[ArchitectureThresholdProfile],
) -> ArchitectureThresholdProfile:
    observed_at = _parse_utc(snapshot.observed_to)
    candidates: list[tuple[int, ArchitectureThresholdProfile]] = []
    for profile in profiles:
        if profile.governance_state != "ACTIVE":
            continue
        effective_from = _parse_utc(profile.effective_from)
        effective_to = _parse_utc(profile.effective_to) if profile.effective_to else None
        if observed_at < effective_from or (effective_to is not None and observed_at >= effective_to):
            continue
        precedence = _scope_precedence(profile, snapshot)
        if precedence >= 0:
            candidates.append((precedence, profile))
    if not candidates:
        raise MonitorValidationError("no active threshold profile applies")
    best_score = max(score for score, _profile in candidates)
    best = [profile for score, profile in candidates if score == best_score]
    if len(best) > 1:
        refs = sorted(f"{profile.profile_id}@{profile.profile_version}" for profile in best)
        raise MonitorValidationError(f"overlapping same-precedence threshold profiles: {refs}")
    return best[0]


def _metric_map(snapshot: ArchitectureMetricSnapshot) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric in snapshot.metrics:
        if not isinstance(metric, dict):
            continue
        name = str(metric.get("name") or "").strip()
        if name:
            result[name] = metric.get("value")
    return result


def _evaluate_indicator(indicator: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    metric_name = str(indicator.get("metric") or "").strip()
    reason_code = str(indicator.get("reason_code") or metric_name or "UNKNOWN").strip()
    boundary = str(indicator.get("affected_boundary") or "").strip()
    if not metric_name or metric_name not in metrics:
        return {
            "indicator_id": indicator.get("indicator_id") or metric_name or reason_code,
            "state": "NOT_EVALUABLE",
            "reason_code": f"MISSING_METRIC:{reason_code}",
            "affected_boundary": boundary,
        }
    try:
        value = float(metrics[metric_name])
        threshold = float(indicator["threshold"])
    except (KeyError, TypeError, ValueError):
        return {
            "indicator_id": indicator.get("indicator_id") or metric_name,
            "state": "NOT_EVALUABLE",
            "reason_code": f"INVALID_INDICATOR:{reason_code}",
            "affected_boundary": boundary,
        }
    comparator = str(indicator.get("comparator") or ">=").strip()
    triggered = {
        ">=": value >= threshold,
        ">": value > threshold,
        "<=": value <= threshold,
        "<": value < threshold,
        "==": value == threshold,
    }.get(comparator)
    if triggered is None:
        return {
            "indicator_id": indicator.get("indicator_id") or metric_name,
            "state": "NOT_EVALUABLE",
            "reason_code": f"INVALID_COMPARATOR:{reason_code}",
            "affected_boundary": boundary,
        }
    severity = str(indicator.get("severity") or "observation").strip()
    return {
        "indicator_id": indicator.get("indicator_id") or metric_name,
        "metric": metric_name,
        "value": value,
        "comparator": comparator,
        "threshold": threshold,
        "state": "TRIGGERED" if triggered else "CLEAR",
        "severity": severity,
        "reason_code": reason_code,
        "affected_boundary": boundary,
    }


def assess_snapshot(
    snapshot: ArchitectureMetricSnapshot,
    profile: ArchitectureThresholdProfile,
    *,
    evaluator_version: str = DEFAULT_EVALUATOR_VERSION,
) -> ArchitectureAssessment:
    if snapshot.metric_definition_set_ref != profile.combination_rule.get("metric_definition_set_ref"):
        return ArchitectureAssessment(
            assessment_id=f"assessment:{content_hash([snapshot.content_hash, profile.content_hash, evaluator_version])}",
            metric_snapshot_ref=snapshot.content_hash,
            threshold_profile_ref=profile.content_hash,
            evaluator_version=evaluator_version,
            indicators=[],
            outcome=AssessmentOutcome.NOT_EVALUABLE,
            triggered_reason_codes=["METRIC_DEFINITION_SET_MISMATCH"],
        ).with_hash()

    metrics = _metric_map(snapshot)
    indicators = [_evaluate_indicator(indicator, metrics) for indicator in profile.indicators]
    if not indicators or any(item["state"] == "NOT_EVALUABLE" for item in indicators):
        outcome = AssessmentOutcome.NOT_EVALUABLE
    else:
        triggered = [item for item in indicators if item["state"] == "TRIGGERED"]
        if not triggered:
            outcome = AssessmentOutcome.NO_ACTION
        elif any(item.get("severity") == "urgent_capacity" for item in triggered):
            outcome = AssessmentOutcome.URGENT_CAPACITY_REVIEW
        elif any(item.get("affected_boundary") in INSTITUTIONAL_BOUNDARIES for item in triggered):
            outcome = AssessmentOutcome.ARCHITECTURE_REVIEW_REQUIRED
        elif len(triggered) >= int(profile.combination_rule.get("optimization_review_trigger_count") or 1):
            outcome = AssessmentOutcome.OPTIMIZATION_REVIEW_REQUIRED
        else:
            outcome = AssessmentOutcome.CONTINUE_OBSERVATION
    reason_codes = sorted(
        str(item["reason_code"])
        for item in indicators
        if item["state"] in {"TRIGGERED", "NOT_EVALUABLE"}
    )
    return ArchitectureAssessment(
        assessment_id=f"assessment:{content_hash([snapshot.content_hash, profile.content_hash, evaluator_version])}",
        metric_snapshot_ref=snapshot.content_hash,
        threshold_profile_ref=profile.content_hash,
        evaluator_version=evaluator_version,
        indicators=indicators,
        outcome=outcome,
        triggered_reason_codes=reason_codes,
    ).with_hash()


def classify_assessment(
    assessment: ArchitectureAssessment,
    *,
    policy_version: str = DEFAULT_POLICY_VERSION,
) -> ArchitectureLearningClassification:
    triggered_boundaries = sorted(
        {
            str(item.get("affected_boundary"))
            for item in assessment.indicators
            if item.get("state") == "TRIGGERED" and str(item.get("affected_boundary") or "").strip()
        }
    )
    if assessment.outcome == AssessmentOutcome.NOT_EVALUABLE:
        target_loop = TargetLoop.OPERATIONAL
        change_class = ChangeClass.NOT_EVALUABLE
        promotion_required = False
        reason_codes = sorted({*assessment.triggered_reason_codes, "NOT_EVALUABLE"})
    elif any(boundary in INSTITUTIONAL_BOUNDARIES for boundary in triggered_boundaries):
        target_loop = TargetLoop.INSTITUTIONAL
        change_class = ChangeClass.ARCHITECTURE_CHANGE
        promotion_required = True
        reason_codes = sorted({*assessment.triggered_reason_codes, "INSTITUTIONAL_BOUNDARY"})
    else:
        target_loop = TargetLoop.OPERATIONAL
        change_class = ChangeClass.BOUNDED_TUNING
        promotion_required = False
        reason_codes = list(assessment.triggered_reason_codes)
    return ArchitectureLearningClassification(
        classification_id=f"classification:{content_hash([assessment.content_hash, policy_version])}",
        assessment_ref=assessment.content_hash,
        target_loop=target_loop,
        change_class=change_class,
        affected_boundaries=triggered_boundaries,
        promotion_required=promotion_required,
        reason_codes=reason_codes,
        policy_version=policy_version,
    ).with_hash()
