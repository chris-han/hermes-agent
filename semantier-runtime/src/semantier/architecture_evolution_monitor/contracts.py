from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from eos.artifact_hashing import content_hash


class MonitorValidationError(ValueError):
    pass


class AssessmentOutcome(str, Enum):
    NO_ACTION = "NO_ACTION"
    CONTINUE_OBSERVATION = "CONTINUE_OBSERVATION"
    OPTIMIZATION_REVIEW_REQUIRED = "OPTIMIZATION_REVIEW_REQUIRED"
    ARCHITECTURE_REVIEW_REQUIRED = "ARCHITECTURE_REVIEW_REQUIRED"
    URGENT_CAPACITY_REVIEW = "URGENT_CAPACITY_REVIEW"
    NOT_EVALUABLE = "NOT_EVALUABLE"


class TargetLoop(str, Enum):
    OPERATIONAL = "OPERATIONAL"
    INSTITUTIONAL = "INSTITUTIONAL"


class ChangeClass(str, Enum):
    BOUNDED_TUNING = "BOUNDED_TUNING"
    ARCHITECTURE_CHANGE = "ARCHITECTURE_CHANGE"
    NOT_EVALUABLE = "NOT_EVALUABLE"


class ReviewState(str, Enum):
    DRAFT = "DRAFT"
    OPEN = "OPEN"
    CONTINUE_CURRENT = "CONTINUE_CURRENT"
    BENCHMARK_REQUESTED = "BENCHMARK_REQUESTED"
    ADR_REQUESTED = "ADR_REQUESTED"
    CODEX_ANALYSIS_REQUESTED = "CODEX_ANALYSIS_REQUESTED"
    REJECTED = "REJECTED"
    CLOSED = "CLOSED"


TERMINAL_REVIEW_STATES = {
    ReviewState.CONTINUE_CURRENT,
    ReviewState.BENCHMARK_REQUESTED,
    ReviewState.ADR_REQUESTED,
    ReviewState.CODEX_ANALYSIS_REQUESTED,
    ReviewState.REJECTED,
    ReviewState.CLOSED,
}


INSTITUTIONAL_BOUNDARIES = {
    "persistence_authority",
    "graph_engine",
    "route_policy",
    "authorization",
    "tenant_boundary",
    "replay",
    "audit",
    "canonical_specification",
    "platform_dependency",
    "cross_team_commitment",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_aware_utc(value: str, field_name: str) -> None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise MonitorValidationError(f"{field_name} must be ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise MonitorValidationError(f"{field_name} must be timezone-aware UTC")
    if parsed.utcoffset().total_seconds() != 0:
        raise MonitorValidationError(f"{field_name} must use UTC offset")


def with_content_hash(payload: dict[str, Any]) -> dict[str, Any]:
    stable = dict(payload)
    stable.pop("content_hash", None)
    stable["content_hash"] = f"sha256:{content_hash(stable)}"
    return stable


@dataclass(frozen=True)
class EvaluationEvidenceRef:
    evidence_ref: str
    evidence_schema_version: str = "evaluation_evidence.v1"
    content_hash: str = ""

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "EvaluationEvidenceRef":
        schema = str(payload.get("evidence_schema_version") or "").strip()
        if schema != "evaluation_evidence.v1":
            raise MonitorValidationError("P1 evidence refs must use evaluation_evidence.v1")
        return cls(
            evidence_ref=str(payload.get("evidence_ref") or "").strip(),
            evidence_schema_version=schema,
            content_hash=str(payload.get("content_hash") or "").strip(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_ref": self.evidence_ref,
            "evidence_schema_version": self.evidence_schema_version,
            "content_hash": self.content_hash,
        }


@dataclass(frozen=True)
class ArchitectureMetricSnapshot:
    snapshot_id: str
    organization_id: str
    environment: str
    architecture_profile: str
    observed_from: str
    observed_to: str
    metric_definition_set_ref: str
    metrics: list[dict[str, Any]]
    evaluation_evidence_refs: list[EvaluationEvidenceRef] = field(default_factory=list)
    schema_version: str = "architecture_metric_snapshot.v1"
    content_hash: str = ""

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ArchitectureMetricSnapshot":
        require_aware_utc(str(payload.get("observed_from") or ""), "observed_from")
        require_aware_utc(str(payload.get("observed_to") or ""), "observed_to")
        evidence_refs = [
            EvaluationEvidenceRef.from_payload(item)
            for item in payload.get("evaluation_evidence_refs", [])
        ]
        snapshot = cls(
            snapshot_id=str(payload.get("snapshot_id") or "").strip(),
            organization_id=str(payload.get("organization_id") or "").strip(),
            environment=str(payload.get("environment") or "").strip(),
            architecture_profile=str(payload.get("architecture_profile") or "").strip(),
            observed_from=str(payload.get("observed_from") or "").strip(),
            observed_to=str(payload.get("observed_to") or "").strip(),
            metric_definition_set_ref=str(payload.get("metric_definition_set_ref") or "").strip(),
            metrics=list(payload.get("metrics") or []),
            evaluation_evidence_refs=evidence_refs,
            content_hash=str(payload.get("content_hash") or "").strip(),
        )
        if snapshot.environment not in {"development", "staging", "production"}:
            raise MonitorValidationError("environment must be development, staging, or production")
        if not snapshot.snapshot_id or not snapshot.organization_id:
            raise MonitorValidationError("snapshot_id and organization_id are required")
        return snapshot.with_hash()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "snapshot_id": self.snapshot_id,
            "organization_id": self.organization_id,
            "environment": self.environment,
            "architecture_profile": self.architecture_profile,
            "observed_from": self.observed_from,
            "observed_to": self.observed_to,
            "metric_definition_set_ref": self.metric_definition_set_ref,
            "metrics": self.metrics,
            "evaluation_evidence_refs": [ref.to_dict() for ref in self.evaluation_evidence_refs],
            "content_hash": self.content_hash,
        }

    def with_hash(self) -> "ArchitectureMetricSnapshot":
        payload = with_content_hash(self.to_dict())
        return ArchitectureMetricSnapshot.from_hashed_payload(payload)

    @classmethod
    def from_hashed_payload(cls, payload: dict[str, Any]) -> "ArchitectureMetricSnapshot":
        return cls(
            snapshot_id=payload["snapshot_id"],
            organization_id=payload["organization_id"],
            environment=payload["environment"],
            architecture_profile=payload["architecture_profile"],
            observed_from=payload["observed_from"],
            observed_to=payload["observed_to"],
            metric_definition_set_ref=payload["metric_definition_set_ref"],
            metrics=list(payload["metrics"]),
            evaluation_evidence_refs=[
                EvaluationEvidenceRef.from_payload(item)
                for item in payload.get("evaluation_evidence_refs", [])
            ],
            content_hash=payload["content_hash"],
        )


@dataclass(frozen=True)
class ArchitectureThresholdProfile:
    profile_id: str
    profile_version: str
    scope: dict[str, Any]
    indicators: list[dict[str, Any]]
    combination_rule: dict[str, Any]
    cooldown_days: int
    effective_from: str
    effective_to: str | None
    governance_state: str
    approved_decision_ref: str
    schema_version: str = "architecture_threshold_profile.v1"
    content_hash: str = ""

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ArchitectureThresholdProfile":
        require_aware_utc(str(payload.get("effective_from") or ""), "effective_from")
        if payload.get("effective_to") is not None:
            require_aware_utc(str(payload.get("effective_to") or ""), "effective_to")
        profile = cls(
            profile_id=str(payload.get("profile_id") or "").strip(),
            profile_version=str(payload.get("profile_version") or "").strip(),
            scope=dict(payload.get("scope") or {}),
            indicators=list(payload.get("indicators") or []),
            combination_rule=dict(payload.get("combination_rule") or {}),
            cooldown_days=int(payload.get("cooldown_days") or 0),
            effective_from=str(payload.get("effective_from") or "").strip(),
            effective_to=(
                str(payload.get("effective_to")).strip()
                if payload.get("effective_to") is not None
                else None
            ),
            governance_state=str(payload.get("governance_state") or "").strip(),
            approved_decision_ref=str(payload.get("approved_decision_ref") or "").strip(),
            content_hash=str(payload.get("content_hash") or "").strip(),
        )
        if profile.governance_state not in {"PROPOSED", "CERTIFIED", "ACTIVE", "SUPERSEDED", "RETIRED"}:
            raise MonitorValidationError("invalid governance_state")
        if not profile.profile_id or not profile.profile_version:
            raise MonitorValidationError("profile_id and profile_version are required")
        return profile.with_hash()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "scope": self.scope,
            "indicators": self.indicators,
            "combination_rule": self.combination_rule,
            "cooldown_days": self.cooldown_days,
            "effective_from": self.effective_from,
            "effective_to": self.effective_to,
            "governance_state": self.governance_state,
            "approved_decision_ref": self.approved_decision_ref,
            "content_hash": self.content_hash,
        }

    def with_hash(self) -> "ArchitectureThresholdProfile":
        return ArchitectureThresholdProfile.from_hashed_payload(with_content_hash(self.to_dict()))

    @classmethod
    def from_hashed_payload(cls, payload: dict[str, Any]) -> "ArchitectureThresholdProfile":
        return cls(
            profile_id=payload["profile_id"],
            profile_version=payload["profile_version"],
            scope=dict(payload["scope"]),
            indicators=list(payload["indicators"]),
            combination_rule=dict(payload["combination_rule"]),
            cooldown_days=int(payload["cooldown_days"]),
            effective_from=payload["effective_from"],
            effective_to=payload.get("effective_to"),
            governance_state=payload["governance_state"],
            approved_decision_ref=payload["approved_decision_ref"],
            content_hash=payload["content_hash"],
        )


@dataclass(frozen=True)
class ArchitectureAssessment:
    assessment_id: str
    metric_snapshot_ref: str
    threshold_profile_ref: str
    evaluator_version: str
    indicators: list[dict[str, Any]]
    outcome: AssessmentOutcome
    triggered_reason_codes: list[str]
    schema_version: str = "architecture_indicator_assessment.v1"
    content_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "assessment_id": self.assessment_id,
            "metric_snapshot_ref": self.metric_snapshot_ref,
            "threshold_profile_ref": self.threshold_profile_ref,
            "evaluator_version": self.evaluator_version,
            "indicators": self.indicators,
            "outcome": self.outcome.value,
            "triggered_reason_codes": self.triggered_reason_codes,
            "content_hash": self.content_hash,
        }

    def with_hash(self) -> "ArchitectureAssessment":
        payload = with_content_hash(self.to_dict())
        return ArchitectureAssessment(
            assessment_id=payload["assessment_id"],
            metric_snapshot_ref=payload["metric_snapshot_ref"],
            threshold_profile_ref=payload["threshold_profile_ref"],
            evaluator_version=payload["evaluator_version"],
            indicators=list(payload["indicators"]),
            outcome=AssessmentOutcome(payload["outcome"]),
            triggered_reason_codes=list(payload["triggered_reason_codes"]),
            content_hash=payload["content_hash"],
        )


@dataclass(frozen=True)
class ArchitectureLearningClassification:
    classification_id: str
    assessment_ref: str
    target_loop: TargetLoop
    change_class: ChangeClass
    affected_boundaries: list[str]
    promotion_required: bool
    reason_codes: list[str]
    policy_version: str
    schema_version: str = "architecture_learning_classification.v1"
    content_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "classification_id": self.classification_id,
            "assessment_ref": self.assessment_ref,
            "target_loop": self.target_loop.value,
            "change_class": self.change_class.value,
            "affected_boundaries": self.affected_boundaries,
            "promotion_required": self.promotion_required,
            "reason_codes": self.reason_codes,
            "policy_version": self.policy_version,
            "content_hash": self.content_hash,
        }

    def with_hash(self) -> "ArchitectureLearningClassification":
        payload = with_content_hash(self.to_dict())
        return ArchitectureLearningClassification(
            classification_id=payload["classification_id"],
            assessment_ref=payload["assessment_ref"],
            target_loop=TargetLoop(payload["target_loop"]),
            change_class=ChangeClass(payload["change_class"]),
            affected_boundaries=list(payload["affected_boundaries"]),
            promotion_required=bool(payload["promotion_required"]),
            reason_codes=list(payload["reason_codes"]),
            policy_version=payload["policy_version"],
            content_hash=payload["content_hash"],
        )


@dataclass(frozen=True)
class ReviewPackage:
    review_id: str
    assessment_ref: str
    classification_ref: str
    evidence_refs: list[dict[str, Any]]
    spec_refs: list[str]
    current_architecture: dict[str, Any]
    workload_profile: dict[str, Any]
    alternatives: list[dict[str, Any]]
    impacts: dict[str, Any]
    unresolved_uncertainties: list[str]
    accountable_owner: str
    required_decision: str
    state: ReviewState = ReviewState.DRAFT
    version: int = 1
    schema_version: str = "architecture_review_package.v1"
    content_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "review_id": self.review_id,
            "assessment_ref": self.assessment_ref,
            "classification_ref": self.classification_ref,
            "evidence_refs": self.evidence_refs,
            "spec_refs": self.spec_refs,
            "current_architecture": self.current_architecture,
            "workload_profile": self.workload_profile,
            "alternatives": self.alternatives,
            "impacts": self.impacts,
            "unresolved_uncertainties": self.unresolved_uncertainties,
            "accountable_owner": self.accountable_owner,
            "required_decision": self.required_decision,
            "state": self.state.value,
            "version": self.version,
            "content_hash": self.content_hash,
        }

    def with_hash(self) -> "ReviewPackage":
        payload = with_content_hash(self.to_dict())
        return ReviewPackage(
            review_id=payload["review_id"],
            assessment_ref=payload["assessment_ref"],
            classification_ref=payload["classification_ref"],
            evidence_refs=list(payload["evidence_refs"]),
            spec_refs=list(payload["spec_refs"]),
            current_architecture=dict(payload["current_architecture"]),
            workload_profile=dict(payload["workload_profile"]),
            alternatives=list(payload["alternatives"]),
            impacts=dict(payload["impacts"]),
            unresolved_uncertainties=list(payload["unresolved_uncertainties"]),
            accountable_owner=payload["accountable_owner"],
            required_decision=payload["required_decision"],
            state=ReviewState(payload["state"]),
            version=int(payload["version"]),
            content_hash=payload["content_hash"],
        )


@dataclass(frozen=True)
class ReviewDecision:
    decision_id: str
    review_id: str
    from_state: ReviewState
    to_state: ReviewState
    idempotency_key: str
    actor_ref: str
    reason_code: str
    decision_payload: dict[str, Any]
    occurred_at: str
    signature: str
    content_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "architecture_review_decision.v1",
            "decision_id": self.decision_id,
            "review_id": self.review_id,
            "from_state": self.from_state.value,
            "to_state": self.to_state.value,
            "idempotency_key": self.idempotency_key,
            "actor_ref": self.actor_ref,
            "reason_code": self.reason_code,
            "decision_payload": self.decision_payload,
            "occurred_at": self.occurred_at,
            "signature": self.signature,
            "content_hash": self.content_hash,
        }

    def with_hash(self) -> "ReviewDecision":
        payload = with_content_hash(self.to_dict())
        return ReviewDecision(
            decision_id=payload["decision_id"],
            review_id=payload["review_id"],
            from_state=ReviewState(payload["from_state"]),
            to_state=ReviewState(payload["to_state"]),
            idempotency_key=payload["idempotency_key"],
            actor_ref=payload["actor_ref"],
            reason_code=payload["reason_code"],
            decision_payload=dict(payload["decision_payload"]),
            occurred_at=payload["occurred_at"],
            signature=payload["signature"],
            content_hash=payload["content_hash"],
        )
