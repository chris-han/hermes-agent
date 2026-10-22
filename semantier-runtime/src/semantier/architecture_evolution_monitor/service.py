from __future__ import annotations

from typing import Any

from eos.artifact_hashing import content_hash

from .contracts import (
    ArchitectureMetricSnapshot,
    ArchitectureThresholdProfile,
    ReviewState,
    utc_now,
)
from .engine import assess_snapshot, classify_assessment, select_threshold_profile
from .profiles import DEFAULT_PROFILES
from .scheduler import MonitorSchedule, list_monitor_schedules, register_monitor_schedule
from .store import ArchitectureEvolutionMonitorStore

SPEC_REFS = [
    "docs/canonical/architecture.md",
    "docs/canonical/institutional-lifecycle-runtime-spec.md",
    "docs/derived/architecture-evolution-monitor-spec.md",
    "docs/derived/governed-execution-boundary-native-design.md",
    "docs/derived/gateway-unified-multitenant-design.md",
    "docs/derived/knowledge_tier_implementation_spec.md",
]


class ArchitectureEvolutionMonitorService:
    def __init__(
        self,
        *,
        store: ArchitectureEvolutionMonitorStore | None = None,
        profiles: list[ArchitectureThresholdProfile] | None = None,
    ) -> None:
        self.store = store or ArchitectureEvolutionMonitorStore()
        self.profiles = profiles or DEFAULT_PROFILES

    def evaluate_snapshot(self, payload: dict[str, Any]) -> dict[str, Any]:
        snapshot = ArchitectureMetricSnapshot.from_payload(payload)
        profile = select_threshold_profile(snapshot, self.profiles)
        assessment = assess_snapshot(snapshot, profile)
        classification = classify_assessment(assessment)
        review_package = None
        self.store.put_profile(profile.to_dict())
        self.store.put_snapshot(snapshot.organization_id, snapshot.to_dict())
        self.store.put_assessment(snapshot.organization_id, assessment.to_dict())
        self.store.put_classification(snapshot.organization_id, classification.to_dict())
        if classification.promotion_required or assessment.outcome.value.endswith("REVIEW_REQUIRED"):
            review_package = self._build_review_package(
                snapshot=snapshot.to_dict(),
                assessment=assessment.to_dict(),
                classification=classification.to_dict(),
                profile=profile.to_dict(),
            )
            self.store.put_review_package(snapshot.organization_id, review_package)
        return {
            "snapshot": snapshot.to_dict(),
            "threshold_profile": profile.to_dict(),
            "assessment": assessment.to_dict(),
            "classification": classification.to_dict(),
            "review_package": review_package,
        }

    def scheduled_evaluate_snapshot(self, payload: dict[str, Any]) -> dict[str, Any]:
        schedule_ref = str(payload.get("schedule_ref") or "").strip()
        if not schedule_ref:
            raise ValueError("schedule_ref is required for scheduled evaluation")
        result = self.evaluate_snapshot(payload)
        result["schedule_ref"] = schedule_ref
        result["notification"] = {
            "kind": "attention_only",
            "review_id": (result.get("review_package") or {}).get("review_id"),
            "schedule_ref": schedule_ref,
        }
        return result

    def create_schedule(
        self,
        *,
        scope_id: str,
        schedule_ref: str,
        organization_id: str,
        environment: str,
        architecture_profile: str,
        cron_expression: str,
    ) -> dict[str, Any]:
        if not schedule_ref.strip():
            raise ValueError("schedule_ref is required")
        if not cron_expression.strip():
            raise ValueError("cron_expression is required")
        schedule = MonitorSchedule(
            schedule_ref=schedule_ref,
            organization_id=organization_id,
            environment=environment,
            architecture_profile=architecture_profile,
            cron_expression=cron_expression,
        )
        return register_monitor_schedule(scope_id=scope_id, schedule=schedule)

    def list_schedules(self, scope_id: str) -> list[dict[str, Any]]:
        return list_monitor_schedules(scope_id)

    def list_scheduled_runs(self, scope_id: str, schedule_ref: str) -> list[dict[str, Any]]:
        return self.store.list_scheduled_runs(scope_id, schedule_ref)

    def _build_review_package(
        self,
        *,
        snapshot: dict[str, Any],
        assessment: dict[str, Any],
        classification: dict[str, Any],
        profile: dict[str, Any],
    ) -> dict[str, Any]:
        evidence_refs = [
            {
                "ref": snapshot["snapshot_id"],
                "schema_version": snapshot["schema_version"],
                "content_hash": snapshot["content_hash"],
            },
            {
                "ref": profile["profile_id"],
                "schema_version": profile["schema_version"],
                "content_hash": profile["content_hash"],
            },
            *snapshot.get("evaluation_evidence_refs", []),
        ]
        package = {
            "schema_version": "architecture_review_package.v1",
            "review_id": f"review:{content_hash([assessment['content_hash'], classification['content_hash']])}",
            "assessment_ref": assessment["content_hash"],
            "classification_ref": classification["content_hash"],
            "evidence_refs": evidence_refs,
            "spec_refs": SPEC_REFS,
            "current_architecture": {
                "architecture_profile": snapshot["architecture_profile"],
                "environment": snapshot["environment"],
                "automatic_mutation_allowed": False,
            },
            "workload_profile": {
                "metric_definition_set_ref": snapshot["metric_definition_set_ref"],
                "observed_from": snapshot["observed_from"],
                "observed_to": snapshot["observed_to"],
            },
            "alternatives": [
                {"alternative_id": "continue_current", "summary": "Retain current architecture"},
                {"alternative_id": "relational_tuning", "summary": "Tune relational indexes and queries"},
                {
                    "alternative_id": "postgresql_19_sql_pgq",
                    "summary": "Use PostgreSQL 19 SQL/PGQ property graph and GRAPH_TABLE over relational facts",
                    "selection_priority": "evaluate_before_extensions",
                },
                {"alternative_id": "materialized_graph_projection", "summary": "Add governed materialized graph projection"},
                {
                    "alternative_id": "postgres_age",
                    "summary": "Evaluate PostgreSQL plus AGE only after native SQL/PGQ insufficiency is evidenced",
                    "requires_evidence": [
                        "heterogeneous_variable_length_traversal_gap",
                        "cypher_or_equivalent_expressiveness_gap",
                        "path_provenance_gap",
                        "deterministic_replay_gap",
                    ],
                },
            ],
            "impacts": {
                "expected_benefit": "bounded architecture review only",
                "migration": "requires separate ADR and implementation plan",
                "non_interference": "no automatic source, store, or infrastructure mutation",
                "rollback": "post-change evidence can open a new review but cannot roll back silently",
                "replay": "assessment and classification refs are hash-pinned",
                "security": "restricted payloads remain referenced by safe hashes",
                "tenant": "organization-scoped review package",
            },
            "unresolved_uncertainties": [
                "benchmark confirmation may be requested through P1 evaluation_evidence.v1"
            ],
            "accountable_owner": "architecture-governance",
            "required_decision": "continue, request benchmark, request ADR, request coding-agent analysis, or reject",
            "state": ReviewState.DRAFT.value,
            "version": 1,
        }
        package["content_hash"] = f"sha256:{content_hash(package)}"
        return package

    def list_reviews(self, organization_id: str) -> list[dict[str, Any]]:
        return self.store.list_review_packages(organization_id)

    def get_review(self, organization_id: str, review_id: str) -> dict[str, Any] | None:
        return self.store.get_review_package(organization_id, review_id)

    def decide_review(
        self,
        *,
        organization_id: str,
        review_id: str,
        expected_version: int,
        expected_hash: str,
        to_state: str,
        idempotency_key: str,
        actor_ref: str,
        reason_code: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        state = ReviewState(to_state)
        if state in {ReviewState.ADR_REQUESTED, ReviewState.CODEX_ANALYSIS_REQUESTED}:
            self._require_institutional_review(
                organization_id=organization_id,
                review_id=review_id,
            )
        decision_seed = [
            organization_id,
            review_id,
            idempotency_key,
            state.value,
            actor_ref,
            reason_code,
            payload,
        ]
        decision = {
            "schema_version": "architecture_review_decision.v1",
            "decision_id": f"decision:{content_hash(decision_seed)}",
            "review_id": review_id,
            "idempotency_key": idempotency_key,
            "actor_ref": actor_ref,
            "reason_code": reason_code,
            "decision_payload": payload,
            "occurred_at": utc_now(),
            "signature": f"sha256:{content_hash(['architecture-review-decision-signature.v1', decision_seed])}",
        }
        return self.store.transition_review(
            organization_id=organization_id,
            review_id=review_id,
            expected_version=expected_version,
            expected_hash=expected_hash,
            to_state=state,
            decision=decision,
        )

    def request_handoff(
        self,
        *,
        organization_id: str,
        review_id: str,
        expected_version: int,
        expected_hash: str,
        to_state: str,
        idempotency_key: str,
        actor_ref: str,
        reason_code: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        decision = self.decide_review(
            organization_id=organization_id,
            review_id=review_id,
            expected_version=expected_version,
            expected_hash=expected_hash,
            to_state=to_state,
            idempotency_key=idempotency_key,
            actor_ref=actor_ref,
            reason_code=reason_code,
            payload=payload,
        )
        grant_type = "codex_analysis" if to_state == "CODEX_ANALYSIS_REQUESTED" else "adr"
        grant = self.store.put_handoff_grant(
            organization_id=organization_id,
            review_id=review_id,
            grant_id=f"grant:{content_hash([review_id, idempotency_key, grant_type])}",
            grant_type=grant_type,
            payload={
                "decision_id": decision["decision_id"],
                "actor_ref": actor_ref,
                "reason_code": reason_code,
                "bounded_scope": payload.get("bounded_scope") or ["analysis_only"],
            },
        )
        return {"decision": decision, "grant": grant}

    def revoke_handoff(
        self,
        *,
        organization_id: str,
        review_id: str,
        expected_version: int,
        expected_hash: str,
        idempotency_key: str,
        actor_ref: str,
        reason_code: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        decision = self.decide_review(
            organization_id=organization_id,
            review_id=review_id,
            expected_version=expected_version,
            expected_hash=expected_hash,
            to_state="OPEN",
            idempotency_key=idempotency_key,
            actor_ref=actor_ref,
            reason_code=reason_code,
            payload=payload,
        )
        return {
            "decision": decision,
            "revoked_grants": self.store.revoke_handoff_grants(
                organization_id=organization_id,
                review_id=review_id,
            ),
        }

    def _require_institutional_review(self, *, organization_id: str, review_id: str) -> None:
        review = self.get_review(organization_id, review_id)
        if review is None:
            raise KeyError(review_id)
        classification = self.store.get_classification_by_ref(
            organization_id,
            str(review.get("classification_ref") or ""),
        )
        if not classification or classification.get("target_loop") != "INSTITUTIONAL":
            raise ValueError("ADR and coding-agent handoffs require institutional classification")

    def dashboard(self, organization_id: str) -> dict[str, Any]:
        reviews = self.list_reviews(organization_id)
        return {
            "schema_version": "architecture_monitor_dashboard.v1",
            "organization_id": organization_id,
            "review_count": len(reviews),
            "open_attention_count": sum(1 for item in reviews if item["state"] in {"DRAFT", "OPEN"}),
            "review_hashes": [
                {"review_id": item["review_id"], "content_hash": item["content_hash"], "state": item["state"]}
                for item in reviews
            ],
            "notifications": [
                {"kind": "attention_only", "review_id": item["review_id"], "state": item["state"]}
                for item in reviews
                if item["state"] in {"DRAFT", "OPEN"}
            ],
        }

    def inspect_review(self, organization_id: str, review_id: str) -> dict[str, Any]:
        review = self.get_review(organization_id, review_id)
        if review is None:
            raise KeyError(review_id)
        persisted_hash = str(review.get("content_hash") or "")
        recomputed_hash = f"sha256:{content_hash({k: v for k, v in review.items() if k != 'content_hash'})}"
        return {
            "schema_version": "architecture_monitor_inspector.v1",
            "organization_id": organization_id,
            "review": review,
            "persisted_hash": persisted_hash,
            "recomputed_hash": recomputed_hash,
            "hash_reconciled": persisted_hash == recomputed_hash,
            "handoff_grants": self.store.list_handoff_grants(organization_id, review_id),
        }
