from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from eos.artifact_hashing import content_hash

from .contracts import DriftFamily, LearningRoute, LOOP_B_FAMILIES
from .store import (
    BASELINE_AMBIGUOUS,
    BASELINE_UNAVAILABLE,
    InstitutionalDriftStore,
)


class InstitutionalDriftService:
    def __init__(self, db_path: str | Path | None = None) -> None:
        self.store = InstitutionalDriftStore(db_path)

    def register_policy(self, payload: dict[str, Any]) -> dict[str, Any]:
        policy = self.store.register_policy(
            organization_id=str(payload["organization_id"]),
            workspace_id=payload.get("workspace_id"),
            asset_selector=dict(payload.get("asset_selector") or {"asset_ref": "*"}),
            monitored_dimensions=[
                str(item) for item in payload.get("monitored_dimensions", [])
            ],
            cadence=str(payload.get("cadence") or "daily"),
            baseline_resolution_mode=str(
                payload.get("baseline_resolution_mode") or "governed_active"
            ),
            severity_thresholds=dict(payload.get("severity_thresholds") or {}),
            allowed_self_healing_level=str(
                payload.get("allowed_self_healing_level") or "H1"
            ),
            owner_role=str(payload.get("owner_role") or "data_owner"),
            reviewer_role=str(payload.get("reviewer_role") or "governance_reviewer"),
            governance_state=str(payload.get("governance_state") or "DRAFT"),
            policy_version=str(payload.get("policy_version") or "v1"),
            effective_from=str(payload["effective_from"]),
            effective_to=payload.get("effective_to"),
            created_by=str(payload.get("created_by") or "system"),
        )
        return policy.to_record()

    def certify_policy(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.store.certify_policy(
            policy_id=str(payload["policy_id"]),
            reviewer_ref=str(payload["reviewer_ref"]),
            reviewer_role=str(payload["reviewer_role"]),
            evidence_hash=str(payload["evidence_hash"]),
        )

    def activate_policy(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.store.activate_policy(
            policy_id=str(payload["policy_id"]),
            approver_ref=str(payload["approver_ref"]),
            approver_role=str(payload["approver_role"]),
            certification_event_id=str(payload["certification_event_id"]),
        )

    def create_baseline(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.store.create_baseline_snapshot(
            organization_id=str(payload["organization_id"]),
            workspace_id=payload.get("workspace_id"),
            asset_ref=str(payload["asset_ref"]),
            metadata_asset_version_ref=str(payload["metadata_asset_version_ref"]),
            activation_set_snapshot_id=str(payload["activation_set_snapshot_id"]),
            resolved_activation_set_hash=str(payload["resolved_activation_set_hash"]),
            ontology_version_ref=str(payload["ontology_version_ref"]),
            semantic_contract_version_ref=str(payload["semantic_contract_version_ref"]),
            data_contract_version_ref=str(payload["data_contract_version_ref"]),
            policy_version_refs=[
                str(item) for item in payload.get("policy_version_refs", [])
            ],
            source_hashes=dict(payload.get("source_hashes") or {}),
            effective_at=str(payload["effective_at"]),
        )

    def observe_and_classify(self, payload: dict[str, Any]) -> dict[str, Any]:
        organization_id = str(payload["organization_id"])
        workspace_id = payload.get("workspace_id")
        asset_ref = str(payload["asset_ref"])
        observed_at = str(payload.get("observed_to") or payload["effective_at"])
        policy_resolution = self.store.resolve_policy(
            organization_id=organization_id,
            workspace_id=workspace_id,
            as_of=observed_at,
            asset_ref=asset_ref,
        )
        if not policy_resolution["ok"]:
            return {"ok": False, "error_code": policy_resolution["error_code"]}
        baseline_resolution = self.store.resolve_baseline(
            organization_id=organization_id,
            workspace_id=workspace_id,
            asset_ref=asset_ref,
            effective_at=observed_at,
        )
        if not baseline_resolution["ok"]:
            return {"ok": False, "error_code": baseline_resolution["error_code"]}
        baseline = baseline_resolution["baseline"]
        policy = policy_resolution["policy"]
        observation = self.store.record_observation(
            policy_id=policy["policy_id"],
            organization_id=organization_id,
            workspace_id=workspace_id,
            asset_ref=asset_ref,
            observed_from=str(payload["observed_from"]),
            observed_to=str(payload["observed_to"]),
            payload=dict(payload.get("observed_payload") or {}),
            source_anchors=[str(item) for item in payload.get("source_anchors", [])],
            collector_version=str(payload.get("collector_version") or "metadata_drift_collector.v1"),
            complete=bool(payload.get("complete", True)),
        )
        run = self.store.create_or_resume_run(
            policy_id=policy["policy_id"],
            organization_id=organization_id,
            workspace_id=workspace_id,
            baseline_snapshot_id=baseline["baseline_snapshot_id"],
            baseline_snapshot_hash=baseline["baseline_snapshot_hash"],
            observation_id=observation.observation_id,
            observation_window=observation.observation_window,
        )
        if run["status"] in {"PENDING", "RUNNING"}:
            self.store.claim_run_lease(
                run_id=run["run_id"],
                lease_owner=str(payload.get("lease_owner") or "institutional_drift_service"),
                lease_expires_at=(
                    datetime.now(timezone.utc) + timedelta(minutes=10)
                ).isoformat(),
            )
        differences = structural_diff(
            baseline_payload=dict(payload.get("baseline_payload") or {}),
            observed_payload=dict(payload.get("observed_payload") or {}),
        )
        semantic_hypotheses = [
            {
                "schema_version": "semantic_drift_hypothesis.v1",
                "hypothesis": str(item),
                "authority_state": "candidate_only",
            }
            for item in payload.get("semantic_hypotheses", [])
        ]
        route = route_finding(
            drift_family=str(payload.get("drift_family") or infer_drift_family(differences)),
            authority_impact=dict(payload.get("authority_impact") or {}),
            complete=bool(payload.get("complete", True)),
            differences=differences,
        )
        finding = self.store.persist_finding(
            run_id=run["run_id"],
            organization_id=organization_id,
            workspace_id=workspace_id,
            drift_family=route["drift_family"],
            drift_subtype=str(payload.get("drift_subtype") or route["drift_family"]),
            baseline_snapshot_id=baseline["baseline_snapshot_id"],
            observation_id=observation.observation_id,
            deterministic_differences=differences,
            semantic_hypotheses=semantic_hypotheses,
            evidence_completeness="COMPLETE" if payload.get("complete", True) else "PARTIAL",
            authority_impact=route["authority_impact"],
            selected_learning_route=route["selected_learning_route"],
            reason_codes=route["reason_codes"],
            severity=route["severity"],
            urgency=route["urgency"],
        )
        return {
            "ok": True,
            "policy": policy,
            "baseline": baseline,
            "observation": observation.to_record(),
            "run": run,
            "finding": finding,
        }

    def verify_recovery(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.store.record_recovery_verification(
            finding_id=str(payload["finding_id"]),
            remediation_ref=str(payload["remediation_ref"]),
            deployed_artifact_refs=[
                str(item) for item in payload.get("deployed_artifact_refs", [])
            ],
            pre_observation_hash=str(payload["pre_observation_hash"]),
            post_observation_hash=str(payload["post_observation_hash"]),
            replay_comparison_result=str(payload["replay_comparison_result"]),
            non_interference_result=str(payload["non_interference_result"]),
            expected_outcome=str(payload["expected_outcome"]),
            observed_outcome=str(payload["observed_outcome"]),
            regression_result=str(payload["regression_result"]),
            evidence=dict(payload.get("evidence") or {}),
        )

    def audit_export(self, finding_id: str) -> dict[str, Any]:
        return self.store.audit_export(finding_id=finding_id)

    def dashboard(self, organization_id: str) -> dict[str, Any]:
        return self.store.dashboard(organization_id=organization_id)

    def uat_operational_and_institutional_paths(self) -> dict[str, Any]:
        policy = self.register_policy(
            {
                "organization_id": "org-uat",
                "asset_selector": {"asset_ref": "dataset:orders"},
                "monitored_dimensions": ["physical_metadata", "classification"],
                "cadence": "daily",
                "effective_from": "2026-07-30T00:00:00+00:00",
                "created_by": "uat-owner",
            }
        )
        cert = self.store.certify_policy(
            policy_id=policy["policy_id"],
            reviewer_ref="uat-reviewer",
            reviewer_role="governance_reviewer",
            evidence_hash=policy["policy_hash"],
        )
        self.store.activate_policy(
            policy_id=policy["policy_id"],
            approver_ref="uat-approver",
            approver_role="governance_reviewer",
            certification_event_id=cert["event_id"],
        )
        baseline = self.create_baseline(
            {
                "organization_id": "org-uat",
                "asset_ref": "dataset:orders",
                "metadata_asset_version_ref": "metadata_asset_version:orders:v1",
                "activation_set_snapshot_id": "kass:uat:v1",
                "resolved_activation_set_hash": content_hash({"kass": "uat-v1"}),
                "ontology_version_ref": "ontology:orders:v1",
                "semantic_contract_version_ref": "semantic_contract:orders:v1",
                "data_contract_version_ref": "data_contract:orders:v1",
                "policy_version_refs": [policy["policy_id"]],
                "source_hashes": {"schema": content_hash({"orders": "v1"})},
                "effective_at": "2026-07-30T00:00:00+00:00",
            }
        )
        base_payload = {"fields": {"order_id": {"type": "string", "required": True}}}
        loop_a = self.observe_and_classify(
            {
                "organization_id": "org-uat",
                "asset_ref": "dataset:orders",
                "effective_at": "2026-07-30T01:00:00+00:00",
                "observed_from": "2026-07-30T00:00:00+00:00",
                "observed_to": "2026-07-30T01:00:00+00:00",
                "baseline_payload": base_payload,
                "observed_payload": {
                    "fields": {
                        "order_id": {"type": "string", "required": True},
                        "optional_note": {"type": "string", "required": False},
                    }
                },
                "drift_family": "physical_metadata",
                "authority_impact": {"authority_tiers": [], "regulated_domain": False},
            }
        )
        loop_b = self.observe_and_classify(
            {
                "organization_id": "org-uat",
                "asset_ref": "dataset:orders",
                "effective_at": "2026-07-30T02:00:00+00:00",
                "observed_from": "2026-07-30T01:00:00+00:00",
                "observed_to": "2026-07-30T02:00:00+00:00",
                "baseline_payload": base_payload,
                "observed_payload": {
                    "fields": {
                        "order_id": {"type": "string", "required": True},
                        "buyer_phone": {"type": "string", "classification": "PII"},
                    }
                },
                "drift_family": "classification",
                "authority_impact": {"authority_tiers": ["T3"], "regulated_domain": True},
                "semantic_hypotheses": ["buyer_phone may change privacy classification"],
            }
        )
        recovery = self.verify_recovery(
            {
                "finding_id": loop_a["finding"]["finding_id"],
                "remediation_ref": "pr:parser-alias-orders-note",
                "deployed_artifact_refs": ["artifact:parser-alias-orders-note"],
                "pre_observation_hash": loop_a["observation"]["observation_hash"],
                "post_observation_hash": content_hash({"orders": "recovered"}),
                "replay_comparison_result": "MATCH",
                "non_interference_result": "PASS",
                "expected_outcome": "INGESTION_RECOVERED",
                "observed_outcome": "INGESTION_RECOVERED",
                "regression_result": "NO_REGRESSION",
                "evidence": {"baseline_snapshot_id": baseline["baseline_snapshot_id"]},
            }
        )
        return {"loop_a": loop_a, "loop_b": loop_b, "recovery": recovery}


def structural_diff(*, baseline_payload: dict[str, Any], observed_payload: dict[str, Any]) -> list[dict[str, Any]]:
    differences: list[dict[str, Any]] = []
    base_fields = dict(baseline_payload.get("fields") or {})
    observed_fields = dict(observed_payload.get("fields") or {})
    for field in sorted(set(observed_fields) - set(base_fields)):
        differences.append(
            {
                "difference_type": "field_added",
                "field": field,
                "observed": observed_fields[field],
                "backward_compatible": not bool(observed_fields[field].get("required")),
            }
        )
    for field in sorted(set(base_fields) - set(observed_fields)):
        differences.append(
            {"difference_type": "field_removed", "field": field, "backward_compatible": False}
        )
    for field in sorted(set(base_fields) & set(observed_fields)):
        before = base_fields[field]
        after = observed_fields[field]
        for key in sorted(set(before) | set(after)):
            if before.get(key) == after.get(key):
                continue
            differences.append(
                {
                    "difference_type": f"field_{key}_changed",
                    "field": field,
                    "baseline": before.get(key),
                    "observed": after.get(key),
                    "backward_compatible": False,
                }
            )
    return differences


def infer_drift_family(differences: list[dict[str, Any]]) -> str:
    if not differences:
        return DriftFamily.PHYSICAL_METADATA.value
    if any("classification" in str(diff.get("difference_type")) for diff in differences):
        return DriftFamily.CLASSIFICATION.value
    return DriftFamily.PHYSICAL_METADATA.value


def route_finding(
    *,
    drift_family: str,
    authority_impact: dict[str, Any],
    complete: bool,
    differences: list[dict[str, Any]],
) -> dict[str, Any]:
    reason_codes: list[str] = []
    if not complete:
        return {
            "drift_family": drift_family,
            "authority_impact": authority_impact,
            "selected_learning_route": LearningRoute.INSUFFICIENT_EVIDENCE.value,
            "reason_codes": ["EVIDENCE_INCOMPLETE"],
            "severity": "LOW",
            "urgency": "OBSERVE",
        }
    if not differences:
        return {
            "drift_family": drift_family,
            "authority_impact": authority_impact,
            "selected_learning_route": LearningRoute.INFORMATION_ONLY.value,
            "reason_codes": ["NO_MATERIAL_DRIFT"],
            "severity": "LOW",
            "urgency": "NONE",
        }
    authority_tiers = set(authority_impact.get("authority_tiers") or [])
    institutional = (
        drift_family in LOOP_B_FAMILIES
        or bool(authority_tiers & {"T1", "T2", "T3", "T4"})
        or bool(authority_impact.get("regulated_domain"))
        or bool(authority_impact.get("mandatory_behavior_changed"))
        or bool(authority_impact.get("uncertain_institutional_impact"))
    )
    if institutional:
        reason_codes.append("INSTITUTIONAL_IMPACT")
        route = LearningRoute.LOOP_B_INSTITUTIONAL.value
        severity = "HIGH"
        urgency = "REVIEW"
    elif all(diff.get("backward_compatible") for diff in differences):
        reason_codes.append("BOUNDED_OPERATIONAL_CHANGE")
        route = LearningRoute.LOOP_A_OPERATIONAL.value
        severity = "LOW"
        urgency = "NORMAL"
    else:
        reason_codes.append("OPERATIONAL_INCIDENT")
        route = LearningRoute.INCIDENT_ONLY.value
        severity = "MEDIUM"
        urgency = "REMEDIATE"
    return {
        "drift_family": drift_family,
        "authority_impact": authority_impact,
        "selected_learning_route": route,
        "reason_codes": reason_codes,
        "severity": severity,
        "urgency": urgency,
    }
