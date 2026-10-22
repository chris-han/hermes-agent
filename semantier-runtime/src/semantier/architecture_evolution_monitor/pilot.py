from __future__ import annotations

from typing import Any

from eos.artifact_hashing import content_hash

from .collectors import collect_architecture_metric_snapshot
from .service import ArchitectureEvolutionMonitorService
from .store import ArchitectureEvolutionMonitorStore


def run_legal_graph_pilot(
    *,
    organization_id: str,
    environment: str,
    observed_from: str,
    observed_to: str,
    non_trigger_workload: dict[str, Any],
    trigger_workload: dict[str, Any],
    evaluation_evidence_refs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    service = ArchitectureEvolutionMonitorService()
    results = []
    for label, workload in (
        ("non_trigger", non_trigger_workload),
        ("trigger", trigger_workload),
    ):
        snapshot = collect_architecture_metric_snapshot(
            organization_id=organization_id,
            environment=environment,
            architecture_profile="legal-authority-relational",
            observed_from=observed_from,
            observed_to=observed_to,
            workload=workload,
            evaluation_evidence_refs=evaluation_evidence_refs,
            snapshot_id=f"legal-graph-pilot:{label}:{content_hash([organization_id, environment, observed_from, observed_to, workload])}",
        )
        evaluation = service.evaluate_snapshot(snapshot.to_dict())
        results.append(
            {
                "case": label,
                "snapshot_ref": evaluation["snapshot"]["content_hash"],
                "assessment_outcome": evaluation["assessment"]["outcome"],
                "classification_loop": evaluation["classification"]["target_loop"],
                "review_ref": (evaluation.get("review_package") or {}).get("content_hash"),
            }
        )
    evidence = {
        "schema_version": "architecture_legal_graph_pilot_evidence.v1",
        "organization_id": organization_id,
        "environment": environment,
        "observed_from": observed_from,
        "observed_to": observed_to,
        "architecture_profile": "legal-authority-relational",
        "cases": results,
        "automatic_adoption": False,
    }
    evidence["content_hash"] = f"sha256:{content_hash(evidence)}"
    ArchitectureEvolutionMonitorStore().put_pilot_evidence(evidence)
    return evidence
