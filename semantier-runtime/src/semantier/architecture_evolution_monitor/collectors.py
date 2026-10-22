from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from eos.artifact_hashing import content_hash

from .contracts import ArchitectureMetricSnapshot, utc_now

METRIC_DEFINITION_SET_REF = "architecture-metric-definitions.v1"


@dataclass(frozen=True)
class MetricCollector:
    family: str
    metric_map: dict[str, str]

    def collect(self, workload: dict[str, Any]) -> list[dict[str, Any]]:
        values = workload.get(self.family)
        if not isinstance(values, dict):
            values = {}
        metrics: list[dict[str, Any]] = []
        for source_key, metric_name in self.metric_map.items():
            if source_key not in values:
                continue
            metrics.append(
                {
                    "name": metric_name,
                    "value": values[source_key],
                    "family": self.family,
                    "source_key": source_key,
                }
            )
        return metrics


COLLECTORS = [
    MetricCollector(
        family="topology",
        metric_map={
            "node_count": "topology.node_count",
            "edge_count": "topology.edge_count",
            "relationship_type_count": "topology.relationship_type_count",
        },
    ),
    MetricCollector(
        family="traversal",
        metric_map={
            "variable_length_ratio": "traversal.variable_length_ratio",
            "heterogeneous_ratio": "traversal.heterogeneous_ratio",
        },
    ),
    MetricCollector(
        family="legal_authority",
        metric_map={
            "heterogeneous_traversal_ratio": "legal_authority.heterogeneous_traversal_ratio",
            "recursive_rows_expanded_p95": "legal_authority.recursive_rows_expanded_p95",
            "path_replay_equality_rate": "legal_authority.path_replay_equality_rate",
            "resolution_latency_ms_p95": "legal_authority.resolution_latency_ms_p95",
            "timeout_rate": "legal_authority.timeout_rate",
            "closure_rebuild_seconds": "legal_authority.closure_rebuild_seconds",
            "resolver_sql_complexity_score": "legal_authority.resolver_sql_complexity_score",
        },
    ),
    MetricCollector(
        family="performance",
        metric_map={
            "latency_ms_p95": "performance.latency_ms_p95",
            "timeout_rate": "performance.timeout_rate",
            "capacity_utilization": "performance.capacity_utilization",
        },
    ),
    MetricCollector(
        family="maintainability",
        metric_map={
            "schema_surgery_count": "maintainability.schema_surgery_count",
            "query_complexity_score": "maintainability.query_complexity_score",
        },
    ),
    MetricCollector(
        family="replay",
        metric_map={
            "trace_completeness_rate": "replay.trace_completeness_rate",
            "path_equality_rate": "replay.path_equality_rate",
        },
    ),
    MetricCollector(
        family="governance",
        metric_map={
            "audit_integrity_rate": "governance.audit_integrity_rate",
            "missing_evidence_count": "governance.missing_evidence_count",
        },
    ),
]


def collect_metrics(workload: dict[str, Any]) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    for collector in COLLECTORS:
        metrics.extend(collector.collect(workload))
    return sorted(metrics, key=lambda item: item["name"])


def collect_architecture_metric_snapshot(
    *,
    organization_id: str,
    environment: str,
    architecture_profile: str,
    observed_from: str,
    observed_to: str,
    workload: dict[str, Any],
    evaluation_evidence_refs: list[dict[str, Any]] | None = None,
    snapshot_id: str | None = None,
) -> ArchitectureMetricSnapshot:
    metrics = collect_metrics(workload)
    resolved_snapshot_id = snapshot_id or f"snapshot:{content_hash([organization_id, environment, architecture_profile, observed_from, observed_to, metrics])}"
    return ArchitectureMetricSnapshot.from_payload(
        {
            "snapshot_id": resolved_snapshot_id,
            "organization_id": organization_id,
            "environment": environment,
            "architecture_profile": architecture_profile,
            "observed_from": observed_from,
            "observed_to": observed_to,
            "metric_definition_set_ref": METRIC_DEFINITION_SET_REF,
            "metrics": metrics,
            "evaluation_evidence_refs": evaluation_evidence_refs or [],
        }
    )


def collect_current_architecture_metric_snapshot(
    *,
    organization_id: str,
    environment: str,
    architecture_profile: str,
    workload: dict[str, Any],
    evaluation_evidence_refs: list[dict[str, Any]] | None = None,
) -> ArchitectureMetricSnapshot:
    now = utc_now()
    return collect_architecture_metric_snapshot(
        organization_id=organization_id,
        environment=environment,
        architecture_profile=architecture_profile,
        observed_from=now,
        observed_to=now,
        workload=workload,
        evaluation_evidence_refs=evaluation_evidence_refs,
    )
