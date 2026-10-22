from __future__ import annotations

from .contracts import ArchitectureThresholdProfile

AGE_DECISION_PROFILE = ArchitectureThresholdProfile.from_payload(
    {
        "profile_id": "age-decision-profile",
        "profile_version": "v1",
        "scope": {
            "organization_id": "*",
            "environment": "*",
            "architecture_profile": "legal-authority-relational",
        },
        "indicators": [
            {
                "indicator_id": "heterogeneous_traversal_ratio",
                "metric": "legal_authority.heterogeneous_traversal_ratio",
                "comparator": ">=",
                "threshold": 0.55,
                "severity": "institutional_review",
                "reason_code": "LEGAL_AUTHORITY_HETEROGENEOUS_TRAVERSAL_HIGH",
                "affected_boundary": "graph_engine",
            },
            {
                "indicator_id": "recursive_rows_expanded",
                "metric": "legal_authority.recursive_rows_expanded_p95",
                "comparator": ">=",
                "threshold": 50000,
                "severity": "operational_review",
                "reason_code": "LEGAL_AUTHORITY_RECURSIVE_ROWS_HIGH",
                "affected_boundary": "bounded_query_tuning",
            },
            {
                "indicator_id": "path_replay_equality",
                "metric": "legal_authority.path_replay_equality_rate",
                "comparator": "<",
                "threshold": 1.0,
                "severity": "institutional_review",
                "reason_code": "LEGAL_AUTHORITY_REPLAY_INCOMPLETE",
                "affected_boundary": "replay",
            },
        ],
        "combination_rule": {
            "metric_definition_set_ref": "architecture-metric-definitions.v1",
            "optimization_review_trigger_count": 1,
        },
        "cooldown_days": 14,
        "effective_from": "2026-07-29T00:00:00+00:00",
        "effective_to": None,
        "governance_state": "ACTIVE",
        "approved_decision_ref": "docs/plans/2026-07-29-p3-architecture-evolution-monitor-skill.md",
    }
)

DEFAULT_PROFILES = [AGE_DECISION_PROFILE]
