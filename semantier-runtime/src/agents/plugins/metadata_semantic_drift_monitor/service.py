from __future__ import annotations

from typing import Any

from eos.artifact_hashing import content_hash


PLUGIN_VERSION = "metadata_semantic_drift_monitor.plugin.v1"


def collect_observation(
    *,
    asset_ref: str,
    fields: dict[str, dict[str, Any]],
    source_ref: str,
) -> dict[str, Any]:
    payload = {
        "asset_ref": asset_ref,
        "fields": fields,
        "source_ref": source_ref,
        "collector_version": PLUGIN_VERSION,
    }
    return {
        "schema_version": "metadata_drift_plugin_observation.v1",
        "asset_ref": asset_ref,
        "normalized_payload": {"fields": fields},
        "source_anchors": [source_ref],
        "collector_version": PLUGIN_VERSION,
        "observation_hash": content_hash(payload),
        "authority_state": "evidence_only",
    }


def prepare_semantic_hypotheses(*, observations: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "schema_version": "semantic_drift_hypothesis.v1",
            "hypothesis": observation,
            "authority_state": "candidate_only",
        }
        for observation in observations
    ]

