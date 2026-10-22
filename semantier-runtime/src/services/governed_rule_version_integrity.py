"""Dependency-light integrity helpers for exact governed rule-version bytes."""

from __future__ import annotations

from typing import Any, Mapping

from eos.artifact_hashing import canonical_json, content_hash

_RULE_HASH_FIELDS = (
    "rule_version_id",
    "rule_candidate_id",
    "organization_id",
    "authority_domain",
    "semantic_tier",
    "jurisdiction",
    "rule_family",
    "rule_state",
    "source_graph_version",
    "source_graph_release_hash",
    "source_graph_rule_id",
    "source_anchor_refs",
    "applicability_scope",
    "governed_rule_text",
    "extracted_rationale",
    "severity",
    "confidence",
    "uncertainty_notes",
    "reviewer_edits",
    "approval_evidence",
    "test_evidence",
    "activation_refs",
    "supersedes_rule_version_id",
)


def governed_rule_version_hash(rule: Mapping[str, Any]) -> str:
    value = str(rule.get("content_hash") or "")
    return value if value.startswith("sha256:") else "sha256:" + value


def verify_governed_rule_version_bytes(rule: Mapping[str, Any]) -> None:
    payload = {field: rule.get(field) for field in _RULE_HASH_FIELDS}
    if str(rule.get("content_hash") or "") != content_hash(canonical_json(payload)):
        raise ValueError("governed rule version content hash mismatch")


__all__ = ["governed_rule_version_hash", "verify_governed_rule_version_bytes"]
