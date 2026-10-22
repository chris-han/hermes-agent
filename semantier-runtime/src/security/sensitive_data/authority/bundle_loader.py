from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from security.sensitive_data.authority.legal_authority import (
    LegalAuthorityRegistry,
    load_t2_legal_authority_registry,
)
from semantier.sensitive_governance_contracts.hashing import canonical_hash

_REPO_ROOT = Path(__file__).resolve().parents[4]
DEMO_BUNDLE_ID = "soe_sensitive_governance_demo_bundle_v1"
DEMO_BUNDLE_PATH = (
    _REPO_ROOT
    / "bootstrap"
    / "common_knowledge"
    / "bundles"
    / f"{DEMO_BUNDLE_ID}.json"
)
REQUIRED_TIERS = frozenset({"T1", "T2", "T3", "T4"})


@dataclass(frozen=True)
class ActivatedSensitiveGovernanceBundle:
    bundle_id: str
    manifest: dict[str, Any]
    artifacts: dict[str, dict[str, Any]]
    manifest_hash: str
    artifact_hashes: dict[str, str]
    t2_authority_registry: LegalAuthorityRegistry

    @property
    def ontology_artifact(self) -> dict[str, Any]:
        return self._artifact_by_tier("T1")

    @property
    def authority_artifacts(self) -> list[dict[str, Any]]:
        return [self._artifact_by_tier("T2"), self._artifact_by_tier("T3")]

    @property
    def policy_artifact(self) -> dict[str, Any]:
        return self._artifact_by_tier("T4")

    def _artifact_by_tier(self, tier: str) -> dict[str, Any]:
        for artifact in self.artifacts.values():
            if artifact.get("tier") == tier:
                return artifact
        raise ValueError(f"missing required bundle tier: {tier}")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"bundle artifact missing: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"bundle artifact must be a JSON object: {path}")
    return payload


def _parse_aware_iso(value: str, *, field: str) -> None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")


def _validate_separate_approval(payload: dict[str, Any], *, field: str) -> None:
    approval = payload.get(field)
    if not isinstance(approval, dict):
        raise ValueError(f"sensitive governance bundle {field} is required")
    required = ["prepared_by", "approved_by", "approved_at"]
    missing = [key for key in required if not str(approval.get(key) or "").strip()]
    if missing:
        raise ValueError(f"sensitive governance bundle {field} missing: {', '.join(missing)}")
    if approval["prepared_by"] == approval["approved_by"]:
        raise ValueError(f"sensitive governance bundle {field} requires separate approval")
    _parse_aware_iso(str(approval["approved_at"]), field=f"{field} approved_at")


def _validate_bundle_activation_evidence(manifest: dict[str, Any]) -> None:
    _validate_separate_approval(manifest, field="dual_role_approval")
    provenance = manifest.get("source_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("sensitive governance bundle source_provenance is required")
    for key in ["source_type", "source_ref", "prepared_by", "approved_by", "approved_at"]:
        if not str(provenance.get(key) or "").strip():
            raise ValueError(f"sensitive governance bundle source_provenance missing: {key}")
    if provenance["prepared_by"] == provenance["approved_by"]:
        raise ValueError("sensitive governance bundle source_provenance requires separate approval")
    _parse_aware_iso(str(provenance["approved_at"]), field="source_provenance approved_at")
    policy_evidence = manifest.get("policy_test_evidence")
    if not isinstance(policy_evidence, dict):
        raise ValueError("sensitive governance bundle policy_test_evidence is required")
    if not policy_evidence.get("positive") or not policy_evidence.get("negative"):
        raise ValueError("sensitive governance bundle requires positive and negative policy tests")
    activation_evidence = manifest.get("activation_evidence")
    if not isinstance(activation_evidence, dict):
        raise ValueError("sensitive governance bundle activation_evidence is required")
    replay_dry_run_hash = str(activation_evidence.get("replay_dry_run_hash") or "").strip()
    if (
        len(replay_dry_run_hash) != 64
        or any(char not in "0123456789abcdef" for char in replay_dry_run_hash)
    ):
        raise ValueError("sensitive governance bundle replay_dry_run_hash is required")
    event_refs = activation_evidence.get("activation_event_refs")
    if not isinstance(event_refs, list) or "sensitive_governance.bundle_activated" not in event_refs:
        raise ValueError("sensitive governance bundle activation event ref is required")
    required_tiers = set(activation_evidence.get("required_artifact_tiers") or [])
    missing_tiers = sorted(REQUIRED_TIERS - required_tiers)
    if missing_tiers:
        raise ValueError(
            "sensitive governance bundle activation evidence missing tiers: "
            + ", ".join(missing_tiers)
        )


def load_demo_bundle(path: Path = DEMO_BUNDLE_PATH) -> ActivatedSensitiveGovernanceBundle:
    manifest = _read_json(path)
    if manifest.get("bundle_id") != DEMO_BUNDLE_ID:
        raise ValueError("unexpected sensitive governance demo bundle id")
    if manifest.get("lifecycle_state") != "active":
        raise ValueError("sensitive governance demo bundle is not active")
    _validate_bundle_activation_evidence(manifest)

    artifact_entries = manifest.get("artifacts")
    if not isinstance(artifact_entries, list):
        raise ValueError("sensitive governance bundle artifacts must be a list")
    tiers = {str(entry.get("tier")) for entry in artifact_entries if isinstance(entry, dict)}
    missing_tiers = sorted(REQUIRED_TIERS - tiers)
    if missing_tiers:
        raise ValueError(f"sensitive governance bundle missing tiers: {', '.join(missing_tiers)}")

    artifacts: dict[str, dict[str, Any]] = {}
    artifact_hashes: dict[str, str] = {}
    for entry in artifact_entries:
        if not isinstance(entry, dict):
            raise ValueError("bundle artifact entry must be an object")
        artifact_id = str(entry.get("artifact_id") or "").strip()
        rel_path = str(entry.get("path") or "").strip()
        if not artifact_id or not rel_path:
            raise ValueError("bundle artifact entry missing artifact_id or path")
        artifact = _read_json(_REPO_ROOT / rel_path)
        if artifact.get("artifact_id") != artifact_id:
            raise ValueError(f"bundle artifact id mismatch for {artifact_id}")
        if artifact.get("lifecycle_state") != "certified":
            raise ValueError(f"bundle artifact is not certified: {artifact_id}")
        artifacts[artifact_id] = artifact
        artifact_hashes[artifact_id] = canonical_hash(artifact)

    bundle = ActivatedSensitiveGovernanceBundle(
        bundle_id=DEMO_BUNDLE_ID,
        manifest=manifest,
        artifacts=artifacts,
        manifest_hash=canonical_hash(manifest),
        artifact_hashes=artifact_hashes,
        t2_authority_registry=load_t2_legal_authority_registry(
            artifact=next(
                artifact for artifact in artifacts.values() if artifact.get("tier") == "T2"
            ),
            bundle_manifest=manifest,
        ),
    )
    return bundle
