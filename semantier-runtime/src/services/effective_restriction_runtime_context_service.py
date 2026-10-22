"""Resolve executable restriction authority from an immutable activation snapshot."""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import re
from typing import Any
from collections.abc import Callable

from contracts.observed_constraint_hypothesis import EffectiveRestrictionRuntimeContextV1
from contracts.ontology_building import ArtifactPinV1
from contracts.tender_sensitive_term_detection_modeling_core import RuntimeContextDTO
from eos.artifact_hashing import content_hash
from services.activation_snapshot_resolver_service import StoreBackedActivationSnapshotResolverService
from storage.governed_rule_store import GovernedRuleStore
from storage.human_grounding_store import HumanGroundingStore
from contracts.accepted_graph_release_v2 import parse_accepted_graph_release_unverified
from services.accepted_graph_rule_projection_service import AcceptedGraphRuleProjectionService
from storage.accepted_graph_release_origin import verify_exact_release_origin_reservation
from storage.ontology_build_store import OntologyBuildStore


def _sha256_ref(value: str) -> str:
    normalized = value if value.startswith("sha256:") else "sha256:" + value
    if re.fullmatch(r"sha256:[0-9a-f]{64}", normalized) is None:
        raise ValueError("authority hash must be an exact lowercase sha256 ref")
    return normalized


class EffectiveRestrictionRuntimeContextService:
    """Build context from stored activation authority, never caller graph pins."""

    def __init__(self, db_path: str | Path, *, clock: Callable[[], datetime] | None = None) -> None:
        self._db_path = Path(db_path)
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._rule_store = GovernedRuleStore(db_path=db_path)
        self._snapshot_resolver = StoreBackedActivationSnapshotResolverService(
            rule_store=self._rule_store
        )

    @staticmethod
    def persist_context(
        *, store: OntologyBuildStore, context: EffectiveRestrictionRuntimeContextV1,
    ) -> ArtifactPinV1:
        """Persist the resolved receipt in the existing workspace artifact store."""
        store.put(context)
        return EffectiveRestrictionRuntimeContextService.context_pin(context)

    @staticmethod
    def context_pin(context: EffectiveRestrictionRuntimeContextV1) -> ArtifactPinV1:
        digest = content_hash(context.model_dump(mode="json", by_alias=True))
        return ArtifactPinV1(
            artifact_kind="effective_restriction_runtime_context.v1",
            artifact_ref="effective-restriction-runtime-context:" + digest[:24],
            artifact_hash="sha256:" + digest,
        )

    def publish_for_policy(
        self, *, store: OntologyBuildStore, organization_id: str, workspace_id: str,
        actor_ref: str, request_id: str, request_hash: str, source_identity_ref: str,
        canonical_source_ir_pin: ArtifactPinV1, graph_intent_pin: ArtifactPinV1,
    ) -> ArtifactPinV1:
        request = dict(
            organization_id=organization_id, workspace_id=workspace_id, actor_ref=actor_ref,
            operation="tender_journey_runtime_binding", request_id=request_id, request_hash=request_hash,
        )
        pin = store.get_operation_result(
            **request, artifact_kind="effective_restriction_runtime_context.v1",
        )
        if pin is None:
            context = self.resolve_for_policy(
                organization_id=organization_id, workspace_id=workspace_id,
                source_identity_ref=source_identity_ref,
                canonical_source_ir_pin=canonical_source_ir_pin, graph_intent_pin=graph_intent_pin,
            )
            pin = store.put_many_with_operation_result(
                **request, artifacts=(context,), result_pin=self.context_pin(context),
            )
        self.resolve_persisted(
            store=store, pin=pin, organization_id=organization_id, workspace_id=workspace_id,
        )
        return pin

    @staticmethod
    def resolve_persisted(
        *, store: OntologyBuildStore, pin: ArtifactPinV1,
        organization_id: str, workspace_id: str,
    ) -> EffectiveRestrictionRuntimeContextV1:
        """Read an exact historical receipt without consulting live authority."""
        if pin.artifact_kind != "effective_restriction_runtime_context.v1":
            raise ValueError("RUNTIME_CONTEXT_RECEIPT_KIND_MISMATCH")
        raw = store.get_raw(
            organization_id=organization_id, workspace_id=workspace_id,
            artifact_ref=pin.artifact_ref,
        )
        if raw is None:
            raise ValueError("RUNTIME_CONTEXT_RECEIPT_NOT_FOUND")
        context = EffectiveRestrictionRuntimeContextV1.model_validate(raw["payload"])
        digest = content_hash(context.model_dump(mode="json", by_alias=True))
        if (
            raw["schema_version"] != context.schema_version
            or raw["artifact_hash"] != pin.artifact_hash
            or pin.artifact_hash != "sha256:" + digest
            or pin.artifact_ref != "effective-restriction-runtime-context:" + digest[:24]
            or context.organization_id != organization_id
            or context.workspace_id != workspace_id
        ):
            raise ValueError("RUNTIME_CONTEXT_RECEIPT_HASH_MISMATCH")
        return context

    def resolve_for_policy(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        source_identity_ref: str,
        canonical_source_ir_pin: ArtifactPinV1,
        graph_intent_pin: ArtifactPinV1,
    ) -> EffectiveRestrictionRuntimeContextV1:
        """Require activated release provenance to match the selected policy."""
        if (
            canonical_source_ir_pin.artifact_kind != "canonical_source_ir"
            or graph_intent_pin.artifact_kind != "graph_intent_brief"
        ):
            raise ValueError("RUNTIME_RELEASE_POLICY_PIN_KIND_MISMATCH")
        context = self.resolve_current(
            organization_id=organization_id, workspace_id=workspace_id,
        )
        store = HumanGroundingStore(self._db_path, graph_backend=None)
        try:
            lineage = store.read_authoritative_release_lineage(
                context.accepted_graph_release_ref,
                organization_id=organization_id, workspace_id=workspace_id,
            )
        finally:
            store._conn.close()
        expected = {
            "accepted_graph_release_hash": context.accepted_graph_release_hash,
            "source_identity_ref": source_identity_ref,
            "canonical_source_ir_ref": canonical_source_ir_pin.artifact_ref,
            "canonical_source_ir_hash": canonical_source_ir_pin.artifact_hash,
            "graph_intent_ref": graph_intent_pin.artifact_ref,
            "graph_intent_hash": graph_intent_pin.artifact_hash,
        }
        if any(lineage.get(key) != value for key, value in expected.items()):
            raise ValueError("RUNTIME_RELEASE_POLICY_LINEAGE_MISMATCH")
        return context

    def resolve_current(
        self,
        *,
        organization_id: str,
        workspace_id: str,
    ) -> EffectiveRestrictionRuntimeContextV1:
        now = self._clock().astimezone(timezone.utc).replace(microsecond=0)
        snapshots = self._rule_store.list_activation_snapshots(
            organization_id=organization_id, workspace_id=workspace_id
        )
        applicable = [
            item for item in snapshots
            if datetime.fromisoformat(str(item["effective_at"]).replace("Z", "+00:00")) <= now
        ]
        snapshot = applicable[0] if applicable else None
        if snapshot is None:
            raise ValueError("activation snapshot is not governed for this context")
        snapshot_payload = {
            "organization_id": snapshot["organization_id"],
            "workspace_id": snapshot["workspace_id"],
            "principal_entitlement_ref": snapshot["principal_entitlement_ref"],
            "activation_resolver_policy_version": snapshot["activation_resolver_policy_version"],
            "effective_at": snapshot["effective_at"],
            "included_activation_ids": snapshot["included_activation_ids"],
            "excluded_activation_evidence": snapshot["excluded_activation_evidence"],
            "prior_snapshot_hash": snapshot["prior_snapshot_hash"],
        }
        expected_resolved = content_hash(snapshot_payload)
        row_payload = {
            "activation_set_snapshot_id": snapshot["activation_set_snapshot_id"],
            **snapshot_payload,
            "resolved_activation_set_hash": expected_resolved,
            "created_at": snapshot["created_at"],
        }
        if (
            snapshot["resolved_activation_set_hash"] != expected_resolved
            or snapshot["content_hash"] != content_hash(row_payload)
        ):
            raise ValueError("activation snapshot hash mismatch")
        activation_set_snapshot_ref = str(snapshot["activation_set_snapshot_id"])
        activation_set_snapshot_hash = _sha256_ref(str(snapshot["content_hash"]))
        runtime_context = RuntimeContextDTO(
            organization_id=organization_id,
            workspace_id=workspace_id,
            actor_id="system:effective-restriction-context-resolver",
            actor_role="system",
        )
        rules, rule_set_hash = self._snapshot_resolver.list_rules_for_activation_snapshot(
            activation_set_snapshot_id=activation_set_snapshot_ref,
            resolved_activation_set_hash=str(snapshot["resolved_activation_set_hash"]),
            context=runtime_context,
            effective_at=now.isoformat(),
        )
        if not rules:
            raise ValueError("activation snapshot contains no executable governed rules")
        lineage: list[dict[str, Any]] = []
        for rule in rules:
            scope = rule.get("applicability_scope")
            if not isinstance(scope, dict):
                raise ValueError("governed rule is missing graph projection lineage")
            release_ref = scope.get("source_graph_version")
            release_hash = scope.get("source_graph_release_hash")
            pattern_ref = scope.get("source_graph_pattern_id")
            pattern_hash = scope.get("source_graph_pattern_hash")
            if not all(isinstance(item, str) and item for item in (release_ref, release_hash, pattern_ref, pattern_hash)):
                raise ValueError("governed rule is missing graph projection lineage")
            lineage.append(
                {
                    "rule_version_ref": rule["rule_version_id"],
                    "rule_version_hash": _sha256_ref(str(rule["content_hash"])),
                    "accepted_graph_release_ref": release_ref,
                    "accepted_graph_release_hash": _sha256_ref(release_hash),
                    "canonical_pattern_ref": pattern_ref,
                    "canonical_pattern_hash": _sha256_ref(pattern_hash),
                }
            )
        release_pairs = {
            (item["accepted_graph_release_ref"], item["accepted_graph_release_hash"])
            for item in lineage
        }
        if len(release_pairs) != 1:
            raise ValueError("activation snapshot spans stale or mixed graph releases")
        lineage.sort(key=lambda item: item["rule_version_ref"].encode("utf-8"))
        projection_hash = "sha256:" + content_hash(lineage)
        projection_ref = "governed-rule-projection:" + projection_hash[7:31]
        release_ref, release_hash = next(iter(release_pairs))
        grounding_store = HumanGroundingStore(self._db_path, graph_backend=None)
        release_payload = grounding_store.get_graph_release(
            release_ref,
            organization_id=organization_id,
            workspace_id=workspace_id,
        )
        release = parse_accepted_graph_release_unverified(release_payload)
        if release.release_hash != release_hash:
            raise ValueError("governed rule projection release pin is stale")
        verify_exact_release_origin_reservation(
            grounding_store._conn,
            graph_version=release.graph_version,
            authority_state="authoritative",
        )
        authoritative_patterns = {
            (pattern.pattern_id, pattern.pattern_hash)
            for pattern, _binding in AcceptedGraphRuleProjectionService._load_v2_patterns(
                grounding_store,
                release,
                organization_id=organization_id,
                workspace_id=workspace_id,
            )
        }
        for item in lineage:
            if (item["canonical_pattern_ref"], item["canonical_pattern_hash"]) not in authoritative_patterns:
                raise ValueError("governed rule projection pattern pin is stale")
        payload = {
            "schemaVersion": "effective_restriction_runtime_context.v1",
            "organizationId": organization_id,
            "workspaceId": workspace_id,
            "acceptedGraphReleaseRef": release_ref,
            "acceptedGraphReleaseHash": release_hash,
            "governedRuleProjectionRef": projection_ref,
            "governedRuleProjectionHash": projection_hash,
            "orderedGovernedRuleVersionRefs": [item["rule_version_ref"] for item in lineage],
            "orderedGovernedRuleVersionSetHash": _sha256_ref(rule_set_hash),
            "activationSetSnapshotRef": activation_set_snapshot_ref,
            "activationSetSnapshotHash": activation_set_snapshot_hash,
            "activationResolverPolicyVersion": snapshot["activation_resolver_policy_version"],
            "resolvedActivationSetHash": _sha256_ref(snapshot["resolved_activation_set_hash"]),
            "effectiveAt": now.isoformat(),
        }
        payload["effectiveContextResolverHash"] = "sha256:" + content_hash(
            {**payload, "resolver_version": "effective_restriction_runtime_context_resolver.v1"}
        )
        return EffectiveRestrictionRuntimeContextV1.model_validate(payload)


__all__ = ["EffectiveRestrictionRuntimeContextService"]
