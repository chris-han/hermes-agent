"""Governed runtime adapter for holographic retrieval-backed projection context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

from memory.holographic_provider import (
    HolographicMemoryProvider,
    ProjectionContextSnapshot,
    sanitize_projection_context,
)

from eos.admissibility_policy_store import ensure_default_admissibility_policy
from eos.artifact_hashing import content_hash
from eos.projection_context_store import create_snapshot

PROVIDER_REF = "hermes-agent/plugins/memory/holographic"
ADAPTER_VERSION = "semantier_memory_adapter_v1"
_ALLOWED_TAG_KEYS = {"authority_domain", "tag_authority_level", "semantic_tier"}
_FORBIDDEN_ONTOLOGY_KEYS = {"redefines_ontology_primitives", "ontology_overrides", "ontology_assertions"}
_SEMANTIC_TIER_ORDER = {
    "T0": 0,
    "T1": 1,
    "T2": 2,
    "T3": 3,
    "T4": 4,
    "T5": 5,
    "T6": 6,
}


def _iter_source_refs(provider_snapshot: ProjectionContextSnapshot) -> Iterable[Dict[str, Any]]:
    for source_ref in provider_snapshot.source_refs or []:
        if isinstance(source_ref, dict):
            yield dict(source_ref)


def _requires_downgraded_authority(
    source_tier: Optional[str],
    active_semantic_tiers: Iterable[str],
) -> bool:
    if not source_tier:
        return False
    source_rank = _SEMANTIC_TIER_ORDER.get(source_tier)
    active_ranks = [
        _SEMANTIC_TIER_ORDER[tier]
        for tier in active_semantic_tiers
        if tier in _SEMANTIC_TIER_ORDER
    ]
    if source_rank is None or not active_ranks:
        return False
    return source_rank < min(active_ranks)


@dataclass
class MemoryAdapterResult:
    projection_context_ref: str
    projection_context_hash: str
    snapshot: Dict[str, Any]


class SemantierMemoryAdapter:
    """Filters provider output and persists immutable projection context snapshots."""

    def __init__(
        self,
        provider: HolographicMemoryProvider,
        *,
        adapter_version: str = ADAPTER_VERSION,
        provider_ref: str = PROVIDER_REF,
    ) -> None:
        self.provider = provider
        self.adapter_version = adapter_version
        self.provider_ref = provider_ref

    def build_projection_context(
        self,
        *,
        org_id: str,
        rea_event: dict,
        authority_domain: str,
        projection_bundle_id: str,
        O_v: str,
        O_tag_v: str,
        K_v: str,
        ConstraintBundle_v: str,
        PrecedenceGraph_v: str,
        ResolutionRules_v: str,
        query_intent: str,
        projection_context_ref: str,
        created_by: str,
        event_id: Optional[str] = None,
        seed_context: Optional[Dict[str, Any]] = None,
        active_source_refs: Optional[Iterable[str]] = None,
        approved_source_refs: Optional[Iterable[str]] = None,
        allowed_tag_keys: Optional[Iterable[str]] = None,
        active_semantic_tiers: Optional[Iterable[str]] = None,
    ) -> MemoryAdapterResult:
        del PrecedenceGraph_v
        del ResolutionRules_v

        provider_snapshot = self.provider.retrieve_projection_context(
            org_id=org_id,
            rea_event=rea_event,
            authority_domain=authority_domain,
            projection_bundle_id=projection_bundle_id,
            knowledge_version=K_v,
            tag_ontology_version=O_tag_v,
            constraint_bundle_version=ConstraintBundle_v,
            query_intent=query_intent,
        )

        selected_context = sanitize_projection_context(
            {
                **(provider_snapshot.selected_context or {}),
                **(seed_context or {}),
            }
        )

        active_sources = set(active_source_refs or [])
        approved_sources = set(approved_source_refs or [])
        allowed_keys = set(allowed_tag_keys or _ALLOWED_TAG_KEYS)
        active_tiers = tuple(active_semantic_tiers or ())
        policy = ensure_default_admissibility_policy()

        excluded_context_refs = list(provider_snapshot.excluded_context_refs or [])
        admissible_source_refs: list[dict[str, Any]] = []
        downgraded_source_refs: list[str] = []
        rejected_summary: list[dict[str, Any]] = []
        for source_ref in _iter_source_refs(provider_snapshot):
            source_id = source_ref.get("source_id") or source_ref.get("source_ref")
            source_domain = source_ref.get("authority_domain")
            status = source_ref.get("governance_state") or source_ref.get("status")
            source_tier = source_ref.get("semantic_tier")

            reject_reason = None
            if source_id and active_sources and source_id not in active_sources:
                reject_reason = "REJECT_NOT_IN_ACTIVE_K_V"
            elif source_id and approved_sources and source_id not in approved_sources:
                reject_reason = "REJECT_NOT_APPROVED"
            elif source_domain and source_domain != authority_domain:
                reject_reason = "REJECT_DOMAIN_MISMATCH"
            elif status in {"DEPRECATED", "REJECTED", "SUPERSEDED"}:
                reject_reason = "REJECT_GOVERNANCE_STATE"

            if reject_reason:
                excluded_context_refs.append(
                    {
                        **source_ref,
                        "reason": reject_reason,
                    }
                )
                rejected_summary.append(
                    {
                        "source_id": source_id,
                        "reason": reject_reason,
                    }
                )
                continue

            admission_decision = "ADMIT"
            context_only = False
            if _requires_downgraded_authority(source_tier, active_tiers):
                admission_decision = "ADMIT_WITH_DOWNGRADED_AUTHORITY"
                context_only = True
                if source_id:
                    downgraded_source_refs.append(source_id)

            admissible_source_refs.append(
                {
                    **source_ref,
                    "admission_decision": admission_decision,
                    "context_only": context_only,
                }
            )

        invalid_ontology_keys = sorted(_FORBIDDEN_ONTOLOGY_KEYS.intersection(selected_context.keys()))
        if invalid_ontology_keys:
            raise ValueError(
                "context_violates_O_v:" + ",".join(invalid_ontology_keys)
            )

        invalid_tag_keys = sorted(
            key for key in selected_context.keys()
            if key.startswith("tag:") and key.split(":", 1)[1] not in allowed_keys
        )
        if invalid_tag_keys:
            raise ValueError("context_violates_O_tag_v:" + ",".join(invalid_tag_keys))

        if selected_context.get("authority_domain") not in (None, authority_domain):
            raise ValueError("context_outside_authority_domain")

        if downgraded_source_refs:
            selected_context["context_only"] = True
            selected_context["downgraded_source_refs"] = sorted(downgraded_source_refs)

        retrieval_query_hash = content_hash(
            {
                "org_id": org_id,
                "authority_domain": authority_domain,
                "projection_bundle_id": projection_bundle_id,
                "O_v": O_v,
                "O_tag_v": O_tag_v,
                "K_v": K_v,
                "ConstraintBundle_v": ConstraintBundle_v,
                "query_intent": query_intent,
                "rea_event": rea_event,
                "seed_context": seed_context or {},
                "source_refs": admissible_source_refs,
            }
        )

        selection_reason = provider_snapshot.selection_reason or "adapter mediated retrieval"
        snapshot = create_snapshot(
            projection_context_ref=projection_context_ref,
            org_id=org_id,
            event_id=event_id,
            knowledge_version=K_v,
            tag_ontology_version=O_tag_v,
            projection_bundle_version=projection_bundle_id,
            constraint_bundle_version=ConstraintBundle_v,
            retrieval_query_hash=retrieval_query_hash,
            provider_ref=self.provider_ref,
            provider_version=provider_snapshot.provider_version,
            adapter_version=self.adapter_version,
            selected_context=selected_context,
            retrieved_source_refs=admissible_source_refs,
            excluded_context_refs=excluded_context_refs,
            admissibility_policy_version=policy["policy_version"],
            rejected_items_summary=rejected_summary,
            selection_reason=selection_reason,
            created_by=created_by,
        )
        return MemoryAdapterResult(
            projection_context_ref=projection_context_ref,
            projection_context_hash=snapshot["projection_context_hash"],
            snapshot=snapshot,
        )
