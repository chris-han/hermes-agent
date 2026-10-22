"""Deterministic retrieval over the currently executable accepted pattern set."""

from __future__ import annotations

from storage.hosted_postgres import sql_query

from datetime import datetime
from pathlib import Path
from fractions import Fraction
import re
from typing import Any, Mapping

from contracts.accepted_graph_release_v2 import parse_accepted_graph_release_unverified
from contracts.evidence_location import EvidenceRef
from contracts.ontology_building import ArtifactPinV1
from contracts.observed_constraint_hypothesis import (
    HypothesisSourceBinding,
    RuntimeSemanticRetrievalSnapshotV1,
    create_runtime_semantic_retrieval_snapshot,
)
from eos.artifact_hashing import content_hash
from services.accepted_graph_rule_projection_service import AcceptedGraphRuleProjectionService
from services.effective_restriction_runtime_context_service import EffectiveRestrictionRuntimeContextService
from storage.governed_rule_store import GovernedRuleStore
from storage.human_grounding_store import HumanGroundingStore
from storage.ontology_build_store import OntologyBuildStore, ontology_build_store_path
from storage.runtime_semantic_hypothesis_artifact_store import RuntimeSemanticHypothesisArtifactStore
from semantier.knowledge_feature_packs.evidence_selector_builder import evidence_ref_identity


RETRIEVAL_POLICY_REF = "runtime-semantic-retrieval.v1"
MAX_RETRIEVED_PATTERNS = 8
RETRIEVAL_POLICY_HASH = "sha256:" + content_hash(
    {
        "policy_ref": RETRIEVAL_POLICY_REF,
        "candidate_scope": "active_authoritative_canonical_patterns",
        "ranking": "unicode_bigram_jaccard_then_pattern_ref",
        "maximum_retrieved_patterns": MAX_RETRIEVED_PATTERNS,
        "zero_overlap_fallback": "retain_active_patterns",
        "material_order": [
            "canonical_pattern", "lexical_cue", "positive_exemplar", "hard_negative"
        ],
    }
)


PERCEPTION_RETRIEVAL_POLICY_REF = "runtime-semantic-retrieval.v2"
PERCEPTION_RETRIEVAL_POLICY_HASH = "sha256:" + content_hash(dict(
    policy_ref=PERCEPTION_RETRIEVAL_POLICY_REF, predecessor_policy_hash=RETRIEVAL_POLICY_HASH,
    query="pinned_perception_non_description_clause_occurrences_with_full_region_fallback",
    source_context="retain_complete_source_region_in_decision_projection"))


def _validate_max_patterns(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("max_patterns must be a positive integer")
    if value > MAX_RETRIEVED_PATTERNS:
        raise ValueError(f"max_patterns must be at most {MAX_RETRIEVED_PATTERNS}")
    return value


def _bigrams(text: str) -> set[str]:
    compact = "".join(str(text).casefold().split())
    if len(compact) < 2:
        return {compact} if compact else set()
    return {compact[index : index + 2] for index in range(len(compact) - 1)}


def _similarity(left: str, right: str) -> Fraction:
    a, b = _bigrams(left), _bigrams(right)
    union = len(a | b)
    return Fraction(len(a & b), union) if union else Fraction(0, 1)


def _rank_semantic_texts(query: str, candidates: list[tuple[str, str]]) -> list[str]:
    scored = [(_similarity(query, text), ref) for ref, text in candidates]
    scored.sort(key=lambda item: (-item[0], item[1].encode("utf-8")))
    return [ref for _score, ref in scored]


def _snake_case_json(value: Any) -> Any:
    """Project governed nested contracts into snapshot-safe machine keys."""
    if isinstance(value, Mapping):
        projected: dict[str, Any] = {}
        for key, nested in value.items():
            snake_key = re.sub(r"(?<!^)(?=[A-Z])", "_", str(key)).lower()
            if snake_key in projected:
                raise ValueError("nested contract key normalization collision")
            projected[snake_key] = _snake_case_json(nested)
        return projected
    if isinstance(value, (list, tuple)):
        return [_snake_case_json(item) for item in value]
    return value


def _validate_source_region(
    *, organization_id: str, workspace_id: str, source_region: Mapping[str, Any]
) -> tuple[str, str, HypothesisSourceBinding, tuple[EvidenceRef, ...]]:
    region_ref = source_region.get("source_region_ref")
    text = source_region.get("text")
    if not isinstance(region_ref, str) or not region_ref or not region_ref.isascii():
        raise ValueError("source_region_ref must be an ASCII-stable reference")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("source region text is required")
    binding = HypothesisSourceBinding.model_validate(source_region.get("source_binding"))
    raw_evidence = source_region.get("evidence_refs")
    if not isinstance(raw_evidence, list) or not raw_evidence:
        raise ValueError("source region evidence_refs must not be empty")
    evidence = tuple(EvidenceRef.model_validate(item) for item in raw_evidence)
    identities = tuple(item.evidence_ref for item in evidence)
    if len(identities) != len(set(identities)):
        raise ValueError("source region evidence_refs contains a duplicate identity")
    for item in evidence:
        if item.organization_id != organization_id or item.workspace_id != workspace_id:
            raise ValueError("source region EvidenceRef scope mismatch")
        if item.source_identity_ref != binding.source_identity_ref:
            raise ValueError("source region EvidenceRef SourceIdentity mismatch")
        if item.selector_hash != item.selector.selector_hash():
            raise ValueError("source region EvidenceRef selector hash mismatch")
        if item.selector_schema_version != item.selector.schema_version:
            raise ValueError("source region EvidenceRef selector schema version mismatch")
        try:
            created_at = (
                item.created_at
                if isinstance(item.created_at, datetime)
                else datetime.fromisoformat(item.created_at.replace("Z", "+00:00"))
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("source region EvidenceRef created_at must be UTC") from exc
        if (
            created_at.tzinfo is None
            or created_at.utcoffset() is None
            or created_at.utcoffset().total_seconds() != 0
        ):
            raise ValueError("source region EvidenceRef created_at must be UTC")
        if item.evidence_ref != evidence_ref_identity(
            organization_id=item.organization_id,
            workspace_id=item.workspace_id,
            source_identity_ref=item.source_identity_ref,
            selector_hash=item.selector_hash,
            evidence_role=item.evidence_role,
        ):
            raise ValueError("source region EvidenceRef identity mismatch")
        if (
            item.provenance.get("source_hash") != binding.source_hash
            or item.provenance.get("canonical_source_ir_ref") != binding.canonical_source_ir_ref
            or item.provenance.get("canonical_source_ir_hash") != binding.canonical_source_ir_hash
        ):
            raise ValueError("source region EvidenceRef provenance mismatch")
    return region_ref, text, binding, evidence


class RuntimeSemanticRetrievalService:
    def __init__(
        self, *, runtime_root: str | Path, db_path: str | Path,
        workspace_root: str | Path | None = None,
    ) -> None:
        self._runtime_root = Path(runtime_root)
        self._db_path = Path(db_path)
        self._workspace_root = Path(workspace_root) if workspace_root is not None else None

    def retrieve_and_persist(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        run_id: str,
        source_region: Mapping[str, Any],
        max_patterns: int = MAX_RETRIEVED_PATTERNS,
        runtime_context_pin: ArtifactPinV1 | None = None,
        perception_pin: ArtifactPinV1 | None = None, typed_source_region=None,
    ) -> RuntimeSemanticRetrievalSnapshotV1:
        max_patterns = _validate_max_patterns(max_patterns)
        region_ref, text, source_binding, evidence_refs = _validate_source_region(
            organization_id=organization_id,
            workspace_id=workspace_id,
            source_region=source_region,
        )

        perception_query = None
        if perception_pin is not None:
            if self._workspace_root is None or typed_source_region is None:
                raise ValueError("PERCEPTION_SOURCE_OWNER_REQUIRED")
            from contracts.runtime_semantic_source_region import source_region_to_legacy_dict
            from services.runtime_restriction_perception_service import RuntimeRestrictionPerceptionService
            if source_region_to_legacy_dict(typed_source_region) != dict(source_region):
                raise ValueError("PERCEPTION_RETRIEVAL_SOURCE_MISMATCH")
            perception_store = OntologyBuildStore(ontology_build_store_path(self._workspace_root))
            try:
                _, perception = RuntimeRestrictionPerceptionService.read_persisted(store=perception_store,
                    organization_id=organization_id, workspace_id=workspace_id, response_pin=perception_pin,
                    source_region=typed_source_region, run_id=run_id)
            finally:
                perception_store.close()
            perception_query = RuntimeRestrictionPerceptionService.retrieval_query(perception, typed_source_region)

        context = EffectiveRestrictionRuntimeContextService(self._db_path).resolve_current(
            organization_id=organization_id, workspace_id=workspace_id
        )
        if runtime_context_pin is not None:
            store = OntologyBuildStore(ontology_build_store_path(
                self._workspace_root if self._workspace_root is not None
                else self._runtime_root / "workspaces" / workspace_id,
            ))
            try:
                pinned = EffectiveRestrictionRuntimeContextService.resolve_persisted(
                    store=store, pin=runtime_context_pin,
                    organization_id=organization_id, workspace_id=workspace_id,
                )
            finally:
                store.close()
            # New inspection must still pass current authority checks, but may
            # never silently substitute a different release or activation set.
            excluded = {"effective_at", "effective_context_resolver_hash"}
            if pinned.model_dump(exclude=excluded) != context.model_dump(exclude=excluded):
                raise ValueError("PINNED_RUNTIME_AUTHORITY_CHANGED")
            context = pinned
        rule_store = GovernedRuleStore(self._db_path)
        active_pairs: set[tuple[str, str]] = set()
        for rule_ref in context.ordered_governed_rule_version_refs:
            rule = rule_store.get_rule_version(rule_ref)
            if rule is None:
                raise ValueError("effective governed rule version is missing")
            scope = rule.get("applicability_scope") or {}
            active_pairs.add(
                (str(scope.get("source_graph_pattern_id") or ""), str(scope.get("source_graph_pattern_hash") or ""))
            )

        grounding = HumanGroundingStore(self._db_path, graph_backend=None)
        release = parse_accepted_graph_release_unverified(
            grounding.get_graph_release(
                context.accepted_graph_release_ref,
                organization_id=organization_id,
                workspace_id=workspace_id,
            )
        )
        patterns = AcceptedGraphRuleProjectionService._load_v2_patterns(
            grounding, release, organization_id=organization_id, workspace_id=workspace_id
        )
        active = [item for item in patterns if (item[0].pattern_id, item[0].pattern_hash) in active_pairs]
        if not active:
            raise ValueError("effective context has no authoritative retrievable patterns")
        semantic_texts = []
        patterns_by_ref = {}
        for pattern, _binding in active:
            semantic_text = " ".join(
                evidence.selector.exact_quote or ""
                for evidence in pattern.source_metadata.evidence_refs
            )
            semantic_texts.append((pattern.pattern_id, semantic_text))
            patterns_by_ref[pattern.pattern_id] = pattern
        ranked_refs = _rank_semantic_texts(perception_query if perception_query is not None else text, semantic_texts)

        material: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()

        def append(kind: str, ref: str, digest: str) -> None:
            key = (kind, ref)
            if key in seen:
                return
            seen.add(key)
            material.append({"kind": kind, "ref": ref, "hash": digest, "rank": len(material) + 1})

        for pattern_ref in ranked_refs[:max_patterns]:
            pattern = patterns_by_ref[pattern_ref]
            append("canonical_pattern", pattern.pattern_id, pattern.pattern_hash)
            for ref in pattern.lexical_evidence_refs:
                append(
                    "lexical_cue",
                    ref,
                    "sha256:" + content_hash(
                        {
                            "schema_version": "canonical_pattern_lexical_cue_binding.v1",
                            "canonical_pattern_ref": pattern.pattern_id,
                            "canonical_pattern_hash": pattern.pattern_hash,
                            "lexical_evidence_ref": ref,
                        }
                    ),
                )
            for kind, refs in (
                ("positive_exemplar", pattern.witness_metadata.positive_exemplar_refs),
                ("hard_negative", pattern.witness_metadata.hard_negative_refs),
            ):
                for ref in refs:
                    row = grounding._conn.execute(
                        sql_query("SELECT witness_hash FROM source_semantic_witness_reviews WHERE witness_ref=?", 'SELECT witness_hash FROM source_semantic_witness_reviews WHERE witness_ref=%s'),
                        (ref,),
                    ).fetchone()
                    if row is None:
                        from services.graph_schema_discovery_service import (
                            resolve_trusted_accepted_witness_by_ref,
                        )

                        witness = resolve_trusted_accepted_witness_by_ref(
                            ref, expected_kind=kind
                        )
                    else:
                        witness = grounding.replay_source_semantic_witness(
                            organization_id=organization_id,
                            workspace_id=workspace_id,
                            witness_ref=ref,
                            witness_hash=str(row["witness_hash"]),
                        )
                    expected_kind = "positive_exemplar" if kind == "positive_exemplar" else "hard_negative"
                    if witness.witness_kind != expected_kind or witness.review.state not in {"accepted", "edited"}:
                        raise ValueError("canonical pattern witness is not admitted retrieval evidence")
                    append(kind, ref, witness.witness_hash)

        snapshot = create_runtime_semantic_retrieval_snapshot(
            organization_id=organization_id,
            workspace_id=workspace_id,
            run_id=run_id,
            query={
                "source_region_ref": region_ref,
                "text_hash": "sha256:" + content_hash(text),
                "source_binding": source_binding.model_dump(mode="json"),
                "evidence_refs": [
                    _snake_case_json(item.model_dump(mode="json")) for item in evidence_refs
                ],
                "max_patterns": max_patterns,
                **({"perception_pin": perception_pin.model_dump(mode="json"), "perception_query": perception_query}
                   if perception_pin is not None else {}),
            },
            ordered_material=material,
            retrieval_policy_ref=PERCEPTION_RETRIEVAL_POLICY_REF if perception_pin is not None else RETRIEVAL_POLICY_REF,
            retrieval_policy_hash=PERCEPTION_RETRIEVAL_POLICY_HASH if perception_pin is not None else RETRIEVAL_POLICY_HASH,
            runtime_context=context,
        )
        return RuntimeSemanticHypothesisArtifactStore(
            self._runtime_root,
            organization_id=organization_id,
            workspace_id=workspace_id,
        ).put_retrieval_snapshot(snapshot)


__all__ = [
    "MAX_RETRIEVED_PATTERNS",
    "RETRIEVAL_POLICY_HASH",
    "RETRIEVAL_POLICY_REF",
    "RuntimeSemanticRetrievalService",
]
