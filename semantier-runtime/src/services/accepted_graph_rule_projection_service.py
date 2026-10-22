"""Project admitted Human-Grounded graph rules into executable candidates."""

from __future__ import annotations

import json
import os
import sqlite3
from eos.db import get_conn
from storage.hosted_postgres import enabled,sql_query
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from contracts.accepted_graph_release_v2 import (
    AcceptedGraphReleaseV2,
    parse_accepted_graph_release_unverified,
)
from contracts.canonical_restriction_pattern import (
    CanonicalRestrictionPatternV1,
    canonical_pattern_bytes_unverified,
    parse_canonical_restriction_pattern_unverified,
)
from contracts.knowledge_graph_phase1 import AcceptedGraphRelease
from eos.artifact_hashing import content_hash
from plugins.tender_sensitive_term_detection_modeling.compiler import DEFAULT_ISSUE_TYPE
from runtime_paths import sqlite_db_path
from storage.governed_rule_store import GovernedRuleStore
from storage.accepted_graph_release_origin import (
    verify_exact_release_origin_reservation,
)
from storage.human_grounding_store import HumanGroundingStore
from storage.knowledge_source_activation_store import KnowledgeSourceActivationStore
from storage.policy_rule_candidate_store import PolicyRuleCandidateStore


def _default_db_path() -> Path:
    return Path(os.environ.get("SEMANTIER_EOS_DB_PATH", str(sqlite_db_path("eos.db", ".semantier/eos.db"))))


@dataclass(frozen=True)
class ProjectedRuleCandidate:
    candidate: dict[str, Any]
    source_graph_rule_id: str
    idempotency_key: str


class AcceptedGraphRuleProjectionService:
    """Make only binding-admitted graph rules executable candidates.

    The service deliberately loads the grounding binding itself. Callers may
    provide a release, but cannot inject a binding from another release.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = Path(db_path) if db_path else _default_db_path()

    def project(
        self,
        *,
        release: AcceptedGraphRelease | AcceptedGraphReleaseV2 | dict[str, Any],
        organization_id: str,
        workspace_id: str | None,
        actor: str,
        actor_authority: str,
    ) -> list[ProjectedRuleCandidate]:
        release_model = (
            release
            if isinstance(release, (AcceptedGraphRelease, AcceptedGraphReleaseV2))
            else parse_accepted_graph_release_unverified(release)
        )
        if isinstance(release_model, AcceptedGraphReleaseV2):
            return self._project_v2(
                release_model,
                organization_id=organization_id,
                workspace_id=workspace_id,
                actor=actor,
                actor_authority=actor_authority,
            )
        release_hash = release_model.release_hash()
        binding = self._load_binding(release_model, release_hash)
        accepted_rule_ids = set(json.loads(binding["accepted_rule_ids_json"] or "[]"))
        nodes = {str(node["node_id"]): node for node in release_model.nodes}
        projected: list[ProjectedRuleCandidate] = []
        candidate_store = PolicyRuleCandidateStore(db_path=self._db_path)
        for rule in release_model.rules:
            rule_id = str(rule["rule_id"])
            if rule_id not in accepted_rule_ids:
                continue
            anchors = list(rule.get("source_anchors") or [])
            match_terms = sorted({str(anchor.get("exact_text") or "") for anchor in anchors if anchor.get("exact_text")})
            if not match_terms:
                raise ValueError(f"accepted graph rule has no anchor text: {rule_id}")
            conditions = self._resolve_nodes(rule.get("condition_node_ids") or [], nodes, rule_id)
            exceptions = self._resolve_nodes(rule.get("exception_node_ids") or [], nodes, rule_id)
            key = content_hash({
                "release_hash": release_hash,
                "source_graph_rule_id": rule_id,
                "organization_id": organization_id,
                "workspace_id": workspace_id,
            })
            scope = {
                "match_terms": match_terms,
                "issue_type": DEFAULT_ISSUE_TYPE,
                "rule_family": "tender_compliance",
                "source_graph_version": release_model.graph_version,
                "source_graph_release_hash": release_hash,
                "source_graph_rule_id": rule_id,
                "source_anchor_refs": [str(anchor["anchor_id"]) for anchor in anchors],
                "conditions": conditions,
                "exceptions": exceptions,
            }
            if rule.get("frame_pattern") is not None:
                scope["frame_pattern"] = {
                    **rule["frame_pattern"],
                    "graph_release_ref": release_model.graph_version,
                }
            approval_evidence = {
                "grounding_event_id": binding["human_label_event_id"],
                "grounding_event_hash": self._grounding_event_hash(binding["human_label_event_id"]),
                "grounding_binding_id": str(binding["binding_id"]),
                "graph_version": release_model.graph_version,
                "graph_release_hash": release_hash,
                "accepted_rule_id": rule_id,
                "grounding_actor_ref": self._grounding_event_actor_ref(binding["human_label_event_id"]),
            }
            candidate = candidate_store.get_candidate_by_idempotency_key(key)
            if candidate is None:
                candidate = candidate_store.create_candidate(
                    organization_id=organization_id,
                    proposed_workspace_scope="workspace" if workspace_id else "organization",
                    proposed_workspace_id=workspace_id,
                    authority_domain="tender_sensitive_detection",
                    semantic_tier="T3",
                    jurisdiction="CN",
                    candidate_idempotency_key=key,
                    rule_family="tender_compliance",
                    candidate_state="PROPOSED",
                    source_anchor_refs=scope["source_anchor_refs"],
                    applicability_scope=scope,
                    extracted_rationale=str(rule.get("reason_code") or "Human-grounded tender sensitive-term rule"),
                    draft_rule_text="; ".join(match_terms),
                    severity="medium",
                    confidence=1.0,
                    uncertainty_notes=None,
                    approval_evidence=approval_evidence,
                    test_evidence={"source_graph_release_hash": release_hash},
                    created_by=actor,
                    created_by_actor_type="plugin",
                )
            if candidate["candidate_state"] == "PROPOSED":
                candidate = candidate_store.request_review(candidate["rule_candidate_id"], actor=actor)
            if candidate["candidate_state"] == "NEEDS_REVIEW":
                candidate = candidate_store.approve(
                    candidate["rule_candidate_id"],
                    actor=actor,
                    actor_authority=actor_authority,
                    approval_evidence=approval_evidence,
                    test_evidence={"source_graph_release_hash": release_hash},
                )
            projected.append(ProjectedRuleCandidate(candidate, rule_id, key))
        return projected

    def _project_v2(
        self,
        release: AcceptedGraphReleaseV2,
        *,
        organization_id: str,
        workspace_id: str | None,
        actor: str,
        actor_authority: str,
    ) -> list[ProjectedRuleCandidate]:
        if workspace_id is None:
            raise ValueError("v2 graph-rule projection requires workspace scope")
        store = HumanGroundingStore(self._db_path, graph_backend=None)
        stored = store.get_graph_release(
            release.graph_version,
            organization_id=organization_id,
            workspace_id=workspace_id,
        )
        if stored != release.as_dict():
            raise ValueError("v2 projection release differs from authoritative store")
        verify_exact_release_origin_reservation(
            store._conn,
            graph_version=release.graph_version,
            authority_state="authoritative",
        )
        patterns = self._load_v2_patterns(
            store, release,
            organization_id=organization_id,
            workspace_id=workspace_id,
        )
        patterns_by_ref = {pattern.pattern_id: pattern for pattern, _ in patterns}
        component_refs = {
            ref
            for pattern, _binding in patterns
            for ref in pattern.composition.member_pattern_refs
        }
        candidate_store = PolicyRuleCandidateStore(db_path=self._db_path)
        projected: list[ProjectedRuleCandidate] = []
        for pattern, binding in patterns:
            if pattern.pattern_id in component_refs:
                continue
            match_terms = sorted({
                str(evidence.selector.exact_quote)
                for evidence in pattern.source_metadata.evidence_refs
                if evidence.selector.exact_quote
            })
            if not match_terms:
                raise ValueError(
                    f"accepted canonical pattern has no exact source text: {pattern.pattern_id}"
                )
            source_anchor_refs = [
                evidence.evidence_ref
                for evidence in pattern.source_metadata.evidence_refs
            ]
            key = content_hash({
                "release_hash": release.release_hash,
                "source_graph_pattern_id": pattern.pattern_id,
                "source_graph_pattern_hash": pattern.pattern_hash,
                "accepted_pattern_binding_hash": binding["binding_hash"],
                "organization_id": organization_id,
                "workspace_id": workspace_id,
            })
            scope = {
                "match_terms": match_terms if pattern.pattern_kind == "atomic" else [],
                "issue_type": DEFAULT_ISSUE_TYPE,
                "rule_family": "tender_compliance",
                "source_graph_version": release.graph_version,
                "source_graph_release_hash": release.release_hash,
                "source_graph_pattern_id": pattern.pattern_id,
                "source_graph_pattern_hash": pattern.pattern_hash,
                # Compatibility alias required by governed_rule_version.v1.
                "source_graph_rule_id": pattern.pattern_id,
                "source_anchor_refs": source_anchor_refs,
                "canonical_restriction_pattern": pattern.model_dump(
                    mode="json", by_alias=True
                ),
            }
            if pattern.pattern_kind == "atomic":
                scope["frame_pattern"] = self._frame_pattern_projection(
                    pattern, release.graph_version
                )
            else:
                scope["composed_frame_pattern"] = (
                    self._composed_frame_pattern_projection(
                        pattern, patterns_by_ref, release.graph_version
                    )
                )
            event_ids = tuple(release.human_label_event_ids)
            events = [store.get_learning_event(event_id) for event_id in event_ids]
            if any(event is None for event in events):
                raise ValueError("v2 projection grounding event is missing")
            approval_evidence = {
                "grounding_event_id": event_ids[0],
                "grounding_event_hash": events[0]["event_hash"],
                "grounding_event_ids": list(event_ids),
                "grounding_event_hashes": [event["event_hash"] for event in events],
                "grounding_binding_id": binding["binding_id"],
                "grounding_binding_hash": binding["binding_hash"],
                "graph_version": release.graph_version,
                "graph_release_hash": release.release_hash,
                "accepted_pattern_id": pattern.pattern_id,
                "accepted_pattern_hash": pattern.pattern_hash,
                "grounding_actor_ref": events[0]["actor_ref"],
            }
            candidate = candidate_store.get_candidate_by_idempotency_key(key)
            if candidate is None:
                candidate = candidate_store.create_candidate(
                    organization_id=organization_id,
                    proposed_workspace_scope="workspace",
                    proposed_workspace_id=workspace_id,
                    authority_domain="tender_sensitive_detection",
                    semantic_tier="T3",
                    jurisdiction="CN",
                    candidate_idempotency_key=key,
                    rule_family="tender_compliance",
                    candidate_state="PROPOSED",
                    source_anchor_refs=source_anchor_refs,
                    applicability_scope=scope,
                    extracted_rationale=(
                        "Accepted canonical restriction pattern projection"
                    ),
                    draft_rule_text="; ".join(match_terms),
                    severity="medium",
                    confidence=1.0,
                    uncertainty_notes=None,
                    approval_evidence=approval_evidence,
                    test_evidence={
                        "source_graph_release_hash": release.release_hash,
                        "source_graph_pattern_hash": pattern.pattern_hash,
                    },
                    created_by=actor,
                    created_by_actor_type="plugin",
                )
            if candidate["candidate_state"] == "PROPOSED":
                candidate = candidate_store.request_review(
                    candidate["rule_candidate_id"], actor=actor
                )
            if candidate["candidate_state"] == "NEEDS_REVIEW":
                candidate = candidate_store.approve(
                    candidate["rule_candidate_id"], actor=actor,
                    actor_authority=actor_authority,
                    approval_evidence=approval_evidence,
                    test_evidence={
                        "source_graph_release_hash": release.release_hash,
                        "source_graph_pattern_hash": pattern.pattern_hash,
                    },
                )
            projected.append(ProjectedRuleCandidate(candidate, pattern.pattern_id, key))
        return projected

    @staticmethod
    def _frame_pattern_projection(
        pattern: CanonicalRestrictionPatternV1, graph_version: str
    ) -> dict[str, Any]:
        if pattern.pattern_kind != "atomic" or len(pattern.frames) != 1:
            raise ValueError("frame projection requires one atomic canonical pattern")
        frame = dict(pattern.frames[0])
        frame.pop("frame_hash", None)
        return {
            "schema_version": "frame_pattern.v0_1",
            "pattern_id": pattern.pattern_id,
            "frame": frame,
            "graph_release_ref": graph_version,
            "source_evidence_refs": list(pattern.lexical_evidence_refs),
            "lifecycle_state": "accepted",
            "pattern_hash": pattern.pattern_hash,
            "corpus_schema_candidate_ref": (
                pattern.lineage_metadata.schema_candidate_ref
            ),
        }

    @classmethod
    def _composed_frame_pattern_projection(
        cls,
        pattern: CanonicalRestrictionPatternV1,
        patterns_by_ref: dict[str, CanonicalRestrictionPatternV1],
        graph_version: str,
    ) -> dict[str, Any]:
        if pattern.composition.operator != "all_of":
            raise ValueError(
                "runtime projection does not support canonical composition operator: "
                f"{pattern.composition.operator}"
            )
        constituents = []
        for component_ref in pattern.composition.member_pattern_refs:
            component = patterns_by_ref.get(component_ref)
            if component is None or component.pattern_kind != "atomic":
                raise ValueError(
                    "composed projection requires exact atomic components"
                )
            constituents.append(cls._frame_pattern_projection(component, graph_version))
        return {
            "schema_version": "composed_frame_pattern.v0_1",
            "pattern_id": pattern.pattern_id,
            "composition_operator": "ALL_OF",
            "constituents": constituents,
            "graph_release_ref": graph_version,
            "source_evidence_refs": list(pattern.lexical_evidence_refs),
            "lifecycle_state": "accepted",
            "pattern_hash": pattern.pattern_hash,
        }

    @staticmethod
    def _load_v2_patterns(
        store: HumanGroundingStore,
        release: AcceptedGraphReleaseV2,
        *,
        organization_id: str,
        workspace_id: str,
    ) -> list[tuple[CanonicalRestrictionPatternV1, dict[str, Any]]]:
        rows = store._conn.execute(
            sql_query("SELECT binding_id FROM accepted_pattern_bindings "
            "WHERE accepted_graph_release_ref=? ORDER BY binding_id",'SELECT binding_id FROM accepted_pattern_bindings WHERE accepted_graph_release_ref=%s ORDER BY binding_id'),
            (release.graph_version,),
        ).fetchall()
        bindings = [
            store._get_unverified_v2_artifact(
                "accepted_pattern_binding", str(row["binding_id"]),
                organization_id=organization_id, workspace_id=workspace_id,
            )
            for row in rows
        ]
        by_pattern = {
            (item["pattern_ref"], item["pattern_hash"]): item for item in bindings
        }
        expected = {
            (item.pattern_ref, item.pattern_hash) for item in release.pattern_membership
        }
        if set(by_pattern) != expected or len(bindings) != len(expected):
            raise ValueError("v2 release pattern binding completeness mismatch")
        payloads: dict[str, dict[str, Any]] = {}
        pattern_rows: dict[str, Any] = {}
        for member in release.pattern_membership:
            row = store._conn.execute(
                sql_query("SELECT * FROM canonical_restriction_patterns WHERE pattern_id=?",'SELECT * FROM canonical_restriction_patterns WHERE pattern_id=%s'),
                (member.pattern_ref,),
            ).fetchone()
            if row is None:
                raise ValueError(
                    f"unverified v2 artifact not found: {member.pattern_ref}"
                )
            if (
                row["organization_id"] != organization_id
                or row["workspace_id"] != workspace_id
            ):
                raise ValueError("unverified v2 artifact tenant scope mismatch")
            try:
                payloads[member.pattern_ref] = json.loads(
                    row["canonical_payload_json"]
                )
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "stored unverified v2 artifact payload is invalid"
                ) from exc
            pattern_rows[member.pattern_ref] = row

        parsed: dict[str, CanonicalRestrictionPatternV1] = {}
        visiting: set[str] = set()

        def resolve(pattern_ref: str) -> CanonicalRestrictionPatternV1:
            if pattern_ref in parsed:
                return parsed[pattern_ref]
            if pattern_ref in visiting:
                raise ValueError("canonical pattern composition cycle")
            payload = payloads.get(pattern_ref)
            if payload is None:
                raise ValueError(
                    "canonical pattern component is absent from release membership"
                )
            visiting.add(pattern_ref)
            component_refs: set[str] = set()
            pending = list(
                payload.get("composition", {}).get("member_pattern_refs", [])
            )
            while pending:
                component_ref = pending.pop()
                if component_ref in component_refs:
                    continue
                component_refs.add(component_ref)
                component = resolve(component_ref)
                pending.extend(component.composition.member_pattern_refs)
            component_models = [parsed[ref] for ref in sorted(component_refs)]
            pattern = parse_canonical_restriction_pattern_unverified(
                payload,
                component_artifacts=tuple(
                    (canonical_pattern_bytes_unverified(item), item.pattern_hash)
                    for item in component_models
                ),
            )
            visiting.remove(pattern_ref)
            row = pattern_rows[pattern_ref]
            if (
                row["pattern_hash"] != pattern.pattern_hash
                or row["organization_id"] != pattern.organization_id
                or row["workspace_id"] != pattern.workspace_id
                or pattern.model_dump(mode="json", by_alias=True) != payload
            ):
                raise ValueError("stored canonical pattern metadata mismatch")
            parsed[pattern_ref] = pattern
            return pattern

        resolved = []
        for member in release.pattern_membership:
            binding = by_pattern[(member.pattern_ref, member.pattern_hash)]
            _normalized, metadata = store._normalize_unverified_v2_artifact(
                "accepted_pattern_binding", binding
            )
            store._validate_unverified_binding_targets_uncommitted(
                "accepted_pattern_binding", metadata
            )
            pattern = resolve(member.pattern_ref)
            if pattern.pattern_hash != member.pattern_hash:
                raise ValueError("v2 projection canonical pattern hash mismatch")
            resolved.append((pattern, binding))
        return resolved

    def activate(
        self,
        *,
        projected: list[ProjectedRuleCandidate],
        organization_id: str,
        workspace_id: str | None,
        actor: str,
        actor_authority: str,
        principal_entitlement_ref: str,
        effective_at: str,
        activation_resolver_policy_version: str = "knowledge_activation_resolver.v1",
    ) -> dict[str, Any]:
        """Turn approved projections into one deterministic activation snapshot."""
        if not projected:
            raise ValueError("cannot activate an empty graph projection")
        rule_store = GovernedRuleStore(db_path=self._db_path)
        activation_store = KnowledgeSourceActivationStore(db_path=self._db_path)
        versions: list[dict[str, Any]] = []
        activation_ids: list[str] = []
        for item in projected:
            version_id = f"grv_{item.idempotency_key.removeprefix('sha256:')[:24]}"
            version = rule_store.get_rule_version(version_id)
            if version is None:
                version = rule_store.create_approved_version(
                    rule_version_id=version_id,
                    rule_candidate_id=item.candidate["rule_candidate_id"],
                    actor=actor,
                    actor_authority=actor_authority,
                )
            activation_id = f"gra_{item.idempotency_key.removeprefix('sha256:')[:24]}"
            activation = rule_store.activate_version(
                version_id,
                activation_ref=activation_id,
                actor=actor,
                actor_authority=actor_authority,
                organization_id=organization_id,
                activation_scope="workspace" if workspace_id else "organization",
                workspace_id=workspace_id,
                effective_from=effective_at,
                approval_event_ref=str(item.candidate["approval_evidence"].get("grounding_event_id")),
            )
            versions.append(version)
            activation_ids.append(activation["activation"]["activation_id"])
        snapshot_basis = {
            "organization_id": organization_id,
            "workspace_id": workspace_id,
            "activation_ids": sorted(activation_ids),
            "effective_at": effective_at,
        }
        snapshot_id = f"kass_{content_hash(snapshot_basis).removeprefix('sha256:')[:24]}"
        snapshot = activation_store.materialize_activation_set_snapshot(
            activation_set_snapshot_id=snapshot_id,
            organization_id=organization_id,
            workspace_id=workspace_id,
            principal_entitlement_ref=principal_entitlement_ref,
            activation_resolver_policy_version=activation_resolver_policy_version,
            effective_at=effective_at,
            included_activation_ids=activation_ids,
            excluded_activation_evidence=[],
        )
        return {"projected": projected, "rule_versions": versions, "activation_ids": activation_ids, "snapshot": snapshot}

    def _grounding_event_hash(self, event_id: str) -> str:
        conn = get_conn(self._db_path)
        try:
            row = conn.execute(
                sql_query("SELECT event_hash FROM knowledge_learning_events WHERE event_id=?",'SELECT event_hash FROM knowledge_learning_events WHERE event_id=%s'),
                (event_id,),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            raise ValueError(f"grounding event not found: {event_id}")
        return str(row[0])

    def _grounding_event_actor_ref(self, event_id: str) -> str:
        conn = get_conn(self._db_path)
        try:
            row = conn.execute(
                sql_query("SELECT event_json FROM knowledge_learning_events WHERE event_id=?",'SELECT event_json FROM knowledge_learning_events WHERE event_id=%s'),
                (event_id,),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            raise ValueError(f"grounding event not found: {event_id}")
        actor_ref = json.loads(row[0]).get("actor_ref")
        if not actor_ref:
            raise ValueError(f"grounding event has no actor_ref: {event_id}")
        return str(actor_ref)

    def _load_binding(self, release: AcceptedGraphRelease, release_hash: str) -> sqlite3.Row:
        if not release.human_label_event_ids:
            raise ValueError("accepted graph release has no grounding event references")
        conn = get_conn(self._db_path)
        if not enabled():conn.row_factory = sqlite3.Row
        try:
            placeholders = ",".join(sql_query("?","%s") for _ in release.human_label_event_ids)
            row = conn.execute(
                sql_query(f"""SELECT * FROM knowledge_builder_assertion_grounding_bindings
                    WHERE graph_version=? AND graph_release_hash=?
                    AND human_label_event_id IN ({placeholders})
                    ORDER BY materialized_at DESC LIMIT 1""",
                    f"""SELECT * FROM knowledge_builder_assertion_grounding_bindings
                    WHERE graph_version=%s AND graph_release_hash=%s
                    AND human_label_event_id IN ({placeholders})
                    ORDER BY materialized_at DESC LIMIT 1"""),
                (release.graph_version, release_hash, *release.human_label_event_ids),
            ).fetchone()
            if row is None:
                row = conn.execute(
                    sql_query(f"""SELECT ('reference:' || binding_id) AS binding_id,
                        concept_id AS assertion_id, human_label_event_id, graph_version,
                        graph_release_hash, accepted_node_ids_json, accepted_edge_ids_json,
                        accepted_rule_ids_json, materialized_at
                        FROM contextgraph_reference_concept_grounding_bindings
                        WHERE graph_version=? AND graph_release_hash=?
                        AND human_label_event_id IN ({placeholders})
                        ORDER BY materialized_at DESC LIMIT 1""",
                        f"""SELECT ('reference:' || binding_id) AS binding_id,
                        concept_id AS assertion_id, human_label_event_id, graph_version,
                        graph_release_hash, accepted_node_ids_json, accepted_edge_ids_json,
                        accepted_rule_ids_json, materialized_at
                        FROM contextgraph_reference_concept_grounding_bindings
                        WHERE graph_version=%s AND graph_release_hash=%s
                        AND human_label_event_id IN ({placeholders})
                        ORDER BY materialized_at DESC LIMIT 1"""),
                    (release.graph_version, release_hash, *release.human_label_event_ids),
                ).fetchone()
            if row is None:
                raise ValueError("accepted graph release has no matching grounding binding")
            return row
        finally:
            conn.close()

    @staticmethod
    def _resolve_nodes(node_ids: list[str], nodes: dict[str, dict[str, Any]], rule_id: str) -> list[dict[str, str]]:
        resolved = []
        for node_id in node_ids:
            node = nodes.get(str(node_id))
            if node is None:
                raise ValueError(f"graph rule {rule_id} references unresolved node: {node_id}")
            resolved.append({"node_id": str(node_id), "text": str(node.get("canonical_label") or node.get("normalized_value") or "")})
        return resolved
