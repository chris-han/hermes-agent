from __future__ import annotations

from storage import hosted_postgres as hosted

from dataclasses import asdict
from typing import Any

from agents.auth_session import RequestContext
from contracts.governed_graph import (
    GraphEdgeV1,
    GraphIndexStatusV1,
    GraphNodeV1,
    GraphPathSummaryV1,
    GraphPositionV1,
    GraphSceneV1,
    GraphWarningV1,
)
from eos.artifact_hashing import content_hash
from eos.db import now_iso
from plugins.knowledge_sensitive_labeling.dashboard.graph_projection import (
    build_authorized_graph_projection,
)
from semantier.graph_kernel import GraphIngestionCoordinator, GraphKernelStore


DEFAULT_PREDICATES = [
    "CONTAINS",
    "DERIVED_FROM",
    "GOVERNED_BY",
    "CONSTRAINS",
    "CONFLICTS_WITH",
    "DEPENDS_ON",
    "PRODUCES",
    "CONSUMES",
    "ACTIVATED_AS",
    "INCLUDED_IN",
    "JUSTIFIED_BY",
    "VALIDATED_AGAINST",
    "USED_CONTEXT",
    "BOUND_TO",
]
READ_STATES = ["candidate", "validated", "approved", "active", "deprecated", "rejected"]


class GovernedGraphService:
    def __init__(
        self,
        *,
        store: GraphKernelStore | None = None,
        ingestion: GraphIngestionCoordinator | None = None,
    ) -> None:
        self.store = store or GraphKernelStore()
        self.ingestion = ingestion or GraphIngestionCoordinator(self.store)

    def resolve_projection(
        self,
        *,
        ctx: RequestContext,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        organization_id = _organization_id(ctx)
        source_ref = _optional_text(request, "source_ref", "sourceRef") or "graph-kernel"
        focus_ref = _optional_text(request, "focus_object_ref", "node_id", "assertion_id", "focusObjectRef")
        snapshot_id = _optional_text(request, "graph_snapshot_ref", "graphSnapshotRef")
        as_of = _optional_text(request, "as_of", "asOf")
        lens = _lens(_optional_text(request, "lens") or "overview")
        assertions = self._select_assertions(
            organization_id=organization_id,
            source_ref=source_ref,
            focus_ref=focus_ref,
            snapshot_id=snapshot_id,
            as_of=as_of,
            limit=_int(request.get("limit"), default=60, maximum=150),
        )
        projection = self._projection_from_assertions(
            ctx=ctx,
            source_ref=source_ref,
            assertions=assertions,
            lens=lens,
            as_of=as_of,
            focus_ref=focus_ref,
        )
        self.store.save_projection_context(
            projection_id=projection["projection_id"],
            organization_id=organization_id,
            workspace_id=ctx.workspace_id,
            source_ref=source_ref,
            focus_ref=focus_ref,
            graph_snapshot_ref=snapshot_id,
            as_of=as_of,
            lens=lens,
        )
        return projection

    def get_projection(
        self,
        *,
        ctx: RequestContext,
        projection_id: str,
    ) -> dict[str, Any]:
        organization_id = _organization_id(ctx)
        context = self.store.get_projection_context(
            organization_id=organization_id,
            workspace_id=ctx.workspace_id,
            projection_id=projection_id,
        )
        if context is None:
            raise KeyError(projection_id)
        return self.resolve_projection(
            ctx=ctx,
            request={
                "source_ref": context["source_ref"],
                "focus_object_ref": context.get("focus_ref"),
                "graph_snapshot_ref": context.get("graph_snapshot_ref"),
                "as_of": context.get("as_of"),
                "lens": context["lens"],
            },
        )

    def get_node(self, *, ctx: RequestContext, node_id: str) -> dict[str, Any]:
        organization_id = _organization_id(ctx)
        node = self._node_by_id(organization_id, node_id)
        if node is None:
            raise KeyError(node_id)
        assertions = self._select_assertions(
            organization_id=organization_id,
            source_ref=None,
            focus_ref=node_id,
            snapshot_id=None,
            as_of=None,
            limit=80,
        )
        return {
            "schema_version": "governed_graph_contract.v1.node_detail",
            "node": asdict(self._node_view(node, index=0)),
            "related_assertions": [self._edge_payload(row) for row in assertions],
        }

    def get_assertion(self, *, ctx: RequestContext, assertion_id: str) -> dict[str, Any]:
        organization_id = _organization_id(ctx)
        row = self._assertion_by_id(organization_id, assertion_id)
        if row is None:
            raise KeyError(assertion_id)
        return {
            "schema_version": "governed_graph_contract.v1.assertion_detail",
            "assertion": self._edge_payload(row),
            "path_summary": _path_summary(row),
        }

    def get_anchor(self, *, ctx: RequestContext, anchor_id: str) -> dict[str, Any]:
        organization_id = _organization_id(ctx)
        rows = self._assertions_by_anchor(organization_id, anchor_id)
        if not rows:
            raise KeyError(anchor_id)
        return {
            "schema_version": "evidence_detail_v1",
            "anchor_id": anchor_id,
            "source_ref": rows[0]["source_ref"],
            "locator": rows[0]["source_anchor"],
            "excerpt": None,
            "disclosure": "excerpt_not_embedded_in_graph_payload",
            "source_assertion_refs": [row["assertion_id"] for row in rows],
        }

    def search(self, *, ctx: RequestContext, request: dict[str, Any]) -> list[dict[str, Any]]:
        organization_id = _organization_id(ctx)
        query = _required_text(request, "query")
        rows = self.store.search_nodes(
            organization_id=organization_id,
            query=query,
            limit=_int(request.get("limit"), default=20, maximum=50),
        )
        return [
            {
                "object_ref": row["node_id"],
                "label": row.get("display_name") or row["canonical_key"],
                "node_kind": _node_kind(row["node_type"]),
                "summary": row["canonical_key"],
                "rank": index + 1,
                "lens_hint": "overview",
                "path_summary": None,
            }
            for index, row in enumerate(rows)
        ]

    def query(self, *, ctx: RequestContext, request: dict[str, Any]) -> dict[str, Any]:
        organization_id = _organization_id(ctx)
        operation = _required_text(request, "operation")
        start_node_id = ""
        if operation not in {"rank_relationships", "explain_selection"}:
            start_node_id = _required_text(request, "node_id", "start_node_id", "object_ref")
        max_depth = _int(request.get("max_depth"), default=2, maximum=5)
        limit = _int(request.get("limit"), default=100, maximum=150)
        predicates = request.get("predicates")
        if predicates is not None and not isinstance(predicates, list):
            raise ValueError("predicates must be a list")
        predicate_list = [str(item) for item in predicates] if predicates else None
        if operation == "rank_relationships":
            return self._rank_relationships(
                organization_id=organization_id,
                request=request,
            )
        if operation == "explain_selection":
            return self._explain_selection(
                organization_id=organization_id,
                request=request,
            )
        if operation in {"neighbors", "expand"}:
            rows = self.store.get_neighbors(
                organization_id=organization_id,
                node_id=start_node_id,
                predicates=predicate_list or DEFAULT_PREDICATES,
                governance_states=READ_STATES,
                limit=limit,
            )
        elif operation == "authority_path":
            rows = self.store.trace_lineage(
                organization_id=organization_id,
                start_node_id=start_node_id,
                predicates=predicate_list or ["GOVERNED_BY", "CONSTRAINS", "ACTIVATED_AS", "INCLUDED_IN"],
                governance_states=["active"],
                max_depth=max_depth,
                limit=limit,
            )
        elif operation == "lineage":
            rows = self.store.trace_lineage(
                organization_id=organization_id,
                start_node_id=start_node_id,
                predicates=predicate_list or DEFAULT_PREDICATES,
                governance_states=READ_STATES,
                max_depth=max_depth,
                limit=limit,
            )
        elif operation == "impact":
            rows = self.store.trace_lineage(
                organization_id=organization_id,
                start_node_id=start_node_id,
                predicates=predicate_list or ["DEPENDS_ON", "PRODUCES", "CONSUMES", "GOVERNED_BY"],
                governance_states=READ_STATES,
                max_depth=max_depth,
                limit=limit,
            )
        else:
            raise ValueError("unsupported graph query operation")
        return {
            "schema_version": "graph_query_result_v1",
            "operation": operation,
            "object_ref": start_node_id,
            "paths": rows,
            "incomplete": len(rows) >= limit,
        }

    def _rank_relationships(self, *, organization_id: str, request: dict[str, Any]) -> dict[str, Any]:
        """Return a stable, read-only relationship ranking for chat and graph UI."""
        limit = _int(request.get("limit"), default=10, maximum=50)
        with self.store._connect() as conn:
            rows = conn.execute(
                hosted.sql_query("""
                SELECT assertion_id, subject_id, predicate, object_id,
                       governance_state, confidence, content_hash
                FROM graph_assertions
                WHERE organization_id = ?
                  AND governance_state IN (?, ?, ?)
                ORDER BY assertion_id
                LIMIT 500
                """, '\n                SELECT assertion_id, subject_id, predicate, object_id,\n                       governance_state, confidence, content_hash\n                FROM graph_assertions\n                WHERE organization_id = %s\n                  AND governance_state IN (%s, %s, %s)\n                ORDER BY assertion_id\n                LIMIT 500\n                '),
                (organization_id, "active", "approved", "validated"),
            ).fetchall()
        ranked = []
        for row in rows:
            confidence = float(row["confidence"] or 0.0)
            state_weight = {"active": 3, "approved": 2, "validated": 1}[row["governance_state"]]
            score = round(state_weight * 1000 + confidence * 100, 6)
            ranked.append({
                "edge_id": row["assertion_id"],
                "source_id": row["subject_id"],
                "predicate": row["predicate"],
                "target_id": row["object_id"],
                "governance_state": row["governance_state"],
                "importance_score": score,
                "evidence_hash": row["content_hash"],
                "ranking_version": "relationship_importance.v1",
            })
        ranked.sort(key=lambda item: (-item["importance_score"], item["edge_id"]))
        return {
            "schema_version": "knowledge_graph_query.v1",
            "operation": "rank_relationships",
            "ranking_version": "relationship_importance.v1",
            "relationships": ranked[:limit],
        }

    def _explain_selection(self, *, organization_id: str, request: dict[str, Any]) -> dict[str, Any]:
        edge_ids = request.get("edge_ids") or request.get("edgeIds") or []
        if not isinstance(edge_ids, list) or not edge_ids:
            raise ValueError("edge_ids must be a non-empty list")
        normalized = [str(edge_id).strip() for edge_id in edge_ids if str(edge_id).strip()]
        if not normalized:
            raise ValueError("edge_ids must contain non-empty identifiers")
        with self.store._connect() as conn:
            rows = conn.execute(
                f"""SELECT assertion_id, subject_id, predicate, object_id, governance_state,
                           source_ref, source_anchor, content_hash
                    FROM graph_assertions
                    WHERE organization_id = {hosted.sql_query("?", "%s")} AND assertion_id IN ({','.join(hosted.sql_query('?', '%s') for _ in normalized)})""",
                (organization_id, *normalized),
            ).fetchall()
        by_id = {row["assertion_id"]: row for row in rows}
        missing = [edge_id for edge_id in normalized if edge_id not in by_id]
        if missing:
            raise KeyError(f"relationship not found: {missing[0]}")
        return {
            "schema_version": "knowledge_graph_query.v1",
            "operation": "explain_selection",
            "ranking_version": "relationship_importance.v1",
            "relationships": [
                {
                    "edge_id": edge_id,
                    "source_id": by_id[edge_id]["subject_id"],
                    "predicate": by_id[edge_id]["predicate"],
                    "target_id": by_id[edge_id]["object_id"],
                    "governance_state": by_id[edge_id]["governance_state"],
                    "source_ref": by_id[edge_id]["source_ref"],
                    "source_anchor": by_id[edge_id]["source_anchor"],
                    "evidence_hash": by_id[edge_id]["content_hash"],
                }
                for edge_id in normalized
            ],
        }

    def compare_conflicts(self, *, ctx: RequestContext, request: dict[str, Any]) -> dict[str, Any]:
        organization_id = _organization_id(ctx)
        left_ref = _required_text(request, "left_assertion_ref")
        right_ref = _required_text(request, "right_assertion_ref")
        left = self._assertion_by_id(organization_id, left_ref)
        right = self._assertion_by_id(organization_id, right_ref)
        if left is None or right is None:
            raise KeyError("assertion not found")
        return {
            "schema_version": "graph_conflict_v1",
            "conflict_ref": f"conflict:{left_ref}:{right_ref}",
            "conflict_type": "semantic",
            "left": self._edge_payload(left),
            "right": self._edge_payload(right),
            "status": "unresolved",
            "resolver_rationale": "No automatic conflict resolver is configured for this comparison.",
            "required_approver_roles": ["legal_reviewer", "compliance_owner"],
        }

    def replay(self, *, ctx: RequestContext, object_ref: str) -> dict[str, Any]:
        organization_id = _organization_id(ctx)
        assertion = self._assertion_by_id(organization_id, object_ref)
        if assertion is None:
            node = self._node_by_id(organization_id, object_ref)
            if node is None:
                raise KeyError(object_ref)
            assertions = self._select_assertions(
                organization_id=organization_id,
                source_ref=None,
                focus_ref=object_ref,
                snapshot_id=None,
                as_of=None,
                limit=20,
            )
        else:
            assertions = [assertion]
        return {
            "schema_version": "lineage_replay_detail_v1",
            "object_ref": object_ref,
            "assertion_refs": [row["assertion_id"] for row in assertions],
            "assertion_hashes": [row["content_hash"] for row in assertions],
            "replay_notice": "Replay uses pinned graph assertion rows and snapshot refs; no live retrieval is performed.",
        }

    def evidence_export(self, *, ctx: RequestContext, request: dict[str, Any]) -> dict[str, Any]:
        organization_id = _organization_id(ctx)
        object_ref = _required_text(request, "target_object_ref", "object_ref")
        payload = self.replay(ctx=ctx, object_ref=object_ref)
        export_payload = {
            "organization_id": organization_id,
            "target_object_ref": object_ref,
            "graph_snapshot_ref": request.get("graph_snapshot_ref"),
            "replay": payload,
        }
        return {
            "schema_version": "governed_graph_evidence_export_v1",
            "export_ref": f"governed_graph_export:{content_hash(export_payload)[:24]}",
            "export_hash": content_hash(export_payload),
            "status": "completed",
            "notice": "Visualization export is a derived evidence chain, not the canonical evidence package.",
        }

    def command(self, *, ctx: RequestContext, request: dict[str, Any]) -> dict[str, Any]:
        organization_id = _organization_id(ctx)
        command = _required_text(request, "command", "action")
        target = _required_text(request, "target_object_ref", "object_ref")
        expected_snapshot = _required_text(request, "expected_graph_snapshot_ref")
        justification = _required_text(request, "justification")
        idempotency_key = _required_text(request, "idempotency_key")
        transition_methods = {
            "validate": self.store.validate_assertion,
            "approve": self.store.approve_assertion,
            "activate": self.store.activate_assertion,
            "reject": self.store.reject_assertion,
            "deprecate": self.store.deprecate_assertion,
        }
        transition_method = transition_methods.get(command)
        if transition_method is not None:
            try:
                updated = transition_method(
                    assertion_id=target,
                    actor_ref=ctx.user_id,
                    governance_event_ref=idempotency_key,
                )
            except (KeyError, ValueError):
                return {
                    "schema_version": "graph_governance_command_result_v1",
                    "status": "rejected",
                    "target_object_ref": target,
                    "graph_snapshot_ref": expected_snapshot,
                    "message": "assertion cannot transition from its current governed state",
                }
            return {
                "schema_version": "graph_governance_command_result_v1",
                "status": "completed",
                "target_object_ref": target,
                "graph_snapshot_ref": expected_snapshot,
                "governed_event_refs": [idempotency_key],
                "message": f"{command} completed for {updated.assertion_id}: {justification}",
            }
        event_commands = {
            "escalate": "escalated",
            "propose_supersession": "supersession_proposed",
            "request_source_correction": "source_correction_requested",
            "request_ontology_correction": "ontology_correction_requested",
            "request_extraction_correction": "extraction_correction_requested",
        }
        event_type = event_commands.get(command)
        if event_type is not None:
            event_ref = self.store.record_graph_governance_event(
                organization_id=organization_id,
                target_object_ref=target,
                event_type=event_type,
                actor_ref=ctx.user_id,
                governance_event_ref=idempotency_key,
                payload={
                    "command": command,
                    "justification": justification,
                    "expected_graph_snapshot_ref": expected_snapshot,
                },
            )
            return {
                "schema_version": "graph_governance_command_result_v1",
                "status": "completed",
                "target_object_ref": target,
                "graph_snapshot_ref": expected_snapshot,
                "governed_event_refs": [event_ref],
                "message": f"{command} recorded for governed review",
            }
        return {
            "schema_version": "graph_governance_command_result_v1",
            "status": "accepted",
            "target_object_ref": target,
            "graph_snapshot_ref": expected_snapshot,
            "governed_event_refs": [idempotency_key],
            "message": "command accepted for governed lifecycle service handoff",
        }

    def index_status(self, *, ctx: RequestContext, source_ref: str) -> dict[str, Any]:
        organization_id = _organization_id(ctx)
        freshness = self.ingestion.get_freshness(
            organization_id=organization_id,
            source_ref=source_ref,
        )
        if freshness is None:
            return asdict(_freshness("stale", source_ref, "index_status_missing"))
        return _freshness_from_row(freshness)

    def index_run(self, *, ctx: RequestContext, request: dict[str, Any]) -> dict[str, Any]:
        organization_id = _organization_id(ctx)
        source_ref = _required_text(request, "source_ref")
        source_text = str(request.get("source_text") or request.get("sourceText") or "")
        idempotency_key = _required_text(request, "idempotency_key")
        if not source_text:
            requested_fingerprint = str(
                request.get("expected_index_fingerprint")
                or content_hash(
                    {
                        "source_ref": source_ref,
                        "reason": request.get("reason"),
                        "idempotency_key": idempotency_key,
                    }
                )
            )
            run = self.ingestion.enqueue_run(
                organization_id=organization_id,
                workspace_id=ctx.workspace_id,
                source_ref=source_ref,
                trigger_type="manual",
                requested_by=ctx.user_id,
                idempotency_key=idempotency_key,
                requested_fingerprint=requested_fingerprint,
                observed_source_hash=None,
            )
            return {
                "schema_version": "governed_graph_index_run_status_v1",
                "outcome": "INDEX_RUN_QUEUED",
                "run": run,
                "freshness": self.index_status(ctx=ctx, source_ref=source_ref),
            }
        result = self.ingestion.index_manual_event(
            organization_id=organization_id,
            workspace_id=ctx.workspace_id,
            source_ref=source_ref,
            source_text=source_text,
            requested_by=ctx.user_id,
            idempotency_key=idempotency_key,
        )
        return {
            "schema_version": "governed_graph_index_run_status_v1",
            **result,
        }

    def _projection_from_assertions(
        self,
        *,
        ctx: RequestContext,
        source_ref: str,
        assertions: list[dict[str, Any]],
        lens: str,
        as_of: str | None,
        focus_ref: str | None,
    ) -> dict[str, Any]:
        organization_id = _organization_id(ctx)
        nodes_by_id = self._nodes_for_assertions(organization_id, assertions)
        node_views = [
            self._node_view(node, index=index)
            for index, node in enumerate(nodes_by_id.values())
        ]
        edge_views = [self._edge_view(row) for row in assertions]
        scene = GraphSceneV1(
            scene_id=f"scene:{lens}",
            title=f"{lens.title()} governed graph",
            lens=_lens(lens),
            node_ids=tuple(node.node_id for node in node_views),
            edge_ids=tuple(edge.edge_id for edge in edge_views),
            layout_profile=_layout_profile(lens),
            focus_node_id=focus_ref if focus_ref in nodes_by_id else (node_views[0].node_id if node_views else None),
            description="Authorized bounded graph projection from Semantier graph kernel.",
        )
        freshness = _freshness("fresh", source_ref, "projection_resolved")
        return build_authorized_graph_projection(
            organization_id=organization_id,
            workspace_id=ctx.workspace_id,
            source_ref=source_ref,
            source_hash=content_hash({"source_ref": source_ref, "assertions": [row["assertion_id"] for row in assertions]}),
            classification_run_ref="graph-kernel-read",
            ontology_version="graph-kernel",
            sensitivity_ontology_version="graph-disclosure-policy",
            resolver_snapshot_hash=content_hash({"assertions": [row["content_hash"] for row in assertions]}),
            redaction_policy_version="governed_graph_route_disclosure.v1",
            nodes=node_views,
            edges=edge_views,
            scenes=[scene],
            disclosure_by_node_id={node.node_id: "allow" for node in node_views},
            graph_snapshot_ref=None,
            graph_snapshot_hash=None,
            assertion_resolver_version="graph_kernel_store.v1",
            as_of=as_of or now_iso(),
            authorization_context_ref=f"authz:{organization_id}:{ctx.workspace_id or 'org'}",
            freshness=freshness,
            warnings=[
                GraphWarningV1(
                    code="bounded_projection",
                    message="Projection is bounded to authorized graph kernel rows.",
                )
            ],
        ).to_dict()

    def _select_assertions(
        self,
        *,
        organization_id: str,
        source_ref: str | None,
        focus_ref: str | None,
        snapshot_id: str | None,
        as_of: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        filters = [hosted.sql_query("a.organization_id = ?", "a.organization_id = %s")]
        params: list[Any] = [organization_id]
        joins = ""
        if snapshot_id:
            joins = hosted.sql_query("JOIN graph_snapshot_assertions gsa ON gsa.assertion_id = a.assertion_id AND gsa.graph_version_id = ?", "JOIN graph_snapshot_assertions gsa ON gsa.assertion_id = a.assertion_id AND gsa.graph_version_id = %s")
            params.insert(0, snapshot_id)
        if source_ref and source_ref != "graph-kernel":
            filters.append(hosted.sql_query('a.source_ref = ?','a.source_ref = %s'))
            params.append(source_ref)
        if focus_ref:
            filters.append(hosted.sql_query('(a.assertion_id = ? OR a.subject_id = ? OR a.object_id = ?)','(a.assertion_id = %s OR a.subject_id = %s OR a.object_id = %s)'))
            params.extend([focus_ref, focus_ref, focus_ref])
        if as_of:
            filters.append(hosted.sql_query('(a.valid_from IS NULL OR a.valid_from <= ?)','(a.valid_from IS NULL OR a.valid_from <= %s)'))
            filters.append(hosted.sql_query('(a.valid_to IS NULL OR a.valid_to > ?)','(a.valid_to IS NULL OR a.valid_to > %s)'))
            params.extend([as_of, as_of])
        with self.store._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT a.*
                FROM graph_assertions a
                {joins}
                WHERE {" AND ".join(filters)}
                ORDER BY a.governance_state, a.assertion_id
                LIMIT {hosted.sql_query("?", "%s")}
                """,
                (*params, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def _nodes_for_assertions(
        self,
        organization_id: str,
        assertions: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        node_ids = sorted({row["subject_id"] for row in assertions} | {row["object_id"] for row in assertions})
        if not node_ids:
            return {}
        with self.store._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT *
                FROM graph_nodes
                WHERE organization_id = {hosted.sql_query("?", "%s")}
                  AND node_id IN ({",".join(hosted.sql_query("?", "%s") for _ in node_ids)})
                ORDER BY node_id
                """,
                (organization_id, *node_ids),
            ).fetchall()
        return {row["node_id"]: dict(row) for row in rows}

    def _node_by_id(self, organization_id: str, node_id: str) -> dict[str, Any] | None:
        with self.store._connect() as conn:
            row = conn.execute(
                hosted.sql_query("SELECT * FROM graph_nodes WHERE organization_id = ? AND node_id = ?", 'SELECT * FROM graph_nodes WHERE organization_id = %s AND node_id = %s'),
                (organization_id, node_id),
            ).fetchone()
        return dict(row) if row else None

    def _assertion_by_id(self, organization_id: str, assertion_id: str) -> dict[str, Any] | None:
        with self.store._connect() as conn:
            row = conn.execute(
                hosted.sql_query("SELECT * FROM graph_assertions WHERE organization_id = ? AND assertion_id = ?", 'SELECT * FROM graph_assertions WHERE organization_id = %s AND assertion_id = %s'),
                (organization_id, assertion_id),
            ).fetchone()
        return dict(row) if row else None

    def _assertions_by_anchor(self, organization_id: str, anchor_id: str) -> list[dict[str, Any]]:
        with self.store._connect() as conn:
            rows = conn.execute(
                hosted.sql_query("""
                SELECT *
                FROM graph_assertions
                WHERE organization_id = ?
                  AND source_anchor = ?
                ORDER BY assertion_id
                LIMIT 50
                """, '\n                SELECT *\n                FROM graph_assertions\n                WHERE organization_id = %s\n                  AND source_anchor = %s\n                ORDER BY assertion_id\n                LIMIT 50\n                '),
                (organization_id, anchor_id),
            ).fetchall()
        return [dict(row) for row in rows]

    def _node_view(self, node: dict[str, Any], *, index: int) -> GraphNodeV1:
        properties = _json_dict(node.get("properties_json"))
        return GraphNodeV1(
            node_id=node["node_id"],
            node_kind=_node_kind(node["node_type"]),
            label=str(node.get("display_name") or properties.get("label") or node["canonical_key"]),
            display_summary=str(properties.get("summary") or node["canonical_key"]),
            content_hash=node["content_hash"],
            governance_state="active",
            semantic_tier=str(properties.get("semantic_tier") or "T3"),
            authority_role=_authority_role(str(properties.get("authority_role") or "structural")),
            jurisdiction=properties.get("jurisdiction"),
            effective_from=properties.get("effective_from"),
            effective_to=properties.get("effective_to"),
            contextual_only=bool(properties.get("contextual_only", False)),
            capabilities=("view", "inspect_evidence", "inspect_authority", "trace_lineage", "trace_impact"),
            detail_ref=f"node:{node['node_id']}",
            source_anchor_refs=tuple(str(item) for item in properties.get("source_anchor_refs", [])),
            position=GraphPositionV1(x=float(80 + (index % 6) * 150), y=float(80 + (index // 6) * 120)),
            metadata={"canonical_key": node["canonical_key"], "node_type": node["node_type"]},
        )

    def _edge_view(self, row: dict[str, Any]) -> GraphEdgeV1:
        return GraphEdgeV1(
            edge_id=row["assertion_id"],
            source_node_id=row["subject_id"],
            target_node_id=row["object_id"],
            predicate=row["predicate"],
            predicate_label=_predicate_label(row["predicate"]),
            predicate_description=f"{row['subject_id']} {row['predicate']} {row['object_id']}",
            direction="forward",
            content_hash=row["content_hash"],
            governance_state=_governance_state(row["governance_state"]),
            semantic_tier=str(row.get("semantic_tier") or "T3"),
            authority_role=_authority_role(row["authority_role"]),
            contextual_only=row["authority_role"] == "interpretive" and row["governance_state"] != "active",
            capabilities=_edge_capabilities(row["governance_state"]),
            detail_ref=f"assertion:{row['assertion_id']}",
            evidence_anchor_refs=(row["source_anchor"],) if row.get("source_anchor") else (),
            assertion_version_ref=row["assertion_id"],
            assertion_hash=row["content_hash"],
            source_assertion_refs=(row["source_ref"],),
            governed_event_refs=(),
            effective_from=row.get("valid_from"),
            effective_to=row.get("valid_to"),
            metadata={
                "assertion_class": row["assertion_class"],
                "extraction_method": row["extraction_method"],
                "extractor_version": row["extractor_version"],
            },
        )

    def _edge_payload(self, row: dict[str, Any]) -> dict[str, Any]:
        return asdict(self._edge_view(row))


def _organization_id(ctx: RequestContext) -> str:
    organization_id = str(ctx.organization_id or "").strip()
    if not organization_id:
        raise ValueError("organization context required")
    return organization_id


def _required_text(payload: dict[str, Any], *keys: str) -> str:
    value = _optional_text(payload, *keys)
    if not value:
        raise ValueError(f"{keys[0]} required")
    return value


def _optional_text(payload: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = payload.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _int(value: Any, *, default: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(1, min(parsed, maximum))


def _json_dict(value: Any) -> dict[str, Any]:
    import json

    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _node_kind(value: str) -> Any:
    return {
        "segment": "source_span",
        "table": "record",
        "column": "field",
        "dataset": "record",
        "law": "authority",
    }.get(value, value if value in {"source", "source_span", "record", "field", "entity", "relation", "classification", "claim", "rule", "policy", "authority", "conflict", "derivative", "actor", "derived_result"} else "entity")


def _authority_role(value: str) -> Any:
    return value if value in {"structural", "binding", "interpretive", "empirical", "contextual", "conflict"} else "structural"


def _governance_state(value: str) -> Any:
    return value if value in {"candidate", "validated", "approved", "active", "deprecated", "rejected"} else "candidate"


def _edge_capabilities(governance_state: str) -> tuple[Any, ...]:
    base = ("view", "inspect_evidence", "inspect_authority", "trace_lineage", "trace_impact", "compare", "export_evidence")
    actions = {
        "candidate": ("validate", "reject", "escalate", "request_source_correction"),
        "validated": ("approve", "activate", "reject", "deprecate", "escalate"),
        "approved": ("activate", "reject", "deprecate", "escalate"),
        "active": ("deprecate", "propose_supersession", "request_source_correction"),
        "deprecated": (),
        "rejected": (),
    }.get(governance_state, ())
    return (*base, *actions)


def _lens(value: str) -> Any:
    return value if value in {"overview", "evidence", "sensitivity", "authority", "conflict", "lineage", "replay", "impact"} else "overview"


def _layout_profile(lens: str) -> Any:
    return {
        "authority": "authority_hierarchy",
        "conflict": "conflict_comparison",
        "lineage": "lineage",
        "impact": "impact_neighborhood",
        "replay": "replay_chain",
    }.get(lens, "evidence_chain")


def _predicate_label(predicate: str) -> str:
    return predicate.lower().replace("_", " ")


def _freshness(status: Any, source_ref: str, message: str) -> GraphIndexStatusV1:
    return GraphIndexStatusV1(
        status=status,
        source_ref=source_ref,
        checked_at=now_iso(),
        index_fingerprint=None,
        message=message,
    )


def _freshness_from_row(row: dict[str, Any]) -> dict[str, Any]:
    status = row.get("status")
    normalized = status if status in {"fresh", "stale", "indexing", "failed_retryable", "failed_terminal"} else "stale"
    return asdict(
        GraphIndexStatusV1(
            status=normalized,
            source_ref=row["source_ref"],
            checked_at=row.get("updated_at") or now_iso(),
            index_fingerprint=row.get("source_fingerprint"),
            message=row.get("error_code"),
        )
    )


def _path_summary(row: dict[str, Any]) -> dict[str, Any]:
    return asdict(
        GraphPathSummaryV1(
            path_id=f"path:{row['assertion_id']}",
            object_refs=(row["subject_id"], row["object_id"]),
            explanation=f"{row['subject_id']} -> {row['predicate']} -> {row['object_id']}",
            incomplete=False,
        )
    )
