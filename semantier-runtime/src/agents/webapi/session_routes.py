from __future__ import annotations

import json
import hashlib
import mimetypes
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

from fastapi import APIRouter
from agents.workspace_session_logs import workspace_session_dir

import os
import re
import sqlite3
import sys
import uuid

from fastapi import HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from agents.hermes_boundary_provider import durable_request_metadata
from agents.auth_session import RequestContext
from agents.runtime_memory_boundary import reconcile_retained_context_items
from agents.governed_context import (
    build_governed_runtime_context_prompt,
    is_analytics_query_message,
)
from agents.sandbox_scope import SandboxScope, sandbox_key_for_request
from agents.session_lineage import build_session_lineage
from agents.workspace_session_logs import (
    WorkspaceSessionResolutionError,
    create_workspace_session_log,
    delete_workspace_session_log,
    derive_workspace_session_transport_metadata,
    get_workspace_session_log_payload,  # noqa: F401 - compatibility export via agents.webapi_gateway
    list_workspace_session_trajectory,
    link_workspace_session_trajectory,
    replace_workspace_session_log_messages,
    resolve_or_create_workspace_session_id,
    resolve_workspace_session_id,  # noqa: F401 - compatibility export via agents.webapi_gateway
    update_workspace_session_title,
)
from agents.webapi.deps import (
    enforce_rate_limit as _enforce_rate_limit,
    ensure_repo_hermes_agent_on_path as _ensure_repo_hermes_agent_on_path,
    require_authenticated_context as _require_authenticated_context,
)
from agents.webapi.embedded_proxy_service import (
    _post_embedded_chat_completion,
    _post_tender_runtime_expert_completion,
    _stream_embedded_chat_completion,
)
from agents.webapi.session_service import (
    _AFFIRMATIVE_FOLLOWUP_MESSAGES,
    _CRON_ERROR_FOLLOWUP_CONTEXT_PROMPT,
    _LOCAL_HERMES_HOME,
    _LOGGER,
    _SHORT_AFFIRMATIVE_FOLLOWUP_CONTEXT_PROMPT,
)
from contracts.knowledge_workbench_context import (
    ActiveFunctionV4,
    KnowledgeWorkbenchContextV4,
    UserIntentV4,
)
from gateway.execution_boundary import (
    BoundaryPaths,
    BoundaryPolicy,
    ExecutionBoundary,
)
from pydantic import ValidationError
from runtime_paths import sqlite_db_path

from semantier.effective_context import EffectiveContextCheckpointService
from semantier.semantica_context_graph.artifact_repository import ContextGraphArtifactRepository
from semantier.semantica_context_graph.bridge import TenderExpertRequest
from semantier.semantica_context_graph.tender_mapper import MAPPER_VERSION
from semantier.semantica_context_graph.workflow_context import (
    WorkflowContextRejected,
    resolve_untrusted_workbench_hint,
)
from storage.human_grounding_store import HumanGroundingStore
from services.ontology_candidate_context_resolver import (
    OntologyCandidateContextRejected,
    resolve_ontology_candidate_context,
)
from storage.ontology_build_store import OntologyBuildStore, ontology_build_store_path

router = APIRouter(tags=["webapi-gateway"])

_SESSION_TITLE_MAX_LENGTH = 100
_PROMPTS_ROOT = Path(__file__).resolve().parents[2] / "prompts" / "agents"


def _branch_session_at_message(
    db: Any,
    source_session_id: str,
    child_session_id: str,
    *,
    title: str,
    branch_point_message_id: str,
    branch_point_sequence: int,
    idempotency_key: str,
    request_hash: str,
    authenticated_context: RequestContext | None = None,
) -> dict[str, Any]:
    # Freeze inherited restriction before the native child can be persisted. A
    # failed branch may leave a harmless restrictive binding, never an escape.
    if authenticated_context is not None:
        from agents.wiki_task_runtime import bind_wiki_task, wiki_profile_actor
        if wiki_profile_actor(authenticated_context.workspace_id, source_session_id):
            bind_wiki_task(authenticated_context, child_session_id, "llm-wiki")
    legacy_branch = getattr(db, "branch_session_at_message", None) or getattr(db, "branch_session", None)
    if callable(legacy_branch):
        return legacy_branch(
            source_session_id,
            child_session_id,
            title=title,
            branch_point_message_id=branch_point_message_id,
            branch_point_sequence=branch_point_sequence,
            idempotency_key=idempotency_key,
            request_hash=request_hash,
        )

    source = db.get_session(source_session_id)
    if source is None:
        raise KeyError("SOURCE_SESSION_NOT_FOUND")
    messages = db.get_messages(source_session_id)
    lineage_root_id = str(
        source.get("lineage_root_id")
        or source.get("parent_session_id")
        or source_session_id
    )
    if branch_point_sequence > len(messages):
        raise KeyError("BRANCH_POINT_NOT_FOUND")
    boundary = messages[branch_point_sequence - 1]
    if (
        str(boundary.get("id") or "") != branch_point_message_id
        or str(boundary.get("role") or "") != "assistant"
    ):
        raise KeyError("BRANCH_POINT_NOT_FOUND")

    with sqlite3.connect(str(db.db_path)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute(
            """CREATE TABLE IF NOT EXISTS semantier_session_branches (
                child_session_id TEXT PRIMARY KEY,
                source_session_id TEXT NOT NULL,
                branch_point_message_id TEXT NOT NULL,
                branch_point_sequence INTEGER NOT NULL,
                title TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                request_hash TEXT NOT NULL
            )"""
        )
        existing_branch = conn.execute(
            "SELECT * FROM semantier_session_branches WHERE child_session_id = ?",
            (child_session_id,),
        ).fetchone()
    if db.get_session(child_session_id) is not None:
        if existing_branch is None or any(
            (
                str(existing_branch["source_session_id"]) != source_session_id,
                str(existing_branch["branch_point_message_id"])
                != branch_point_message_id,
                int(existing_branch["branch_point_sequence"])
                != branch_point_sequence,
                str(existing_branch["title"]) != title,
                str(existing_branch["idempotency_key"]) != idempotency_key,
                str(existing_branch["request_hash"]) != request_hash,
            )
        ):
            raise ValueError("IDEMPOTENCY_CONFLICT")
        return {
            "id": child_session_id,
            "session_id": child_session_id,
            "parent_session_id": source_session_id,
            "lineage_root_id": lineage_root_id,
            "branch_point_message_id": branch_point_message_id,
            "branch_point_sequence": branch_point_sequence,
            "message_count": len(db.get_messages(child_session_id)),
            "idempotent": True,
        }

    source_model_config = source.get("model_config")
    if isinstance(source_model_config, str):
        try:
            parsed_model_config = json.loads(source_model_config)
        except json.JSONDecodeError:
            parsed_model_config = {}
    elif isinstance(source_model_config, dict):
        parsed_model_config = dict(source_model_config)
    else:
        parsed_model_config = {}
    parsed_model_config["_branched_from"] = source_session_id
    db.create_session(
        child_session_id,
        source=str(source.get("source") or "api_server"),
        model=source.get("model"),
        model_config=parsed_model_config,
        system_prompt=source.get("_system_prompt_resolved") or source.get("system_prompt"),
        user_id=source.get("user_id"),
        parent_session_id=source_session_id,
        display_name=title,
    )
    try:
        # The child owns a new transcript occurrence through the inclusive
        # boundary. Preserve message payloads, never source-row identities.
        db.append_messages_batch(
            child_session_id,
            [
                {
                    key: value
                    for key, value in message.items()
                    if key not in {
                        "id", "session_id", "message_uid", "_row_id",
                        "_db_row_snapshot", "_canonical_row",
                    }
                }
                for message in messages[:branch_point_sequence]
            ],
        )
        with sqlite3.connect(str(db.db_path)) as conn:
            conn.execute(
                """INSERT INTO semantier_session_branches(
                    child_session_id, source_session_id, branch_point_message_id,
                    branch_point_sequence, title, idempotency_key, request_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    child_session_id,
                    source_session_id,
                    branch_point_message_id,
                    branch_point_sequence,
                    title,
                    idempotency_key,
                    request_hash,
                ),
            )
            conn.commit()
    except BaseException:
        db.delete_session(child_session_id)
        raise
    return {
        "id": child_session_id,
        "session_id": child_session_id,
        "parent_session_id": source_session_id,
        "lineage_root_id": lineage_root_id,
        "branch_point_message_id": branch_point_message_id,
        "branch_point_sequence": branch_point_sequence,
        "message_count": branch_point_sequence,
        "idempotent": False,
    }
_WEBAPI_BUNDLED_SKILL_INVOCATION_PROMPT = (
    _PROMPTS_ROOT / "WEBAPI_BUNDLED_SKILL_INVOCATION.md"
)


def _filter_effective_context_events(
    events: list[Any],
    *,
    state_effect: str | None = None,
    source_type: str | None = None,
    actor: str | None = None,
    sequence_from: int | None = None,
    sequence_to: int | None = None,
) -> list[Any]:
    return [
        event
        for event in events
        if (not state_effect or event.state_effect.value == state_effect)
        and (
            not source_type
            or any(
                ref.ref_type.value == source_type
                for ref in (*event.input_refs, *event.output_refs)
            )
        )
        and (not actor or event.actor_ref == actor)
        and (sequence_from is None or event.sequence >= sequence_from)
        and (sequence_to is None or event.sequence <= sequence_to)
    ]


def _compat_attr(name: str):
    compat_module = sys.modules.get("agents.webapi_gateway")
    if compat_module is not None and name in vars(compat_module):
        return getattr(compat_module, name)
    return globals()[name]


def _normalize_session_label(label: str | None) -> str | None:
    if not isinstance(label, str):
        return None
    normalized = re.sub(r"\s+", " ", label).strip()
    if not normalized:
        return None
    if len(normalized) <= _SESSION_TITLE_MAX_LENGTH:
        return normalized
    return normalized[: _SESSION_TITLE_MAX_LENGTH - 3].rstrip() + "..."


def _set_unique_session_title(db, session_id: str, label: str | None) -> str | None:
    """Keep native collision numbering inside the session title length contract."""
    candidate = db.sanitize_title(label)
    while True:
        try:
            db.set_session_title(session_id, candidate or "")
            return candidate
        except ValueError as exc:
            if "already in use by session" not in str(exc):
                raise
            candidate = db.get_next_title_in_lineage(candidate or "")
            match = re.fullmatch(r"(.*)( #\d+)", candidate)
            if match:
                stem, suffix = match.groups()
                candidate = stem[: _SESSION_TITLE_MAX_LENGTH - len(suffix)].rstrip() + suffix
            # Truncation may collide with a previously shortened title. Resolve
            # that lineage too, including suffix growth from #9 to #10.


_ALLOWED_V4_ACTIVE_FUNCTIONS = frozenset(
    {
        ("contextgraph-studio", "source_registration", "sources"),
        ("contextgraph-studio", "schema_discovery", "schema"),
        ("contextgraph-studio", "candidate_extraction", "extract"),
        ("contextgraph-studio", "human_grounding", "ground"),
        ("contextgraph-studio", "graph_exploration", "graph"),
        ("contextgraph-studio", "tender_inspection", "inspect"),
        ("contextgraph-studio", "release_comparison", "compare"),
        ("contextgraph-studio", "evaluation", "evaluate"),
        ("semantica-showcase2", "knowledge-graph_exploration", "knowledge-graph"),
        ("semantica-showcase2", "ontology_exploration", "ontology"),
        ("semantica-showcase2", "embedding_exploration", "embedding"),
        ("semantica-showcase2", "semantic-network_exploration", "semantic-network"),
        ("semantica-showcase2", "temporal_exploration", "temporal"),
        ("semantica-showcase2", "analytics_exploration", "analytics"),
        ("semantica-showcase2", "lineage_exploration", "lineage"),
    }
)


def _allowlisted_v4_active_function(
    value: ActiveFunctionV4 | None,
) -> dict[str, str] | None:
    if value is None:
        return None
    identity = (value.surface, value.function, value.tab)
    if identity not in _ALLOWED_V4_ACTIVE_FUNCTIONS:
        return None
    return value.model_dump(mode="json")


def _resolve_v4_user_intent(
    value: UserIntentV4 | None,
    trusted: dict[str, Any],
) -> dict[str, Any] | None:
    if value is None:
        return None
    resolved = {
        "kind": value.kind,
        "source": value.source,
        "explicitness": value.explicitness,
        "targetType": None,
        "targetIds": [],
    }
    governed_targets = {
        "node": tuple(trusted.get("selectedNodeIds") or ()),
        "edge": tuple(trusted.get("selectedEdgeIds") or ()),
    }
    allowed_ids = governed_targets.get(value.target_type or "", ())
    requested_ids = tuple(value.target_ids)
    if requested_ids and all(item in allowed_ids for item in requested_ids):
        resolved["targetType"] = value.target_type
        resolved["targetIds"] = list(requested_ids)
    return resolved


def _resolve_v4_workbench_selection(
    ctx: RequestContext,
    raw: dict[str, Any],
) -> dict[str, Any]:
    try:
        parsed = KnowledgeWorkbenchContextV4.model_validate(raw)
    except ValidationError as exc:
        raise HTTPException(
            status_code=400, detail="invalid KnowledgeWorkbenchContext.v4 schema"
        ) from exc
    organization_id = str(ctx.organization_id or "").strip()
    workspace_id = str(ctx.workspace_id or "").strip()
    if not organization_id or not workspace_id:
        raise HTTPException(status_code=401, detail="authenticated tenant context required")
    unsupported = (
        parsed.selected_exemplar_evaluation_ref,
        parsed.selected_canonical_pattern_ref,
        parsed.selected_observed_hypothesis_ref,
        parsed.selected_finding_derivation_ref,
    )
    if any(unsupported):
        raise HTTPException(
            status_code=409,
            detail="stale or invalid workbench context: governed resolver unavailable",
        )
    graph_pins = (parsed.graph_ref, parsed.graph_version, parsed.graph_hash)
    if any(graph_pins) and not all(graph_pins):
        raise HTTPException(
            status_code=400, detail="graphRef, graphVersion, and graphHash are required"
        )
    if (
        parsed.selected_node_ids or parsed.selected_edge_ids or parsed.source_anchors
    ) and not all(graph_pins):
        raise HTTPException(
            status_code=400, detail="graph pins are required for graph selections"
        )
    if (
        parsed.selected_rule_ids
        or parsed.selected_candidate_id is not None
        or parsed.selected_evidence_refs
    ):
        raise HTTPException(
            status_code=409,
            detail="stale or invalid workbench context: governed selection resolver unavailable",
        )
    if (
        parsed.selected_source_semantic_witness_ref is not None
        and parsed.source_semantic_witness_set_ref is None
    ):
        raise HTTPException(
            status_code=409,
            detail="stale or invalid workbench context: witness set is required",
        )
    trusted = {
        "schemaVersion": "knowledge_workbench_context.v4",
        "activeFunction": _allowlisted_v4_active_function(parsed.active_function),
        "userIntent": None,
        "graphRef": None,
        "graphVersion": None,
        "graphHash": None,
        "authorityState": None,
        "runMode": None,
        "candidateGraphId": None,
        "acceptedReleaseId": None,
        "acceptedReleaseVersion": None,
        "selectedNodeIds": [],
        "selectedEdgeIds": [],
        "selectedRuleIds": [],
        "selectedCandidateId": None,
        "selectedEvidenceRefs": [],
        "activeSourceIdentityRef": None,
        "sourceAnchors": [],
        "governanceState": "candidate",
        "hasAcceptedRelease": False,
        "extractionRunId": None,
        "providerRef": None,
        "providerCommit": None,
        "targetEvidenceRef": None,
        "activeRuleVersionId": None,
        "graphRuleId": None,
        "originEvidenceRef": None,
        "schemaCandidateRef": None,
        "schemaCandidateHash": None,
        "schemaDecisionRef": None,
        "schemaDecisionHash": None,
        "selectedSchemaAmbiguityIds": [],
        "gateOperatingMode": None,
        "gateFallbackReason": None,
        "canonicalSourceIrRef": None,
        "canonicalSourceIrHash": None,
        "graphIntentBriefRef": None,
        "graphIntentBriefHash": None,
        "graphIntentReviewState": None,
        "graphIntentSelectedQuestionIds": [],
        "graphIntentUnresolvedMaterialQuestionIds": [],
        "sourceSemanticWitnessSetRef": None,
        "sourceSemanticWitnessSetHash": None,
        "selectedSourceSemanticWitnessRef": None,
        "selectedSourceSemanticWitnessHash": None,
        "selectedExemplarEvaluationRef": None,
        "selectedExemplarEvaluationHash": None,
        "selectedCanonicalPatternRef": None,
        "selectedCanonicalPatternHash": None,
        "selectedObservedHypothesisRef": None,
        "selectedObservedHypothesisHash": None,
        "selectedFindingDerivationRef": None,
        "selectedFindingDerivationHash": None,
    }
    trusted["userIntent"] = _resolve_v4_user_intent(parsed.user_intent, trusted)
    if parsed.source_semantic_witness_set_ref is None:
        return KnowledgeWorkbenchContextV4.model_validate(trusted).model_dump(
            mode="json", by_alias=True
        )

    db_path = Path(
        os.environ.get(
            "SEMANTIER_EOS_DB_PATH",
            str(sqlite_db_path("eos.db", ".semantier/eos.db")),
        )
    )
    from storage import hosted_postgres as hosted
    if not hosted.enabled() and not db_path.is_file():
        raise HTTPException(status_code=409, detail="stale or invalid workbench context")
    store = HumanGroundingStore(db_path, graph_backend=None)
    try:
        row = store._conn.execute(
            hosted.sql_query(
                """SELECT artifact_ref FROM source_semantic_witness_manifests
                WHERE organization_id=? AND workspace_id=? AND manifest_ref=?
                  AND content_hash=?""",
                """SELECT artifact_ref FROM source_semantic_witness_manifests
                WHERE organization_id=%s AND workspace_id=%s AND manifest_ref=%s
                  AND content_hash=%s""",
            ),
            (
                organization_id,
                workspace_id,
                parsed.source_semantic_witness_set_ref,
                parsed.source_semantic_witness_set_hash,
            ),
        ).fetchone()
        if row is None:
            raise ValueError("witness manifest pin is missing")
        manifest = store.replay_source_semantic_witness_manifest(
            organization_id=organization_id,
            workspace_id=workspace_id,
            manifest_ref=parsed.source_semantic_witness_set_ref,
            content_hash=parsed.source_semantic_witness_set_hash,
            artifact_ref=row["artifact_ref"],
        )
        if parsed.selected_source_semantic_witness_ref is not None:
            selected = store.replay_source_semantic_witness(
                organization_id=organization_id,
                workspace_id=workspace_id,
                witness_ref=parsed.selected_source_semantic_witness_ref,
                witness_hash=parsed.selected_source_semantic_witness_hash,
            )
            if (
                selected.witness_id,
                selected.witness_hash,
            ) not in {
                (entry.witness_ref, entry.witness_hash)
                for entry in manifest.witnesses
            }:
                raise ValueError("selected witness is not in the pinned witness set")
    except (KeyError, ValueError) as exc:
        raise HTTPException(
            status_code=409, detail="stale or invalid workbench context"
        ) from exc
    finally:
        store._conn.close()

    trusted["sourceSemanticWitnessSetRef"] = manifest.manifest_ref
    trusted["sourceSemanticWitnessSetHash"] = manifest.content_hash
    if parsed.selected_source_semantic_witness_ref is not None:
        trusted["selectedSourceSemanticWitnessRef"] = selected.witness_id
        trusted["selectedSourceSemanticWitnessHash"] = selected.witness_hash
    trusted["activeSourceIdentityRef"] = manifest.source_binding.source_identity_ref
    trusted["canonicalSourceIrRef"] = manifest.source_binding.canonical_source_ir_ref
    trusted["canonicalSourceIrHash"] = manifest.source_binding.canonical_source_ir_hash
    trusted["userIntent"] = _resolve_v4_user_intent(parsed.user_intent, trusted)
    return KnowledgeWorkbenchContextV4.model_validate(trusted).model_dump(
        mode="json", by_alias=True
    )


def _resolve_workbench_context_hint(ctx: RequestContext, raw: Any) -> str | None:
    """Re-resolve browser graph context before bounded specialist use."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise HTTPException(status_code=400, detail="knowledge_workbench_context must be an object")
    active = raw.get("activeFunction")
    if isinstance(active, dict) and active.get("surface") == "semantic-projection-showcase":
        from contracts.semantic_projection_context import resolve_semantic_projection_context
        if not ctx.organization_id or not ctx.workspace_id:
            raise HTTPException(status_code=400, detail="governed workspace required")
        if raw.get("schemaVersion") != "knowledge_workbench_context.v2" or any(raw.get(k) for k in ("graphRef", "graphVersion", "graphHash", "candidateGraphId", "acceptedReleaseId", "workflowSimulation")):
            raise HTTPException(status_code=400, detail="recorded specimen cannot carry graph authority")
        try:
            selection = resolve_semantic_projection_context(raw.get("semanticProjection"))
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return json.dumps({"schemaVersion": "knowledge_workbench_context.v2", "activeFunction": {"surface": "semantic-projection-showcase", "function": "recorded_semantic_projection", "tab": "specimen"}, "graphRef": None, "authorityState": "candidate", "semanticProjection": selection}, ensure_ascii=False, sort_keys=True)
    v4_context = None
    if raw.get("schemaVersion") == "knowledge_workbench_context.v4":
        v4_context = _resolve_v4_workbench_selection(ctx, raw)
    organization_id = str(ctx.organization_id or "").strip()
    workspace_id = str(ctx.workspace_id or "").strip()
    artifact_id = str(raw.get("graphRef") or raw.get("graph_ref") or "").strip()
    graph_version = str(raw.get("graphVersion") or raw.get("graph_version") or "").strip()
    graph_hash = str(raw.get("graphHash") or raw.get("graph_hash") or "").strip()
    if v4_context is not None and not any((artifact_id, graph_version, graph_hash)):
        # Navigation and pinned source witnesses do not require a graph. The
        # v4 owner has already rebuilt them without browser-supplied authority.
        return json.dumps(v4_context, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    active_function = raw.get("activeFunction")
    surface = (
        str(active_function.get("surface") or "").strip()
        if isinstance(active_function, dict)
        else ""
    )
    candidate_ref = str(raw.get("candidateGraphId") or "").strip()
    is_ontology_candidate = (
        raw.get("schemaVersion") == "knowledge_workbench_context.v2"
        and surface in {"ontology-workbench", "ontology-workbench2"}
        and bool(candidate_ref or graph_hash)
    )
    if is_ontology_candidate:
        if (
            not organization_id
            or not workspace_id
            or not candidate_ref
            or not graph_hash
            or artifact_id
            or graph_version
        ):
            raise HTTPException(
                status_code=400,
                detail="candidateGraphId and graphHash are required without graphRef/graphVersion",
            )
        store = OntologyBuildStore(ontology_build_store_path(ctx.workspace_root))
        try:
            resolved_candidate = resolve_ontology_candidate_context(
                store,
                organization_id=organization_id,
                workspace_id=workspace_id,
                candidate_ref=candidate_ref,
                candidate_hash=graph_hash,
                browser_hint=raw,
            )
        except OntologyCandidateContextRejected as exc:
            raise HTTPException(
                status_code=409,
                detail="stale or invalid ontology candidate context",
            ) from exc
        finally:
            store.close()
        return json.dumps(
            resolved_candidate,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    if not organization_id or not artifact_id or not graph_version or not graph_hash:
        raise HTTPException(status_code=400, detail="graphRef, graphVersion, and graphHash are required")
    repository = ContextGraphArtifactRepository(
        ctx.workspace_root,
        organization_id=organization_id,
        workspace_id=ctx.workspace_id,
    )
    try:
        canonical, pointers = resolve_untrusted_workbench_hint(
            repository,
            tenant_id=organization_id,
            domain="tender-sensitive",
            artifact_id=artifact_id,
            graph_version=graph_version,
            graph_hash=graph_hash,
            browser_hint=raw,
        )
    except WorkflowContextRejected as exc:
        raise HTTPException(status_code=409, detail="stale or invalid graph context") from exc
    resolved_context = dict(v4_context or {})
    resolved_context.update(
        {
            "graphRef": canonical.graph_ref,
            "graphVersion": canonical.graph_version,
            "graphHash": canonical.graph_hash,
            "authorityState": canonical.authority_state,
            "runMode": canonical.run_mode,
            "selectedNodeIds": pointers["selected_node_ids"],
            "selectedEdgeIds": pointers["selected_edge_ids"],
            "sourceAnchors": pointers["source_anchors"],
        }
    )
    if v4_context is not None:
        parsed_v4 = KnowledgeWorkbenchContextV4.model_validate(raw)
        resolved_context["userIntent"] = _resolve_v4_user_intent(
            parsed_v4.user_intent, resolved_context
        )
        resolved_context = KnowledgeWorkbenchContextV4.model_validate(
            resolved_context
        ).model_dump(mode="json", by_alias=True)
    return json.dumps(
        resolved_context,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


async def _resolve_tender_journey_context_hint(ctx: RequestContext, session_id: str, raw: Any) -> str | None:
    """Resolve navigation-only references through the existing authenticated owners."""
    if raw is None:
        return None
    from contracts.tender_journey import TenderJourneyChatSelectionV1
    from agents.tender_journey_tool_api import TenderJourneyToolAPI
    from services.runtime_errors import RuntimeContextError
    try:
        selection = TenderJourneyChatSelectionV1.model_validate(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="INVALID_TENDER_JOURNEY_SELECTION") from exc
    try:
        api = TenderJourneyToolAPI(context=ctx, session_ref=session_id)
        episode = None
        if selection.transition_id is not None:
            saved = await api.invoke(operation="action", payload={
                "action": "read_episode", "journey_id": selection.journey_id,
                "arguments": {"transition_id": selection.transition_id},
            })
            episode = saved["episode"]
        current = await api.invoke(operation="resolve", payload={"journey_id": selection.journey_id})
        return json.dumps({
            "schema_version": "tender_journey_chat_context.v1",
            "current_journey": current["journey"],
            "selected_episode": episode,
        }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="JOURNEY_EPISODE_NOT_FOUND") from exc
    except (RuntimeContextError, PermissionError) as exc:
        raise HTTPException(status_code=403, detail="FORBIDDEN") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def _append_workbench_context(system_message: str | None, context_json: str | None) -> str | None:
    if not context_json:
        return system_message
    bounded = f"{system_message}\n" if system_message else ""
    try:
        parsed = json.loads(context_json)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict) and parsed.get("semanticProjection"):
        from contracts.semantic_projection_context import SEMANTIC_PROJECTION_POLICY
        bounded += SEMANTIC_PROJECTION_POLICY + "\n"
    return bounded + context_json


async def _maybe_run_tender_runtime_expert(
    *,
    ctx: RequestContext,
    parent_session_id: str,
    message: str,
    model: str | None,
    context_json: str | None,
    enabled: bool,
) -> Any | None:
    """Use the isolated Runtime Expert only for an explicit graph-workbench turn.

    Main Hermes owns routing only.  The specialist request is reconstructed from
    the server-resolved graph pin; browser-provided profile/stage/certification
    claims never enter the request.
    """
    if not enabled or not context_json:
        return None
    context = json.loads(context_json)
    if not context.get("graphRef"):
        return None
    repository = ContextGraphArtifactRepository(
        ctx.workspace_root,
        organization_id=str(ctx.organization_id or ""),
        workspace_id=ctx.workspace_id,
    )
    metadata = repository.get(str(context["graphRef"]))
    request = TenderExpertRequest(
        organization_id=str(ctx.organization_id or ""),
        workspace_id=ctx.workspace_id,
        parent_session_id=parent_session_id,
        graph_ref=metadata.artifact_id,
        graph_version=metadata.graph_version,
        graph_hash=metadata.content_hash,
        semantica_commit=metadata.semantica_commit,
        message=message,
        run_mode=str(context.get("runMode") or "evaluation_baseline"),
        selected_node_ids=tuple(str(value) for value in context.get("selectedNodeIds", ())),
        selected_edge_ids=tuple(str(value) for value in context.get("selectedEdgeIds", ())),
        source_anchors=tuple(
            value for value in context.get("sourceAnchors", ()) if isinstance(value, dict)
        ),
        mapper_version=MAPPER_VERSION,
        request_id=f"runtime-expert:{uuid.uuid4().hex}",
    )
    result = await _post_tender_runtime_expert_completion(
        request=request,
        hermes_home=ctx.hermes_home,
        model=model,
    )
    return {
        "ok": result.status == "completed",
        "status": result.status,
        "request_id": result.request_id,
        "specialist_session_id": result.specialist_session_id,
        "message": result.message,
        "evidence": list(result.evidence),
        "graph_refs": list(result.graph_refs),
        "interaction": result.interaction,
        "error_code": result.error_code,
    }


def _prepare_governed_analytics_context(
    ctx: RequestContext,
    *,
    user_message: str = "",
) -> dict[str, Any] | None:
    if user_message and not is_analytics_query_message(user_message):
        return None

    try:
        from agents import gateway_identity

        org_context = gateway_identity.get_user_organization_context(ctx.user_id)
    except Exception as exc:
        _LOGGER.warning("governed analytics preflight skipped: %s", exc)
        return None
    if not isinstance(org_context, dict):
        return None

    organization_id = str(
        org_context.get("organization_id") or ctx.organization_id or ""
    ).strip()
    dataset_type = str(
        org_context.get("dataset_type") or ctx.dataset_type or ""
    ).strip().upper()
    active_dataset_version_id = str(
        org_context.get("active_dataset_version_id") or ""
    ).strip()
    dataset_key = str(org_context.get("dataset_key") or "").strip()
    if not organization_id or not dataset_type:
        return None

    activation_context = None
    try:
        from agents.governed_context import resolve_governed_activation_context

        activation_context = resolve_governed_activation_context(
            user_id=ctx.user_id,
            workspace_id=ctx.workspace_id,
        )
    except Exception as exc:
        _LOGGER.warning("governed activation context resolution failed: %s", exc)

    def _with_activation_context(payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(activation_context, dict):
            return payload
        updated = dict(payload)
        updated["activation_set_snapshot_id"] = activation_context.get(
            "activation_set_snapshot_id"
        )
        updated["activation_resolver_policy_version"] = activation_context.get(
            "activation_resolver_policy_version"
        )
        updated["resolved_activation_set_hash"] = activation_context.get(
            "resolved_activation_set_hash"
        )
        return updated

    if dataset_type == "REAL":
        try:
            from eos.real_company_setup_orchestration import run_real_company_setup

            return _with_activation_context(run_real_company_setup(
                organization_id=organization_id,
                dataset_type=dataset_type,
                active_dataset_version_id=active_dataset_version_id or None,
                requested_by=ctx.user_id,
                idempotency_key=f"web-session-preflight:{ctx.workspace_id}:{organization_id}",
                member_role=ctx.member_role,
                can_change_settings=ctx.can_change_settings,
            ))
        except Exception as exc:
            _LOGGER.warning("real-company setup preflight failed: %s", exc)
            return None

    if dataset_type != "DEMO" or not dataset_key:
        return None

    try:
        from eos.real_company_lakehouse_refresh import refresh_lakehouse
        from plugins.governed_analytics import _check_analytics_requirements

        ok, message = _check_analytics_requirements()
        if ok:
            return _with_activation_context({"ok": True, "setup_status": "DEMO_ACTIVE", "refreshed": False})
        if not str(message).startswith("LAKEHOUSE_"):
            return _with_activation_context({"ok": False, "setup_status": "DEMO_ACTIVE", "error": message})
        return _with_activation_context(refresh_lakehouse(
            organization_id=organization_id,
            dataset_version_id=dataset_key,
            requested_by=ctx.user_id,
            idempotency_key=f"web-session-preflight:{ctx.workspace_id}:{organization_id}:{dataset_key}",
            replace=True,
        ))
    except Exception as exc:
        _LOGGER.warning("demo lakehouse preflight failed: %s", exc)
        return None



@router.get("/sessions")
async def list_sessions(
    request: Request,
    limit: int = Query(default=50, ge=1, le=500),
):
    def read_in_worker():
        ctx = _require_authenticated_context(request)
        payload = {"sessions": _list_workspace_sessions_primary(ctx, limit=limit)}
        return JSONResponse(_normalize_sessions(payload, ctx))
    return await run_in_threadpool(read_in_worker)


@router.get("/sessions/{session_key:path}/trajectory")
@router.get("/sessions/{session_key}/trajectory")
async def get_session_trajectory(request: Request, session_key: str) -> Any:
    ctx = _require_authenticated_context(request)
    internal_id = _resolve_workspace_runtime_session_id(ctx, session_key)
    if not internal_id:
        raise HTTPException(status_code=400, detail="session_key required")
    rows = list_workspace_session_trajectory(ctx.hermes_home, internal_id)
    lineage_rows = [
        row for row in rows if row.get("record_type") == "session_lineage"
    ]
    trajectory_rows = [
        row for row in rows if row.get("record_type") != "session_lineage"
    ]
    lineage_links = [
        {
            "parentSessionId": _external_session_id(
                str(row.get("parent_session_id") or ""), ctx
            ),
            "lineageRootId": _external_session_id(
                str(row.get("lineage_root_id") or ""), ctx
            ),
            "branchPointMessageId": row.get("branch_point_message_id"),
            "branchPointSequence": row.get("branch_point_message_sequence"),
            "parentTrajectoryRef": row.get("parent_trajectory_ref"),
        }
        for row in lineage_rows
    ]

    if not trajectory_rows:
        run_trajectory, run_source_file = _trajectory_from_latest_run(
            ctx.hermes_home,
            internal_id,
        )
        if run_trajectory is not None:
            return JSONResponse(
                {
                    "session_id": internal_id,
                    "sessionKey": _external_session_id(internal_id, ctx),
                    "title": session_key,
                    "source_file": run_source_file,
                    "trajectory": run_trajectory,
                    "lineageLinks": lineage_links,
                }
            )
        if lineage_links:
            return JSONResponse(
                {
                    "session_id": internal_id,
                    "sessionKey": _external_session_id(internal_id, ctx),
                    "title": session_key,
                    "source_file": "",
                    "trajectory": {
                        "conversations": [],
                        "model": "",
                        "completed": False,
                        "timestamp": "",
                    },
                    "lineageLinks": lineage_links,
                }
            )
        raise HTTPException(
            status_code=404, detail="No trajectory found for this session"
        )

    for row in trajectory_rows:
        if "conversations" in row:
            trajectory = _merge_session_artifact_metadata(
                row, ctx.hermes_home, internal_id
            )
            return JSONResponse(
                {
                    "session_id": internal_id,
                    "sessionKey": _external_session_id(internal_id, ctx),
                    "title": str(row.get("title") or session_key),
                    "source_file": "",
                    "trajectory": trajectory,
                    "lineageLinks": lineage_links,
                }
            )

    trajectory = _merge_session_artifact_metadata(
        {
            "conversations": trajectory_rows,
            "model": "",
            "completed": True,
            "timestamp": "",
        },
        ctx.hermes_home,
        internal_id,
    )
    return JSONResponse(
        {
            "session_id": internal_id,
            "sessionKey": _external_session_id(internal_id, ctx),
            "title": session_key,
            "source_file": "",
            "trajectory": trajectory,
            "lineageLinks": lineage_links,
        }
    )

@router.get("/sessions/{session_key:path}/artifacts/{artifact_path:path}/raw")
@router.get("/sessions/{session_key}/artifacts/{artifact_path:path}/raw")
async def get_session_artifact_raw(
    request: Request,
    session_key: str,
    artifact_path: str,
) -> Any:
    ctx = _require_authenticated_context(request)
    internal_id = _resolve_workspace_runtime_session_id(ctx, session_key)
    if not internal_id:
        raise HTTPException(status_code=400, detail="session_key required")
    artifact_file = _resolve_session_artifact_file(
        ctx.hermes_home,
        internal_id,
        artifact_path,
    )
    if artifact_file is None:
        raise HTTPException(status_code=404, detail="Artifact not found")
    try:
        content = artifact_file.read_bytes()
    except OSError as exc:
        raise HTTPException(status_code=404, detail="Artifact not found") from exc
    media_type, _ = mimetypes.guess_type(artifact_file.name)
    return Response(
        content=content,
        media_type=media_type or "application/octet-stream",
        headers={
            "Content-Disposition": f'inline; filename="{artifact_file.name}"',
        },
    )


@router.post("/sessions")
async def create_session(request: Request) -> Any:
    ctx = _require_authenticated_context(request)
    _enforce_rate_limit("session_create_user", request=request, ctx=ctx)
    _enforce_rate_limit(
        "session_create_tenant",
        request=request,
        ctx=ctx,
        subject=str(ctx.organization_id or ctx.workspace_id),
    )
    try:
        body = await request.json()
    except Exception:
        body = {}
    label = body.get("label") if isinstance(body, dict) else None
    normalized_label = _normalize_session_label(label)
    friendly_id = body.get("friendlyId") if isinstance(body, dict) else None
    explicit_id = body.get("id") if isinstance(body, dict) else None
    _prepare_governed_analytics_context(ctx)
    session = _create_session_record(
        ctx=ctx,
        label=normalized_label,
        session_id=(
            friendly_id
            if isinstance(friendly_id, str) and friendly_id.strip()
            else explicit_id
            if isinstance(explicit_id, str)
            else None
        ),
    )
    return {
        "ok": True,
        "sessionKey": _external_session_id(session["id"], ctx),
        "friendlyId": _external_session_id(session["id"], ctx),
        "entry": _to_semantier_session_summary(session, ctx),
        "modelApplied": False,
    }


@router.post("/sessions/{session_key:path}/branches")
async def branch_session(session_key: str, request: Request) -> Any:
    """Create an isolated session through an inclusive assistant boundary."""
    ctx = _require_authenticated_context(request)
    _enforce_rate_limit("session_create_user", request=request, ctx=ctx)
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}

    idempotency_key = str(
        request.headers.get("Idempotency-Key") or body.get("idempotencyKey") or ""
    ).strip()
    if not idempotency_key:
        raise HTTPException(status_code=400, detail="IDEMPOTENCY_KEY_REQUIRED")
    branch_point = body.get("branchPoint")
    if not isinstance(branch_point, dict):
        raise HTTPException(status_code=400, detail="BRANCH_POINT_REQUIRED")
    message_id = str(branch_point.get("messageId") or "").strip()
    sequence = branch_point.get("sequence")
    if not message_id or not isinstance(sequence, int) or sequence < 1:
        raise HTTPException(status_code=409, detail="BRANCH_POINT_NOT_PERSISTED")

    requested_session_key = unquote(str(session_key or "")).strip()
    if ":" in requested_session_key:
        requested_workspace_id = requested_session_key.split(":", 1)[0].strip()
        if requested_workspace_id and requested_workspace_id != str(ctx.workspace_id).strip():
            raise HTTPException(status_code=404, detail="SOURCE_SESSION_NOT_FOUND")
    source_id = _resolve_workspace_runtime_session_id(ctx, requested_session_key)
    if not source_id or not _session_belongs_to_workspace(source_id, ctx):
        raise HTTPException(status_code=404, detail="SOURCE_SESSION_NOT_FOUND")

    # A branch is only exposed when the P0 checkpoint is already verified. It
    # is safer to fail closed than to silently branch from transcript text.
    organization_id = str(ctx.organization_id or "").strip()
    if not organization_id:
        raise HTTPException(status_code=422, detail="BRANCH_CONTEXT_INCOMPLETE")
    try:
        from semantier.effective_context import EffectiveContextCheckpointService, EffectiveContextStore

        ecc_store = EffectiveContextStore()
        ecc_service = EffectiveContextCheckpointService(ecc_store)
        checkpoint = ecc_store.get_checkpoint_by_boundary(
            organization_id, ctx.workspace_id, session_key.strip(), message_id
        ) or ecc_store.get_checkpoint_by_boundary(
            organization_id, ctx.workspace_id, source_id, message_id
        )
        if checkpoint is None or checkpoint.verification_status != "verified":
            raise HTTPException(status_code=422, detail="BRANCH_CONTEXT_INCOMPLETE")
        if not ecc_service.verify_checkpoint(
            organization_id, ctx.workspace_id, checkpoint.checkpoint_id
        ):
            raise HTTPException(status_code=422, detail="BRANCH_CONTEXT_INCOMPLETE")
    except HTTPException:
        raise
    except Exception as exc:
        _LOGGER.warning("branch checkpoint verification failed: %s", exc)
        raise HTTPException(status_code=422, detail="BRANCH_CONTEXT_INCOMPLETE") from exc

    source = _get_workspace_session_detail_primary(ctx, source_id)
    source_title = str((source or {}).get("title") or "").strip()
    requested_title = str(body.get("title") or "").strip()
    title = _normalize_session_label(requested_title or f"⎇ {source_title or 'New Session'}")
    if title and not title.startswith("⎇ "):
        title = _normalize_session_label(f"⎇ {title}")
    request_hash = hashlib.sha256(
        json.dumps(
            {
                "workspace_id": str(ctx.workspace_id),
                "source_session_id": source_id,
                "checkpoint_id": checkpoint.checkpoint_id,
                "checkpoint_hash": checkpoint.checkpoint_hash,
                "branch_point": {
                    "message_id": message_id,
                    "sequence": sequence,
                },
                "title": title,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    child_id = f"{ctx.workspace_id}:branch_{hashlib.sha256(idempotency_key.encode('utf-8')).hexdigest()[:24]}"

    db = _workspace_session_db(ctx)
    try:
        idempotent_replay = db.get_session(child_id) is not None
        child = _branch_session_at_message(
            db,
            source_id,
            child_id,
            title=title or "⎇ New Session",
            branch_point_message_id=message_id,
            branch_point_sequence=sequence,
            idempotency_key=idempotency_key,
            request_hash=request_hash,
            authenticated_context=ctx,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except KeyError as exc:
        code = str(exc).strip("'")
        raise HTTPException(
            status_code=404 if code in {"SOURCE_SESSION_NOT_FOUND", "BRANCH_POINT_NOT_FOUND"} else 409,
            detail=code,
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    finally:
        db.close()

    try:
        child_checkpoint = ecc_service.branch_checkpoint(
            organization_id=organization_id,
            workspace_id=ctx.workspace_id,
            source_checkpoint_id=checkpoint.checkpoint_id,
            child_session_id=_external_session_id(child_id, ctx),
            boundary_message_id=message_id,
            actor_ref=str(ctx.user_id or "workspace-branch"),
            authority_context_ref=str(
                ecc_store.get_payload(
                    organization_id, ctx.workspace_id, checkpoint.authority_pins_ref
                ).get("historical_admission_proof_ref")
                or ""
            ),
        )
    except Exception as exc:
        cleanup_db = _workspace_session_db(ctx)
        try:
            cleanup_db.delete_session(child_id, sessions_dir=ctx.hermes_home / "sessions")
        finally:
            cleanup_db.close()
        _LOGGER.exception("branch checkpoint creation failed for %s", child_id)
        raise HTTPException(status_code=422, detail="BRANCH_CONTEXT_INCOMPLETE") from exc

    try:
        transport = derive_workspace_session_transport_metadata(
            platform="webchat",
            session_id=child_id,
            session_key=_external_session_id(child_id, ctx),
            platform_session_key=_external_session_id(child_id, ctx),
            origin_user_id=ctx.user_id,
        )
        create_workspace_session_log(
            ctx.hermes_home,
            session_id=child_id,
            title=title,
            source="api_server",
            platform="webchat",
            session_key=transport.platform_session_key,
            chat_id=transport.chat_id,
            thread_id=transport.thread_id,
            origin_user_id=transport.origin_user_id,
            workspace_id=ctx.workspace_id,
            workspace_owner_id=ctx.workspace_id,
        )
        replace_workspace_session_log_messages(ctx.hermes_home, child_id, [])
        trajectory_link = link_workspace_session_trajectory(
            ctx.hermes_home,
            source_session_id=source_id,
            child_session_id=child_id,
            lineage_root_id=str(child.get("lineage_root_id") or source_id),
            branch_point_message_id=message_id,
            boundary_message_sequence=sequence,
        )
    except Exception as exc:
        _LOGGER.exception("branch child log creation failed for %s", child_id)
        cleanup_db = _workspace_session_db(ctx)
        try:
            cleanup_db.delete_session(child_id, sessions_dir=ctx.hermes_home / "sessions")
        finally:
            cleanup_db.close()
        delete_workspace_session_log(ctx.hermes_home, child_id)
        raise HTTPException(status_code=500, detail="BRANCH_LINK_FAILED") from exc

    child["id"] = child_id
    child["session_id"] = child_id
    child["title"] = title
    child["parent_session_id"] = source_id
    child["branch_point_message_id"] = message_id
    child["branch_point_sequence"] = sequence
    child["lineage_root_id"] = child.get("lineage_root_id") or source_id
    return {
        "ok": True,
        "session": {
            **_to_semantier_session_summary(child, ctx),
            "parentSessionId": _external_session_id(source_id, ctx),
            "lineageRootId": _external_session_id(str(child["lineage_root_id"]), ctx),
            "branchPointMessageId": message_id,
            "branchPointSequence": sequence,
            "isBranch": True,
        },
        "continuity": {
            "trajectoryLinkCount": int(
                trajectory_link.get("trajectoryLinkCount") or 0
            ),
            "parentTrajectoryRef": str(trajectory_link.get("parentTrajectoryRef") or ""),
            "branchView": {
                "checkpointId": child_checkpoint.checkpoint_id,
                "checkpointHash": child_checkpoint.checkpoint_hash,
                "sourceSessionId": _external_session_id(source_id, ctx),
                "branchPointMessageId": message_id,
                "branchPointSequence": sequence,
                "knowledgeStateRef": child_checkpoint.knowledge_state_ref,
                "activeContractRef": child_checkpoint.active_contract_ref,
                "authorityPinsRef": child_checkpoint.authority_pins_ref,
                "transcriptSnapshotRef": child_checkpoint.transcript_snapshot_ref,
                "trajectoryRange": {
                    "fromSequence": child_checkpoint.trajectory_from_sequence,
                    "toSequence": child_checkpoint.trajectory_to_sequence,
                },
                "parentCheckpointId": child_checkpoint.parent_checkpoint_id,
                "lineageRootCheckpointId": child_checkpoint.lineage_root_checkpoint_id,
            },
            "contextSnapshotHash": child_checkpoint.checkpoint_hash,
            "checkpointId": child_checkpoint.checkpoint_id,
            "idempotentReplay": idempotent_replay,
        },
    }


@router.patch("/sessions")
async def rename_session(request: Request) -> Any:
    ctx = _require_authenticated_context(request)
    body = await request.json()
    session_key = ""
    label: str | None = None
    if isinstance(body, dict):
        session_key = str(
            body.get("sessionKey") or body.get("friendlyId") or ""
        ).strip()
        raw_label = body.get("label")
        label = _normalize_session_label(raw_label)
    if not session_key:
        raise HTTPException(status_code=400, detail="sessionKey required")
    session = _update_session_record(ctx=ctx, session_key=session_key, label=label)
    return {
        "ok": True,
        "sessionKey": _external_session_id(session["id"], ctx),
        "entry": _to_semantier_session_summary(session, ctx),
    }


@router.delete("/sessions")
async def delete_session(
    request: Request,
    sessionKey: str | None = Query(default=None),
    friendlyId: str | None = Query(default=None),
) -> Any:
    ctx = _require_authenticated_context(request)
    resolved = (sessionKey or "").strip() or (friendlyId or "").strip()
    if not resolved:
        raise HTTPException(status_code=400, detail="sessionKey required")
    _delete_session_record(ctx, resolved)
    return {"ok": True, "sessionKey": resolved}


def _extract_session_list(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        items = payload.get("items")
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
        sessions = payload.get("sessions")
        if isinstance(sessions, list):
            return [item for item in sessions if isinstance(item, dict)]
    return []


def _latest_session_run_payload(
    hermes_home: Path,
    session_id: str,
) -> tuple[dict[str, Any] | None, str]:
    runs_dir = workspace_session_dir(hermes_home, session_id) / "runs"
    if not runs_dir.exists() or not runs_dir.is_dir():
        return None, ""

    latest_payload: dict[str, Any] | None = None
    latest_source = ""
    latest_sort_key = float("-inf")

    for candidate in runs_dir.glob("*.json"):
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        updated_at = payload.get("updatedAt")
        created_at = payload.get("createdAt")
        updated_num = float(updated_at) if isinstance(updated_at, (int, float)) else 0.0
        created_num = float(created_at) if isinstance(created_at, (int, float)) else 0.0
        sort_key = max(updated_num, created_num)
        if sort_key < latest_sort_key:
            continue
        latest_sort_key = sort_key
        latest_payload = payload
        latest_source = candidate.name

    return latest_payload, latest_source


def _session_run_artifact_metadata(
    hermes_home: Path,
    session_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    runs_dir = workspace_session_dir(hermes_home, session_id) / "runs"
    if not runs_dir.exists() or not runs_dir.is_dir():
        return [], []

    artifacts: list[dict[str, Any]] = []
    artifact_keys: set[str] = set()
    lifecycle_events: list[dict[str, Any]] = []
    event_keys: set[str] = set()

    for candidate in sorted(runs_dir.glob("*.json")):
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue

        payload_artifacts = payload.get("artifacts")
        if isinstance(payload_artifacts, list):
            for item in payload_artifacts:
                if not isinstance(item, dict):
                    continue
                keys = _artifact_metadata_keys(item)
                if keys and keys & artifact_keys:
                    continue
                artifact_keys.update(keys)
                artifacts.append(item)

        payload_events = payload.get("lifecycleEvents")
        if isinstance(payload_events, list):
            for item in payload_events:
                if not isinstance(item, dict):
                    continue
                event_artifact = item.get("artifact")
                artifact_key = ""
                if isinstance(event_artifact, dict):
                    artifact_key = str(
                        event_artifact.get("artifact_id")
                        or event_artifact.get("path")
                        or event_artifact.get("filename")
                        or ""
                    ).strip()
                key = "|".join(
                    [
                        str(item.get("type") or "").strip(),
                        str(item.get("timestamp") or "").strip(),
                        artifact_key,
                    ]
                )
                if key and key in event_keys:
                    continue
                if key:
                    event_keys.add(key)
                lifecycle_events.append(item)

    return artifacts, lifecycle_events


def _session_artifact_file_metadata(
    hermes_home: Path,
    session_id: str,
    known_keys: set[str] | None = None,
) -> list[dict[str, Any]]:
    artifacts_dir = workspace_session_dir(hermes_home, session_id) / "artifacts"
    if not artifacts_dir.exists() or not artifacts_dir.is_dir():
        return []

    known = known_keys or set()
    artifacts: list[dict[str, Any]] = []
    for candidate in sorted(artifacts_dir.rglob("*")):
        if not candidate.is_file():
            continue
        try:
            resolved = candidate.resolve()
            size_bytes = candidate.stat().st_size
            relative_path = candidate.relative_to(artifacts_dir).as_posix()
            updated_at = (
                datetime.fromtimestamp(
                    candidate.stat().st_mtime,
                    tz=timezone.utc,
                )
                .isoformat()
                .replace("+00:00", "Z")
            )
        except OSError:
            continue
        key = str(resolved)
        if key in known or candidate.name in known or relative_path in known:
            continue
        known.add(key)
        artifacts.append(
            {
                "kind": "session_artifact",
                "filename": candidate.name,
                "path": key,
                "relative_path": relative_path,
                "size_bytes": size_bytes,
                "raw_url": _session_artifact_raw_url(session_id, relative_path),
                "updated_at": updated_at,
            }
        )
    return artifacts


def _artifact_metadata_key(item: dict[str, Any]) -> str:
    for field in ("artifact_id", "path", "relative_path", "filename"):
        value = str(item.get(field) or "").strip()
        if value:
            return value
    return ""


def _artifact_metadata_keys(item: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    for field in ("artifact_id", "path", "relative_path", "filename"):
        value = str(item.get(field) or "").strip()
        if value:
            keys.add(value)
            if field == "path":
                try:
                    keys.add(str(Path(value).resolve()))
                except OSError:
                    pass
    return keys


def _session_artifacts_dir(hermes_home: Path, session_id: str) -> Path:
    return workspace_session_dir(hermes_home, session_id) / "artifacts"


def _relative_session_artifact_path(
    hermes_home: Path,
    session_id: str,
    artifact_path: str,
) -> str | None:
    artifacts_dir = _session_artifacts_dir(hermes_home, session_id)
    raw_path = str(artifact_path or "").strip()
    if not raw_path:
        return None
    try:
        artifact_candidate = Path(raw_path)
        if not artifact_candidate.is_absolute():
            artifact_candidate = artifacts_dir / artifact_candidate
        resolved_dir = artifacts_dir.resolve()
        resolved_file = artifact_candidate.resolve()
        if not resolved_file.is_relative_to(resolved_dir):
            return None
        return resolved_file.relative_to(resolved_dir).as_posix()
    except (OSError, ValueError):
        return None


def _resolve_session_artifact_file(
    hermes_home: Path,
    session_id: str,
    artifact_path: str,
) -> Path | None:
    relative_path = _relative_session_artifact_path(
        hermes_home,
        session_id,
        artifact_path,
    )
    if relative_path is None:
        return None
    candidate = _session_artifacts_dir(hermes_home, session_id) / relative_path
    if not candidate.is_file():
        return None
    return candidate


def _session_artifact_raw_url(session_id: str, relative_path: str) -> str:
    return (
        "/sessions/"
        f"{quote(session_id, safe='')}"
        "/artifacts/"
        f"{quote(relative_path, safe='/')}"
        "/raw"
    )


def _enrich_session_artifact_metadata(
    hermes_home: Path,
    session_id: str,
    item: dict[str, Any],
) -> dict[str, Any]:
    relative_path = _relative_session_artifact_path(
        hermes_home,
        session_id,
        str(item.get("path") or item.get("relative_path") or ""),
    )
    if relative_path is None:
        return item
    enriched = dict(item)
    enriched.setdefault("relative_path", relative_path)
    enriched.setdefault(
        "raw_url",
        _session_artifact_raw_url(session_id, relative_path),
    )
    return enriched


def _lifecycle_event_key(item: dict[str, Any]) -> str:
    event_artifact = item.get("artifact")
    artifact_key = ""
    if isinstance(event_artifact, dict):
        artifact_key = _artifact_metadata_key(event_artifact)
    return "|".join(
        [
            str(item.get("type") or "").strip(),
            str(item.get("timestamp") or "").strip(),
            artifact_key,
        ]
    )


def _merge_session_artifact_metadata(
    trajectory: dict[str, Any],
    hermes_home: Path,
    session_id: str,
) -> dict[str, Any]:
    merged = dict(trajectory)
    artifacts: list[dict[str, Any]] = []
    artifact_keys: set[str] = set()
    for item in merged.get("artifacts", []):
        if not isinstance(item, dict):
            continue
        enriched = _enrich_session_artifact_metadata(hermes_home, session_id, item)
        keys = _artifact_metadata_keys(enriched)
        if keys and keys & artifact_keys:
            continue
        artifact_keys.update(keys)
        artifacts.append(enriched)
    artifact_keys.discard("")

    run_artifacts, run_events = _session_run_artifact_metadata(
        hermes_home,
        session_id,
    )
    for item in run_artifacts:
        keys = _artifact_metadata_keys(item)
        if keys and keys & artifact_keys:
            continue
        enriched = _enrich_session_artifact_metadata(hermes_home, session_id, item)
        artifact_keys.update(keys)
        artifact_keys.update(_artifact_metadata_keys(enriched))
        artifacts.append(enriched)

    artifacts.extend(
        _session_artifact_file_metadata(hermes_home, session_id, artifact_keys)
    )
    if artifacts:
        merged["artifacts"] = artifacts

    lifecycle_events = [
        item for item in merged.get("lifecycleEvents", []) if isinstance(item, dict)
    ]
    event_keys = {_lifecycle_event_key(item) for item in lifecycle_events}
    event_keys.discard("")
    for item in run_events:
        key = _lifecycle_event_key(item)
        if key and key in event_keys:
            continue
        if key:
            event_keys.add(key)
        lifecycle_events.append(item)
    if lifecycle_events:
        merged["lifecycleEvents"] = lifecycle_events

    return merged


def _session_run_artifact_tools(hermes_home: Path, session_id: str) -> list[str]:
    runs_dir = workspace_session_dir(hermes_home, session_id) / "runs"
    if not runs_dir.exists() or not runs_dir.is_dir():
        return []

    discovered: list[str] = []
    seen: set[str] = set()
    for candidate in sorted(runs_dir.glob("*.json")):
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        tool_calls = payload.get("toolCalls")
        if not isinstance(tool_calls, list):
            continue
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            name = str(call.get("name") or "").strip()
            key = name.lower()
            if not name or key in seen:
                continue
            seen.add(key)
            discovered.append(name)
    return discovered


def _session_run_artifact_skill_attachments(
    hermes_home: Path,
    session_id: str,
) -> list[dict[str, str]]:
    runs_dir = workspace_session_dir(hermes_home, session_id) / "runs"
    if not runs_dir.exists() or not runs_dir.is_dir():
        return []

    attachments: list[dict[str, str]] = []
    seen: set[str] = set()
    for candidate in sorted(runs_dir.glob("*.json")):
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        tool_calls = payload.get("toolCalls")
        if not isinstance(tool_calls, list):
            continue
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            marker_type = str(call.get("type") or call.get("kind") or "").strip().lower()
            if marker_type != "skill":
                continue
            name = str(call.get("name") or "").strip()
            key = name.lower()
            if not name or key in seen:
                continue
            seen.add(key)
            attachments.append({"type": "skill", "name": name})
    return attachments


def _tool_names_from_tool_specs(raw_tools: Any) -> list[str]:
    if not isinstance(raw_tools, list):
        return []
    discovered: list[str] = []
    seen: set[str] = set()
    for item in raw_tools:
        name = ""
        if isinstance(item, str):
            name = item
        elif isinstance(item, dict):
            function = item.get("function")
            if isinstance(function, dict):
                name = str(function.get("name") or "").strip()
            if not name:
                name = str(item.get("name") or item.get("tool") or "").strip()
        key = name.lower()
        if name and key not in seen:
            seen.add(key)
            discovered.append(name)
    return discovered


def _session_tools_from_messages(messages: list[dict[str, Any]]) -> list[str]:
    discovered: list[str] = []
    seen: set[str] = set()

    def add(name: str) -> None:
        normalized = str(name or "").strip()
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            discovered.append(normalized)

    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "session_meta":
            tools_payload = message.get("tools")
            if tools_payload is None:
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    try:
                        decoded = json.loads(content)
                    except json.JSONDecodeError:
                        decoded = None
                    if isinstance(decoded, dict):
                        tools_payload = decoded.get("tools")
                    elif isinstance(decoded, list):
                        tools_payload = decoded
            for name in _tool_names_from_tool_specs(tools_payload):
                add(name)
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            if isinstance(function, dict):
                add(str(function.get("name") or ""))
            else:
                add(str(call.get("name") or call.get("tool_name") or ""))
        if message.get("role") == "tool":
            add(str(message.get("tool_name") or message.get("name") or ""))

    return discovered


def _session_attachments_from_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, str]]:
    attachments: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(kind: str, name: str) -> None:
        normalized = str(name or "").strip()
        key = (kind, normalized.lower())
        if not normalized or key in seen:
            return
        seen.add(key)
        attachments.append({"type": kind, "name": normalized})

    for name in _session_tools_from_messages(messages):
        add("tool", name)

    for message in messages:
        if not isinstance(message, dict):
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            if _activity_tool_call_name(call) != "skill_view":
                continue
            function = call.get("function")
            if not isinstance(function, dict):
                continue
            arguments = function.get("arguments")
            if isinstance(arguments, str) and arguments.strip():
                try:
                    decoded_args = json.loads(arguments)
                except json.JSONDecodeError:
                    decoded_args = None
            elif isinstance(arguments, dict):
                decoded_args = arguments
            else:
                decoded_args = None
            if isinstance(decoded_args, dict):
                add("skill", str(decoded_args.get("name") or decoded_args.get("skill") or ""))

    return attachments


def _hydrate_sparse_session_log_payload(
    ctx: RequestContext,
    session_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    rewritten = dict(payload)
    messages = rewritten.get("messages")
    if not isinstance(messages, list) or not messages:
        message_payload = _get_workspace_session_messages_primary(ctx, session_id)
        if message_payload is not None:
            _, hydrated_messages = message_payload
            if hydrated_messages:
                rewritten["messages"] = hydrated_messages
                rewritten["message_count"] = len(hydrated_messages)
                messages = hydrated_messages

    tools = rewritten.get("tools")
    if not isinstance(tools, list) or not tools:
        hydrated_tools = _session_tools_from_messages(
            messages if isinstance(messages, list) else []
        )
        if hydrated_tools:
            rewritten["tools"] = hydrated_tools
        else:
            run_tools = _session_run_artifact_tools(ctx.hermes_home, session_id)
            if run_tools:
                rewritten["tools"] = run_tools
    attachments = rewritten.get("attachments")
    if not isinstance(attachments, list) or not attachments:
        hydrated_attachments = _session_attachments_from_messages(
            messages if isinstance(messages, list) else []
        )
        hydrated_attachments.extend(
            item
            for item in _session_run_artifact_skill_attachments(
                ctx.hermes_home,
                session_id,
            )
            if item not in hydrated_attachments
        )
        if hydrated_attachments:
            rewritten["attachments"] = hydrated_attachments
    return rewritten


def _activity_timestamp(value: Any) -> str:
    if isinstance(value, (int, float)):
        raw = float(value)
        if raw > 10_000_000_000:
            raw = raw / 1000.0
        return (
            datetime.fromtimestamp(raw, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )
    if isinstance(value, str) and value.strip():
        return value.strip()
    return ""


def _session_execution_boundary(
    ctx: RequestContext,
    *,
    session_id: str,
) -> ExecutionBoundary:
    session_root = workspace_session_dir(ctx.hermes_home, session_id)
    runs_root = (session_root / "runs").resolve()
    uploads_root = (session_root / "uploads").resolve()
    artifacts_root = (session_root / "artifacts").resolve()
    logs_root = (session_root / "logs").resolve()
    for path in (runs_root, uploads_root, artifacts_root, logs_root):
        path.mkdir(parents=True, exist_ok=True)
    workspace_root = ctx.hermes_home.resolve()
    return ExecutionBoundary(
        source="webapi_chat_stream",
        session_id=session_id,
        user_id=ctx.user_id,
        workspace_id=ctx.workspace_id,
        paths=BoundaryPaths(
            hermes_home=workspace_root,
            terminal_cwd=workspace_root,
            runs_root=runs_root,
            uploads_root=uploads_root,
            artifacts_root=artifacts_root,
            logs_root=logs_root,
        ),
        policy=BoundaryPolicy(
            provider_fallback_enabled=False,
            require_boundary=True,
            allowed_read_roots=(workspace_root, uploads_root, artifacts_root),
            allowed_write_roots=(runs_root, uploads_root, artifacts_root, logs_root),
        ),
        audit_metadata={
            "authority_source": "semantier_authenticated_context",
            "transport": "webapi_chat_stream",
        },
        subprocess_env={
            "SEMANTIER_WORKSPACE_RUNS_DIR": str(runs_root),
            "SEMANTIER_WORKSPACE_ARTIFACTS_DIR": str(artifacts_root),
            "WIKI_PATH": str((workspace_root / "wiki").resolve()),
        },
    )


def _skill_command_slug(value: str) -> str:
    normalized = value.strip().lower().replace("_", "-").replace(" ", "-")
    normalized = re.sub(r"[^a-z0-9-]", "", normalized)
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-")
    return normalized


def _webapi_bundled_skill_invocation_prompt(
    *,
    skill_name: str,
    skill_content: str,
    user_instruction: str,
) -> str:
    template = _WEBAPI_BUNDLED_SKILL_INVOCATION_PROMPT.read_text(encoding="utf-8")
    return template.format(
        skill_name=skill_name,
        skill_content=skill_content,
        user_instruction=user_instruction,
    ).strip()


def _parse_web_skill_slash_message(message: str) -> tuple[str, str] | None:
    if not message.startswith("/"):
        return None
    match = re.match(r"^/([^\s/]+)(.*)$", message)
    if match is None:
        return None
    command = _skill_command_slug(match.group(1))
    remainder = match.group(2) or ""
    if remainder and not remainder[0].isspace():
        return None
    if not command:
        return None
    return command, remainder[1:] if remainder.startswith(" ") else remainder


def _expand_bundled_web_skill_slash_command(
    ctx: RequestContext,
    *,
    session_id: str,
    message: str,
    command: str,
) -> tuple[str, str | None] | None:
    try:
        from agents.hermes_bundled_skill_adapter import (
            list_bundled_skill_inventory,
            load_bundled_skill_content,
        )
        from agent.skill_utils import parse_frontmatter, skill_matches_platform
        from agents.runtime_inventory import _disabled_skill_names
    except Exception as exc:
        _LOGGER.warning("Bundled slash skill expansion unavailable: %s", exc)
        return None

    disabled = _disabled_skill_names(ctx.hermes_home)
    boundary = _session_execution_boundary(ctx, session_id=session_id)
    manifest = list_bundled_skill_inventory(boundary=boundary)
    entry = next(
        (
            item
            for item in manifest.entries
            if command
            in {
                _skill_command_slug(item.name),
                _skill_command_slug(Path(item.skill_id).name),
            }
        ),
        None,
    )
    if entry is None:
        return None
    if entry.name in disabled or entry.skill_id in disabled:
        raise _slash_command_http_error(
            "slash_command_disabled_skill",
            command,
            f"The **{entry.name}** skill is installed but disabled.",
        )

    raw_content = load_bundled_skill_content(
        entry.name,
        manifest_entry=entry,
        boundary=boundary,
    )
    frontmatter, body = parse_frontmatter(raw_content)
    if not skill_matches_platform(frontmatter):
        raise _slash_command_http_error(
            "slash_command_platform_incompatible_skill",
            command,
            f"The **{entry.name}** skill is not compatible with this platform.",
        )

    parsed = _parse_web_skill_slash_message(message)
    user_instruction = parsed[1] if parsed is not None else ""
    expanded_message = _webapi_bundled_skill_invocation_prompt(
        skill_name=entry.name,
        skill_content=body.strip(),
        user_instruction=user_instruction,
    )
    _persist_session_skill_marker(
        ctx,
        session_id=session_id,
        skill_name=entry.name,
        slash_command=f"/{command}",
        source="bundled",
    )
    return expanded_message, entry.name


def _webapi_builtin_slash_commands() -> set[str]:
    try:
        from hermes_cli.commands import COMMANDS

        return {str(command).strip().lstrip("/") for command in COMMANDS}
    except Exception:
        return {
            "clear",
            "help",
            "model",
            "new",
            "save",
            "skills",
            "skin",
        }


def _webapi_plugin_command_resolver(command: str) -> Any:
    try:
        from hermes_cli.plugins import get_plugin_command_handler

        return get_plugin_command_handler(command)
    except Exception:
        return None


def _slash_command_http_error(kind: str, command: str, message: str | None = None) -> HTTPException:
    status = 400 if kind != "slash_command_expansion_failed" else 500
    detail = {
        "code": kind,
        "command": f"/{command.lstrip('/')}",
        "message": message or kind,
    }
    return HTTPException(status_code=status, detail=detail)


@contextmanager
def _webapi_skill_command_runtime_bound(
    ctx: RequestContext,
    *,
    session_id: str,
):
    resolved_home = Path(ctx.hermes_home).expanduser().resolve()
    try:
        from runtime_paths import bind_workspace_session_env

        binder = bind_workspace_session_env(resolved_home, session_id)
    except Exception:
        binder = None
    if binder is not None:
        try:
            binder.__enter__()
        except Exception:
            binder = None
        else:
            try:
                yield
            except BaseException:
                exc_info = sys.exc_info()
                suppress = binder.__exit__(*exc_info)
                if not suppress:
                    raise
            else:
                binder.__exit__(None, None, None)
            return

    previous = {
        "HERMES_HOME": os.environ.get("HERMES_HOME"),
        "SEMANTIER_WORKSPACE_ID": os.environ.get("SEMANTIER_WORKSPACE_ID"),
        "SEMANTIER_WRITE_SAFE_ROOT": os.environ.get("SEMANTIER_WRITE_SAFE_ROOT"),
        "SEMANTIER_TERMINAL_CWD": os.environ.get("SEMANTIER_TERMINAL_CWD"),
    }
    context_token = None
    reset_active_hermes_home = None
    try:
        from hermes_constants import (
            reset_active_hermes_home as _reset_active_hermes_home,
            set_active_hermes_home,
        )

        context_token = set_active_hermes_home(resolved_home)
        reset_active_hermes_home = _reset_active_hermes_home
    except Exception:
        context_token = None
        reset_active_hermes_home = None

    os.environ["HERMES_HOME"] = str(resolved_home)
    os.environ["SEMANTIER_WORKSPACE_ID"] = str(ctx.workspace_id)
    os.environ["SEMANTIER_WRITE_SAFE_ROOT"] = str(resolved_home)
    os.environ["SEMANTIER_TERMINAL_CWD"] = str(resolved_home)
    try:
        yield
    finally:
        if context_token is not None and reset_active_hermes_home is not None:
            try:
                reset_active_hermes_home(context_token)
            except Exception:
                pass
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _reset_webapi_skill_command_caches() -> None:
    try:
        import agent.skill_commands as skill_commands

        skill_commands._skill_commands = {}
        skill_commands._skill_commands_platform = None
    except Exception:
        pass
    try:
        from agent.skill_utils import _external_dirs_cache_clear

        _external_dirs_cache_clear()
    except Exception:
        pass


def _persist_session_skill_marker(
    ctx: RequestContext,
    *,
    session_id: str,
    skill_name: str,
    slash_command: str,
    source: str,
) -> None:
    run_id = f"skill-{_skill_command_slug(skill_name) or 'skill'}"
    run_path = workspace_session_dir(ctx.hermes_home, session_id) / "runs" / f"{run_id}.json"
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    payload = {
        "id": run_id,
        "runId": run_id,
        "sessionKey": session_id,
        "status": "complete",
        "createdAt": now,
        "updatedAt": now,
        "toolCalls": [
            {
                "id": f"skill:{skill_name}",
                "type": "skill",
                "kind": "skill",
                "phase": "complete",
                "name": skill_name,
                "args": {
                    "slashCommand": slash_command,
                    "source": source,
                },
                "preview": f"Loaded skill: {skill_name}",
            }
        ],
    }
    run_path.parent.mkdir(parents=True, exist_ok=True)
    run_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _expand_web_skill_slash_command(
    ctx: RequestContext,
    *,
    session_id: str,
    message: str,
) -> tuple[str, str | None]:
    if not message.startswith("/"):
        return message, None
    from agents.wiki_task_runtime import wiki_profile_actor
    command = message.split(maxsplit=1)[0]
    selected_wiki = command in ("/llm-wiki", "/research/llm-wiki")
    if selected_wiki:
        # Use only manifest-pinned raw bundled content: dynamic skill rendering
        # can run inline shell outside model-tool dispatch.
        bundled = _expand_bundled_web_skill_slash_command(
            ctx, session_id=session_id, message=message, command="llm-wiki"
        )
        if bundled is None:
            raise _slash_command_http_error("slash_command_expansion_failed", "llm-wiki", "Pinned documentary wiki skill unavailable")
        return bundled
    if wiki_profile_actor(ctx.workspace_id, session_id) is not None:
        raise _slash_command_http_error("slash_command_unsupported", command, "This documentary wiki session cannot execute dynamic slash skills")
    try:
        from agent.skill_commands import (
            DynamicSkillPayloadBuildError,
            classify_slash_command,
            expand_dynamic_skill_command,
        )
    except Exception as exc:
        _LOGGER.warning("Installed slash skill expansion unavailable: %s", exc)
        if message.startswith("/"):
            raise _slash_command_http_error(
                "slash_command_expansion_failed",
                message.split(maxsplit=1)[0].lstrip("/"),
                "Slash command expansion is unavailable.",
            )
        return message, None

    with _webapi_skill_command_runtime_bound(ctx, session_id=session_id):
        _reset_webapi_skill_command_caches()
        classification = classify_slash_command(
            message,
            built_in_commands=_webapi_builtin_slash_commands(),
            plugin_command_resolver=_webapi_plugin_command_resolver,
            platform="webapi",
        )
        if classification is None:
            return message, None
        if classification.kind in {"built_in", "plugin"}:
            raise _slash_command_http_error(
                "slash_command_unsupported",
                classification.command,
                "This WebAPI session endpoint does not dispatch slash commands.",
            )
        if classification.kind == "unknown":
            bundled_expansion = _expand_bundled_web_skill_slash_command(
                ctx,
                session_id=session_id,
                message=message,
                command=classification.command,
            )
            if bundled_expansion is not None:
                return bundled_expansion
            raise _slash_command_http_error(
                "slash_command_unknown",
                classification.command,
                classification.message,
            )
        if classification.kind in {"disabled_skill", "platform_incompatible_skill"}:
            raise _slash_command_http_error(
                f"slash_command_{classification.kind}",
                classification.command,
                classification.message,
            )
        try:
            invocation = expand_dynamic_skill_command(message, task_id=session_id)
        except DynamicSkillPayloadBuildError as exc:
            bundled_expansion = _expand_bundled_web_skill_slash_command(
                ctx,
                session_id=session_id,
                message=message,
                command=classification.command,
            )
            if bundled_expansion is not None:
                return bundled_expansion
            raise _slash_command_http_error(
                "slash_command_expansion_failed",
                classification.command,
                str(exc),
            ) from exc
        if invocation is None:
            bundled_expansion = _expand_bundled_web_skill_slash_command(
                ctx,
                session_id=session_id,
                message=message,
                command=classification.command,
            )
            if bundled_expansion is not None:
                return bundled_expansion
            raise _slash_command_http_error(
                "slash_command_unknown",
                classification.command,
                f"Unknown command `/{classification.command}`.",
            )
        slash_command = invocation.command_key
        skill_name = invocation.skill_name
        _persist_session_skill_marker(
            ctx,
            session_id=session_id,
            skill_name=skill_name,
            slash_command=slash_command,
            source="installed",
        )
        _LOGGER.info(
            "Expanded dynamic skill slash command",
            extra={
                "session_id": session_id,
                "command_key": invocation.command_key,
                "skill_name": skill_name,
            },
        )
        return invocation.expanded_message, skill_name


def _activity_tool_call_name(call: dict[str, Any]) -> str:
    function = call.get("function")
    if isinstance(function, dict):
        return str(function.get("name") or "").strip()
    return str(call.get("name") or call.get("tool_name") or "").strip()


def _session_activity_from_messages(
    messages: list[dict[str, Any]],
    *,
    session_key: str,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    final_assistant_message: dict[str, Any] | None = None
    if messages:
        first = messages[0]
        events.append(
            {
                "type": "session",
                "time": _activity_timestamp(first.get("timestamp")),
                "text": "Session started",
                "details": {
                    "event": "session",
                    "phase": "started",
                    "sessionKey": session_key,
                },
            }
        )

    for message in messages:
        if not isinstance(message, dict):
            continue
        timestamp = _activity_timestamp(message.get("timestamp"))
        role = str(message.get("role") or "").strip()
        if role == "assistant":
            content = message.get("content")
            tool_calls = message.get("tool_calls")
            if isinstance(content, str) and content.strip() and not tool_calls:
                final_assistant_message = message
            if not isinstance(tool_calls, list):
                continue
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                name = _activity_tool_call_name(call)
                if not name:
                    continue
                details: dict[str, Any] = {
                    "event": "tool",
                    "phase": "started",
                    "toolName": name,
                    "sessionKey": session_key,
                }
                call_id = str(call.get("id") or "").strip()
                if call_id:
                    details["toolCallId"] = call_id
                function = call.get("function")
                if isinstance(function, dict) and function.get("arguments"):
                    details["args"] = function.get("arguments")
                events.append(
                    {
                        "type": "tool",
                        "time": timestamp,
                        "text": name,
                        "details": details,
                    }
                )
        elif role == "tool":
            name = str(message.get("tool_name") or message.get("name") or "").strip()
            if not name:
                continue
            details = {
                "event": "tool",
                "phase": "complete",
                "toolName": name,
                "sessionKey": session_key,
            }
            call_id = str(message.get("tool_call_id") or "").strip()
            if call_id:
                details["toolCallId"] = call_id
            content = message.get("content")
            if isinstance(content, str) and content:
                details["preview"] = content[:500]
            events.append(
                {
                    "type": "tool",
                    "time": timestamp,
                    "text": name,
                    "details": details,
                }
            )

    if final_assistant_message is not None:
        events.append(
            {
                "type": "assistant_complete",
                "time": _activity_timestamp(final_assistant_message.get("timestamp")),
                "text": "Complete",
                "details": {
                    "event": "done",
                    "state": "final",
                    "errorMessage": None,
                    "sessionKey": session_key,
                },
            }
        )

    return events


def _trajectory_from_latest_run(
    hermes_home: Path,
    session_id: str,
) -> tuple[dict[str, Any] | None, str]:
    run_payload, source_file = _latest_session_run_payload(hermes_home, session_id)
    if not isinstance(run_payload, dict):
        return None, ""

    assistant_text = run_payload.get("assistantText")
    thinking_text = run_payload.get("thinkingText")
    status = str(run_payload.get("status") or "").strip().lower()

    conversations: list[dict[str, str]] = []
    if isinstance(thinking_text, str) and thinking_text.strip():
        conversations.append({"from": "assistant", "value": thinking_text.strip()})
    if isinstance(assistant_text, str) and assistant_text.strip():
        conversations.append({"from": "assistant", "value": assistant_text.strip()})

    artifacts, lifecycle_events = _session_run_artifact_metadata(
        hermes_home, session_id
    )
    artifact_keys = {
        str(
            item.get("artifact_id") or item.get("path") or item.get("filename") or ""
        ).strip()
        for item in artifacts
        if isinstance(item, dict)
    }
    artifacts.extend(
        _session_artifact_file_metadata(hermes_home, session_id, artifact_keys)
    )
    extra: dict[str, Any] = {}
    if artifacts:
        extra["artifacts"] = artifacts
    if lifecycle_events:
        extra["lifecycleEvents"] = lifecycle_events

    if not conversations and not extra:
        return None, ""

    model = run_payload.get("model")
    created_at = run_payload.get("createdAt")
    updated_at = run_payload.get("updatedAt")
    timestamp = ""
    source_ts = updated_at if isinstance(updated_at, (int, float)) else created_at
    if isinstance(source_ts, (int, float)):
        timestamp = (
            datetime.fromtimestamp(float(source_ts) / 1000.0, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )

    return (
        {
            "conversations": conversations,
            "model": str(model) if isinstance(model, str) else "",
            "completed": status == "complete",
            "timestamp": timestamp,
            **extra,
        },
        source_file,
    )


def _session_identifier(session: dict[str, Any]) -> str:
    return str(session.get("id") or session.get("session_id") or "").strip()


def _workspace_session_prefix(ctx: RequestContext) -> str:
    return f"{ctx.workspace_id}:"


def _is_legacy_public_session(session_id: str, ctx: RequestContext) -> bool:
    return ctx.workspace_id == "public" and ":" not in session_id


def _session_belongs_to_workspace(session_id: str, ctx: RequestContext) -> bool:
    if not session_id:
        return False
    return session_id.startswith(
        _workspace_session_prefix(ctx)
    ) or _is_legacy_public_session(session_id, ctx)


def _external_session_id(session_id: str, ctx: RequestContext) -> str:
    prefix = _workspace_session_prefix(ctx)
    if session_id.startswith(prefix):
        return session_id[len(prefix) :]
    return session_id


def _internal_session_id(session_key: str, ctx: RequestContext) -> str:
    resolved = session_key.strip()
    if not resolved:
        return ""
    prefix = _workspace_session_prefix(ctx)
    if resolved.startswith(prefix):
        return resolved
    return f"{prefix}{resolved}"


def _resolve_or_create_workspace_runtime_session_id(
    ctx: RequestContext,
    *,
    session_key: str,
    create_if_missing: bool,
    title: str | None = None,
) -> str:
    requested = (session_key or "").strip()
    if not requested:
        return ""
    workspace_id = str(getattr(ctx, "workspace_id", "") or "").strip()
    if not workspace_id:
        return ""

    try:
        transport_session_id = (
            requested
            if requested.startswith(f"{workspace_id}:")
            else f"{workspace_id}:{requested}"
        )
        transport = derive_workspace_session_transport_metadata(
            platform="webchat",
            session_id=transport_session_id,
            session_key=requested,
            platform_session_key=requested,
            origin_user_id=getattr(ctx, "user_id", None),
        )
        return resolve_or_create_workspace_session_id(
            ctx.hermes_home,
            workspace_id=workspace_id,
            alias=requested,
            preferred_session_id=requested,
            platform_session_key=transport.platform_session_key,
            chat_id=transport.chat_id,
            thread_id=transport.thread_id,
            origin_user_id=transport.origin_user_id,
            source="api_server",
            platform="webchat",
            title=title,
            create_if_missing=create_if_missing,
        )
    except WorkspaceSessionResolutionError:
        # Alias miss / mapping absence is handled as not-found semantics.
        return ""
    except Exception as exc:
        _LOGGER.exception(
            "workspace session resolution failed for workspace=%s key=%s",
            getattr(ctx, "workspace_id", ""),
            requested,
        )
        raise HTTPException(
            status_code=500, detail="Workspace session resolution failed"
        ) from exc


def _normalize_workspace_session_route_key(
    ctx: RequestContext, session_key: str
) -> str:
    requested = unquote(str(session_key or "").strip())
    if not requested:
        return ""
    if "/" not in requested and "\\" not in requested:
        return requested

    try:
        candidate = Path(requested).expanduser().resolve(strict=False)
        sessions_root = (Path(ctx.hermes_home) / "sessions").resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return requested

    if candidate.parent == sessions_root:
        return candidate.name
    if candidate.name == "logs" and candidate.parent.parent == sessions_root:
        return candidate.parent.name
    if (
        candidate.parent.name == "logs"
        and candidate.parent.parent.parent == sessions_root
    ):
        return candidate.parent.parent.name
    return requested


def _resolve_workspace_runtime_session_id(ctx: RequestContext, session_key: str) -> str:
    requested = _normalize_workspace_session_route_key(ctx, session_key)
    if not requested:
        return ""

    # An explicit resource lookup cannot use transport/origin aliases: that
    # producer path may select another conversation and bind the requested key.
    from agents.workspace_session_logs import _read_json, _session_segment

    home = Path(ctx.hermes_home).expanduser().resolve()
    sessions = home / "sessions"
    index_path = sessions / "sessions.json"
    index = (
        _read_json(index_path, {})
        if index_path.resolve().is_relative_to(home)
        else {}
    )
    if not isinstance(index, dict):
        index = {}
    internal = _internal_session_id(requested, ctx)
    candidates = list(dict.fromkeys((internal, requested)))
    for candidate in tuple(candidates):
        entry = index.get(candidate)
        alias_target = str(entry.get("session_id") or "").strip() if isinstance(entry, dict) else ""
        if (alias_target and _session_belongs_to_workspace(alias_target, ctx)
                and alias_target not in candidates):
            candidates.append(alias_target)
    for candidate in candidates:
        if not _session_belongs_to_workspace(candidate, ctx):
            continue
        # Existing producer path helpers mkdir, even on a miss. Construct only
        # read paths here and keep symlink targets inside the governed home.
        # Non-own colon inputs never reach log-path tail normalization.
        segment = _session_segment(home, candidate)
        logs = sessions / segment / "logs"
        filename = f"{segment}.json" if segment.startswith("session_") else f"session_{segment}.json"
        log_path = logs / filename
        jsonl_path = logs / f"{segment}.jsonl"
        legacy_resolved = ""
        if log_path.resolve().is_relative_to(home) and log_path.is_file():
            payload = _read_json(log_path, None)
            legacy_resolved = (
                str(payload.get("session_id") or "").strip()
                if isinstance(payload, dict) else ""
            ) or candidate
        elif jsonl_path.resolve().is_relative_to(home) and jsonl_path.is_file():
            legacy_resolved = candidate
        if legacy_resolved and _session_belongs_to_workspace(legacy_resolved, ctx):
            return legacy_resolved
        # Retained pre-canonical logs may live directly under sessions/.
        for name in dict.fromkeys((candidate, quote(candidate, safe="-_."))):
            if "/" in name or "\\" in name:
                continue
            legacy_path = sessions / f"session_{name}.json"
            if legacy_path.resolve().is_relative_to(home) and legacy_path.is_file():
                payload = _read_json(legacy_path, None)
                legacy_resolved = str(payload.get("session_id") or "").strip() if isinstance(payload, dict) else ""
                if legacy_resolved and _session_belongs_to_workspace(legacy_resolved, ctx):
                    return legacy_resolved
        for legacy_path in sessions.glob("session_*.json"):
            if legacy_path.resolve().is_relative_to(home) and legacy_path.is_file():
                payload = _read_json(legacy_path, None)
                if isinstance(payload, dict) and payload.get("session_id") == candidate:
                    return candidate

    db = _workspace_session_db(ctx, read_only=True)
    try:
        for candidate in candidates:
            if _session_belongs_to_workspace(candidate, ctx) and db.get_session(candidate) is not None:
                return candidate
    finally:
        db.close()
    return ""


def _ensure_runtime_session_hydrated(ctx: RequestContext, session_key: str) -> str:
    runtime_session_id = _resolve_workspace_runtime_session_id(ctx, session_key)
    if not runtime_session_id:
        runtime_session_id = _resolve_or_create_workspace_runtime_session_id(
            ctx,
            session_key=(session_key or "").strip(),
            create_if_missing=True,
        )
    if not runtime_session_id:
        raise HTTPException(status_code=404, detail="Session not found")

    db = _workspace_session_db(ctx)
    try:
        if db.get_session(runtime_session_id) is None:
            db.create_session(runtime_session_id, "api_server")
        return runtime_session_id
    finally:
        db.close()


def _rewrite_session_record_for_workspace(
    session: dict[str, Any],
    ctx: RequestContext,
) -> dict[str, Any]:
    rewritten = dict(session)
    session_id = _session_identifier(rewritten)
    external_id = _external_session_id(session_id, ctx)
    if "id" in rewritten or session_id:
        rewritten["id"] = external_id
    if "session_id" in rewritten or session_id:
        rewritten["session_id"] = external_id
    session_key = str(rewritten.get("session_key") or "").strip()
    if session_key:
        rewritten["session_key"] = _external_session_id(session_key, ctx)
    return rewritten


def _session_runtime_home() -> Path:
    return (
        Path(os.environ.get("SEMANTIER_LOCAL_STATE_DIR") or str(_LOCAL_HERMES_HOME))
        .expanduser()
        .resolve()
    )


def _workspace_session_db(ctx: RequestContext, *, read_only: bool = False):
    from hermes_state_backend import open_session_db, postgres_enabled

    # Sessions are written to the global runtime state.db (under HERMES_HOME /
    # SEMANTIER_LOCAL_STATE_DIR), not the workspace runtime home state.db.
    # Session isolation is enforced downstream by _session_belongs_to_workspace
    # (prefix filter on session IDs), so opening the shared DB here is safe.
    runtime_home = _session_runtime_home()
    path = runtime_home / "state.db"
    return open_session_db(db_path=path, read_only=read_only if postgres_enabled() else read_only and path.is_file())


def _session_source_to_platform(
    source: str | None, fallback: str | None = None
) -> str | None:
    normalized = str(source or "").strip()
    if fallback:
        return fallback
    if normalized == "api_server":
        return "webchat"
    return normalized or None


_CRON_STOP_FOOTER_RE = re.compile(
    r'To stop or manage this job, send me a new message \(e\.g\. "'
    r'(?P<command>stop reminder [^"\r\n]+)'
    r'"\)\.'
)

_SESSION_MEMORY_HEADERS: tuple[tuple[str, str], ...] = (
    ("memory", "MEMORY (your personal notes)"),
    ("user_profile", "USER PROFILE (who the user is)"),
)
_SESSION_SEARCH_MEMORY_SNAPSHOT_LIMIT = 5
_SESSION_SEARCH_MEMORY_CONTENT_LIMIT = 12_000


def _is_memory_separator_line(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and set(stripped) == {"═"}


def _extract_session_memory_snapshots(
    system_prompt: str | None,
) -> list[dict[str, str]]:
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        return []

    lines = system_prompt.splitlines()
    markers: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        for kind, prefix in _SESSION_MEMORY_HEADERS:
            if not line.startswith(prefix):
                continue
            content_start = index + 1
            if content_start < len(lines) and _is_memory_separator_line(
                lines[content_start]
            ):
                content_start += 1
            block_start = index
            if index > 0 and _is_memory_separator_line(lines[index - 1]):
                block_start = index - 1
            markers.append(
                {
                    "kind": kind,
                    "title": line.strip(),
                    "block_start": block_start,
                    "content_start": content_start,
                }
            )
            break

    snapshots: list[dict[str, str]] = []
    for marker_index, marker in enumerate(markers):
        end_index = len(lines)
        if marker_index + 1 < len(markers):
            end_index = int(markers[marker_index + 1]["block_start"])
        else:
            for index in range(int(marker["content_start"]), len(lines)):
                if lines[index].startswith("Conversation started:"):
                    end_index = index
                    break

        content_lines = lines[int(marker["content_start"]) : end_index]
        content = "\n".join(content_lines).strip()
        if content:
            snapshots.append(
                {
                    "kind": str(marker["kind"]),
                    "title": str(marker["title"]),
                    "content": content,
                    "source": "session_db.system_prompt",
                }
            )
    return snapshots


def _session_search_snapshot_content(payload: Any) -> str | None:
    if not isinstance(payload, dict) or payload.get("success") is not True:
        return None

    selected: dict[str, Any] = {}
    for key in ("mode", "query", "results", "source", "total", "truncated"):
        if key in payload:
            selected[key] = payload[key]
    if not selected:
        selected = payload

    content = json.dumps(selected, ensure_ascii=False, indent=2, sort_keys=True)
    if len(content) > _SESSION_SEARCH_MEMORY_CONTENT_LIMIT:
        content = (
            content[:_SESSION_SEARCH_MEMORY_CONTENT_LIMIT].rstrip()
            + "\n...[truncated]"
        )
    return content


def _extract_session_search_memory_snapshots(
    messages: list[dict[str, Any]],
) -> list[dict[str, str]]:
    snapshots: list[dict[str, str]] = []
    for message in messages:
        if message.get("role") != "tool" or message.get("tool_name") != "session_search":
            continue
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            continue
        snapshot_content = _session_search_snapshot_content(payload)
        if not snapshot_content:
            continue
        title_query = payload.get("query") if isinstance(payload, dict) else None
        title = "SESSION SEARCH MEMORY (retrieved during this session)"
        if isinstance(title_query, str) and title_query.strip():
            title = f"{title}: {title_query.strip()}"
        snapshots.append(
            {
                "kind": "session_search",
                "title": title,
                "content": snapshot_content,
                "source": "session_db.messages.session_search",
            }
        )
        if len(snapshots) >= _SESSION_SEARCH_MEMORY_SNAPSHOT_LIMIT:
            break
    return snapshots


def _message_fill_input_actions(content: str) -> list[dict[str, str]]:
    match = _CRON_STOP_FOOTER_RE.search(content)
    if match is None:
        return []
    command = match.group("command").strip()
    if not command:
        return []
    return [{"type": "fill_input", "label": command, "value": command}]


def _add_workspace_message_actions(normalized: dict[str, Any]) -> None:
    if normalized.get("role") != "assistant":
        return
    content = normalized.get("content")
    if not isinstance(content, str):
        return
    actions = _message_fill_input_actions(content)
    if actions:
        normalized["actions"] = actions


def _db_message_to_workspace_message(message: dict[str, Any]) -> dict[str, Any]:
    timestamp = message.get("timestamp")
    created_at = None
    if isinstance(timestamp, (int, float)):
        created_at = datetime.fromtimestamp(
            float(timestamp), tz=timezone.utc
        ).isoformat()
    normalized: dict[str, Any] = {
        "id": message.get("id"),
        "message_id": message.get("message_id"),
        "message_sequence": message.get("message_sequence"),
        "role": message.get("role") or "assistant",
        "content": message.get("content") or "",
    }
    if created_at:
        normalized["created_at"] = created_at
    if isinstance(timestamp, (int, float)):
        normalized["timestamp"] = float(timestamp)
    if isinstance(message.get("tool_calls"), list):
        normalized["tool_calls"] = message["tool_calls"]
    if isinstance(message.get("tool_call_id"), str):
        normalized["tool_call_id"] = message["tool_call_id"]
    if isinstance(message.get("tool_name"), str):
        normalized["tool_name"] = message["tool_name"]
    if isinstance(message.get("finish_reason"), str):
        normalized["finish_reason"] = message["finish_reason"]
    if isinstance(message.get("reasoning"), str):
        normalized["reasoning"] = message["reasoning"]
    if isinstance(message.get("reasoning_content"), str):
        normalized["reasoning_content"] = message["reasoning_content"]
    if isinstance(message.get("display_metadata"), dict):
        normalized["metadata"] = message["display_metadata"]
    _add_workspace_message_actions(normalized)
    return normalized


def _db_session_to_workspace_record(
    session: dict[str, Any],
    *,
    last_active: float | None = None,
) -> dict[str, Any]:
    session_id = str(session.get("id") or session.get("session_id") or "").strip()
    started_at_raw = session.get("started_at")
    started_at = (
        float(started_at_raw) if isinstance(started_at_raw, (int, float)) else 0.0
    )
    if last_active is None:
        last_active_raw = session.get("last_active")
        if isinstance(last_active_raw, (int, float)):
            last_active = float(last_active_raw)
        else:
            last_active = started_at
    title = str(session.get("title") or "").strip() or None
    display_name = str(session.get("display_name") or title or "").strip() or None
    source = str(session.get("source") or "").strip() or None
    platform = _session_source_to_platform(source)
    model = str(session.get("model") or "").strip() or None
    message_count_raw = session.get("message_count")
    message_count = int(message_count_raw) if isinstance(message_count_raw, int) else 0
    return {
        "id": session_id,
        "session_id": session_id,
        "session_key": session_id,
        "started_at": started_at,
        "last_active": last_active,
        "message_count": message_count,
        "is_active": False,
        "title": title,
        "display_name": display_name,
        "source": source,
        "model": model,
        "platform": platform,
        "parent_session_id": session.get("parent_session_id"),
        "lineage_root_id": session.get("lineage_root_id"),
        "branch_point_message_id": session.get("branch_point_message_id"),
        "branch_point_sequence": session.get("branch_point_sequence"),
        "is_branch": bool(session.get("parent_session_id")),
    }


def _list_workspace_sessions_primary(
    ctx: RequestContext,
    *,
    limit: int,
    offset: int = 0,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    db = _workspace_session_db(ctx, read_only=True)
    try:
        db_limit = max(db.session_count(), limit + offset, 1)
        for session in db.search_sessions(limit=db_limit, offset=0):
            record = _db_session_to_workspace_record(session)
            session_id = _session_identifier(record)
            if not _session_belongs_to_workspace(session_id, ctx):
                continue
            rows.append(record)
    finally:
        db.close()

    rows.sort(
        key=lambda item: (
            -float(item.get("last_active") or item.get("started_at") or 0),
            -float(item.get("started_at") or 0),
            str(item.get("session_id") or item.get("id") or ""),
        )
    )
    return rows[offset : offset + limit]


def _get_workspace_session_detail_primary(
    ctx: RequestContext,
    session_id: str,
) -> dict[str, Any] | None:
    db = _workspace_session_db(ctx, read_only=True)
    try:
        session = db.get_session(session_id)
        if session is None:
            return None
        messages = db.get_messages(session_id, limit=1, latest=True)
        started_at = session.get("started_at")
        last_active = float(started_at) if isinstance(started_at, (int, float)) else 0.0
        if messages:
            last_ts = messages[-1].get("timestamp")
            if isinstance(last_ts, (int, float)):
                last_active = float(last_ts)
        return _db_session_to_workspace_record(
            session,
            last_active=last_active,
        )
    finally:
        db.close()


def _get_workspace_session_messages_primary(
    ctx: RequestContext,
    session_id: str,
    *,
    limit: int | None = None,
    offset: int = 0,
    latest: bool = False,
) -> tuple[str, list[dict[str, Any]]] | None:
    db = _workspace_session_db(ctx, read_only=True)
    try:
        session = db.get_session(session_id)
        if session is not None:
            messages = [
                _db_message_to_workspace_message(item)
                for item in db.get_messages(
                    session_id, limit=limit, offset=offset, latest=latest
                )
            ]
            return session_id, messages
    finally:
        db.close()
    return None


def _get_workspace_session_latest_descendant_primary(
    ctx: RequestContext,
    session_id: str,
) -> tuple[str, list[str]] | None:
    db = _workspace_session_db(ctx)
    try:
        sid = db.resolve_session_id(session_id)
        if (
            not sid
            or not _session_belongs_to_workspace(sid, ctx)
            or db.get_session(sid) is None
        ):
            return None

        rows = db.list_session_records()

        children: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            child_id = str(row["id"] or "").strip()
            parent_id = str(row["parent_session_id"] or "").strip()
            if (
                child_id
                and parent_id
                and _session_belongs_to_workspace(child_id, ctx)
                and _session_belongs_to_workspace(parent_id, ctx)
            ):
                children.setdefault(parent_id, []).append(
                    {
                        "id": child_id,
                        "started_at": row["started_at"],
                    }
                )

        def started(row: dict[str, Any]) -> float:
            try:
                return float(row.get("started_at") or 0)
            except (TypeError, ValueError):
                return 0.0

        current = sid
        path = [sid]
        seen = {sid}
        while children.get(current):
            candidates = [row for row in children[current] if row["id"] not in seen]
            if not candidates:
                break
            candidates.sort(key=started, reverse=True)
            current = candidates[0]["id"]
            path.append(current)
            seen.add(current)
        return current, path
    finally:
        db.close()


def _workspace_session_records(
    payload: Any, ctx: RequestContext
) -> list[dict[str, Any]]:
    rows = _extract_session_list(payload)
    return [
        _rewrite_session_record_for_workspace(item, ctx)
        for item in rows
        if _session_belongs_to_workspace(_session_identifier(item), ctx)
    ]


def _timestamp_to_epoch_ms(value: Any) -> int:
    if value in (None, ""):
        return 0
    try:
        return int(float(value) * 1000)
    except (TypeError, ValueError):
        return 0


def _iso_timestamp_to_epoch_seconds(value: Any) -> float:
    if not isinstance(value, str):
        return 0.0
    raw = value.strip()
    if not raw:
        return 0.0
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _session_created_at_ms(session: dict[str, Any]) -> int:
    return _timestamp_to_epoch_ms(session.get("started_at"))


def _session_updated_at_ms(session: dict[str, Any]) -> int:
    last_active = session.get("last_active")
    if last_active not in (None, "", 0, 0.0, "0", "0.0"):
        return _timestamp_to_epoch_ms(last_active)
    return _session_created_at_ms(session)


def _to_semantier_session_summary(
    session: dict[str, Any],
    ctx: RequestContext,
) -> dict[str, Any]:
    session_id = _external_session_id(_session_identifier(session), ctx)
    title = str(session.get("title") or "").strip()
    label = title or session_id
    summary = {
        "key": session_id,
        "friendlyId": session_id,
        "label": label,
        "title": title or label,
        "derivedTitle": title or label,
        "updatedAt": _session_updated_at_ms(session),
        "createdAt": _session_created_at_ms(session),
    }
    optional_fields = {
        "source": session.get("source"),
        "platform": session.get("platform"),
        "model": session.get("model"),
        "isActive": session.get("is_active"),
        "messageCount": session.get("message_count"),
        "parentSessionId": (
            _external_session_id(str(session["parent_session_id"]), ctx)
            if session.get("parent_session_id")
            else None
        ),
        "lineageRootId": (
            _external_session_id(str(session["lineage_root_id"]), ctx)
            if session.get("lineage_root_id")
            else None
        ),
        "branchPointMessageId": session.get("branch_point_message_id"),
        "branchPointSequence": session.get("branch_point_sequence"),
        "isBranch": session.get("is_branch"),
    }
    for key, value in optional_fields.items():
        if value is not None:
            summary[key] = value
    return summary


def _normalize_sessions(payload: Any, ctx: RequestContext) -> list[dict[str, Any]]:
    summaries = [
        _to_semantier_session_summary(item, ctx)
        for item in _workspace_session_records(payload, ctx)
    ]
    return sorted(
        summaries, key=lambda item: (-int(item["updatedAt"]), str(item["key"]))
    )


def _merge_system_message(
    ctx: RequestContext,
    *,
    user_message: str,
    requested_system_message: str | None,
) -> str | None:
    governed_context = build_governed_runtime_context_prompt(
        user_id=ctx.user_id,
        workspace_id=ctx.workspace_id,
        user_message=user_message,
    )
    parts = [
        requested_system_message,
        governed_context,
    ]
    return "\n\n".join(part.strip() for part in parts if part and part.strip()) or None


def _cron_error_followup_context_system_message(
    ctx: RequestContext,
    *,
    session_id: str,
    user_message: str,
) -> str | None:
    normalized_message = " ".join((user_message or "").strip().lower().split())
    if normalized_message not in _AFFIRMATIVE_FOLLOWUP_MESSAGES:
        return None

    db = _workspace_session_db(ctx)
    try:
        persisted = db.get_messages(session_id)
    finally:
        db.close()

    if persisted and not any(item.get("role") == "user" for item in persisted):
        for item in reversed(persisted):
            if item.get("role") != "assistant":
                continue
            content = str(item.get("content") or "").strip()
            lowered = content.lower()
            if "cronjob response:" in lowered and (
                "failed" in lowered or "error" in lowered
            ):
                return _CRON_ERROR_FOLLOWUP_CONTEXT_PROMPT.format(
                    latest_cron_output=content
                )

    return None


def _short_affirmative_followup_context_system_message(
    ctx: RequestContext,
    *,
    session_id: str,
    user_message: str,
) -> str | None:
    normalized_message = " ".join((user_message or "").strip().lower().split())
    if normalized_message not in _AFFIRMATIVE_FOLLOWUP_MESSAGES:
        return None

    db = _workspace_session_db(ctx)
    try:
        persisted = db.get_messages(session_id)
    finally:
        db.close()

    for item in reversed(persisted or []):
        if item.get("role") != "assistant":
            continue
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        return _SHORT_AFFIRMATIVE_FOLLOWUP_CONTEXT_PROMPT.format(
            latest_assistant_message=content
        )
    return None


def _with_cron_error_followup_context(
    ctx: RequestContext,
    *,
    session_id: str,
    user_message: str,
    system_message: str | None,
) -> str | None:
    cron_context = _cron_error_followup_context_system_message(
        ctx,
        session_id=session_id,
        user_message=user_message,
    )
    affirmative_context = _short_affirmative_followup_context_system_message(
        ctx,
        session_id=session_id,
        user_message=user_message,
    )
    parts = [system_message, cron_context, affirmative_context]
    return "\n\n".join(part.strip() for part in parts if part and part.strip()) or None


def _normalize_workspace_session_route_key(
    ctx: RequestContext, session_key: str
) -> str:
    requested = unquote(str(session_key or "").strip())
    if not requested:
        return ""
    if "/" not in requested and "\\" not in requested:
        return requested

    try:
        candidate = Path(requested).expanduser().resolve(strict=False)
        sessions_root = (Path(ctx.hermes_home) / "sessions").resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return requested

    if candidate.parent == sessions_root:
        return candidate.name
    if candidate.name == "logs" and candidate.parent.parent == sessions_root:
        return candidate.parent.name
    if (
        candidate.parent.name == "logs"
        and candidate.parent.parent.parent == sessions_root
    ):
        return candidate.parent.parent.name
    return requested


def _create_session_record(
    ctx: RequestContext,
    *,
    label: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    normalized_label = _normalize_session_label(label)
    requested_session_key = (
        session_id or ""
    ).strip() or f"session_{uuid.uuid4().hex[:12]}"
    resolved_session_id = _resolve_or_create_workspace_runtime_session_id(
        ctx,
        session_key=requested_session_key,
        create_if_missing=True,
        title=normalized_label,
    )
    if not resolved_session_id:
        raise HTTPException(status_code=500, detail="Failed to create session")

    resolved_label = normalized_label
    payload_sandbox_key = sandbox_key_for_request(
        SandboxScope(
            workspace_id=ctx.workspace_id,
            lane="interactive_session",
            scope_id=resolved_session_id,
            adapter_key=None,
            network_enabled=False,
        )
    )
    transport = derive_workspace_session_transport_metadata(
        platform="webchat",
        session_id=resolved_session_id,
        session_key=resolved_session_id,
        platform_session_key=resolved_session_id,
        origin_user_id=ctx.user_id,
    )
    payload = create_workspace_session_log(
        ctx.hermes_home,
        session_id=resolved_session_id,
        title=resolved_label,
        source="api_server",
        platform="webchat",
        session_key=transport.platform_session_key,
        chat_id=transport.chat_id,
        thread_id=transport.thread_id,
        origin_user_id=transport.origin_user_id,
        workspace_id=ctx.workspace_id,
        sandbox_key=payload_sandbox_key,
        workspace_owner_id=ctx.workspace_id,
    )
    db = _workspace_session_db(ctx)
    try:
        db.create_session(resolved_session_id, "api_server")
        if resolved_label:
            resolved_label = _set_unique_session_title(db, resolved_session_id, resolved_label)
    finally:
        db.close()
    if resolved_label != normalized_label:
        payload = update_workspace_session_title(
            ctx.hermes_home, resolved_session_id, resolved_label
        )
    return {
        "id": resolved_session_id,
        "session_id": resolved_session_id,
        "title": payload.get("title"),
        "display_name": payload.get("display_name"),
        "source": payload.get("source"),
        "started_at": _iso_timestamp_to_epoch_seconds(payload.get("session_start")),
        "last_active": _iso_timestamp_to_epoch_seconds(payload.get("last_updated")),
        "message_count": payload.get("message_count") or 0,
    }


def _update_session_record(
    ctx: RequestContext,
    *,
    session_key: str,
    label: str | None = None,
) -> dict[str, Any]:
    resolved = _resolve_workspace_runtime_session_id(ctx, session_key)
    if not resolved:
        raise HTTPException(status_code=404, detail="Session not found")
    normalized_label = _normalize_session_label(label)
    payload = update_workspace_session_title(
        ctx.hermes_home, resolved, normalized_label
    )
    if payload is None:
        raise HTTPException(status_code=404, detail="Session not found")
    db = _workspace_session_db(ctx)
    try:
        db.create_session(resolved, str(payload.get("source") or "api_server"))
        resolved_label = _set_unique_session_title(db, resolved, normalized_label)
        if resolved_label != normalized_label:
            payload = update_workspace_session_title(ctx.hermes_home, resolved, resolved_label)
    finally:
        db.close()
    return {
        "id": resolved,
        "session_id": resolved,
        "title": payload.get("title"),
        "display_name": payload.get("display_name"),
        "source": payload.get("source"),
        "started_at": _iso_timestamp_to_epoch_seconds(payload.get("session_start")),
        "last_active": _iso_timestamp_to_epoch_seconds(payload.get("last_updated")),
        "message_count": payload.get("message_count") or 0,
    }


def _delete_session_record(ctx: RequestContext, session_key: str) -> None:
    resolved = _resolve_workspace_runtime_session_id(ctx, session_key)
    if not resolved:
        raise HTTPException(status_code=404, detail="Session not found")

    artifact_deleted = delete_workspace_session_log(ctx.hermes_home, resolved, exact=True)
    db_deleted = False
    db = _workspace_session_db(ctx)
    try:
        sessions_dir = ctx.hermes_home / "sessions"
        runtime_session_id = db.resolve_session_id(resolved)
        if runtime_session_id == resolved:
            db_deleted = bool(db.delete_session(resolved, sessions_dir=sessions_dir))
    finally:
        db.close()

    if artifact_deleted is None and not db_deleted:
        raise HTTPException(status_code=404, detail="Session not found")


@router.api_route(
    "/api/sessions",
    methods=["GET", "POST", "PATCH", "DELETE"],
)
async def api_sessions(
    request: Request,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    sessionKey: str | None = Query(default=None),
    friendlyId: str | None = Query(default=None),
) -> Any:
    ctx = await run_in_threadpool(_require_authenticated_context, request)
    if request.method == "GET":
        def read_in_worker():
            payload = {
                "sessions": _list_workspace_sessions_primary(
                    ctx, limit=limit, offset=offset
                )
            }
            sessions = _workspace_session_records(payload, ctx)
            return {
                "sessions": sessions,
                "total": len(sessions),
                "limit": limit,
                "offset": offset,
            }
        return await run_in_threadpool(read_in_worker)
    if request.method == "POST":
        return await create_session(request)
    if request.method == "PATCH":
        return await rename_session(request)
    resolved = (sessionKey or "").strip() or (friendlyId or "").strip()
    if not resolved:
        raise HTTPException(status_code=400, detail="sessionKey required")
    _delete_session_record(ctx, resolved)
    return {"ok": True, "sessionKey": resolved}


@router.api_route(
    "/api/sessions/search",
    methods=["GET"],
)
async def api_sessions_search(
    request: Request,
    q: str = Query(default=""),
    limit: int = Query(default=20, ge=1, le=500),
) -> Any:
    ctx = _require_authenticated_context(request)
    _ensure_repo_hermes_agent_on_path()
    from hermes_cli.web_routers.sessions import search_sessions
    from hermes_constants import set_hermes_home_override,reset_hermes_home_override

    token=set_hermes_home_override(str(_session_runtime_home()))
    try:
        payload = await search_sessions(q=q, limit=limit)
    finally:
        reset_hermes_home_override(token)
    # Scope results to the authenticated workspace. Workspace-scoped session
    # ids carry a `{workspace_id}:` prefix; older unscoped ids are dropped
    # rather than leaked across tenants.
    prefix = _workspace_session_prefix(ctx)
    results = payload.get("results") if isinstance(payload, dict) else None
    if isinstance(results, list):
        scoped: list[dict[str, Any]] = []
        for row in results:
            if not isinstance(row, dict):
                continue
            sid = str(row.get("session_id") or "")
            if not sid.startswith(prefix):
                continue
            rewritten = dict(row)
            rewritten["session_id"] = _external_session_id(sid, ctx)
            scoped.append(rewritten)
        return {"results": scoped}
    return {"results": []}


@router.api_route(
    "/api/sessions/{session_id}",
    methods=["GET", "DELETE"],
)
async def api_session_detail(session_id: str, request: Request) -> Any:
    def execute_in_worker():
        ctx = _require_authenticated_context(request)
        if request.method == "DELETE":
            _delete_session_record(ctx, session_id)
            return {"ok": True}
        detail = _get_workspace_session_detail_primary(
            ctx,
            _resolve_workspace_runtime_session_id(ctx, session_id),
        )
        if not isinstance(detail, dict):
            raise HTTPException(status_code=404, detail="Session not found")
        return _rewrite_session_record_for_workspace(detail, ctx)
    return await run_in_threadpool(execute_in_worker)


@router.get("/api/sessions/{session_id}/lineage")
async def api_session_lineage(session_id: str, request: Request) -> Any:
    ctx = _require_authenticated_context(request)
    runtime_session_id = _resolve_workspace_runtime_session_id(ctx, session_id)
    db = _workspace_session_db(ctx)
    try:
        lineage = build_session_lineage(
            db,
            runtime_session_id,
            belongs_to_workspace=lambda candidate: _session_belongs_to_workspace(candidate, ctx),
        )
    finally:
        db.close()
    if lineage is None:
        raise HTTPException(status_code=404, detail="Session not found")

    sessions = []
    for item in lineage["sessions"]:
        external_id = _external_session_id(str(item["id"]), ctx)
        parent_id = item.get("parent_session_id")
        sessions.append(
            {
                "key": external_id,
                "title": item.get("title") or external_id,
                "parentSessionId": _external_session_id(str(parent_id), ctx) if parent_id else None,
                "createdAt": _timestamp_to_epoch_ms(item.get("started_at")),
                "branchPointMessageId": item.get("branch_point_message_id"),
                "branchPointSequence": item.get("branch_point_sequence"),
                "isCurrent": bool(item.get("is_current")),
            }
        )
    return {
        "currentSessionId": _external_session_id(lineage["current_session_id"], ctx),
        "rootSessionId": _external_session_id(lineage["root_session_id"], ctx),
        "sessions": sessions,
    }


@router.get("/api/sessions/{session_id}/memory-snapshot")
async def api_session_memory_snapshot(session_id: str, request: Request) -> Any:
    ctx = _require_authenticated_context(request)
    runtime_session_id = _resolve_workspace_runtime_session_id(ctx, session_id)
    if not runtime_session_id:
        raise HTTPException(status_code=404, detail="Session not found")

    db = _workspace_session_db(ctx)
    try:
        session = db.get_session(runtime_session_id)
        messages = db.get_messages(runtime_session_id)
    finally:
        db.close()
    if not isinstance(session, dict):
        raise HTTPException(status_code=404, detail="Session not found")

    snapshots = _extract_session_memory_snapshots(session.get("system_prompt"))
    snapshots.extend(_extract_session_search_memory_snapshots(messages))
    return {
        "session_id": _external_session_id(runtime_session_id, ctx),
        "sessionKey": _external_session_id(runtime_session_id, ctx),
        "snapshots": snapshots,
    }


@router.get("/api/sessions/{session_id:path}/log")
@router.get("/api/sessions/{session_id}/log")
async def api_session_log(session_id: str, request: Request) -> Any:
    ctx = _require_authenticated_context(request)
    runtime_session_id = _resolve_workspace_runtime_session_id(ctx, session_id)
    payload = _compat_attr("get_workspace_session_log_payload")(
        ctx.hermes_home,
        runtime_session_id,
    )
    if not isinstance(payload, dict):
        raise HTTPException(status_code=404, detail="Session not found")

    rewritten = _hydrate_sparse_session_log_payload(ctx, runtime_session_id, payload)
    rewritten["session_id"] = _external_session_id(runtime_session_id, ctx)
    rewritten["session_key"] = _external_session_id(runtime_session_id, ctx)
    return rewritten


@router.get("/api/sessions/{session_id:path}/activity")
@router.get("/api/sessions/{session_id}/activity")
async def api_session_activity(session_id: str, request: Request) -> Any:
    return _api_session_activity_payload(session_id, request)


@router.get("/api/sessions/{session_id:path}/effective-context")
@router.get("/api/sessions/{session_id}/effective-context")
async def api_session_effective_context(session_id: str, request: Request) -> Any:
    """Expose a read-only, tenant-scoped projection of the ECC ledger."""
    ctx = _require_authenticated_context(request)
    runtime_session_id = _resolve_workspace_runtime_session_id(ctx, session_id)
    if not runtime_session_id:
        raise HTTPException(status_code=404, detail="Session not found")

    organization_id = str(ctx.organization_id or "").strip()
    workspace_id = str(ctx.workspace_id or "").strip()
    if not organization_id or not workspace_id:
        raise HTTPException(status_code=404, detail="Effective context unavailable")

    service = EffectiveContextCheckpointService()
    all_events = service.store.list_events(
        organization_id, workspace_id, runtime_session_id
    )
    state_effect_filter = request.query_params.get("state_effect")
    source_type_filter = request.query_params.get("source_type")
    actor_filter = request.query_params.get("actor")
    try:
        sequence_from = int(request.query_params["sequence_from"])
    except (KeyError, TypeError, ValueError):
        sequence_from = None
    try:
        sequence_to = int(request.query_params["sequence_to"])
    except (KeyError, TypeError, ValueError):
        sequence_to = None
    events = _filter_effective_context_events(
        all_events,
        state_effect=state_effect_filter,
        source_type=source_type_filter,
        actor=actor_filter,
        sequence_from=sequence_from,
        sequence_to=sequence_to,
    )
    checkpoints = service.store.list_checkpoints(
        organization_id, workspace_id, runtime_session_id
    )
    current = checkpoints[-1] if checkpoints else None
    verified = False
    verification_reason = "NO_CHECKPOINT"
    restored: dict[str, Any] | None = None
    if current is not None:
        try:
            verified = service.verify_checkpoint(
                organization_id, workspace_id, current.checkpoint_id
            )
            verification_reason = "VERIFIED" if verified else "CHECKPOINT_OR_CHAIN_INVALID"
            if verified:
                restored = service.restore_read_only(
                    organization_id, workspace_id, current.checkpoint_id
                )
        except (KeyError, ValueError) as exc:
            verification_reason = str(exc)

    closure_gaps: list[str] = []
    if current is not None:
        for ref_name in (
            "knowledge_state_ref",
            "active_contract_ref",
            "authority_pins_ref",
            "transcript_snapshot_ref",
        ):
            ref = getattr(current, ref_name)
            try:
                service.store.get_payload(organization_id, workspace_id, ref)
            except KeyError:
                closure_gaps.append(ref)

    lineage = [
        {
            "checkpoint_id": checkpoint.checkpoint_id,
            "parent_checkpoint_id": checkpoint.parent_checkpoint_id,
            "lineage_root_checkpoint_id": checkpoint.lineage_root_checkpoint_id,
            "boundary_event_sequence": checkpoint.boundary_event_sequence,
            "verification_status": checkpoint.verification_status,
        }
        for checkpoint in checkpoints
    ]
    diffs: list[dict[str, Any]] = []
    for parent, child in zip(checkpoints, checkpoints[1:]):
        parent_data = parent.model_dump(mode="json")
        child_data = child.model_dump(mode="json")
        changed_fields = sorted(
            key for key in child_data if parent_data.get(key) != child_data.get(key)
        )
        diffs.append(
            {
                "from_checkpoint_id": parent.checkpoint_id,
                "to_checkpoint_id": child.checkpoint_id,
                "comparison_kind": "parent_child",
                "changed_fields": changed_fields,
                "knowledge_state_changed": parent.knowledge_state_ref != child.knowledge_state_ref,
                "active_contract_changed": parent.active_contract_ref != child.active_contract_ref,
                "authority_pins_changed": parent.authority_pins_ref != child.authority_pins_ref,
                "compaction": {
                    "started": any(
                        event.event_type.value == "CONTEXT_COMPACTION_STARTED"
                        and parent.boundary_event_sequence < event.sequence <= child.boundary_event_sequence
                        for event in all_events
                    ),
                    "committed": any(
                        event.event_type.value == "CONTEXT_COMPACTION_COMMITTED"
                        and parent.boundary_event_sequence < event.sequence <= child.boundary_event_sequence
                        for event in all_events
                    ),
                    "event_ids": [
                        event.event_id
                        for event in all_events
                        if event.event_type.value in {
                            "CONTEXT_COMPACTION_STARTED",
                            "CONTEXT_COMPACTION_COMMITTED",
                        }
                        and parent.boundary_event_sequence < event.sequence <= child.boundary_event_sequence
                    ],
                },
                "risk_flags": sorted(
                    flag
                    for flag, changed in {
                        "active_contract_drift": parent.active_contract_ref != child.active_contract_ref,
                        "authority_pin_change": parent.authority_pins_ref != child.authority_pins_ref,
                        "knowledge_state_drift": parent.knowledge_state_ref != child.knowledge_state_ref,
                    }.items()
                    if changed
                ),
            }
        )
    eval_result = None
    if isinstance(restored, dict):
        knowledge_state = restored.get("knowledge_state")
        if isinstance(knowledge_state, dict):
            eval_result = knowledge_state.get("eval")
    eval_result = eval_result or service.store.get_latest_eval_result(
        organization_id, workspace_id, runtime_session_id
    )
    metric_names = (
        "restoration_fidelity",
        "branch_equivalence",
        "active_contract_retention",
        "knowledge_closure_recall",
        "authority_pin_fidelity",
        "compaction_fidelity",
        "memory_utility",
        "contamination",
    )
    eval_results = service.store.list_eval_results(
        organization_id, workspace_id, runtime_session_id
    )
    reference_result = next(
        (item for item in reversed(eval_results) if item.get("mode") == "reference_checkpoint"),
        None,
    )
    agent_result = next(
        (item for item in reversed(eval_results) if item.get("mode") == "persistent"),
        None,
    )
    eval_comparison = None
    if reference_result or agent_result:
        reference_metrics = reference_result.get("metrics", {}) if reference_result else {}
        agent_metrics = agent_result.get("metrics", {}) if agent_result else {}
        eval_comparison = {
            "agent_result_id": agent_result.get("result_id") if agent_result else None,
            "reference_result_id": reference_result.get("result_id") if reference_result else None,
            "agent_checkpoint_id": agent_result.get("checkpoint_id") if agent_result else None,
            "reference_checkpoint_id": reference_result.get("checkpoint_id") if reference_result else None,
            "metric_deltas": {
                name: agent_metrics.get(name) - reference_metrics.get(name)
                if isinstance(agent_metrics.get(name), (int, float))
                and isinstance(reference_metrics.get(name), (int, float))
                else None
                for name in metric_names
            },
            "first_failure_point": agent_result.get("first_failed_round") if agent_result else None,
        }
    metric_names = (
        "restoration_fidelity",
        "branch_equivalence",
        "active_contract_retention",
        "knowledge_closure_recall",
        "authority_pin_fidelity",
        "compaction_fidelity",
        "memory_utility",
        "contamination",
    )
    if isinstance(eval_result, dict):
        eval_result = {
            **eval_result,
            "metrics": {
                name: eval_result.get("metrics", {}).get(name)
                for name in metric_names
            },
            "first_failure_point": eval_result.get("first_failed_round"),
        }
    else:
        eval_result = {
            "status": "unavailable",
            "metrics": {name: None for name in metric_names},
            "first_failure_point": None,
        }

    def _event_payload(event: Any) -> dict[str, Any]:
        return {
            "event_id": event.event_id,
            "sequence": event.sequence,
            "occurred_at": event.occurred_at,
            "actor_ref": event.actor_ref,
            "event_type": event.event_type.value,
            "state_effect": event.state_effect.value,
            "input_refs": [ref.model_dump(mode="json") for ref in event.input_refs],
            "output_refs": [ref.model_dump(mode="json") for ref in event.output_refs],
            "cause_event_ids": event.cause_event_ids,
            "authority_context_ref": event.authority_context_ref,
            "content_hash": event.content_hash,
            "prior_event_hash": event.prior_event_hash,
            "type": event.event_type.value,
            "time": event.occurred_at,
            "text": event.event_type.value,
            "details": {
                "sequence": event.sequence,
                "state_effect": event.state_effect.value,
                "source": "semantier.effective_context",
                "actor_ref": event.actor_ref,
                "event_type": event.event_type.value,
            },
        }

    evidence = {
        "schema": "effective_context_observatory_evidence.v1",
        "session_id": _external_session_id(runtime_session_id, ctx),
        "checkpoint_id": current.checkpoint_id if current else None,
        "lineage": lineage,
        "diffs": diffs,
        "event_count": len(all_events),
        "eval": eval_result,
        "eval_comparison": eval_comparison,
    }
    evidence["bundle_hash"] = hashlib.sha256(
        json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    return {
        "source": "semantier.effective_context",
        "tenant": {"organization_id": organization_id, "workspace_id": workspace_id},
        "session_id": _external_session_id(runtime_session_id, ctx),
        "boundary_cursor": {
            "event_sequence": current.boundary_event_sequence if current else len(all_events),
            "checkpoint_id": current.checkpoint_id if current else None,
        },
        "checkpoint": current.model_dump(mode="json") if current else None,
        "checkpoints": [checkpoint.model_dump(mode="json") for checkpoint in checkpoints],
        "verification": {
            "status": "verified" if verified else (current.verification_status if current else "degraded"),
            "reason": verification_reason,
        },
        "restored": restored,
        "closure": {"gaps": closure_gaps, "closed": not closure_gaps and current is not None},
        "branchability": {
            "status": "branchable" if verified and not closure_gaps else "degraded",
            "remediation": None if verified and not closure_gaps else "verify_checkpoint_and_restore_missing_payloads",
        },
        "lineage": lineage,
        "diffs": diffs,
        "filters": {
            "state_effect": state_effect_filter,
            "source_type": source_type_filter,
            "actor": actor_filter,
            "sequence_from": sequence_from,
            "sequence_to": sequence_to,
        },
        "events": [_event_payload(event) for event in events],
        "eval": eval_result,
        "eval_comparison": eval_comparison,
        "evidence": evidence,
    }


def _api_session_activity_payload(session_id: str, request: Request) -> dict[str, Any]:
    ctx = _require_authenticated_context(request)
    runtime_session_id = _resolve_workspace_runtime_session_id(ctx, session_id)
    payload = _get_workspace_session_messages_primary(ctx, runtime_session_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Session not found")
    response_session_id, messages = payload
    external_id = _external_session_id(response_session_id, ctx)
    return {
        "session_id": external_id,
        "sessionKey": external_id,
        "events": _session_activity_from_messages(
            messages,
            session_key=external_id,
        ),
    }


@router.api_route(
    "/api/sessions/{session_id}/latest-descendant",
    methods=["GET"],
)
async def api_session_latest_descendant(session_id: str, request: Request) -> Any:
    ctx = _require_authenticated_context(request)
    # Resolve through the workspace-scoped resolver first; if the session
    # does not belong to this workspace, fail closed with 404 rather than
    # consult the cross-workspace runtime store.
    runtime_session_id = _resolve_workspace_runtime_session_id(ctx, session_id)
    prefix = _workspace_session_prefix(ctx)
    if not runtime_session_id or not runtime_session_id.startswith(prefix):
        raise HTTPException(status_code=404, detail="Session not found")
    resolved = _get_workspace_session_latest_descendant_primary(ctx, runtime_session_id)
    if resolved is None:
        raise HTTPException(status_code=404, detail="Session not found")
    latest, path = resolved
    return {
        "requested_session_id": _external_session_id(
            path[0] if path else runtime_session_id, ctx
        ),
        "session_id": _external_session_id(latest, ctx),
        "path": [_external_session_id(item, ctx) for item in path],
        "changed": bool(path and latest != path[0]),
    }


@router.api_route(
    "/api/sessions/{session_id}/messages",
    methods=["GET"],
)
async def api_session_messages(
    session_id: str,
    request: Request,
    limit: int | None = Query(default=None, ge=0),
    offset: int = Query(default=0, ge=0),
    order: str | None = Query(default=None),
) -> Any:
    if order not in (None, "oldest", "latest"):
        raise HTTPException(
            status_code=400,
            detail="order must be one of: oldest, latest",
        )
    default_page = limit is None
    page_limit = 500 if limit is None else min(limit, 500)
    latest_page = order == "latest" or (order is None and default_page)
    def context_in_worker():
        ctx = _require_authenticated_context(request)
        return ctx, _resolve_workspace_runtime_session_id(ctx, session_id)
    ctx, runtime_session_id = await run_in_threadpool(context_in_worker)
    from agents.hermes_embedded_api import recover_authenticated_api_session
    try:
        await recover_authenticated_api_session(ctx, runtime_session_id)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    payload = await run_in_threadpool(_get_workspace_session_messages_primary,
        ctx,
        runtime_session_id,
        limit=page_limit,
        offset=offset,
        latest=latest_page,
    )
    if payload is None:
        raise HTTPException(status_code=404, detail="Session not found")
    response_session_id, messages = payload
    rewritten = {
        "session_id": response_session_id,
        "messages": messages,
        "pagination": {
            "limit": page_limit,
            "offset": offset,
            "order": order or ("latest" if default_page else "oldest"),
            "returned": len(messages),
        },
    }
    rewritten["session_id"] = _external_session_id(
        response_session_id,
        ctx,
    )
    return rewritten


def _run_explicit_chat_tool(
    ctx: RequestContext, session_id: str, message: str, action: dict
) -> dict:
    """An authenticated explicit action invokes the registered tool without a model turn."""
    if not ctx.authenticated:
        raise HTTPException(401, "Authentication required")
    if not isinstance(action, dict) or action.get("name") != "screen_resumes":
        raise HTTPException(400, "Explicit chat tool unavailable")
    args = action.get("arguments")
    if not isinstance(args, dict) or args.get("mode") != "lineage_v1":
        raise HTTPException(400, "Explicit screening requires lineage_v1")
    runtime_id = _resolve_workspace_runtime_session_id(ctx, session_id)
    if not runtime_id:
        raise HTTPException(404, "Session not found")
    from agents.launcher import _load_db_workspace_config

    config = _load_db_workspace_config(ctx.workspace_id)
    plugins = config.get("plugins") or {}
    toolsets = (config.get("platform_toolsets") or {}).get("api_server") or []
    if (
        "auto_resume_screening" not in (plugins.get("enabled") or [])
        or "auto_resume_screening" in (plugins.get("disabled") or [])
        or "auto_resume_screening" not in toolsets
    ):
        raise HTTPException(503, "Screening tool not enabled for this workspace")
    _ensure_repo_hermes_agent_on_path()
    from runtime_paths import bind_workspace_session_env
    from gateway.execution_boundary import bind_execution_boundary
    from gateway.session_context import set_session_vars, clear_session_vars
    from model_tools import handle_function_call
    from tools.registry import registry

    if registry.get_entry("screen_resumes") is None:
        raise HTTPException(503, "Registered screening tool unavailable")
    call_id = "explicit-" + uuid.uuid4().hex
    boundary = _session_execution_boundary(ctx, session_id=runtime_id)
    with (
        bind_execution_boundary(boundary),
        bind_workspace_session_env(ctx.hermes_home, runtime_id),
    ):
        tokens = set_session_vars(
            platform="api_server",
            source="webchat",
            user_id=ctx.user_id,
            workspace_owner_id=ctx.workspace_id,
            session_id=runtime_id,
            session_key=runtime_id,
            hermes_home=str(ctx.hermes_home),
            async_delivery=False,
        )
        try:
            output = handle_function_call(
                "screen_resumes",
                args,
                session_id=runtime_id,
                task_id=runtime_id,
                tool_call_id=call_id,
                user_task=message,
                enabled_tools=["screen_resumes"],
                enabled_toolsets=["auto_resume_screening"],
            )
        finally:
            clear_session_vars(tokens)
    db = _workspace_session_db(ctx)
    try:
        db.append_messages_batch(
            runtime_id,
            [
                {"role": "user", "content": message},
                {
                    "role": "tool",
                    "tool_name": "screen_resumes",
                    "tool_call_id": call_id,
                    "content": output,
                    "display_kind": "explicit_tool_result",
                    "display_metadata": {
                        "invocation": "user_explicit",
                        "model_ran": False,
                    },
                },
            ],
        )
    finally:
        db.close()
    return {
        "ok": True,
        "session_id": _external_session_id(runtime_id, ctx),
        "tool_result": json.loads(output),
        "model_ran": False,
    }

@router.post("/api/sessions/{session_id}/chat")
async def api_session_chat(session_id: str, request: Request) -> Any:
    ctx = _require_authenticated_context(request)
    _enforce_rate_limit("chat_user", request=request, ctx=ctx)
    _enforce_rate_limit(
        "chat_tenant",
        request=request,
        ctx=ctx,
        subject=str(ctx.organization_id or ctx.workspace_id),
    )
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}

    workbench_context_json = _resolve_workbench_context_hint(
        ctx, body.get("knowledge_workbench_context")
    )

    raw_message = body.get("message")
    message = raw_message.strip() if isinstance(raw_message, str) else ""
    if not message:
        raise HTTPException(status_code=400, detail="message required")

    from services.learning_review_commands import maybe_learning_review_command
    review = maybe_learning_review_command(ctx, message, body.get("requestIdentity"))
    if review is not None:
        return review

    try:
        request_metadata = durable_request_metadata(body.get("requestIdentity"), raw_message, body.get("requestHash"))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if "tool_action" in body:
        return _run_explicit_chat_tool(ctx, session_id, message, body["tool_action"])

    raw_model = body.get("model")
    model = (
        raw_model.strip() if isinstance(raw_model, str) and raw_model.strip() else None
    )
    runtime_expert_result = await _maybe_run_tender_runtime_expert(
        ctx=ctx,
        parent_session_id=session_id,
        message=message,
        model=model,
        context_json=workbench_context_json,
        enabled=body.get("runtime_expert") is True,
    )
    if runtime_expert_result is not None:
        return runtime_expert_result
    raw_system = body.get("system_message")
    requested_system_message = (
        raw_system.strip()
        if isinstance(raw_system, str) and raw_system.strip()
        else None
    )
    system_message = _merge_system_message(
        ctx,
        user_message=message,
        requested_system_message=requested_system_message,
    )
    system_message = _append_workbench_context(system_message, workbench_context_json)

    _prepare_governed_analytics_context(ctx, user_message=message)
    internal_session_id = _ensure_runtime_session_hydrated(ctx, session_id)
    expanded_message, _expanded_skill = _expand_web_skill_slash_command(
        ctx,
        session_id=internal_session_id,
        message=message,
    )
    from agents.wiki_task_runtime import prepare_wiki_task
    expanded_message = prepare_wiki_task(ctx, internal_session_id, _expanded_skill, expanded_message)
    system_message = _with_cron_error_followup_context(
        ctx,
        session_id=internal_session_id,
        user_message=message,
        system_message=system_message,
    )
    if ctx.organization_id:
        reconcile_retained_context_items(context=ctx)
    data, response_session_id = await _post_embedded_chat_completion(
        session_id=internal_session_id,
        upload_session_id=session_id,
        message=expanded_message,
        request_metadata=request_metadata,
        model=model,
        system_message=system_message,
        hermes_home=ctx.hermes_home,
        user_id=ctx.user_id,
        workspace_id=ctx.workspace_id,
    )
    return {
        "ok": True,
        "session_id": _external_session_id(response_session_id, ctx),
        "run_id": _external_session_id(response_session_id, ctx),
        "message": (((data.get("choices") or [{}])[0] or {}).get("message") or {}).get(
            "content"
        ),
        "usage": data.get("usage"),
    }


@router.post("/api/sessions/{session_id}/chat/stream")
async def api_session_chat_stream(session_id: str, request: Request) -> Response:
    ctx = _require_authenticated_context(request)
    _enforce_rate_limit("chat_user", request=request, ctx=ctx)
    _enforce_rate_limit(
        "chat_tenant",
        request=request,
        ctx=ctx,
        subject=str(ctx.organization_id or ctx.workspace_id),
    )
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}

    tender_context_json = await _resolve_tender_journey_context_hint(
        ctx, session_id, body.get("tender_journey_context")
    )
    workbench_context_json = _resolve_workbench_context_hint(
        ctx, body.get("knowledge_workbench_context")
    )

    raw_message = body.get("message")
    message = raw_message.strip() if isinstance(raw_message, str) else ""
    if not message:
        raise HTTPException(status_code=400, detail="message required")

    from services.learning_review_commands import maybe_learning_review_command
    review = maybe_learning_review_command(ctx, message, body.get("requestIdentity"))
    if review is not None:
        return review

    try:
        request_metadata = durable_request_metadata(body.get("requestIdentity"), raw_message, body.get("requestHash"))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    raw_model = body.get("model")
    model = (
        raw_model.strip() if isinstance(raw_model, str) and raw_model.strip() else None
    )
    raw_system = body.get("system_message")
    requested_system_message = (
        raw_system.strip()
        if isinstance(raw_system, str) and raw_system.strip()
        else None
    )
    system_message = _merge_system_message(
        ctx,
        user_message=message,
        requested_system_message=requested_system_message,
    )
    system_message = _append_workbench_context(system_message, workbench_context_json)
    system_message = _append_workbench_context(system_message, tender_context_json)
    from agents.webapi.presentation_context import presentation_system_context
    system_message = _append_workbench_context(
        system_message, presentation_system_context(ctx, message, body.get("presentation_context")),
    )

    _prepare_governed_analytics_context(ctx, user_message=message)
    internal_session_id = _ensure_runtime_session_hydrated(ctx, session_id)
    expanded_message, _expanded_skill = _expand_web_skill_slash_command(
        ctx,
        session_id=internal_session_id,
        message=message,
    )
    from agents.wiki_task_runtime import prepare_wiki_task
    expanded_message = prepare_wiki_task(ctx, internal_session_id, _expanded_skill, expanded_message)
    system_message = _with_cron_error_followup_context(
        ctx,
        session_id=internal_session_id,
        user_message=message,
        system_message=system_message,
    )
    if ctx.organization_id:
        reconcile_retained_context_items(context=ctx)
    return await _stream_embedded_chat_completion(
        session_id=internal_session_id,
        upload_session_id=session_id,
        message=expanded_message,
        request_metadata=request_metadata,
        model=model,
        system_message=system_message,
        hermes_home=ctx.hermes_home,
        user_id=ctx.user_id,
        workspace_id=ctx.workspace_id,
    )


@router.post("/api/sessions/send")
async def api_sessions_send(request: Request) -> Any:
    ctx = _require_authenticated_context(request)
    _enforce_rate_limit("chat_user", request=request, ctx=ctx)
    _enforce_rate_limit(
        "chat_tenant",
        request=request,
        ctx=ctx,
        subject=str(ctx.organization_id or ctx.workspace_id),
    )
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}

    workbench_context_json = _resolve_workbench_context_hint(
        ctx, body.get("knowledge_workbench_context")
    )

    raw_session_key = (
        body.get("sessionKey") or body.get("friendlyId") or body.get("session_id")
    )
    session_key = raw_session_key.strip() if isinstance(raw_session_key, str) else ""
    if not session_key:
        raise HTTPException(status_code=400, detail="sessionKey required")

    raw_message = body.get("message")
    message = raw_message.strip() if isinstance(raw_message, str) else ""
    if not message:
        raise HTTPException(status_code=400, detail="message required")

    from services.learning_review_commands import maybe_learning_review_command
    review = maybe_learning_review_command(ctx, message, body.get("requestIdentity"))
    if review is not None:
        return review

    try:
        request_metadata = durable_request_metadata(body.get("requestIdentity"), raw_message, body.get("requestHash"))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    raw_model = body.get("model")
    model = (
        raw_model.strip() if isinstance(raw_model, str) and raw_model.strip() else None
    )
    runtime_expert_result = await _maybe_run_tender_runtime_expert(
        ctx=ctx,
        parent_session_id=session_key,
        message=message,
        model=model,
        context_json=workbench_context_json,
        enabled=body.get("runtime_expert") is True,
    )
    if runtime_expert_result is not None:
        return runtime_expert_result
    raw_system = body.get("system_message")
    requested_system_message = (
        raw_system.strip()
        if isinstance(raw_system, str) and raw_system.strip()
        else None
    )
    system_message = _merge_system_message(
        ctx,
        user_message=message,
        requested_system_message=requested_system_message,
    )
    system_message = _append_workbench_context(system_message, workbench_context_json)

    _prepare_governed_analytics_context(ctx, user_message=message)
    internal_session_id = _ensure_runtime_session_hydrated(ctx, session_key)
    expanded_message, _expanded_skill = _expand_web_skill_slash_command(
        ctx,
        session_id=internal_session_id,
        message=message,
    )
    from agents.wiki_task_runtime import prepare_wiki_task
    expanded_message = prepare_wiki_task(ctx, internal_session_id, _expanded_skill, expanded_message)
    system_message = _with_cron_error_followup_context(
        ctx,
        session_id=internal_session_id,
        user_message=message,
        system_message=system_message,
    )
    if ctx.organization_id:
        reconcile_retained_context_items(context=ctx)
    data, response_session_id = await _post_embedded_chat_completion(
        session_id=internal_session_id,
        message=expanded_message,
        request_metadata=request_metadata,
        model=model,
        system_message=system_message,
        hermes_home=ctx.hermes_home,
        user_id=ctx.user_id,
        workspace_id=ctx.workspace_id,
    )
    external_session_id = _external_session_id(response_session_id, ctx)
    return {
        "ok": True,
        "sessionKey": external_session_id,
        "session_id": external_session_id,
        "runId": external_session_id,
        "run_id": external_session_id,
        "message": (((data.get("choices") or [{}])[0] or {}).get("message") or {}).get(
            "content"
        ),
        "usage": data.get("usage"),
    }
