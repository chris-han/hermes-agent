from __future__ import annotations

from storage import hosted_postgres as hosted


import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
import re
from pathlib import Path
from time import monotonic
from uuid import uuid4
from hashlib import sha256

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from agents.route_policy import classify_authorization_class
from agents.webapi.deps import require_route_authorization as _require_route_authorization
from storage.canonical_term_store import CanonicalTermStore
from storage.knowledge_builder_evaluation_store import KnowledgeBuilderEvaluationStore
from storage.knowledge_builder_store import KnowledgeBuilderStore
from storage.runtime_semantic_index_store import RuntimeSemanticIndexStore
from storage.runtime_feedback_store import RuntimeFeedbackStore
from storage.semantic_feedback_event_store import SemanticFeedbackEventStore
from contracts.knowledge_feature_pack import (
    content_hash,
    HumanDecision,
    HumanLabel,
    LearningEventEnvelope,
    LearningEventType,
)
from contracts.knowledge_graph_phase1 import CandidateEdge, CandidateRelationType, GraphDelta, SensitiveTermCandidateGraph
from contracts.accepted_graph_release_lineage import (
    AcceptedGraphReleaseLineage,
    AcceptedGraphReleaseLineageV2,
)
from contracts.accepted_graph_release_v2 import AcceptedGraphReleaseV2
from contracts.corpus_graph_schema import CorpusGraphSchemaCandidate
from contracts.governed_extract import ExtractResult
from contracts.schema_decision import SchemaDecision
from contracts.knowledge_reasoning_phase1 import (
    EvidenceObligation,
    KnowledgeScope,
    TenderPackageDocument,
)
from knowledge_extraction.service import KnowledgeExtractionService
from knowledge_extraction.config import load_knowledge_extraction_config, SUPPORTED_PROVIDERS
from runtime_paths import sqlite_db_path
from semantier.knowledge_feature_packs.falkordb_backend import (
    FalkorDBGraphBackend,
    InMemoryGraphBackend,
)
from semantier.knowledge_feature_packs.grounding_workbench import HumanGroundingWorkbench
from storage.human_grounding_store import HumanGroundingStore
from storage.corpus_graph_schema_candidate_store import (
    CorpusGraphSchemaCandidateStore,
)
from storage.graph_intent_brief_store import GraphIntentBriefStore
from storage.schema_decision_store import SchemaDecisionStore
from storage.profile_relation_type_proposal_store import ProfileRelationTypeProposalStore
from services.corpus_graph_schema_profile_registry import CorpusGraphSchemaProfileRegistry
from services.knowledge_relation_taxonomy_service import KnowledgeRelationTaxonomyService
from services.materialized_accepted_graph_bundle_reader import (
    MaterializedAcceptedGraphBundleFailure,
    MaterializedAcceptedGraphBundleReader,
)
from services.semantic_graph_projection_service import (
    SemanticGraphProjectionFailure,
    SemanticGraphProjectionService,
)
from services.semantic_layout_engine import (
    GraphLayoutLimitExceeded,
    SemanticLayoutEngine,
    load_default_semantic_layout_profile,
)
from services.graph_coherence_analyzer import (
    GraphCoherenceAnalyzer,
    GraphCoherenceLimitExceeded,
    load_default_graph_coherence_policy,
)
from services.graph_coherence_artifact_store import (
    GraphCoherenceArtifactConflict,
    GraphCoherenceArtifactStore,
)
from services.accepted_graph_rule_projection_service import AcceptedGraphRuleProjectionService
from services.accepted_semantic_graph_materialization_service import (
    AcceptedSemanticGraphMaterializationService,
    ResolveMaterializationInputsCommand,
    ResolveReviewedPatternInputsCommand,
)
from storage.accepted_graph_release_origin import (
    verify_exact_release_origin_reservation,
)
from services.canonical_skill_registry import CANONICAL_JOURNEY_REF
from services.regex_improvement_proposal import evaluate_regex_proposal
from eos.artifact_hashing import canonical_json
from storage.knowledge_extraction_store import KnowledgeExtractionStore
from storage.knowledge_source_compilation_store import KnowledgeSourceCompilationStore
from storage.benchmark_read_model_store import BenchmarkReadModelStore
from storage.evaluation_control_plane_store import EvaluationControlPlaneStore
from contracts.knowledge_evaluation import KnowledgeBenchmarkRunRequestV1
from contracts.ai_grounding import (
    AiGroundingRequest,
    BatchAcceptRequest,
    StructuredGenerationError,
)
from contracts.knowledge_extraction import KnowledgeBuilderExtractionRunRequestV2
from contracts.contextgraph_reference import ReferenceConcept
from knowledge_extraction.contracts_chinese_concept_boundary_ic1_ic7 import (
    CurationAction,
)
from agents.hermes_current_agent_llm import HermesCurrentAgentStructuredLlm
from prompts.loader import load_repo_prompt
from semantier.knowledge_feature_packs.semantica_grounding_assistant import (
    GroundingCandidateInput,
    SemanticaGroundingAssistant,
)
from semantier.knowledge_feature_packs.semantica_adapter import SEMANTICA_COMMIT
from semantier.knowledge_feature_packs.relationship_extraction import (
    RELATION_EXTRACTION_SCHEMA,
    build_relationship_extraction_input,
)
from semantier.semantica_context_graph.artifact_repository import (
    ArtifactConflict,
    ContextGraphArtifactRepository,
)
from semantier.semantica_context_graph.native_builder import (
    GRAPH_ID,
    native_graph_from_accepted_release,
    native_graph_from_accepted_release_bundle,
)
from semantier.semantica_context_graph.accepted_release_conformance import (
    validate_accepted_release_conformance,
)
from services.accepted_graph_release_checkpoint import (
    accepted_release_source_manifest_hash,
    verify_accepted_release_checkpoint,
)

router = APIRouter(tags=["webapi-gateway"])


def _authz(request: Request, method: str, path: str):
    authz_class = classify_authorization_class(method, path)
    if authz_class is None:
        raise HTTPException(status_code=500, detail="Route auth class not configured")
    return _require_route_authorization(
        request,
        authorization_class=authz_class,
        route_hint=f"{method} {path}",
    )


async def _json_body(request: Request) -> dict:
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="JSON body required") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON object body required")
    return body


def _eos_path() -> Path:
    return Path(os.environ.get("SEMANTIER_EOS_DB_PATH", str(sqlite_db_path("eos.db", ".semantier/eos.db"))))


def _extraction_store() -> KnowledgeExtractionStore:
    return KnowledgeExtractionStore(_eos_path())


def _current_agent_structured_llm(hermes_home: Path) -> HermesCurrentAgentStructuredLlm:
    return HermesCurrentAgentStructuredLlm(hermes_home=hermes_home)


def _few_shot_relationship_proposer(
    hermes_home: Path, *, negative_examples: list[dict[str, object]] | None = None
):
    """Return a synchronous adapter for the async active-agent LLM bridge.

    Extraction is currently a synchronous service boundary. The single LLM
    request runs in a worker thread so an async FastAPI request does not try to
    nest an event loop, while the provider remains the active configured agent.
    """
    def propose(elements: list[dict[str, object]]) -> dict[str, object]:
        llm = _current_agent_structured_llm(hermes_home)
        prompt = load_repo_prompt(
            "contextgraph/semantica_relationship_extraction_v1.md",
            allowed_namespace="contextgraph",
        )
        input_text = build_relationship_extraction_input(
            elements, negative_examples=negative_examples
        )

        def run() -> dict[str, object]:
            result = asyncio.run(llm.generate_structured(
                instructions=prompt.text,
                input_text=input_text,
                json_schema=RELATION_EXTRACTION_SCHEMA,
                schema_name="semantier_semantic_relationship_extraction_v1",
                purpose="contextgraph_ai_extract",
            ))
            return result.parsed

        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="semantic-extraction") as pool:
            return pool.submit(run).result()

    return propose


def _benchmark_store() -> BenchmarkReadModelStore:
    return BenchmarkReadModelStore(_eos_path())


def _benchmark_limit(request: Request) -> int:
    try:
        limit = int(request.query_params.get("limit") or 50)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="limit must be an integer") from exc
    if not 1 <= limit <= 200:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 200")
    return limit


def _projection_hash(payload: dict) -> str:
    return content_hash({key: value for key, value in payload.items() if key != "content_hash"})


def _validated_compilation_anchor(*, anchor: dict, compilation: dict, source: dict,
                                  canonical_document_id: str) -> dict:
    if anchor.get("knowledge_source_compilation_run_id") != canonical_document_id:
        raise ValueError("canonical compilation anchor run mismatch")
    if anchor.get("knowledge_source_id") != compilation["knowledge_source_id"]:
        raise ValueError("canonical compilation anchor source mismatch")
    if anchor.get("source_hash") != compilation["source_hash"] or anchor.get("source_hash") != source["source_hash"]:
        raise ValueError("canonical compilation anchor source hash mismatch")
    payload = dict(anchor.get("anchor_payload") or {})
    start_raw = payload.get("start_offset", payload.get("start_char"))
    end_raw = payload.get("end_offset", payload.get("end_char"))
    try:
        start, end = int(start_raw), int(end_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("canonical compilation anchor offsets are required integers") from exc
    text = source["source_text"]
    if not 0 <= start < end <= len(text):
        raise ValueError("canonical compilation anchor offsets are out of bounds")
    exact = text[start:end]
    pinned_exact = payload.get("exact_text", payload.get("text"))
    if pinned_exact is not None and pinned_exact != exact:
        raise ValueError("canonical compilation anchor exact_text mismatch")
    pinned_hash = payload.get("local_text_hash", payload.get("text_hash"))
    exact_hash = content_hash(exact)
    if pinned_hash is not None and pinned_hash != exact_hash:
        raise ValueError("canonical compilation anchor local_text_hash mismatch")
    return {
        "anchor_id": anchor["compilation_anchor_ref"],
        "canonical_document_anchor_ref": anchor["canonical_document_anchor_ref"],
        "document_hash": source["source_hash"],
        "document_part": payload.get("document_part", "paragraph"),
        "paragraph_index": payload.get("paragraph_index", 0),
        "start_offset": start,
        "end_offset": end,
        "local_text_hash": exact_hash,
        "source_id": source["source_id"],
        "source_ref": f"knowledge-source:{source['source_id']}",
        "source_hash": source["source_hash"],
        "compilation_run_id": canonical_document_id,
    }


def _hydrate_scope(scope: dict, builder: KnowledgeBuilderStore) -> dict:
    hydrated = {**scope, "schema_version": "knowledge_scope.v1"}
    documents = []
    for document in scope.get("documents", []):
        elements = []
        for element in document.get("elements", []):
            source = builder.get_source(str(element.get("source_id") or ""))
            if source is None or source["source_hash"] != document["document_hash"]:
                raise ValueError("tender package canonical source unavailable or changed")
            if element.get("source_hash") != source["source_hash"] or element.get("document_hash") != source["source_hash"]:
                raise ValueError("tender package anchor source/document hash mismatch")
            try:
                start = int(element["start_offset"])
                end = int(element["end_offset"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("tender package anchor offsets are required integers") from exc
            if not 0 <= start < end <= len(source["source_text"]):
                raise ValueError("tender package anchor offsets are out of bounds")
            exact_text = source["source_text"][start:end]
            if element.get("exact_text") is not None and element["exact_text"] != exact_text:
                raise ValueError("tender package anchor exact_text mismatch")
            expected_local_hash = content_hash(exact_text)
            if element.get("local_text_hash") is not None and element["local_text_hash"] != expected_local_hash:
                raise ValueError("tender package anchor local_text_hash mismatch")
            elements.append({**element, "text": exact_text, "exact_text": exact_text,
                             "local_text_hash": expected_local_hash})
        documents.append({**document, "elements": elements})
    hydrated["documents"] = documents
    return hydrated


def _require_org(owner: dict | None, organization_id: str, *, label: str) -> dict:
    if owner is None or owner.get("organization_id") != organization_id:
        raise HTTPException(status_code=404, detail=f"{label} not found")
    return owner


def _require_owner(owner: dict | None, organization_id: str, workspace_id: str | None,
                   *, label: str) -> dict:
    owner = _require_org(owner, organization_id, label=label)
    if owner.get("workspace_id") != workspace_id:
        raise HTTPException(status_code=404, detail=f"{label} not found")
    return owner


def _page_limit(request: Request) -> int:
    try:
        limit = int(request.query_params.get("limit") or 100)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="limit must be an integer") from exc
    if not 1 <= limit <= 500:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 500")
    return limit


def _anchor_context_text(anchor_ref: str, source_text: str | None) -> str | None:
    if not source_text:
        return None
    match = re.search(r":paragraph:(\d+)(?::|$)", anchor_ref)
    if match is None:
        return None
    paragraphs = source_text.splitlines()
    index = int(match.group(1))
    return paragraphs[index] if 0 <= index < len(paragraphs) else None


def _reference_anchor_docx_location(
    anchor_ref: str, exact_text: str, source_text: str | None
) -> dict[str, Any] | None:
    """Project governed Markdown coordinates into a native DOCX structure.

    Reference-concept anchors are indexed against normalized Markdown.  Table
    rows therefore cannot be treated as Flyfish paragraph ordinals.  This
    projection preserves the governed term/span coordinate, maps Markdown
    table blocks to DOCX table/row/cell paths, and fails closed when the
    coordinate does not reproduce ``exact_text``.
    """

    if not source_text or not exact_text:
        return None
    match = re.search(
        r":paragraph:(\d+):term:(\d+)-(\d+):span:(\d+)-(\d+)(?::|$)",
        anchor_ref,
    )
    if match is None:
        return None
    paragraph_index, term_start, _term_end, span_start, span_end = (
        int(value) for value in match.groups()
    )
    lines = source_text.splitlines(keepends=True)
    if not 0 <= paragraph_index < len(lines):
        return None
    absolute_start = term_start + span_start
    absolute_end = term_start + span_end
    if source_text[absolute_start:absolute_end] != exact_text:
        return None

    line_start = sum(len(line) for line in lines[:paragraph_index])
    line = lines[paragraph_index].rstrip("\r\n")
    relative_start = absolute_start - line_start
    relative_end = absolute_end - line_start
    if relative_start < 0 or relative_end > len(line):
        return None

    table_ordinal = -1
    row_ordinal = -1
    in_table = False
    target_table: int | None = None
    target_row: int | None = None
    for index, candidate in enumerate(lines[: paragraph_index + 1]):
        stripped = candidate.rstrip("\r\n").strip()
        is_table_row = stripped.startswith("|") and stripped.endswith("|")
        if not is_table_row:
            in_table = False
            continue
        next_stripped = (
            lines[index + 1].rstrip("\r\n").strip()
            if index + 1 < len(lines)
            else ""
        )
        next_cells = (
            [cell.strip() for cell in next_stripped[1:-1].split("|")]
            if next_stripped.startswith("|") and next_stripped.endswith("|")
            else []
        )
        starts_new_adjacent_table = bool(next_cells) and all(
            re.fullmatch(r":?-{3,}:?", cell) is not None
            for cell in next_cells
        )
        if not in_table or (starts_new_adjacent_table and row_ordinal >= 0):
            table_ordinal += 1
            row_ordinal = -1
            in_table = True
        cells = [cell.strip() for cell in stripped[1:-1].split("|")]
        is_separator = bool(cells) and all(
            re.fullmatch(r":?-{3,}:?", cell) is not None for cell in cells
        )
        if not is_separator:
            row_ordinal += 1
        if index == paragraph_index:
            if is_separator:
                return None
            target_table = table_ordinal
            target_row = row_ordinal

    if target_table is not None and target_row is not None:
        pipe_positions = [index for index, value in enumerate(line) if value == "|"]
        for cell_index, (left, right) in enumerate(
            zip(pipe_positions, pipe_positions[1:])
        ):
            raw_start = left + 1
            raw_end = right
            while raw_start < raw_end and line[raw_start].isspace():
                raw_start += 1
            while raw_end > raw_start and line[raw_end - 1].isspace():
                raw_end -= 1
            if raw_start <= relative_start and relative_end <= raw_end:
                return {
                    "kind": "docx_structural_range",
                    "structuralPath": [
                        f"table:{target_table}",
                        f"row:{target_row}",
                        f"cell:{cell_index}",
                    ],
                    "startOffset": relative_start - raw_start,
                    "endOffset": relative_end - raw_start,
                    "exactText": exact_text,
                }
        return None

    paragraph_ordinal = sum(
        1
        for candidate in lines[:paragraph_index]
        if not (
            candidate.rstrip("\r\n").strip().startswith("|")
            and candidate.rstrip("\r\n").strip().endswith("|")
        )
        and candidate.rstrip("\r\n").strip()
    )
    return {
        "kind": "docx_structural_range",
        "structuralPath": [f"paragraph:{paragraph_ordinal}"],
        "startOffset": relative_start,
        "endOffset": relative_end,
        "exactText": exact_text,
    }


def _reference_concept_candidate(
    concept: dict, expressions: list[dict] | None = None, *, source_text: str | None = None
) -> dict:
    label = str(concept.get("canonical_label") or concept.get("concept_id") or "")
    source_anchor_refs = [str(ref) for ref in concept.get("source_anchor_refs", [])]
    expression = next(
        (item for item in (expressions or []) if item.get("concept_id") == concept.get("concept_id")),
        None,
    )
    evidence_refs = [
        {"evidence_ref": anchor_ref, "selector_hash": content_hash(anchor_ref)}
        for anchor_ref in source_anchor_refs
    ]
    return {
        "assertion_id": concept["concept_id"],
        "concept_id": concept["concept_id"],
        "candidate_graph_id": concept.get("candidate_graph_id"),
        "confidence": 1.0,
        "grounding_state": concept.get("grounding_state", "candidate"),
        "source_document_hash": concept.get("source_document_hash"),
        "source_anchors": [
            {
                "anchor_id": anchor_ref,
                "exact_text": exact_text,
                **(
                    {"presentation_location": location}
                    if (
                        location := _reference_anchor_docx_location(
                            anchor_ref, exact_text, source_text
                        )
                    )
                    else {}
                ),
            }
            for anchor_ref in source_anchor_refs
            for exact_text in [str(expression.get("text") if expression else label)]
        ],
        "evidence_refs": evidence_refs,
        "normalized_assertion": {
            "subject": {"text": label},
            "predicate": "reference_concept",
            "object": {"text": expression.get("text") if expression else label},
        },
        "extraction_run_id": concept.get("extraction_run_id"),
    }


def _canonical_evidence_review_payload(
    *, concept: dict, expressions: list[dict], evidence_ref: str,
    source_text: str | None = None,
) -> tuple[dict[str, Any], str]:
    candidate = _reference_concept_candidate(concept, expressions, source_text=source_text)
    anchors = candidate["source_anchors"]
    anchor_index = next(
        (index for index, anchor in enumerate(anchors) if anchor["anchor_id"] == evidence_ref),
        None,
    )
    if anchor_index is None:
        raise ValueError("evidenceRef is not canonical evidence for reference concept")
    concept_label = str(concept.get("canonical_label") or concept["concept_id"])
    context_text = str(anchors[anchor_index].get("exact_text") or concept_label)
    start_offset = context_text.find(concept_label)
    return ({
        "decision_scope": "evidence_occurrence",
        "concept_id": concept["concept_id"],
        "concept_label": concept_label,
        "matched_text": concept_label,
        "context_text": context_text,
        "start_offset": start_offset if start_offset >= 0 else None,
        "end_offset": start_offset + len(concept_label) if start_offset >= 0 else None,
    }, f"{concept['concept_id']}::evidence::{anchor_index}")


def _reference_concept_ai_suggestion(concept: dict) -> dict:
    anchor_refs = [str(ref) for ref in concept.get("source_anchor_refs", [])]
    split_proposals = _reference_concept_split_proposals(concept)
    return {
        "assertion_id": concept["concept_id"],
        "concept_id": concept["concept_id"],
        "suggestion_status": "supported",
        "status": "supported",
        "confidence": 1.0,
        "evidence_anchor_count": len(anchor_refs),
        "evidence_anchor_refs": anchor_refs,
        "provider": "semantier_reference_concept",
        "provider_version": "contextgraph_reference_concept.v1",
        "model": "deterministic_reference_metadata",
        "threshold": 1.0,
        "rationale": "Reference concept extracted from governed source-list evidence.",
        "issues": [],
        "split_proposals": split_proposals,
    }


_MODIFIER_PREFIXES = (
    "仅限", "限于", "仅允许", "不得", "禁止", "不包括", "排除", "适用于", "针对",
)


def _reference_concept_split_proposals(concept: dict) -> list[dict[str, Any]]:
    label = str(concept.get("canonical_label") or "")
    prefix = next(
        (item for item in sorted(_MODIFIER_PREFIXES, key=len, reverse=True)
         if label.startswith(item) and len(label) > len(item)),
        None,
    )
    if prefix is None:
        return []
    return [{
        "offsets": [len(prefix)],
        "segments": [prefix, label[len(prefix):]],
        "confidence": 0.96,
        "rationale": "Deterministic modifier prefix directly governs the following concept span.",
    }]


@router.post("/api/knowledge/builder/sources")
async def create_knowledge_builder_source(request: Request):
    route_path = "/api/knowledge/builder/sources"
    ctx = _authz(request, "POST", route_path)
    body = await _json_body(request)
    try:
        source = KnowledgeBuilderStore().create_source(
            organization_id=str(ctx.organization_id),
            workspace_id=ctx.workspace_id,
            source_kind=str(body.get("sourceKind") or body.get("source_kind") or "text"),
            source_ref=str(body.get("sourceRef") or body.get("source_ref") or "inline_source"),
            source_text=str(body.get("sourceText") or body.get("source_text") or ""),
            source_metadata=body.get("sourceMetadata") or body.get("source_metadata") or {},
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "source": source}


@router.post("/api/knowledge/builder/discovery-runs")
async def create_knowledge_builder_discovery_run(request: Request):
    route_path = "/api/knowledge/builder/discovery-runs"
    ctx = _authz(request, "POST", route_path)
    body = await _json_body(request)
    try:
        run = KnowledgeBuilderStore().create_discovery_run(
            organization_id=str(ctx.organization_id),
            workspace_id=ctx.workspace_id,
            source_id=body.get("sourceId") or body.get("source_id"),
            source_kind=str(body.get("sourceKind") or body.get("source_kind") or "text"),
            source_ref=str(body.get("sourceRef") or body.get("source_ref") or "inline_source"),
            source_text=body.get("sourceText") or body.get("source_text"),
            source_metadata=body.get("sourceMetadata") or body.get("source_metadata") or {},
            model_provenance=body.get("modelProvenance") or body.get("model_provenance"),
            procedure_provenance=body.get("procedureProvenance") or body.get("procedure_provenance"),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "run": run}


@router.get("/api/knowledge/builder/discovery-runs/{run_id}")
async def get_knowledge_builder_discovery_run(run_id: str, request: Request):
    route_path = "/api/knowledge/builder/discovery-runs/{run_id}"
    ctx = _authz(request, "GET", route_path)
    run = KnowledgeBuilderStore().get_discovery_run(run_id)
    if (run is None or run["organization_id"] != ctx.organization_id or
            run.get("workspace_id") != ctx.workspace_id):
        raise HTTPException(status_code=404, detail="discovery run not found")
    return {"ok": True, "run": run}


@router.post("/api/knowledge/builder/canonical-terms")
async def create_knowledge_builder_canonical_term(request: Request):
    route_path = "/api/knowledge/builder/canonical-terms"
    ctx = _authz(request, "POST", route_path)
    body = await _json_body(request)
    try:
        candidate = CanonicalTermStore().create_candidate(
            discovery_run_id=str(body.get("discoveryRunId") or body.get("discovery_run_id") or ""),
            domain=str(body.get("domain") or "tender_compliance"),
            canonical_label=str(body.get("canonicalLabel") or body.get("canonical_label") or ""),
            definition=str(body.get("definition") or ""),
            aliases=[str(item) for item in body.get("aliases") or []],
            allowed_contexts=[str(item) for item in body.get("allowedContexts") or body.get("allowed_contexts") or []],
            prohibited_contexts=[str(item) for item in body.get("prohibitedContexts") or body.get("prohibited_contexts") or []],
            source_anchor_refs=[str(item) for item in body.get("sourceAnchorRefs") or body.get("source_anchor_refs") or []],
            evidence_summary=str(body.get("evidenceSummary") or body.get("evidence_summary") or ""),
            proposed_runtime_effect=body.get("proposedRuntimeEffect") or body.get("proposed_runtime_effect") or {},
            governance_state=str(body.get("governanceState") or body.get("governance_state") or "PROPOSED"),
            created_by=str(ctx.user_id),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "termCandidate": candidate}


@router.post("/api/knowledge/builder/evaluation-datasets")
async def create_knowledge_builder_evaluation_dataset(request: Request):
    route_path = "/api/knowledge/builder/evaluation-datasets"
    ctx = _authz(request, "POST", route_path)
    body = await _json_body(request)
    store = KnowledgeBuilderEvaluationStore()
    try:
        if body.get("useTenderUatFixture") or body.get("use_tender_uat_fixture"):
            dataset = store.create_tender_uat_dataset(
                organization_id=str(ctx.organization_id),
                workspace_id=ctx.workspace_id,
                discovery_run_id=body.get("discoveryRunId") or body.get("discovery_run_id"),
                created_by=str(ctx.user_id),
            )
        else:
            dataset = store.create_dataset(
                organization_id=str(ctx.organization_id),
                workspace_id=ctx.workspace_id,
                discovery_run_id=body.get("discoveryRunId") or body.get("discovery_run_id"),
                domain=str(body.get("domain") or "tender_compliance"),
                name=str(body.get("name") or "Knowledge Builder evaluation dataset"),
                description=str(body.get("description") or ""),
                examples=list(body.get("examples") or []),
                created_by=str(ctx.user_id),
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "evaluationDataset": dataset}


@router.get("/api/knowledge/builder/evaluation-datasets/{dataset_id}")
async def get_knowledge_builder_evaluation_dataset(dataset_id: str, request: Request):
    route_path = "/api/knowledge/builder/evaluation-datasets/{dataset_id}"
    ctx = _authz(request, "GET", route_path)
    dataset = KnowledgeBuilderEvaluationStore().get_dataset(dataset_id)
    if dataset is None or dataset["organization_id"] != ctx.organization_id:
        raise HTTPException(status_code=404, detail="evaluation dataset not found")
    return {"ok": True, "evaluationDataset": dataset}


@router.post("/api/knowledge/builder/evaluation-runs")
async def create_knowledge_builder_evaluation_run(request: Request):
    route_path = "/api/knowledge/builder/evaluation-runs"
    ctx = _authz(request, "POST", route_path)
    body = await _json_body(request)
    try:
        evaluation_run = KnowledgeBuilderEvaluationStore().create_evaluation_run(
            evaluation_dataset_id=str(
                body.get("evaluationDatasetId") or body.get("evaluation_dataset_id") or ""
            ),
            discovery_run_id=body.get("discoveryRunId") or body.get("discovery_run_id"),
            created_by=str(ctx.user_id),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "evaluationRun": evaluation_run}


@router.post("/api/knowledge/builder/evaluation-results/{result_id}/rating")
async def rate_knowledge_builder_evaluation_result(result_id: str, request: Request):
    route_path = "/api/knowledge/builder/evaluation-results/{result_id}/rating"
    _authz(request, "POST", route_path)
    body = await _json_body(request)
    try:
        result = KnowledgeBuilderEvaluationStore().update_result_rating(
            evaluation_result_id=result_id,
            human_rating=str(body.get("humanRating") or body.get("human_rating") or ""),
            explanation_acceptance=str(
                body.get("explanationAcceptance") or body.get("explanation_acceptance") or "needs_review"
            ),
            error_labels=[str(item) for item in body.get("errorLabels") or body.get("error_labels") or []],
            reviewer_notes=body.get("reviewerNotes") or body.get("reviewer_notes"),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "evaluationResult": result}


@router.post("/api/knowledge/builder/benchmark-runs")
async def create_knowledge_benchmark_run(request: Request):
    route_path = "/api/knowledge/builder/benchmark-runs"
    ctx = _authz(request, "POST", route_path)
    body = await _json_body(request)
    try:
        launch = KnowledgeBenchmarkRunRequestV1.model_validate(body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if launch.profileId == 'policy-version-engineering.v1':
        if (launch.executionMode != 'real' or launch.layers != ('graph',) or launch.profileVersion != '1'
            or launch.benchmarkFamilyRef != 'knowledge-evaluation-stack.v2'):
            raise HTTPException(400, 'POLICY_REAL_GRAPH_EVALUATION_REQUIRED')
        from agents.webapi.expense_workbench_routes import owner, policy_revision_owner, PolicyGraphEvalCommand
        try:
            policy_command = PolicyGraphEvalCommand.model_validate(launch.caseFilters or {})
        except ValueError as exc:
            raise HTTPException(400, 'INVALID_POLICY_EVALUATION_PINS') from exc
        def execute_policy_evaluation():
            with owner(request) as expense_service:
                result = policy_revision_owner(expense_service).evaluate(**policy_command.model_dump())
                run, report = result['run'], result['report']
                result['benchmarkOrchestration'] = dict(
                    benchmark_orchestration_id=run['artifact_ref'],
                    organization_id=str(ctx.organization_id), workspace_id=ctx.workspace_id,
                    profile_id=launch.profileId, profile_version=launch.profileVersion,
                    execution_mode='real', requested_layers=['graph'], operational_status='COMPLETED',
                    created_at=run['produced_at'], warnings=[], child_run_refs=[dict(
                        layer='graph', evaluation_run_id=run['artifact_ref'], operational_status='COMPLETED',
                        certification_result='NOT_EVALUABLE', run_validity='VALID')])
                return result
        from starlette.concurrency import run_in_threadpool
        return await run_in_threadpool(execute_policy_evaluation)
    orchestration_id = f"benchmark-orchestration:{uuid4().hex}"
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    children = [
        {
            "layer": layer,
            "evaluation_run_id": f"evaluation-run:{uuid4().hex}",
            "evaluation_evidence_id": None,
            "operational_status": "QUEUED",
            "run_validity": None,
            "certification_result": None,
        }
        for layer in launch.layers
    ]
    record = {
        "schema_version": "knowledge_benchmark_orchestration.v1",
        "benchmark_orchestration_id": orchestration_id,
        "organization_id": str(ctx.organization_id),
        "workspace_id": ctx.workspace_id,
        "benchmark_family_ref": launch.benchmarkFamilyRef,
        "profile_id": launch.profileId,
        "profile_version": launch.profileVersion,
        "requested_layers": list(launch.layers),
        "execution_mode": launch.executionMode,
        "operational_status": "QUEUED",
        "warnings": [
            {
                "code": "RUNTIME_PROFILE_UNAVAILABLE_FOR_PATH_CRITICAL_CLASSIFICATION",
                "providers": list(launch.providers),
            }
        ],
        "child_run_refs": children,
        "created_at": now,
        "completed_at": None,
        "stale_reason_code": None,
    }
    record["content_hash"] = _projection_hash(record)
    store = _benchmark_store()
    try:
        stored = store.put_orchestration(record)
        for child in children:
            layer_record = {
                "schema_version": "knowledge_benchmark_run.v1",
                "benchmark_orchestration_id": orchestration_id,
                "organization_id": str(ctx.organization_id),
                "workspace_id": ctx.workspace_id,
                "evaluation_run_id": child["evaluation_run_id"],
                "layer": child["layer"],
                "provider_id": launch.providers[0] if child["layer"] == "extraction" and launch.providers else None,
                "profile_id": launch.profileId,
                "profile_version": launch.profileVersion,
                "execution_mode": launch.executionMode,
                "operational_status": "QUEUED",
                "evaluation_evidence_id": None,
                "run_validity": None,
                "certification_result": None,
                "benchmark_family_ref": launch.benchmarkFamilyRef,
                "scenario_ref": f"pending:{child['layer']}",
                "subject_manifest_ref": f"pending:{child['evaluation_run_id']}:subject",
                "environment_manifest_ref": f"pending:{child['evaluation_run_id']}:environment",
                "evaluator_bundle_ref": f"pending:{child['evaluation_run_id']}:evaluator",
                "reference_release_id": launch.referenceReleaseId if child["layer"] == "graph" else None,
                "reference_release_hash": None,
                "created_at": now,
                "completed_at": None,
                "summary": {},
                "warnings": record["warnings"],
                "stale_reason_code": None,
            }
            layer_record["content_hash"] = _projection_hash(layer_record)
            store.put_layer_run(layer_record)
    finally:
        store.close()
    return {"ok": True, "benchmarkOrchestration": stored}


@router.get("/api/knowledge/builder/benchmark-runs")
async def list_knowledge_benchmark_runs(request: Request):
    route_path = "/api/knowledge/builder/benchmark-runs"
    ctx = _authz(request, "GET", route_path)
    store = _benchmark_store()
    try:
        rows, cursor = store.list_orchestrations(
            str(ctx.organization_id), ctx.workspace_id,
            limit=_benchmark_limit(request), cursor=request.query_params.get("cursor"),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        store.close()
    return {"ok": True, "benchmarkOrchestrations": rows, "nextCursor": cursor}


@router.get("/api/knowledge/builder/benchmark-runs/{benchmark_orchestration_id}")
async def get_knowledge_benchmark_run(benchmark_orchestration_id: str, request: Request):
    route_path = "/api/knowledge/builder/benchmark-runs/{benchmark_orchestration_id}"
    ctx = _authz(request, "GET", route_path)
    store = _benchmark_store()
    try:
        row = store.get_orchestration(benchmark_orchestration_id, str(ctx.organization_id), ctx.workspace_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="benchmark orchestration not found") from exc
    finally:
        store.close()
    return {"ok": True, "benchmarkOrchestration": row}


@router.get("/api/knowledge/builder/benchmark-runs/{benchmark_orchestration_id}/layers")
async def list_knowledge_benchmark_layers(benchmark_orchestration_id: str, request: Request):
    route_path = "/api/knowledge/builder/benchmark-runs/{benchmark_orchestration_id}/layers"
    ctx = _authz(request, "GET", route_path)
    store = _benchmark_store()
    try:
        rows = store.list_layers(benchmark_orchestration_id, str(ctx.organization_id), ctx.workspace_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="benchmark orchestration not found") from exc
    finally:
        store.close()
    return {"ok": True, "benchmarkLayerRuns": rows}


def _benchmark_layer_owned(run_id: str, ctx) -> tuple[BenchmarkReadModelStore, dict]:
    store = _benchmark_store()
    try:
        return store, store.get_layer_run(run_id, str(ctx.organization_id), ctx.workspace_id)
    except KeyError as exc:
        store.close()
        raise HTTPException(status_code=404, detail="benchmark layer run not found") from exc


@router.get("/api/knowledge/builder/benchmark-layer-runs/{evaluation_run_id}")
async def get_knowledge_benchmark_layer(evaluation_run_id: str, request: Request):
    route_path = "/api/knowledge/builder/benchmark-layer-runs/{evaluation_run_id}"
    ctx = _authz(request, "GET", route_path)
    store, row = _benchmark_layer_owned(evaluation_run_id, ctx)
    try:
        canonical = None
        if row.get("evaluation_evidence_id"):
            canonical_store = EvaluationControlPlaneStore(_eos_path())
            try:
                canonical = canonical_store.get_evidence(row["evaluation_evidence_id"])
            finally:
                canonical_store.close()
    finally:
        store.close()
    return {"ok": True, "benchmarkLayerRun": row, "canonicalEvidence": canonical}


@router.get("/api/knowledge/builder/benchmark-layer-runs/{evaluation_run_id}/progress")
async def get_knowledge_benchmark_progress(evaluation_run_id: str, request: Request):
    from eos.evaluation_progress import canonical_progress
    ctx = _authz(request, "GET", "/api/knowledge/builder/benchmark-layer-runs/{evaluation_run_id}/progress")
    store, row = _benchmark_layer_owned(evaluation_run_id, ctx)
    try:
        if not row.get("evaluation_evidence_id"):
            raise HTTPException(409, "Evaluation evidence unavailable")
        canonical_store = EvaluationControlPlaneStore(_eos_path())
        try:
            progress = canonical_progress(canonical_store, run_id=evaluation_run_id,
                evidence_id=row["evaluation_evidence_id"], organization_id=str(ctx.organization_id),
                workspace_id=ctx.workspace_id)
        except KeyError as exc:
            raise HTTPException(404, "Evaluation evidence unavailable") from exc
        except ValueError as exc:
            raise HTTPException(409, "Evaluation source integrity failure") from exc
        finally:
            canonical_store.close()
    finally:
        store.close()
    return {"ok": True, "progress": progress}


@router.get("/api/knowledge/builder/evaluation-progress/{source_id}")
async def get_knowledge_diagnostic_progress(source_id: str, request: Request):
    from eos.evaluation_progress import diagnostic_progress
    _authz(request, "GET", "/api/knowledge/builder/evaluation-progress/{source_id}")
    try:
        progress = diagnostic_progress(source_id, Path(__file__).resolve().parents[3])
    except KeyError as exc:
        raise HTTPException(404, "Registered diagnostic unavailable") from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(409, "Registered diagnostic source unavailable or integrity failure") from exc
    return {"ok": True, "progress": progress}


@router.get("/api/knowledge/builder/benchmark-layer-runs/{evaluation_run_id}/cases")
async def list_knowledge_benchmark_cases(evaluation_run_id: str, request: Request):
    route_path = "/api/knowledge/builder/benchmark-layer-runs/{evaluation_run_id}/cases"
    ctx = _authz(request, "GET", route_path)
    store, _ = _benchmark_layer_owned(evaluation_run_id, ctx)
    try:
        rows, cursor = store.list_cases(
            evaluation_run_id, str(ctx.organization_id), ctx.workspace_id,
            limit=_benchmark_limit(request), cursor=request.query_params.get("cursor"),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        store.close()
    return {"ok": True, "benchmarkCaseResults": rows, "nextCursor": cursor}


@router.get("/api/knowledge/builder/benchmark-layer-runs/{evaluation_run_id}/challenge-slices")
async def list_knowledge_benchmark_challenge_slices(evaluation_run_id: str, request: Request):
    route_path = "/api/knowledge/builder/benchmark-layer-runs/{evaluation_run_id}/challenge-slices"
    ctx = _authz(request, "GET", route_path)
    store, _ = _benchmark_layer_owned(evaluation_run_id, ctx)
    try:
        rows = store.list_challenge_slices(evaluation_run_id, str(ctx.organization_id), ctx.workspace_id)
    finally:
        store.close()
    return {"ok": True, "challengeSlices": rows}


@router.get("/api/knowledge/builder/benchmark-layer-runs/{evaluation_run_id}/artifacts")
async def list_knowledge_benchmark_artifacts(evaluation_run_id: str, request: Request):
    route_path = "/api/knowledge/builder/benchmark-layer-runs/{evaluation_run_id}/artifacts"
    ctx = _authz(request, "GET", route_path)
    store, _ = _benchmark_layer_owned(evaluation_run_id, ctx)
    try:
        rows = store.list_artifacts(evaluation_run_id, str(ctx.organization_id), ctx.workspace_id)
    finally:
        store.close()
    return {"ok": True, "benchmarkArtifacts": rows}


@router.post("/api/knowledge/builder/promotions")
async def promote_knowledge_builder_runtime_semantics(request: Request):
    route_path = "/api/knowledge/builder/promotions"
    ctx = _authz(request, "POST", route_path)
    body = await _json_body(request)
    try:
        authority = RuntimeSemanticIndexStore().promote_and_activate(
            organization_id=str(ctx.organization_id),
            workspace_id=ctx.workspace_id,
            term_candidate_id=str(body.get("termCandidateId") or body.get("term_candidate_id") or ""),
            semantic_relation_candidate_ids=[
                str(item)
                for item in body.get("semanticRelationCandidateIds")
                or body.get("semantic_relation_candidate_ids")
                or []
            ],
            evaluation_run_id=str(body.get("evaluationRunId") or body.get("evaluation_run_id") or ""),
            approved_by=str(ctx.user_id),
            activated_by=str(ctx.user_id),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "authorityVersion": authority}


@router.post("/api/knowledge/builder/read-model-rebuilds")
async def rebuild_knowledge_builder_runtime_semantic_index(request: Request):
    route_path = "/api/knowledge/builder/read-model-rebuilds"
    ctx = _authz(request, "POST", route_path)
    body = await _json_body(request)
    try:
        rebuild = RuntimeSemanticIndexStore().rebuild_read_models(
            authority_version_id=str(body.get("authorityVersionId") or body.get("authority_version_id") or ""),
            rebuilt_by=str(ctx.user_id),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "readModelRebuild": rebuild}


@router.get("/api/knowledge/builder/runtime-semantic-index/{authority_version_id}")
async def get_knowledge_builder_runtime_semantic_index(authority_version_id: str, request: Request):
    route_path = "/api/knowledge/builder/runtime-semantic-index/{authority_version_id}"
    _authz(request, "GET", route_path)
    store = RuntimeSemanticIndexStore()
    authority = store.get_authority_version(authority_version_id)
    if authority is None:
        raise HTTPException(status_code=404, detail="runtime semantic authority version not found")
    return {
        "ok": True,
        "authorityVersion": authority,
        "readModel": store.get_read_model(authority_version_id),
    }


@router.get("/api/knowledge/builder/feedback-deltas")
async def list_knowledge_builder_feedback_deltas(request: Request):
    route_path = "/api/knowledge/builder/feedback-deltas"
    _authz(request, "GET", route_path)
    discovery_run_id = request.query_params.get("discoveryRunId") or request.query_params.get("discovery_run_id")
    return {
        "ok": True,
        "feedbackDeltas": RuntimeFeedbackStore().list_candidate_deltas(discovery_run_id),
    }


@router.post("/api/knowledge/builder/extraction-runs")
async def create_knowledge_builder_extraction_run(request: Request):
    route_path = "/api/knowledge/builder/extraction-runs"
    ctx = _authz(request, "POST", route_path)
    body = await _json_body(request)
    schema_version = body.get("schemaVersion") or body.get("schema_version")
    if schema_version not in {
        "knowledge_builder_extraction_run_request.v1",
        "knowledge_builder_extraction_run_request.v2",
    }:
        raise HTTPException(
            status_code=400,
            detail="knowledge_builder_extraction_run_request.v1 or .v2 required",
        )
    try:
        configured = load_knowledge_extraction_config()
        if schema_version == "knowledge_builder_extraction_run_request.v2":
            parsed = KnowledgeBuilderExtractionRunRequestV2.model_validate(body)
            provider = parsed.provider or configured.provider
            profile = parsed.profile or configured.profile
            source_id = parsed.source_id
            discovery_run_id = parsed.discovery_run_id
            document_id = parsed.document_id
            package_id = parsed.tender_package_id
            canonical_ir_ref = parsed.canonical_ir_ref
            source_identity_ref = parsed.source_identity_ref
            parser_trace_id = parsed.parser_trace_id
            provider_options = parsed.provider_options
            source_role = parsed.source_role
            workflow_kind = parsed.workflow_kind
        else:
            provider = body.get("provider", configured.provider)
            profile = body.get("profile") or configured.profile
            source_id = None
            discovery_run_id = str(body.get("discoveryRunId") or body.get("discovery_run_id") or "")
            document_id = str(body.get("documentId") or body.get("document_id") or "")
            package_id = body.get("tenderPackageId") or body.get("tender_package_id")
            canonical_ir_ref = body.get("canonicalIrRef") or body.get("canonical_ir_ref")
            source_identity_ref = body.get("sourceIdentityRef") or body.get("source_identity_ref")
            parser_trace_id = body.get("parserTraceId") or body.get("parser_trace_id")
            provider_options = body.get("providerOptions") or body.get("provider_options") or {}
            source_role = "legacy_unknown"
            workflow_kind = "legacy_candidate_assertion"
        if provider not in SUPPORTED_PROVIDERS:
            raise ValueError(
                "provider must be semantica, semantica_service, langextract, or legacy"
            )
        from knowledge_extraction.service import governed_artifact_writer_for

        artifact_root = Path(ctx.workspace_root) / "runs" / "knowledge-builder" / "artifacts"
        relationship_proposer = None
        relationship_feedback_refs: list[str] = []
        if str(provider) == "semantica" and provider_options.get("relationshipExtraction", True) is not False:
            feedback_store = SemanticFeedbackEventStore(
                ctx.workspace_root, workspace_id=str(ctx.workspace_id)
            )
            negative_examples = []
            for event in feedback_store.list_ai_extract_negative_examples(
                profile_ref=str(profile)
            ):
                corrected = event.corrected_result or {}
                negative_examples.append({
                    "feedback_id": event.feedback_id,
                    "event_hash": event.event_hash,
                    "reason_code": event.reason_code or "",
                    "concept_label": corrected.get("concept_label") or "",
                    "matched_text": corrected.get("matched_text") or "",
                    "context_text": corrected.get("context_text") or "",
                })
                relationship_feedback_refs.append(
                    f"{event.feedback_id}@{event.event_hash}"
                )
            relationship_proposer = _few_shot_relationship_proposer(
                ctx.hermes_home, negative_examples=negative_examples
            )
        extraction_canonical_ir_ref = canonical_ir_ref
        if provider_options.pop("canonicalIrHandoffOnly", False) is True:
            extraction_canonical_ir_ref = None
        run = KnowledgeExtractionService(
            db_path=_eos_path(),
            artifact_writer=governed_artifact_writer_for(artifact_root),
            relationship_proposer=relationship_proposer,
            relationship_feedback_refs=relationship_feedback_refs,
        ).start(
            organization_id=str(ctx.organization_id),
            workspace_id=ctx.workspace_id,
            source_id=source_id,
            discovery_run_id=discovery_run_id,
            document_id=document_id,
            provider=str(provider),
            profile=str(profile),
            package_id=package_id,
            canonical_ir_ref=extraction_canonical_ir_ref,
            source_identity_ref=source_identity_ref,
            parser_trace_id=parser_trace_id,
            provider_options=provider_options,
            source_role=source_role,
            workflow_kind=workflow_kind,
        )
    except (ValueError, ValidationError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    # The governed CanonicalSourceIR reference is request-validated by the
    # extraction service but is not part of the legacy extraction-run row.
    # Carry it across this immediate Source -> Schema handoff so the Studio
    # never asks the operator to upload the same document a second time.
    return {
        "ok": True,
        "extractionRun": {**run, "canonical_ir_ref": canonical_ir_ref},
    }


@router.get("/api/knowledge/builder/extraction-capability")
async def get_knowledge_builder_extraction_capability(request: Request):
    route_path = "/api/knowledge/builder/extraction-capability"
    _authz(request, "GET", route_path)
    try:
        configured = load_knowledge_extraction_config()
        service = KnowledgeExtractionService(db_path=_eos_path())
        providers = {name: service.capability(name) for name in SUPPORTED_PROVIDERS}
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    selected = providers[configured.provider]
    return {
        "ok": True,
        "schemaVersion": "knowledge_extraction_capability.v1",
        "configuredProvider": configured.provider,
        "configuredProfile": configured.profile,
        "state": selected["state"],
        "providers": providers,
        "fallbackPolicy": "explicit_new_run_only",
        "silentFallback": False,
    }


@router.get("/api/knowledge/builder/extraction-runs")
async def list_knowledge_builder_extraction_runs(request: Request):
    route_path = "/api/knowledge/builder/extraction-runs"
    ctx = _authz(request, "GET", route_path)
    store = _extraction_store()
    try:
        limit = int(request.query_params.get("limit") or 100)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="limit must be an integer") from exc
    try:
        runs = store.list_extraction_runs(
            organization_id=str(ctx.organization_id),
            workspace_id=str(ctx.workspace_id) if ctx.workspace_id else None,
            source_id=request.query_params.get("sourceId") or request.query_params.get("source_id"),
            limit=limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "schemaVersion": "knowledge_builder_extraction_runs_response.v1", "extractionRuns": runs}


def _graph_delta_preview(assertion, store: KnowledgeExtractionStore) -> dict:
    grounding = HumanGroundingStore(_eos_path(), graph_backend=_CandidateOnlyPreviewBackend())
    graph = grounding.get_candidate_graph(assertion.candidate_graph_id)
    owner = grounding.get_candidate_graph_owner(assertion.candidate_graph_id)
    base = owner.get("allowed_base_graph_version") if owner else None
    if not base:
        return {"available": False, "reason": "no_materialized_base"}
    try:
        base_release = grounding.get_graph_release(
            base, organization_id=owner.get("organization_id"),
            workspace_id=owner.get("workspace_id"),
        )
    except ValueError:
        return {"available": False, "reason": "no_materialized_base"}
    normalized = assertion.normalized_assertion.model_dump(mode="json")
    anchors = tuple(anchor.anchor_id for anchor in assertion.source_anchors)
    subject = normalized.get("subject")
    obj = normalized.get("object")
    predicate = normalized.get("predicate")
    operations = []
    if subject and obj and predicate and normalized.get("candidate_relation_id"):
        subject_id = f"candidate_subject_{assertion.assertion_id}"
        object_id = f"candidate_object_{assertion.assertion_id}"
        for ordinal, (node_id, value) in enumerate(((subject_id, subject), (object_id, obj)), 1):
            operations.append({"operation_id": f"preview_node_{ordinal}", "action": "node_add",
                               "target_refs": [node_id], "after": {"node_id": node_id, **value},
                               "source_anchor_refs": list(anchors)})
        operations.append({"operation_id": "preview_relation_1", "action": "edge_add",
                           "target_refs": [normalized["candidate_relation_id"]],
                           "after": {"edge_id": normalized["candidate_relation_id"],
                                     "source_node_id": subject_id, "target_node_id": object_id,
                                     "relation_type": predicate},
                           "source_anchor_refs": list(anchors)})
    elif graph.nodes:
        node = graph.nodes[0]
        operations.append({"operation_id": "preview_node_1", "action": "node_add",
                           "target_refs": [node.node_id], "after": node.model_dump(mode="json"),
                           "source_anchor_refs": list(anchors)})
    else:
        return {"available": False, "reason": "candidate_topology_unavailable"}
    base_release_hash = base_release.get("release_hash") or content_hash(base_release)
    projection = {
        "candidate_graph_id": assertion.candidate_graph_id,
        "assertion_id": assertion.assertion_id,
        "candidate_relation_id": normalized.get("candidate_relation_id"),
        "subject_candidate_node_id": f"candidate_subject_{assertion.assertion_id}" if subject else None,
        "object_candidate_node_id": f"candidate_object_{assertion.assertion_id}" if obj else None,
        "from_graph_version": base,
        "base_release_hash": base_release_hash,
        "transition": "allocate_next_materialized_graph_version",
        "evidence_anchor_refs": list(anchors),
        "operations": operations,
    }
    preview_seed = content_hash(projection)
    delta_payload = {"graph_delta_id": f"preview_{preview_seed[7:31]}",
                     "candidate_graph_id": assertion.candidate_graph_id,
                     "from_graph_version": base,
                     "to_graph_version": f"KG_v{int(preview_seed[7:23], 16) + 1}",
                     "human_label_event_id": "preview_validation_only",
                     "justification": "Pending reviewer justification.", "operations": operations}
    validated = GraphDelta.model_validate(delta_payload).model_dump(mode="json")
    validated.pop("human_label_event_id")
    return {"available": True, "schemaVersion": "graph_delta_preview_projection.v1",
            "graphDelta": validated, "previewHash": preview_seed,
            "stalenessToken": preview_seed, "baseGraphVersion": base,
            "baseReleaseHash": base_release_hash,
            "transition": "allocate_next_materialized_graph_version",
            "ownedTargetRefs": [item for item in (
                projection["subject_candidate_node_id"], projection["object_candidate_node_id"],
                projection["candidate_relation_id"],
            ) if item],
            "evidenceAnchorRefs": list(anchors)}


class _CandidateOnlyPreviewBackend:
    def materialize(self, _release):
        raise RuntimeError("preview backend cannot materialize")


def _release_graph_backend():
    if os.environ.get("SEMANTIER_F10_FIXTURE") == "1":
        return InMemoryGraphBackend()
    host = os.environ.get("SEMANTIER_FALKORDB_HOST")
    if host:
        return FalkorDBGraphBackend.connect(
            host=host,
            port=int(os.environ.get("SEMANTIER_FALKORDB_PORT") or 6379),
        )
    lite_path = os.environ.get("SEMANTIER_FALKORDB_LITE_PATH")
    if not lite_path:
        lite_path = str(_eos_path().with_name("falkordb-lite.db"))
    return FalkorDBGraphBackend.open_lite(lite_path)


def _persist_accepted_release_origin(
    *,
    release: dict,
    ctx,
    release_lineage: AcceptedGraphReleaseLineage | AcceptedGraphReleaseLineageV2 | None = None,
    authority_state: str = "authoritative",
) -> str:
    graph_version = str(release["graph_version"])
    extraction_store = _extraction_store()
    owner = extraction_store.get_candidate_graph_owner(
        str(release.get("candidate_graph_id") or "")
    )
    lineage_id = str((owner or {}).get("source_id") or "").strip()
    if not lineage_id:
        raise ValueError("accepted graph release is missing governed lineage ownership")
    if release_lineage is not None:
        if isinstance(release_lineage, AcceptedGraphReleaseLineageV2):
            grounding = HumanGroundingStore(
                _eos_path(), graph_backend=_CandidateOnlyPreviewBackend()
            )
            origin_id = verify_exact_release_origin_reservation(
                grounding._conn, graph_version=graph_version,
                authority_state="authoritative",
            )
            origin = grounding._conn.execute(
                hosted.sql_query("SELECT source_id,release_lineage_json FROM contextgraph_artifact_origins "
                "WHERE origin_id=?", 'SELECT source_id,release_lineage_json FROM contextgraph_artifact_origins WHERE origin_id=%s'), (origin_id,),
            ).fetchone()
            if AcceptedGraphReleaseLineageV2.parse_unverified(
                json.loads(origin["release_lineage_json"])
            ) != release_lineage:
                raise ValueError("accepted v2 release origin lineage mismatch")
            return str(origin["source_id"])
        verified = extraction_store.get_release_artifact_origin_lineage(
            graph_version=graph_version,
            organization_id=str(ctx.organization_id),
            workspace_id=str(ctx.workspace_id) if ctx.workspace_id else "",
            workspace_root=ctx.workspace_root,
            require_materialized_checkpoint=False,
        )
        if (
            verified["provenance_status"] != "validated_v3_retry"
            or AcceptedGraphReleaseLineage.from_dict(verified["lineage"])
            != release_lineage
        ):
            raise ValueError("accepted release origin does not match exact lineage")
        return lineage_id
    extraction_store.record_release_artifact_origin(
        graph_version=graph_version,
        candidate_graph_id=str(release.get("candidate_graph_id") or ""),
        source_id=lineage_id,
        lineage_id=lineage_id,
        extraction_run_id=str(owner.get("extraction_run_id") or ""),
        organization_id=str(ctx.organization_id),
        workspace_id=str(ctx.workspace_id),
        source_identity_ref=owner.get("source_identity_ref"),
        release_hash=str(release["release_hash"]),
        release_lineage=release_lineage,
        authority_state=authority_state,
    )
    return lineage_id


def _checkpoint_accepted_release_artifact(
    *,
    release: dict,
    ctx,
    release_lineage: AcceptedGraphReleaseLineage | AcceptedGraphReleaseLineageV2 | None = None,
    semantic_bundle: dict | None = None,
) -> dict:
    if semantic_bundle is not None:
        if semantic_bundle.get("release") != release:
            raise ValueError(
                "semantic bundle release differs from checkpoint release"
            )
        AcceptedGraphReleaseV2.parse_unverified(release)
    repository = ContextGraphArtifactRepository(
        ctx.workspace_root,
        organization_id=str(ctx.organization_id),
        workspace_id=str(ctx.workspace_id),
    )
    graph_version = str(release["graph_version"])
    graph = (
        native_graph_from_accepted_release_bundle(semantic_bundle)
        if semantic_bundle is not None
        else native_graph_from_accepted_release(release)
    )
    if semantic_bundle is not None:
        validate_accepted_release_conformance(
            bundle=semantic_bundle,
            graph=graph,
        )
    source_manifest_hash = accepted_release_source_manifest_hash(
        release,
        release_lineage=release_lineage,
    )
    lineage_id = _persist_accepted_release_origin(
        release=release,
        ctx=ctx,
        release_lineage=release_lineage,
    )
    try:
        repository.create_checkpoint(
            graph,
            lineage_id=lineage_id,
            graph_id=GRAPH_ID,
            graph_version=graph_version,
            artifact_kind="authoritative",
            run_mode="authoritative",
            semantica_commit=(
                release_lineage.compiler_semantica_commit
                if release_lineage is not None
                else SEMANTICA_COMMIT
            ),
            source_manifest_hash=source_manifest_hash,
            grounding_manifest_hash=str(release["release_hash"]),
            governance_ref=f"AcceptedGraphRelease:{graph_version}",
            created_by_user_id=str(ctx.user_id),
            artifact_id=graph_version,
        )
    except ArtifactConflict:
        pass
    metadata = verify_accepted_release_checkpoint(
        repository,
        graph=graph,
        release=release,
        lineage_id=lineage_id,
        release_lineage=release_lineage,
    )
    return metadata


def _verify_recovered_exact_release_lineage(
    *,
    release: dict,
    ctx,
    expected_lineage: AcceptedGraphReleaseLineage,
) -> AcceptedGraphReleaseLineage:
    """Verify an existing exact release without reconstructing missing lineage."""
    persisted = _extraction_store().get_release_artifact_origin_lineage(
        graph_version=str(release["graph_version"]),
        organization_id=str(ctx.organization_id),
        workspace_id=str(ctx.workspace_id) if ctx.workspace_id else "",
        workspace_root=ctx.workspace_root,
        require_materialized_checkpoint=False,
    )
    if persisted["provenance_status"] != "validated_v3_retry":
        raise ValueError("existing accepted release does not have exact v3 lineage")
    actual = AcceptedGraphReleaseLineage.from_dict(persisted["lineage"])
    if actual != expected_lineage:
        raise ValueError("persisted exact release lineage does not match retry pins")
    return actual


def _resolve_v3_release_inputs(
    *,
    body: dict,
    ctx,
    candidate_graph_id: str,
) -> tuple[
    CorpusGraphSchemaCandidate,
    SchemaDecision,
    ExtractResult,
    SensitiveTermCandidateGraph,
]:
    """Resolve and validate v2 release inputs before materialization."""
    allowed = {
        "schemaVersion",
        "schema_version",
        "humanEventId",
        "human_event_id",
        "extractRunRef",
        "extract_run_ref",
        "extractResultHash",
        "extract_result_hash",
    }
    if set(body) - allowed:
        raise ValueError("accepted_graph_release_request.v2 contains unknown fields")
    for camel, snake in (
        ("schemaVersion", "schema_version"),
        ("humanEventId", "human_event_id"),
        ("extractRunRef", "extract_run_ref"),
        ("extractResultHash", "extract_result_hash"),
    ):
        if camel in body and snake in body:
            raise ValueError(
                f"accepted_graph_release_request.v2 contains duplicate aliases: "
                f"{camel}, {snake}"
            )
    extract_run_ref = str(
        body.get("extractRunRef") or body.get("extract_run_ref") or ""
    ).strip()
    extract_result_hash = str(
        body.get("extractResultHash") or body.get("extract_result_hash") or ""
    ).strip()
    if re.fullmatch(r"extract-run-[A-Za-z0-9._-]+", extract_run_ref) is None:
        raise ValueError("extractRunRef is invalid")
    if re.fullmatch(r"sha256:[0-9a-f]{64}", extract_result_hash) is None:
        raise ValueError("extractResultHash must be an exact sha256 ref")

    requested_workspace_root = Path(ctx.workspace_root).resolve()
    workspace_id = str(ctx.workspace_id)
    workspace_root = (
        requested_workspace_root.parent.parent
        if requested_workspace_root.name == workspace_id
        and requested_workspace_root.parent.name == "workspaces"
        else requested_workspace_root
    )
    workspace_dir = workspace_root / "workspaces" / workspace_id
    runs_root = (workspace_dir / "runs").resolve()
    result_path = (
        runs_root / extract_run_ref / "artifacts" / "extract_result.v1.json"
    ).resolve()
    if runs_root not in result_path.parents or not result_path.is_file():
        raise ValueError("governed ExtractResult not found")
    extract_result = ExtractResult.from_dict(
        json.loads(result_path.read_text(encoding="utf-8"))
    )
    if (
        extract_result.extract_run_id != extract_run_ref
        or extract_result.extract_result_hash != extract_result_hash
    ):
        raise ValueError("ExtractResult exact pin mismatch")

    decision_ref = str(extract_result.schema_binding.get("schema_decision_ref") or "")
    profile_registry = CorpusGraphSchemaProfileRegistry(
        repository_root=Path(__file__).resolve().parents[3]
    )
    taxonomy_service = KnowledgeRelationTaxonomyService(
        repository_root=Path(__file__).resolve().parents[3],
        profile_registry=profile_registry,
        profile_relation_type_proposal_store=ProfileRelationTypeProposalStore(
            workspace_root,
            workspace_id=workspace_id,
            organization_id=str(ctx.organization_id),
        ),
    )
    decision = SchemaDecisionStore(
        workspace_root, workspace_id=workspace_id
    ).get_contract(
        decision_ref, taxonomy_service=taxonomy_service
    )
    if decision.schema_version != "schema_decision.v3":
        raise ValueError("new release requires Candidate.v4/Decision.v3")
    if decision.schema_decision_hash != extract_result.schema_binding.get(
        "schema_decision_hash"
    ):
        raise ValueError("SchemaDecision exact pin mismatch")
    candidate = CorpusGraphSchemaCandidate.from_dict(
        CorpusGraphSchemaCandidateStore(
            workspace_root, workspace_id=workspace_id
        ).get_by_hash(str(extract_result.schema_binding.get("candidate_hash") or ""))
    )

    source_registry = KnowledgeBuilderStore(_eos_path(), workspace_root=workspace_root)
    try:
        intent_binding = dict(candidate.graph_intent_binding or {})
        graph_intent_store = GraphIntentBriefStore(
            workspace_root,
            organization_id=str(ctx.organization_id),
            workspace_id=workspace_id,
            source_registry=source_registry,
        )
        graph_intent = graph_intent_store.get(
            str(intent_binding.get("graph_intent_ref") or ""),
            graph_intent_hash=str(intent_binding.get("graph_intent_hash") or ""),
        )
        source_identity, _canonical_source = graph_intent_store.resolve_evidence_source(
            source_identity_ref=graph_intent.source_binding.source_identity_ref,
            canonical_source_ir_ref=graph_intent.source_binding.canonical_source_ir_ref,
            canonical_source_ir_hash=graph_intent.source_binding.canonical_source_ir_hash,
        )
        graph_intent_store.verify_source_binding(
            graph_intent.source_binding,
            source_identity=source_identity,
        )
    finally:
        source_registry.close()
    if (
        graph_intent.review.state != intent_binding.get("review_state")
        or graph_intent.source_binding.as_dict() != candidate.source_binding
    ):
        raise ValueError("GraphIntent exact pin mismatch")
    extraction_store = _extraction_store()
    owner = extraction_store.get_candidate_graph_owner(candidate_graph_id)
    if owner is None:
        raise ValueError("candidate graph not found")
    source = dict(candidate.source_binding or {})
    candidate_graph = HumanGroundingStore(
        _eos_path(), graph_backend=_CandidateOnlyPreviewBackend()
    ).get_candidate_graph(candidate_graph_id)
    exact_owner_pins = {
        "organization": (owner.get("organization_id"), str(ctx.organization_id)),
        "workspace": (owner.get("workspace_id"), workspace_id),
        "source identity": (
            owner.get("source_identity_ref"),
            source.get("source_identity_ref"),
        ),
        "materialization extraction run": (
            owner.get("extraction_run_id"),
            candidate_graph.extraction_run_id,
        ),
        "materialization source": (
            owner.get("materialization_source_hash"),
            candidate_graph.source_document_hash,
        ),
    }
    for label, (actual, expected) in exact_owner_pins.items():
        if actual != expected:
            raise ValueError(f"candidate graph owner {label} pin mismatch")
    AcceptedGraphReleaseLineageV2.validate_pre_release_inputs(
        candidate=candidate, decision=decision, extract_result=extract_result,
        candidate_graph=candidate_graph, candidate_graph_id=candidate_graph_id,
    )
    return candidate, decision, extract_result, candidate_graph


def _materialize_v2_release_from_exact_inputs(
    *,
    lineage_inputs: tuple[
        CorpusGraphSchemaCandidate, SchemaDecision, ExtractResult,
        SensitiveTermCandidateGraph,
    ],
    candidate_graph_id: str,
    human_event_ids: tuple[str, ...],
    ctx,
) -> dict:
    """Route a v4/v3 request through the sole sealed v2 materializer."""

    candidate, decision, extract_result, _candidate_graph = lineage_inputs
    if (
        candidate.schema_version != "corpus_graph_schema_candidate.v4"
        or decision.schema_version != "schema_decision.v3"
    ):
        raise ValueError("v2 release materialization requires Candidate.v4/Decision.v3")
    requested = Path(ctx.workspace_root).resolve()
    workspace_id = str(ctx.workspace_id)
    runtime_root = (
        requested.parent.parent
        if requested.name == workspace_id and requested.parent.name == "workspaces"
        else requested
    )
    source = dict(candidate.source_binding or {})
    intent = dict(candidate.graph_intent_binding or {})
    service = AcceptedSemanticGraphMaterializationService(
        runtime_root=runtime_root,
        workspace_root=runtime_root,
        db_path=_eos_path(),
        organization_id=str(ctx.organization_id),
        workspace_id=workspace_id,
        graph_backend=_release_graph_backend(),
    )
    return service.materialize_accepted_graph_release_v2(
        ResolveReviewedPatternInputsCommand(
            materialization=ResolveMaterializationInputsCommand(
                organization_id=str(ctx.organization_id), workspace_id=workspace_id,
                source_identity_ref=str(source["source_identity_ref"]),
                lineage_id=str(source["lineage_id"]),
                graph_intent_ref=str(intent["graph_intent_ref"]),
                graph_intent_hash=str(intent["graph_intent_hash"]),
                schema_candidate_ref=candidate.candidate_id,
                schema_candidate_hash=candidate.candidate_hash,
                schema_decision_ref=decision.schema_decision_id,
                schema_decision_hash=decision.schema_decision_hash,
                extract_result_ref=extract_result.extract_run_id,
                extract_result_hash=extract_result.extract_result_hash,
            ),
            candidate_graph_id=candidate_graph_id,
            human_label_event_ids=human_event_ids,
        )
    )


def _retry_v2_release_from_exact_lineage(
    *, graph_version: str, store: HumanGroundingStore, graph_backend, ctx,
) -> dict:
    """Resume one failed v2 release from its immutable persisted command pins."""

    workspace_id = str(ctx.workspace_id)
    organization_id = str(ctx.organization_id)
    release, _status = store.get_graph_release_projection(
        graph_version,
        organization_id=organization_id,
        workspace_id=workspace_id,
    )
    origin_rows = store._conn.execute(
        hosted.sql_query("SELECT release_lineage_json,release_lineage_hash,authority_state "
        "FROM contextgraph_artifact_origins WHERE artifact_id=? "
        "AND organization_id=? AND workspace_id=? "
        "AND relation_kind='accepted_release' ORDER BY created_at,origin_id", "SELECT release_lineage_json,release_lineage_hash,authority_state FROM contextgraph_artifact_origins WHERE artifact_id=%s AND organization_id=%s AND workspace_id=%s AND relation_kind='accepted_release' ORDER BY created_at,origin_id"),
        (graph_version, organization_id, workspace_id),
    ).fetchall()
    if len(origin_rows) != 1:
        raise ValueError("exact v2 release origin lineage is unavailable")
    origin = origin_rows[0]
    if origin["authority_state"] not in {"candidate", "authoritative"}:
        raise ValueError("exact v2 release origin authority state is invalid")
    try:
        lineage = AcceptedGraphReleaseLineageV2.parse_unverified(
            json.loads(origin["release_lineage_json"])
        )
    except (TypeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("exact v2 release lineage metadata is invalid") from exc
    if origin["release_lineage_hash"] != lineage.lineage_hash:
        raise ValueError("exact v2 release lineage hash mismatch")
    accepted_release = AcceptedGraphReleaseV2.parse_unverified(release)
    if (
        lineage.accepted_graph_release_ref != accepted_release.graph_version
        or lineage.accepted_graph_release_hash != accepted_release.release_hash
        or lineage.candidate_graph_id != accepted_release.candidate_graph_id
        or (lineage.organization_id, lineage.workspace_id)
        != (organization_id, workspace_id)
    ):
        raise ValueError("exact v2 release lineage does not match stored release")
    requested = Path(ctx.workspace_root).resolve()
    runtime_root = (
        requested.parent.parent
        if requested.name == workspace_id and requested.parent.name == "workspaces"
        else requested
    )
    service = AcceptedSemanticGraphMaterializationService(
        runtime_root=runtime_root,
        workspace_root=runtime_root,
        db_path=_eos_path(),
        organization_id=organization_id,
        workspace_id=workspace_id,
        graph_backend=graph_backend,
    )
    bundle = service.materialize_accepted_graph_release_v2(
        ResolveReviewedPatternInputsCommand(
            materialization=ResolveMaterializationInputsCommand(
                organization_id=organization_id,
                workspace_id=workspace_id,
                source_identity_ref=lineage.source_identity_ref,
                lineage_id=lineage.lineage_id,
                graph_intent_ref=lineage.graph_intent_ref,
                graph_intent_hash=lineage.graph_intent_hash,
                schema_candidate_ref=lineage.schema_candidate_ref,
                schema_candidate_hash=lineage.schema_candidate_hash,
                schema_decision_ref=lineage.schema_decision_ref,
                schema_decision_hash=lineage.schema_decision_hash,
                extract_result_ref=lineage.schema_validation_extract_run_ref,
                extract_result_hash=lineage.schema_validation_extract_result_hash,
            ),
            candidate_graph_id=lineage.candidate_graph_id,
            human_label_event_ids=tuple(accepted_release.human_label_event_ids),
        )
    )
    if bundle["release"]["graph_version"] != graph_version:
        raise ValueError("v2 retry allocated a different release identity")
    return bundle


def _canonical_edit_delta(
    *, assertion, edited_assertion: dict, event_id: str, justification: str,
    source_anchor_refs: tuple[str, ...], to_graph_version: str,
) -> GraphDelta:
    grounding = HumanGroundingStore(_eos_path(), graph_backend=_CandidateOnlyPreviewBackend())
    graph = grounding.get_candidate_graph(assertion.candidate_graph_id)
    # Candidate graph builders may use provider-native node identifiers (for
    # example ``concept_entity_*``) rather than the assertion store's legacy
    # ``candidate_node_<assertion>`` convention. Bind the edit to the node
    # carrying the assertion's evidence anchor, preserving the canonical
    # graph identity instead of inventing a target that is absent.
    assertion_anchor_ids = {anchor.anchor_id for anchor in assertion.source_anchors}
    node = next(
        (item for item in graph.nodes
         if assertion_anchor_ids.intersection(anchor.anchor_id for anchor in item.source_anchors)),
        None,
    )
    target_id = node.node_id if node is not None else f"candidate_node_{assertion.assertion_id}"
    if node is None:
        raise ValueError("edited assertion candidate node is absent from candidate graph")
    label = next((
        str(edited_assertion.get(field) or "").strip()
        for field in ("subject_text", "object_text", "predicate_text")
        if str(edited_assertion.get(field) or "").strip()
    ), "")
    if not label:
        raise ValueError("edited assertion must contain subject_text, object_text, or predicate_text")
    before = node.model_dump(mode="json")
    after = {
        **before,
        "canonical_label": label,
        "normalized_value": label.casefold(),
        "rationale": "Reviewer corrected the normalized assertion.",
    }
    owner = grounding.get_candidate_graph_owner(assertion.candidate_graph_id) or {}
    return GraphDelta.model_validate({
        "graph_delta_id": f"delta_{uuid4().hex}",
        "candidate_graph_id": assertion.candidate_graph_id,
        "from_graph_version": owner.get("allowed_base_graph_version"),
        "to_graph_version": to_graph_version,
        "human_label_event_id": event_id,
        "justification": justification,
        "operations": [{
            "operation_id": f"edit_{assertion.assertion_id}",
            "action": "node_edit",
            "target_refs": [target_id],
            "before": before,
            "after": after,
            "source_anchor_refs": list(source_anchor_refs),
        }],
    })


@router.get("/api/knowledge/builder/assertion-candidates/{assertion_id}/graph-delta-preview")
async def get_knowledge_builder_graph_delta_preview(assertion_id: str, request: Request):
    route_path = "/api/knowledge/builder/assertion-candidates/{assertion_id}/graph-delta-preview"
    ctx = _authz(request, "GET", route_path)
    store = _extraction_store()
    _require_owner(store.get_assertion_owner(assertion_id), str(ctx.organization_id),
                   str(ctx.workspace_id) if ctx.workspace_id else None, label="assertion candidate")
    assertion = store.get_assertion(assertion_id)
    if assertion is None:
        raise HTTPException(status_code=404, detail="assertion candidate not found")
    return {"ok": True, **_graph_delta_preview(assertion, store)}


@router.get("/api/knowledge/builder/graph-releases/{graph_version}/compare")
async def compare_knowledge_builder_graph_release(graph_version: str, base: str, request: Request):
    route_path = "/api/knowledge/builder/graph-releases/{graph_version}/compare"
    ctx = _authz(request, "GET", route_path)
    store = HumanGroundingStore(_eos_path(), graph_backend=_release_graph_backend())
    try:
        return store.compare_graph_versions(
            base,
            graph_version,
            organization_id=str(ctx.organization_id),
            workspace_id=str(ctx.workspace_id) if ctx.workspace_id else None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/api/knowledge/builder/graph-releases")
async def list_knowledge_builder_graph_releases(request: Request):
    route_path = "/api/knowledge/builder/graph-releases"
    ctx = _authz(request, "GET", route_path)
    store = HumanGroundingStore(_eos_path(), graph_backend=None)
    releases = store.list_reference_graph_versions(
        organization_id=str(ctx.organization_id),
        workspace_id=str(ctx.workspace_id) if ctx.workspace_id else None,
    )

    def graph_version_key(item: dict) -> tuple[int, str]:
        graph_version = str(item.get("graph_version") or "")
        match = re.search(r"(\d+)$", graph_version)
        return (int(match.group(1)) if match else -1, graph_version)

    return {
        "ok": True,
        "schema_version": "workspace_graph_release_inventory.v1",
        "graph_releases": sorted(releases, key=graph_version_key, reverse=True),
    }


@router.post("/api/knowledge/builder/graph-releases/batch-delete")
async def batch_delete_knowledge_builder_graph_releases(request: Request):
    route_path = "/api/knowledge/builder/graph-releases/batch-delete"
    ctx = _authz(request, "POST", route_path)
    payload = await request.json()
    graph_versions = payload.get("graph_versions") if isinstance(payload, dict) else None
    if not isinstance(graph_versions, list) or not all(
        isinstance(item, str) for item in graph_versions
    ):
        raise HTTPException(status_code=400, detail="graph_versions must be a list of strings")
    store = HumanGroundingStore(_eos_path(), graph_backend=None)
    return store.batch_delete_empty_graph_releases(
        graph_versions,
        organization_id=str(ctx.organization_id),
        workspace_id=str(ctx.workspace_id) if ctx.workspace_id else None,
    )


@router.get("/api/knowledge/builder/graph-releases/{graph_version}/lineage")
async def get_knowledge_builder_graph_release_lineage(
    graph_version: str, request: Request, expected_release_hash: str | None = None,
):
    """Read the existing immutable lineage through full materialized-bundle validation."""
    route_path = "/api/knowledge/builder/graph-releases/{graph_version}/lineage"
    ctx = _authz(request, "GET", route_path)
    store = HumanGroundingStore(_eos_path(), graph_backend=None)
    organization_id = str(ctx.organization_id)
    workspace_id = str(ctx.workspace_id or "")
    try:
        record = store.read_materialized_release_record(
            graph_version, organization_id=organization_id, workspace_id=workspace_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Graph release not found") from exc
    if expected_release_hash is not None and record["content_hash"] != expected_release_hash:
        raise HTTPException(status_code=409, detail="Graph release hash mismatch")
    try:
        bundle = MaterializedAcceptedGraphBundleReader(
            store=store, organization_id=organization_id, workspace_id=workspace_id,
        ).read(record["release"])
    except (ValueError, MaterializedAcceptedGraphBundleFailure) as exc:
        raise HTTPException(status_code=409, detail="Graph release lineage is unavailable") from exc
    return bundle["lineage"]


@router.get("/api/knowledge/builder/graph-releases/{graph_version}/projection")
async def get_knowledge_builder_graph_release_projection(graph_version: str, request: Request):
    route_path = "/api/knowledge/builder/graph-releases/{graph_version}/projection"
    ctx = _authz(request, "GET", route_path)
    store = HumanGroundingStore(_eos_path(), graph_backend=None)
    try:
        record = store.read_materialized_release_record(
            graph_version,
            organization_id=str(ctx.organization_id),
            workspace_id=str(ctx.workspace_id) if ctx.workspace_id else "",
        )
    except ValueError:
        return _semantic_projection_error(
            status_code=404,
            code="GRAPH_RELEASE_NOT_FOUND",
            message="Graph release not found",
            retryable=False,
        )
    if record.get("retirement_state", "active") != "active":
        return _semantic_projection_error(
            status_code=404,
            code="GRAPH_RELEASE_NOT_FOUND",
            message="Graph release not found",
            retryable=False,
        )
    lifecycle_status = record["lifecycle_status"]
    if lifecycle_status == "pending":
        return _semantic_projection_error(
            status_code=409,
            code="GRAPH_RELEASE_NOT_MATERIALIZED",
            message="Graph release is not materialized",
            retryable=True,
        )
    if lifecycle_status == "failed":
        return _semantic_projection_error(
            status_code=409,
            code="GRAPH_RELEASE_MATERIALIZATION_FAILED",
            message="Graph release materialization failed",
            retryable=False,
        )
    if record["lineage_contract"] != "exact_v4_v3":
        return _semantic_projection_error(
            status_code=422,
            code="GRAPH_RELEASE_VERSION_UNSUPPORTED",
            message="Graph release version is unsupported",
            retryable=False,
        )
    try:
        projection = _semantic_graph_projection_service(store=store, ctx=ctx).project(
            record["release"]
        )
    except SemanticGraphProjectionFailure as exc:
        return _semantic_projection_error(
            status_code=exc.status_code,
            code=exc.code,
            message=exc.message,
            retryable=exc.retryable,
        )
    except MaterializedAcceptedGraphBundleFailure as exc:
        return _semantic_projection_error(
            status_code=409, code=exc.code,
            message="Graph release bundle validation failed", retryable=False,
        )
    except GraphLayoutLimitExceeded:
        return _semantic_projection_error(
            status_code=413,
            code="GRAPH_RELEASE_LIMIT_EXCEEDED",
            message="Graph release exceeds projection limits",
            retryable=False,
        )
    except ValueError:
        return _semantic_projection_error(
            status_code=409,
            code="GRAPH_RELEASE_INTEGRITY_MISMATCH",
            message="Graph release integrity validation failed",
            retryable=False,
        )
    return projection.as_dict()


@router.post(
    "/api/knowledge/builder/graph-releases/{graph_version}/projection/coherence-analysis"
)
async def create_knowledge_builder_graph_release_coherence_analysis(
    graph_version: str, request: Request,
):
    route_path = (
        "/api/knowledge/builder/graph-releases/"
        "{graph_version}/projection/coherence-analysis"
    )
    ctx = _authz(request, "POST", route_path)
    raw_body = await request.body()
    try:
        body = json.loads(raw_body) if raw_body else {}
    except (TypeError, ValueError):
        body = None
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON object body required")
    policy = load_default_graph_coherence_policy()
    requested_policy_ref = body.get("analyzer_policy_ref", policy.policy_ref)
    requested_policy_hash = body.get("analyzer_policy_hash", policy.policy_hash)
    if (
        requested_policy_ref != policy.policy_ref
        or requested_policy_hash != policy.policy_hash
    ):
        return _semantic_projection_error(
            status_code=422, code="COHERENCE_POLICY_UNSUPPORTED",
            message="Coherence analyzer policy is unsupported", retryable=False,
        )
    store = HumanGroundingStore(_eos_path(), graph_backend=None)
    try:
        record = store.read_materialized_release_record(
            graph_version,
            organization_id=str(ctx.organization_id),
            workspace_id=str(ctx.workspace_id) if ctx.workspace_id else "",
        )
    except ValueError:
        return _semantic_projection_error(
            status_code=404, code="GRAPH_RELEASE_NOT_FOUND",
            message="Graph release not found", retryable=False,
        )
    if record.get("retirement_state", "active") != "active":
        return _semantic_projection_error(
            status_code=404, code="GRAPH_RELEASE_NOT_FOUND",
            message="Graph release not found", retryable=False,
        )
    lifecycle_status = record["lifecycle_status"]
    if lifecycle_status == "pending":
        return _semantic_projection_error(
            status_code=409, code="GRAPH_RELEASE_NOT_MATERIALIZED",
            message="Graph release is not materialized", retryable=True,
        )
    if lifecycle_status == "failed":
        return _semantic_projection_error(
            status_code=409, code="GRAPH_RELEASE_MATERIALIZATION_FAILED",
            message="Graph release materialization failed", retryable=False,
        )
    if record["lineage_contract"] != "exact_v4_v3":
        return _semantic_projection_error(
            status_code=422, code="GRAPH_RELEASE_VERSION_UNSUPPORTED",
            message="Graph release version is unsupported", retryable=False,
        )
    try:
        projection = _semantic_graph_projection_service(store=store, ctx=ctx).project(
            record["release"]
        )
        analysis = _graph_coherence_analyzer().analyze(
            accepted_release=record["release"], projection=projection,
        )
        _graph_coherence_artifact_store(ctx=ctx).persist(analysis)
    except SemanticGraphProjectionFailure as exc:
        return _semantic_projection_error(
            status_code=exc.status_code, code=exc.code,
            message=exc.message, retryable=exc.retryable,
        )
    except MaterializedAcceptedGraphBundleFailure as exc:
        return _semantic_projection_error(
            status_code=409, code=exc.code,
            message="Graph release bundle validation failed", retryable=False,
        )
    except (GraphLayoutLimitExceeded, GraphCoherenceLimitExceeded):
        return _semantic_projection_error(
            status_code=413, code="GRAPH_RELEASE_LIMIT_EXCEEDED",
            message="Graph release exceeds analysis limits", retryable=False,
        )
    except GraphCoherenceArtifactConflict:
        return _semantic_projection_error(
            status_code=409, code="COHERENCE_ARTIFACT_CONFLICT",
            message="Coherence artifact identity conflicts with persisted bytes",
            retryable=False,
        )
    except ValueError:
        return _semantic_projection_error(
            status_code=409, code="GRAPH_RELEASE_INTEGRITY_MISMATCH",
            message="Graph release integrity validation failed", retryable=False,
        )
    return analysis.as_dict()


def _semantic_graph_projection_service(*, store: HumanGroundingStore, ctx) -> SemanticGraphProjectionService:
    profile = load_default_semantic_layout_profile()
    return SemanticGraphProjectionService(
        bundle_reader=MaterializedAcceptedGraphBundleReader(
            store=store,
            organization_id=str(ctx.organization_id),
            workspace_id=str(ctx.workspace_id) if ctx.workspace_id else "",
        ),
        taxonomy_service=KnowledgeRelationTaxonomyService(),
        layout_engine=SemanticLayoutEngine(profile),
        schema_decision_store=SchemaDecisionStore(
            ctx.hermes_home, workspace_id=str(ctx.workspace_id)
        ),
    )


def _graph_coherence_analyzer() -> GraphCoherenceAnalyzer:
    return GraphCoherenceAnalyzer(load_default_graph_coherence_policy())


def _graph_coherence_artifact_store(*, ctx) -> GraphCoherenceArtifactStore:
    return GraphCoherenceArtifactStore(
        workspace_root=ctx.hermes_home,
        organization_id=str(ctx.organization_id),
        workspace_id=str(ctx.workspace_id),
    )


def _semantic_projection_error(
    *, status_code: int, code: str, message: str, retryable: bool
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "ok": False,
            "schema_version": "semantic_graph_projection_error.v1",
            "error": {
                "code": code,
                "message": message,
                "retryable": retryable,
            },
        },
    )


@router.get("/api/knowledge/builder/extraction-runs/{run_id}")
async def get_knowledge_builder_extraction_run(run_id: str, request: Request):
    route_path = "/api/knowledge/builder/extraction-runs/{run_id}"
    ctx = _authz(request, "GET", route_path)
    run = _extraction_store().get_extraction_run(run_id)
    _require_owner(run, str(ctx.organization_id), str(ctx.workspace_id) if ctx.workspace_id else None, label="extraction run")
    return {"ok": True, "extractionRun": run}


@router.get("/api/knowledge/builder/extraction-runs/{run_id}/occurrences")
async def list_knowledge_builder_extraction_occurrences(run_id: str, request: Request):
    route_path = "/api/knowledge/builder/extraction-runs/{run_id}/occurrences"
    ctx = _authz(request, "GET", route_path)
    store = _extraction_store()
    _require_owner(store.get_extraction_run(run_id), str(ctx.organization_id), str(ctx.workspace_id) if ctx.workspace_id else None, label="extraction run")
    return {
        "ok": True,
        "schemaVersion": "knowledge_builder_extraction_occurrences_response.v1",
        "occurrences": [item.model_dump(mode="json") for item in store.list_occurrences(run_id)],
    }


@router.post("/api/knowledge/builder/extraction-runs/{run_id}/ai-grounding-suggestions")
async def create_knowledge_builder_ai_grounding_suggestions(run_id: str, request: Request):
    route_path = "/api/knowledge/builder/extraction-runs/{run_id}/ai-grounding-suggestions"
    ctx = _authz(request, "POST", route_path)
    store = _extraction_store()
    run = _require_owner(
        store.get_extraction_run(run_id), str(ctx.organization_id),
        str(ctx.workspace_id) if ctx.workspace_id else None, label="extraction run",
    )
    body = await _json_body(request)
    try:
        grounding_request = AiGroundingRequest.model_validate(body)
    except ValidationError as exc:
        detail = str(exc)
        if body.get("schemaVersion") == "knowledge_builder_ai_grounding_request.v1" or (
            "confidenceThreshold" in body or "confidence_threshold" in body
        ):
            detail = (
                "knowledge_builder_ai_grounding_request.v2 required; "
                "confidenceThreshold is not supported by structured AI Ground"
            )
        raise HTTPException(status_code=400, detail=detail) from exc
    if grounding_request.extraction_run_id != run_id:
        raise HTTPException(status_code=400, detail="extractionRunId must match route run id")
    assertions = store.list_assertions(
        run_id, organization_id=str(ctx.organization_id),
        workspace_id=str(ctx.workspace_id) if ctx.workspace_id else None, limit=500,
    )
    if grounding_request.candidate_ids is not None:
        requested_ids = set(grounding_request.candidate_ids)
        assertions = [item for item in assertions if item.assertion_id in requested_ids]
        if {item.assertion_id for item in assertions} != requested_ids:
            reference = store.list_reference_concepts(extraction_run_id=run_id)
            concepts = [
                item for item in reference["concepts"]
                if item["concept_id"] in requested_ids
            ]
            if {item["concept_id"] for item in concepts} == requested_ids:
                suggestions = [_reference_concept_ai_suggestion(item) for item in concepts]
                return {
                    "ok": True,
                    "schemaVersion": "knowledge_builder_ai_grounding_suggestions_response.v2",
                    "authorityState": "candidate_only",
                    "assessmentSource": "llm_structured",
                    "aiGroundingRun": {
                        "ai_grounding_run_id": f"ai_ground_ref_{uuid4().hex}",
                        "extraction_run_id": run_id,
                        "provider": "semantier_reference_concept",
                        "model": "deterministic_reference_metadata",
                        "agent_id": "semantier-core",
                        "status": "completed",
                    },
                    "suggestions": suggestions,
                    "splitsApplied": [],
                    "summary": {
                        "total": len(suggestions),
                        "supported": len(suggestions),
                        "unsupported": 0,
                        "ambiguous": 0,
                        "needs_edit": 0,
                        "provider_error": 0,
                    },
                }
            raise HTTPException(status_code=400, detail="candidateIds contain unknown assertions")
    if not assertions:
        reference = store.list_reference_concepts(extraction_run_id=run_id)
        concepts = reference["concepts"]
        if concepts:
            suggestions = [_reference_concept_ai_suggestion(item) for item in concepts]
            return {
                "ok": True,
                "schemaVersion": "knowledge_builder_ai_grounding_suggestions_response.v2",
                "authorityState": "candidate_only",
                "assessmentSource": "llm_structured",
                "aiGroundingRun": {
                    "ai_grounding_run_id": f"ai_ground_ref_{uuid4().hex}",
                    "extraction_run_id": run_id,
                    "provider": "semantier_reference_concept",
                    "model": "deterministic_reference_metadata",
                    "agent_id": "semantier-core",
                    "status": "completed",
                },
                "suggestions": suggestions,
                "splitsApplied": [],
                "summary": {
                    "total": len(suggestions),
                    "supported": len(suggestions),
                    "unsupported": 0,
                    "ambiguous": 0,
                    "needs_edit": 0,
                    "provider_error": 0,
                },
            }
        raise HTTPException(status_code=400, detail="AI Ground requires at least one candidate")
    prompt = load_repo_prompt(
        "contextgraph/ai_ground_v2.md", allowed_namespace="contextgraph"
    )
    ai_run = store.create_ai_grounding_run(
        run_id, organization_id=str(ctx.organization_id),
        workspace_id=str(ctx.workspace_id) if ctx.workspace_id else None,
        prompt_ref=prompt.relative_name, prompt_hash=prompt.sha256,
    )
    candidates = []
    for assertion in assertions:
        normalized = assertion.normalized_assertion
        candidates.append(GroundingCandidateInput(
            assertion_id=assertion.assertion_id,
            subject_text=json.dumps(normalized.subject, ensure_ascii=False, sort_keys=True),
            predicate_text=normalized.predicate or "",
            object_text=json.dumps(normalized.object, ensure_ascii=False, sort_keys=True),
            evidence=tuple({
                "anchor_id": anchor.anchor_id,
                "exact_text": anchor.exact_text,
                "document_part": anchor.document_part,
                "start_offset": anchor.start_offset,
                "end_offset": anchor.end_offset,
            } for anchor in assertion.source_anchors),
            source_hash=assertion.source_document_hash,
            candidate_graph_version=assertion.candidate_graph_id,
        ))
    started = monotonic()
    try:
        outcome = await SemanticaGroundingAssistant(
            _current_agent_structured_llm(ctx.hermes_home), batch_size=12, max_concurrency=4
        ).ground(candidates=candidates, prompt=prompt)
        latency_ms = max(0, round((monotonic() - started) * 1000))
        generations = outcome.generations
        if not generations:
            raise ValueError("AI Ground requires at least one candidate")
        provider = generations[0].provider
        model = generations[0].model
        agent_id = generations[0].agent_id
        if any((item.provider, item.model, item.agent_id) != (provider, model, agent_id)
               for item in generations):
            raise ValueError("active provider attribution changed during AI Ground run")
        usage: dict[str, int] = {}
        for generation in generations:
            for key, value in generation.usage.items():
                if isinstance(value, int):
                    usage[key] = usage.get(key, 0) + value
        response_hash = sha256("\n".join(item.text for item in generations).encode()).hexdigest()
        result_payloads = [item.model_dump(mode="json") for item in outcome.results]
        completed_run = store.complete_ai_grounding_run(
            ai_run["ai_grounding_run_id"], results=result_payloads,
            provider=provider, model=model, agent_id=agent_id, usage=usage,
            latency_ms=latency_ms, response_hash=response_hash,
        )
    except asyncio.CancelledError:
        store.fail_ai_grounding_run(
            ai_run["ai_grounding_run_id"], failure_reason="AI Ground request cancelled",
            latency_ms=max(0, round((monotonic() - started) * 1000)),
        )
        raise
    except Exception as exc:
        store.fail_ai_grounding_run(
            ai_run["ai_grounding_run_id"], failure_reason=str(exc),
            latency_ms=max(0, round((monotonic() - started) * 1000)),
        )
        raise HTTPException(status_code=502, detail=f"AI Ground provider failed: {exc}") from exc
    latest = store.list_latest_ai_grounding_results(run_id)
    persisted = latest["suggestions"]
    statuses = ("supported", "unsupported", "ambiguous", "needs_edit", "provider_error")
    counts = {status: sum(item["status"] == status for item in persisted) for status in statuses}
    return {
        "ok": True,
        "schemaVersion": "knowledge_builder_ai_grounding_suggestions_response.v2",
        "authorityState": "candidate_only",
        "assessmentSource": "llm_structured",
        "aiGroundingRun": completed_run,
        "suggestions": persisted,
        "summary": {"total": len(persisted), **counts},
    }


@router.get("/api/knowledge/builder/assertion-candidates")
async def list_knowledge_builder_assertion_candidates(request: Request):
    route_path = "/api/knowledge/builder/assertion-candidates"
    ctx = _authz(request, "GET", route_path)
    store = _extraction_store()
    extraction_run_id = request.query_params.get("extractionRunId") or request.query_params.get("extraction_run_id")
    if extraction_run_id:
        _require_owner(store.get_extraction_run(extraction_run_id), str(ctx.organization_id), str(ctx.workspace_id) if ctx.workspace_id else None, label="extraction run")
    limit = _page_limit(request)
    cursor = request.query_params.get("cursor")
    assertions_with_sentinel = store.list_assertions(
        extraction_run_id, organization_id=str(ctx.organization_id),
        workspace_id=str(ctx.workspace_id) if ctx.workspace_id else None,
        limit=limit + 1, cursor=cursor,
    )
    has_more = len(assertions_with_sentinel) > limit
    assertions = assertions_with_sentinel[:limit]
    assertion_ids = [item.assertion_id for item in assertions]
    histories = store.list_assertion_review_histories(
        assertion_ids, organization_id=str(ctx.organization_id),
        workspace_id=str(ctx.workspace_id) if ctx.workspace_id else None,
    )
    ai_grounding = store.list_latest_ai_grounding_results(extraction_run_id) if extraction_run_id else {
        "assessmentSource": "legacy_threshold", "suggestions": []
    }
    return {
        "ok": True,
        "schemaVersion": "knowledge_builder_assertion_candidates_response.v1",
        "assertionCandidates": [item.model_dump(mode="json") for item in assertions],
        "reviewHistories": histories,
        "aiGroundingAssessmentSource": ai_grounding["assessmentSource"],
        "aiGroundingSuggestions": ai_grounding["suggestions"],
        "page": {
            "limit": limit,
            "returnedCount": len(assertions),
            "nextCursor": assertions[-1].assertion_id if has_more and assertions else None,
        },
    }


@router.post("/api/knowledge/builder/assertion-candidates/batch-grounding-events")
async def create_knowledge_builder_batch_grounding_events(request: Request):
    route_path = "/api/knowledge/builder/assertion-candidates/batch-grounding-events"
    ctx = _authz(request, "POST", route_path)
    try:
        batch = BatchAcceptRequest.model_validate(await _json_body(request))
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    store = _extraction_store()
    workspace_id = str(ctx.workspace_id) if ctx.workspace_id else None
    prepared: list[LearningEventEnvelope] = []
    extraction_run_id: str | None = None
    batch_id = f"ground_batch_{uuid4().hex}"
    occurred_at = datetime.now(timezone.utc)
    try:
        for item in batch.items:
            owner = _require_owner(
                store.get_assertion_owner(item.assertion_id), str(ctx.organization_id),
                workspace_id, label="assertion candidate",
            )
            if extraction_run_id is None:
                extraction_run_id = owner["extraction_run_id"]
            elif extraction_run_id != owner["extraction_run_id"]:
                raise ValueError("batch candidates must belong to one extraction run")
            assertion = store.get_assertion(item.assertion_id)
            if assertion is None:
                raise ValueError("assertion candidate not found")
            valid_refs = store.valid_evidence_anchor_refs(item.assertion_id)
            source_refs = tuple(item.evidence_anchor_refs)
            if not source_refs or not set(source_refs).issubset(valid_refs):
                raise ValueError("batch evidence anchor refs are not canonical evidence")
            history = store.list_events(item.assertion_id)
            event_id = f"human_event_{uuid4().hex}"
            prepared.append(LearningEventEnvelope(
                event_id=event_id,
                case_id=item.assertion_id,
                event_type=LearningEventType.HUMAN_ACCEPT,
                occurred_at=occurred_at,
                source_document_hash=assertion.source_document_hash,
                source_anchor_refs=source_refs,
                graph_version="KG_v0",
                rule_version="tender_rules.v1",
                evaluation_corpus_version="tender_sensitive_v1",
                actor_ref=f"user:{ctx.user_id}",
                human_label=HumanLabel(
                    decision=HumanDecision.ACCEPT,
                    certainty=batch.certainty,
                    reason_code=batch.reason_code,
                    justification=batch.justification,
                ),
                prior_event_hash=history[-1]["event_hash"] if history else None,
            ))
        if extraction_run_id is None:
            raise ValueError("batch must contain candidates")
        return store.append_human_accept_batch(
            events=prepared, batch_id=batch_id,
            organization_id=str(ctx.organization_id), workspace_id=workspace_id,
            extraction_run_id=extraction_run_id,
            idempotency_key=batch.batch_idempotency_key,
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/api/knowledge/builder/reference-concepts")
async def list_knowledge_builder_reference_concepts(request: Request):
    route_path = "/api/knowledge/builder/reference-concepts"
    ctx = _authz(request, "GET", route_path)
    store = _extraction_store()
    extraction_run_id = request.query_params.get("extractionRunId") or request.query_params.get("extraction_run_id")
    if extraction_run_id:
        _require_owner(
            store.get_extraction_run(extraction_run_id), str(ctx.organization_id),
            str(ctx.workspace_id) if ctx.workspace_id else None, label="extraction run",
        )
    reference = store.list_reference_concepts(extraction_run_id=extraction_run_id)
    concepts = reference["concepts"]
    source_text = None
    if extraction_run_id:
        extraction_run = store.get_extraction_run(extraction_run_id) or {}
        source = KnowledgeBuilderStore(_eos_path()).get_source(
            str(extraction_run.get("source_id") or "")
        ) or {}
        source_text = str(source.get("source_text") or "") or None
    concept_ids = [item["concept_id"] for item in concepts]
    histories = store.list_reference_concept_review_histories(
        concept_ids, organization_id=str(ctx.organization_id),
        workspace_id=str(ctx.workspace_id) if ctx.workspace_id else None,
    )
    suggestions = [_reference_concept_ai_suggestion(item) for item in concepts]
    return {
        "ok": True,
        "schemaVersion": "contextgraph_reference_concepts_response.v1",
        "referenceConcepts": concepts,
        "canonicalExpressions": reference["canonical_expressions"],
        "assertionCandidates": [
            _reference_concept_candidate(
                item, reference["canonical_expressions"], source_text=source_text
            )
            for item in concepts
        ],
        "reviewHistories": histories,
        "aiGroundingAssessmentSource": "llm_structured",
        "aiGroundingSuggestions": suggestions,
        "page": {
            "limit": len(concepts),
            "returnedCount": len(concepts),
            "nextCursor": None,
        },
    }


@router.get("/api/contextgraph/extraction-runs/{run_id}/candidate-spans")
async def list_contextgraph_extraction_candidate_spans(run_id: str, request: Request):
    route_path = "/api/contextgraph/extraction-runs/{run_id}/candidate-spans"
    ctx = _authz(request, "GET", route_path)
    store = _extraction_store()
    _require_owner(
        store.get_extraction_run(run_id),
        str(ctx.organization_id),
        str(ctx.workspace_id) if ctx.workspace_id else None,
        label="extraction run",
    )
    reference = store.list_reference_concepts(extraction_run_id=run_id)
    concepts = reference["concepts"]
    spans = [
        {
            "candidate_span_id": item["concept_id"],
            "run_id": run_id,
            "exact_text": item.get("canonical_label") or "",
            "semantic_role": item.get("semantic_role") or item.get("category"),
            "source_anchor_refs": list(item.get("source_anchor_refs") or []),
            "grounding_state": item.get("grounding_state") or "candidate",
            "needs_boundary_review": item.get("grounding_state") not in {"human_accepted", "grounded"},
        }
        for item in concepts
    ]
    return {
        "ok": True,
        "schemaVersion": "contextgraph_candidate_spans_response.v1",
        "runId": run_id,
        "candidateSpans": spans,
        "page": {
            "limit": len(spans),
            "returnedCount": len(spans),
            "nextCursor": None,
        },
    }


@router.get("/api/contextgraph/extraction-runs/{run_id}/learning-events")
async def list_contextgraph_extraction_learning_events(run_id: str, request: Request):
    route_path = "/api/contextgraph/extraction-runs/{run_id}/learning-events"
    ctx = _authz(request, "GET", route_path)
    store = _extraction_store()
    _require_owner(
        store.get_extraction_run(run_id),
        str(ctx.organization_id),
        str(ctx.workspace_id) if ctx.workspace_id else None,
        label="extraction run",
    )
    reference = store.list_reference_concepts(extraction_run_id=run_id)
    concept_ids = [item["concept_id"] for item in reference["concepts"]]
    history_map = store.list_reference_concept_review_histories(
        concept_ids,
        organization_id=str(ctx.organization_id),
        workspace_id=str(ctx.workspace_id) if ctx.workspace_id else None,
    )
    events = [
        event
        for concept_id in concept_ids
        for event in history_map.get(concept_id, [])
    ]
    events.sort(key=lambda item: (item.get("occurred_at") or "", item.get("event_id") or ""))
    return {
        "ok": True,
        "schemaVersion": "contextgraph_boundary_learning_events_response.v1",
        "runId": run_id,
        "learningEvents": events,
    }


@router.post("/api/contextgraph/extraction-runs/{run_id}/boundary-actions")
async def create_contextgraph_extraction_boundary_action(run_id: str, request: Request):
    route_path = "/api/contextgraph/extraction-runs/{run_id}/boundary-actions"
    ctx = _authz(request, "POST", route_path)
    body = await _json_body(request)
    try:
        action = CurationAction.model_validate(body)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not action.sourceSpanIds:
        raise HTTPException(status_code=400, detail="sourceSpanIds is required")

    store = _extraction_store()
    _require_owner(
        store.get_extraction_run(run_id),
        str(ctx.organization_id),
        str(ctx.workspace_id) if ctx.workspace_id else None,
        label="extraction run",
    )

    def _graph_version_number(graph_version: str | None) -> int:
        value = str(graph_version or "")
        if value.startswith("KG_v") and value.removeprefix("KG_v").isdigit():
            return int(value.removeprefix("KG_v"))
        return 0

    def _split_ids(source_span_id: str, offsets: list[int]) -> list[str]:
        ids: list[str] = []
        points = [*offsets]
        boundaries = ["start", *points, "end"]
        for index in range(len(boundaries) - 1):
            digest = content_hash(
                {
                    "contract": "contextgraph-boundary-split-v1",
                    "source_span_id": source_span_id,
                    "left": boundaries[index],
                    "right": boundaries[index + 1],
                    "ordinal": index,
                }
            ).removeprefix("sha256:")
            ids.append(f"ccs_v1_{digest}")
        return ids

    candidate_graph_ids: set[str] = set()
    concept_items: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for concept_id in action.sourceSpanIds:
        owner = _require_owner(
            store.get_reference_concept_owner(concept_id),
            str(ctx.organization_id),
            str(ctx.workspace_id) if ctx.workspace_id else None,
            label="reference concept",
        )
        if owner.get("extraction_run_id") != run_id:
            raise HTTPException(status_code=409, detail="boundary action spans must belong to the target extraction run")
        concept = store.get_reference_concept(concept_id)
        if concept is None:
            raise HTTPException(status_code=404, detail="reference concept not found")
        graph_id = str(owner.get("candidate_graph_id") or "")
        if graph_id:
            candidate_graph_ids.add(graph_id)
        concept_items.append((concept_id, owner, concept))

    if len(candidate_graph_ids) > 1:
        raise HTTPException(status_code=409, detail="boundary action spans must belong to one candidate graph")
    if candidate_graph_ids:
        graph_owner = store.get_candidate_graph_owner(next(iter(candidate_graph_ids))) or {}
        current_version = _graph_version_number(graph_owner.get("allowed_base_graph_version"))
        if action.expectedGraphVersion != current_version:
            raise HTTPException(
                status_code=409,
                detail=f"stale boundary action: expectedGraphVersion={action.expectedGraphVersion}, current={current_version}",
            )

    if action.actionType == "merge" and len(action.sourceSpanIds) < 2:
        raise HTTPException(status_code=422, detail="merge requires at least two sourceSpanIds")
    if action.actionType == "split" and len(action.sourceSpanIds) != 1:
        raise HTTPException(status_code=422, detail="split requires exactly one sourceSpanId")
    if action.actionType == "edit_role" and not action.targetSemanticRole:
        raise HTTPException(status_code=422, detail="edit_role requires targetSemanticRole")

    if action.actionType == "accept":
        event_type = LearningEventType.HUMAN_ACCEPT
        decision = HumanDecision.ACCEPT
        grounding_state = "human_accepted"
    elif action.actionType == "reject":
        event_type = LearningEventType.HUMAN_REJECT
        decision = HumanDecision.REJECT
        grounding_state = "human_rejected"
    else:
        event_type = LearningEventType.HUMAN_UNCERTAIN
        decision = HumanDecision.UNCERTAIN
        grounding_state = "human_edited"

    grounding_store = HumanGroundingStore(
        _eos_path(), graph_backend=_CandidateOnlyPreviewBackend()
    )
    workbench = HumanGroundingWorkbench(grounding_store)
    stored_events: list[dict] = []
    refreshed: list[dict] = []
    replacement_span_ids = list(action.sourceSpanIds)
    relation_proposals: list[dict[str, Any]] = []

    modifier_labels = {
        "仅限", "限于", "仅允许", "不得", "禁止", "不包括", "排除", "适用于", "针对",
    }

    def _relation_proposal(
        source_id: str,
        source_label: str,
        target_id: str,
        target_label: str,
        anchor_refs: tuple[str, ...],
    ) -> dict[str, Any] | None:
        if source_label.strip() not in modifier_labels:
            return None
        return {
            "relationId": f"relation_{content_hash({'source': source_id, 'target': target_id, 'type': 'modifier_of'})[7:23]}",
            "sourceNodeId": source_id,
            "targetNodeId": target_id,
            "relationType": "modifier_of",
            "sourceLabel": source_label,
            "targetLabel": target_label,
            "sourceAnchorRefs": list(anchor_refs),
            "confidence": 0.65,
            "groundingState": "candidate",
        }

    if action.actionType == "split":
        replacement_span_ids = _split_ids(action.sourceSpanIds[0], action.splitOffsetsAbsolute or [])
    elif action.actionType == "merge":
        digest = content_hash(
            {
                "contract": "contextgraph-boundary-merge-v1",
                "source_span_ids": list(action.sourceSpanIds),
            }
        ).removeprefix("sha256:")
        replacement_span_ids = [f"ccs_v1_{digest}"]

    if action.actionType == "split":
        source_concept = concept_items[0][2]
        label = str(source_concept.get("canonical_label") or "")
        offsets = [0, *(action.splitOffsetsAbsolute or []), len(label)]
        split_labels = [label[left:right] for left, right in zip(offsets, offsets[1:])]
        existing_concepts_by_label = {
            str(item.get("canonical_label") or ""): str(item["concept_id"])
            for item in store.list_reference_concepts(
                extraction_run_id=run_id
            )["concepts"]
            if item["concept_id"] != action.sourceSpanIds[0]
        }
        replacement_span_ids = [
            existing_concepts_by_label.get(split_label, replacement_span_ids[index])
            for index, split_label in enumerate(split_labels)
        ]
        anchor_refs = tuple(source_concept.get("source_anchor_refs") or ())
        for index, (source_label, target_label) in enumerate(
            zip(split_labels, split_labels[1:])
        ):
            proposal = _relation_proposal(
                replacement_span_ids[index], source_label,
                replacement_span_ids[index + 1], target_label, anchor_refs,
            )
            if proposal:
                relation_proposals.append(proposal)
    elif action.actionType == "merge":
        for left_item, right_item in zip(concept_items, concept_items[1:]):
            left_concept = left_item[2]
            right_concept = right_item[2]
            proposal = _relation_proposal(
                left_item[0], str(left_concept.get("canonical_label") or ""),
                right_item[0], str(right_concept.get("canonical_label") or ""),
                tuple(left_concept.get("source_anchor_refs") or ())
                + tuple(right_concept.get("source_anchor_refs") or ()),
            )
            if proposal:
                relation_proposals.append(proposal)

    for concept_id, owner, concept in concept_items:
        source_anchor_refs = tuple(concept.get("source_anchor_refs") or ())
        if not source_anchor_refs:
            raise HTTPException(status_code=400, detail="reference concept has no evidence anchors")
        histories = store.list_reference_concept_review_histories(
            [concept_id],
            organization_id=str(ctx.organization_id),
            workspace_id=str(ctx.workspace_id) if ctx.workspace_id else None,
        )
        history = histories.get(concept_id, [])
        boundary_metadata: dict[str, Any] = {
            "boundary_action": action.actionType,
            "action_id": action.actionId,
        }
        if action.actionType == "split":
            boundary_metadata["split_offsets_absolute"] = list(action.splitOffsetsAbsolute or [])
            boundary_metadata["replacement_span_ids"] = replacement_span_ids
        if action.actionType == "merge":
            boundary_metadata["merge_source_span_ids"] = list(action.sourceSpanIds)
            boundary_metadata["replacement_span_ids"] = replacement_span_ids
        if relation_proposals:
            boundary_metadata["relation_proposals"] = relation_proposals
        if action.actionType == "edit_role":
            boundary_metadata["target_semantic_role"] = action.targetSemanticRole
        event = LearningEventEnvelope(
            event_id=f"human_event_{uuid4().hex}",
            case_id=concept_id,
            event_type=event_type,
            occurred_at=datetime.now(timezone.utc),
            source_document_hash=str(owner.get("source_document_hash") or concept.get("source_document_hash")),
            source_anchor_refs=source_anchor_refs,
            graph_version="KG_v0",
            rule_version="tender_rules.v1",
            evaluation_corpus_version="tender_sensitive_v1",
            actor_ref=f"user:{ctx.user_id}",
            human_label=HumanLabel(
                decision=decision,
                certainty="",
                reason_code=action.reasonCode,
                justification=str(action.comment or ""),
            ),
            downstream_outcome=boundary_metadata,
            prior_event_hash=history[-1]["event_hash"] if history else None,
        )
        result = workbench.submit_review(human_event=event, delta=None)
        stored_events.append(result["human_event"])
        if action.actionType == "split":
            label = str(concept.get("canonical_label") or "")
            offsets = [0, *(action.splitOffsetsAbsolute or []), len(label)]
            if any(left >= right for left, right in zip(offsets, offsets[1:])):
                raise HTTPException(status_code=422, detail="split offsets must produce non-empty concepts")

            def _split_anchor(anchor_ref: str, left: int, right: int) -> str:
                match = re.match(r"^(.*:span:)(\d+)-(\d+)$", str(anchor_ref))
                if match is None:
                    return str(anchor_ref)
                span_start = int(match.group(2))
                return f"{match.group(1)}{span_start + left}-{span_start + right}"

            replacements = tuple(
                ReferenceConcept(
                    schema_version="contextgraph_reference_concept.v1",
                    concept_id=replacement_span_ids[index],
                    canonical_label=label[left:right],
                    category=concept.get("category"),
                    prohibition_type=concept.get("prohibition_type"),
                    applies_to=tuple(concept.get("applies_to") or ()),
                    source_anchor_refs=tuple(
                        _split_anchor(anchor, left, right)
                        for anchor in source_anchor_refs
                    ),
                    grounding_state="human_edited",
                )
                for index, (left, right) in enumerate(zip(offsets, offsets[1:]))
            )
            store.replace_reference_concept_with_splits(concept_id, replacements)
        elif action.actionType in {"accept", "reject", "merge", "edit_role"}:
            store.update_reference_concept_grounding_state(concept_id, grounding_state)
        refreshed.extend(
            {
                "candidateSpanId": replacement_id,
                "groundingState": grounding_state,
            }
            for replacement_id in replacement_span_ids
        )

    latest_graph_version = 0
    if candidate_graph_ids:
        graph_owner = store.get_candidate_graph_owner(next(iter(candidate_graph_ids))) or {}
        latest_graph_version = _graph_version_number(graph_owner.get("allowed_base_graph_version"))

    return {
        "ok": True,
        "schemaVersion": "contextgraph_boundary_action_response.v1",
        "actionId": action.actionId,
        "storedEvent": stored_events[0] if len(stored_events) == 1 else {"events": stored_events},
        "replacementSpanIds": replacement_span_ids,
        "relationProposals": relation_proposals,
        "newGraphVersion": latest_graph_version,
        "refreshedDiagnostics": refreshed,
    }


@router.get("/api/contextgraph/extraction-runs/{run_id}/candidate-graph/{graph_version}")
async def get_contextgraph_extraction_candidate_graph_version(run_id: str, graph_version: str, request: Request):
    route_path = "/api/contextgraph/extraction-runs/{run_id}/candidate-graph/{graph_version}"
    ctx = _authz(request, "GET", route_path)
    store = _extraction_store()
    _require_owner(
        store.get_extraction_run(run_id),
        str(ctx.organization_id),
        str(ctx.workspace_id) if ctx.workspace_id else None,
        label="extraction run",
    )
    grounding_store = HumanGroundingStore(_eos_path(), graph_backend=_release_graph_backend())
    try:
        release = grounding_store.get_graph_release(
            graph_version,
            organization_id=str(ctx.organization_id),
            workspace_id=str(ctx.workspace_id) if ctx.workspace_id else None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    source_run_id = str(release.get("source_extraction_run_id") or "")
    if not source_run_id:
        candidate_graph_id = str(release.get("candidate_graph_id") or "")
        if candidate_graph_id:
            graph_owner = store.get_candidate_graph_owner(candidate_graph_id) or {}
            source_run_id = str(graph_owner.get("extraction_run_id") or "")
    if source_run_id and source_run_id != run_id:
        raise HTTPException(status_code=404, detail="graph release not found for extraction run")
    return {
        "ok": True,
        "schemaVersion": "contextgraph_candidate_graph_version_response.v1",
        "runId": run_id,
        "graphVersion": graph_version,
        "graphRelease": release,
    }


@router.get("/api/knowledge/builder/reference-concepts/{concept_id}")
async def get_knowledge_builder_reference_concept(concept_id: str, request: Request):
    route_path = "/api/knowledge/builder/reference-concepts/{concept_id}"
    ctx = _authz(request, "GET", route_path)
    store = _extraction_store()
    _require_owner(
        store.get_reference_concept_owner(concept_id), str(ctx.organization_id),
        str(ctx.workspace_id) if ctx.workspace_id else None, label="reference concept",
    )
    concept = store.get_reference_concept(concept_id)
    if concept is None:
        raise HTTPException(status_code=404, detail="reference concept not found")
    history = store.list_reference_concept_review_histories(
        [concept_id], organization_id=str(ctx.organization_id),
        workspace_id=str(ctx.workspace_id) if ctx.workspace_id else None,
    )
    reference = store.list_reference_concepts(extraction_run_id=concept["extraction_run_id"])
    extraction_run = store.get_extraction_run(str(concept["extraction_run_id"])) or {}
    source = KnowledgeBuilderStore(_eos_path()).get_source(
        str(extraction_run.get("source_id") or "")
    ) or {}
    source_text = str(source.get("source_text") or "") or None
    evidence_reviews = SemanticFeedbackEventStore(
        ctx.workspace_root, workspace_id=str(ctx.workspace_id)
    ).list_for_original_result(concept_id)
    return {
        "ok": True,
        "schemaVersion": "contextgraph_reference_concept_response.v1",
        "referenceConcept": concept,
        "assertionCandidate": _reference_concept_candidate(
            concept, reference["canonical_expressions"], source_text=source_text
        ),
        "learningEvents": history.get(concept_id, []),
        "candidateTopology": {"schema_version": "contextgraph_reference_concept.v1"},
        "acceptedTopology": None,
        "evidenceObligations": [],
        "occurrences": [],
        "evidenceReviews": [event.as_dict() for event in evidence_reviews],
    }


@router.post("/api/knowledge/builder/reference-concepts/{concept_id}/evidence-reviews")
async def create_knowledge_builder_reference_concept_evidence_review(
    concept_id: str, request: Request
):
    route_path = "/api/knowledge/builder/reference-concepts/{concept_id}/evidence-reviews"
    ctx = _authz(request, "POST", route_path)
    body = await _json_body(request)
    if body.get("schemaVersion") != "reference_concept_evidence_review_request.v1":
        raise HTTPException(
            status_code=400,
            detail="reference_concept_evidence_review_request.v1 required",
        )
    decision = str(body.get("decision") or "")
    if decision not in {"accept", "reject"}:
        raise HTTPException(status_code=400, detail="decision must be accept or reject")
    reason_code = str(body.get("reasonCode") or "") or None
    allowed_reasons = {
        "wrong_substring", "wrong_context", "wrong_concept", "duplicate", "other"
    }
    if decision == "reject" and reason_code not in allowed_reasons:
        raise HTTPException(status_code=400, detail="a supported rejection reason is required")
    evidence_ref = str(body.get("evidenceRef") or "")
    if not evidence_ref:
        raise HTTPException(status_code=400, detail="evidenceRef is required")
    store = _extraction_store()
    _require_owner(
        store.get_reference_concept_owner(concept_id), str(ctx.organization_id),
        str(ctx.workspace_id) if ctx.workspace_id else None, label="reference concept",
    )
    concept = store.get_reference_concept(concept_id)
    if concept is None:
        raise HTTPException(status_code=404, detail="reference concept not found")
    if evidence_ref not in store.valid_reference_concept_evidence_anchor_refs(concept_id):
        raise HTTPException(status_code=400, detail="evidenceRef is not canonical evidence for reference concept")
    run_id = str(concept.get("extraction_run_id") or "")
    extraction_run = store.get_extraction_run(run_id) or {}
    reference = store.list_reference_concepts(extraction_run_id=run_id)
    source = KnowledgeBuilderStore(_eos_path()).get_source(
        str(extraction_run.get("source_id") or "")
    ) or {}
    corrected_result, finding_id = _canonical_evidence_review_payload(
        concept=concept,
        expressions=reference["canonical_expressions"],
        evidence_ref=evidence_ref,
        source_text=str(source.get("source_text") or "") or None,
    )
    event = SemanticFeedbackEventStore(
        ctx.workspace_root, workspace_id=str(ctx.workspace_id)
    ).create_event(
        run_id=run_id,
        finding_id=finding_id,
        action="confirm" if decision == "accept" else "dismiss",
        original_result_ref=concept_id,
        corrected_result=corrected_result,
        reason_code=reason_code,
        justification=str(body.get("comment") or "") or None,
        evidence_ref=evidence_ref,
        graph_version_ref=str(concept.get("candidate_graph_id") or "candidate_graph"),
        profile_ref=str(extraction_run.get("profile_ref") or "unknown"),
        profile_version=str(extraction_run.get("provider_version") or "1"),
        skill_ref=CANONICAL_JOURNEY_REF,
        reviewer=str(ctx.user_id),
    )
    return {"ok": True, "evidenceReview": event.as_dict()}


@router.post("/api/knowledge/builder/evidence-reviews/{feedback_id}/regex-proposals")
async def create_knowledge_builder_regex_improvement_proposal(
    feedback_id: str, request: Request
):
    route_path = "/api/knowledge/builder/evidence-reviews/{feedback_id}/regex-proposals"
    ctx = _authz(request, "POST", route_path)
    body = await _json_body(request)
    if body.get("schemaVersion") != "regex_improvement_proposal_request.v1":
        raise HTTPException(status_code=400, detail="regex_improvement_proposal_request.v1 required")
    try:
        feedback = SemanticFeedbackEventStore(
            ctx.workspace_root, workspace_id=str(ctx.workspace_id)
        ).get_event(feedback_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="evidence review not found")
    if feedback.action != "dismiss":
        raise HTTPException(status_code=400, detail="regex proposals require rejected evidence")
    corrected = feedback.corrected_result or {}
    raw_positive_examples = body.get("positiveExamples", [])
    if not isinstance(raw_positive_examples, list) or len(raw_positive_examples) > 50:
        raise HTTPException(status_code=400, detail="positiveExamples must contain at most 50 values")
    positive_examples = [str(value) for value in raw_positive_examples if str(value)]
    if not positive_examples:
        raise HTTPException(status_code=400, detail="positiveExamples are required")
    if any(len(value) > 2_000 for value in positive_examples):
        raise HTTPException(status_code=400, detail="positiveExamples values must not exceed 2000 characters")
    try:
        llm = _current_agent_structured_llm(ctx.hermes_home)
        result = await llm.generate_structured(
            instructions=load_repo_prompt(
                "contextgraph/regex_improvement_proposal_v1.md",
                allowed_namespace="contextgraph",
            ).text,
            input_text=canonical_json({
                "concept_label": corrected.get("concept_label"),
                "negative_examples": [corrected.get("context_text")],
                "positive_examples": positive_examples,
                "reason_code": feedback.reason_code,
            }),
            json_schema={
                "type": "object", "additionalProperties": False,
                "required": ["regex", "rationale"],
                "properties": {
                    "regex": {"type": "string", "maxLength": 500},
                    "rationale": {"type": "string", "maxLength": 500},
                },
            },
            schema_name="semantier_regex_improvement_proposal_v1",
            purpose="contextgraph_ai_extract_regex_proposal",
        )
    except StructuredGenerationError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"regex proposal provider unavailable: {exc}",
        ) from exc
    try:
        proposal = evaluate_regex_proposal(
            concept_label=str(corrected.get("concept_label") or ""),
            regex=str(result.parsed.get("regex") or ""),
            rationale=str(result.parsed.get("rationale") or ""),
            negative_examples=[str(corrected.get("context_text") or "")],
            positive_examples=positive_examples,
            feedback_refs=[feedback.feedback_id],
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    artifact_dir = Path(ctx.workspace_root) / "runs" / "knowledge-builder" / "artifacts" / "regex-proposals"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_name = f"{proposal['proposal_hash'][7:]}.json"
    artifact_path = artifact_dir / artifact_name
    artifact_path.write_text(canonical_json(proposal), encoding="utf-8")
    return {
        "ok": True,
        "regexProposal": proposal,
        "artifactRef": str(artifact_path.relative_to(ctx.workspace_root)),
    }


@router.get("/api/knowledge/builder/reference-concepts/{concept_id}/graph-delta-preview")
async def preview_knowledge_builder_reference_concept_graph_delta(concept_id: str, request: Request):
    route_path = "/api/knowledge/builder/reference-concepts/{concept_id}/graph-delta-preview"
    ctx = _authz(request, "GET", route_path)
    store = _extraction_store()
    _require_owner(
        store.get_reference_concept_owner(concept_id), str(ctx.organization_id),
        str(ctx.workspace_id) if ctx.workspace_id else None, label="reference concept",
    )
    concept = store.get_reference_concept(concept_id)
    if concept is None:
        raise HTTPException(status_code=404, detail="reference concept not found")
    return {
        "ok": True,
        "available": False,
        "reason": "Reference concept accept uses source-anchor admission without GraphDelta edit.",
        "evidenceAnchorRefs": list(concept.get("source_anchor_refs", [])),
    }


@router.post("/api/knowledge/builder/reference-concepts/{concept_id}/grounding-events")
async def create_knowledge_builder_reference_concept_grounding_event(concept_id: str, request: Request):
    route_path = "/api/knowledge/builder/reference-concepts/{concept_id}/grounding-events"
    ctx = _authz(request, "POST", route_path)
    body = await _json_body(request)
    schema_version = body.get("schemaVersion") or body.get("schema_version")
    if schema_version != "learning_event_grounding_request.v1":
        raise HTTPException(status_code=400, detail="learning_event_grounding_request.v1 required")
    try:
        decision = HumanDecision(str(body.get("decision") or ""))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    is_reground = bool(body.get("reground"))
    if is_reground and decision is not HumanDecision.UNCERTAIN:
        raise HTTPException(status_code=400, detail="reground requires an uncertain review decision")
    store = _extraction_store()
    owner = _require_owner(
        store.get_reference_concept_owner(concept_id), str(ctx.organization_id),
        str(ctx.workspace_id) if ctx.workspace_id else None, label="reference concept",
    )
    concept = store.get_reference_concept(concept_id)
    if concept is None:
        raise HTTPException(status_code=404, detail="reference concept not found")
    source_anchor_refs = tuple(
        body.get("evidenceAnchorRefs")
        or body.get("evidence_anchor_refs")
        or concept.get("source_anchor_refs")
        or ()
    )
    if decision in {HumanDecision.ACCEPT, HumanDecision.EDIT} and not source_anchor_refs:
        raise HTTPException(status_code=400, detail="accept/edit review requires non-empty evidence anchor refs")
    valid_evidence_refs = store.valid_reference_concept_evidence_anchor_refs(concept_id)
    if not set(source_anchor_refs).issubset(valid_evidence_refs):
        raise HTTPException(status_code=400, detail="review evidence anchor refs are not canonical evidence for reference concept")
    event_id = f"human_event_{uuid4().hex}"
    history = store.list_reference_concept_review_histories(
        [concept_id], organization_id=str(ctx.organization_id),
        workspace_id=str(ctx.workspace_id) if ctx.workspace_id else None,
    ).get(concept_id, [])
    event_type = {
        HumanDecision.ACCEPT: LearningEventType.HUMAN_ACCEPT,
        HumanDecision.EDIT: LearningEventType.HUMAN_EDIT,
        HumanDecision.REJECT: LearningEventType.HUMAN_REJECT,
        HumanDecision.UNCERTAIN: LearningEventType.HUMAN_UNCERTAIN,
    }[decision]
    label = HumanLabel(
        decision=decision,
        certainty=str(body.get("certainty") or ""),
        reason_code=str(body.get("reasonCode") or body.get("reason_code") or ""),
        justification=str(body.get("justification") or ""),
        process_labels=body.get("processLabels") or body.get("process_labels") or {},
    )
    graph_delta = None
    graph_delta_payload = body.get("graphDelta") or body.get("graph_delta")
    if decision is HumanDecision.EDIT:
        if not isinstance(graph_delta_payload, dict):
            raise HTTPException(status_code=400, detail="edit review requires a GraphDelta")
        graph_delta_input = dict(graph_delta_payload)
        graph_delta_input["human_label_event_id"] = event_id
        graph_delta_input["candidate_graph_id"] = owner["candidate_graph_id"]
        graph_delta_input["from_graph_version"] = (
            HumanGroundingStore(_eos_path(), graph_backend=_CandidateOnlyPreviewBackend())
            .get_candidate_graph_owner(owner["candidate_graph_id"] or "")
            or {}
        ).get("allowed_base_graph_version")
        graph_delta_input["justification"] = label.justification
        try:
            graph_delta = GraphDelta.model_validate(graph_delta_input)
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if graph_delta.candidate_graph_id != owner["candidate_graph_id"]:
            raise HTTPException(status_code=400, detail="GraphDelta candidate graph does not match reference concept")
        for operation in graph_delta.operations:
            if not set(operation.source_anchor_refs).issubset(set(source_anchor_refs)):
                raise HTTPException(status_code=400, detail="GraphDelta source anchors must be bound to review evidence")

    grounding_store = HumanGroundingStore(
        _eos_path(), graph_backend=_CandidateOnlyPreviewBackend()
    )
    review_graph_version = graph_delta.from_graph_version if graph_delta else "KG_v0"
    try:
        if decision in {HumanDecision.ACCEPT, HumanDecision.EDIT} and owner.get("candidate_graph_id"):
            graph_id = str(owner["candidate_graph_id"])
            candidate_owner = _require_owner(
                store.get_candidate_graph_owner(graph_id), str(ctx.organization_id),
                str(ctx.workspace_id) if ctx.workspace_id else None, label="candidate graph",
            )
            review_graph_version = candidate_owner.get("allowed_base_graph_version") or "KG_v0"
            graph = grounding_store.get_candidate_graph(graph_id)
            if (
                graph.extraction_run_id != owner["extraction_run_id"]
                or graph.source_document_hash != owner.get("source_document_hash")
                or concept.get("candidate_graph_hash") != graph.graph_hash()
            ):
                raise HTTPException(status_code=409, detail="reference review candidate binding mismatch")
            # Bind this explicit review to the immutable proposal already produced.
            # This records provenance; it performs no inference or authority transition.
            proposal_receipt = grounding_store.append_event(LearningEventEnvelope(
                event_id=f"machine_receipt_{uuid4().hex}", case_id=concept_id,
                event_type=LearningEventType.MACHINE_PROPOSAL,
                occurred_at=datetime.now(timezone.utc),
                source_document_hash=graph.source_document_hash,
                source_anchor_refs=source_anchor_refs,
                graph_version=review_graph_version,
                rule_version="tender_rules.v1", evaluation_corpus_version="tender_sensitive_v1",
                machine_proposal=graph.machine_proposal,
                prior_event_hash=history[-1]["event_hash"] if history else None,
            ))
            history.append(proposal_receipt)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    event = LearningEventEnvelope(
        event_id=event_id,
        case_id=concept_id,
        event_type=event_type,
        occurred_at=datetime.now(timezone.utc),
        source_document_hash=str(owner.get("source_document_hash") or concept.get("source_document_hash")),
        source_anchor_refs=source_anchor_refs,
        graph_version=review_graph_version,
        rule_version="tender_rules.v1",
        evaluation_corpus_version="tender_sensitive_v1",
        actor_ref=f"user:{ctx.user_id}",
        human_label=label,
        graph_delta_ref=graph_delta.graph_delta_id if graph_delta else None,
        downstream_outcome=(
            {"action": "reground", "source_block_id": str(body.get("sourceBlockId") or "")}
            if is_reground
            else None
        ),
        prior_event_hash=history[-1]["event_hash"] if history else None,
    )
    try:
        result = HumanGroundingWorkbench(grounding_store).submit_review(
            human_event=event, delta=graph_delta
        )
        if decision is HumanDecision.ACCEPT:
            store.update_reference_concept_grounding_state(concept_id, "human_accepted")
        elif decision is HumanDecision.EDIT:
            store.update_reference_concept_grounding_state(concept_id, "human_edited")
        elif decision is HumanDecision.REJECT:
            store.update_reference_concept_grounding_state(concept_id, "human_rejected")
        elif is_reground:
            store.update_reference_concept_grounding_state(concept_id, "candidate")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "learningEvent": result["human_event"],
            "graphDelta": result.get("graph_delta"), "graphRelease": result.get("release")}


@router.get("/api/knowledge/builder/candidate-graphs/{candidate_graph_id}")
async def inspect_knowledge_builder_candidate_graph(
    candidate_graph_id: str, request: Request, expected_graph_hash: str | None = None,
):
    route_path = "/api/knowledge/builder/candidate-graphs/{candidate_graph_id}"
    ctx = _authz(request, "GET", route_path)
    owner = _require_owner(
        _extraction_store().get_candidate_graph_owner(candidate_graph_id),
        str(ctx.organization_id), str(ctx.workspace_id) if ctx.workspace_id else None,
        label="candidate graph",
    )
    try:
        grounding = HumanGroundingStore(_eos_path(), graph_backend=_CandidateOnlyPreviewBackend())
        graph = grounding.get_candidate_graph(candidate_graph_id)
        if expected_graph_hash is not None and expected_graph_hash != graph.graph_hash():
            raise ValueError("candidate graph hash mismatch")
        if graph.extraction_run_id != owner["extraction_run_id"]:
            raise ValueError("candidate graph extraction run mismatch")
        extraction = _extraction_store()
        concepts = extraction.list_reference_concepts(extraction_run_id=owner["extraction_run_id"])["concepts"]
        concept_ids = [item["concept_id"] for item in concepts if item.get("candidate_graph_id") == candidate_graph_id]
        histories = extraction.list_reference_concept_review_histories(
            concept_ids, organization_id=str(ctx.organization_id),
            workspace_id=str(ctx.workspace_id) if ctx.workspace_id else None,
        )
        proposals = []
        for history in histories.values():
            for recorded in history:
                if (recorded.get("human_label") or {}).get("reason_code") != "typed_relation_review":
                    continue
                event = grounding.get_learning_event(recorded["event_id"])
                if (event is None or event.get("event_type") != "human_edit"
                        or (event.get("human_label") or {}).get("decision") != "edit"
                        or event.get("reviewed_candidate_graph_hash") != graph.graph_hash()):
                    raise ValueError("recorded relation candidate hash mismatch")
                delta = grounding.get_candidate_graph_delta(candidate_graph_id, event.get("graph_delta_ref") or "")
                if (delta is None or delta.human_label_event_id != event["event_id"]
                        or delta.delta_hash() != event.get("reviewed_graph_delta_hash")):
                    raise ValueError("recorded relation GraphDelta hash or event mismatch")
                if len(delta.operations) != 1:
                    raise ValueError("recorded relation operation mismatch")
                operation = delta.operations[0]
                after = CandidateEdge.model_validate(operation.after)
                before = next((edge for edge in graph.edges if edge.edge_id == after.edge_id), None)
                if (before is None or operation.action.value != "edge_edit"
                        or list(operation.target_refs) != [before.edge_id]
                        or operation.before != before.model_dump(mode="json")
                        or after.model_dump(mode="json") != {**operation.before, "relation_type": after.relation_type.value}):
                    raise ValueError("recorded relation original edge mismatch")
                proposals.append({"relation": after.model_dump(mode="json"), "learningEvent": event,
                    "graphDelta": {**delta.model_dump(mode="json"), "content_hash": delta.delta_hash()}})
        return {
            "recordedRelationProposals": proposals,
            "reviewEvents": [event for events in histories.values() for event in events],
            "candidateGraph": graph.model_dump(mode="json", by_alias=True),
            "candidateGraphHash": graph.graph_hash(),
            "relationTypeIds": [item.value for item in CandidateRelationType],
            "extractionRunId": owner["extraction_run_id"],
        }
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/api/knowledge/builder/candidate-graphs/{candidate_graph_id}/relation-edits")
async def edit_knowledge_builder_candidate_graph_relation(candidate_graph_id: str, request: Request):
    """Persist a typed relation edit as candidate-only Human Grounding work."""
    route_path = "/api/knowledge/builder/candidate-graphs/{candidate_graph_id}/relation-edits"
    ctx = _authz(request, "POST", route_path)
    body = await _json_body(request)
    if (body.get("schemaVersion") or body.get("schema_version")) != "contextgraph_relation_edit_request.v1":
        raise HTTPException(status_code=400, detail="contextgraph_relation_edit_request.v1 required")
    edge_id = str(body.get("edgeId") or body.get("edge_id") or "").strip()
    relation_type = str(body.get("relationType") or body.get("relation_type") or "").strip()
    if not edge_id or not relation_type:
        raise HTTPException(status_code=400, detail="edgeId and relationType are required")
    store = _extraction_store()
    owner = _require_owner(
        store.get_candidate_graph_owner(candidate_graph_id),
        str(ctx.organization_id), str(ctx.workspace_id) if ctx.workspace_id else None,
        label="candidate graph",
    )
    grounding_store = HumanGroundingStore(_eos_path(), graph_backend=_CandidateOnlyPreviewBackend())
    try:
        graph = grounding_store.get_candidate_graph(candidate_graph_id)
        if "expectedGraphHash" in body and body["expectedGraphHash"] != graph.graph_hash():
            raise HTTPException(status_code=409, detail="candidate graph hash mismatch; reload before correction")
        edge = next((item for item in graph.edges if item.edge_id == edge_id), None)
        if edge is None:
            raise ValueError("candidate graph relation not found")
        concept_bundle = store.list_reference_concepts(extraction_run_id=owner["extraction_run_id"])
        concept = next(
            (
                item for item in concept_bundle["concepts"]
                if item.get("candidate_graph_id") == candidate_graph_id
                and set(item.get("source_anchor_refs") or ()).intersection(
                    anchor.anchor_id for anchor in edge.source_anchors
                )
            ),
            None,
        )
        if concept is None:
            raise ValueError("candidate graph relation is not bound to a reference concept")
        source_anchor_refs = tuple(
            body.get("evidenceAnchorRefs") or body.get("evidence_anchor_refs")
            or concept.get("source_anchor_refs") or ()
        )
        valid_refs = store.valid_reference_concept_evidence_anchor_refs(concept["concept_id"])
        if not source_anchor_refs or not set(source_anchor_refs).issubset(valid_refs):
            raise ValueError("relation edit evidence anchors are not canonical for the concept")
        before = edge.model_dump(mode="json")
        after = {**before, "relation_type": relation_type}
        event_id = f"human_event_{uuid4().hex}"
        history = store.list_reference_concept_review_histories(
            [concept["concept_id"]], organization_id=str(ctx.organization_id),
            workspace_id=str(ctx.workspace_id) if ctx.workspace_id else None,
        ).get(concept["concept_id"], [])
        delta = GraphDelta.model_validate({
            "graph_delta_id": f"delta_{uuid4().hex}",
            "candidate_graph_id": candidate_graph_id,
            "from_graph_version": owner.get("allowed_base_graph_version"),
            "to_graph_version": "KG_v1",
            "human_label_event_id": event_id,
            "justification": str(body.get("justification") or "Reviewer edited a proposed typed relation."),
            "operations": [{
                "operation_id": f"edit_{edge_id}", "action": "edge_edit",
                "target_refs": [edge_id], "before": before, "after": after,
                "source_anchor_refs": list(source_anchor_refs),
            }],
        })
        event = LearningEventEnvelope(
            event_id=event_id, case_id=concept["concept_id"],
            event_type=LearningEventType.HUMAN_EDIT,
            occurred_at=datetime.now(timezone.utc),
            source_document_hash=str(owner.get("source_document_hash") or graph.source_document_hash),
            source_anchor_refs=source_anchor_refs,
            graph_version=delta.from_graph_version or "KG_v0",
            rule_version="tender_rules.v1", evaluation_corpus_version="tender_sensitive_v1",
            actor_ref=f"user:{ctx.user_id}",
            human_label=HumanLabel(
                decision=HumanDecision.EDIT,
                certainty=str(body.get("certainty") or "certain"),
                reason_code="typed_relation_review",
                justification=delta.justification,
            ),
            graph_delta_ref=delta.graph_delta_id,
            prior_event_hash=history[-1]["event_hash"] if history else None,
        )
        result = HumanGroundingWorkbench(grounding_store).submit_review(
            human_event=event, delta=delta,
        )
        store.update_reference_concept_grounding_state(concept["concept_id"], "human_edited")
        return {"ok": True, "relation": after, "learningEvent": result["human_event"], "graphDelta": result["graph_delta"]}
    except (ValueError, ValidationError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _batch_human_event_id(
    *,
    batch_idempotency_key: str,
    organization_id: str,
    workspace_id: str | None,
    candidate_graph_id: str,
    concept_id: str,
    extract_run_ref: str,
    extract_result_hash: str,
) -> str:
    command = {
        "batch_idempotency_key": batch_idempotency_key,
        "candidate_graph_id": candidate_graph_id,
        "concept_id": concept_id,
        "extract_result_hash": extract_result_hash,
        "extract_run_ref": extract_run_ref,
        "organization_id": organization_id,
        "workspace_id": workspace_id,
    }
    return f"human_event_batch_{sha256(canonical_json(command).encode('utf-8')).hexdigest()}"


def _batch_response_id(event_ids: tuple[str, ...]) -> str:
    digest = sha256(
        canonical_json({"human_event_ids": sorted(event_ids)}).encode("utf-8")
    ).hexdigest()
    return f"ground_batch_{digest}"


@router.post("/api/knowledge/builder/reference-concepts/batch-grounding-events")
async def create_knowledge_builder_reference_concept_batch_grounding_events(
    request: Request,
):
    route_path = "/api/knowledge/builder/reference-concepts/batch-grounding-events"
    ctx = _authz(request, "POST", route_path)
    body = await _json_body(request)
    request_schema = body.get("schemaVersion") or body.get("schema_version")
    if request_schema not in {
        "learning_event_grounding_batch_request.v1",
        "learning_event_grounding_batch_request.v2",
    }:
        raise HTTPException(
            status_code=400,
            detail="learning_event_grounding_batch_request.v1 or v2 required",
        )
    if request_schema == "learning_event_grounding_batch_request.v2":
        allowed = {
            "schemaVersion",
            "schema_version",
            "batchIdempotencyKey",
            "batch_idempotency_key",
            "decision",
            "items",
            "certainty",
            "reasonCode",
            "reason_code",
            "justification",
            "extractRunRef",
            "extract_run_ref",
            "extractResultHash",
            "extract_result_hash",
        }
        if set(body) - allowed:
            raise HTTPException(
                status_code=400,
                detail="learning_event_grounding_batch_request.v2 contains unknown fields",
            )
        for camel, snake in (
            ("schemaVersion", "schema_version"),
            ("batchIdempotencyKey", "batch_idempotency_key"),
            ("reasonCode", "reason_code"),
            ("extractRunRef", "extract_run_ref"),
            ("extractResultHash", "extract_result_hash"),
        ):
            if camel in body and snake in body:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "learning_event_grounding_batch_request.v2 contains "
                        f"duplicate aliases: {camel}, {snake}"
                    ),
                )
        batch_idempotency_key = str(
            body.get("batchIdempotencyKey")
            or body.get("batch_idempotency_key")
            or ""
        ).strip()
        if not batch_idempotency_key:
            raise HTTPException(
                status_code=400,
                detail="batchIdempotencyKey is required for v2 batch grounding",
            )
    else:
        batch_idempotency_key = None
    if str(body.get("decision") or "") != "accept":
        raise HTTPException(
            status_code=400,
            detail="reference concept batch grounding supports accept only",
        )
    store = _extraction_store()
    workspace_id = str(ctx.workspace_id) if ctx.workspace_id else None
    events = []
    extraction_run_id: str | None = None
    candidate_graph_id: str | None = None
    review_graph_version: str | None = None
    occurred_at = datetime.now(timezone.utc)
    try:
        for item in body.get("items") or []:
            concept_id = str(
                item.get("assertionId")
                or item.get("conceptId")
                or item.get("concept_id")
                or ""
            )
            owner = _require_owner(
                store.get_reference_concept_owner(concept_id),
                str(ctx.organization_id),
                workspace_id,
                label="reference concept",
            )
            if extraction_run_id is None:
                extraction_run_id = owner["extraction_run_id"]
            elif extraction_run_id != owner["extraction_run_id"]:
                raise ValueError("batch concepts must belong to one extraction run")
            if candidate_graph_id is None:
                candidate_graph_id = owner["candidate_graph_id"]
                candidate_owner = store.get_candidate_graph_owner(candidate_graph_id)
                if candidate_owner is None:
                    raise ValueError("candidate graph not found")
                review_graph_version = str(
                    candidate_owner.get("allowed_base_graph_version") or "KG_v0"
                )
            elif candidate_graph_id != owner["candidate_graph_id"]:
                raise ValueError("batch concepts must belong to one candidate graph")
            concept = store.get_reference_concept(concept_id)
            if concept is None:
                raise ValueError("reference concept not found")
            source_refs = tuple(
                item.get("evidenceAnchorRefs")
                or item.get("evidence_anchor_refs")
                or concept.get("source_anchor_refs")
                or ()
            )
            if not source_refs or not set(source_refs).issubset(
                store.valid_reference_concept_evidence_anchor_refs(concept_id)
            ):
                raise ValueError(
                    "batch evidence anchor refs are not canonical evidence"
                )
            history = store.list_reference_concept_review_histories(
                [concept_id],
                organization_id=str(ctx.organization_id),
                workspace_id=workspace_id,
            ).get(concept_id, [])
            events.append(
                LearningEventEnvelope(
                    event_id=(
                        _batch_human_event_id(
                            batch_idempotency_key=batch_idempotency_key,
                            organization_id=str(ctx.organization_id),
                            workspace_id=workspace_id,
                            candidate_graph_id=str(owner["candidate_graph_id"]),
                            concept_id=concept_id,
                            extract_run_ref=str(
                                body.get("extractRunRef")
                                or body.get("extract_run_ref")
                                or ""
                            ),
                            extract_result_hash=str(
                                body.get("extractResultHash")
                                or body.get("extract_result_hash")
                                or ""
                            ),
                        )
                        if batch_idempotency_key is not None
                        else f"human_event_{uuid4().hex}"
                    ),
                    case_id=concept_id,
                    event_type=LearningEventType.HUMAN_ACCEPT,
                    occurred_at=occurred_at,
                    source_document_hash=str(
                        owner.get("source_document_hash")
                        or concept.get("source_document_hash")
                    ),
                    source_anchor_refs=source_refs,
                    graph_version=str(review_graph_version or "KG_v0"),
                    rule_version="tender_rules.v1",
                    evaluation_corpus_version="tender_sensitive_v1",
                    actor_ref=f"user:{ctx.user_id}",
                    human_label=HumanLabel(
                        decision=HumanDecision.ACCEPT,
                        certainty=str(body.get("certainty") or ""),
                        reason_code=str(
                            body.get("reasonCode") or body.get("reason_code") or ""
                        ),
                        justification=str(body.get("justification") or ""),
                    ),
                    prior_event_hash=history[-1]["event_hash"] if history else None,
                )
            )
        if extraction_run_id is None:
            raise ValueError("batch must contain concepts")
        if not candidate_graph_id:
            raise ValueError("batch concepts are not bound to a candidate graph")
        candidate_owner = store.get_candidate_graph_owner(candidate_graph_id)
        if candidate_owner is None:
            raise ValueError("candidate graph not found")
        canonical_journey = (
            candidate_owner.get("workflow_kind") == "reference_graph_build"
        )
        if (
            canonical_journey
            and request_schema != "learning_event_grounding_batch_request.v2"
        ):
            raise ValueError("canonical graph builds require exact v3 release lineage")
        lineage_inputs = None
        if request_schema == "learning_event_grounding_batch_request.v2":
            lineage_inputs = _resolve_v3_release_inputs(
                body={
                    "schemaVersion": "accepted_graph_release_request.v2",
                    "humanEventId": "batch-pending",
                    "extractRunRef": body.get("extractRunRef")
                    or body.get("extract_run_ref"),
                    "extractResultHash": body.get("extractResultHash")
                    or body.get("extract_result_hash"),
                },
                ctx=ctx,
                candidate_graph_id=candidate_graph_id,
            )
            events.sort(key=lambda event: event.case_id)
        release_store = HumanGroundingStore(
            _eos_path(), graph_backend=_release_graph_backend()
        )
        event_ids = tuple(event.event_id for event in events)
        persisted: list[dict[str, Any]] | None = None
        if batch_idempotency_key is not None:
            command_hash = content_hash(
                {
                    "actor_ref": f"user:{ctx.user_id}",
                    "batch_idempotency_key": batch_idempotency_key,
                    "candidate_graph_id": candidate_graph_id,
                    "certainty": str(body.get("certainty") or ""),
                    "decision": "accept",
                    "extract_result_hash": str(
                        body.get("extractResultHash")
                        or body.get("extract_result_hash")
                        or ""
                    ),
                    "extract_run_ref": str(
                        body.get("extractRunRef")
                        or body.get("extract_run_ref")
                        or ""
                    ),
                    "items": [
                        {
                            "concept_id": event.case_id,
                            "source_anchor_refs": sorted(event.source_anchor_refs),
                        }
                        for event in events
                    ],
                    "justification": str(body.get("justification") or ""),
                    "organization_id": str(ctx.organization_id),
                    "reason_code": str(
                        body.get("reasonCode") or body.get("reason_code") or ""
                    ),
                    "workspace_id": workspace_id,
                }
            )
            persisted = release_store.append_reference_review_batch(
                events=tuple(events),
                batch_id=_batch_response_id(event_ids),
                organization_id=str(ctx.organization_id),
                workspace_id=workspace_id,
                candidate_graph_id=candidate_graph_id,
                idempotency_key=batch_idempotency_key,
                command_hash=command_hash,
            )
        if lineage_inputs is not None:
            if persisted is None:
                raise ValueError("v2 batch release requires persisted review events")
            candidate, decision, extract_result, _candidate_graph = lineage_inputs
            relation_occurrences = extract_result.output.get("relation_occurrences")
            if not isinstance(relation_occurrences, list):
                raise ValueError(
                    "v2 batch release requires governed relation occurrences"
                )
            decided_at = str(persisted[0]["occurred_at"])
            existing_occurrences = {
                item["extracted_occurrence_ref"]: item
                for item in release_store.list_relation_occurrence_groundings(
                    extract_result_ref=extract_result.extract_run_id,
                    organization_id=str(ctx.organization_id), workspace_id=str(workspace_id),
                )
            }
            for occurrence in relation_occurrences:
                existing = existing_occurrences.get(str(occurrence["occurrence_ref"]))
                if existing is not None:
                    if (
                        existing["outcome"] != "accepted"
                        or existing["extract_result_hash"] != extract_result.extract_result_hash
                        or existing["schema_decision_ref"] != decision.schema_decision_id
                        or existing["schema_decision_hash"] != decision.schema_decision_hash
                        or existing["extracted_occurrence_hash"] != occurrence["occurrence_hash"]
                    ):
                        raise ValueError("existing occurrence Grounding does not accept exact batch inputs")
                    continue
                release_store.put_relation_occurrence_grounding(
                    {
                        "organization_id": str(ctx.organization_id),
                        "workspace_id": workspace_id,
                        "extract_result_ref": extract_result.extract_run_id,
                        "extract_result_hash": extract_result.extract_result_hash,
                        "schema_decision_ref": decision.schema_decision_id,
                        "schema_decision_hash": decision.schema_decision_hash,
                        "extracted_occurrence_ref": str(
                            occurrence["occurrence_ref"]
                        ),
                        "outcome": "accepted",
                        "corrected_occurrence_ref": None,
                        "corrected_occurrence_hash": None,
                        "supersedes_occurrence_ref": None,
                        "supersedes_occurrence_hash": None,
                        "relation_usage_grounding_decision_ref": None,
                        "relation_usage_grounding_decision_hash": None,
                        "actor_ref": f"user:{ctx.user_id}",
                        "decided_at": decided_at,
                    },
                    extract_result=extract_result,
                )
            bundle = _materialize_v2_release_from_exact_inputs(
                lineage_inputs=lineage_inputs,
                candidate_graph_id=candidate_graph_id,
                human_event_ids=tuple(item["event_id"] for item in persisted),
                ctx=ctx,
            )
            release = bundle["release"]
            lineage = AcceptedGraphReleaseLineageV2.parse_unverified(
                bundle["lineage"]
            )
            artifact = _checkpoint_accepted_release_artifact(
                release=release, ctx=ctx, release_lineage=lineage,
                semantic_bundle=bundle,
            )
            for event in events:
                store.update_reference_concept_grounding_state(
                    event.case_id, "human_accepted"
                )
            return {
                "ok": True,
                "schemaVersion": "learning_event_grounding_batch_response.v2",
                "authorityState": "accepted_graph_released",
                "batchId": _batch_response_id(event_ids),
                "extractionRunId": extraction_run_id,
                "events": persisted, "graphRelease": release,
                "graphArtifact": artifact, "releaseLineage": bundle["lineage"],
                "occurrenceGroundingManifest": bundle[
                    "occurrence_grounding_manifest"
                ],
                "canonicalPatternManifest": bundle[
                    "canonical_pattern_manifest"
                ],
            }
        release_state = release_store.get_release_state_for_reviews(
            candidate_graph_id=candidate_graph_id,
            human_event_ids=event_ids,
            organization_id=str(ctx.organization_id),
            workspace_id=workspace_id,
        )
        if release_state is not None:
            release = release_state["release"]
            if release_state["status"] != "materialized":
                release = release_store.retry_graph_release(
                    str(release["graph_version"]),
                    organization_id=str(ctx.organization_id),
                    workspace_id=workspace_id,
                )
            if persisted is None:
                persisted = []
                for event_id in event_ids:
                    existing_event = release_store.get_learning_event(event_id)
                    if existing_event is None:
                        raise ValueError("materialized release review event not found")
                    persisted.append(existing_event)
        else:
            if persisted is None:
                grounding_store = HumanGroundingStore(
                    _eos_path(), graph_backend=_CandidateOnlyPreviewBackend()
                )
                persisted = []
                for event in events:
                    review = HumanGroundingWorkbench(grounding_store).submit_review(
                        human_event=event,
                        delta=None,
                    )
                    persisted.append(review["human_event"])
            for event in events:
                store.update_reference_concept_grounding_state(
                    event.case_id, "human_accepted"
                )
            release = release_store.materialize_reviewed_candidates(
                candidate_graph_id=candidate_graph_id,
                human_event_ids=tuple(event["event_id"] for event in persisted),
                organization_id=str(ctx.organization_id),
                workspace_id=workspace_id,
            )
        artifact = _checkpoint_accepted_release_artifact(
            release=release,
            ctx=ctx,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "ok": True,
        "schemaVersion": "learning_event_grounding_batch_response.v1",
        "authorityState": "accepted_graph_released",
        "batchId": (
            _batch_response_id(event_ids)
            if batch_idempotency_key is not None
            else f"ground_batch_{uuid4().hex}"
        ),
        "extractionRunId": extraction_run_id,
        "events": persisted,
        "graphRelease": release,
        "graphArtifact": artifact,
    }


@router.post("/api/knowledge/builder/reference-concepts/{concept_id}/release")
async def release_knowledge_builder_reference_concept(
    concept_id: str, request: Request
):
    route_path = "/api/knowledge/builder/reference-concepts/{concept_id}/release"
    ctx = _authz(request, "POST", route_path)
    body = await _json_body(request)
    request_schema = body.get("schemaVersion") or body.get("schema_version")
    if request_schema not in {
        "accepted_graph_release_request.v1",
        "accepted_graph_release_request.v2",
    }:
        raise HTTPException(
            status_code=400,
            detail="accepted_graph_release_request.v1 or v2 required",
        )
    human_event_id = str(
        body.get("humanEventId") or body.get("human_event_id") or ""
    ).strip()
    if not human_event_id:
        raise HTTPException(status_code=400, detail="humanEventId is required")
    extraction = _extraction_store()
    owner = _require_owner(
        extraction.get_reference_concept_owner(concept_id),
        str(ctx.organization_id),
        str(ctx.workspace_id) if ctx.workspace_id else None,
        label="reference concept",
    )
    candidate_graph_id = str(owner.get("candidate_graph_id") or "")
    if not candidate_graph_id:
        raise HTTPException(
            status_code=409,
            detail="reference concept is not bound to a candidate graph",
        )
    store = HumanGroundingStore(_eos_path(), graph_backend=_release_graph_backend())
    try:
        if (
            owner.get("workflow_kind") == "reference_graph_build"
            and request_schema != "accepted_graph_release_request.v2"
        ):
            raise ValueError("canonical graph builds require exact v3 release lineage")
        lineage_inputs = (
            _resolve_v3_release_inputs(
                body=body,
                ctx=ctx,
                candidate_graph_id=candidate_graph_id,
            )
            if request_schema == "accepted_graph_release_request.v2"
            else None
        )
        if lineage_inputs is not None:
            bundle = _materialize_v2_release_from_exact_inputs(
                lineage_inputs=lineage_inputs,
                candidate_graph_id=candidate_graph_id,
                human_event_ids=(human_event_id,),
                ctx=ctx,
            )
            release = bundle["release"]
            release_lineage = AcceptedGraphReleaseLineageV2.parse_unverified(
                bundle["lineage"]
            )
            artifact = _checkpoint_accepted_release_artifact(
                release=release, ctx=ctx, release_lineage=release_lineage,
                semantic_bundle=bundle,
            )
            return {
                "ok": True, "graphRelease": release, "graphArtifact": artifact,
                "releaseLineage": bundle["lineage"],
                "occurrenceGroundingManifest": bundle[
                    "occurrence_grounding_manifest"
                ],
                "canonicalPatternManifest": bundle[
                    "canonical_pattern_manifest"
                ],
            }
        release_state = store.get_release_state_for_reviews(
            candidate_graph_id=candidate_graph_id,
            human_event_ids=(human_event_id,),
            organization_id=str(ctx.organization_id),
            workspace_id=str(ctx.workspace_id) if ctx.workspace_id else None,
        )
        if release_state is None:
            release = store.materialize_reviewed_candidate(
                candidate_graph_id=candidate_graph_id,
                human_event_id=human_event_id,
                organization_id=str(ctx.organization_id),
                workspace_id=str(ctx.workspace_id) if ctx.workspace_id else None,
            )
        else:
            release = release_state["release"]
            if release_state["status"] != "materialized":
                release = store.retry_graph_release(
                    str(release["graph_version"]),
                    organization_id=str(ctx.organization_id),
                    workspace_id=str(ctx.workspace_id) if ctx.workspace_id else None,
                )
        artifact = _checkpoint_accepted_release_artifact(
            release=release,
            ctx=ctx,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"ok": True, "graphRelease": release, "graphArtifact": artifact}


@router.get("/api/knowledge/builder/assertion-candidates/{assertion_id}")
async def get_knowledge_builder_assertion_candidate(assertion_id: str, request: Request):
    route_path = "/api/knowledge/builder/assertion-candidates/{assertion_id}"
    ctx = _authz(request, "GET", route_path)
    store = _extraction_store()
    owner = _require_owner(store.get_assertion_owner(assertion_id), str(ctx.organization_id), str(ctx.workspace_id) if ctx.workspace_id else None, label="assertion candidate")
    assertion = store.get_assertion(assertion_id)
    occurrences = store.list_assertion_occurrences(assertion_id)
    topologies = store.get_assertion_topologies(assertion_id)
    return {
        "ok": True,
        "schemaVersion": "knowledge_builder_assertion_candidate_response.v1",
        "assertionCandidate": assertion.model_dump(mode="json") if assertion else None,
        "occurrences": [item.model_dump(mode="json") for item in occurrences],
        "evidenceObligations": store.list_evidence_obligations(assertion_id),
        "learningEvents": store.list_events(assertion_id),
        "candidateTopology": topologies["candidate_topology"],
        "acceptedTopology": topologies["accepted_topology"],
    }


@router.post("/api/knowledge/builder/assertion-candidates/{assertion_id}/grounding-events")
async def create_knowledge_builder_grounding_event(assertion_id: str, request: Request):
    route_path = "/api/knowledge/builder/assertion-candidates/{assertion_id}/grounding-events"
    ctx = _authz(request, "POST", route_path)
    body = await _json_body(request)
    schema_version = body.get("schemaVersion") or body.get("schema_version")
    if schema_version != "learning_event_grounding_request.v1":
        raise HTTPException(status_code=400, detail="learning_event_grounding_request.v1 required")
    try:
        decision = HumanDecision(str(body.get("decision") or ""))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    edited_assertion_present = "editedAssertion" in body or "edited_assertion" in body
    edited_assertion = (
        body["editedAssertion"] if "editedAssertion" in body else body.get("edited_assertion")
    )
    if decision is HumanDecision.EDIT:
        if not edited_assertion_present or not isinstance(edited_assertion, dict):
            raise HTTPException(
                status_code=400, detail="edit review requires an edited assertion object"
            )
    elif edited_assertion_present:
        raise HTTPException(
            status_code=400, detail="edited assertion is only valid for edit review"
        )
    store = _extraction_store()
    _require_owner(store.get_assertion_owner(assertion_id), str(ctx.organization_id), str(ctx.workspace_id) if ctx.workspace_id else None, label="assertion candidate")
    assertion = store.get_assertion(assertion_id)
    if assertion is None:
        raise HTTPException(status_code=404, detail="assertion candidate not found")
    try:
        event_type = {
            HumanDecision.ACCEPT: LearningEventType.HUMAN_ACCEPT,
            HumanDecision.EDIT: LearningEventType.HUMAN_EDIT,
            HumanDecision.REJECT: LearningEventType.HUMAN_REJECT,
            HumanDecision.UNCERTAIN: LearningEventType.HUMAN_UNCERTAIN,
        }[decision]
        label = HumanLabel(
            decision=decision,
            certainty=str(body.get("certainty") or ""),
            reason_code=str(body.get("reasonCode") or body.get("reason_code") or ""),
            justification=str(body.get("justification") or ""),
            process_labels=body.get("processLabels") or body.get("process_labels") or {},
        )
        history = store.list_events(assertion_id)
        event_id = f"human_event_{uuid4().hex}"
        graph_delta_payload = body.get("graphDelta") or body.get("graph_delta")
        if graph_delta_payload:
            graph_delta_input = dict(graph_delta_payload)
            submitted_from_graph_version = graph_delta_input.get("from_graph_version")
            graph_delta_input["human_label_event_id"] = event_id
            graph_delta_input["justification"] = label.justification
            candidate_owner = store.get_candidate_graph_owner(assertion.candidate_graph_id)
            if candidate_owner is None or candidate_owner.get("organization_id") != str(ctx.organization_id):
                raise ValueError("candidate graph not found for organization")
            if candidate_owner.get("workspace_id") != (str(ctx.workspace_id) if ctx.workspace_id else None):
                raise ValueError("candidate graph not found for workspace")
            graph_delta_input["from_graph_version"] = candidate_owner.get("allowed_base_graph_version")
            graph_delta = GraphDelta.model_validate(graph_delta_input)
        else:
            graph_delta = None
        if graph_delta and graph_delta.candidate_graph_id != assertion.candidate_graph_id:
            raise ValueError("GraphDelta candidate graph does not match assertion candidate")
        if "evidenceAnchorRefs" in body:
            source_anchor_refs = tuple(body.get("evidenceAnchorRefs") or ())
        elif "evidence_anchor_refs" in body:
            source_anchor_refs = tuple(body.get("evidence_anchor_refs") or ())
        else:
            source_anchor_refs = tuple(anchor.anchor_id for anchor in assertion.source_anchors)
        if decision in {HumanDecision.ACCEPT, HumanDecision.EDIT} and not source_anchor_refs:
            raise ValueError("accept/edit review requires non-empty evidence anchor refs")
        valid_evidence_refs = store.valid_evidence_anchor_refs(assertion_id)
        if not set(source_anchor_refs).issubset(valid_evidence_refs):
            raise ValueError("review evidence anchor refs are not canonical evidence for assertion")
        if decision in {HumanDecision.ACCEPT, HumanDecision.EDIT} and graph_delta:
            evidence_refs = set(source_anchor_refs)
            for operation in graph_delta.operations:
                if not operation.source_anchor_refs:
                    raise ValueError("GraphDelta operation source anchor refs are required")
                if not set(operation.source_anchor_refs).issubset(evidence_refs):
                    raise ValueError(
                        "GraphDelta operation source anchor refs must be bound to review evidence"
                    )
            supplied_preview_hash = body.get("graphDeltaPreviewHash") or body.get("graph_delta_preview_hash")
            if not supplied_preview_hash:
                raise ValueError("accept/edit review requires graph delta preview hash")
            current_preview = _graph_delta_preview(assertion, store)
            if (not current_preview.get("available") or
                    supplied_preview_hash != current_preview.get("previewHash")):
                raise ValueError("graph delta preview is stale; reload before grounding")
            if list(source_anchor_refs) != current_preview.get("evidenceAnchorRefs"):
                raise ValueError("submitted evidence anchors differ from canonical preview")
            canonical_delta = current_preview["graphDelta"]
            if decision is HumanDecision.EDIT:
                graph_delta = _canonical_edit_delta(
                    assertion=assertion,
                    edited_assertion=edited_assertion,
                    event_id=event_id,
                    justification=label.justification,
                    source_anchor_refs=source_anchor_refs,
                    to_graph_version=canonical_delta["to_graph_version"],
                )
            else:
                submitted_projection = {
                    "candidate_graph_id": graph_delta_input.get("candidate_graph_id"),
                    "from_graph_version": submitted_from_graph_version,
                    "operations": graph_delta_input.get("operations"),
                }
                canonical_projection = {
                    "candidate_graph_id": canonical_delta["candidate_graph_id"],
                    "from_graph_version": canonical_delta.get("from_graph_version"),
                    "operations": canonical_delta["operations"],
                }
                if content_hash(submitted_projection) != content_hash(canonical_projection):
                    raise ValueError("submitted GraphDelta differs from canonical preview")
                graph_delta_input["graph_delta_id"] = f"delta_{uuid4().hex}"
                graph_delta_input["to_graph_version"] = canonical_delta["to_graph_version"]
                graph_delta = GraphDelta.model_validate(graph_delta_input)
        event = LearningEventEnvelope(
            event_id=event_id,
            case_id=assertion_id,
            event_type=event_type,
            occurred_at=datetime.now(timezone.utc),
            source_document_hash=assertion.source_document_hash,
            source_anchor_refs=source_anchor_refs,
            graph_version=graph_delta.from_graph_version if graph_delta and graph_delta.from_graph_version else "KG_v0",
            rule_version="tender_rules.v1",
            evaluation_corpus_version="tender_sensitive_v1",
            actor_ref=f"user:{ctx.user_id}",
            human_label=label,
            graph_delta_ref=graph_delta.graph_delta_id if graph_delta else None,
            downstream_outcome=(
                {"edited_assertion": edited_assertion}
                if decision is HumanDecision.EDIT else None
            ),
            prior_event_hash=history[-1]["event_hash"] if history else None,
        )
        if decision is HumanDecision.ACCEPT and graph_delta is not None:
            raise ValueError("accept without edit cannot persist a GraphDelta")
        if decision is HumanDecision.EDIT and graph_delta is None:
            raise ValueError("edit review requires a GraphDelta")
        grounding_store = HumanGroundingStore(
            _eos_path(), graph_backend=_CandidateOnlyPreviewBackend()
        )
        result = HumanGroundingWorkbench(grounding_store).submit_review(
            human_event=event, delta=graph_delta
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "learningEvent": result["human_event"],
            "graphDelta": result.get("graph_delta"), "graphRelease": result.get("release")}


@router.post("/api/knowledge/builder/graph-releases/{graph_version}/retry-materialization")
async def retry_knowledge_builder_graph_release(graph_version: str, request: Request):
    route_path = "/api/knowledge/builder/graph-releases/{graph_version}/retry-materialization"
    ctx = _authz(request, "POST", route_path)
    # Use the configured external backend when present, otherwise the governed
    # in-process FalkorDBLite backend. Retry must follow the same selection as
    # the initial authenticated release path.
    graph_backend = _release_graph_backend()
    store = HumanGroundingStore(_eos_path(), graph_backend=graph_backend)
    try:
        lineage_contract = store.get_release_lineage_contract(
            graph_version,
            organization_id=str(ctx.organization_id),
            workspace_id=str(ctx.workspace_id) if ctx.workspace_id else None,
        )
        if lineage_contract == "exact_v4_v3":
            bundle = _retry_v2_release_from_exact_lineage(
                graph_version=graph_version,
                store=store,
                graph_backend=graph_backend,
                ctx=ctx,
            )
            release_lineage_v2 = AcceptedGraphReleaseLineageV2.parse_unverified(
                bundle["lineage"]
            )
            artifact = _checkpoint_accepted_release_artifact(
                release=bundle["release"],
                ctx=ctx,
                release_lineage=release_lineage_v2,
                semantic_bundle=bundle,
            )
            return {
                "ok": True,
                "graphRelease": bundle["release"],
                "graphArtifact": artifact,
            }
        release_lineage = None
        try:
            retry_preflight = _extraction_store().get_release_artifact_origin_lineage(
                graph_version=graph_version,
                organization_id=str(ctx.organization_id),
                workspace_id=str(ctx.workspace_id) if ctx.workspace_id else "",
                workspace_root=ctx.workspace_root,
                require_materialized_checkpoint=False,
            )
        except ValueError as exc:
            if (
                lineage_contract == "legacy"
                and str(exc) == "accepted release artifact origin not found"
            ):
                retry_preflight = None
            else:
                raise
        if retry_preflight is not None:
            if lineage_contract == "legacy":
                if retry_preflight["provenance_status"] != "legacy":
                    raise ValueError("legacy release lineage contract mismatch")
            elif retry_preflight["provenance_status"] != "validated_v3_retry":
                raise ValueError("exact v3 release lineage is unavailable")
            else:
                release_lineage = AcceptedGraphReleaseLineage.from_dict(
                    retry_preflight["lineage"]
                )
        elif lineage_contract == "exact_v3":
            raise ValueError("exact v3 release lineage is unavailable")
        release = store.retry_graph_release(
            graph_version, organization_id=str(ctx.organization_id),
            workspace_id=str(ctx.workspace_id) if ctx.workspace_id else None,
        )
        artifact = (
            _checkpoint_accepted_release_artifact(
                release=release,
                ctx=ctx,
                release_lineage=release_lineage,
            )
            if release_lineage is not None
            else None
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"ok": True, "graphRelease": release, "graphArtifact": artifact}


@router.post("/api/knowledge/builder/assertion-candidates/{assertion_id}/release")
async def release_knowledge_builder_assertion_candidate(assertion_id: str, request: Request):
    route_path = "/api/knowledge/builder/assertion-candidates/{assertion_id}/release"
    ctx = _authz(request, "POST", route_path)
    body = await _json_body(request)
    request_schema = body.get("schemaVersion") or body.get("schema_version")
    if request_schema not in {
        "accepted_graph_release_request.v1", "accepted_graph_release_request.v2",
    }:
        raise HTTPException(
            status_code=400,
            detail="accepted_graph_release_request.v1 or v2 required",
        )
    human_event_id = str(body.get("humanEventId") or body.get("human_event_id") or "").strip()
    if not human_event_id:
        raise HTTPException(status_code=400, detail="humanEventId is required")
    extraction = _extraction_store()
    _require_owner(
        extraction.get_assertion_owner(assertion_id), str(ctx.organization_id),
        str(ctx.workspace_id) if ctx.workspace_id else None, label="assertion candidate",
    )
    assertion = extraction.get_assertion(assertion_id)
    if assertion is None:
        raise HTTPException(status_code=404, detail="assertion candidate not found")
    store = HumanGroundingStore(_eos_path(), graph_backend=_release_graph_backend())
    try:
        if request_schema == "accepted_graph_release_request.v2":
            lineage_inputs = _resolve_v3_release_inputs(
                body=body, ctx=ctx,
                candidate_graph_id=assertion.candidate_graph_id,
            )
            if (
                lineage_inputs[0].schema_version
                == "corpus_graph_schema_candidate.v4"
                and lineage_inputs[1].schema_version == "schema_decision.v3"
            ):
                bundle = _materialize_v2_release_from_exact_inputs(
                    lineage_inputs=lineage_inputs,
                    candidate_graph_id=assertion.candidate_graph_id,
                    human_event_ids=(human_event_id,), ctx=ctx,
                )
                release_lineage = AcceptedGraphReleaseLineageV2.parse_unverified(
                    bundle["lineage"]
                )
                artifact = _checkpoint_accepted_release_artifact(
                    release=bundle["release"], ctx=ctx,
                    release_lineage=release_lineage,
                    semantic_bundle=bundle,
                )
                return {
                    "ok": True, "graphRelease": bundle["release"],
                    "graphArtifact": artifact,
                    "releaseLineage": bundle["lineage"],
                    "occurrenceGroundingManifest": bundle[
                        "occurrence_grounding_manifest"
                    ],
                    "canonicalPatternManifest": bundle[
                        "canonical_pattern_manifest"
                    ],
                }
            raise ValueError(
                "assertion v2 release requires Candidate.v4/Decision.v3"
            )
        release = store.materialize_reviewed_candidate(
            candidate_graph_id=assertion.candidate_graph_id,
            human_event_id=human_event_id,
            organization_id=str(ctx.organization_id),
            workspace_id=str(ctx.workspace_id) if ctx.workspace_id else None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"ok": True, "graphRelease": release}


@router.post("/api/knowledge/builder/graph-releases/{graph_version}/project-activate")
async def project_activate_knowledge_builder_graph_release(graph_version: str, request: Request):
    """Project one accepted release into the governed tender activation chain."""
    route_path = "/api/knowledge/builder/graph-releases/{graph_version}/project-activate"
    ctx = _authz(request, "POST", route_path)
    actor_authority = str(ctx.member_role or "").strip()
    body = await _json_body(request)
    effective_at = str(body.get("effectiveAt") or body.get("effective_at") or "").strip()
    if not effective_at:
        raise HTTPException(status_code=400, detail="effectiveAt is required")
    store = HumanGroundingStore(_eos_path(), graph_backend=_release_graph_backend())
    try:
        release = store.get_graph_release(
            graph_version, organization_id=str(ctx.organization_id),
            workspace_id=str(ctx.workspace_id) if ctx.workspace_id else None,
        )
        if "expectedReleaseHash" in body and body["expectedReleaseHash"] != release.get("release_hash"):
            raise HTTPException(status_code=409, detail="accepted release hash mismatch; reload before activation")
        service = AcceptedGraphRuleProjectionService(_eos_path())
        projected = service.project(
            release=release,
            organization_id=str(ctx.organization_id),
            workspace_id=str(ctx.workspace_id) if ctx.workspace_id else None,
            actor=f"user:{ctx.user_id}",
            actor_authority=actor_authority,
        )
        activation = service.activate(
            projected=projected,
            organization_id=str(ctx.organization_id),
            workspace_id=str(ctx.workspace_id) if ctx.workspace_id else None,
            actor=f"user:{ctx.user_id}",
            actor_authority=actor_authority,
            principal_entitlement_ref=str(body.get("principalEntitlementRef") or f"membership:{ctx.organization_id}:{ctx.user_id}"),
            effective_at=effective_at,
            activation_resolver_policy_version=str(body.get("activationResolverPolicyVersion") or "knowledge_activation_resolver.v1"),
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"ok": True, "graphRelease": release, "projection": activation}


@router.get("/api/knowledge/builder/tender-packages/{package_id}")
async def get_knowledge_builder_tender_package(package_id: str, request: Request):
    route_path = "/api/knowledge/builder/tender-packages/{package_id}"
    ctx = _authz(request, "GET", route_path)
    scope = _extraction_store().get_knowledge_scope(package_id)
    _require_owner(scope, str(ctx.organization_id), str(ctx.workspace_id) if ctx.workspace_id else None, label="tender package")
    try:
        hydrated = _hydrate_scope(scope, KnowledgeBuilderStore(_eos_path()))
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"ok": True, "schemaVersion": "knowledge_builder_tender_package_response.v1",
            "tenderPackage": hydrated}


@router.post("/api/knowledge/builder/tender-packages")
async def create_knowledge_builder_tender_package(request: Request):
    route_path = "/api/knowledge/builder/tender-packages"
    ctx = _authz(request, "POST", route_path)
    body = await _json_body(request)
    if (body.get("schemaVersion") or body.get("schema_version")) != "knowledge_builder_tender_package_request.v1":
        raise HTTPException(status_code=400, detail="knowledge_builder_tender_package_request.v1 required")
    organization_id = str(ctx.organization_id)
    workspace_id = str(ctx.workspace_id) if ctx.workspace_id else None
    discovery_run_id = str(body.get("discoveryRunId") or body.get("discovery_run_id") or "")
    builder = KnowledgeBuilderStore(_eos_path())
    discovery = builder.get_discovery_run(discovery_run_id)
    try:
        if (discovery is None or discovery.get("organization_id") != organization_id or
                discovery.get("workspace_id") != workspace_id):
            raise ValueError("discovery run not found")
        raw_documents = body.get("documents")
        if not isinstance(raw_documents, list):
            raise ValueError("documents must be an array")
        documents = []
        for raw in raw_documents:
            if not isinstance(raw, dict):
                raise ValueError("each document must be an object")
            if any(key in raw for key in ("text", "sourceText", "source_text", "elements")):
                raise ValueError("client document text/elements are not accepted")
            source_id = raw.get("sourceId") or raw.get("source_id")
            document_discovery_id = raw.get("discoveryRunId") or raw.get("discovery_run_id")
            canonical_document_id = raw.get("canonicalDocumentId") or raw.get("canonical_document_id")
            if sum(bool(item) for item in (source_id, document_discovery_id, canonical_document_id)) != 1:
                raise ValueError("document must reference exactly one sourceId, discoveryRunId, or canonicalDocumentId")
            compilation_anchors = []
            if canonical_document_id:
                compilation_store = KnowledgeSourceCompilationStore(_eos_path())
                compilation = compilation_store.get_run(str(canonical_document_id))
                if compilation is None:
                    raise ValueError("canonical document compilation not found")
                if (compilation["organization_id"] != organization_id or
                        compilation.get("workspace_id") != workspace_id):
                    raise ValueError("canonical document compilation not found")
                governed_source_id = compilation["knowledge_source_id"]
                mapped_source = builder.find_source_by_governed_id(governed_source_id)
                source_id = mapped_source["source_id"] if mapped_source else None
                compilation_anchors = compilation_store.list_anchors(str(canonical_document_id))
                mapped_refs = set(compilation.get("mapped_source_anchor_refs") or ())
                actual_refs = {item["compilation_anchor_ref"] for item in compilation_anchors}
                if mapped_refs != actual_refs:
                    raise ValueError("canonical document compilation anchor manifest mismatch")
            if document_discovery_id:
                document_discovery = builder.get_discovery_run(str(document_discovery_id))
                if (document_discovery is None or
                        document_discovery.get("organization_id") != organization_id or
                        document_discovery.get("workspace_id") != workspace_id):
                    raise ValueError("document discovery run not found")
                source_id = document_discovery["source_id"]
            if not source_id:
                raise ValueError("document sourceId or discoveryRunId is required")
            source = builder.get_source(str(source_id))
            if (source is None or source.get("organization_id") != organization_id or
                    source.get("workspace_id") != workspace_id):
                raise ValueError("document source not found")
            if canonical_document_id and (
                compilation["knowledge_source_id"] != (source.get("source_metadata") or {}).get("knowledge_source_id") or
                compilation["source_hash"] != source["source_hash"]
            ):
                raise ValueError("canonical document compilation source identity/hash mismatch")
            expected_hash = raw.get("documentHash") or raw.get("document_hash")
            if expected_hash and expected_hash != source["source_hash"]:
                raise ValueError("document hash mismatch")
            document_id = str(raw.get("documentId") or raw.get("document_id") or "")
            anchor_key = f"{source['source_id']}:{source['source_hash']}:0:{len(source['source_text'])}"
            elements = ({
                "anchor_id": f"source-anchor:{sha256(anchor_key.encode()).hexdigest()}",
                "document_hash": source["source_hash"],
                "document_part": "paragraph",
                "paragraph_index": 0,
                "start_offset": 0,
                "end_offset": len(source["source_text"]),
                # Keep the canonical element self-contained for in-process
                # Semantica.  The extraction adapter is intentionally
                # provider-neutral and must not recover text from an
                # unmanaged source later in the pipeline.
                "text": source["source_text"],
                "exact_text": source["source_text"],
                "source_id": source["source_id"],
                "source_ref": f"knowledge-source:{source['source_id']}",
                "source_hash": source["source_hash"],
            },) if not compilation_anchors else tuple(
                _validated_compilation_anchor(
                    anchor=anchor, compilation=compilation, source=source,
                    canonical_document_id=str(canonical_document_id),
                ) for anchor in compilation_anchors
            )
            documents.append(TenderPackageDocument(
                document_id=document_id, document_hash=source["source_hash"],
                role=str(raw.get("role") or ""), elements=elements,
            ))
        package_id = str(body.get("packageId") or body.get("package_id") or f"package_{uuid4().hex}")
        scope = KnowledgeScope(package_id=package_id, documents=tuple(documents))
        result = _extraction_store().put_knowledge_scope(
            discovery_run_id=discovery_run_id, organization_id=organization_id,
            workspace_id=workspace_id, scope=scope,
        )
    except ValueError as exc:
        detail = str(exc)
        if "immutable knowledge_builder_knowledge_scopes" in detail:
            detail = "tender package immutable record conflict"
        raise HTTPException(status_code=400, detail=detail) from exc
    return {"ok": True, "schemaVersion": "knowledge_builder_tender_package_response.v1",
            "tenderPackage": _hydrate_scope(result, builder)}


@router.get("/api/knowledge/builder/evidence-obligations")
async def list_knowledge_builder_evidence_obligations(request: Request):
    route_path = "/api/knowledge/builder/evidence-obligations"
    ctx = _authz(request, "GET", route_path)
    store = _extraction_store()
    assertion_id = request.query_params.get("assertionId") or request.query_params.get("assertion_id")
    if assertion_id:
        _require_owner(store.get_assertion_owner(assertion_id), str(ctx.organization_id), str(ctx.workspace_id) if ctx.workspace_id else None, label="assertion candidate")
    obligations = store.list_evidence_obligations(
        assertion_id, organization_id=str(ctx.organization_id),
        workspace_id=str(ctx.workspace_id) if ctx.workspace_id else None,
        limit=_page_limit(request), cursor=request.query_params.get("cursor"),
    )
    return {"ok": True, "schemaVersion": "knowledge_builder_evidence_obligations_response.v1",
            "evidenceObligations": [{**item, "schema_version": "evidence_obligation.v1"}
                                    for item in obligations]}


@router.post("/api/knowledge/builder/evidence-obligations")
async def create_knowledge_builder_evidence_obligation(request: Request):
    route_path = "/api/knowledge/builder/evidence-obligations"
    ctx = _authz(request, "POST", route_path)
    body = await _json_body(request)
    if (body.get("schemaVersion") or body.get("schema_version")) != "knowledge_builder_evidence_obligation_request.v1":
        raise HTTPException(status_code=400, detail="knowledge_builder_evidence_obligation_request.v1 required")
    assertion_id = str(body.get("assertionId") or body.get("assertion_id") or "")
    store = _extraction_store()
    _require_owner(store.get_assertion_owner(assertion_id), str(ctx.organization_id),
                   str(ctx.workspace_id) if ctx.workspace_id else None,
                   label="assertion candidate")
    try:
        obligation = EvidenceObligation(
            obligation_id=str(body.get("obligationId") or body.get("obligation_id") or f"obligation_{uuid4().hex}"),
            assertion_id=assertion_id,
            obligation_type=str(body.get("obligationType") or body.get("obligation_type") or ""),
            primary_anchor_ref=str(body.get("primaryAnchorRef") or body.get("primary_anchor_ref") or ""),
            allowed_companion_roles=tuple(body.get("allowedCompanionRoles") or body.get("allowed_companion_roles") or ()),
            query_terms=tuple(body.get("queryTerms") or body.get("query_terms") or ()),
        )
        result = store.put_evidence_obligation(obligation)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "schemaVersion": "knowledge_builder_evidence_obligation_response.v1",
            "evidenceObligation": {**result, "schema_version": "evidence_obligation.v1"}}


@router.post("/api/knowledge/builder/evidence-obligations/{obligation_id}/resolve")
async def resolve_knowledge_builder_evidence_obligation(obligation_id: str, request: Request):
    route_path = "/api/knowledge/builder/evidence-obligations/{obligation_id}/resolve"
    ctx = _authz(request, "POST", route_path)
    body = await _json_body(request)
    if (body.get("schemaVersion") or body.get("schema_version")) != "knowledge_builder_evidence_obligation_resolution_request.v1":
        raise HTTPException(status_code=400, detail="knowledge_builder_evidence_obligation_resolution_request.v1 required")
    store = _extraction_store()
    _require_owner(store.get_evidence_obligation_owner(obligation_id), str(ctx.organization_id), str(ctx.workspace_id) if ctx.workspace_id else None, label="evidence obligation")
    try:
        obligation = store.resolve_evidence_obligation(
            obligation_id=obligation_id,
            status=str(body.get("status") or ""),
            supporting_anchor_refs=tuple(body.get("supportingAnchorRefs") or body.get("supporting_anchor_refs") or ()),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True,
            "schemaVersion": "knowledge_builder_evidence_obligation_resolution_response.v1",
            "evidenceObligation": {**obligation, "schema_version": "evidence_obligation.v1"}}
