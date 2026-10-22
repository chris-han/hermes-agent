from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Any
from uuid import uuid4
from dataclasses import asdict

from contracts.tender_sensitive_term_detection_modeling_core import RuntimeContextDTO
from eos.artifact_hashing import canonical_json, content_hash as compute_content_hash
from services.governed_document_service import load_verified_normalized_document_artifact
from services.knowledge_source_compilation_service import (
    StoreBackedKnowledgeSourceCompilationService,
)
from plugins.tender_sensitive_term_detection_modeling.frame_semantics import (
    compile_frame_pattern_candidate,
)
from services.knowledge_source_governance_service import (
    StoreBackedKnowledgeSourceGovernanceService,
)
from services.policy_rule_candidate_service import StoreBackedPolicyRuleCandidateService

PROFILE_VERSION = "sensitive_lexicon_docx.v1"
NORMALIZATION_POLICY_VERSION = "sensitive_lexicon_normalization.v1"
DEFAULT_RULE_FAMILY = "tender_compliance"
DEFAULT_ISSUE_TYPE = "sensitive_expression"
DEFAULT_SEVERITY = "medium"

_HEADER_ALIASES = {
    "term": {"敏感词", "敏感词汇", "违规敏感词", "词汇", "关键词", "term"},
    "category": {"分类", "类别", "场景分类", "违规分类", "风险类型", "category"},
    "severity": {"风险等级", "严重程度", "severity"},
    "rationale": {"说明", "依据", "违规示例", "示例", "rationale"},
}


def _assert_ascii(value: str, field: str) -> str:
    normalized = (value or "").strip()
    if not normalized:
        raise ValueError(f"{field} is required")
    try:
        normalized.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{field} must be an ASCII-stable identifier") from exc
    return normalized


def _assert_optional_ascii(value: str | None, field: str) -> str | None:
    if value is None or not str(value).strip():
        return None
    return _assert_ascii(str(value), field)


def normalize_sensitive_term(value: str) -> str:
    text = unicodedata.normalize("NFKC", value or "")
    text = " ".join(text.split())
    punctuation_map = str.maketrans(
        {
            "，": ",",
            "。": ".",
            "：": ":",
            "；": ";",
            "（": "(",
            "）": ")",
            "【": "[",
            "】": "]",
            "“": '"',
            "”": '"',
            "‘": "'",
            "’": "'",
        }
    )
    return text.translate(punctuation_map).strip().lower()


def _header_role(text: str) -> str | None:
    normalized = normalize_sensitive_term(text)
    for role, aliases in _HEADER_ALIASES.items():
        if normalized in {normalize_sensitive_term(alias) for alias in aliases}:
            return role
    return None


def _compilation_anchor_ref(
    *,
    knowledge_source_id: str,
    run_id: str,
    canonical_anchor_ref: str,
    compiler_profile_version: str,
) -> str:
    digest = compute_content_hash(
        canonical_json(
            {
                "knowledge_source_id": knowledge_source_id,
                "run_id": run_id,
                "canonical_anchor_ref": canonical_anchor_ref,
                "compiler_profile_version": compiler_profile_version,
            }
        )
    ).removeprefix("sha256:")[:24]
    return f"ksca_{digest}"


def _candidate_idempotency_key(
    *,
    organization_id: str,
    proposed_workspace_scope: str,
    proposed_workspace_id: str | None,
    knowledge_source_id: str,
    source_hash: str,
    canonical_anchor_ref: str,
    normalized_term: str,
    rule_family: str,
    compiler_profile_version: str,
) -> str:
    return compute_content_hash(
        canonical_json(
            {
                "organization_id": organization_id,
                "proposed_workspace_scope": proposed_workspace_scope,
                "proposed_workspace_id": proposed_workspace_id,
                "knowledge_source_id": knowledge_source_id,
                "source_hash": source_hash,
                "canonical_anchor_ref": canonical_anchor_ref,
                "compiler_profile_version": compiler_profile_version,
                "normalization_policy_version": NORMALIZATION_POLICY_VERSION,
                "normalized_term": normalized_term,
                "rule_family": rule_family,
            }
        )
    )


def _find_superseded_candidate(
    *,
    candidate_service: StoreBackedPolicyRuleCandidateService,
    organization_id: str,
    proposed_workspace_scope: str,
    proposed_workspace_id: str | None,
    knowledge_source_id: str,
    canonical_anchor_ref: str,
    normalized_term: str,
    rule_family: str,
    compiler_profile_version: str,
) -> str | None:
    for candidate in candidate_service.list_candidates(rule_family=rule_family):
        if candidate.get("organization_id") != organization_id:
            continue
        if candidate.get("proposed_workspace_scope") != proposed_workspace_scope:
            continue
        if (candidate.get("proposed_workspace_id") or None) != (proposed_workspace_id or None):
            continue
        scope = candidate.get("applicability_scope") or {}
        if scope.get("knowledge_source_id") != knowledge_source_id:
            continue
        if scope.get("canonical_anchor_ref") != canonical_anchor_ref:
            continue
        if scope.get("normalized_term") != normalized_term:
            continue
        if scope.get("compiler_profile_version") == compiler_profile_version:
            continue
        return str(candidate["rule_candidate_id"])
    return None


def _canonical_anchor_ref(document: dict[str, Any], node: dict[str, Any], source_hash: str) -> str:
    # Retained canonical v2 artifacts can carry the original explicit ref.
    if node.get("anchor_ref"):
        return str(node["anchor_ref"])
    evidence_by_id = {item["id"]: item for item in document.get("evidence", [])}
    for evidence_id in node.get("evidence_ids", []):
        evidence = evidence_by_id.get(evidence_id) or {}
        anchor = evidence.get("anchor") or {}
        if (anchor.get("document_schema_version") == "canonical_document.v2"
                and anchor.get("normalized_document_artifact_ref")
                and anchor.get("source_hash") == source_hash
                and anchor.get("local_text_hash") == node.get("local_text_hash")):
            return str(evidence_id)
    raise RuntimeError("CANONICAL_ANCHOR_REQUIRED")


def _table_entries(document: dict[str, Any], source_hash: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for table in document.get("tables") or []:
        rows = table.get("rows") or []
        if not rows:
            continue
        header_cells = rows[0].get("cells") or []
        roles = [_header_role(str(cell.get("text") or "")) for cell in header_cells]
        if "term" not in roles:
            continue
        term_index = roles.index("term")
        for row in rows[1:]:
            cells = row.get("cells") or []
            if term_index >= len(cells):
                continue
            term_cell = cells[term_index]
            term_text = str(term_cell.get("text") or "").strip()
            if not term_text:
                continue
            values: dict[str, str | None] = {
                "term": term_text,
                "category": None,
                "severity": None,
                "rationale": None,
            }
            for index, role in enumerate(roles):
                if role in values and index < len(cells):
                    values[role] = str(cells[index].get("text") or "").strip() or None
            terms = [
                part.strip()
                for part in term_text.replace("；", ";").replace("，", ",").replace("、", ",").split(",")
                if part.strip()
            ]
            for term in terms or [term_text]:
                entries.append(
                    {
                        "layout": "table_with_header",
                        "term_text": term,
                        "term_category": values["category"],
                        "severity": values["severity"] or DEFAULT_SEVERITY,
                        "rationale": values["rationale"],
                        "canonical_anchor_ref": _canonical_anchor_ref(document, term_cell, source_hash),
                        "canonical_anchor": term_cell,
                    }
                )
    return entries


def _paragraph_entries(document: dict[str, Any], source_hash: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    current_category: str | None = None
    for block in document.get("blocks") or []:
        text = str(block.get("text") or "").strip()
        if not text:
            continue
        if text.endswith(("：", ":")) and len(text) <= 40:
            current_category = text.rstrip("：:")
            continue
        if current_category is None:
            continue
        parts = [part.strip() for part in text.replace("；", ";").replace("，", ",").split(",")]
        for part in parts:
            term = part.strip(" -•\t")
            if not term:
                continue
            entries.append(
                {
                    "layout": "paragraph_list_with_category_heading",
                    "term_text": term,
                    "term_category": current_category,
                    "severity": DEFAULT_SEVERITY,
                    "rationale": None,
                    "canonical_anchor_ref": _canonical_anchor_ref(document, block, source_hash),
                    "canonical_anchor": block,
                }
            )
    return entries


def _compile_entries(document: dict[str, Any], source_hash: str) -> list[dict[str, Any]]:
    table_entries = _table_entries(document, source_hash)
    if table_entries:
        return table_entries
    paragraph_entries = _paragraph_entries(document, source_hash)
    if paragraph_entries:
        return paragraph_entries
    raise RuntimeError("IMPORT_SCHEMA_UNSUPPORTED")


def compile_sensitive_lexicon(
    *,
    normalized_document_artifact_ref: str,
    organization_id: str,
    workspace_id: str | None,
    knowledge_source_id: str,
    source_ref: str,
    created_by: str,
    proposed_workspace_scope: str = "workspace",
    proposed_workspace_id: str | None = None,
    source_version: str | None = None,
    db_path: str | Path | None = None,
    compiler_profile_version: str = PROFILE_VERSION,
    accepted_corpus_schema_projection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    organization_id = _assert_ascii(organization_id, "organization_id")
    workspace_id = _assert_optional_ascii(workspace_id, "workspace_id")
    knowledge_source_id = _assert_ascii(knowledge_source_id, "knowledge_source_id")
    created_by = _assert_ascii(created_by, "created_by")
    proposed_workspace_scope = _assert_ascii(proposed_workspace_scope, "proposed_workspace_scope")
    proposed_workspace_id = _assert_optional_ascii(
        proposed_workspace_id or workspace_id,
        "proposed_workspace_id",
    )
    compiler_profile_version = _assert_ascii(compiler_profile_version, "compiler_profile_version")

    context = RuntimeContextDTO(
        organization_id=organization_id,
        workspace_id=workspace_id,
        actor_id=created_by,
        actor_role="plugin",
    )
    artifact = load_verified_normalized_document_artifact(
        normalized_document_artifact_ref=normalized_document_artifact_ref,
    )
    document = artifact.get("document")
    if not isinstance(document, dict):
        raise RuntimeError("INVALID_CANONICAL_DOCUMENT_ARTIFACT")
    if document.get("schema_version") != "canonical_document.v2":
        raise RuntimeError("CANONICAL_DOCUMENT_V2_REQUIRED")
    if (document.get("parser") or {}).get("method") not in {"ooxml", "docx_ooxml"}:
        raise RuntimeError("DOCX_CANONICAL_DOCUMENT_REQUIRED")
    source_hash = str(artifact.get("source_hash") or "")
    if not source_hash:
        raise RuntimeError("SOURCE_HASH_REQUIRED")

    source_service = StoreBackedKnowledgeSourceGovernanceService(db_path=db_path)
    source = source_service.register_controlled_document(
        knowledge_source_id=knowledge_source_id,
        source_ref=source_ref,
        source_hash=source_hash,
        semantic_purpose="sensitive_lexicon",
        context=context,
        metadata={"source_version": source_version},
    )
    source_service.assert_source_routable(source["source_kind"], source["semantic_purpose"])
    if source["source_lifecycle_status"] not in {"approved", "validated"}:
        raise RuntimeError("KNOWLEDGE_SOURCE_NOT_ADMISSIBLE")
    source_service.resolve_controlled_document_compiler_route(knowledge_source_id)

    entries = _compile_entries(document, source_hash)
    run_id = f"kscr_{uuid4().hex}"
    compilation_service = StoreBackedKnowledgeSourceCompilationService(db_path=db_path)
    candidate_service = StoreBackedPolicyRuleCandidateService(db_path=db_path)
    candidates: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    frame_candidates: list[dict[str, Any]] = []
    anchor_refs: list[str] = []
    seen: set[tuple[str, str, str | None]] = set()
    compiled_rows: list[dict[str, Any]] = []
    for entry in entries:
        normalized_term = normalize_sensitive_term(str(entry["term_text"]))
        if not normalized_term:
            continue
        dedupe_key = (normalized_term, str(entry["canonical_anchor_ref"]), entry.get("term_category"))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        anchor_ref = _compilation_anchor_ref(
            knowledge_source_id=knowledge_source_id,
            run_id=run_id,
            canonical_anchor_ref=str(entry["canonical_anchor_ref"]),
            compiler_profile_version=compiler_profile_version,
        )
        anchor_refs.append(anchor_ref)
        compiled_rows.append(
            {
                "entry": entry,
                "normalized_term": normalized_term,
                "anchor_ref": anchor_ref,
                "idempotency_key": _candidate_idempotency_key(
                    organization_id=organization_id,
                    proposed_workspace_scope=proposed_workspace_scope,
                    proposed_workspace_id=proposed_workspace_id,
                    knowledge_source_id=knowledge_source_id,
                    source_hash=source_hash,
                    canonical_anchor_ref=str(entry["canonical_anchor_ref"]),
                    normalized_term=normalized_term,
                    rule_family=DEFAULT_RULE_FAMILY,
                    compiler_profile_version=compiler_profile_version,
                ),
            }
        )

    run = compilation_service.create_run(
        knowledge_source_compilation_run_id=run_id,
        knowledge_source_id=knowledge_source_id,
        organization_id=organization_id,
        workspace_id=workspace_id,
        semantic_purpose="sensitive_lexicon",
        source_kind="controlled_document",
        source_ref=source_ref,
        source_artifact_ref=str(artifact.get("source_artifact_ref") or source_ref),
        source_hash=source_hash,
        normalized_document_artifact_ref=normalized_document_artifact_ref,
        normalized_document_artifact_hash=str(artifact.get("artifact_hash") or ""),
        document_schema_version=str(document.get("schema_version") or ""),
        parser_method=str((document.get("parser") or {}).get("method") or ""),
        parser_version=str((document.get("parser") or {}).get("version") or ""),
        provider_artifact_refs=[],
        provider_artifact_hashes=[],
        compiler_profile_version=compiler_profile_version,
        mapped_source_anchor_refs=anchor_refs,
        created_by=created_by,
    )

    for compiled in compiled_rows:
        entry = compiled["entry"]
        anchor_ref = compiled["anchor_ref"]
        compilation_service.create_anchor(
            compilation_anchor_ref=anchor_ref,
            knowledge_source_compilation_run_id=run_id,
            knowledge_source_id=knowledge_source_id,
            canonical_document_anchor_ref=str(entry["canonical_anchor_ref"]),
            source_hash=source_hash,
            compiler_profile_version=compiler_profile_version,
            anchor_payload={
                "knowledge_source_id": knowledge_source_id,
                "knowledge_source_compilation_run_id": run_id,
                "canonical_document_anchor_ref": entry["canonical_anchor_ref"],
                "source_hash": source_hash,
                "compiler_profile_version": compiler_profile_version,
            },
        )
        normalized_term = compiled["normalized_term"]
        idempotency_key = compiled["idempotency_key"]
        frame_pattern_payload = None
        if accepted_corpus_schema_projection is not None:
            try:
                frame_pattern = compile_frame_pattern_candidate(
                    source_term=str(entry["term_text"]),
                    source_example=entry.get("rationale"),
                    source_evidence_refs=(anchor_ref,),
                    accepted_schema_projection=accepted_corpus_schema_projection,
                    graph_release_ref="pending_graph_release",
                )
            except ValueError:
                frame_pattern = None
            if frame_pattern is not None:
                frame_pattern_payload = asdict(frame_pattern)
                frame_candidates.append(frame_pattern_payload)
        candidate = candidate_service.create_candidate_idempotent(
            idempotency_key=idempotency_key,
            source_anchor_refs=[anchor_ref],
            applicability_scope={
                "match_terms": [entry["term_text"]],
                "normalized_term": normalized_term,
                "term_category": entry.get("term_category"),
                "issue_type": DEFAULT_ISSUE_TYPE,
                "knowledge_source_id": knowledge_source_id,
                "knowledge_source_compilation_run_id": run_id,
                "canonical_anchor_ref": entry["canonical_anchor_ref"],
                "compiler_profile_version": compiler_profile_version,
                "normalization_policy_version": NORMALIZATION_POLICY_VERSION,
                "frame_pattern": frame_pattern_payload,
            },
            context=context,
            candidate_payload={
                "proposed_workspace_scope": proposed_workspace_scope,
                "proposed_workspace_id": proposed_workspace_id,
                "authority_domain": "tender_compliance",
                "semantic_tier": "T4",
                "jurisdiction": "CN",
                "rule_family": DEFAULT_RULE_FAMILY,
                "candidate_state": "PROPOSED",
                "extracted_rationale": str(entry.get("rationale") or "Sensitive lexicon DOCX candidate."),
                "draft_rule_text": f"Flag tender wording containing {entry['term_text']}.",
                "severity": str(entry.get("severity") or DEFAULT_SEVERITY),
                "confidence": 1.0,
                "uncertainty_notes": None,
                "supersedes_candidate_id": _find_superseded_candidate(
                    candidate_service=candidate_service,
                    organization_id=organization_id,
                    proposed_workspace_scope=proposed_workspace_scope,
                    proposed_workspace_id=proposed_workspace_id,
                    knowledge_source_id=knowledge_source_id,
                    canonical_anchor_ref=str(entry["canonical_anchor_ref"]),
                    normalized_term=normalized_term,
                    rule_family=DEFAULT_RULE_FAMILY,
                    compiler_profile_version=compiler_profile_version,
                ),
            },
        )
        candidates.append(candidate)
        evidence_payload = {
            "compiler_profile_version": compiler_profile_version,
            "normalization_policy_version": NORMALIZATION_POLICY_VERSION,
            "candidate_state_snapshot_display_only": True,
            "current_lifecycle_authority": "policy_rule_candidates.rule_candidate_id",
        }
        if candidate["candidate_state"] == "PROPOSED":
            evidence_payload.update(
                {
                    "term_text": entry["term_text"],
                    "normalized_term": normalized_term,
                }
            )
        evidence_rows.append(
            compilation_service.persist_knowledge_builder_evidence(
                organization_id=organization_id,
                workspace_id=workspace_id,
                knowledge_source_id=knowledge_source_id,
                knowledge_source_compilation_run_id=run_id,
                candidate=candidate,
                source_anchor_refs=[anchor_ref],
                evidence_payload=evidence_payload,
            )
        )
    return {
        "status": "completed",
        "knowledge_source": source,
        "compilation_run": run,
        "candidates": candidates,
        "knowledge_builder_evidence": evidence_rows,
        "compiler_profile_version": compiler_profile_version,
        "normalization_policy_version": NORMALIZATION_POLICY_VERSION,
        "frame_candidates": frame_candidates,
    }
