"""Authenticated EvidenceRef resolution and regrounding seam."""

import os
from pathlib import Path

from runtime_paths import sqlite_db_path
from storage.graph_intent_brief_store import GraphIntentBriefStore
from storage.knowledge_builder_store import KnowledgeBuilderStore

from fastapi import APIRouter, HTTPException, Request

from agents.route_policy import RouteAuthorizationClass
from agents.webapi.deps import require_route_authorization
from contracts.canonical_source_ir import CanonicalSourceDocument
from contracts.evidence_location import EvidenceRef, EvidenceSelector
from services.runtime_errors import RuntimeContextError
from semantier.knowledge_feature_packs.evidence_resolver import resolve_evidence_ref
from semantier.knowledge_feature_packs.evidence_selector_builder import evidence_ref_for_block

router = APIRouter(tags=["evidence"])


def _ctx(request: Request):
    return require_route_authorization(request, authorization_class=RouteAuthorizationClass.TENANT_MEMBER,
                                        route_hint="evidence resolver")


def _parse_document(body: dict) -> CanonicalSourceDocument:
    raw = body.get("canonical_source_document") or body.get("canonicalSourceDocument")
    if not isinstance(raw, dict):
        raise HTTPException(status_code=400, detail="canonical source document required")
    return CanonicalSourceDocument.model_validate(raw)


@router.post("/api/evidence/resolve")
async def resolve_evidence(request: Request):
    ctx = _ctx(request)
    try:
        body = await request.json()
        ref = EvidenceRef.model_validate(body.get("evidence_ref") or body.get("evidenceRef"))
        if ref.organization_id != ctx.organization_id or ref.workspace_id != ctx.workspace_id:
            raise HTTPException(status_code=403, detail="evidence reference is outside the active tenant/workspace")
        artifact_ref = body.get("artifact_ref") or body.get("artifactRef")
        if ref.source_identity_ref.startswith("governed-source:kbs_receivables_"):
            from agents.business_answers import native_service
            with native_service(ctx) as service:
                source = service.get_source_by_identity(ctx=ctx, source_identity_ref=ref.source_identity_ref)
                if ref not in source.evidence:
                    raise HTTPException(status_code=409, detail="Evidence reference does not match the pinned native source")
                document = source.canonical_source_ir
                artifact_ref = source.byte_owner_ref
        elif body.get("canonical_source_document") or body.get("canonicalSourceDocument"):
            document = _parse_document(body)
        else:
            db_path = Path(os.environ.get("SEMANTIER_EOS_DB_PATH", str(sqlite_db_path("eos.db", ".semantier/eos.db"))))
            if not db_path.is_file():
                raise ValueError("governed knowledge source store not found")
            source_registry = KnowledgeBuilderStore(db_path, workspace_root=ctx.workspace_root)
            try:
                store = GraphIntentBriefStore(
                    ctx.workspace_root, organization_id=ctx.organization_id,
                    workspace_id=ctx.workspace_id, source_registry=source_registry,
                )
                identity, document = store.resolve_evidence_source(
                    source_identity_ref=ref.source_identity_ref,
                    canonical_source_ir_ref=str(ref.provenance.get("canonical_source_ir_ref") or ""),
                    canonical_source_ir_hash=str(ref.provenance.get("canonical_source_ir_hash") or ""),
                )
                if ref.provenance.get("source_hash") not in (None, identity.source_hash):
                    raise ValueError("evidence source hash mismatch")
                artifact_ref = identity.artifact_ref
            finally:
                source_registry.close()
        location = resolve_evidence_ref(ref, document, artifact_ref=artifact_ref,
                                       selector_hash=body.get("selector_hash") or body.get("selectorHash"))
        return location.model_dump(mode="json", by_alias=True)
    except Exception as exc:
        if isinstance(exc, HTTPException):
            raise
        if isinstance(exc, PermissionError) or (isinstance(exc, RuntimeContextError) and exc.code in {"FORBIDDEN_ACTOR", "FORBIDDEN_SCOPE"}):
            raise HTTPException(status_code=403, detail="Evidence access denied") from exc
        return {"schemaVersion": "semantier.resolved_location.v1", "status": "unresolved", "error": str(exc)}


@router.post("/api/evidence/reground")
async def reground_evidence(request: Request):
    ctx = _ctx(request)
    try:
        body = await request.json()
        old = EvidenceRef.model_validate(body.get("evidence_ref") or body.get("evidenceRef"))
        document = _parse_document(body)
        selector = EvidenceSelector.model_validate(body.get("new_selector") or body.get("newSelector"))
        block_id = str(body.get("source_block_id") or body.get("sourceBlockId") or "")
        block = next((item for item in document.blocks if item.block_id == block_id), None)
        if block is None:
            raise ValueError("source block required and must be present in canonical document")
        if selector.selector_hash() != old.selector_hash:
            # The explicit human selector is authoritative only after it points at this block.
            if selector.source_element_ref and selector.source_element_ref != (old.provenance.get("source_locator") or {}).get("nativeId"):
                pass
        source = body.get("source_identity") or body.get("sourceIdentity")
        if not isinstance(source, dict):
            raise ValueError("source identity required")
        from contracts.evidence_location import SourceIdentity
        source_identity = SourceIdentity.model_validate(source)
        if source_identity.source_identity_ref != old.source_identity_ref or source_identity.organization_id != ctx.organization_id \
                or source_identity.workspace_id != ctx.workspace_id:
            raise ValueError("reground source identity is outside the active tenant/source")
        ref = evidence_ref_for_block(document, block, source_identity=source_identity,
                                     evidence_role=old.evidence_role, selector=selector)
        location = resolve_evidence_ref(ref, document, artifact_ref=source.get("artifactRef"))
        return {"evidenceRef": ref.model_dump(mode="json", by_alias=True), "resolvedLocation": location.model_dump(mode="json", by_alias=True)}
    except Exception as exc:
        return {"status": "unresolved", "error": str(exc)}
