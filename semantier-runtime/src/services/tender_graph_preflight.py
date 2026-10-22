"""Fresh readiness checks over the existing Tender intake owners."""
from __future__ import annotations

from agents.auth_session import RequestContext
from services.corpus_graph_schema_profile_registry import CorpusGraphSchemaProfileRegistry
from services.workspace_artifact_service import WorkspaceArtifactService
from storage.ontology_build_store import ontology_build_store_path

TENDER_PROFILE_ID = "tender_corpus_schema_induction.v0_1"


def tender_graph_preflight(*, ctx: RequestContext, artifacts: WorkspaceArtifactService) -> dict:
    if not isinstance(artifacts, WorkspaceArtifactService):
        raise TypeError("graph preflight requires the production artifact owner")
    # The artifact owner's read and upload operations share this current DB-backed
    # membership check. UI bundle labels are not a separate write entitlement.
    artifacts._access(ctx)
    if artifacts._store._path.resolve() != ontology_build_store_path(ctx.workspace_root).resolve():
        raise ValueError("JOURNEY_WORKSPACE_OWNER_MISMATCH")
    artifacts._store.list_artifact_pins(organization_id=ctx.organization_id,
        workspace_id=ctx.workspace_id, schema_version="workspace_artifact.v1")
    try:
        profile = CorpusGraphSchemaProfileRegistry().resolve(TENDER_PROFILE_ID)
    except (KeyError, ValueError, OSError):
        return {"status": "BLOCKED", "blocker_code": "TENDER_SCHEMA_PROFILE_UNAVAILABLE",
                "recovery_owner": "CorpusGraphSchemaProfileRegistry"}
    if profile.source_kind != "docx":
        return {"status": "BLOCKED", "blocker_code": "TENDER_SCHEMA_PROFILE_UNAVAILABLE",
                "recovery_owner": "CorpusGraphSchemaProfileRegistry"}
    return {"status": "READY", "profile_id": profile.profile_id,
            "profile_hash": profile.content_hash,
            "authorization_owner": "RuntimeAuthorityResolver",
            "authorization_class": "TENANT_MEMBER",
            "source_access_owner": "WorkspaceArtifactService"}


def require_tender_graph_preflight(*, ctx: RequestContext, artifacts: WorkspaceArtifactService) -> dict:
    result = tender_graph_preflight(ctx=ctx, artifacts=artifacts)
    if result["status"] != "READY":
        raise ValueError(result["blocker_code"])
    return result
