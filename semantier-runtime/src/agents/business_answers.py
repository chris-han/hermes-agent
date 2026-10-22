"""Shared native delegation for WebAPI and installed business-answer tools."""
from contextlib import contextmanager

from agents.auth_db import get_workspace_config
from contracts.business_answers_receivables import ReceivablesRunRequest
from services.runtime_context_proposal_provider import RuntimeContextProposalProviderRegistry
from services.runtime_context_transition_service import RuntimeAuthorityResolver
from storage.knowledge_builder_store import KnowledgeBuilderStore
from storage.ontology_build_store import OntologyBuildStore, ontology_build_store_path

_PROVIDER_REGISTRY = RuntimeContextProposalProviderRegistry.compose_from_packaged_resources()


def require_enabled(ctx):
    config = get_workspace_config(str(ctx.workspace_id or ""))
    plugins = config.get("plugins", {})
    if "business_answers" not in plugins.get("enabled", []) or "business_answers" in plugins.get("disabled", []):
        raise PermissionError("BUSINESS_ANSWERS_PLUGIN_DISABLED")


@contextmanager
def native_service(ctx):
    from services.business_answers_receivables_service import BusinessAnswersReceivablesService
    if not ctx.authenticated or not ctx.user_id or not ctx.organization_id or not ctx.workspace_id:
        raise PermissionError("BUSINESS_ANSWERS_AUTHORITY_REQUIRED")
    registry = KnowledgeBuilderStore(workspace_root=ctx.workspace_root)
    store = OntologyBuildStore(ontology_build_store_path(ctx.workspace_root))
    try:
        resolver = RuntimeAuthorityResolver(graph_intent_store=None, ontology_build_store=store,
            runtime_context_store=None, authority_store=None, provider_registry=_PROVIDER_REGISTRY)
        yield BusinessAnswersReceivablesService(build_store=store, knowledge_store=registry,
            authority_resolver=resolver)
    finally:
        store.close()
        registry.close()


def create_run(*, ctx, request):
    require_enabled(ctx)
    body = request if isinstance(request, ReceivablesRunRequest) else ReceivablesRunRequest.model_validate(request)
    with native_service(ctx) as service:
        return service.create_run(ctx=ctx, request=body)


def get_run(*, ctx, run_ref):
    with native_service(ctx) as service:
        return service.get_run(ctx=ctx, run_ref=run_ref)


def get_answer(*, ctx, answer_ref):
    with native_service(ctx) as service:
        return service.get_answer(ctx=ctx, answer_ref=answer_ref)


def get_answer_projection(*, ctx, answer_ref):
    with native_service(ctx) as service:
        answer = service.get_answer(ctx=ctx, answer_ref=answer_ref)
        run = service.get_run(ctx=ctx, run_ref=answer.run_ref)
        return {"ok": True, "answer": answer.model_dump(mode="json"),
            "current_state": service.answer_state(ctx=ctx, answer=answer),
            "basis": _basis_projection(service, ctx, run)}


def _basis_projection(service, ctx, run):
    basis = service.get_run_basis(ctx=ctx, run=run)
    source = basis["source"].model_dump(mode="json", by_alias=True, include={
        "artifact_ref", "artifact_hash", "import_id", "file_id", "source_identity_ref", "source_hash",
        "original_name", "parser_profile", "parser_profile_hash", "canonical_source_ir_pin", "sheet_name",
        "header_row", "headers", "row_numbers", "evidence"})
    return {"source": source, "report": basis["report"].model_dump(mode="json"),
        "manifest": basis["manifest"].model_dump(mode="json"),
        "snapshot": basis["snapshot"].model_dump(mode="json")}


def _run_projection(service, ctx, run):
    return {"ok": True, "run": run.model_dump(mode="json"),
        "basis": _basis_projection(service, ctx, run)}


def create_run_projection(*, ctx, request):
    require_enabled(ctx)
    body = request if isinstance(request, ReceivablesRunRequest) else ReceivablesRunRequest.model_validate(request)
    with native_service(ctx) as service:
        run = service.create_run(ctx=ctx, request=body)
        return _run_projection(service, ctx, run)


def get_run_projection(*, ctx, run_ref):
    with native_service(ctx) as service:
        run = service.get_run(ctx=ctx, run_ref=run_ref)
        return _run_projection(service, ctx, run)
