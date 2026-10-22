"""Governed HTTP surface for runtime-context policy and element authority."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query, Request

from agents.auth_session import RequestContext
from agents.route_policy import RouteAuthorizationClass
from agents.webapi.deps import require_route_authorization
from contracts.knowledge_classification import RuntimeContextDTO
from contracts.ontology_building import ArtifactPinV1
from contracts.runtime_context_authority import (
    RuntimeContextElementAdmissionReceiptV1,
    RuntimeContextScopePolicyV1,
)
from services.runtime_context_authority_issuance_service import (
    AdmitRuntimeContextElementCommandV1,
    IssueRuntimeContextScopePolicyCommandV1,
    RuntimeContextAuthorityIssuanceService,
)
from services.runtime_context_proposal_provider import (
    RuntimeContextProposalProviderRegistry,
)
from services.runtime_context_transition_service import RuntimeAuthorityResolver
from storage.graph_intent_brief_store import GraphIntentBriefStore
from storage.knowledge_builder_store import KnowledgeBuilderStore
from storage.ontology_build_store import OntologyBuildStore, ontology_build_store_path
from storage.runtime_context_authority_store import RuntimeContextAuthorityStore
from storage.runtime_working_context_store import RuntimeContextError, RuntimeTaskContextStore


router = APIRouter(tags=["runtime-context-authority"])
ROUTE_RUNTIME_CONTEXT_SCOPE_POLICIES = "/api/runtime-context-scope-policies"
ROUTE_RUNTIME_CONTEXT_SCOPE_POLICY = "/api/runtime-context-scope-policies/{policy_ref}"
ROUTE_RUNTIME_CONTEXT_ADMISSIONS = "/api/runtime-context-admissions"
ROUTE_RUNTIME_CONTEXT_ADMISSION = "/api/runtime-context-admissions/{admission_ref}"

# Trusted composition reads the one known package resource once. Request paths
# only consult this frozen registry and never derive implementation identity from disk.
_PROVIDER_REGISTRY = RuntimeContextProposalProviderRegistry.compose_from_packaged_resources()


def _actor(ctx: RequestContext) -> RuntimeContextDTO:
    if not ctx.organization_id or not ctx.workspace_id:
        raise RuntimeContextError("MISSING_WORKSPACE")
    return RuntimeContextDTO(
        organization_id=str(ctx.organization_id),
        workspace_id=str(ctx.workspace_id),
        actor_id=str(ctx.user_id),
    )


def _components(ctx: RequestContext):
    if not ctx.organization_id or not ctx.workspace_id:
        raise RuntimeContextError("MISSING_WORKSPACE")
    registry = KnowledgeBuilderStore(workspace_root=ctx.workspace_root)
    build_store = OntologyBuildStore(ontology_build_store_path(ctx.workspace_root))
    authority_store = RuntimeContextAuthorityStore(build_store)
    resolver = RuntimeAuthorityResolver(
        graph_intent_store=GraphIntentBriefStore(
            ctx.workspace_root,
            organization_id=str(ctx.organization_id),
            workspace_id=str(ctx.workspace_id),
            source_registry=registry,
        ),
        ontology_build_store=build_store,
        runtime_context_store=RuntimeTaskContextStore(build_store),
        authority_store=authority_store,
        provider_registry=_PROVIDER_REGISTRY,
    )
    service = RuntimeContextAuthorityIssuanceService(
        authority_store=authority_store,
        resolver=resolver,
        provider_registry=_PROVIDER_REGISTRY,
    )
    return registry, authority_store, resolver, service


def _translate(exc: RuntimeContextError) -> HTTPException:
    if exc.code in {"FORBIDDEN_ACTOR", "FORBIDDEN_SESSION"}:
        return HTTPException(status_code=403, detail=exc.code)
    if exc.code in {
        "FORBIDDEN_SCOPE",
        "SESSION_NOT_FOUND",
        "OBJECT_NOT_FOUND",
        "UNSUPPORTED_ARTIFACT_KIND",
    }:
        return HTTPException(status_code=404, detail="NOT_FOUND")
    if exc.code == "MISSING_WORKSPACE":
        return HTTPException(status_code=409, detail=exc.code)
    return HTTPException(status_code=409, detail=exc.code)


@router.post(
    ROUTE_RUNTIME_CONTEXT_SCOPE_POLICIES,
    response_model=RuntimeContextScopePolicyV1,
)
def issue_runtime_context_scope_policy(
    body: IssueRuntimeContextScopePolicyCommandV1, request: Request
) -> RuntimeContextScopePolicyV1:
    ctx = require_route_authorization(
        request,
        authorization_class=RouteAuthorizationClass.TENANT_ADMIN,
        route_hint=ROUTE_RUNTIME_CONTEXT_SCOPE_POLICIES,
    )
    registry = authority_store = None
    try:
        registry, authority_store, _resolver, service = _components(ctx)
        return service.issue_scope_policy(
            actor_context=_actor(ctx),
            command=body,
            issued_at=datetime.now(timezone.utc),
        )
    except RuntimeContextError as exc:
        raise _translate(exc) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="MALFORMED_CONTRACT") from exc
    finally:
        if authority_store is not None:
            authority_store.close()
        if registry is not None:
            registry.close()


@router.get(
    ROUTE_RUNTIME_CONTEXT_SCOPE_POLICY,
    response_model=RuntimeContextScopePolicyV1,
)
def get_runtime_context_scope_policy(
    policy_ref: str,
    request: Request,
    policy_hash: str = Query(pattern=r"^sha256:[a-f0-9]{64}$"),
) -> RuntimeContextScopePolicyV1:
    ctx = require_route_authorization(
        request,
        authorization_class=RouteAuthorizationClass.TENANT_MEMBER,
        route_hint=ROUTE_RUNTIME_CONTEXT_SCOPE_POLICY,
    )
    registry = authority_store = None
    try:
        registry, authority_store, resolver, _service = _components(ctx)
        value = resolver.resolve_pins(
            actor_context=_actor(ctx),
            pins=(
                ArtifactPinV1(
                    artifact_kind="runtime_context_scope_policy",
                    artifact_ref=policy_ref,
                    artifact_hash=policy_hash,
                ),
            ),
        )[0]
        if not isinstance(value, RuntimeContextScopePolicyV1):
            raise RuntimeContextError("INTEGRITY_UNPROVABLE")
        return value
    except RuntimeContextError as exc:
        raise _translate(exc) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="MALFORMED_CONTRACT") from exc
    finally:
        if authority_store is not None:
            authority_store.close()
        if registry is not None:
            registry.close()


@router.post(
    ROUTE_RUNTIME_CONTEXT_ADMISSIONS,
    response_model=RuntimeContextElementAdmissionReceiptV1,
)
def admit_runtime_context_element(
    body: AdmitRuntimeContextElementCommandV1, request: Request
) -> RuntimeContextElementAdmissionReceiptV1:
    ctx = require_route_authorization(
        request,
        authorization_class=RouteAuthorizationClass.TENANT_MEMBER,
        route_hint=ROUTE_RUNTIME_CONTEXT_ADMISSIONS,
    )
    registry = authority_store = None
    try:
        registry, authority_store, _resolver, service = _components(ctx)
        return service.admit_element(
            actor_context=_actor(ctx),
            command=body,
            admitted_at=datetime.now(timezone.utc),
        )
    except RuntimeContextError as exc:
        raise _translate(exc) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="MALFORMED_CONTRACT") from exc
    finally:
        if authority_store is not None:
            authority_store.close()
        if registry is not None:
            registry.close()


@router.get(
    ROUTE_RUNTIME_CONTEXT_ADMISSION,
    response_model=RuntimeContextElementAdmissionReceiptV1,
)
def get_runtime_context_admission(
    admission_ref: str,
    request: Request,
    admission_hash: str = Query(pattern=r"^sha256:[a-f0-9]{64}$"),
) -> RuntimeContextElementAdmissionReceiptV1:
    ctx = require_route_authorization(
        request,
        authorization_class=RouteAuthorizationClass.TENANT_MEMBER,
        route_hint=ROUTE_RUNTIME_CONTEXT_ADMISSION,
    )
    registry = authority_store = None
    try:
        registry, authority_store, resolver, _service = _components(ctx)
        value = resolver.resolve_pins(
            actor_context=_actor(ctx),
            pins=(
                ArtifactPinV1(
                    artifact_kind="runtime_context_element_admission",
                    artifact_ref=admission_ref,
                    artifact_hash=admission_hash,
                ),
            ),
        )[0]
        if not isinstance(value, RuntimeContextElementAdmissionReceiptV1):
            raise RuntimeContextError("INTEGRITY_UNPROVABLE")
        return value
    except RuntimeContextError as exc:
        raise _translate(exc) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="MALFORMED_CONTRACT") from exc
    finally:
        if authority_store is not None:
            authority_store.close()
        if registry is not None:
            registry.close()


__all__ = ["router"]
