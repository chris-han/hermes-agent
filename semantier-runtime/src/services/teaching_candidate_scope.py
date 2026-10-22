"""Governed DEVELOPMENT boundary for non-authoritative teaching execution."""
from agents.auth_db import load_organizations
from agents.auth_session import RequestContext
from contracts.knowledge_classification import RuntimeContextDTO
from services.runtime_context_transition_service import RuntimeAuthorityResolver


def require_teaching_scope(context, *, organization_id, workspace_id):
    # Only authenticated handler context may enter, never tool/model arguments.
    if (not isinstance(context, RequestContext) or not context.authenticated
        or (context.organization_id, context.workspace_id) != (organization_id, workspace_id)):
        raise PermissionError("TEACHING_CANDIDATE_AUTHENTICATED_SCOPE_REQUIRED")
    actor = RuntimeContextDTO(organization_id=organization_id, workspace_id=workspace_id,
                              actor_id=context.user_id)
    # This owner resolves DB-backed identity/membership; it has no artifact dependencies.
    RuntimeAuthorityResolver.require_current_actor(actor_context=actor)
    organization = load_organizations().get(organization_id)
    if organization is None or organization.get("dataset_type") != "DEVELOPMENT":
        raise PermissionError("TEACHING_CANDIDATE_DEVELOPMENT_ONLY")
