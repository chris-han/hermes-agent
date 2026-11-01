"""Incumbent native-host edge only. No native task/transport qualification implied."""
import json
from pathlib import Path


def native_file_guard(boundary):
    from gateway.execution_boundary import DiskIOBoundaryGuard, ExecutionBoundary
    if not isinstance(boundary, ExecutionBoundary):
        raise TypeError('core scope or validated native execution boundary required')
    return DiskIOBoundaryGuard(boundary)


def wiki_scope(boundary):
    from gateway.execution_boundary import current_execution_boundary
    from agents.auth_session import RequestContext
    from agents.gateway_identity import get_user_organization_context
    from agents.core_file_scope import CoreFileScope
    from runtime_paths import workspace_root_path
    if (boundary is None or current_execution_boundary() is not boundary
            or boundary.audit_metadata.get('authority_source') != 'semantier_authenticated_context'
            or boundary.audit_metadata.get('wiki_task_profile') != 'documentary-wiki.v1'):
        raise PermissionError('authenticated server-owned wiki profile required')
    org = get_user_organization_context(boundary.user_id) or {}
    root = workspace_root_path(boundary.workspace_id)
    ctx = RequestContext(authenticated=True, user_id=boundary.user_id,
        workspace_id=boundary.workspace_id, workspace_slug=boundary.workspace_id,
        organization_id=org.get('organization_id'), workspace_root=root,
        hermes_home=root, membership_status=org.get('membership_status', 'unassigned'))
    if boundary.paths.hermes_home != root:
        raise PermissionError('native workspace root mismatch')
    return CoreFileScope.issue(ctx, boundary.session_id)


def dispatch_native(name, args, **kwargs):
    from gateway.execution_boundary import current_execution_boundary
    from agents.wiki_task_runtime import dispatch
    return dispatch(wiki_scope(current_execution_boundary()), name, args, **kwargs)


def register_wiki_tools():
    from tools.registry import registry
    prompts = Path(__file__).resolve().parents[1] / 'prompts/agents'
    for schema in json.loads((prompts/'wiki_task_tools.json').read_text()):
        name = schema['name']
        registry.register(name=name, toolset='file', schema=schema,
            handler=lambda args, _name=name, **kw: dispatch_native(_name, args, **kw))


def read_native_work_receipt(args, **kwargs):
    """Native edge supplies only scope; governed core revalidates live authority."""
    from contracts.workflow_execution import meeting_context_from_execution_boundary
    from agents.auth_session import background_identity_context
    from agents.core_work_receipts import read_receipt
    bound = meeting_context_from_execution_boundary()
    from agents.gateway_identity import get_user_record
    user = get_user_record(bound.user_id) or {}
    scope = {name: getattr(bound, name) for name in ('user_id', 'workspace_id', 'organization_id')}
    scope['workspace_slug'] = user.get('workspace_slug')
    context = background_identity_context(**scope)
    return json.dumps(read_receipt(context, args), ensure_ascii=False)


def register_core_work_receipt_tool():
    # Complete native builtin discovery before retirement; a later lazy import
    # must not re-register the removed scheduler tools. No agent is started.
    from model_tools import registry
    prompts = Path(__file__).resolve().parents[1] / 'prompts/agents'
    schema = json.loads((prompts / 'core_work_receipt_tool.json').read_text())
    registry.register(name=schema['name'], toolset='cronjob', schema=schema,
                      handler=read_native_work_receipt)
    # Old schedule/script mutation has no selected server owner. Native IDs
    # must not leak back through an otherwise enabled cron toolset.
    registry.deregister('cronjob_manage')
    registry.deregister('cronjob')
