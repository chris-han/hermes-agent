from __future__ import annotations

from pathlib import Path
import re

_USER_MEMORY_BLOCKED_LINE_RE = re.compile(
    r"(?i)(organization_id|member_role|membership_status|dataset_type|industry_code|"
    r"active organization|default organization|current organization|organization context|"
    r"用户公司是|你的公司是|当前组织是|默认组织是|组织id|组织 ID|组织上下文|成员角色)"
)

_USER_MEMORY_BOUNDARY_NOTE = (
    "组织身份、成员角色和组织级 authority context 由 governed runtime context 注入，不应保存在 USER.md。"
)


def sanitize_user_memory_profile(hermes_home: Path) -> bool:
    """Strip authority-bearing organization assertions from USER.md.

    USER.md is unmanaged contextual memory. It must not carry organization
    identity, membership, or other authority-bearing facts that can conflict
    with governed runtime context.
    """

    user_md_path = hermes_home / "memories" / "USER.md"
    if not user_md_path.exists():
        return False

    original = user_md_path.read_text(encoding="utf-8")
    lines = original.splitlines()

    kept_lines: list[str] = []
    removed_any = False
    note_present = False
    for line in lines:
        stripped = line.strip()
        if stripped == _USER_MEMORY_BOUNDARY_NOTE:
            note_present = True
            kept_lines.append(line)
            continue
        if _USER_MEMORY_BLOCKED_LINE_RE.search(line):
            removed_any = True
            continue
        kept_lines.append(line)

    if not removed_any:
        return False

    sanitized_lines = [line for line in kept_lines if line.strip()]
    if not note_present:
        sanitized_lines.append(_USER_MEMORY_BOUNDARY_NOTE)
    body = "\n".join(sanitized_lines).strip()
    if body:
        body += "\n"
    user_md_path.write_text(body, encoding="utf-8")
    return True


def reconcile_retained_context_items(*, context) -> None:
    """Reconcile governed managed items before a new frozen Hermes snapshot.

    A projection failure propagates: loading an old managed original would
    contradict the retained correction/suppression intent.
    """
    from services.retained_context_item_service import RetainedContextItemService
    from storage.ontology_build_store import OntologyBuildStore, ontology_build_store_path

    RetainedContextItemService._check(context)
    store = OntologyBuildStore(ontology_build_store_path(context.workspace_root))
    try:
        RetainedContextItemService(store=store).reconcile(context=context)
    finally:
        store.close()


def reconcile_retained_context_for_boundary(boundary, *, context=None) -> None:
    """Use an authenticated context or current governed identity, never memory."""
    from agents.auth_session import RequestContext
    from agents.gateway_identity import get_user_record, get_user_organization_context
    from runtime_paths import workspace_hermes_home_path
    from storage.ontology_build_store import OntologyBuildStore, ontology_build_store_path

    if not boundary.workspace_id:
        return
    workspace_home = workspace_hermes_home_path(boundary.workspace_id).resolve()
    store_path = ontology_build_store_path(workspace_home)
    if not store_path.exists():
        return
    store = OntologyBuildStore(store_path)
    try:
        has_managed_intent = store.has_retained_context_changes(workspace_id=boundary.workspace_id)
    finally:
        store.close()
    if not has_managed_intent:
        return
    if (boundary.paths.hermes_home is None
            or Path(boundary.paths.hermes_home).resolve() != workspace_home):
        raise PermissionError('retained context requires governed workspace runtime home')

    if context is None:
        if not boundary.user_id or not boundary.workspace_id:
            raise PermissionError('retained context reconciliation requires governed actor/workspace')
        user = get_user_record(boundary.user_id)
        if not user or user.get('workspace_id') != boundary.workspace_id:
            raise PermissionError('execution boundary actor does not own the workspace')
        organization = get_user_organization_context(boundary.user_id)
        if not organization or not organization.get('organization_id'):
            # Managed intent already exists: skipping recovery would let a
            # stale original become a new frozen memory snapshot. A missing
            # governed organization cannot be inferred from memory.
            raise PermissionError('managed retained context recovery requires governed organization')
        context = RequestContext(authenticated=True, user_id=boundary.user_id,
            workspace_id=boundary.workspace_id, workspace_slug=boundary.workspace_id,
            workspace_root=boundary.paths.hermes_home, hermes_home=boundary.paths.hermes_home,
            organization_id=organization['organization_id'],
            membership_status=organization.get('membership_status', 'unassigned'))
    if (context.user_id != boundary.user_id or context.workspace_id != boundary.workspace_id
            or Path(context.hermes_home).resolve() != Path(boundary.paths.hermes_home).resolve()):
        raise PermissionError('retained context does not match execution boundary')
    if not context.organization_id:
        raise PermissionError('managed retained context recovery requires governed organization')
    reconcile_retained_context_items(context=context)
