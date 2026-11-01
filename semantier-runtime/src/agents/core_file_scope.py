"""Authenticated core upload/wiki file scope, not a native session or identity owner."""
from dataclasses import dataclass
from pathlib import Path

from agents.auth_session import RequestContext
from runtime_paths import workspace_session_segment


@dataclass(frozen=True)
class FilePaths:
    hermes_home: Path
    uploads_root: Path


@dataclass(frozen=True)
class CoreFileScope:
    context: RequestContext
    session_id: str

    @classmethod
    def issue(cls, context, session_id):
        scope = cls(context, session_id)
        scope.validate()
        return scope

    @property
    def workspace_id(self): return self.context.workspace_id

    @property
    def user_id(self): return self.context.user_id

    @property
    def paths(self):
        root = self.context.workspace_root.absolute()
        segment = workspace_session_segment(self.workspace_id, self.session_id)
        return FilePaths(root, root / 'sessions' / segment / 'uploads')

    def validate(self):
        ctx = self.context
        if not isinstance(ctx, RequestContext) or not ctx.authenticated:
            raise PermissionError('authenticated core context required')
        from contracts.knowledge_classification import RuntimeContextDTO
        from services.runtime_context_transition_service import RuntimeAuthorityResolver
        from services.runtime_errors import RuntimeContextError
        actor = RuntimeContextDTO(organization_id=ctx.organization_id,
            workspace_id=ctx.workspace_id, actor_id=ctx.user_id)
        try:
            RuntimeAuthorityResolver.require_current_actor(actor_context=actor)
            _, org = RuntimeAuthorityResolver._current_identity_context(actor)
            if org.get('membership_status') != 'active':
                raise PermissionError('current workspace membership required')
        except RuntimeContextError as exc:
            raise PermissionError('current file authority required') from exc
        from agents.gateway_identity import resolve_workspace_paths
        root, home = resolve_workspace_paths(ctx.workspace_id)
        reject_links(root)
        reject_links(ctx.workspace_root.absolute())
        if ctx.workspace_root.absolute() != root or ctx.hermes_home.absolute() != home:
            raise PermissionError('workspace root binding mismatch')
        from agents.auth_db import is_postgresql
        if is_postgresql():
            from agents.auth_db_postgres import connection
            with connection() as c:
                row = c.execute('SELECT workspace_root FROM semantier_identity.workspace_runtime_bindings WHERE user_id=%s AND workspace_id=%s', (ctx.user_id, ctx.workspace_id)).fetchone()
            if not row or row['workspace_root'] != str(ctx.workspace_root.absolute()):
                raise PermissionError('current workspace root binding required')
        workspace_session_segment(ctx.workspace_id, self.session_id)

    def _resolve(self, path):
        self.validate()
        path = Path(path).absolute()
        reject_links(path)
        uploads = self.paths.uploads_root
        if path != uploads and path.parent != uploads:
            raise PermissionError('outside authorized upload root')
        if path.resolve() != path:
            raise PermissionError('noncanonical upload path')
        return path

    def resolve_read_path(self, path, *, purpose='document_input'):
        return self._resolve(path)

    def resolve_write_path(self, path, *, purpose='document_import'):
        return self._resolve(path)


def reject_links(path):
    for part in (path, *path.parents):
        if part.is_symlink() or (hasattr(part, 'is_junction') and part.is_junction()):
            raise PermissionError('symlink/junction file scope rejected')


def file_guard(boundary):
    if isinstance(boundary, CoreFileScope):
        boundary.validate()
        return boundary
    from agents.hermes_business_adapter import native_file_guard
    return native_file_guard(boundary)
