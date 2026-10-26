"""Resolve SDK credentials against current governed identity and explicit grants."""
from dataclasses import dataclass
from urllib.parse import urlsplit
from fastapi import HTTPException, Request
from agents.webapi.deps import require_route_authorization
from agents.route_policy import RouteAuthorizationClass


@dataclass(frozen=True)
class GatewayActor:
    user_id: str
    organization_id: str
    workspace_id: str
    permissions: frozenset[str]
    token_id: str | None = None
    account_id: str | None = None
    currency: str | None = None
    client_namespace: str | None = None


def current_membership(user_id, organization_id, workspace_id):
    from agents.gateway_identity import get_user_record, get_user_membership_for_organization
    user = get_user_record(user_id)
    membership = get_user_membership_for_organization(user_id=user_id, organization_id=organization_id)
    return bool(user and user.get('workspace_id') == workspace_id and membership and membership.get('membership_status') == 'active')


def csrf(request: Request):
    origin = request.headers.get('origin')
    expected = urlsplit(str(request.base_url))
    if not origin or (urlsplit(origin).scheme,urlsplit(origin).netloc) != (expected.scheme,expected.netloc):
        raise HTTPException(403, 'same-origin mutation required')
    if request.headers.get('x-semantier-csrf') != '1':
        raise HTTPException(403, 'CSRF header required')


async def actor_for(request: Request, service, permission: str, *, browser_only=False):
    if not service or not service.enabled:
        raise HTTPException(503, 'model gateway disabled')
    header = request.headers.get('authorization', '')
    if header.startswith('Bearer ') and not browser_only:
        token = await service.store.token(header[7:])
        if not token:
            raise HTTPException(401, 'invalid gateway credential')
        uid, org, ws = token['user_id'],token['organization_id'],token['workspace_id']
        permissions = set(token['permissions']) & await service.store.permissions(org,uid)
        tid = token['token_id']
        aid, currency = token.get('account_id'), token.get('currency')
        client_namespace = None
        if aid is not None:
            async with await service.store._connection() as connection:
                access = await (await connection.execute("SELECT 1 FROM semantier_gateway.account_access WHERE organization_id=%s AND account_id=%s AND currency=%s AND user_id=%s AND status='ACTIVE'", (org, aid, currency, uid))).fetchone()
                device = await (await connection.execute('SELECT context FROM semantier_gateway.device_connections WHERE token_id=%s', (tid,))).fetchone()
                if device:
                    from contracts.model_gateway import canonical_hash
                    client_namespace=canonical_hash([device['context']['runtime_ref'],device['context']['profile_ref']])
            if not access:
                raise HTTPException(403, 'current account access required')
    else:
        ctx = require_route_authorization(request, authorization_class=RouteAuthorizationClass.TENANT_MEMBER,
                                          route_hint=f'{request.method} {request.url.path}')
        uid, org, ws = ctx.user_id,ctx.organization_id,ctx.workspace_id
        asserted_org=request.headers.get('x-semantier-organization-id')
        if asserted_org is not None and asserted_org!=org:
            raise HTTPException(409,'organization context changed; refresh before continuing')
        permissions = await service.store.permissions(org,uid)
        tid = None
        aid, currency, client_namespace = None, None, None
        if request.method not in {'GET','HEAD'}:
            csrf(request)
    if not org or not current_membership(uid,org,ws):
        raise HTTPException(403, 'current active membership required')
    required={permission} if isinstance(permission,str) else set(permission)
    if not required & permissions:
        raise HTTPException(403, 'explicit '+ ' or '.join(sorted(required))+' permission required')
    return GatewayActor(uid,org,ws,frozenset(permissions),tid,aid,currency,client_namespace)
