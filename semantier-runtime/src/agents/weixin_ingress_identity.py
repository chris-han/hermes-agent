from __future__ import annotations

from dataclasses import dataclass

from agents.gateway_identity import resolve_weixin_owner_correlation


class WeixinIngressOwnerResolutionError(RuntimeError):
    """Base class for Weixin ingress ownership resolution failures."""


class WeixinIngressOwnerNotFound(WeixinIngressOwnerResolutionError):
    """Raised when no active owner correlation matches inbound Weixin data."""


class WeixinIngressOwnerConflict(WeixinIngressOwnerResolutionError):
    """Raised when multiple active owners match inbound Weixin data."""


@dataclass(frozen=True)
class WeixinIngressOwnerContext:
    owner_user_id: str
    owner_workspace_id: str
    account_id: str | None = None
    external_user_id: str | None = None
    chat_id: str | None = None
    platform_session_key: str | None = None


def resolve_weixin_ingress_owner(
    *,
    account_id: str | None,
    external_user_id: str | None,
    chat_id: str | None = None,
    platform_session_key: str | None = None,
) -> WeixinIngressOwnerContext:
    try:
        match = resolve_weixin_owner_correlation(
            account_id=account_id,
            external_user_id=external_user_id,
            chat_id=chat_id,
            platform_session_key=platform_session_key,
        )
    except ValueError as exc:
        raise WeixinIngressOwnerConflict(str(exc)) from exc

    if match is None:
        raise WeixinIngressOwnerNotFound(
            "no active owner correlation for inbound Weixin identity",
        )

    owner_user_id = str(match.get("owner_user_id") or "").strip()
    owner_workspace_id = str(match.get("owner_workspace_id") or "").strip()
    if not owner_user_id or not owner_workspace_id:
        raise WeixinIngressOwnerConflict(
            "active owner correlation is missing owner identifiers",
        )

    return WeixinIngressOwnerContext(
        owner_user_id=owner_user_id,
        owner_workspace_id=owner_workspace_id,
        account_id=str(match.get("linked_account_id") or "").strip() or None,
        external_user_id=str(match.get("linked_external_user_id") or "").strip()
        or None,
        chat_id=str(match.get("chat_id") or "").strip() or None,
        platform_session_key=str(match.get("platform_session_key") or "").strip()
        or None,
    )
