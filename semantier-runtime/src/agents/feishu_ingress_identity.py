from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agents.gateway_identity import resolve_feishu_owner_correlation


class FeishuIngressOwnerResolutionError(RuntimeError):
    """Base class for Feishu ingress ownership resolution failures."""


class FeishuIngressOwnerNotFound(FeishuIngressOwnerResolutionError):
    """Raised when no active owner correlation matches inbound Feishu data."""


class FeishuIngressOwnerConflict(FeishuIngressOwnerResolutionError):
    """Raised when multiple owners match inbound Feishu data."""


@dataclass(frozen=True)
class FeishuIngressOwnerContext:
    owner_user_id: str
    owner_workspace_id: str
    open_id: str | None = None
    union_id: str | None = None
    chat_id: str | None = None
    platform_session_key: str | None = None


def stamp_feishu_ingress_owner(event: Any) -> None:
    """Bind transport-authenticated Feishu aliases before Hermes derives session keys."""
    source = event.source
    raw_event = getattr(getattr(event, "raw_message", None), "event", None)
    sender_id = getattr(getattr(raw_event, "sender", None), "sender_id", None)
    open_id = str(getattr(sender_id, "open_id", "") or "").strip()
    union_id = str(getattr(sender_id, "union_id", "") or "").strip()
    if not open_id:
        candidate = str(getattr(source, "user_id", "") or "")
        open_id = candidate if candidate.startswith("ou_") else ""
    union_id = union_id or str(getattr(source, "user_id_alt", "") or "")
    owner = resolve_feishu_ingress_owner(
        open_id=open_id or None,
        union_id=union_id or None,
        chat_id=getattr(source, "chat_id", None),
        platform_session_key=getattr(event, "message_id", None),
    )
    source.workspace_owner_id = owner.owner_workspace_id
    source.user_id = open_id or owner.open_id or source.user_id


def resolve_feishu_ingress_owner(
    *,
    open_id: str | None = None,
    union_id: str | None = None,
    chat_id: str | None = None,
    platform_session_key: str | None = None,
) -> FeishuIngressOwnerContext:
    try:
        match = resolve_feishu_owner_correlation(
            open_id=open_id,
            union_id=union_id,
            chat_id=chat_id,
            platform_session_key=platform_session_key,
        )
    except ValueError as exc:
        raise FeishuIngressOwnerConflict(str(exc)) from exc
    if match is None:
        raise FeishuIngressOwnerNotFound("no active owner correlation for inbound Feishu identity")
    owner_user_id = str(match.get("owner_user_id") or "").strip()
    owner_workspace_id = str(match.get("owner_workspace_id") or "").strip()
    if not owner_user_id or not owner_workspace_id:
        raise FeishuIngressOwnerConflict("active Feishu correlation is missing owner identifiers")
    return FeishuIngressOwnerContext(
        owner_user_id=owner_user_id,
        owner_workspace_id=owner_workspace_id,
        open_id=str(match.get("linked_external_user_id") or "").strip() or None,
        union_id=str(match.get("linked_external_union_id") or "").strip() or None,
        chat_id=str(match.get("chat_id") or "").strip() or None,
        platform_session_key=str(match.get("platform_session_key") or "").strip() or None,
    )
