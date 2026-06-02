"""Compatibility shim for optional Semantier Weixin ingress identity helpers."""

from __future__ import annotations


class WeixinIngressOwnerResolutionError(RuntimeError):
    """Raised when a Weixin inbound message cannot be mapped to an owner."""


def resolve_weixin_ingress_owner(**_kwargs):
    """Repo-only fallback: no governed owner resolution is available here."""
    raise WeixinIngressOwnerResolutionError(
        "Weixin ingress owner resolution is unavailable in this checkout."
    )
