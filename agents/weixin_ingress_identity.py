"""Optional governed Weixin ingress identity seam."""


class WeixinIngressOwnerResolutionError(RuntimeError):
    pass


def resolve_weixin_ingress_owner(**_kwargs):
    raise WeixinIngressOwnerResolutionError(
        "Weixin ingress owner resolution is unavailable in this checkout."
    )
