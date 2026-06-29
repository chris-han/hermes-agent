"""Compatibility shim for the Slack platform adapter."""

from plugins.platforms.slack.adapter import *  # noqa: F401,F403
from plugins.platforms.slack.adapter import (  # noqa: F401
    AsyncApp,
    AsyncSocketModeHandler,
    AsyncWebClient,
    _resolve_slack_proxy_url,
    _slash_user_id,
    is_host_excluded_by_no_proxy,
    resolve_proxy_url,
)
