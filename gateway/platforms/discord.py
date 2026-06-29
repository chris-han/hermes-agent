"""Compatibility shim for the Discord platform adapter."""

import sys
import types

from plugins.platforms.discord import adapter as _adapter
from plugins.platforms.discord.adapter import *  # noqa: F401,F403

for _name in dir(_adapter):
    if _name.startswith("__"):
        continue
    globals()[_name] = getattr(_adapter, _name)

try:
    import discord as discord  # noqa: F401
except ImportError:
    discord = None
    DISCORD_AVAILABLE = False
else:
    DISCORD_AVAILABLE = getattr(_adapter, "DISCORD_AVAILABLE", True)


def _define_discord_view_classes():
    result = _adapter._define_discord_view_classes()
    for _view_name in (
        "ExecApprovalView",
        "SlashConfirmView",
        "UpdatePromptView",
        "ModelPickerView",
        "ClarifyChoiceView",
    ):
        if hasattr(_adapter, _view_name):
            globals()[_view_name] = getattr(_adapter, _view_name)
    return result


def check_discord_requirements() -> bool:
    result = _adapter.check_discord_requirements()
    if result:
        _define_discord_view_classes()
    globals()["DISCORD_AVAILABLE"] = getattr(_adapter, "DISCORD_AVAILABLE", result)
    globals()["discord"] = getattr(_adapter, "discord", discord)
    return result


class _DiscordShimModule(types.ModuleType):
    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name not in {"_adapter", "__class__", "_define_discord_view_classes"} and hasattr(_adapter, name):
            setattr(_adapter, name, value)


sys.modules[__name__].__class__ = _DiscordShimModule

del _name
