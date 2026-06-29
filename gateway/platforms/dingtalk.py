"""Compatibility shim for the DingTalk platform adapter."""

import sys
import types

from plugins.platforms.dingtalk import adapter as _adapter
from plugins.platforms.dingtalk.adapter import *  # noqa: F401,F403

for _name in dir(_adapter):
    if _name.startswith("__"):
        continue
    globals()[_name] = getattr(_adapter, _name)


class _DingTalkShimModule(types.ModuleType):
    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name not in {"_adapter", "__class__"} and hasattr(_adapter, name):
            setattr(_adapter, name, value)


sys.modules[__name__].__class__ = _DingTalkShimModule

del _name
