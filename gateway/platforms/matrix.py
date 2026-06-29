"""Compatibility shim for the Matrix platform adapter."""

import sys

from plugins.platforms.matrix import adapter as _adapter
from plugins.platforms.matrix.adapter import *  # noqa: F401,F403

for _name in dir(_adapter):
    if _name.startswith("__"):
        continue
    globals()[_name] = getattr(_adapter, _name)

del _name

sys.modules[__name__] = _adapter
