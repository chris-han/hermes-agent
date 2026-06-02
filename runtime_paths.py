"""Compatibility shim for the Semantier runtime-path helpers.

This submodule checkout runs tests from ``hermes-agent`` directly, while the
canonical ``runtime_paths.py`` lives in the parent repo's ``src/`` tree.
Execute that source in this module namespace so tests can patch globals like
``_WORKSPACES_ROOT`` and affect the exported helper functions normally.
"""

from __future__ import annotations

from pathlib import Path


_UPSTREAM_RUNTIME_PATHS = Path(__file__).resolve().parent.parent / "src" / "runtime_paths.py"

if not _UPSTREAM_RUNTIME_PATHS.exists():
    raise ModuleNotFoundError(
        f"Missing upstream runtime_paths implementation at {_UPSTREAM_RUNTIME_PATHS}"
    )

exec(compile(_UPSTREAM_RUNTIME_PATHS.read_text(encoding="utf-8"), str(_UPSTREAM_RUNTIME_PATHS), "exec"))
