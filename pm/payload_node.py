"""Relocatable shared Node selection for sealed Python Playwright consumers."""
from __future__ import annotations

import os
from pathlib import Path
from typing import MutableMapping


def configure_payload_node(project_root: Path, env: MutableMapping[str, str] | None = None) -> None:
    """Use the payload's recorded Node, preserving an explicit operator override.

    This only reads installed facts. It must never acquire tools or resolve a
    user-global Node for an isolated packaged runtime.
    """
    from pm.environments import _payload_manifest, _payload_path
    from pm.lock import Facts
    from pm.registry import get_package
    from pm.store import current_target

    env = os.environ if env is None else env
    if env.get("PLAYWRIGHT_NODEJS_PATH"):
        return
    root = Path(project_root).resolve()
    manifest = _payload_manifest(root)
    if not manifest or not manifest.get("store"):
        return
    store = _payload_path(root, manifest["store"], "store")
    fact = Facts(store / "facts.json").get("node")
    if not fact:
        return  # Older/minimal payloads retain Playwright's private runtime.
    entry = (store / fact["entry"]).resolve()
    if not entry.is_relative_to(store):
        raise RuntimeError("packaged Node entry escapes its tool store")
    binary = get_package("node").binary(entry, current_target())
    if binary is None or not binary.is_file():
        raise RuntimeError("packaged shared Node runtime is missing; repair the installation")
    env["PLAYWRIGHT_NODEJS_PATH"] = str(binary)
