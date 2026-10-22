"""Decode governed workspace-upload identity references without disk access."""

from __future__ import annotations

import base64
import json
from pathlib import PurePosixPath
from typing import Any


KNOWLEDGE_UPLOAD_REF_PREFIX = "knowledge-upload:v1:"


def decode_governed_upload_ref(
    ref: str,
    *,
    expected_workspace_id: str | None = None,
) -> str:
    """Return the validated workspace-relative path embedded in ``ref``.

    The identity is decoded before any filesystem operation. Its path is
    restricted to the governed ``wiki/uploads`` subtree, while the caller's
    filesystem boundary remains responsible for final resolved containment.
    """
    if not ref.startswith(KNOWLEDGE_UPLOAD_REF_PREFIX):
        raise ValueError("invalid governed upload reference")
    encoded = ref[len(KNOWLEDGE_UPLOAD_REF_PREFIX):]
    try:
        payload: Any = json.loads(
            base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)),
        )
    except Exception as exc:
        raise ValueError("invalid governed upload reference") from exc
    if not isinstance(payload, dict):
        raise ValueError("invalid governed upload reference")

    workspace_id = payload.get("workspaceId")
    if (
        expected_workspace_id is not None
        and workspace_id != expected_workspace_id
    ):
        raise ValueError("governed upload reference workspace mismatch")

    relative_value = payload.get("relativePath")
    if not isinstance(relative_value, str) or not relative_value.strip():
        raise ValueError("invalid governed upload reference")
    relative = PurePosixPath(relative_value)
    if (
        relative.is_absolute()
        or relative.parts[:2] != ("wiki", "uploads")
        or ".." in relative.parts
    ):
        raise ValueError("governed upload reference escapes upload root")
    return relative.as_posix()
