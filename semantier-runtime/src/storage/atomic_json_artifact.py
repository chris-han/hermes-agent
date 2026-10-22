"""Cross-process publication for immutable JSON artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4
from storage.governed_file_io import exclusive_file_lock, sync_directory, clean_publication_temps


def publish_immutable_json(
    target: Path,
    payload: dict[str, Any],
    *,
    conflict_error: str,
    equivalent: Callable[[dict[str, Any], dict[str, Any]], bool] | None = None,
) -> dict[str, Any]:
    """Atomically publish once or return the equivalent immutable winner."""
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
    lock_path = target.with_name(f".{target.name}.publish.lock")
    equivalent = equivalent or (lambda existing, proposed: existing == proposed)

    with exclusive_file_lock(lock_path):
        clean_publication_temps(target)
        temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
        try:
            if target.is_file():
                existing = json.loads(target.read_text(encoding="utf-8"))
                if not isinstance(existing, dict) or not equivalent(existing, payload):
                    raise ValueError(conflict_error)
                sync_directory(target.parent)
                return existing

            with temporary.open("x", encoding="utf-8") as temporary_file:
                temporary_file.write(encoded)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary, target)
            sync_directory(target.parent)
            return payload
        finally:
            if temporary.exists():
                temporary.unlink()


__all__ = ["publish_immutable_json"]
