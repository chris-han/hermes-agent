from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import re
import shutil
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_payload(payload: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-") or "document"


@dataclass(frozen=True)
class BoundSession:
    workspace_home: Path
    session_id: str
    session_root: Path

    @property
    def uploads_root(self) -> Path:
        return self.session_root / "uploads"

    @property
    def artifacts_root(self) -> Path:
        return self.session_root / "artifacts"

    @property
    def tmp_root(self) -> Path:
        return self.session_root / "tmp"


def _session_env(name: str) -> str:
    try:
        from gateway.session_context import get_session_env, session_context_engaged
    except ImportError:
        return os.environ.get(name) or os.environ.get(f"HERMES_{name}") or ""
    value = get_session_env(f"HERMES_{name}", "")
    # An explicitly cleared task must not inherit another request's process env.
    if session_context_engaged():
        return value
    return value or os.environ.get(name, "")


def _active_workspace_home() -> str:
    session_home = _session_env("SESSION_HERMES_HOME")
    if session_home:
        return session_home
    try:
        from runtime_paths import current_workspace_hermes_home

        value = current_workspace_hermes_home()
        if value:
            return value
    except Exception:
        pass
    return _session_env("SESSION_HERMES_HOME") or os.environ.get("HERMES_HOME", "")


def bound_session(args: dict[str, Any]) -> BoundSession:
    workspace_home_raw = _active_workspace_home().strip()
    session_id = _session_env("SESSION_ID").strip()
    if not workspace_home_raw or not session_id:
        raise RuntimeError("SESSION_BINDING_REQUIRED")
    try:
        from runtime_paths import workspace_session_runs_root_from_hermes_home

        runs_root = workspace_session_runs_root_from_hermes_home(
            Path(workspace_home_raw), session_id
        )
    except Exception as exc:
        raise RuntimeError("SESSION_BINDING_INVALID") from exc
    session_root = runs_root.parent.resolve()
    requested_session = str(args.get("session_id") or "").strip()
    if requested_session and requested_session != session_id:
        raise RuntimeError("SESSION_CLAIM_MISMATCH")
    requested_workspace = str(args.get("workspace_id") or "").strip()
    if requested_workspace and requested_workspace != Path(workspace_home_raw).name:
        raise RuntimeError("WORKSPACE_CLAIM_MISMATCH")
    return BoundSession(
        workspace_home=Path(workspace_home_raw).expanduser().resolve(),
        session_id=session_id,
        session_root=session_root,
    )


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _upload_filename_match_key(name: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(name)).casefold().strip()
    return re.sub(r"[\s_\-]+", "", normalized)


def _resolve_upload_filename(bound: BoundSession, raw: str) -> Path | None:
    """Resolve a model-supplied upload filename without leaving the session root."""
    if not raw.startswith("uploads/"):
        return None
    requested_name = Path(raw).name
    requested_key = _upload_filename_match_key(requested_name)
    if not requested_key:
        return None
    matches = [
        candidate
        for candidate in bound.uploads_root.iterdir()
        if candidate.is_file() and _upload_filename_match_key(candidate.name) == requested_key
    ] if bound.uploads_root.is_dir() else []
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        raise RuntimeError("UPLOAD_RESOLUTION_AMBIGUOUS")
    return None


def resolve_session_ref(bound: BoundSession, ref: str, *, for_write: bool = False) -> Path:
    raw = str(ref or "").strip()
    if not raw:
        raise RuntimeError("ARTIFACT_REF_REQUIRED")
    if raw.startswith('sessions/') or Path(raw).is_absolute():
        allocated = (bound.workspace_home / raw).resolve()
        if _inside(allocated, bound.session_root):
            raw = str(allocated.relative_to(bound.session_root))
        elif Path(raw).is_absolute():
            return _import_hermes_document_cache_ref(bound, Path(raw))
        else:
            raise RuntimeError("CROSS_SCOPE_ARTIFACT_REF_REJECTED")
    if raw.startswith(("http://", "https://", "file://")):
        raise RuntimeError("REMOTE_ARTIFACT_REF_REJECTED")
    if raw == "wiki" or raw.startswith("wiki/"):
        candidate = (bound.workspace_home / raw).resolve()
        if not _inside(candidate, bound.workspace_home):
            raise RuntimeError("CROSS_SCOPE_ARTIFACT_REF_REJECTED")
        if not for_write and not candidate.exists():
            raise RuntimeError("ARTIFACT_NOT_FOUND")
        return candidate
    candidate = (bound.session_root / raw).resolve()
    if not _inside(candidate, bound.session_root):
        raise RuntimeError("CROSS_SCOPE_ARTIFACT_REF_REJECTED")
    if not candidate.exists() and raw.startswith("uploads/"):
        resolved_upload = _resolve_upload_filename(bound, raw)
        if resolved_upload is not None:
            return resolved_upload
        staged = (bound.artifacts_root / raw).resolve()
        if _inside(staged, bound.artifacts_root) and staged.is_file():
            bound.uploads_root.mkdir(parents=True, exist_ok=True)
            destination = (bound.uploads_root / Path(raw).name).resolve()
            if not _inside(destination, bound.uploads_root):
                raise RuntimeError("CROSS_SCOPE_ARTIFACT_REF_REJECTED")
            if not destination.exists():
                shutil.copy2(staged, destination)
            return destination
    if not for_write and not candidate.exists():
        raise RuntimeError("ARTIFACT_NOT_FOUND")
    return candidate


def _allowed_document_cache_roots() -> list[Path]:
    roots = []
    for raw in (
        os.environ.get("HERMES_DOCUMENT_CACHE_DIR"),
        os.environ.get("SEMANTIER_HERMES_DOCUMENT_CACHE_DIR"),
    ):
        if raw:
            roots.append(Path(raw).expanduser().resolve())
    roots.append((Path.home() / ".hermes" / "cache" / "documents").resolve())
    return list(dict.fromkeys(roots))


def _import_hermes_document_cache_ref(bound: BoundSession, source: Path) -> Path:
    resolved = source.expanduser().resolve()
    if not resolved.is_file():
        raise RuntimeError("ARTIFACT_NOT_FOUND")
    if not any(_inside(resolved, root) for root in _allowed_document_cache_roots()):
        raise RuntimeError("ABSOLUTE_ARTIFACT_REF_REJECTED")
    bound.uploads_root.mkdir(parents=True, exist_ok=True)
    safe_name = slug(resolved.name)
    if not Path(safe_name).suffix and resolved.suffix:
        safe_name = f"{safe_name}{resolved.suffix.lower()}"
    digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:12]
    destination = (bound.uploads_root / f"imported-{digest}-{safe_name}").resolve()
    if not _inside(destination, bound.uploads_root):
        raise RuntimeError("CROSS_SCOPE_ARTIFACT_REF_REJECTED")
    if not destination.exists():
        shutil.copy2(resolved, destination)
    return destination


def media_type_for(path: Path, hint: str | None = None) -> str:
    if hint:
        return hint
    guessed, _ = mimetypes.guess_type(path.name)
    if guessed:
        return guessed
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    if suffix == ".xlsx":
        return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    return "application/octet-stream"


def artifact_ref(bound: BoundSession, path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(bound.session_root.resolve()))
    except ValueError:
        try:
            relative_to_workspace = resolved.relative_to(bound.workspace_home.resolve())
        except ValueError:
            raise
        if relative_to_workspace.parts and relative_to_workspace.parts[0] == "wiki":
            return str(relative_to_workspace)
        raise


def write_json_artifact(
    bound: BoundSession, *, area: str, name: str, payload: dict[str, Any]
) -> tuple[str, Path]:
    root = bound.artifacts_root / area
    root.mkdir(parents=True, exist_ok=True)
    artifact = dict(payload)
    artifact["artifact_hash"] = sha256_payload(artifact)
    path = root / f"{slug(name)}.json"
    path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return artifact_ref(bound, path), path
