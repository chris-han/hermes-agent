from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from gateway.execution_boundary import (
    BoundaryPathRejected,
    DiskIOBoundaryGuard,
    ExecutionBoundary,
    GovernedExecutionBoundaryRequired,
)


EXPOSE_HERMES_BUNDLED_SKILLS_ENV = "SEMANTIER_EXPOSE_HERMES_BUNDLED_SKILLS"
MANAGED_BUNDLED_SKILL_ROOT_NAME = "hermes-bundled-skills"


class SkillManifestChanged(RuntimeError):
    pass


class BundledSkillScriptExecutionBlocked(RuntimeError):
    pass


@dataclass(frozen=True)
class GovernedDiagnosticInspectionContext:
    user_id: str
    workspace_id: str
    organization_id: str
    role: str
    request_id: str
    inspected_at_utc: str
    audit_actor: str
    audit_source: str
    authority_source: str = "governed_auth_state"

    def valid(self) -> bool:
        return all(
            str(value or "").strip()
            for value in (
                self.user_id,
                self.workspace_id,
                self.organization_id,
                self.request_id,
                self.inspected_at_utc,
                self.audit_actor,
                self.audit_source,
            )
        ) and self.authority_source == "governed_auth_state"

    def to_manifest_context(self) -> dict[str, str]:
        return {
            "type": "governed_diagnostic_inspection",
            "user_id": self.user_id,
            "workspace_id": self.workspace_id,
            "organization_id": self.organization_id,
            "role": self.role,
            "request_id": self.request_id,
            "audit_actor": self.audit_actor,
            "audit_source": self.audit_source,
            "authority_source": self.authority_source,
            "inspected_at_utc": self.inspected_at_utc,
        }


@dataclass(frozen=True)
class BundledSkillManifestEntry:
    name: str
    skill_id: str
    source_ref: str
    source_path: Path
    skill_md_sha256: str
    supporting_file_hashes: Mapping[str, str]
    workspace_id: str
    session_id: str | None
    generated_at_utc: str
    audit_metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def allowed_files(self) -> set[Path]:
        files = {self.source_path / "SKILL.md"}
        files.update(self.source_path / rel for rel in self.supporting_file_hashes)
        return {path.resolve() for path in files}

    def to_record(self, *, disabled_skill_names: Iterable[str] = ()) -> dict[str, Any]:
        disabled_names = {str(name) for name in disabled_skill_names}
        enabled = self.skill_id not in disabled_names and self.name not in disabled_names
        return {
            "id": self.skill_id,
            "slug": self.skill_id.replace("/", "-"),
            "name": self.name,
            "description": _frontmatter(self.source_path / "SKILL.md").get("description", ""),
            "author": "Hermes bundled skill",
            "triggers": [],
            "tags": ["bundled"],
            "homepage": None,
            "category": self.skill_id.split("/", 1)[0] if "/" in self.skill_id else "general",
            "icon": "B",
            "content": "",
            "fileCount": 1 + len(self.supporting_file_hashes),
            "sourcePath": str(self.source_path.resolve()),
            "installed": True,
            "enabled": enabled,
            "builtin": True,
            "sourceTier": "application",
            "sourceLabel": "Bundled",
            "canEdit": False,
            "canUninstall": False,
            "canModify": True,
            "platformCompatibility": [],
            "version": None,
            "installedVersion": None,
            "latestVersion": None,
            "contentHash": self.skill_md_sha256,
            "latestHash": None,
            "updateStatus": None,
            "hubIdentifier": f"hermes-agent/skills/{self.skill_id}",
            "hubSource": "hermes-agent",
            "hubUpdatedAt": None,
            "packageType": "bundled-skill",
            "packagePath": self.source_ref,
            "canUpdate": False,
            "security": {
                "level": "boundary-controlled",
                "flags": ["execution-boundary-required", "manifest-pinned"],
                "score": 0,
            },
            "bundledSkillManifest": {
                "name": self.name,
                "skillId": self.skill_id,
                "sourceRef": self.source_ref,
                "skillMdSha256": self.skill_md_sha256,
                "supportingFileHashes": dict(self.supporting_file_hashes),
                "workspaceId": self.workspace_id,
                "sessionId": self.session_id,
                "generatedAtUtc": self.generated_at_utc,
                "auditMetadata": dict(self.audit_metadata),
            },
        }


@dataclass(frozen=True)
class BundledSkillManifest:
    entries: tuple[BundledSkillManifestEntry, ...]
    context: Mapping[str, str]
    generated_at_utc: str

    def require(self, name_or_id: str) -> BundledSkillManifestEntry:
        for entry in self.entries:
            if entry.name == name_or_id or entry.skill_id == name_or_id:
                return entry
        raise KeyError(name_or_id)


def bundled_skill_exposure_enabled() -> bool:
    return str(os.environ.get(EXPOSE_HERMES_BUNDLED_SKILLS_ENV, "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def repo_bundled_skills_root() -> Path:
    from agents.launcher import _repo_hermes_agent_root
    return _repo_hermes_agent_root() / "skills"


def managed_bundled_skill_root(runtime_root: Path) -> Path:
    return runtime_root / "managed" / MANAGED_BUNDLED_SKILL_ROOT_NAME


def raw_bundled_skill_roots() -> tuple[Path, ...]:
    return (repo_bundled_skills_root().resolve(),)


def forbidden_global_external_skill_roots(runtime_root: Path) -> set[Path]:
    return {
        *raw_bundled_skill_roots(),
        managed_bundled_skill_root(runtime_root).resolve(),
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frontmatter(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    if not raw.startswith("---\n"):
        return {}
    end = raw.find("\n---", 4)
    if end == -1:
        return {}
    try:
        parsed = yaml.safe_load(raw[4:end]) or {}
    except yaml.YAMLError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _skill_dirs(source_root: Path) -> Iterable[Path]:
    if not source_root.exists():
        return ()
    return sorted({path.parent for path in source_root.rglob("SKILL.md")})


def _context_from_boundary(boundary: ExecutionBoundary) -> dict[str, str]:
    if not boundary.workspace_id or not boundary.session_id:
        raise GovernedExecutionBoundaryRequired(
            "SEMANTIER_EXECUTION_BOUNDARY_REQUIRED: bundled skill inventory requires workspace and session"
        )
    return {
        "type": "execution_boundary",
        "workspace_id": boundary.workspace_id,
        "session_id": boundary.session_id,
        "user_id": boundary.user_id or "",
        "authority_source": str(
            boundary.audit_metadata.get("authority_source")
            or "semantier_authenticated_context"
        ),
    }


def _manifest_context(
    *,
    auth_context: GovernedDiagnosticInspectionContext | None,
    boundary: ExecutionBoundary | None,
) -> dict[str, str] | None:
    if boundary is not None:
        return _context_from_boundary(boundary)
    if auth_context is not None and auth_context.valid():
        return auth_context.to_manifest_context()
    return None


def list_bundled_skill_inventory(
    *,
    auth_context: GovernedDiagnosticInspectionContext | None = None,
    boundary: ExecutionBoundary | None = None,
    source_root: Path | None = None,
) -> BundledSkillManifest:
    generated_at = _utc_now()
    context = _manifest_context(auth_context=auth_context, boundary=boundary)
    if not bundled_skill_exposure_enabled() or context is None:
        return BundledSkillManifest(entries=(), context={}, generated_at_utc=generated_at)

    root = (source_root or repo_bundled_skills_root()).resolve()
    entries: list[BundledSkillManifestEntry] = []
    for skill_dir in _skill_dirs(root):
        skill_md = skill_dir / "SKILL.md"
        try:
            relative = skill_dir.relative_to(root).as_posix()
        except ValueError:
            continue
        meta = _frontmatter(skill_md)
        name = str(meta.get("name") or skill_dir.name).strip()
        if not name:
            continue
        supporting_hashes: dict[str, str] = {}
        for path in sorted(item for item in skill_dir.rglob("*") if item.is_file()):
            if path.name == "SKILL.md":
                continue
            supporting_hashes[path.relative_to(skill_dir).as_posix()] = _sha256(path)
        entries.append(
            BundledSkillManifestEntry(
                name=name,
                skill_id=relative,
                source_ref=f"hermes-agent/skills/{relative}",
                source_path=skill_dir.resolve(),
                skill_md_sha256=_sha256(skill_md),
                supporting_file_hashes=supporting_hashes,
                workspace_id=context["workspace_id"],
                session_id=context.get("session_id"),
                generated_at_utc=generated_at,
                audit_metadata=context,
            )
        )
    return BundledSkillManifest(
        entries=tuple(entries),
        context=context,
        generated_at_utc=generated_at,
    )


def _assert_entry_current(entry: BundledSkillManifestEntry) -> None:
    skill_md = entry.source_path / "SKILL.md"
    if _sha256(skill_md) != entry.skill_md_sha256:
        raise SkillManifestChanged(
            f"BUNDLED_SKILL_MANIFEST_CHANGED: {entry.source_ref}/SKILL.md changed after inventory"
        )
    for relative, expected_hash in entry.supporting_file_hashes.items():
        path = entry.source_path / relative
        if not path.exists() or _sha256(path) != expected_hash:
            raise SkillManifestChanged(
                f"BUNDLED_SKILL_MANIFEST_CHANGED: {entry.source_ref}/{relative} changed after inventory"
            )


def _assert_entry_matches_boundary(
    entry: BundledSkillManifestEntry,
    boundary: ExecutionBoundary,
) -> None:
    if entry.workspace_id != boundary.workspace_id or entry.session_id != boundary.session_id:
        raise BoundaryPathRejected(
            "BUNDLED_SKILL_MANIFEST_CONTEXT_MISMATCH: manifest entry cannot be reused across boundaries"
        )


def load_bundled_skill_content(
    name_or_id: str,
    *,
    manifest_entry: BundledSkillManifestEntry,
    boundary: ExecutionBoundary | None,
) -> str:
    if boundary is None:
        raise GovernedExecutionBoundaryRequired(
            "SEMANTIER_EXECUTION_BOUNDARY_REQUIRED: bundled skill invocation requires boundary"
        )
    if manifest_entry.name != name_or_id and manifest_entry.skill_id != name_or_id:
        raise KeyError(name_or_id)
    _assert_entry_matches_boundary(manifest_entry, boundary)
    _assert_entry_current(manifest_entry)
    guard = DiskIOBoundaryGuard(boundary)
    skill_path = guard.resolve_read_path(
        manifest_entry.source_path / "SKILL.md",
        purpose="skill_asset",
        manifest_allowed_files=manifest_entry.allowed_files,
    )
    return skill_path.read_text(encoding="utf-8")


def resolve_bundled_skill_read_path(
    requested_path: str | Path,
    *,
    manifest_entry: BundledSkillManifestEntry | None,
    boundary: ExecutionBoundary | None,
    purpose: str = "read",
) -> Path:
    guard = DiskIOBoundaryGuard(boundary)
    allowed = manifest_entry.allowed_files if manifest_entry is not None else None
    return guard.resolve_read_path(
        requested_path,
        purpose=purpose,
        manifest_allowed_files=allowed,
    )


def resolve_bundled_skill_write_path(
    requested_path: str | Path,
    *,
    boundary: ExecutionBoundary | None,
    purpose: str = "tool_output",
) -> Path:
    return DiskIOBoundaryGuard(boundary).resolve_write_path(requested_path, purpose=purpose)


def import_document_for_bundled_skill(
    requested_path: str | Path,
    *,
    boundary: ExecutionBoundary | None,
    allowlisted_document_cache_roots: Iterable[Path] = (),
) -> Path:
    guard = DiskIOBoundaryGuard(boundary)
    source = Path(requested_path).expanduser().resolve()
    uploads_root = boundary.paths.uploads_root if boundary is not None else None
    if uploads_root is None:
        raise GovernedExecutionBoundaryRequired(
            "SEMANTIER_UPLOADS_ROOT_REQUIRED: bundled document import requires uploads_root"
        )

    for cache_root in allowlisted_document_cache_roots:
        resolved_root = Path(cache_root).expanduser().resolve()
        try:
            rel = source.relative_to(resolved_root)
        except ValueError:
            continue
        target = guard.resolve_write_path(uploads_root / rel, purpose="document_import")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        return target

    return guard.resolve_read_path(source, purpose="document_input")


def execute_bundled_skill_script(*_args: Any, **_kwargs: Any) -> None:
    raise BundledSkillScriptExecutionBlocked(
        "BUNDLED_SKILL_SCRIPT_EXECUTION_BLOCKED: raw Hermes bundled skill scripts are disabled "
        "for authenticated Semantier sessions unless routed through boundary-controlled tools"
    )
