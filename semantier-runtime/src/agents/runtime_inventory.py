from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml

from agents.hermes_bundled_skill_adapter import (
    GovernedDiagnosticInspectionContext,
    forbidden_global_external_skill_roots,
    list_bundled_skill_inventory,
)
from runtime_paths import platform_runtime_root

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|$)", re.DOTALL)
_SOURCE_META = {
    "workspace": {
        "sourceTier": "workspace",
        "sourceLabel": "User",
    },
    "application": {
        "sourceTier": "application",
        "sourceLabel": "Shared",
    },
}
_INSTALL_LOCK_RELATIVE_PATH = Path(".semantier") / "install-lock.json"
_REPO_APPLICATION_SKILLS_ROOT = Path(__file__).resolve().parents[1] / "skills"
_RESERVED_APPLICATION_SKILL_IDS = frozenset(
    {
        "governance/architecture-evolution-monitor",
    }
)
_RESERVED_APPLICATION_SKILL_NAMES = frozenset(
    {
        "architecture-evolution-monitor",
        "architecture evolution monitor",
    }
)


def _literal_string_assignments(path: Path) -> dict[str, str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError):
        return {}

    values: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            if not isinstance(node.value.value, str):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    values[target.id] = node.value.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                values[node.target.id] = node.value.value
    return values


def _plugin_constant_resolver(plugin_dir: Path, tree: ast.Module) -> dict[str, str]:
    module_aliases: dict[str, str] = {}
    constants: dict[str, str] = {}

    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
            module_name = node.module.split(".", 1)[0]
            module_constants = _literal_string_assignments(plugin_dir / f"{module_name}.py")
            for alias in node.names:
                if alias.name == "*":
                    continue
                local_name = alias.asname or alias.name
                if alias.name == module_name:
                    module_aliases[local_name] = module_name
                elif alias.name in module_constants:
                    constants[local_name] = module_constants[alias.name]
        elif isinstance(node, ast.ImportFrom) and node.level == 1 and node.module is None:
            for alias in node.names:
                module_aliases[alias.asname or alias.name] = alias.name.split(".", 1)[0]

    for local_name, module_name in module_aliases.items():
        module_constants = _literal_string_assignments(plugin_dir / f"{module_name}.py")
        for constant_name, value in module_constants.items():
            constants[f"{local_name}.{constant_name}"] = value

    constants.update(_literal_string_assignments(plugin_dir / "__init__.py"))
    return constants


def _resolve_string_expr(node: ast.AST | None, constants: dict[str, str]) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value.strip()
    if isinstance(node, ast.Name):
        return constants.get(node.id, "").strip()
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return constants.get(f"{node.value.id}.{node.attr}", "").strip()
    return ""


def _is_register_tool_call(node: ast.Call) -> bool:
    func = node.func
    return isinstance(func, ast.Attribute) and func.attr == "register_tool"

_TOOLSET_LABELS = {
    "hermes-cli": "Hermes CLI",
    "hermes-weixin": "Hermes Weixin",
    "hermes-feishu": "Hermes Feishu",
    "hermes-api-server": "Hermes API Server",
    "meeting-coordinator": "Meeting Coordinator",
}

_TOOLSET_DESCRIPTIONS = {
    "hermes-cli": "CLI runtime toolset for local terminal sessions.",
    "hermes-weixin": "Weixin gateway runtime toolset.",
    "hermes-feishu": "Feishu gateway runtime toolset.",
    "hermes-api-server": "Embedded API server runtime toolset.",
    "meeting-coordinator": "Feishu meeting RSVP monitoring and follow-up orchestration.",
}


def _normalized_platform_compatibility(value: Any) -> list[str]:
    return sorted(dict.fromkeys(_normalize_string_list(value)))


def _configured_toolset_platforms(config: dict[str, Any]) -> dict[str, list[str]]:
    platform_toolsets = config.get("platform_toolsets")
    if not isinstance(platform_toolsets, dict):
        return {}

    result: dict[str, list[str]] = {}
    for platform, raw_toolsets in platform_toolsets.items():
        platform_name = _normalize_text(platform)
        if not platform_name:
            continue
        for toolset_name in _normalize_string_list(raw_toolsets):
            platforms = result.setdefault(toolset_name, [])
            if platform_name not in platforms:
                platforms.append(platform_name)
    return {
        toolset_name: sorted(platforms)
        for toolset_name, platforms in result.items()
    }


def _plugin_configured_platforms(
    *,
    plugin_name: str,
    toolsets: dict[str, list[str]],
    toolset_platforms: dict[str, list[str]],
) -> list[str]:
    platforms: set[str] = set(toolset_platforms.get(plugin_name, []))
    for toolset_name in toolsets:
        platforms.update(toolset_platforms.get(toolset_name, []))
    return sorted(platforms)


def _skill_platform_compatibility(
    *,
    skill_meta: dict[str, Any],
    plugin_meta: dict[str, Any],
) -> list[str]:
    direct = _normalized_platform_compatibility(
        skill_meta.get("platform_compatibility")
        or skill_meta.get("platformCompatibility")
    )
    if direct:
        return direct

    hermes_meta = skill_meta.get("metadata")
    if isinstance(hermes_meta, dict):
        hermes_section = hermes_meta.get("hermes")
        if isinstance(hermes_section, dict):
            inherited = _normalized_platform_compatibility(
                hermes_section.get("platform_compatibility")
                or hermes_section.get("platformCompatibility")
            )
            if inherited:
                return inherited

    return _normalized_platform_compatibility(
        plugin_meta.get("platform_compatibility")
        or plugin_meta.get("platformCompatibility")
    )


def _plugin_enabled_names(config: dict[str, Any]) -> set[str]:
    plugins = config.get("plugins")
    if not isinstance(plugins, dict):
        return set()
    return {
        name
        for name in _normalize_string_list(plugins.get("enabled"))
        if name
    }


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [_normalize_text(item) for item in value if _normalize_text(item)]
    single = _normalize_text(value)
    return [single] if single else []


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "item"


def _normalized_skill_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "-", _normalize_text(value).lower()).strip("-")


def _is_reserved_application_skill_record(record: dict[str, Any]) -> bool:
    record_id = _normalize_text(record.get("id"))
    name = _normalize_text(record.get("name"))
    return (
        record_id in _RESERVED_APPLICATION_SKILL_IDS
        or _normalized_skill_key(record_id) in _RESERVED_APPLICATION_SKILL_NAMES
        or _normalized_skill_key(name) in _RESERVED_APPLICATION_SKILL_NAMES
    )


def _read_frontmatter(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return {}, ""

    match = _FRONTMATTER_RE.match(raw)
    if not match:
        return {}, raw

    try:
        parsed = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        parsed = {}
    return parsed if isinstance(parsed, dict) else {}, raw


def _read_yaml_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _disabled_skill_names(runtime_root: Path) -> set[str]:
    config = _read_yaml_file(runtime_root / "config.yaml")
    skills_config = config.get("skills")
    if not isinstance(skills_config, dict):
        return set()
    disabled = skills_config.get("disabled")
    if not isinstance(disabled, list):
        return set()
    return {str(item).strip() for item in disabled if str(item).strip()}


def _count_files(root: Path) -> int:
    return sum(1 for entry in root.rglob("*") if entry.is_file())


def _read_hub_lock(skills_root: Path) -> dict[str, dict[str, Any]]:
    lock_path = skills_root / ".hub" / "lock.json"
    try:
        raw = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    installed = raw.get("installed")
    if not isinstance(installed, dict):
        return {}
    return {
        _normalize_text(name): entry
        for name, entry in installed.items()
        if _normalize_text(name) and isinstance(entry, dict)
    }


def _read_install_lock(runtime_root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    lock_path = runtime_root / _INSTALL_LOCK_RELATIVE_PATH
    try:
        raw = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"plugins": {}, "skills": {}, "tools": {}}
    result: dict[str, dict[str, dict[str, Any]]] = {
        "plugins": {},
        "skills": {},
        "tools": {},
    }
    if not isinstance(raw, dict):
        return result
    for key in result:
        section = raw.get(key)
        if not isinstance(section, dict):
            continue
        result[key] = {
            _normalize_text(name): entry
            for name, entry in section.items()
            if _normalize_text(name) and isinstance(entry, dict)
        }
    return result


def _repo_marketplace_root() -> Path:
    return Path(__file__).resolve().parents[2] / "semantier-skills"


def _repo_core_plugins_root() -> Path:
    return Path(__file__).resolve().parents[1] / "plugins"


def _marketplace_entries_by_path() -> dict[str, dict[str, Any]]:
    index_path = _repo_marketplace_root() / "marketplace" / "index.json"
    try:
        raw = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    entries = raw.get("skills") if isinstance(raw, dict) else None
    if not isinstance(entries, list):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        path = _normalize_text(entry.get("path")).strip("/")
        if path:
            result[path] = entry
    return result


def _directory_content_hash(root: Path) -> str | None:
    if not root.exists() or not root.is_dir():
        return None
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if "__pycache__" in path.parts:
            continue
        try:
            relative = path.relative_to(root).as_posix()
            data = path.read_bytes()
        except OSError:
            continue
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(data)
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def _version_text(value: Any) -> str | None:
    text = _normalize_text(value)
    return text or None


def _package_update_metadata(
    *,
    installed_dir: Path,
    installed_meta: dict[str, Any],
    install_entry: dict[str, Any] | None,
    latest_dir: Path | None,
    latest_marketplace_entry: dict[str, Any] | None,
    source_kind: str,
) -> dict[str, Any]:
    installed_version = (
        _version_text(installed_meta.get("version"))
        or _version_text((install_entry or {}).get("version"))
    )
    latest_meta = _read_yaml_file(latest_dir / "plugin.yaml") if latest_dir else {}
    latest_version = (
        _version_text(latest_meta.get("version"))
        or _version_text((latest_marketplace_entry or {}).get("version"))
    )
    installed_hash = (
        _version_text((install_entry or {}).get("content_hash"))
        or _directory_content_hash(installed_dir)
    )
    latest_hash = _directory_content_hash(latest_dir) if latest_dir else None
    can_update = bool(source_kind == "workspace" and latest_dir and latest_dir.exists())
    update_status = None
    if can_update:
        update_status = (
            "update_available"
            if (
                (latest_version and installed_version and latest_version != installed_version)
                or (latest_hash and installed_hash and latest_hash != installed_hash)
            )
            else "up_to_date"
        )
    elif source_kind == "workspace":
        update_status = "unavailable"

    return {
        "version": installed_version,
        "installedVersion": installed_version,
        "latestVersion": latest_version,
        "contentHash": installed_hash,
        "latestHash": latest_hash,
        "updateStatus": update_status,
        "hubIdentifier": _version_text((install_entry or {}).get("identifier"))
        or _version_text((latest_marketplace_entry or {}).get("id")),
        "hubSource": _version_text((install_entry or {}).get("source"))
        or _version_text((latest_marketplace_entry or {}).get("source")),
        "hubUpdatedAt": _version_text((install_entry or {}).get("updated_at")),
        "packageType": _version_text((install_entry or {}).get("type"))
        or _version_text((latest_marketplace_entry or {}).get("type")),
        "packagePath": _version_text((install_entry or {}).get("path"))
        or _version_text((latest_marketplace_entry or {}).get("path")),
        "canUpdate": can_update,
    }


def _external_skill_roots(runtime_root: Path) -> list[Path]:
    config = _read_yaml_file(runtime_root / "config.yaml")
    skills_cfg = config.get("skills")
    if not isinstance(skills_cfg, dict):
        return []
    external_dirs = skills_cfg.get("external_dirs")
    if isinstance(external_dirs, str):
        external_dirs = [external_dirs]
    if not isinstance(external_dirs, list):
        return []

    roots: list[Path] = []
    seen: set[Path] = set()
    forbidden_roots = forbidden_global_external_skill_roots(runtime_root)
    local_skills_root = (runtime_root / "skills").resolve()
    shared_skills_root = (platform_runtime_root() / "skills").resolve()
    for item in external_dirs:
        text = _normalize_text(item)
        if not text:
            continue
        path = Path(text).expanduser()
        if not path.is_absolute():
            path = runtime_root / path
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if (
            resolved == local_skills_root
            or resolved == shared_skills_root
            or resolved in forbidden_roots
            or resolved in seen
            or not resolved.exists()
        ):
            continue
        seen.add(resolved)
        roots.append(resolved)
    return roots


def _is_plugin_external_skill_root(skills_root: Path, runtime_root: Path) -> bool:
    try:
        skills_root.resolve().relative_to((runtime_root / "plugins").resolve())
    except ValueError:
        return False
    return True


def _skill_dirs(skills_root: Path) -> list[Path]:
    if not skills_root.exists():
        return []
    return sorted({path.parent for path in skills_root.rglob("SKILL.md")})


def _skill_md_sha256(skill_dir: Path) -> str | None:
    try:
        digest = hashlib.sha256((skill_dir / "SKILL.md").read_bytes()).hexdigest()
    except OSError:
        return None
    return f"sha256:{digest}"


def _is_selectable_skill(skill_dir: Path, skills_root: Path) -> bool:
    try:
        if skill_dir.relative_to(skills_root).parts[:1] == ("retired",):
            return False
    except ValueError:
        return False
    metadata, _ = _read_frontmatter(skill_dir / "SKILL.md")
    hermes = metadata.get("metadata")
    if isinstance(hermes, dict):
        hermes = hermes.get("hermes")
    return not (isinstance(hermes, dict) and hermes.get("selectable") is False)


def _category_from_skill_meta(metadata: dict[str, Any], skill_dir: Path, skills_root: Path) -> str:
    direct = _normalize_text(metadata.get("category"))
    if direct:
        return direct
    hermes_meta = metadata.get("metadata")
    if isinstance(hermes_meta, dict):
        hermes_section = hermes_meta.get("hermes")
        if isinstance(hermes_section, dict):
            category = _normalize_text(hermes_section.get("category"))
            if category:
                return category

    try:
        relative = skill_dir.relative_to(skills_root)
    except ValueError:
        return "general"
    parts = relative.parts
    if len(parts) >= 2:
        return parts[0]
    return "general"


def _plugin_skill_root_manifest(skills_root: Path) -> dict[str, Any]:
    manifest = skills_root / "plugin.yaml"
    if not manifest.exists():
        return {}
    return _read_yaml_file(manifest)


def _is_repo_owned_plugin_package(skills_root: Path, plugin_meta: dict[str, Any]) -> bool:
    plugin_name = _normalize_text(plugin_meta.get("name")) or skills_root.name
    if not plugin_name:
        return False
    candidate = _repo_core_plugins_root() / plugin_name
    return candidate.is_dir() and (candidate / "plugin.yaml").is_file()


def _build_skill_record(
    skill_dir: Path,
    *,
    skills_root: Path,
    source_kind: str,
    hub_entry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta, raw_content = _read_frontmatter(skill_dir / "SKILL.md")
    plugin_meta = _plugin_skill_root_manifest(skills_root)
    relative = skill_dir.relative_to(skills_root)
    rel_path = skill_dir.name if relative.as_posix() == "." else relative.as_posix()
    name = _normalize_text(meta.get("name")) or Path(rel_path).name
    hermes_meta = meta.get("metadata")
    hermes_section = {}
    if isinstance(hermes_meta, dict) and isinstance(hermes_meta.get("hermes"), dict):
        hermes_section = hermes_meta["hermes"]

    tags = _normalize_string_list(meta.get("tags"))
    if not tags and isinstance(hermes_section, dict):
        tags = _normalize_string_list(hermes_section.get("tags"))

    manifest_source = _normalize_text(plugin_meta.get("source")).lower()
    manifest_read_only = plugin_meta.get("read_only") is True
    effective_source_kind = (
        "application"
        if (
            source_kind == "workspace"
            and (manifest_source == "shared" or manifest_read_only)
            and _is_repo_owned_plugin_package(skills_root, plugin_meta)
        )
        else source_kind
    )
    source_meta = _SOURCE_META[effective_source_kind]
    is_workspace_owned = effective_source_kind == "workspace"
    category = _category_from_skill_meta(meta, skill_dir, skills_root)
    if category == "general":
        category = _normalize_text(plugin_meta.get("category")) or category
    platform_compatibility = _skill_platform_compatibility(
        skill_meta=meta,
        plugin_meta=plugin_meta,
    )
    package_meta: dict[str, Any] = {}
    if (
        plugin_meta
        and source_kind == "workspace"
        and skills_root.parent.name == "plugins"
    ):
        plugin_name = skills_root.name
        runtime_root = skills_root.parent.parent
        install_lock = _read_install_lock(runtime_root)
        install_entry = install_lock.get("plugins", {}).get(plugin_name)
        package_path = _normalize_text((install_entry or {}).get("path")) or f"plugins/{plugin_name}"
        marketplace_entry = _marketplace_entries_by_path().get(package_path)
        latest_dir = (_repo_marketplace_root() / package_path).resolve()
        if not latest_dir.exists():
            latest_dir = None
        package_meta = _package_update_metadata(
            installed_dir=skills_root,
            installed_meta=plugin_meta,
            install_entry=install_entry,
            latest_dir=latest_dir,
            latest_marketplace_entry=marketplace_entry,
            source_kind=source_kind,
        )
    return {
        "id": rel_path,
        "slug": _slugify(rel_path),
        "name": name,
        "description": _normalize_text(meta.get("description")),
        "author": _normalize_text(meta.get("author")) or "Semantier Runtime",
        "triggers": _normalize_string_list(meta.get("triggers")),
        "tags": tags,
        "homepage": None,
        "category": category,
        "icon": "✨" if effective_source_kind == "workspace" else "🧩",
        "content": raw_content,
        "fileCount": _count_files(skill_dir),
        "sourcePath": str(skill_dir.resolve()),
        "installed": True,
        "enabled": True,
        "builtin": False,
        "sourceTier": source_meta["sourceTier"],
        "sourceLabel": source_meta["sourceLabel"],
        "canEdit": is_workspace_owned,
        "canUninstall": is_workspace_owned,
        "canModify": is_workspace_owned,
        "platformCompatibility": platform_compatibility,
        "version": package_meta.get("version") or (
            _version_text(meta.get("version"))
            or _version_text(plugin_meta.get("version"))
            or _version_text((hub_entry or {}).get("version"))
        ),
        "installedVersion": package_meta.get("installedVersion") or (
            _version_text(meta.get("version"))
            or _version_text(plugin_meta.get("version"))
            or _version_text((hub_entry or {}).get("version"))
        ),
        "latestVersion": package_meta.get("latestVersion")
        or _version_text((hub_entry or {}).get("latest_version")),
        "contentHash": package_meta.get("contentHash")
        or _normalize_text((hub_entry or {}).get("content_hash"))
        or (
            _skill_md_sha256(skill_dir)
            if effective_source_kind == "application"
            and isinstance(hermes_section, dict)
            and hermes_section.get("immutable") is True
            else None
        )
        or None,
        "latestHash": package_meta.get("latestHash"),
        "updateStatus": package_meta.get("updateStatus"),
        "hubIdentifier": package_meta.get("hubIdentifier")
        or _normalize_text((hub_entry or {}).get("identifier"))
        or None,
        "hubSource": package_meta.get("hubSource")
        or _normalize_text((hub_entry or {}).get("source"))
        or None,
        "hubUpdatedAt": package_meta.get("hubUpdatedAt")
        or _normalize_text((hub_entry or {}).get("updated_at"))
        or None,
        "packageType": package_meta.get("packageType"),
        "packagePath": package_meta.get("packagePath"),
        "canUpdate": bool(package_meta.get("canUpdate") or (hub_entry or {}).get("identifier")),
        "security": {
            "level": "safe",
            "flags": [],
            "score": 0,
        },
    }


def list_skills_inventory(
    *,
    workspace_hermes_home: Path | None,
    governed_diagnostic_context: GovernedDiagnosticInspectionContext | None = None,
    execution_boundary: Any | None = None,
) -> dict[str, Any]:
    shared_root = platform_runtime_root()
    seen: dict[str, dict[str, Any]] = {}
    managed_package_availability = None

    shared_skills_root = shared_root / "skills"
    shared_disabled = _disabled_skill_names(shared_root)
    shared_hub_lock = _read_hub_lock(shared_skills_root)
    shared_local_skill_names: set[str] = set()
    for skills_root in [shared_skills_root, _REPO_APPLICATION_SKILLS_ROOT, *_external_skill_roots(shared_root)]:
        if skills_root == _REPO_APPLICATION_SKILLS_ROOT and not skills_root.exists():
            continue
        for skill_dir in _skill_dirs(skills_root):
            if not _is_selectable_skill(skill_dir, skills_root):
                continue
            meta, _ = _read_frontmatter(skill_dir / "SKILL.md")
            skill_name = _normalize_text(meta.get("name")) or skill_dir.name
            if skills_root == shared_skills_root:
                shared_local_skill_names.add(skill_name)
            elif (
                skill_name in shared_local_skill_names
                and _is_plugin_external_skill_root(skills_root, shared_root)
            ):
                continue
            record = _build_skill_record(
                skill_dir,
                skills_root=skills_root,
                source_kind="application",
                hub_entry=shared_hub_lock.get(skill_name),
            )
            if record.get("canModify") is True:
                record["enabled"] = (
                    str(record.get("id") or "") not in shared_disabled
                    and str(record.get("name") or "") not in shared_disabled
                )
            if _is_reserved_application_skill_record(record):
                record["builtin"] = True
                record["canEdit"] = False
                record["canUninstall"] = False
                record["canModify"] = False
            seen[record["id"]] = record

    if workspace_hermes_home is not None:
        workspace_skills_root = workspace_hermes_home / "skills"
        workspace_disabled = _disabled_skill_names(workspace_hermes_home)
        workspace_hub_lock = _read_hub_lock(workspace_skills_root)
        workspace_local_skill_names: set[str] = set()
        for skills_root in [workspace_skills_root, *_external_skill_roots(workspace_hermes_home)]:
            for skill_dir in _skill_dirs(skills_root):
                if not _is_selectable_skill(skill_dir, skills_root):
                    continue
                meta, _ = _read_frontmatter(skill_dir / "SKILL.md")
                skill_name = _normalize_text(meta.get("name")) or skill_dir.name
                if skills_root == workspace_skills_root:
                    workspace_local_skill_names.add(skill_name)
                elif (
                    skill_name in workspace_local_skill_names
                    and _is_plugin_external_skill_root(skills_root, workspace_hermes_home)
                ):
                    continue
                record = _build_skill_record(
                    skill_dir,
                    skills_root=skills_root,
                    source_kind="workspace",
                    hub_entry=workspace_hub_lock.get(skill_name),
                )
                if _is_reserved_application_skill_record(record):
                    continue
                if record.get("canModify") is True:
                    record["enabled"] = (
                        str(record.get("id") or "") not in workspace_disabled
                        and str(record.get("name") or "") not in workspace_disabled
                    )
                seen[record["id"]] = record
    else:
        workspace_disabled = set()

    # Read the exact authority-backed index used by skill_view. Disk state may
    # deny invokability, but never grants it when verification is unavailable.
    if workspace_hermes_home is not None:
        from hermes_constants import set_hermes_home_override, reset_hermes_home_override
        from tools.skill_review_capture import generation, managed_skill_index, managed_home, pin_generation
        token = set_hermes_home_override(workspace_hermes_home)
        try:
            packages = {}
            try:
                packages = generation()
                with pin_generation():
                    packages = generation()
                    managed_index = managed_skill_index()
                managed_error = None
            except (ValueError, PermissionError, OSError, KeyError, TypeError):
                managed_index = {}
                managed_error = "managed_package_verification_unavailable"
            if managed_home():
                managed_package_availability = "unavailable" if managed_error else "available"
            controlled = set(packages)
            if managed_error and managed_home():
                controlled.update(str(row["name"]) for row in seen.values() if row.get("sourceTier") == "workspace")
            seen = {key: row for key, row in seen.items() if row.get("name") not in controlled and key not in controlled}
            for name in sorted(controlled):
                directory = managed_index.get(name)
                if managed_error is None and directory is None:
                    continue
                if directory is not None:
                    record = _build_skill_record(directory, skills_root=directory.parent, source_kind="workspace")
                    record.update(contentHash=packages[name]["version_hash"], runtimeAvailability="available", enabled=True, invokable=True)
                else:
                    record = {"description": "Managed package verification unavailable", "content": "", "category": "general", "sourceTier": "workspace", "sourceLabel": "Managed · unavailable", "enabled": False, "invokable": False, "runtimeAvailability": "unavailable", "nonInvokableReason": managed_error}
                record.update(id=name, name=name, slug=_slugify(name), canEdit=False, canUninstall=False, canModify=False, canUpdate=False)
                seen[name] = record
        finally:
            reset_hermes_home_override(token)

    # Bundled skills are a governed diagnostic surface. A normal inventory
    # request must not expose them merely because the launcher enabled the
    # platform-wide feature flag; the route supplies a valid diagnostic
    # context only for an authorized inspection.
    bundled_manifest = (
        list_bundled_skill_inventory(
            auth_context=governed_diagnostic_context,
            boundary=execution_boundary,
        )
        if governed_diagnostic_context is not None
        else list_bundled_skill_inventory()
    )
    for entry in bundled_manifest.entries:
        seen[entry.skill_id] = entry.to_record(
            disabled_skill_names=workspace_disabled,
        )

    skills = sorted(seen.values(), key=lambda item: (item["sourceLabel"], item["name"].lower()))
    categories = sorted(
        {"All", *[_normalize_text(skill.get("category")) or "general" for skill in skills]},
        key=lambda value: value.lower(),
    )

    workspace_slug = workspace_hermes_home.parent.name if workspace_hermes_home is not None else None
    return {
        "skills": skills,
        "managedPackageAvailability": managed_package_availability,
        "total": len(skills),
        "page": 1,
        "categories": categories,
        "workspaceSlug": workspace_slug,
        "bundledSkillManifest": {
            "generatedAtUtc": bundled_manifest.generated_at_utc,
            "context": dict(bundled_manifest.context),
            "entries": [
                {
                    "name": entry.name,
                    "skillId": entry.skill_id,
                    "sourceRef": entry.source_ref,
                    "skillMdSha256": entry.skill_md_sha256,
                    "supportingFileHashes": dict(entry.supporting_file_hashes),
                    "workspaceId": entry.workspace_id,
                    "sessionId": entry.session_id,
                    "generatedAtUtc": entry.generated_at_utc,
                    "auditMetadata": dict(entry.audit_metadata),
                }
                for entry in bundled_manifest.entries
            ],
        },
    }


def _tool_names_from_plugin_source(plugin_dir: Path) -> dict[str, list[str]]:
    toolset_tools: dict[str, list[str]] = {}
    init_path = plugin_dir / "__init__.py"
    if not init_path.exists():
        return toolset_tools

    try:
        raw = init_path.read_text(encoding="utf-8")
    except OSError:
        return toolset_tools

    try:
        tree = ast.parse(raw, filename=str(init_path))
    except SyntaxError:
        return toolset_tools

    constants = _plugin_constant_resolver(plugin_dir, tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not _is_register_tool_call(node):
            continue
        keywords = {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg}
        tool_name = _resolve_string_expr(keywords.get("name"), constants)
        toolset_name = _resolve_string_expr(keywords.get("toolset"), constants)
        if not tool_name or not toolset_name:
            continue
        tools = toolset_tools.setdefault(toolset_name, [])
        if tool_name not in tools:
            tools.append(tool_name)
    return toolset_tools


def _plugin_inventory(plugins_root: Path) -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    if not plugins_root.exists():
        return inventory

    for plugin_dir in sorted(path for path in plugins_root.iterdir() if path.is_dir()):
        meta = _read_yaml_file(plugin_dir / "plugin.yaml")
        name = _normalize_text(meta.get("name")) or plugin_dir.name
        provides_hooks = _normalize_string_list(meta.get("provides_hooks"))
        tools_by_toolset = _tool_names_from_plugin_source(plugin_dir)
        record = {
            "id": plugin_dir.name,
            "sourcePath": str(plugin_dir.resolve()),
            "description": _normalize_text(meta.get("description")),
            "version": _version_text(meta.get("version")),
            "category": _normalize_text(meta.get("category")),
            "ui_extensions": meta.get("ui_extensions") if isinstance(meta.get("ui_extensions"), list) else [],
            "provides_hooks": provides_hooks,
            "toolsets": tools_by_toolset,
            "platform_compatibility": _normalized_platform_compatibility(
                meta.get("platform_compatibility")
                or meta.get("platformCompatibility")
            ),
        }
        inventory[plugin_dir.name] = record
        if name:
            inventory[name] = record
    return inventory


def _plugin_category(
    *,
    name: str,
    provides_hooks: list[str],
    toolsets: dict[str, list[str]],
) -> str:
    has_hooks = len(provides_hooks) > 0
    has_tools = any(len(tools) > 0 for tools in toolsets.values())
    if has_hooks and "guard" in name:
        return "Runtime Guard"
    if has_hooks and has_tools:
        return "Hybrid"
    if has_hooks:
        return "Hook Plugin"
    if has_tools:
        return "Tool Provider"
    return "Support Plugin"


def list_plugins_inventory(*, workspace_hermes_home: Path | None) -> dict[str, Any]:
    shared_root = platform_runtime_root()
    seen: dict[str, dict[str, Any]] = {}

    def collect(
        *,
        runtime_root: Path,
        source_kind: str,
    ) -> None:
        plugins_root = runtime_root / "plugins"
        if not plugins_root.exists():
            return
        config = _read_yaml_file(runtime_root / "config.yaml")
        inventory = _plugin_inventory(plugins_root)
        install_lock = _read_install_lock(runtime_root)
        marketplace_by_path = _marketplace_entries_by_path()
        enabled_names = _plugin_enabled_names(config)
        toolset_platforms = _configured_toolset_platforms(config)
        source_meta = _SOURCE_META[source_kind]

        for plugin_dir in sorted(path for path in plugins_root.iterdir() if path.is_dir()):
            plugin_meta = inventory.get(plugin_dir.name, {})
            name = _normalize_text(plugin_dir.name)
            toolsets = plugin_meta.get("toolsets")
            if not isinstance(toolsets, dict):
                toolsets = {}
            provides_hooks = plugin_meta.get("provides_hooks")
            if not isinstance(provides_hooks, list):
                provides_hooks = []
            ui_extensions = plugin_meta.get("ui_extensions")
            if not isinstance(ui_extensions, list):
                ui_extensions = []
            flat_tools = sorted(
                {
                    tool
                    for tools in toolsets.values()
                    if isinstance(tools, list)
                    for tool in tools
                    if _normalize_text(tool)
                }
            )
            package_path = _normalize_text(
                (install_lock.get("plugins", {}).get(name) or {}).get("path")
            ) or f"plugins/{name}"
            latest_marketplace_entry = marketplace_by_path.get(package_path)
            latest_dir = (_repo_marketplace_root() / package_path).resolve()
            if not latest_dir.exists():
                latest_dir = None
            package_meta = _package_update_metadata(
                installed_dir=plugin_dir,
                installed_meta=plugin_meta,
                install_entry=install_lock.get("plugins", {}).get(name),
                latest_dir=latest_dir,
                latest_marketplace_entry=latest_marketplace_entry,
                source_kind=source_kind,
            )

            record = {
                "id": name,
                "name": name,
                "label": _resolve_toolset_label(name),
                "description": _normalize_text(plugin_meta.get("description")),
                "sourceTier": source_meta["sourceTier"],
                "sourceLabel": source_meta["sourceLabel"],
                "enabled": name in enabled_names,
                "category": _normalize_text(plugin_meta.get("category")) or _plugin_category(
                    name=name,
                    provides_hooks=provides_hooks,
                    toolsets=toolsets,
                ),
                "toolsets": sorted(toolsets.keys()),
                "tools": flat_tools,
                "uiExtensions": ui_extensions,
                "providesHooks": provides_hooks,
                "platformCompatibility": (
                    _normalized_platform_compatibility(
                        plugin_meta.get("platform_compatibility")
                    )
                    or _plugin_configured_platforms(
                        plugin_name=name,
                        toolsets=toolsets,
                        toolset_platforms=toolset_platforms,
                    )
                ),
                "sourcePath": str(plugin_dir.resolve()),
                "canModify": source_kind == "workspace",
                **package_meta,
            }
            if source_kind == "workspace" and name in seen:
                existing = seen[name]
                if existing.get("sourceTier") == "application":
                    continue
            seen[name] = record

    collect(runtime_root=shared_root, source_kind="application")
    if workspace_hermes_home is not None:
        collect(runtime_root=workspace_hermes_home, source_kind="workspace")

    items = sorted(seen.values(), key=lambda item: item["label"].lower())
    categories = sorted({"All", *[item["category"] for item in items]})
    return {
        "plugins": items,
        "total": len(items),
        "categories": categories,
    }


def _resolve_toolset_label(name: str) -> str:
    if name in _TOOLSET_LABELS:
        return _TOOLSET_LABELS[name]
    return " ".join(part.capitalize() for part in name.replace("_", "-").split("-") if part)


def _resolve_toolset_description(
    name: str,
    plugin_meta: dict[str, Any] | None,
    *,
    platforms: list[str],
) -> str:
    if plugin_meta:
        description = _normalize_text(plugin_meta.get("description"))
        if description:
            return description
        provides_hooks = plugin_meta.get("provides_hooks")
        if isinstance(provides_hooks, list) and provides_hooks:
            return f"Runtime hook surface providing: {', '.join(str(item) for item in provides_hooks)}."
    if name in _TOOLSET_DESCRIPTIONS:
        return _TOOLSET_DESCRIPTIONS[name]
    if platforms:
        return f"Configured for: {', '.join(platforms)}."
    return "Configured runtime toolset."


def _toolset_tools(name: str, plugin_meta: dict[str, Any] | None) -> list[str]:
    if plugin_meta and isinstance(plugin_meta.get("toolsets"), dict):
        raw = plugin_meta["toolsets"].get(name)
        if isinstance(raw, list):
            return [tool for tool in raw if _normalize_text(tool)]
    return []


def _plugin_meta_for_toolset(
    plugin_inventory: dict[str, dict[str, Any]],
    toolset_name: str,
) -> dict[str, Any] | None:
    plugin_meta = plugin_inventory.get(toolset_name)
    if plugin_meta is not None:
        return plugin_meta

    for candidate in plugin_inventory.values():
        toolsets = candidate.get("toolsets")
        if isinstance(toolsets, dict) and toolset_name in toolsets:
            return candidate
    return None


def _append_toolsets_from_config(
    *,
    result: dict[str, dict[str, Any]],
    config: dict[str, Any],
    source_kind: str,
    plugins_root: Path,
) -> None:
    platform_toolsets = config.get("platform_toolsets")
    if not isinstance(platform_toolsets, dict):
        return

    plugin_inventory = _plugin_inventory(plugins_root)
    runtime_root = plugins_root.parent
    install_lock = _read_install_lock(runtime_root)
    marketplace_by_path = _marketplace_entries_by_path()
    source_meta = _SOURCE_META[source_kind]

    for platform, raw_toolsets in platform_toolsets.items():
        toolsets = _normalize_string_list(raw_toolsets)
        for toolset_name in toolsets:
            record = result.get(toolset_name)
            if record is None:
                plugin_meta = _plugin_meta_for_toolset(plugin_inventory, toolset_name)
                provider_name = _normalize_text((plugin_meta or {}).get("id")) or toolset_name
                package_path = _normalize_text(
                    (install_lock.get("plugins", {}).get(provider_name) or {}).get("path")
                ) or f"plugins/{provider_name}"
                latest_dir = (_repo_marketplace_root() / package_path).resolve()
                if not latest_dir.exists():
                    latest_dir = None
                installed_dir_text = _normalize_text((plugin_meta or {}).get("sourcePath"))
                installed_dir = Path(installed_dir_text) if installed_dir_text else plugins_root / provider_name
                package_meta = _package_update_metadata(
                    installed_dir=installed_dir,
                    installed_meta=plugin_meta or {},
                    install_entry=install_lock.get("plugins", {}).get(provider_name),
                    latest_dir=latest_dir,
                    latest_marketplace_entry=marketplace_by_path.get(package_path),
                    source_kind=source_kind,
                )
                record = {
                    "id": toolset_name,
                    "name": toolset_name,
                    "label": _resolve_toolset_label(toolset_name),
                    "description": _resolve_toolset_description(
                        toolset_name,
                        plugin_meta,
                        platforms=[platform],
                    ),
                    "tools": _toolset_tools(toolset_name, plugin_meta),
                    "enabled": True,
                    "available": True,
                    "configured": True,
                    "sourceTier": source_meta["sourceTier"],
                    "sourceLabel": source_meta["sourceLabel"],
                    "builtin": False,
                    "canModify": source_kind == "workspace",
                    "platforms": [platform],
                    **package_meta,
                }
                result[toolset_name] = record
                continue

            if source_kind == "workspace" and record.get("sourceTier") != "application":
                record["sourceTier"] = source_meta["sourceTier"]
                record["sourceLabel"] = source_meta["sourceLabel"]
                record["canModify"] = True
            platforms = record.setdefault("platforms", [])
            if platform not in platforms:
                platforms.append(platform)
            record["configured"] = True
            record["enabled"] = True


def list_toolsets_inventory(*, workspace_hermes_home: Path | None) -> dict[str, Any]:
    toolsets: dict[str, dict[str, Any]] = {}
    shared_root = platform_runtime_root()

    _append_toolsets_from_config(
        result=toolsets,
        config=_read_yaml_file(shared_root / "config.yaml"),
        source_kind="application",
        plugins_root=shared_root / "plugins",
    )

    if workspace_hermes_home is not None:
        _append_toolsets_from_config(
            result=toolsets,
            config=_read_yaml_file(workspace_hermes_home / "config.yaml"),
            source_kind="workspace",
            plugins_root=workspace_hermes_home / "plugins",
        )

    items = []
    for record in toolsets.values():
        platforms = sorted(dict.fromkeys(_normalize_string_list(record.get("platforms"))))
        items.append(
            {
                **{
                    key: value
                    for key, value in record.items()
                    if key != "platforms"
                },
                "platformCompatibility": platforms,
            }
        )
    items.sort(key=lambda item: item["label"].lower())
    return {
        "tools": items,
        "total": len(items),
    }
