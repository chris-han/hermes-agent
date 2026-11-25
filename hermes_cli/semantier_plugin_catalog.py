"""Resolve pinned Semantier plugin releases from the provider's public index."""
import json
import re
from urllib.request import Request, urlopen


REPO = "chris-han/semantier-skills"
INDEX = f"https://api.github.com/repos/{REPO}/contents/marketplace/index.json?ref=main"


def load_semantier_plugin_catalog():
    with urlopen(Request(INDEX, headers={"Accept": "application/vnd.github.raw+json"}), timeout=10) as response:
        document = json.load(response)
    rows = {}
    for row in document["skills"]:
        if row.get("type") != "plugin" or "package_ref" not in row:
            continue
        name = row["name"]
        path = f"plugins/{name}"
        if (not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", name)
                or row["repo"] != f"https://github.com/{REPO}"
                or row["source"] != "custom-marketplace"
                or row["path"] != path or row["id"] != f"{REPO}/{path}"
                or not re.fullmatch(r"[a-f0-9]{40}", row["package_ref"])
                or not re.fullmatch(r"sha256:[a-f0-9]{64}", row["content_hash"])
                or not isinstance(row["version"], str) or not row["version"] or name in rows):
            raise ValueError("Invalid pinned Semantier plugin release")
        release = {key: row[key] for key in ("id", "source", "path", "version", "package_ref", "content_hash")}
        rows[name] = {"name": name, "repo": REPO, "sha": row["package_ref"], "subdir": path,
                      "provider": "semantier", "release": release, "description": row.get("description", "")}
    return rows


def resolve_selected_release(selection, identifier, ref):
    if not isinstance(selection, dict):
        raise ValueError("Invalid Semantier selection")
    rows = load_semantier_plugin_catalog()
    row = next((r for r in rows.values() if r["release"] == selection), None)
    if not row or identifier != f"{REPO}#{row['subdir']}" or ref != row["sha"]:
        raise ValueError("Selected Semantier release is unavailable")
    return row["release"]


def verify_release_tree(release, manifest, root):
    import hashlib
    if (manifest.get("name") != release["path"].split("/")[-1]
            or str(manifest.get("version")) != release["version"]):
        raise ValueError("Semantier manifest differs from the selected release")
    digest = hashlib.sha256()
    for file in sorted(root.rglob("*")):
        relative = file.relative_to(root)
        if file.is_symlink():
            raise ValueError("Semantier package contains a symbolic link")
        if file.is_file() and "__pycache__" not in relative.parts:
            digest.update(relative.as_posix().encode() + b"\0" + file.read_bytes() + b"\0")
    if "sha256:" + digest.hexdigest() != release["content_hash"]:
        raise ValueError("Semantier package differs from the selected content hash")


def project_installed_tutor_progress(release, project_ref, plan_ref, objectives):
    """Use installed Tutor to project grades supplied by the connected Workspace."""
    import inspect
    from pathlib import Path

    from hermes_cli.plugins import discover_plugins, get_plugin_manager
    from hermes_cli.plugins_cmd_git import _read_install_metadata
    from hermes_cli.plugins_cmd_install import _read_manifest_for_install
    from tools.registry import registry

    record = _read_install_metadata().get("tutor", {})
    if (not isinstance(release, dict) or release.get("id") != f"{REPO}/plugins/tutor"
            or record.get("semantier_release") != release or record.get("revision") != release.get("package_ref")
            or record.get("pinned") is not True):
        raise ValueError("Install the matching Semantier Tutor release in this profile")
    manager = get_plugin_manager()
    installed = manager.home_path / "plugins" / "tutor"
    verify_release_tree(release, _read_manifest_for_install(installed), installed)
    discover_plugins()
    entry = registry.snapshot_registration("tutor_progress", scope=manager.scope_key)
    if not entry or entry.toolset != "tutor" or not any(
            row["name"] == "tutor" and row["enabled"] and row["version"] == release["version"]
            for row in manager.list_plugins()):
        raise ValueError("Enable the installed Tutor toolset")
    origin = inspect.getsourcefile(entry.handler)
    if not origin or not Path(origin).resolve().is_relative_to(installed.resolve()):
        raise ValueError("Tutor handler origin mismatch")
    args = {"project_ref": project_ref, "plan_ref": plan_ref}

    def progress(name, arguments):
        if name != "tutor_progress" or arguments != args:
            raise ValueError("Invalid Tutor operation")
        return {"objectives": objectives}

    result = json.loads(registry.dispatch("tutor_progress", args, scope=manager.scope_key, tutor_core=progress))
    return {"mastery": [objective["mastery"] for objective in result["objectives"]], "release": release}
