"""Resolve pinned Semantier plugin releases from the provider's public index."""
import json
import re
from urllib.request import urlopen


REPO = "chris-han/semantier-skills"
INDEX = f"https://raw.githubusercontent.com/{REPO}/main/marketplace/index.json"


def load_semantier_plugin_catalog():
    with urlopen(INDEX, timeout=10) as response:
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
