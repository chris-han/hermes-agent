import hashlib
import io
import json

import pytest

from hermes_cli import semantier_plugin_catalog as catalog


def test_selected_release_survives_lookup_and_tree_check(monkeypatch, tmp_path):
    data = b"name: tutor\nversion: 0.1.0\n"
    row = dict(name="tutor", type="plugin", path="plugins/tutor", source="custom-marketplace",
               repo=f"https://github.com/{catalog.REPO}", id=f"{catalog.REPO}/plugins/tutor",
               version="0.1.0", package_ref="b" * 40,
               content_hash="sha256:" + hashlib.sha256(b"plugin.yaml\0" + data + b"\0").hexdigest())
    monkeypatch.setattr(catalog, "urlopen", lambda *a, **k: io.BytesIO(json.dumps({"skills": [row]}).encode()))
    entry = catalog.load_semantier_plugin_catalog()["tutor"]
    identifier = f"{catalog.REPO}#plugins/tutor"
    release = catalog.resolve_selected_release(entry["release"], identifier, entry["sha"])
    with pytest.raises(ValueError):
        catalog.resolve_selected_release(release, "other/repo#plugins/tutor", entry["sha"])
    manifest = {"name": "tutor", "version": "0.1.0"}
    tmp_path = tmp_path / "package"
    tmp_path.mkdir()
    (tmp_path / "plugin.yaml").write_bytes(data)
    catalog.verify_release_tree(release, manifest, tmp_path)
    (tmp_path / "plugin.yaml").write_bytes(data + b" ")
    with pytest.raises(ValueError):
        catalog.verify_release_tree(release, manifest, tmp_path)


def test_connected_progress_requires_exact_installed_provider(monkeypatch):
    from hermes_cli import plugins_cmd_git
    release = {"id": f"{catalog.REPO}/plugins/tutor", "package_ref": "a"*40}
    monkeypatch.setattr(plugins_cmd_git, "_read_install_metadata", lambda: {
        "tutor": {"pinned": True, "revision": release["package_ref"], "catalog": {"name": "tutor"}}
    })
    with pytest.raises(ValueError, match="matching Semantier Tutor"):
        catalog.project_installed_tutor_progress(release, "project", "plan", [])
