from __future__ import annotations

from pathlib import Path


def _write_plugin(root: Path, name: str) -> None:
    plugin_dir = root / "plugins" / name
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        f"name: {name}\nversion: 0.1.0\nkind: standalone\n",
        encoding="utf-8",
    )
    (plugin_dir / "__init__.py").write_text(
        "def register(ctx):\n    return None\n",
        encoding="utf-8",
    )


def test_discover_plugins_scans_shared_runtime_and_workspace(monkeypatch, tmp_path):
    from hermes_cli import plugins as plugins_mod

    shared_home = tmp_path / ".semantier-home"
    workspace_home = tmp_path / "workspaces" / "ws-123" / ".hermes"
    _write_plugin(shared_home, "shared_plugin")
    _write_plugin(workspace_home, "workspace_plugin")
    workspace_home.mkdir(parents=True, exist_ok=True)
    (workspace_home / "config.yaml").write_text(
        "plugins:\n"
        "  enabled:\n"
        "    - shared_plugin\n"
        "    - workspace_plugin\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("SEMANTIER_LOCAL_STATE_DIR", str(shared_home))
    monkeypatch.setenv("HERMES_HOME", str(workspace_home))
    plugins_mod._plugin_manager = None

    plugins_mod.discover_plugins(force=True)
    manager = plugins_mod.get_plugin_manager()

    assert manager._plugins["shared_plugin"].enabled is True
    assert manager._plugins["workspace_plugin"].enabled is True


def test_workspace_plugin_overrides_shared_plugin_by_key(monkeypatch, tmp_path):
    from hermes_cli import plugins as plugins_mod

    shared_home = tmp_path / ".semantier-home"
    workspace_home = tmp_path / "workspaces" / "ws-123" / ".hermes"
    _write_plugin(shared_home, "tiered_plugin")
    _write_plugin(workspace_home, "tiered_plugin")
    workspace_home.mkdir(parents=True, exist_ok=True)
    (workspace_home / "config.yaml").write_text(
        "plugins:\n"
        "  enabled:\n"
        "    - tiered_plugin\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("SEMANTIER_LOCAL_STATE_DIR", str(shared_home))
    monkeypatch.setenv("HERMES_HOME", str(workspace_home))
    plugins_mod._plugin_manager = None

    plugins_mod.discover_plugins(force=True)
    manager = plugins_mod.get_plugin_manager()

    assert manager._plugins["tiered_plugin"].manifest.path.startswith(str(workspace_home))
