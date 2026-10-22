"""Semantier-owned default workspace plugin seeding."""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from threading import RLock

import yaml


DEFAULT_WORKSPACE_PLUGINS = ("requirements_change_management",)
_DEFAULT_PLUGIN_SEED_LOCK = RLock()


def _skills_plugins_dir() -> Path:
    override = os.environ.get("SEMANTIER_SKILLS_PLUGINS_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "semantier-skills" / "plugins"


def ensure_default_workspace_plugins(home: Path | str) -> None:
    """Install/refresh Semantier defaults while preserving explicit user disable."""
    # Inventory and action handlers can bind the same home concurrently. Keep
    # directory refresh and configuration reconciliation in one critical section.
    with _DEFAULT_PLUGIN_SEED_LOCK:
        _seed_default_workspace_plugins(home)


def _seed_default_workspace_plugins(home: Path | str) -> None:
    target_home = Path(home).expanduser().resolve()
    target_home.mkdir(parents=True, exist_ok=True)
    plugins_dir = target_home / "plugins"
    plugins_dir.mkdir(parents=True, exist_ok=True)
    source_root = _skills_plugins_dir()

    config_path = target_home / "config.yaml"
    config = {}
    config_text = ""
    if config_path.exists():
        config_text = config_path.read_text(encoding="utf-8")
        loaded = yaml.safe_load(config_text)
        if isinstance(loaded, dict):
            config = loaded
    plugins_config = config.setdefault("plugins", {})
    enabled = list(plugins_config.get("enabled") or [])
    disabled = set(plugins_config.get("disabled") or [])

    for name in DEFAULT_WORKSPACE_PLUGINS:
        source = source_root / name
        if not (source / "plugin.yaml").exists():
            raise FileNotFoundError(f"default Semantier plugin source missing: {source}")
        target = plugins_dir / name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(
            source,
            target,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
        )
        if name not in disabled and name not in enabled:
            enabled.append(name)

    plugins_config["enabled"] = enabled
    if disabled:
        plugins_config["disabled"] = sorted(disabled)
    next_config_text = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
    if next_config_text != config_text:
        config_path.write_text(next_config_text, encoding="utf-8")


__all__ = ["DEFAULT_WORKSPACE_PLUGINS", "ensure_default_workspace_plugins"]
