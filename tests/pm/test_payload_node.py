"""Sealed Playwright uses its payload-owned Node without environment leakage."""
import json
import os
from pathlib import Path

import pytest

from pm.payload_node import configure_payload_node


def payload(tmp_path):
    root = tmp_path / "payload"
    repo = root / "hermes-agent"
    repo.mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps({"repo": "hermes-agent", "store": "tools"}))
    store = root / "tools"
    binary = store / "node-pinned" / ("node.exe" if os.name == "nt" else "bin/node")
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"payload-node")
    (store / "facts.json").write_text(json.dumps({"schema": 1, "packages": {"node": {"entry": "node-pinned"}}}))
    return repo, binary


def test_selects_relocated_payload_node_without_mutating_process(tmp_path):
    repo, binary = payload(tmp_path)
    env = {}
    before = os.environ.get("PLAYWRIGHT_NODEJS_PATH")
    configure_payload_node(repo, env)
    assert Path(env["PLAYWRIGHT_NODEJS_PATH"]) == binary
    assert os.environ.get("PLAYWRIGHT_NODEJS_PATH") == before


def test_preserves_explicit_node_override(tmp_path):
    repo, _ = payload(tmp_path)
    env = {"PLAYWRIGHT_NODEJS_PATH": "operator-selected-node"}
    configure_payload_node(repo, env)
    assert env["PLAYWRIGHT_NODEJS_PATH"] == "operator-selected-node"


def test_source_checkout_does_not_select_an_unrelated_node(tmp_path):
    env = {}
    configure_payload_node(tmp_path, env)
    assert env == {}


def test_missing_packaged_node_fails_before_playwright_launch(tmp_path):
    repo, binary = payload(tmp_path)
    binary.unlink()
    with pytest.raises(RuntimeError, match="shared Node runtime is missing"):
        configure_payload_node(repo, {})


def test_facts_cannot_redirect_node_outside_the_payload(tmp_path):
    repo, _binary = payload(tmp_path)
    facts = repo.parent / "tools/facts.json"
    facts.write_text(json.dumps({"schema": 1, "packages": {"node": {"entry": "../../foreign"}}}))
    with pytest.raises(RuntimeError, match="escapes"):
        configure_payload_node(repo, {})


def test_normal_payload_activation_selects_shared_node(tmp_path, monkeypatch):
    import sys
    from pm.environments import activate_dependencies, site_packages

    repo, binary = payload(tmp_path)
    manifest = repo.parent / "manifest.json"
    manifest.write_text(json.dumps({"repo": "hermes-agent", "store": "tools", "venv": "venv"}))
    site_packages(repo.parent / "venv").mkdir(parents=True)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delenv("PLAYWRIGHT_NODEJS_PATH", raising=False)
    monkeypatch.setenv("PYTHONPATH", "")
    monkeypatch.setenv("PATH", os.environ.get("PATH", ""))
    monkeypatch.setenv("VIRTUAL_ENV", "unrelated-shell-environment")
    activate_dependencies(repo)
    assert Path(os.environ["PLAYWRIGHT_NODEJS_PATH"]) == binary
    assert "VIRTUAL_ENV" not in os.environ