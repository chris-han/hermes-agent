from __future__ import annotations

import sys
from types import SimpleNamespace

from gateway.run import _governed_context_prompt_for_source


def test_ungoverned_source_cannot_activate_semantier_context():
    assert _governed_context_prompt_for_source(
        SimpleNamespace(workspace_owner_id=None),
        "show revenue",
    ) is None


def test_governed_source_delegates_to_parent_authority(monkeypatch):
    observed = {}

    def build(**kwargs):
        observed.update(kwargs)
        return "governed-context"

    # Hermes' standalone CI checkout intentionally has no Semantier parent
    # source tree. Substitute only that external authority boundary here; the
    # parent repository's integration suite exercises the real bridge module.
    monkeypatch.setitem(
        sys.modules,
        "agents.governed_context",
        SimpleNamespace(build_governed_runtime_context_prompt=build),
    )

    result = _governed_context_prompt_for_source(
        SimpleNamespace(workspace_owner_id="workspace-123"),
        "show revenue",
    )

    assert result == "governed-context"
    assert observed == {
        "workspace_id": "workspace-123",
        "user_id": None,
        "user_message": "show revenue",
    }
