"""Dependency-light owner of the canonical WorkingContext selection-policy pin."""

from __future__ import annotations

from contracts.ontology_building import ArtifactPinV1
from eos.artifact_hashing import content_hash

_SELECTOR_DIGEST = content_hash(
    {
        "selector": "working_context_explicit_selection",
        "version": "v1",
        "behavior": "exact selected ids over admitted ContextGraph",
    }
)
_SELECTION_POLICY = ArtifactPinV1(
    artifact_kind="working_context_selector",
    artifact_ref=f"working-context-selector:explicit-v1:{_SELECTOR_DIGEST}",
    artifact_hash=f"sha256:{_SELECTOR_DIGEST}",
)


def working_context_selection_policy() -> ArtifactPinV1:
    return _SELECTION_POLICY


__all__ = ["working_context_selection_policy"]
