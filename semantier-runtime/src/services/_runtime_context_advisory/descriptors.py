"""Pure retained advisory product descriptor values."""

from __future__ import annotations

import re
from typing import Mapping

from contracts.ontology_building import ArtifactPinV1
from contracts.runtime_context_reconstruction import CapturedJsonV1, CapturedSourceV1
from eos.artifact_hashing import canonical_json, content_hash

from .common import (
    PinKey,
    _expect_dependencies,
    _expect_identity,
    _payload,
    _require_supplied_source,
    _validate_closure,
)


_PROFILE_VALUE = {
    "failure_mode": "reject",
    "max_capture_nodes": 2048,
    "max_dependency_depth": 32,
    "max_pairs": 8,
    "max_total_bytes": 33_554_432,
    "pair_order": "request",
    "profile_id": "context_only_empirical_v1",
    "purpose": "NEXT_PROPOSAL",
    "schema_version": "advisory_context_profile.v1",
    "source_order": "kind_ref_hash_utf8",
    "usage": "CONTEXT_ONLY",
}
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_PROFILE_DIGEST = content_hash(_PROFILE_VALUE)
_PROFILE_PIN = ArtifactPinV1(
    artifact_kind="advisory_context_profile",
    artifact_ref=f"advisory-context-profile:{_PROFILE_DIGEST}",
    artifact_hash=f"sha256:{_PROFILE_DIGEST}",
)


def verify_advisory_context_profile_v1(
    *, source: CapturedSourceV1, closure: Mapping[PinKey, CapturedSourceV1]
) -> tuple[ArtifactPinV1, ...]:
    closure = _validate_closure(closure)
    source = _require_supplied_source(source, closure)
    source, payload = _payload(source)
    encoded = canonical_json(_PROFILE_VALUE)
    if payload != _PROFILE_VALUE or source.payload.canonical_json != encoded:
        raise ValueError("advisory context profile must equal the exact product value")
    _expect_identity(
        source,
        kind=_PROFILE_PIN.artifact_kind,
        ref=_PROFILE_PIN.artifact_ref,
        digest=_PROFILE_PIN.artifact_hash,
    )
    return _expect_dependencies(source, (), closure)


def verify_advisory_context_operator_v1(
    *, source: CapturedSourceV1, closure: Mapping[PinKey, CapturedSourceV1]
) -> tuple[ArtifactPinV1, ...]:
    closure = _validate_closure(closure)
    source = _require_supplied_source(source, closure)
    source, payload = _payload(source)
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "operator_id",
        "implementation_resource_id",
        "implementation_hash",
        "profile",
    }:
        raise ValueError("advisory context operator must contain the exact fields")
    try:
        profile = ArtifactPinV1.model_validate(payload["profile"])
    except (TypeError, ValueError) as exc:
        raise ValueError("advisory context operator profile pin is invalid") from exc
    implementation_hash = payload["implementation_hash"]
    expected = {
        "schema_version": "advisory_context_operator.v1",
        "operator_id": "context_only_empirical_v1",
        "implementation_resource_id": "advisory_operator_v1",
        "implementation_hash": implementation_hash,
        "profile": profile.model_dump(mode="json"),
    }
    if (
        payload != expected
        or profile != _PROFILE_PIN
        or not isinstance(implementation_hash, str)
        or _SHA256.fullmatch(implementation_hash) is None
        or source.payload.canonical_json != canonical_json(expected)
    ):
        raise ValueError("advisory context operator must equal the exact product value")
    digest = content_hash(expected)
    _expect_identity(
        source,
        kind="advisory_context_operator",
        ref=f"advisory-context-operator:{digest}",
        digest=f"sha256:{digest}",
    )
    dependencies = _expect_dependencies(source, (profile,), closure)
    profile_source = closure.get(
        (profile.artifact_kind, profile.artifact_ref, profile.artifact_hash)
    )
    if profile_source is None:
        raise ValueError("advisory context operator profile capture is missing")
    verify_advisory_context_profile_v1(source=profile_source, closure=closure)
    return dependencies


def _advisory_operator_descriptor(
    *, profile: ArtifactPinV1, operator_resource_hash: str
) -> CapturedJsonV1:
    if not isinstance(profile, ArtifactPinV1) or profile != _PROFILE_PIN:
        raise ValueError("operator descriptor requires the exact product profile pin")
    if (
        not isinstance(operator_resource_hash, str)
        or _SHA256.fullmatch(operator_resource_hash) is None
    ):
        raise ValueError("operator_resource_hash must be an exact sha256 hash")
    value = {
        "schema_version": "advisory_context_operator.v1",
        "operator_id": "context_only_empirical_v1",
        "implementation_resource_id": "advisory_operator_v1",
        "implementation_hash": operator_resource_hash,
        "profile": profile.model_dump(mode="json"),
    }
    return CapturedJsonV1(
        media_type="application/json",
        canonical_json=canonical_json(value),
        bytes_hash="sha256:" + content_hash(value),
    )


__all__ = [
    "verify_advisory_context_operator_v1",
    "verify_advisory_context_profile_v1",
]
