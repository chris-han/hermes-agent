"""Shared pure helpers for retained advisory native verification."""

from __future__ import annotations

import base64
import binascii
import json
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Mapping

from contracts.ontology_building import ArtifactPinV1
from contracts.runtime_context_reconstruction import CapturedJsonV1, CapturedSourceV1


PinKey = tuple[str, str, str]


def _pin_key(pin: ArtifactPinV1) -> PinKey:
    return (pin.artifact_kind, pin.artifact_ref, pin.artifact_hash)


def _sorted_pins(pins: tuple[ArtifactPinV1, ...]) -> tuple[ArtifactPinV1, ...]:
    return tuple(sorted(pins, key=_pin_key))


def _validate_closure(
    closure: Mapping[PinKey, CapturedSourceV1],
) -> dict[PinKey, CapturedSourceV1]:
    resolved: dict[PinKey, CapturedSourceV1] = {}
    refs: dict[tuple[str, str], str] = {}
    for supplied_key, source in closure.items():
        validated = CapturedSourceV1.model_validate(source.model_dump(mode="json"))
        actual_key = _pin_key(validated.source)
        if supplied_key != actual_key:
            raise ValueError("retained closure key does not match captured source pin")
        ref_key = actual_key[:2]
        previous = refs.setdefault(ref_key, actual_key[2])
        if previous != actual_key[2]:
            raise ValueError(
                "retained closure contains one source ref with conflicting hashes"
            )
        resolved[actual_key] = validated
    return resolved


def _payload(source: CapturedSourceV1) -> tuple[CapturedSourceV1, Any]:
    validated = CapturedSourceV1.model_validate(source.model_dump(mode="json"))
    captured = CapturedJsonV1.model_validate(validated.payload.model_dump(mode="json"))
    return validated, json.loads(captured.canonical_json)


def _require_supplied_source(
    source: CapturedSourceV1,
    closure: Mapping[PinKey, CapturedSourceV1],
) -> CapturedSourceV1:
    validated = CapturedSourceV1.model_validate(source.model_dump(mode="json"))
    retained = closure.get(_pin_key(validated.source))
    if retained is None or retained != validated:
        raise ValueError("supplied source does not equal its retained closure entry")
    return validated


def _captured_json(value: Any, label: str) -> tuple[CapturedJsonV1, Any]:
    try:
        captured = CapturedJsonV1.model_validate(value)
        decoded = json.loads(captured.canonical_json)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not exact captured canonical JSON") from exc
    return captured, decoded


def _strict_base64(value: Any, label: str) -> bytes:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be strict canonical base64")
    try:
        raw = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"{label} must be strict canonical base64") from exc
    if base64.b64encode(raw).decode("ascii") != value:
        raise ValueError(f"{label} must be strict canonical base64")
    return raw


def _require_relative_ref(value: str, label: str) -> None:
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        not value
        or posix.is_absolute()
        or windows.is_absolute()
        or "\\" in value
        or ".." in posix.parts
    ):
        raise ValueError(f"{label} must be a governed relative artifact ref")


def _expect_identity(
    source: CapturedSourceV1, *, kind: str, ref: str, digest: str
) -> None:
    if _pin_key(source.source) != (kind, ref, digest):
        raise ValueError(f"{kind} retained source pin does not match native identity")


def _expect_dependencies(
    source: CapturedSourceV1,
    expected: tuple[ArtifactPinV1, ...],
    closure: Mapping[PinKey, CapturedSourceV1],
) -> tuple[ArtifactPinV1, ...]:
    expected = _sorted_pins(expected)
    if tuple(source.dependencies) != expected:
        raise ValueError(
            "retained source dependencies do not match native material edges"
        )
    source_key = _pin_key(source.source)
    for pin in expected:
        key = _pin_key(pin)
        if key == source_key:
            raise ValueError("retained source dependency cycle")
        dependency = closure.get(key)
        if dependency is None or _pin_key(dependency.source) != key:
            raise ValueError("retained source dependency is missing or conflicts")
    return expected
