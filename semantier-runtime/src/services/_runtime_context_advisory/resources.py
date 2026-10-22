"""Pure retained advisory composite-resource format primitives."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Mapping, Sequence

from eos.artifact_hashing import canonical_json


COMPOSITE_RESOURCE_IDS = frozenset(
    {
        "advisory_kernel_v1",
        "advisory_native_v1",
        "advisory_operator_v1",
        "advisory_representation_v1",
    }
)
RAW_RESOURCE_IDS = frozenset(
    {
        "advisory_profile_v1",
        "compiler_semantic_compiler",
        "compiler_validation_compiler",
        "compiler_rule_plan_compiler",
        "compiler_prompt",
        "compiler_configuration",
        "compiler_frame_schema",
        "compiler_profile",
    }
)
_MAX_COMPOSITE_BYTES = 33_554_432


@dataclass(frozen=True, slots=True)
class ResourceRequirementV1:
    resource_id: str
    name: str
    bytes_hash: str


@dataclass(frozen=True, slots=True)
class ResourceBytesV1:
    resource_id: str
    raw_bytes: bytes


@dataclass(frozen=True, slots=True)
class ModuleAllocationV1:
    name: str
    resource_id: str
    raw_bytes: bytes


@dataclass(frozen=True, slots=True)
class CompositeMemberV1:
    name: str
    byte_length: int
    bytes_hash: str
    raw_bytes: bytes


@dataclass(frozen=True, slots=True)
class CompositeResourceV1:
    raw_bytes: bytes
    members: tuple[CompositeMemberV1, ...]
    requires: tuple[ResourceRequirementV1, ...]


@dataclass(frozen=True, slots=True)
class ValidatedResourceInventoryV1:
    resources: tuple[ResourceBytesV1, ...]
    composites: tuple[CompositeResourceV1, ...]
    charged_composite_bytes: int


_ALL_RESOURCE_IDS = COMPOSITE_RESOURCE_IDS | RAW_RESOURCE_IDS
_COMPILER_CODE_RESOURCE_IDS = frozenset(
    {
        "compiler_semantic_compiler",
        "compiler_validation_compiler",
        "compiler_rule_plan_compiler",
    }
)
_MODULE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMPOSITE_FIELDS = {"format_version", "members", "requires"}
_MEMBER_FIELDS = {"name", "byte_length", "bytes_hash", "bytes_base64"}
_REQUIREMENT_FIELDS = {"resource_id", "name", "bytes_hash"}


class _DuplicateJsonKey(ValueError):
    pass


def _sha256(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _require_bytes(value: object, label: str) -> bytes:
    if type(value) is not bytes:
        raise ValueError(f"{label} must be bytes")
    return value


def _require_module_name(value: object) -> str:
    if not isinstance(value, str) or _MODULE_NAME.fullmatch(value) is None:
        raise ValueError("member name must be an ASCII dotted Python module name")
    return value


def _require_hash(value: object, label: str = "bytes_hash") -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be an exact sha256 hash")
    return value


def _require_resource_id(value: object) -> str:
    if not isinstance(value, str) or value not in _ALL_RESOURCE_IDS:
        raise ValueError("unknown resource_id")
    return value


def _require_sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, (tuple, list)):
        raise ValueError(f"{label} must be a finite array")
    return value


def _validate_requirement(value: object) -> ResourceRequirementV1:
    if isinstance(value, ResourceRequirementV1):
        resource_id = value.resource_id
        name = value.name
        bytes_hash = value.bytes_hash
    elif isinstance(value, Mapping):
        if set(value) != _REQUIREMENT_FIELDS:
            raise ValueError("requirement must contain the exact fields")
        resource_id = value["resource_id"]
        name = value["name"]
        bytes_hash = value["bytes_hash"]
    else:
        raise ValueError("requirement must contain the exact fields")
    return ResourceRequirementV1(
        resource_id=_require_resource_id(resource_id),
        name=_require_module_name(name),
        bytes_hash=_require_hash(bytes_hash),
    )


def _requirement_key(value: ResourceRequirementV1) -> tuple[bytes, bytes, bytes]:
    return tuple(
        item.encode("utf-8")
        for item in (value.resource_id, value.name, value.bytes_hash)
    )  # type: ignore[return-value]


def encode_composite_resource(
    *,
    members: Sequence[tuple[str, bytes]],
    requires: Sequence[ResourceRequirementV1] = (),
) -> bytes:
    members = _require_sequence(members, "members")  # type: ignore[assignment]
    requires = _require_sequence(requires, "requires")  # type: ignore[assignment]
    encoded_members: list[dict[str, object]] = []
    names: set[str] = set()
    total_member_bytes = 0
    for value in members:
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise ValueError("member source must contain name and raw bytes")
        name = _require_module_name(value[0])
        raw = _require_bytes(value[1], f"member {name} raw bytes")
        if name in names:
            raise ValueError("duplicate member name")
        names.add(name)
        total_member_bytes += len(raw)
        if total_member_bytes > _MAX_COMPOSITE_BYTES:
            raise ValueError("composite member bytes exceed the 32 MiB bound")
        encoded_members.append(
            {
                "name": name,
                "byte_length": len(raw),
                "bytes_hash": _sha256(raw),
                "bytes_base64": base64.b64encode(raw).decode("ascii"),
            }
        )
    validated_requirements = tuple(_validate_requirement(value) for value in requires)
    requirement_keys = [_requirement_key(value) for value in validated_requirements]
    if len(requirement_keys) != len(set(requirement_keys)):
        raise ValueError("duplicate requirement")
    payload = {
        "format_version": 1,
        "members": sorted(
            encoded_members, key=lambda item: str(item["name"]).encode("utf-8")
        ),
        "requires": [
            {
                "resource_id": value.resource_id,
                "name": value.name,
                "bytes_hash": value.bytes_hash,
            }
            for value in sorted(validated_requirements, key=_requirement_key)
        ],
    }
    encoded = canonical_json(payload).encode("utf-8")
    if len(encoded) > _MAX_COMPOSITE_BYTES:
        raise ValueError("complete composite bytes exceed the 32 MiB bound")
    return encoded


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKey(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant: {value}")


def decode_composite_resource(raw_bytes: bytes) -> CompositeResourceV1:
    raw_bytes = _require_bytes(raw_bytes, "composite")
    if len(raw_bytes) > _MAX_COMPOSITE_BYTES:
        raise ValueError("complete composite bytes exceed the 32 MiB bound")
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("composite must be canonical UTF-8 JSON") from exc
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except _DuplicateJsonKey:
        raise
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        raise ValueError("composite must be canonical JSON") from exc
    if not isinstance(payload, dict) or set(payload) != _COMPOSITE_FIELDS:
        raise ValueError("composite must contain the exact fields")
    if type(payload["format_version"]) is not int or payload["format_version"] != 1:
        raise ValueError("format_version must be integer 1")
    member_values = payload["members"]
    requirement_values = payload["requires"]
    if not isinstance(member_values, list):
        raise ValueError("members must be an array")
    if not isinstance(requirement_values, list):
        raise ValueError("requires must be an array")
    if len(member_values) > len(raw_bytes) or len(requirement_values) > len(raw_bytes):
        raise ValueError("composite arrays exceed their charged byte bound")

    members: list[CompositeMemberV1] = []
    declared_total = 0
    encoded_total = 0
    for value in member_values:
        if not isinstance(value, dict) or set(value) != _MEMBER_FIELDS:
            raise ValueError("member must contain the exact fields")
        name = _require_module_name(value["name"])
        byte_length = value["byte_length"]
        if type(byte_length) is not int or byte_length < 0:
            raise ValueError(
                "member byte_length must be a nonnegative nonboolean integer"
            )
        bytes_hash = _require_hash(value["bytes_hash"])
        bytes_base64 = value["bytes_base64"]
        if not isinstance(bytes_base64, str):
            raise ValueError("member bytes_base64 must be strict canonical base64")
        encoded_total += len(bytes_base64)
        declared_total += byte_length
        if encoded_total > len(raw_bytes) or declared_total > len(raw_bytes):
            raise ValueError("member arrays exceed the charged composite byte bound")
        try:
            raw = base64.b64decode(bytes_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(
                "member bytes_base64 must be strict canonical base64"
            ) from exc
        if base64.b64encode(raw).decode("ascii") != bytes_base64:
            raise ValueError("member bytes_base64 must be strict canonical base64")
        if len(raw) != byte_length:
            raise ValueError("member byte_length mismatch")
        if _sha256(raw) != bytes_hash:
            raise ValueError("member bytes hash mismatch")
        members.append(CompositeMemberV1(name, byte_length, bytes_hash, raw))

    names = [value.name.encode("utf-8") for value in members]
    if len(names) != len(set(names)):
        raise ValueError("duplicate member name")
    if names != sorted(names):
        raise ValueError("composite members must be UTF-8 name ordered")

    requirements = tuple(_validate_requirement(value) for value in requirement_values)
    requirement_keys = [_requirement_key(value) for value in requirements]
    if len(requirement_keys) != len(set(requirement_keys)):
        raise ValueError("duplicate requirement")
    if requirement_keys != sorted(requirement_keys):
        raise ValueError("composite requirements must be UTF-8 tuple ordered")
    if canonical_json(payload).encode("utf-8") != raw_bytes:
        raise ValueError("composite must use the exact canonical JSON encoding")
    return CompositeResourceV1(raw_bytes, tuple(members), requirements)


def _validate_resource(value: object) -> ResourceBytesV1:
    if not isinstance(value, ResourceBytesV1):
        raise ValueError("resources must be ResourceBytesV1 values")
    return ResourceBytesV1(
        resource_id=_require_resource_id(value.resource_id),
        raw_bytes=_require_bytes(value.raw_bytes, "resource raw bytes"),
    )


def _validate_allocation(value: object) -> ModuleAllocationV1:
    if not isinstance(value, ModuleAllocationV1):
        raise ValueError("allocations must be ModuleAllocationV1 values")
    return ModuleAllocationV1(
        name=_require_module_name(value.name),
        resource_id=_require_resource_id(value.resource_id),
        raw_bytes=_require_bytes(value.raw_bytes, "allocation raw bytes"),
    )


def validate_resource_inventory(
    *,
    resources: Sequence[ResourceBytesV1],
    allocations: Sequence[ModuleAllocationV1],
) -> ValidatedResourceInventoryV1:
    resources = _require_sequence(resources, "resources")  # type: ignore[assignment]
    allocations = _require_sequence(allocations, "allocations")  # type: ignore[assignment]
    validated_resources = tuple(_validate_resource(value) for value in resources)
    resource_ids = [value.resource_id for value in validated_resources]
    if len(resource_ids) != len(set(resource_ids)):
        raise ValueError("duplicate resource_id")

    validated_allocations = tuple(_validate_allocation(value) for value in allocations)
    allocation_names = [value.name for value in validated_allocations]
    if len(allocation_names) != len(set(allocation_names)):
        raise ValueError("each retained module must have one physical owner")
    resources_by_id = {value.resource_id: value for value in validated_resources}
    for value in validated_allocations:
        if value.resource_id not in resources_by_id:
            raise ValueError("module allocation names an unsupplied resource")

    composite_resources = tuple(
        value
        for value in validated_resources
        if value.resource_id in COMPOSITE_RESOURCE_IDS
    )
    charged_composite_bytes = sum(len(value.raw_bytes) for value in composite_resources)
    if charged_composite_bytes > _MAX_COMPOSITE_BYTES:
        raise ValueError("cumulative complete composite bytes exceed the 32 MiB bound")

    composites_by_id = {
        value.resource_id: decode_composite_resource(value.raw_bytes)
        for value in composite_resources
    }
    allocations_by_resource: dict[str, dict[str, ModuleAllocationV1]] = {}
    for value in validated_allocations:
        allocations_by_resource.setdefault(value.resource_id, {})[value.name] = value

    for resource_id, composite in composites_by_id.items():
        members = {value.name: value for value in composite.members}
        owned = allocations_by_resource.get(resource_id, {})
        missing = set(members) - set(owned)
        extra = set(owned) - set(members)
        if missing:
            raise ValueError("composite member has a missing module allocation")
        if extra:
            raise ValueError("composite resource has an extra module allocation")
        for name, member in members.items():
            if owned[name].raw_bytes != member.raw_bytes:
                raise ValueError(
                    "module allocation bytes conflict with composite member"
                )

    for resource_id in RAW_RESOURCE_IDS:
        owned = allocations_by_resource.get(resource_id, {})
        if not owned:
            continue
        if resource_id not in _COMPILER_CODE_RESOURCE_IDS:
            raise ValueError("non-code raw resource cannot own a repository module")
        if len(owned) != 1:
            raise ValueError("a raw compiler resource may own only one whole module")
        resource = resources_by_id[resource_id]
        allocation = next(iter(owned.values()))
        if allocation.raw_bytes != resource.raw_bytes:
            raise ValueError(
                "compiler module allocation must equal the whole raw resource"
            )

    allocations_by_binding = {
        (value.resource_id, value.name): value for value in validated_allocations
    }
    for composite in composites_by_id.values():
        for requirement in composite.requires:
            allocation = allocations_by_binding.get(
                (requirement.resource_id, requirement.name)
            )
            if allocation is None:
                raise ValueError("unresolved requirement binding")
            if _sha256(allocation.raw_bytes) != requirement.bytes_hash:
                raise ValueError("conflicting requirement byte binding")

    ordered_resources = tuple(
        sorted(validated_resources, key=lambda value: value.resource_id.encode("utf-8"))
    )
    ordered_composites = tuple(
        composites_by_id[value.resource_id]
        for value in ordered_resources
        if value.resource_id in composites_by_id
    )
    return ValidatedResourceInventoryV1(
        resources=ordered_resources,
        composites=ordered_composites,
        charged_composite_bytes=charged_composite_bytes,
    )
