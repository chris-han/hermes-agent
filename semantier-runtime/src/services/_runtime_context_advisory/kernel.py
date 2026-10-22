"""Pure authenticated kernel for retained advisory capture verification."""

from __future__ import annotations

import base64
import binascii
from hashlib import sha256
import json
from typing import Callable, Mapping

from contracts.ontology_building import ArtifactPinV1
from contracts.runtime_context_reconstruction import (
    CapturedJsonV1,
    CapturedSourceV1,
    PackagedVerifierEntryV1,
    PackagedVerifierRegistryV1,
    RetainedPackageResourceV1,
    RetainedVerifierPackageV1,
    TrustedRetainedVerifierRegistrationV1,
    packaged_verifier_registry_pin,
)
from eos.artifact_hashing import canonical_json, content_hash

from .representation import verify_advisory_episode_representation_v1
from .resources import (
    ModuleAllocationV1,
    ResourceBytesV1,
    ResourceRequirementV1,
    decode_composite_resource,
    validate_resource_inventory,
)


_MAX_TOTAL_BYTES = 33_554_432
_MAX_CAPTURE_NODES = 2048
_MAX_DEPENDENCY_DEPTH = 32
_MAX_PAIRS = 8
_REGISTRY_VERSION = "context_only_empirical_v1"
_KERNEL_ID = "semantier.offline_artifact_integrity.v1"

_RESOURCE_MEDIA_TYPES = {
    "advisory_kernel_v1": "application/json",
    "advisory_native_v1": "application/json",
    "advisory_operator_v1": "application/json",
    "advisory_representation_v1": "application/json",
    "advisory_profile_v1": "application/json",
    "compiler_semantic_compiler": "text/x-python",
    "compiler_validation_compiler": "text/x-python",
    "compiler_rule_plan_compiler": "text/x-python",
    "compiler_prompt": "text/plain",
    "compiler_configuration": "application/json",
    "compiler_frame_schema": "application/json",
    "compiler_profile": "application/json",
}

# This is release policy, not package-selected ownership. Values deliberately omit
# hashes: startup binds the completed installed bytes after this module exists.
_MODULE_ALLOCATION = (
    ("contracts", "advisory_kernel_v1"),
    ("contracts.canonical_source_ir", "advisory_kernel_v1"),
    ("contracts.evidence_location", "advisory_kernel_v1"),
    ("contracts.graph_intent_brief", "advisory_kernel_v1"),
    ("contracts.normative_semantics", "advisory_kernel_v1"),
    ("contracts.ontology_artifact_integrity", "advisory_kernel_v1"),
    ("contracts.ontology_audit", "advisory_kernel_v1"),
    ("contracts.ontology_build_inputs", "advisory_kernel_v1"),
    ("contracts.ontology_building", "advisory_kernel_v1"),
    ("contracts.ontology_paired_evaluation", "advisory_kernel_v1"),
    ("contracts.ontology_validation", "advisory_kernel_v1"),
    ("contracts.runtime_context_reconstruction", "advisory_kernel_v1"),
    ("contracts.runtime_learning_trial", "advisory_kernel_v1"),
    ("contracts.runtime_working_context", "advisory_kernel_v1"),
    ("contracts.validation_contract", "advisory_kernel_v1"),
    ("eos", "advisory_kernel_v1"),
    ("eos.artifact_hashing", "advisory_kernel_v1"),
    ("semantier", "advisory_kernel_v1"),
    ("semantier.knowledge_feature_packs", "advisory_kernel_v1"),
    (
        "semantier.knowledge_feature_packs.canonical_source_ir_adapter",
        "advisory_kernel_v1",
    ),
    (
        "semantier.knowledge_feature_packs.evidence_selector_builder",
        "advisory_kernel_v1",
    ),
    ("services", "advisory_kernel_v1"),
    ("services._runtime_context_advisory", "advisory_kernel_v1"),
    ("services._runtime_context_advisory.build", "advisory_native_v1"),
    ("services._runtime_context_advisory.common", "advisory_kernel_v1"),
    ("services._runtime_context_advisory.compiler", "advisory_native_v1"),
    ("services._runtime_context_advisory.descriptors", "advisory_native_v1"),
    ("services._runtime_context_advisory.empirical", "advisory_native_v1"),
    ("services._runtime_context_advisory.kernel", "advisory_kernel_v1"),
    ("services._runtime_context_advisory.operator", "advisory_operator_v1"),
    ("services._runtime_context_advisory.paired_inputs", "advisory_native_v1"),
    ("services._runtime_context_advisory.paired_outputs", "advisory_native_v1"),
    ("services._runtime_context_advisory.representation", "advisory_representation_v1"),
    ("services._runtime_context_advisory.resources", "advisory_kernel_v1"),
    ("services._runtime_context_advisory.source", "advisory_native_v1"),
    ("services._runtime_context_advisory.trial", "advisory_native_v1"),
    ("services._runtime_context_advisory.trial_build", "advisory_native_v1"),
    ("services.external_validation_compiler", "compiler_validation_compiler"),
    ("services.normative_semantic_compiler", "compiler_semantic_compiler"),
    ("services.ontology_rule_plan_compiler", "compiler_rule_plan_compiler"),
    ("services.runtime_context_reconstruction_owners", "advisory_native_v1"),
    ("services.runtime_learning_trial_service", "advisory_native_v1"),
)

_NATIVE_REQUIREMENTS = (
    "contracts.canonical_source_ir",
    "contracts.evidence_location",
    "contracts.graph_intent_brief",
    "contracts.normative_semantics",
    "contracts.ontology_artifact_integrity",
    "contracts.ontology_build_inputs",
    "contracts.ontology_building",
    "contracts.ontology_paired_evaluation",
    "contracts.ontology_validation",
    "contracts.runtime_context_reconstruction",
    "contracts.runtime_learning_trial",
    "contracts.validation_contract",
    "eos.artifact_hashing",
    "semantier.knowledge_feature_packs.canonical_source_ir_adapter",
    "semantier.knowledge_feature_packs.evidence_selector_builder",
    "services",
    "services._runtime_context_advisory",
    "services._runtime_context_advisory.common",
)
_PRODUCT_REQUIREMENTS = (
    "contracts.ontology_building",
    "contracts.runtime_context_reconstruction",
    "contracts.runtime_learning_trial",
    "eos.artifact_hashing",
    "services",
    "services._runtime_context_advisory",
    "services._runtime_context_advisory.common",
)

# kind, schema, mode, ref field, hash field, prefix
_ENTRY_SPECS = (
    (
        "advisory_compiler_inputs",
        "advisory_compiler_inputs.v1",
        "native_owner",
        None,
        None,
        None,
    ),
    (
        "advisory_context_operator",
        "advisory_context_operator.v1",
        "full_canonical_json",
        None,
        None,
        None,
    ),
    (
        "advisory_context_profile",
        "advisory_context_profile.v1",
        "full_canonical_json",
        None,
        None,
        None,
    ),
    (
        "advisory_graph_intent_publication_capture",
        "advisory_graph_intent_publication_capture.v1",
        "native_owner",
        None,
        None,
        None,
    ),
    (
        "advisory_paired_suite_review",
        "advisory_paired_suite_review.v1",
        "self_bound_envelope",
        "decision_ref",
        "decision_hash",
        "advisory-paired-suite-review",
    ),
    (
        "advisory_source_capture",
        "advisory_source_capture.v1",
        "native_owner",
        None,
        None,
        None,
    ),
    (
        "advisory_trial_build_inputs",
        "advisory_trial_build_inputs.v1",
        "self_bound_envelope",
        "input_ref",
        "input_hash",
        "advisory-trial-build-inputs",
    ),
    (
        "advisory_trial_variant",
        "advisory_trial_variant.v1",
        "native_owner",
        None,
        None,
        None,
    ),
    (
        "canonical_evidence_adequacy_assessment",
        "canonical_evidence_adequacy_assessment.v1",
        "native_owner",
        None,
        None,
        None,
    ),
    (
        "canonical_evidence_boundary",
        "canonical_evidence_boundary.v1",
        "native_owner",
        None,
        None,
        None,
    ),
    ("canonical_source_ir", "canonical_source_ir.v1", "native_owner", None, None, None),
    ("competency_contract", "competency_contract.v1", "native_owner", None, None, None),
    ("compiled_rule_plan", "compiled_rule_plan.v1", "native_owner", None, None, None),
    (
        "empirical_episode",
        "empirical_episode.v1",
        "self_bound_envelope",
        "episode_ref",
        "episode_hash",
        "empirical-episode",
    ),
    (
        "empirical_episode_assessment",
        "empirical_episode_assessment.v1",
        "self_bound_envelope",
        "assessment_ref",
        "assessment_hash",
        "empirical-episode-assessment",
    ),
    (
        "empirical_episode_plan",
        "empirical_episode_plan.v1",
        "self_bound_envelope",
        "plan_ref",
        "plan_hash",
        "empirical-episode-plan",
    ),
    ("evidence_ref", "semantier.evidence_ref.v1", "native_owner", None, None, None),
    (
        "exploratory_graph_snapshot",
        "exploratory_graph_snapshot.v1",
        "native_owner",
        None,
        None,
        None,
    ),
    (
        "exploratory_graph_snapshot_integrity",
        "exploratory_graph_snapshot_integrity.v1",
        "native_owner",
        None,
        None,
        None,
    ),
    ("graph_intent_brief", "graph_intent_brief.v1", "native_owner", None, None, None),
    (
        "normative_semantic_candidate_set",
        "normative_semantic_candidate_set.v1",
        "native_owner",
        None,
        None,
        None,
    ),
    (
        "ontology_build_inputs",
        "ontology_build_inputs.v1",
        "native_owner",
        None,
        None,
        None,
    ),
    (
        "ontology_build_manifest",
        "ontology_build_manifest.v1",
        "native_owner",
        None,
        None,
        None,
    ),
    (
        "ontology_paired_evaluation_report",
        "ontology_paired_evaluation_report.v1",
        "self_bound_envelope",
        "report_ref",
        "report_hash",
        "ontology-paired-evaluation-report",
    ),
    (
        "ontology_paired_evaluation_run",
        "ontology_paired_evaluation_run.v1",
        "self_bound_envelope",
        "run_ref",
        "run_hash",
        "ontology-paired-evaluation-run",
    ),
    (
        "ontology_paired_obligation_suite",
        "ontology_paired_obligation_suite.v1",
        "self_bound_envelope",
        "suite_ref",
        "suite_hash",
        "ontology-paired-obligation-suite",
    ),
    (
        "ontology_source_first_compiler",
        "ontology_source_first_compiler_source.v1",
        "native_owner",
        None,
        None,
        None,
    ),
    (
        "paired_evaluation_policy",
        "paired_evaluation_policy.v1",
        "native_owner",
        None,
        None,
        None,
    ),
    (
        "paired_evaluator_bundle",
        "paired_evaluator_bundle.v1",
        "self_bound_envelope",
        "evaluator_ref",
        "evaluator_hash",
        "paired-evaluator-bundle",
    ),
    (
        "paired_observed_facts",
        "paired_observed_facts.v1",
        "self_bound_envelope",
        "facts_ref",
        "facts_hash",
        "paired-observed-facts",
    ),
    (
        "runtime_semantic_evaluation_bundle",
        "runtime_semantic_evaluation_bundle.v1",
        "native_owner",
        None,
        None,
        None,
    ),
    (
        "source_identity",
        "semantier.source_identity.v1",
        "native_owner",
        None,
        None,
        None,
    ),
    ("validation_contract", "validation_contract.v1", "native_owner", None, None, None),
)

_SCHEMA_FIELDS = {
    "canonical_source_ir": "canonicalIrVersion",
    "evidence_ref": "schemaVersion",
    "ontology_source_first_compiler": None,
    "paired_evaluation_policy": None,
    "source_identity": "schemaVersion",
}
_COMPILER_ROLES = {
    "semantic_compiler": (
        "compiler_semantic_compiler",
        "repo:src/services/normative_semantic_compiler.py",
    ),
    "validation_compiler": (
        "compiler_validation_compiler",
        "repo:src/services/external_validation_compiler.py",
    ),
    "rule_plan_compiler": (
        "compiler_rule_plan_compiler",
        "repo:src/services/ontology_rule_plan_compiler.py",
    ),
    "prompt": (
        "compiler_prompt",
        "repo:src/prompts/normative_semantic_compilation_v1.txt",
    ),
    "configuration": (
        "compiler_configuration",
        "repo:src/services/runtime_context_advisory_assets/compiler_configuration_v1.json",
    ),
    "frame_schema": (
        "compiler_frame_schema",
        "repo:src/services/runtime_context_advisory_assets/compiler_frame_schema_v1.json",
    ),
    "profile": (
        "compiler_profile",
        "repo:src/services/runtime_context_advisory_assets/compiler_profile_v1.json",
    ),
}


def _digest(raw: bytes) -> str:
    return "sha256:" + sha256(raw).hexdigest()


def _pin_key(pin: ArtifactPinV1) -> tuple[str, str, str]:
    return (pin.artifact_kind, pin.artifact_ref, pin.artifact_hash)


def _expected_entries() -> tuple[PackagedVerifierEntryV1, ...]:
    return tuple(
        PackagedVerifierEntryV1(
            entry_id=kind,
            artifact_kind=kind,
            schema_versions=(schema,),
            verification_mode=mode,
            ref_field=ref_field,
            hash_field=hash_field,
            ref_prefix=prefix,
            resource_id="advisory_native_v1",
            entrypoint=f"verify_{kind}_v1",
        )
        for kind, schema, mode, ref_field, hash_field, prefix in _ENTRY_SPECS
    )


def _enforce_total_bytes(parts: tuple[int, ...]) -> None:
    total = 0
    for part in parts:
        if type(part) is not int or part < 0:
            raise ValueError("decoded byte charges must be nonnegative integers")
        total += part
        if total > _MAX_TOTAL_BYTES:
            raise ValueError("decoded typed parts exceed the 32 MiB bound")


def _strict_trusted_rows(
    rows: object,
    expected_ids: tuple[str, ...],
    *,
    label: str,
    callable_values: bool = False,
) -> dict[str, object]:
    if type(rows) is not tuple:
        raise ValueError(f"trusted {label} must be an exact immutable tuple")
    result: dict[str, object] = {}
    keys: list[str] = []
    for row in rows:
        if type(row) is not tuple or len(row) != 2 or type(row[0]) is not str:
            raise ValueError(f"trusted {label} row is malformed")
        key, value = row
        if key in result:
            raise ValueError(f"trusted {label} contains a duplicate identifier")
        if callable_values:
            if not callable(value):
                raise ValueError("trusted verifier registration is missing a callable")
        elif type(value) is not str:
            raise ValueError("trusted resource hash must be a string")
        result[key] = value
        keys.append(key)
    if tuple(keys) != expected_ids:
        raise ValueError(f"trusted {label} must equal the complete selected release")
    return result


def _expected_requirements(
    names: tuple[str, ...], member_bytes: Mapping[str, bytes]
) -> tuple[ResourceRequirementV1, ...]:
    return tuple(
        ResourceRequirementV1(
            resource_id="advisory_kernel_v1",
            name=name,
            bytes_hash=_digest(member_bytes[name]),
        )
        for name in names
    )


def _validate_inventory(raw_resources: Mapping[str, bytes]) -> dict[str, bytes]:
    composites = {
        resource_id: decode_composite_resource(raw_resources[resource_id])
        for resource_id in (
            "advisory_kernel_v1",
            "advisory_native_v1",
            "advisory_operator_v1",
            "advisory_representation_v1",
        )
    }
    owned_members = {
        (resource_id, member.name): member.raw_bytes
        for resource_id, composite in composites.items()
        for member in composite.members
    }
    expected_bindings = set(_MODULE_ALLOCATION)
    actual_bindings = set(owned_members)
    for name, resource_id in _MODULE_ALLOCATION:
        if resource_id.startswith("compiler_"):
            actual_bindings.add((resource_id, name))
    if actual_bindings != {(resource, name) for name, resource in expected_bindings}:
        raise ValueError("packaged module ownership differs from the fixed allocation")

    allocations = []
    member_bytes: dict[str, bytes] = {}
    for name, resource_id in _MODULE_ALLOCATION:
        raw = (
            raw_resources[resource_id]
            if resource_id.startswith("compiler_")
            else owned_members[(resource_id, name)]
        )
        allocations.append(
            ModuleAllocationV1(name=name, resource_id=resource_id, raw_bytes=raw)
        )
        member_bytes[name] = raw
    validate_resource_inventory(
        resources=tuple(
            ResourceBytesV1(resource_id=resource_id, raw_bytes=raw)
            for resource_id, raw in raw_resources.items()
        ),
        allocations=tuple(allocations),
    )

    representation_requirement = (
        ResourceRequirementV1(
            resource_id="advisory_representation_v1",
            name="services._runtime_context_advisory.representation",
            bytes_hash=_digest(
                member_bytes["services._runtime_context_advisory.representation"]
            ),
        ),
    )
    expected = {
        "advisory_kernel_v1": representation_requirement,
        "advisory_native_v1": _expected_requirements(
            _NATIVE_REQUIREMENTS, member_bytes
        ),
        "advisory_operator_v1": _expected_requirements(
            _PRODUCT_REQUIREMENTS, member_bytes
        ),
        "advisory_representation_v1": _expected_requirements(
            _PRODUCT_REQUIREMENTS, member_bytes
        ),
    }
    for resource_id, requirements in expected.items():
        if composites[resource_id].requires != requirements:
            raise ValueError("composite requirements differ from the fixed release")
    return member_bytes


def _authenticate_details(
    *,
    package: RetainedVerifierPackageV1,
    registry_pin: ArtifactPinV1,
    trusted: TrustedRetainedVerifierRegistrationV1,
) -> tuple[
    PackagedVerifierRegistryV1,
    dict[str, bytes],
    dict[str, Callable[..., object]],
    dict[str, bytes],
]:
    if (
        type(package) is not RetainedVerifierPackageV1
        or type(registry_pin) is not ArtifactPinV1
        or type(trusted) is not TrustedRetainedVerifierRegistrationV1
    ):
        raise ValueError("package authentication requires exact typed trusted inputs")
    if (
        type(package.registry) is not CapturedJsonV1
        or type(package.registry.canonical_json) is not str
        or len(package.registry.canonical_json) > _MAX_TOTAL_BYTES
    ):
        raise ValueError("retained registry text exceeds its typed byte bound")
    registry_bytes = package.registry.canonical_json.encode("utf-8")
    _enforce_total_bytes((len(registry_bytes),))
    if (
        type(trusted.registry_bytes) is not bytes
        or len(trusted.registry_bytes) > _MAX_TOTAL_BYTES
        or registry_bytes != trusted.registry_bytes
    ):
        raise ValueError("registry bytes do not match trusted registration")
    try:
        registry_pin = ArtifactPinV1.model_validate(
            registry_pin.model_dump(mode="json")
        )
        registry = package.parsed_registry()
        captured_registry = CapturedJsonV1.model_validate(
            package.registry.model_dump(mode="json")
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("retained verifier registry is invalid") from exc

    if (
        captured_registry != package.registry
        or package.registry.bytes_hash != _digest(registry_bytes)
        or canonical_json(registry.model_dump(mode="json")).encode("utf-8")
        != registry_bytes
    ):
        raise ValueError("registry bytes are not the exact normalized canonical value")
    if (
        packaged_verifier_registry_pin(registry) != registry_pin
        or type(trusted.registry_pin) is not ArtifactPinV1
        or trusted.registry_pin != registry_pin
    ):
        raise ValueError("registry bytes and pin do not match trusted registration")
    limits = registry.limits
    if (
        registry.registry_version != _REGISTRY_VERSION
        or registry.kernel_id != _KERNEL_ID
        or registry.kernel_resource != "advisory_kernel_v1"
        or limits.max_capture_nodes != _MAX_CAPTURE_NODES
        or limits.max_dependency_depth != _MAX_DEPENDENCY_DEPTH
        or limits.max_total_bytes != _MAX_TOTAL_BYTES
        or registry.entries != _expected_entries()
    ):
        raise ValueError("registry differs from the fixed advisory release")
    manifests = {manifest.resource_id: manifest for manifest in registry.resources}
    if set(manifests) != set(_RESOURCE_MEDIA_TYPES) or any(
        manifest.media_type != _RESOURCE_MEDIA_TYPES[resource_id]
        for resource_id, manifest in manifests.items()
    ):
        raise ValueError("resource manifest differs from the fixed advisory release")

    resource_hashes = _strict_trusted_rows(
        trusted.resource_hashes,
        tuple(sorted(_RESOURCE_MEDIA_TYPES)),
        label="resource hashes",
    )
    verifier_values = _strict_trusted_rows(
        trusted.verifiers,
        tuple(entry.entry_id for entry in registry.entries),
        label="verifiers",
        callable_values=True,
    )
    declared_lengths = tuple(manifest.byte_length for manifest in registry.resources)
    _enforce_total_bytes((len(registry_bytes), *declared_lengths))
    if type(package.resources) is not tuple:
        raise ValueError("package resources must be an exact immutable tuple")
    resource_ids: list[str] = []
    for resource in package.resources:
        if (
            type(resource) is not RetainedPackageResourceV1
            or type(resource.resource_id) is not str
            or type(resource.bytes_base64) is not str
        ):
            raise ValueError("package resource envelope is not exact typed data")
        resource_ids.append(resource.resource_id)
    if tuple(resource_ids) != tuple(sorted(manifests)):
        raise ValueError("package resources must match the manifest exactly")
    package_resources = {
        resource.resource_id: resource for resource in package.resources
    }
    for resource_id, manifest in manifests.items():
        encoded_length = len(package_resources[resource_id].bytes_base64)
        expected_length = 4 * ((manifest.byte_length + 2) // 3)
        if encoded_length != expected_length:
            raise ValueError(
                "retained resource encoded body violates its declared bound"
            )
    try:
        validated_package = RetainedVerifierPackageV1.model_validate(
            package.model_dump(mode="json")
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("retained verifier package is invalid") from exc
    if validated_package != package:
        raise ValueError("retained verifier package is not exact normalized typed data")
    package_resources = {
        resource.resource_id: resource for resource in validated_package.resources
    }
    raw_resources: dict[str, bytes] = {}
    for resource_id, manifest in manifests.items():
        resource = package_resources[resource_id]
        raw = _strict_base64(resource.bytes_base64, f"retained resource {resource_id}")
        digest = _digest(raw)
        if (
            len(raw) != manifest.byte_length
            or digest != manifest.bytes_hash
            or digest != resource_hashes[resource_id]
        ):
            raise ValueError("resource bytes do not match manifest and trusted release")
        raw_resources[resource_id] = raw
    member_bytes = _validate_inventory(raw_resources)
    return (
        registry,
        raw_resources,
        {key: value for key, value in verifier_values.items()},  # type: ignore[misc]
        member_bytes,
    )


def _authenticate_advisory_package(
    *,
    package: RetainedVerifierPackageV1,
    registry_pin: ArtifactPinV1,
    trusted: TrustedRetainedVerifierRegistrationV1,
) -> PackagedVerifierRegistryV1:
    registry, _resources, _verifiers, _members = _authenticate_details(
        package=package, registry_pin=registry_pin, trusted=trusted
    )
    return registry


def _payload(source: CapturedSourceV1) -> dict[str, object]:
    try:
        validated = CapturedSourceV1.model_validate(source.model_dump(mode="json"))
        value = json.loads(validated.payload.canonical_json)
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("capture must contain exact canonical JSON") from exc
    if validated != source or not isinstance(value, dict):
        raise ValueError("capture must contain one exact canonical JSON object")
    return value


def _strict_base64(value: object, label: str) -> bytes:
    if type(value) is not str:
        raise ValueError(f"{label} must be strict canonical base64")
    try:
        raw = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"{label} must be strict canonical base64") from exc
    if base64.b64encode(raw).decode("ascii") != value:
        raise ValueError(f"{label} must be strict canonical base64")
    return raw


def _validate_identity(
    source: CapturedSourceV1,
    entry: PackagedVerifierEntryV1,
    payload: Mapping[str, object],
) -> None:
    if entry.verification_mode == "native_owner":
        return
    if entry.verification_mode == "full_canonical_json":
        if source.source.artifact_hash != source.payload.bytes_hash:
            raise ValueError("full canonical JSON identity does not bind payload bytes")
        return
    assert entry.ref_field is not None
    assert entry.hash_field is not None
    assert entry.ref_prefix is not None
    if set((entry.ref_field, entry.hash_field)) - set(payload):
        raise ValueError("self-bound envelope identity fields are missing")
    basis = {
        key: value
        for key, value in payload.items()
        if key not in (entry.ref_field, entry.hash_field)
    }
    digest = content_hash(basis)
    expected_ref = f"{entry.ref_prefix}:{digest}"
    expected_hash = f"sha256:{digest}"
    if (
        payload[entry.ref_field] != expected_ref
        or payload[entry.hash_field] != expected_hash
        or source.source.artifact_ref != expected_ref
        or source.source.artifact_hash != expected_hash
    ):
        raise ValueError("self-bound envelope identity does not bind exact content")


def _validate_selector(
    *,
    source: CapturedSourceV1,
    payload: Mapping[str, object],
    registry_pin: ArtifactPinV1,
    entries: Mapping[str, PackagedVerifierEntryV1],
) -> PackagedVerifierEntryV1:
    entry = entries.get(source.verifier.entry_id)
    if (
        entry is None
        or source.verifier.registry != registry_pin
        or entry.entry_id != source.source.artifact_kind
        or entry.artifact_kind != source.source.artifact_kind
    ):
        raise ValueError("capture selector does not name its exact registry entry")
    field = _SCHEMA_FIELDS.get(entry.entry_id, "schema_version")
    if field is None:
        if {"schema_version", "schemaVersion", "canonicalIrVersion"} & set(payload):
            raise ValueError("selector-only capture must not invent a schema field")
    elif payload.get(field) not in entry.schema_versions:
        raise ValueError("capture schema does not match the selected registry entry")
    return entry


def _validate_package_leaf(
    *,
    source: CapturedSourceV1,
    payload: Mapping[str, object],
    profile: ArtifactPinV1,
    raw_resources: Mapping[str, bytes],
    member_bytes: Mapping[str, bytes],
) -> None:
    kind = source.source.artifact_kind
    if kind == "advisory_context_profile":
        if (
            source.source != profile
            or source.payload.canonical_json.encode("utf-8")
            != raw_resources["advisory_profile_v1"]
        ):
            raise ValueError("profile capture differs from the authenticated profile")
    elif kind == "advisory_context_operator":
        expected = {
            "schema_version": "advisory_context_operator.v1",
            "operator_id": "context_only_empirical_v1",
            "implementation_resource_id": "advisory_operator_v1",
            "implementation_hash": _digest(raw_resources["advisory_operator_v1"]),
            "profile": profile.model_dump(mode="json"),
        }
        if payload != expected or source.dependencies != (profile,):
            raise ValueError(
                "operator descriptor differs from authenticated product values"
            )
    elif kind == "advisory_compiler_inputs":
        resources = payload.get("compiler_resources")
        if not isinstance(resources, list) or len(resources) != len(_COMPILER_ROLES):
            raise ValueError("compiler role resources are incomplete")
        for value, (role, (resource_id, resource_ref)) in zip(
            resources, _COMPILER_ROLES.items(), strict=True
        ):
            if not isinstance(value, dict):
                raise ValueError("compiler role resource must be an object")
            raw = _strict_base64(value.get("bytes_base64"), f"compiler {role} bytes")
            if (
                value.get("role") != role
                or value.get("resource_ref") != resource_ref
                or value.get("bytes_hash") != _digest(raw_resources[resource_id])
                or raw != raw_resources[resource_id]
            ):
                raise ValueError(
                    "compiler role does not equal authenticated package bytes"
                )
    elif kind == "advisory_trial_build_inputs":
        if payload.get("producer_hash") != _digest(
            member_bytes["services.runtime_learning_trial_service"]
        ):
            raise ValueError(
                "trial build producer differs from authenticated member bytes"
            )


def _preflight_outcomes(
    outcomes: object,
) -> tuple[tuple[ArtifactPinV1, ArtifactPinV1], ...]:
    if type(outcomes) is not tuple or not 1 <= len(outcomes) <= _MAX_PAIRS:
        raise ValueError("outcomes must contain one to eight ordered pairs")
    result: list[tuple[ArtifactPinV1, ArtifactPinV1]] = []
    pairs: set[tuple[tuple[str, str, str], tuple[str, str, str]]] = set()
    episodes: set[tuple[str, str, str]] = set()
    for pair in outcomes:
        if type(pair) is not tuple or len(pair) != 2:
            raise ValueError("each outcome must be an exact ordered pair")
        episode, assessment = pair
        if type(episode) is not ArtifactPinV1 or type(assessment) is not ArtifactPinV1:
            raise ValueError("outcome members must be exact artifact pins")
        if (
            episode.artifact_kind != "empirical_episode"
            or assessment.artifact_kind != "empirical_episode_assessment"
        ):
            raise ValueError("outcome pair kinds are invalid")
        key = (_pin_key(episode), _pin_key(assessment))
        if key in pairs or key[0] in episodes:
            raise ValueError("outcome pairs contain a duplicate episode selection")
        pairs.add(key)
        episodes.add(key[0])
        result.append((episode, assessment))
    return tuple(result)


def _capture_closure(sources: object) -> dict[tuple[str, str, str], CapturedSourceV1]:
    if type(sources) is not tuple or len(sources) > _MAX_CAPTURE_NODES:
        raise ValueError("sources exceed the 2048 capture-node bound")
    keys: list[tuple[str, str, str]] = []
    refs: set[tuple[str, str]] = set()
    closure: dict[tuple[str, str, str], CapturedSourceV1] = {}
    for source in sources:
        if type(source) is not CapturedSourceV1:
            raise ValueError("sources must contain exact captured source values")
        key = _pin_key(source.source)
        if key in closure or key[:2] in refs:
            raise ValueError("sources contain duplicate or conflicting identities")
        if any(
            pin.artifact_kind
            in {"frozen_context_formation", "runtime_context_formation_command"}
            for pin in (source.source, *source.dependencies)
        ):
            raise ValueError("capture closure contains a forward formation reference")
        closure[key] = source
        keys.append(key)
        refs.add(key[:2])
    if keys != sorted(
        keys, key=lambda key: tuple(item.encode("utf-8") for item in key)
    ):
        raise ValueError("sources must use UTF-8 kind/ref/hash order")
    return closure


def _authenticated_profile_pin(
    raw_resources: Mapping[str, bytes],
) -> ArtifactPinV1:
    raw = raw_resources["advisory_profile_v1"]
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "authenticated profile resource must be canonical JSON"
        ) from exc
    if not isinstance(value, dict) or canonical_json(value).encode("utf-8") != raw:
        raise ValueError("authenticated profile resource must be canonical JSON")
    digest = sha256(raw).hexdigest()
    return ArtifactPinV1(
        artifact_kind="advisory_context_profile",
        artifact_ref=f"advisory-context-profile:{digest}",
        artifact_hash=f"sha256:{digest}",
    )


def _verify_native_closure(
    *,
    roots: tuple[ArtifactPinV1, ...],
    closure: Mapping[tuple[str, str, str], CapturedSourceV1],
    registry_pin: ArtifactPinV1,
    registry: PackagedVerifierRegistryV1,
    raw_resources: Mapping[str, bytes],
    verifiers: Mapping[str, Callable[..., object]],
    member_bytes: Mapping[str, bytes],
) -> None:
    entries = {entry.entry_id: entry for entry in registry.entries}
    profile = _authenticated_profile_pin(raw_resources)
    validated_entries: dict[tuple[str, str, str], PackagedVerifierEntryV1] = {}
    for key, source in closure.items():
        payload = _payload(source)
        entry = _validate_selector(
            source=source,
            payload=payload,
            registry_pin=registry_pin,
            entries=entries,
        )
        _validate_package_leaf(
            source=source,
            payload=payload,
            profile=profile,
            raw_resources=raw_resources,
            member_bytes=member_bytes,
        )
        _validate_identity(source, entry, payload)
        validated_entries[key] = entry

    visiting: set[tuple[str, str, str]] = set()
    visited: set[tuple[str, str, str]] = set()

    def visit(pin: ArtifactPinV1, depth: int) -> None:
        key = _pin_key(pin)
        if depth > _MAX_DEPENDENCY_DEPTH:
            raise ValueError("capture dependency depth exceeds 32")
        if key in visiting:
            raise ValueError("capture dependency graph contains a cycle")
        if key in visited:
            return
        source = closure.get(key)
        if source is None or source.source != pin:
            raise ValueError("capture closure is missing an exact dependency")
        entry = validated_entries[key]
        visiting.add(key)
        try:
            returned = verifiers[entry.entry_id](source=source, closure=closure)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"native verifier rejected {entry.entry_id}") from exc
        if type(returned) is not tuple or any(
            type(dependency) is not ArtifactPinV1 for dependency in returned
        ):
            raise ValueError("native verifier returned malformed dependencies")
        if returned != source.dependencies:
            raise ValueError("declared dependencies differ from exact native returns")
        for dependency in returned:
            visit(dependency, depth + 1)
        visiting.remove(key)
        visited.add(key)

    for root in roots:
        visit(root, 1)
    if visited != set(closure):
        raise ValueError("capture closure contains unreachable extras")


def _verify_advisory_native_capture_v1(
    *,
    root: ArtifactPinV1,
    verifier_registry: ArtifactPinV1,
    sources: tuple[CapturedSourceV1, ...],
    verifier_package: RetainedVerifierPackageV1,
    trusted: TrustedRetainedVerifierRegistrationV1,
) -> None:
    if (
        type(root) is not ArtifactPinV1
        or type(verifier_registry) is not ArtifactPinV1
        or verifier_registry.artifact_kind != "packaged_verifier_registry"
    ):
        raise ValueError("native root and registry must use exact typed values")
    closure = _capture_closure(sources)
    registry, raw_resources, verifiers, member_bytes = _authenticate_details(
        package=verifier_package,
        registry_pin=verifier_registry,
        trusted=trusted,
    )
    _enforce_total_bytes(
        (
            len(verifier_package.registry.canonical_json.encode("utf-8")),
            *(len(raw) for raw in raw_resources.values()),
            *(len(source.payload.canonical_json.encode("utf-8")) for source in sources),
        )
    )
    _verify_native_closure(
        roots=(root,),
        closure=closure,
        registry_pin=verifier_registry,
        registry=registry,
        raw_resources=raw_resources,
        verifiers=verifiers,
        member_bytes=member_bytes,
    )


def verify_advisory_capture_v1(
    *,
    outcomes: tuple[tuple[ArtifactPinV1, ArtifactPinV1], ...],
    profile: ArtifactPinV1,
    operator: ArtifactPinV1,
    verifier_registry: ArtifactPinV1,
    sources: tuple[CapturedSourceV1, ...],
    representation: CapturedJsonV1,
    verifier_package: RetainedVerifierPackageV1,
    trusted: TrustedRetainedVerifierRegistrationV1,
) -> None:
    outcomes = _preflight_outcomes(outcomes)
    if (
        type(profile) is not ArtifactPinV1
        or profile.artifact_kind != "advisory_context_profile"
        or type(operator) is not ArtifactPinV1
        or operator.artifact_kind != "advisory_context_operator"
        or type(verifier_registry) is not ArtifactPinV1
        or verifier_registry.artifact_kind != "packaged_verifier_registry"
        or type(representation) is not CapturedJsonV1
    ):
        raise ValueError("kernel roots and representation must use exact typed values")
    closure = _capture_closure(sources)
    registry, raw_resources, verifiers, member_bytes = _authenticate_details(
        package=verifier_package,
        registry_pin=verifier_registry,
        trusted=trusted,
    )
    try:
        representation = CapturedJsonV1.model_validate(
            representation.model_dump(mode="json")
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(
            "representation must be exact captured canonical JSON"
        ) from exc
    _enforce_total_bytes(
        (
            len(verifier_package.registry.canonical_json.encode("utf-8")),
            *(len(raw) for raw in raw_resources.values()),
            *(len(source.payload.canonical_json.encode("utf-8")) for source in sources),
            len(representation.canonical_json.encode("utf-8")),
        )
    )

    profile_source = closure.get(_pin_key(profile))
    if profile_source is None:
        raise ValueError("profile capture is missing")
    _payload(profile_source)
    if (
        profile_source.payload.canonical_json.encode("utf-8")
        != raw_resources["advisory_profile_v1"]
    ):
        raise ValueError("profile capture differs from the authenticated profile")
    roots = tuple(pin for pair in outcomes for pin in pair) + (profile, operator)
    _verify_native_closure(
        roots=roots,
        closure=closure,
        registry_pin=verifier_registry,
        registry=registry,
        raw_resources=raw_resources,
        verifiers=verifiers,
        member_bytes=member_bytes,
    )
    verify_advisory_episode_representation_v1(
        outcomes=outcomes,
        closure=closure,
        representation=representation,
    )
