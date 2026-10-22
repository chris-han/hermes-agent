"""Pure retained verification for the initial advisory compiler family."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import json
from typing import Any, Mapping

from contracts.canonical_source_ir import content_hash as canonical_ir_content_hash
from contracts.graph_intent_brief import GraphIntentBrief
from contracts.normative_semantics import NormativeSemanticCandidateSetV1
from contracts.ontology_building import (
    ArtifactPinV1,
    CanonicalEvidenceBoundaryV1,
    CompetencyContractV1,
)
from contracts.ontology_validation import CompiledRulePlanV1
from contracts.runtime_context_reconstruction import CapturedJsonV1, CapturedSourceV1
from contracts.runtime_learning_trial import (
    AdvisoryCompilerInputsV1,
    AdvisoryTrialVariantV1,
)
from contracts.validation_contract import ValidationContractV1
from eos.artifact_hashing import canonical_json, content_hash

from .common import (
    PinKey,
    _expect_dependencies,
    _expect_identity,
    _payload,
    _pin_key,
    _require_supplied_source,
    _strict_base64,
    _validate_closure,
)
from .source import (
    _canonical_source_ir_ref,
    _source_member_value,
    _source_member_view,
)


_RESOURCE_REFS = {
    "semantic_compiler": "repo:src/services/normative_semantic_compiler.py",
    "validation_compiler": "repo:src/services/external_validation_compiler.py",
    "rule_plan_compiler": "repo:src/services/ontology_rule_plan_compiler.py",
    "prompt": "repo:src/prompts/normative_semantic_compilation_v1.txt",
    "configuration": (
        "repo:src/services/runtime_context_advisory_assets/"
        "compiler_configuration_v1.json"
    ),
    "frame_schema": (
        "repo:src/services/runtime_context_advisory_assets/"
        "compiler_frame_schema_v1.json"
    ),
    "profile": (
        "repo:src/services/runtime_context_advisory_assets/compiler_profile_v1.json"
    ),
}

_FRAME_SCHEMA_MAPPING = {
    "entity_distinction": "#/$defs/EntityDistinctionFrameV1",
    "exception": "#/$defs/ExceptionFrameV1",
    "precedence": "#/$defs/PrecedenceFrameV1",
    "qualification": "#/$defs/QualificationFrameV1",
    "rule": "#/$defs/RuleFrameV1",
    "scope": "#/$defs/ScopeFrameV1",
    "temporal_condition": "#/$defs/TemporalConditionFrameV1",
}

_FRAME_COMMON_NATIVE_KEYS = (
    "frame_id",
    "display_name",
    "evidence_refs",
    "source_region_refs",
)
_FRAME_NATIVE_KEYS = {
    "entity_distinction": (
        "left_entity_ref",
        "right_entity_ref",
        "outcome_effect",
    ),
    "exception": ("applies_to_rule_ref", "conditions"),
    "precedence": (
        "higher_priority_ref",
        "lower_priority_ref",
        "condition_refs",
    ),
    "qualification": ("qualified_term", "definition"),
    "rule": (
        "modality",
        "actor_ref",
        "object_ref",
        "conditions",
        "consequence",
        "exception_refs",
        "scope_refs",
        "qualification_refs",
        "temporal_condition_refs",
        "precedence_refs",
    ),
    "scope": ("applies_to_rule_ref", "actor_scope", "object_scope"),
    "temporal_condition": (
        "effective_from",
        "effective_until",
        "event_time_ref",
    ),
}
_INITIAL_FRAME_KINDS = (
    "entity_distinction",
    "exception",
    "precedence",
    "qualification",
    "rule",
    "scope",
    "temporal_condition",
)
_CANDIDATE_NATIVE_KEYS = (
    "organization_id",
    "workspace_id",
    "source_identity_ref",
    "source_hash",
    "canonical_source_ir_ref",
    "canonical_source_ir_hash",
    "competency_contract_ref",
    "competency_contract_hash",
    "normative_schema_ref",
    "normative_schema_hash",
    "profile_ref",
    "profile_hash",
    "compiler_ref",
    "compiler_hash",
    "prompt_hash",
    "provider_ref",
    "model_ref",
    "config_hash",
)


@dataclass(frozen=True, slots=True)
class _InputContext:
    inputs: AdvisoryCompilerInputsV1
    retained_json: Any
    source_object: dict[str, Any]
    boundary: CanonicalEvidenceBoundaryV1
    competency: CompetencyContractV1
    profile: dict[str, Any]
    dependencies: tuple[ArtifactPinV1, ...]


def _model_payload(source: CapturedSourceV1, contract: type[Any], label: str) -> Any:
    _source, payload = _payload(source)
    value = contract.model_validate(payload)
    if value.model_dump(mode="json") != payload:
        raise ValueError(f"retained {label} encoding mismatch")
    return value


def _resolve(
    closure: Mapping[PinKey, CapturedSourceV1], pin: ArtifactPinV1
) -> CapturedSourceV1:
    resolved = closure.get((pin.artifact_kind, pin.artifact_ref, pin.artifact_hash))
    if resolved is None:
        raise ValueError("retained compiler dependency is missing")
    return resolved


def _boundary_value(source: CapturedSourceV1) -> CanonicalEvidenceBoundaryV1:
    boundary = _model_payload(source, CanonicalEvidenceBoundaryV1, "compiler boundary")
    _expect_identity(
        source,
        kind="canonical_evidence_boundary",
        ref=boundary.boundary_ref,
        digest=boundary.boundary_hash,
    )
    return boundary


def _validate_boundary_native_value(
    boundary: CanonicalEvidenceBoundaryV1,
) -> None:
    basis = {
        "organization_id": boundary.organization_id,
        "workspace_id": boundary.workspace_id,
        "source_identity_ref": boundary.source_identity_ref,
        "source_hash": boundary.source_hash,
        "canonical_source_ir_ref": boundary.canonical_source_ir_ref,
        "canonical_source_ir_hash": boundary.canonical_source_ir_hash,
        "intended_domain": boundary.intended_domain,
        "intended_use": boundary.intended_use,
        "target_time": boundary.target_time.isoformat(),
        "jurisdiction": boundary.jurisdiction,
        "explicit_exclusions": boundary.explicit_exclusions,
    }
    digest = content_hash(basis)
    if (
        boundary.boundary_ref != f"canonical-boundary:{digest}"
        or boundary.boundary_hash != f"sha256:{digest}"
        or boundary.target_time != boundary.created_at
        or boundary.jurisdiction != "unresolved"
        or boundary.explicit_exclusions != ("certification", "admission", "activation")
    ):
        raise ValueError("canonical evidence boundary native producer mismatch")


def _source_object_for_graph_intent(
    graph_source: CapturedSourceV1,
    closure: Mapping[PinKey, CapturedSourceV1],
) -> tuple[GraphIntentBrief, dict[str, Any]]:
    _graph_source, graph_payload = _payload(graph_source)
    graph_intent = _accepted_graph_intent_payload(graph_payload)
    _expect_identity(
        graph_source,
        kind="graph_intent_brief",
        ref=graph_intent.graph_intent_id,
        digest=graph_intent.graph_intent_hash,
    )
    source_pins = tuple(
        pin
        for pin in graph_source.dependencies
        if pin.artifact_kind == "advisory_source_capture"
    )
    if len(source_pins) != 1:
        raise ValueError("GraphIntent must declare exactly one source capture")
    source_capture = _resolve(closure, source_pins[0])
    view = _source_member_view(source_capture, closure)
    return graph_intent, _source_object_for_graph_intent_value(graph_intent, view)


def _accepted_graph_intent_payload(payload: Any) -> GraphIntentBrief:
    if not isinstance(payload, dict):
        raise ValueError("retained competency GraphIntent must be an object")
    graph_intent = GraphIntentBrief.from_dict(payload)
    if graph_intent.as_dict() != payload or graph_intent.review.state != "accepted":
        raise ValueError("initial competency requires an accepted native GraphIntent")
    return graph_intent


def _accepted_graph_intent_value(brief: GraphIntentBrief) -> GraphIntentBrief:
    if not isinstance(brief, GraphIntentBrief):
        raise ValueError("initial competency requires an accepted native GraphIntent")
    return _accepted_graph_intent_payload(brief.as_dict())


def _source_object_for_graph_intent_value(
    graph_intent: GraphIntentBrief,
    view: Any,
) -> dict[str, Any]:
    raw = _strict_base64(view.payload["source_bytes_base64"], "compiler source bytes")
    try:
        source_object = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("compiler source bytes must contain canonical JSON") from exc
    if not isinstance(source_object, dict):
        raise ValueError("compiler source must be one JSON object")
    binding = graph_intent.source_binding
    if (
        binding.organization_id != view.identity.organization_id
        or binding.workspace_id != view.identity.workspace_id
        or binding.source_identity_ref != view.identity.source_identity_ref
        or binding.source_hash != view.identity.source_hash
        or view.canonical.source_identity_ref != view.identity.source_identity_ref
    ):
        raise ValueError("GraphIntent compiler source binding mismatch")
    return source_object


def _competency_value(
    source: CapturedSourceV1,
    closure: Mapping[PinKey, CapturedSourceV1],
) -> tuple[CompetencyContractV1, tuple[ArtifactPinV1, ...]]:
    competency = _model_payload(source, CompetencyContractV1, "competency contract")
    _expect_identity(
        source,
        kind="competency_contract",
        ref=competency.competency_contract_ref,
        digest=competency.competency_contract_hash,
    )
    boundary_pin = ArtifactPinV1(
        artifact_kind="canonical_evidence_boundary",
        artifact_ref=competency.boundary_ref,
        artifact_hash=competency.boundary_hash,
    )
    graph_pin = ArtifactPinV1(
        artifact_kind="graph_intent_brief",
        artifact_ref=competency.graph_intent_brief_ref,
        artifact_hash=competency.graph_intent_brief_hash,
    )
    dependencies = _expect_dependencies(source, (boundary_pin, graph_pin), closure)
    boundary = _boundary_value(_resolve(closure, boundary_pin))
    graph_intent, source_object = _source_object_for_graph_intent(
        _resolve(closure, graph_pin), closure
    )
    _validate_competency_native_value(
        competency=competency,
        boundary=boundary,
        graph_intent=graph_intent,
        source_object=source_object,
    )
    return competency, dependencies


def _validate_competency_native_value(
    *,
    competency: CompetencyContractV1,
    boundary: CanonicalEvidenceBoundaryV1,
    graph_intent: GraphIntentBrief,
    source_object: dict[str, Any],
) -> None:
    binding = graph_intent.source_binding
    if (
        (competency.organization_id, competency.workspace_id)
        != (boundary.organization_id, boundary.workspace_id)
        or (competency.organization_id, competency.workspace_id)
        != (binding.organization_id, binding.workspace_id)
        or competency.intended_domain != boundary.intended_domain
        or competency.in_scope != (boundary.intended_use,)
        or competency.out_of_scope != boundary.explicit_exclusions
        or boundary.source_identity_ref != binding.source_identity_ref
        or boundary.source_hash != binding.source_hash
        or boundary.canonical_source_ir_ref != binding.canonical_source_ir_ref
        or boundary.canonical_source_ir_hash != binding.canonical_source_ir_hash
    ):
        raise ValueError("initial competency scope or source binding mismatch")

    raw_cases = source_object.get("competency_oracle")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("initial competency source oracle is missing")
    expected_questions = []
    for case in raw_cases:
        if not isinstance(case, dict):
            raise ValueError("initial competency oracle member must be an object")
        expected_questions.append(
            {
                "competency_id": str(case["competency_id"]),
                "question": str(case["question"]),
                "criticality": "proposed_critical",
                "review_state": "unresolved",
                "required_semantic_dimensions": list(
                    case["required_semantic_dimensions"]
                ),
                "acceptance_obligations": [str(case["expected_result"])],
                "evidence_refs": list(case["evidence_refs"]),
                "provenance_kind": "source_derived",
            }
        )
    if [
        item.model_dump(mode="json") for item in competency.questions
    ] != expected_questions:
        raise ValueError("initial competency questions do not match source oracle")
    if (
        competency.predecessor_ref is not None
        or competency.predecessor_hash is not None
    ):
        raise ValueError("initial competency cannot be a successor")

    _validate_competency_identity(competency)


def _validate_competency_identity(competency: CompetencyContractV1) -> None:
    basis = {
        "organization_id": competency.organization_id,
        "workspace_id": competency.workspace_id,
        "boundary_ref": competency.boundary_ref,
        "boundary_hash": competency.boundary_hash,
        "graph_intent_brief_ref": competency.graph_intent_brief_ref,
        "graph_intent_brief_hash": competency.graph_intent_brief_hash,
        "intended_domain": competency.intended_domain,
        "questions": [item.model_dump(mode="json") for item in competency.questions],
    }
    digest = content_hash(basis)
    if (
        competency.competency_contract_ref != f"competency-contract:{digest}"
        or competency.competency_contract_hash != f"sha256:{digest}"
    ):
        raise ValueError("competency contract does not match initial native identity")


def _validate_live_competency_relationships(
    *,
    competency: CompetencyContractV1,
    boundary: CanonicalEvidenceBoundaryV1,
    graph_intent: GraphIntentBrief,
) -> None:
    if (
        (competency.boundary_ref, competency.boundary_hash)
        != (boundary.boundary_ref, boundary.boundary_hash)
        or (
            competency.graph_intent_brief_ref,
            competency.graph_intent_brief_hash,
        )
        != (graph_intent.graph_intent_id, graph_intent.graph_intent_hash)
        or boundary.intended_domain
        != graph_intent.represented_or_governed_subject.description
        or boundary.intended_use != graph_intent.document_function.description
    ):
        raise ValueError("initial competency scope or source binding mismatch")


def _json_pointer(value: Any, pointer: str) -> Any:
    if pointer == "":
        return value
    current = value
    for encoded in pointer[1:].split("/"):
        token = encoded.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            try:
                current = current[int(token)]
            except (ValueError, IndexError) as exc:
                raise ValueError("compiler JSON pointer does not resolve") from exc
        elif isinstance(current, dict) and token in current:
            current = current[token]
        else:
            raise ValueError("compiler JSON pointer does not resolve")
    return current


def _validate_frame_schema(raw: bytes) -> None:
    try:
        schema = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("compiler frame schema must be canonical JSON") from exc
    if canonical_json(schema).encode("utf-8") != raw or not isinstance(schema, dict):
        raise ValueError("compiler frame schema must be canonical JSON")
    definitions = schema.get("$defs")
    discriminator = schema.get("discriminator")
    if (
        set(schema) != {"$defs", "discriminator", "oneOf"}
        or not isinstance(definitions, dict)
        or set(definitions)
        != {
            *(
                target.removeprefix("#/$defs/")
                for target in _FRAME_SCHEMA_MAPPING.values()
            ),
            "SemanticPredicateV1",
        }
        or discriminator
        != {"mapping": _FRAME_SCHEMA_MAPPING, "propertyName": "frame_kind"}
        or schema["oneOf"]
        != [
            {"$ref": f"#/$defs/{name}"}
            for name in (
                "RuleFrameV1",
                "ExceptionFrameV1",
                "ScopeFrameV1",
                "QualificationFrameV1",
                "PrecedenceFrameV1",
                "TemporalConditionFrameV1",
                "EntityDistinctionFrameV1",
            )
        ]
    ):
        raise ValueError(
            "compiler frame schema does not describe the native frame union"
        )
    for kind, target in _FRAME_SCHEMA_MAPPING.items():
        frame = definitions[target.removeprefix("#/$defs/")]
        if not isinstance(frame, dict) or frame.get("properties", {}).get(
            "frame_kind"
        ) != {
            "const": kind,
            "default": kind,
            "title": "Frame Kind",
            "type": "string",
        }:
            raise ValueError("compiler frame schema discriminator member mismatch")


def _validate_resources(
    inputs: AdvisoryCompilerInputsV1, source_object: dict[str, Any]
) -> dict[str, Any]:
    resources = {item.role: item for item in inputs.compiler_resources}
    if tuple(resources) != tuple(_RESOURCE_REFS):
        raise ValueError("compiler resource roles are not in the native order")
    raw_by_role: dict[str, bytes] = {}
    for role, resource in resources.items():
        if resource.resource_ref != _RESOURCE_REFS[role]:
            raise ValueError("compiler resource ref does not match its native role")
        raw_by_role[role] = _strict_base64(
            resource.bytes_base64, f"compiler {role} bytes"
        )
    for role in ("semantic_compiler", "validation_compiler", "rule_plan_compiler"):
        try:
            text = raw_by_role[role].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("compiler module resource must be UTF-8") from exc
        if not text:
            raise ValueError("compiler module resource cannot be empty")
    try:
        prompt = raw_by_role["prompt"].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("compiler prompt resource must be UTF-8") from exc
    if not prompt:
        raise ValueError("compiler prompt resource cannot be empty")
    if raw_by_role["configuration"] != b"{}":
        raise ValueError("compiler configuration must be exact empty JSON")
    _validate_frame_schema(raw_by_role["frame_schema"])
    try:
        profile = json.loads(raw_by_role["profile"].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("compiler profile must be canonical JSON") from exc
    if (
        canonical_json(profile).encode("utf-8") != raw_by_role["profile"]
        or not isinstance(profile, dict)
        or set(profile)
        != {
            "profile_id",
            "schema_version",
            "semantic_profile",
            "source_fixture_id",
            "source_schema_version",
            "usage",
        }
        or profile["profile_id"] != "controlled_internal_policy_v1"
        or profile["schema_version"] != "advisory_native_compiler_profile.v1"
        or profile["semantic_profile"] != "semantier_deterministic_v1"
        or profile["usage"] != "CONTEXT_ONLY"
        or profile["source_fixture_id"] != source_object.get("fixture_id")
        or profile["source_schema_version"] != source_object.get("schema_version")
    ):
        raise ValueError("compiler profile does not match the retained source")
    return profile


def _input_value(
    source: CapturedSourceV1,
    closure: Mapping[PinKey, CapturedSourceV1],
) -> _InputContext:
    inputs = _model_payload(source, AdvisoryCompilerInputsV1, "compiler inputs")
    _expect_identity(
        source,
        kind="advisory_compiler_inputs",
        ref=inputs.input_ref,
        digest=inputs.input_hash,
    )
    dependencies = _expect_dependencies(
        source, (inputs.source, inputs.competency, inputs.boundary), closure
    )
    source_capture = _resolve(closure, inputs.source)
    view = _source_member_view(source_capture, closure)
    retained_json, source_object = _compiler_source_values(inputs, view)
    boundary = _boundary_value(_resolve(closure, inputs.boundary))
    competency, _competency_dependencies = _competency_value(
        _resolve(closure, inputs.competency), closure
    )
    profile = _validate_input_native_value(
        inputs=inputs,
        boundary=boundary,
        competency=competency,
        view=view,
        source_object=source_object,
    )
    return _InputContext(
        inputs=inputs,
        retained_json=retained_json,
        source_object=source_object,
        boundary=boundary,
        competency=competency,
        profile=profile,
        dependencies=dependencies,
    )


def _compiler_source_values(
    inputs: AdvisoryCompilerInputsV1,
    view: Any,
) -> tuple[Any, dict[str, Any]]:
    raw_source = _strict_base64(
        view.payload["source_bytes_base64"], "compiler source bytes"
    )
    try:
        retained_json = json.loads(raw_source.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("compiler source bytes must contain JSON") from exc
    source_object = _json_pointer(retained_json, inputs.source_object_locator)
    if not isinstance(source_object, dict):
        raise ValueError("compiler source object locator must select an object")
    oracle = _json_pointer(retained_json, inputs.oracle_locator)
    if not isinstance(oracle, list):
        raise ValueError("compiler oracle locator must select an array")
    return retained_json, source_object


def _validate_input_native_value(
    *,
    inputs: AdvisoryCompilerInputsV1,
    boundary: CanonicalEvidenceBoundaryV1,
    competency: CompetencyContractV1,
    view: Any,
    source_object: dict[str, Any],
) -> dict[str, Any]:
    canonical_payload = view.payload["canonical_source_ir"]
    canonical_hash = canonical_ir_content_hash(canonical_payload)
    if (
        (inputs.organization_id, inputs.workspace_id)
        != (view.identity.organization_id, view.identity.workspace_id)
        or (inputs.organization_id, inputs.workspace_id)
        != (boundary.organization_id, boundary.workspace_id)
        or (inputs.organization_id, inputs.workspace_id)
        != (competency.organization_id, competency.workspace_id)
        or boundary.source_identity_ref != view.identity.source_identity_ref
        or boundary.source_hash != view.identity.source_hash
        or boundary.canonical_source_ir_hash != canonical_hash
        or boundary.canonical_source_ir_ref != _canonical_source_ir_ref(canonical_hash)
        or inputs.boundary.artifact_ref != competency.boundary_ref
        or inputs.boundary.artifact_hash != competency.boundary_hash
    ):
        raise ValueError("compiler input source, boundary, or competency mismatch")
    return _validate_resources(inputs, source_object)


def _validate_live_input_relationships(
    *,
    inputs: AdvisoryCompilerInputsV1,
    boundary: CanonicalEvidenceBoundaryV1,
    competency: CompetencyContractV1,
    source_capture_pin: ArtifactPinV1,
) -> None:
    if (
        inputs.source != source_capture_pin
        or inputs.boundary
        != ArtifactPinV1(
            artifact_kind="canonical_evidence_boundary",
            artifact_ref=boundary.boundary_ref,
            artifact_hash=boundary.boundary_hash,
        )
        or inputs.competency
        != ArtifactPinV1(
            artifact_kind="competency_contract",
            artifact_ref=competency.competency_contract_ref,
            artifact_hash=competency.competency_contract_hash,
        )
    ):
        raise ValueError("compiler input source, boundary, or competency mismatch")


def _validated_live_value(value: Any, contract: type[Any], label: str) -> Any:
    if not isinstance(value, contract):
        raise ValueError(f"live {label} must be a typed native value")
    return contract.model_validate(value.model_dump(mode="json"))


def _validate_live_compiler_input_native_values(
    *,
    inputs: AdvisoryCompilerInputsV1,
    boundary: CanonicalEvidenceBoundaryV1,
    competency: CompetencyContractV1,
    brief: GraphIntentBrief,
    source_capture: CapturedJsonV1,
) -> None:
    inputs = _validated_live_value(inputs, AdvisoryCompilerInputsV1, "compiler inputs")
    boundary = _validated_live_value(
        boundary, CanonicalEvidenceBoundaryV1, "compiler boundary"
    )
    competency = _validated_live_value(
        competency, CompetencyContractV1, "competency contract"
    )
    source_capture = _validated_live_value(
        source_capture, CapturedJsonV1, "source capture"
    )

    view = _source_member_value(source_capture)
    _retained_json, source_object = _compiler_source_values(inputs, view)
    _validate_boundary_native_value(boundary)
    graph_intent = _accepted_graph_intent_value(brief)
    _source_object_for_graph_intent_value(graph_intent, view)
    _validate_live_competency_relationships(
        competency=competency,
        boundary=boundary,
        graph_intent=graph_intent,
    )
    _validate_competency_native_value(
        competency=competency,
        boundary=boundary,
        graph_intent=graph_intent,
        source_object=source_object,
    )
    source_capture_pin = ArtifactPinV1(
        artifact_kind="advisory_source_capture",
        artifact_ref=(
            "advisory-source-capture:"
            + source_capture.bytes_hash.removeprefix("sha256:")
        ),
        artifact_hash=source_capture.bytes_hash,
    )
    _validate_live_input_relationships(
        inputs=inputs,
        boundary=boundary,
        competency=competency,
        source_capture_pin=source_capture_pin,
    )
    _validate_input_native_value(
        inputs=inputs,
        boundary=boundary,
        competency=competency,
        view=view,
        source_object=source_object,
    )


def _initial_input_dependency(source: CapturedSourceV1) -> ArtifactPinV1:
    if (
        len(source.dependencies) != 1
        or source.dependencies[0].artifact_kind != "advisory_compiler_inputs"
    ):
        raise ValueError("initial compiler output must declare one exact input")
    return source.dependencies[0]


def _source_sections(source_object: dict[str, Any]) -> dict[str, str]:
    sections = source_object.get("source_sections")
    if not isinstance(sections, list) or not sections:
        raise ValueError("compiler source sections are missing")
    mapping: dict[str, str] = {}
    for section in sections:
        if (
            not isinstance(section, dict)
            or set(section) != {"section_id", "text", "evidence_ref"}
            or not all(isinstance(section[key], str) for key in section)
            or section["section_id"] in mapping
        ):
            raise ValueError("compiler source section member is invalid")
        mapping[section["section_id"]] = section["evidence_ref"]
    return mapping


def _validate_candidate_frames(
    candidate: NormativeSemanticCandidateSetV1,
    source_object: dict[str, Any],
) -> None:
    if tuple(frame.frame_kind for frame in candidate.frames) != _INITIAL_FRAME_KINDS:
        raise ValueError("initial candidate has an incompatible native frame profile")
    sections = _source_sections(source_object)
    for frame in candidate.frames:
        _validate_candidate_frame_native_value(frame)
        try:
            expected_evidence = tuple(
                sections[region] for region in frame.source_region_refs
            )
        except KeyError as exc:
            raise ValueError("candidate frame source region is not retained") from exc
        if frame.evidence_refs != expected_evidence:
            raise ValueError("candidate frame evidence is not source-local")
        for predicate in getattr(frame, "conditions", ()):
            if predicate.evidence_refs != frame.evidence_refs:
                raise ValueError("candidate predicate evidence is not source-local")


def _validate_candidate_frame_native_value(frame: Any) -> None:
    payload = frame.model_dump(mode="json")
    keys = (*_FRAME_COMMON_NATIVE_KEYS, *_FRAME_NATIVE_KEYS[frame.frame_kind])
    basis = {key: payload[key] for key in keys}
    if frame.frame_hash != "sha256:" + content_hash(basis):
        raise ValueError("candidate frame does not match native identity")
    if frame.display_name_zh is not None or frame.epistemic_status != "FOUND":
        raise ValueError("candidate frame does not retain native defaults")


def _initial_candidate_value(
    source: CapturedSourceV1,
    closure: Mapping[PinKey, CapturedSourceV1],
) -> tuple[NormativeSemanticCandidateSetV1, _InputContext, tuple[ArtifactPinV1, ...]]:
    candidate = _model_payload(
        source, NormativeSemanticCandidateSetV1, "normative candidate"
    )
    _expect_identity(
        source,
        kind="normative_semantic_candidate_set",
        ref=candidate.candidate_set_ref,
        digest=candidate.candidate_set_hash,
    )
    input_pin = _initial_input_dependency(source)
    dependencies = _expect_dependencies(source, (input_pin,), closure)
    input_context = _input_value(_resolve(closure, input_pin), closure)
    _validate_initial_candidate_input_values(
        candidate=candidate,
        inputs=input_context.inputs,
        boundary=input_context.boundary,
    )
    _validate_candidate_frames(candidate, input_context.source_object)
    _validate_initial_candidate_identity(candidate)
    return candidate, input_context, dependencies


def _validate_initial_candidate_input_values(
    *,
    candidate: NormativeSemanticCandidateSetV1,
    inputs: AdvisoryCompilerInputsV1,
    boundary: CanonicalEvidenceBoundaryV1,
) -> None:
    resources = {item.role: item for item in inputs.compiler_resources}
    if (
        candidate.change_cause != "initial_compilation"
        or candidate.predecessor_candidate_set_ref is not None
        or candidate.predecessor_candidate_set_hash is not None
        or candidate.triggering_ref is not None
        or candidate.completeness_state != "COMPLETE"
        or candidate.unresolved_slot_refs != ()
    ):
        raise ValueError("derived or unresolved candidate is unsupported")
    expected = {
        "organization_id": inputs.organization_id,
        "workspace_id": inputs.workspace_id,
        "source_identity_ref": boundary.source_identity_ref,
        "source_hash": boundary.source_hash,
        "canonical_source_ir_ref": boundary.canonical_source_ir_ref,
        "canonical_source_ir_hash": boundary.canonical_source_ir_hash,
        "competency_contract_ref": inputs.competency.artifact_ref,
        "competency_contract_hash": inputs.competency.artifact_hash,
        "normative_schema_ref": resources["frame_schema"].resource_ref,
        "normative_schema_hash": resources["frame_schema"].bytes_hash,
        "profile_ref": resources["profile"].resource_ref,
        "profile_hash": resources["profile"].bytes_hash,
        "compiler_ref": resources["semantic_compiler"].resource_ref,
        "compiler_hash": resources["semantic_compiler"].bytes_hash,
        "prompt_hash": resources["prompt"].bytes_hash,
        "provider_ref": inputs.provider_ref,
        "model_ref": inputs.model_ref,
        "config_hash": resources["configuration"].bytes_hash,
    }
    if any(getattr(candidate, key) != value for key, value in expected.items()):
        raise ValueError("candidate does not match its exact producing input")
    if candidate.compiled_at != inputs.recorded_at:
        raise ValueError("candidate compiled_at does not match its producing input")


def _validate_initial_candidate_identity(
    candidate: NormativeSemanticCandidateSetV1,
) -> None:
    basis = {
        **{key: getattr(candidate, key) for key in _CANDIDATE_NATIVE_KEYS},
        "frames": [frame.model_dump(mode="json") for frame in candidate.frames],
    }
    digest = content_hash(basis)
    if (
        candidate.candidate_set_ref != f"normative-candidate-set:{digest}"
        or candidate.candidate_set_hash != f"sha256:{digest}"
    ):
        raise ValueError("candidate does not match initial native identity")


def _trial_receipt_value(
    source: CapturedSourceV1,
    closure: Mapping[PinKey, CapturedSourceV1],
) -> tuple[AdvisoryTrialVariantV1, tuple[ArtifactPinV1, ...]]:
    receipt = _model_payload(source, AdvisoryTrialVariantV1, "advisory trial variant")
    _expect_identity(
        source,
        kind="advisory_trial_variant",
        ref=receipt.variant_ref,
        digest=receipt.variant_hash,
    )
    dependencies = _expect_dependencies(
        source, (receipt.source, receipt.inputs), closure
    )
    return receipt, dependencies


def _expected_trial_frames(
    predecessor: NormativeSemanticCandidateSetV1,
    receipt: AdvisoryTrialVariantV1,
    *,
    original: NormativeSemanticCandidateSetV1,
) -> tuple[Any, ...]:
    if receipt.variant == "OMIT_EXCEPTION":
        exceptions = tuple(
            frame for frame in predecessor.frames if frame.frame_kind == "exception"
        )
        if len(exceptions) != 1:
            raise ValueError("OMIT_EXCEPTION requires one source exception")
        return tuple(frame for frame in predecessor.frames if frame != exceptions[0])
    if receipt.variant in {"PROXY_THRESHOLD_20", "PROXY_THRESHOLD_10"}:
        matches = tuple(
            (frame, index)
            for frame in predecessor.frames
            if frame.frame_kind == "rule"
            for index, predicate in enumerate(frame.conditions)
            if predicate.operator == "GE"
        )
        if len(matches) != 1:
            raise ValueError("threshold trial requires one GE predicate")
        target, index = matches[0]
        threshold = 20 if receipt.variant == "PROXY_THRESHOLD_20" else 10
        payload = target.model_dump(mode="json")
        payload["conditions"][index]["expected_value"] = threshold
        keys = (*_FRAME_COMMON_NATIVE_KEYS, *_FRAME_NATIVE_KEYS["rule"])
        payload["frame_hash"] = "sha256:" + content_hash(
            {key: payload[key] for key in keys}
        )
        changed = type(target).model_validate(payload)
        return tuple(
            changed if frame.frame_id == target.frame_id else frame
            for frame in predecessor.frames
        )
    original_exceptions = tuple(
        frame for frame in original.frames if frame.frame_kind == "exception"
    )
    if len(original_exceptions) != 1 or receipt.added_frame != original_exceptions[0]:
        raise ValueError("repair added frame is not the original source exception")
    if any(
        frame.frame_id == receipt.added_frame.frame_id for frame in predecessor.frames
    ):
        raise ValueError("repair added frame already exists")
    return tuple(
        sorted(
            (*predecessor.frames, receipt.added_frame),
            key=lambda frame: frame.frame_id.encode("utf-8"),
        )
    )


def _candidate_value(
    source: CapturedSourceV1,
    closure: Mapping[PinKey, CapturedSourceV1],
    *,
    _visited: frozenset[PinKey] = frozenset(),
    _depth: int = 0,
) -> tuple[NormativeSemanticCandidateSetV1, _InputContext, tuple[ArtifactPinV1, ...]]:
    key = _pin_key(source.source)
    if key in _visited or _depth >= 32:
        raise ValueError("candidate/receipt predecessor cycle or depth overflow")
    visited = _visited | {key}
    if (
        len(source.dependencies) == 1
        and source.dependencies[0].artifact_kind == "advisory_compiler_inputs"
    ):
        return _initial_candidate_value(source, closure)
    if (
        len(source.dependencies) != 1
        or source.dependencies[0].artifact_kind != "advisory_trial_variant"
    ):
        raise ValueError("candidate must declare one exact input or producer receipt")

    candidate = _model_payload(
        source, NormativeSemanticCandidateSetV1, "derived normative candidate"
    )
    _expect_identity(
        source,
        kind="normative_semantic_candidate_set",
        ref=candidate.candidate_set_ref,
        digest=candidate.candidate_set_hash,
    )
    receipt_pin = source.dependencies[0]
    dependencies = _expect_dependencies(source, (receipt_pin,), closure)
    receipt_source = _resolve(closure, receipt_pin)
    receipt_key = _pin_key(receipt_source.source)
    if receipt_key in visited:
        raise ValueError("candidate/receipt predecessor cycle")
    receipt, _ = _trial_receipt_value(receipt_source, closure)
    if receipt.candidate != source.source:
        raise ValueError("trial receipt subject does not equal derived candidate")
    predecessor_source = _resolve(closure, receipt.source)
    direct_predecessor_source = predecessor_source
    predecessor, context, _ = _candidate_value(
        predecessor_source,
        closure,
        _visited=visited | {receipt_key},
        _depth=_depth + 1,
    )
    if receipt.inputs != ArtifactPinV1(
        artifact_kind="advisory_compiler_inputs",
        artifact_ref=context.inputs.input_ref,
        artifact_hash=context.inputs.input_hash,
    ):
        raise ValueError(
            "trial receipt input does not match declared predecessor ancestry"
        )

    original = predecessor
    while (
        original.change_cause != "initial_compilation"
        or original.compiler_ref.endswith("#advisory-trial-variant.v1")
    ):
        predecessor_pin = predecessor_source.dependencies[0]
        predecessor_receipt, _ = _trial_receipt_value(
            _resolve(closure, predecessor_pin), closure
        )
        predecessor_source = _resolve(closure, predecessor_receipt.source)
        original, _original_context, _ = _candidate_value(
            predecessor_source,
            closure,
            _visited=visited | {receipt_key},
            _depth=_depth + 1,
        )
        break

    # Preserve the retained adapter's frame-before-predecessor-proof error order.
    _expected_trial_frames(predecessor, receipt, original=original)
    prior_receipt = None
    if receipt.variant == "ADD_MISSING_FRAME":
        if (
            len(direct_predecessor_source.dependencies) != 1
            or direct_predecessor_source.dependencies[0].artifact_kind
            != "advisory_trial_variant"
        ):
            raise ValueError("repair predecessor is not the native omitted variant")
        prior_receipt, _ = _trial_receipt_value(
            _resolve(closure, direct_predecessor_source.dependencies[0]), closure
        )
    _validate_trial_candidate_value(
        candidate=candidate,
        inputs=context.inputs,
        predecessor=predecessor,
        receipt=receipt,
        original=original,
        prior_receipt=prior_receipt,
    )
    return candidate, context, dependencies


def _validate_trial_candidate_value(
    *,
    candidate: NormativeSemanticCandidateSetV1,
    inputs: AdvisoryCompilerInputsV1,
    predecessor: NormativeSemanticCandidateSetV1,
    receipt: AdvisoryTrialVariantV1,
    original: NormativeSemanticCandidateSetV1,
    prior_receipt: AdvisoryTrialVariantV1 | None,
) -> None:
    """Check the complete native mutation using typed historical values only."""
    frames = _expected_trial_frames(predecessor, receipt, original=original)
    payload = predecessor.model_dump(mode="json")
    payload["frames"] = [frame.model_dump(mode="json") for frame in frames]
    payload["compiled_at"] = receipt.recorded_at.isoformat()
    if receipt.variant == "ADD_MISSING_FRAME":
        if prior_receipt is None:
            raise ValueError("repair predecessor is not the native omitted variant")
        if prior_receipt.variant != "OMIT_EXCEPTION":
            raise ValueError("repair predecessor is not the native omitted variant")
        if receipt.recorded_at != inputs.recorded_at + timedelta(seconds=2):
            raise ValueError("repair receipt has unsupported saved time")
        payload.update(
            predecessor_candidate_set_ref=predecessor.candidate_set_ref,
            predecessor_candidate_set_hash=predecessor.candidate_set_hash,
            change_cause="bounded_repair",
            triggering_ref="trial-repair:add-missing-frame",
        )
        digest = content_hash(
            {
                "predecessor_candidate_set_ref": predecessor.candidate_set_ref,
                "predecessor_candidate_set_hash": predecessor.candidate_set_hash,
                "repair_ref": "trial-repair:add-missing-frame",
                "frames": payload["frames"],
            }
        )
    else:
        if (
            predecessor.compiler_ref
            != inputs.resource("semantic_compiler").resource_ref
            or predecessor.change_cause != "initial_compilation"
            or predecessor.predecessor_candidate_set_ref is not None
            or predecessor.predecessor_candidate_set_hash is not None
            or predecessor.triggering_ref is not None
        ):
            raise ValueError("trial receipt source is not an initial native candidate")
        if receipt.recorded_at != inputs.recorded_at + timedelta(seconds=1):
            raise ValueError("trial receipt has unsupported saved time")
        semantic = inputs.resource("semantic_compiler")
        payload.update(
            compiler_ref=semantic.resource_ref + "#advisory-trial-variant.v1",
            compiler_hash=semantic.bytes_hash,
            predecessor_candidate_set_ref=None,
            predecessor_candidate_set_hash=None,
            change_cause="initial_compilation",
            triggering_ref=None,
        )
        digest = content_hash(
            {
                **{key: payload[key] for key in _CANDIDATE_NATIVE_KEYS},
                "frames": payload["frames"],
            }
        )
    payload.update(
        candidate_set_ref=f"normative-candidate-set:{digest}",
        candidate_set_hash=f"sha256:{digest}",
    )
    expected = NormativeSemanticCandidateSetV1.model_validate(payload)
    if candidate != expected:
        raise ValueError("derived candidate does not match exact native mutation")


def _validation_value(
    source: CapturedSourceV1,
    closure: Mapping[PinKey, CapturedSourceV1],
) -> tuple[ValidationContractV1, _InputContext, tuple[ArtifactPinV1, ...]]:
    validation = _model_payload(source, ValidationContractV1, "validation contract")
    _expect_identity(
        source,
        kind="validation_contract",
        ref=validation.validation_contract_ref,
        digest=validation.validation_contract_hash,
    )
    input_pin = _initial_input_dependency(source)
    dependencies = _expect_dependencies(source, (input_pin,), closure)
    input_context = _input_value(_resolve(closure, input_pin), closure)
    oracle = _json_pointer(
        input_context.retained_json, input_context.inputs.oracle_locator
    )
    expected = _validate_validation_input_values(
        validation=validation,
        inputs=input_context.inputs,
        boundary=input_context.boundary,
        profile=input_context.profile,
        human_oracle_hash="sha256:" + content_hash(oracle),
    )
    _validate_validation_source_members(
        validation=validation,
        source_object=input_context.source_object,
        expected_oracle_ref=expected["human_oracle_ref"],
    )
    _validate_validation_identity(validation=validation, expected=expected)
    return validation, input_context, dependencies


def _validate_validation_input_values(
    *,
    validation: ValidationContractV1,
    inputs: AdvisoryCompilerInputsV1,
    boundary: CanonicalEvidenceBoundaryV1,
    profile: dict[str, Any],
    human_oracle_hash: str,
) -> dict[str, Any]:
    resources = {item.role: item for item in inputs.compiler_resources}
    expected_oracle_ref = f"{boundary.source_identity_ref}#{inputs.oracle_locator}"
    expected = {
        "organization_id": inputs.organization_id,
        "workspace_id": inputs.workspace_id,
        "source_identity_ref": boundary.source_identity_ref,
        "source_hash": boundary.source_hash,
        "boundary_ref": boundary.boundary_ref,
        "boundary_hash": boundary.boundary_hash,
        "competency_contract_ref": inputs.competency.artifact_ref,
        "competency_contract_hash": inputs.competency.artifact_hash,
        "normative_schema_ref": resources["frame_schema"].resource_ref,
        "normative_schema_hash": resources["frame_schema"].bytes_hash,
        "human_oracle_ref": expected_oracle_ref,
        "human_oracle_hash": human_oracle_hash,
    }
    if any(getattr(validation, key) != value for key, value in expected.items()):
        raise ValueError("validation does not match its exact producing input")
    independence = validation.independence
    if (
        independence.compiler_ref != resources["validation_compiler"].resource_ref
        or independence.compiler_hash != resources["validation_compiler"].bytes_hash
        or independence.run_ref != f"validation-compile-run:{profile['profile_id']}"
        or independence.prompt_hash != resources["prompt"].bytes_hash
        or independence.config_hash != resources["configuration"].bytes_hash
        or independence.provider_ref != inputs.provider_ref
        or independence.model_ref != inputs.model_ref
        or independence.source_region_coverage_ref != "source-region-coverage:complete"
        or independence.semantic_candidate_input_ref is not None
    ):
        raise ValueError("validation independence member mismatch")
    if (
        validation.review_state != "proposed"
        or validation.predecessor_validation_contract_ref is not None
        or validation.predecessor_validation_contract_hash is not None
        or validation.review_decision_ref is not None
        or validation.compiled_at != inputs.recorded_at
    ):
        raise ValueError("initial validation lineage or saved time mismatch")
    return expected


def _validate_validation_source_members(
    *,
    validation: ValidationContractV1,
    source_object: dict[str, Any],
    expected_oracle_ref: str,
) -> None:
    dimensions = [
        item
        for item in source_object.get("semantic_dimensions", [])
        if isinstance(item, dict) and item.get("critical") is True
    ]
    cases = [
        item
        for item in source_object.get("competency_oracle", [])
        if isinstance(item, dict)
    ]
    if not dimensions or not cases:
        raise ValueError("validation source dimensions or oracle cases are missing")
    expected_obligations: list[dict[str, Any]] = []
    for dimension in dimensions:
        name = str(dimension["dimension"])
        competency_ids = tuple(
            str(case["competency_id"])
            for case in cases
            if name in case.get("required_semantic_dimensions", [])
        ) or tuple(str(case["competency_id"]) for case in cases)
        evidence_refs = tuple(str(ref) for ref in dimension["evidence_refs"])
        obligation_hash = "sha256:" + content_hash(
            {
                "semantic_dimension": name,
                "competency_ids": competency_ids,
                "evidence_refs": evidence_refs,
                "human_oracle_ref": expected_oracle_ref,
            }
        )
        expected_obligations.append(
            {
                "obligation_id": f"obligation.{name}",
                "obligation_hash": obligation_hash,
                "semantic_dimension": name,
                "competency_ids": list(competency_ids),
                "acceptance_criterion": (f"candidate preserves source-grounded {name}"),
                "evidence_refs": list(evidence_refs),
                "provenance_kind": "authority_reviewed",
            }
        )
    expected_obligations.sort(key=lambda item: item["obligation_id"].encode("utf-8"))
    if [item.model_dump(mode="json") for item in validation.obligations] != (
        expected_obligations
    ):
        raise ValueError("validation obligations do not match native source members")

    by_competency = {
        str(case["competency_id"]): tuple(
            item["obligation_id"]
            for item in expected_obligations
            if str(case["competency_id"]) in item["competency_ids"]
        )
        for case in cases
    }
    expected_tests: list[dict[str, Any]] = []
    for case in cases:
        competency_id = str(case["competency_id"])
        test_basis = {
            "competency_id": competency_id,
            "case_kind": case["case_kind"],
            "question": case["question"],
            "expected_result": case["expected_result"],
            "required_obligation_ids": by_competency[competency_id],
            "evidence_refs": tuple(str(ref) for ref in case["evidence_refs"]),
        }
        expected_tests.append(
            {
                "test_id": f"competency-test.{competency_id}",
                "test_hash": "sha256:" + content_hash(test_basis),
                **{
                    key: list(value) if isinstance(value, tuple) else value
                    for key, value in test_basis.items()
                },
                "provenance_kind": "authority_reviewed",
            }
        )
    expected_tests.sort(key=lambda item: item["test_id"].encode("utf-8"))
    if [item.model_dump(mode="json") for item in validation.executable_tests] != (
        expected_tests
    ):
        raise ValueError("validation tests do not match native oracle members")


def _validate_validation_identity(
    *,
    validation: ValidationContractV1,
    expected: dict[str, Any],
) -> None:
    basis = {
        **expected,
        "obligations": [
            item.model_dump(mode="json") for item in validation.obligations
        ],
        "executable_tests": [
            item.model_dump(mode="json") for item in validation.executable_tests
        ],
        "independence": validation.independence.model_dump(mode="json"),
    }
    digest = content_hash(basis)
    if (
        validation.validation_contract_ref != f"validation-contract:{digest}"
        or validation.validation_contract_hash != f"sha256:{digest}"
    ):
        raise ValueError("validation contract does not match native identity")


def _compiled_predicate(frame_id: str, predicate: Any) -> dict[str, Any]:
    return {
        "predicate_id": predicate.predicate_id,
        "operator": predicate.operator,
        "fact_ref": predicate.fact_ref,
        "expected_value": predicate.expected_value,
        "ir_node_ref": frame_id,
        "evidence_refs": list(predicate.evidence_refs),
    }


def _plan_value(
    source: CapturedSourceV1,
    closure: Mapping[PinKey, CapturedSourceV1],
) -> tuple[CompiledRulePlanV1, tuple[ArtifactPinV1, ...]]:
    plan = _model_payload(source, CompiledRulePlanV1, "compiled rule plan")
    _expect_identity(
        source,
        kind="compiled_rule_plan",
        ref=plan.plan_ref,
        digest=plan.plan_hash,
    )
    by_kind = {pin.artifact_kind: pin for pin in source.dependencies}
    expected_kinds = {
        "advisory_compiler_inputs",
        "normative_semantic_candidate_set",
        "validation_contract",
    }
    if set(by_kind) != expected_kinds or len(source.dependencies) != 3:
        raise ValueError("compiled plan must declare its exact three native inputs")
    input_pin = by_kind["advisory_compiler_inputs"]
    candidate_pin = by_kind["normative_semantic_candidate_set"]
    validation_pin = by_kind["validation_contract"]
    dependencies = _expect_dependencies(
        source, (candidate_pin, validation_pin, input_pin), closure
    )
    candidate_source = _resolve(closure, candidate_pin)
    validation_source = _resolve(closure, validation_pin)
    candidate, candidate_inputs, _candidate_dependencies = _candidate_value(
        candidate_source, closure
    )
    validation, validation_inputs, _validation_dependencies = _validation_value(
        validation_source, closure
    )
    if (
        validation_source.dependencies != (input_pin,)
        or candidate_inputs.inputs != validation_inputs.inputs
        or input_pin
        != ArtifactPinV1(
            artifact_kind="advisory_compiler_inputs",
            artifact_ref=candidate_inputs.inputs.input_ref,
            artifact_hash=candidate_inputs.inputs.input_hash,
        )
    ):
        raise ValueError("compiled plan inputs do not share one producing input")
    _validate_plan_native_value(
        plan=plan,
        candidate=candidate,
        validation=validation,
        inputs=candidate_inputs.inputs,
        competency=candidate_inputs.competency,
        profile=candidate_inputs.profile,
    )
    return plan, dependencies


def _validate_plan_native_value(
    *,
    plan: CompiledRulePlanV1,
    candidate: NormativeSemanticCandidateSetV1,
    validation: ValidationContractV1,
    inputs: AdvisoryCompilerInputsV1,
    competency: CompetencyContractV1,
    profile: dict[str, Any],
) -> None:
    rule_plan_resource = inputs.resource("rule_plan_compiler")
    if (
        (plan.organization_id, plan.workspace_id)
        != (inputs.organization_id, inputs.workspace_id)
        or (plan.organization_id, plan.workspace_id)
        != (candidate.organization_id, candidate.workspace_id)
        or plan.candidate_set_ref != candidate.candidate_set_ref
        or plan.candidate_set_hash != candidate.candidate_set_hash
        or plan.validation_contract_ref != validation.validation_contract_ref
        or plan.validation_contract_hash != validation.validation_contract_hash
        or plan.compiler_ref != rule_plan_resource.resource_ref
        or plan.compiler_hash != rule_plan_resource.bytes_hash
        or plan.semantic_profile != profile["semantic_profile"]
        or plan.compiled_at != candidate.compiled_at
    ):
        raise ValueError("compiled plan does not match its native inputs")

    competency_refs = tuple(question.competency_id for question in competency.questions)
    exceptions = {
        frame.frame_id: frame
        for frame in candidate.frames
        if frame.frame_kind == "exception"
    }
    expected_rules: list[dict[str, Any]] = []
    for frame in candidate.frames:
        if frame.frame_kind != "rule":
            continue
        expected_rules.append(
            {
                "rule_id": frame.frame_id,
                "ir_node_ref": frame.frame_id,
                "competency_refs": list(competency_refs),
                "evidence_refs": list(frame.evidence_refs),
                "predicates": [
                    _compiled_predicate(frame.frame_id, predicate)
                    for predicate in frame.conditions
                ],
                "exception_predicates": [
                    _compiled_predicate(exception.frame_id, predicate)
                    for ref in frame.exception_refs
                    if (exception := exceptions.get(ref)) is not None
                    for predicate in exception.conditions
                ],
                "result_when_matched": "MATCH",
            }
        )
    if [item.model_dump(mode="json") for item in plan.rules] != expected_rules:
        raise ValueError("compiled plan rules do not preserve native membership")
    basis = {
        "candidate_set_ref": plan.candidate_set_ref,
        "candidate_set_hash": plan.candidate_set_hash,
        "validation_contract_ref": plan.validation_contract_ref,
        "validation_contract_hash": plan.validation_contract_hash,
        "compiler_ref": plan.compiler_ref,
        "compiler_hash": plan.compiler_hash,
        "semantic_profile": plan.semantic_profile,
        "rules": [item.model_dump(mode="json") for item in plan.rules],
    }
    digest = content_hash(basis)
    if (
        plan.plan_ref != f"compiled-rule-plan:{digest}"
        or plan.plan_hash != f"sha256:{digest}"
    ):
        raise ValueError("compiled plan does not match native identity")


def _validate_stored_initial_output_values(
    *,
    inputs: AdvisoryCompilerInputsV1,
    boundary: CanonicalEvidenceBoundaryV1,
    competency: CompetencyContractV1,
    candidate: NormativeSemanticCandidateSetV1,
    validation: ValidationContractV1,
    plan: CompiledRulePlanV1,
) -> None:
    """Validate storage-native relationships; no current access/source-byte claim."""
    if (
        (boundary.organization_id, boundary.workspace_id)
        != (inputs.organization_id, inputs.workspace_id)
        or (competency.organization_id, competency.workspace_id)
        != (inputs.organization_id, inputs.workspace_id)
        or inputs.boundary
        != ArtifactPinV1(
            artifact_kind="canonical_evidence_boundary",
            artifact_ref=boundary.boundary_ref,
            artifact_hash=boundary.boundary_hash,
        )
        or inputs.competency
        != ArtifactPinV1(
            artifact_kind="competency_contract",
            artifact_ref=competency.competency_contract_ref,
            artifact_hash=competency.competency_contract_hash,
        )
        or (competency.boundary_ref, competency.boundary_hash)
        != (boundary.boundary_ref, boundary.boundary_hash)
        or competency.predecessor_ref is not None
        or competency.predecessor_hash is not None
    ):
        raise ValueError("stored compiler input boundary or competency mismatch")
    _validate_boundary_native_value(boundary)
    _validate_competency_identity(competency)
    profile = json.loads(
        _strict_base64(inputs.resource("profile").bytes_base64, "compiler profile")
    )
    _validate_initial_candidate_input_values(
        candidate=candidate, inputs=inputs, boundary=boundary
    )
    if tuple(frame.frame_kind for frame in candidate.frames) != _INITIAL_FRAME_KINDS:
        raise ValueError("initial candidate has an incompatible native frame profile")
    for frame in candidate.frames:
        _validate_candidate_frame_native_value(frame)
    _validate_initial_candidate_identity(candidate)
    expected = _validate_validation_input_values(
        validation=validation,
        inputs=inputs,
        boundary=boundary,
        profile=profile,
        human_oracle_hash=validation.human_oracle_hash,
    )
    _validate_validation_identity(validation=validation, expected=expected)
    _validate_plan_native_value(
        plan=plan,
        candidate=candidate,
        validation=validation,
        inputs=inputs,
        competency=competency,
        profile=profile,
    )


def _validate_advisory_trial_output_values(
    *,
    inputs: AdvisoryCompilerInputsV1,
    source: NormativeSemanticCandidateSetV1,
    original: NormativeSemanticCandidateSetV1,
    prior_receipt: AdvisoryTrialVariantV1 | None,
    validation: ValidationContractV1,
    competency: CompetencyContractV1,
    candidate: NormativeSemanticCandidateSetV1,
    plan: CompiledRulePlanV1,
    receipt: AdvisoryTrialVariantV1,
) -> None:
    """Verify exact typed publication values without producers or source IO."""
    candidate = _validated_live_value(
        candidate, NormativeSemanticCandidateSetV1, "candidate"
    )
    plan = _validated_live_value(plan, CompiledRulePlanV1, "plan")
    receipt = _validated_live_value(receipt, AdvisoryTrialVariantV1, "derivation")
    input_pin = ArtifactPinV1(
        artifact_kind="advisory_compiler_inputs",
        artifact_ref=inputs.input_ref,
        artifact_hash=inputs.input_hash,
    )
    source_pin = ArtifactPinV1(
        artifact_kind="normative_semantic_candidate_set",
        artifact_ref=source.candidate_set_ref,
        artifact_hash=source.candidate_set_hash,
    )
    original_pin = ArtifactPinV1(
        artifact_kind="normative_semantic_candidate_set",
        artifact_ref=original.candidate_set_ref,
        artifact_hash=original.candidate_set_hash,
    )
    candidate_pin = ArtifactPinV1(
        artifact_kind="normative_semantic_candidate_set",
        artifact_ref=candidate.candidate_set_ref,
        artifact_hash=candidate.candidate_set_hash,
    )
    if (
        (receipt.organization_id, receipt.workspace_id)
        != (inputs.organization_id, inputs.workspace_id)
        or receipt.inputs != input_pin
        or receipt.source != source_pin
        or receipt.candidate != candidate_pin
    ):
        raise ValueError("ADVISORY_TRIAL_DERIVATION_MISMATCH")
    if receipt.variant == "ADD_MISSING_FRAME":
        if (
            prior_receipt is None
            or prior_receipt.variant != "OMIT_EXCEPTION"
            or prior_receipt.inputs != input_pin
            or prior_receipt.source != original_pin
            or prior_receipt.candidate != source_pin
            or (prior_receipt.organization_id, prior_receipt.workspace_id)
            != (inputs.organization_id, inputs.workspace_id)
        ):
            raise ValueError("repair predecessor is not the native omitted variant")
    elif source != original or prior_receipt is not None:
        raise ValueError("trial receipt source is not an initial native candidate")
    _validate_trial_candidate_value(
        candidate=candidate,
        inputs=inputs,
        predecessor=source,
        receipt=receipt,
        original=original,
        prior_receipt=prior_receipt,
    )
    profile = json.loads(
        _strict_base64(inputs.resource("profile").bytes_base64, "compiler profile")
    )
    _validate_plan_native_value(
        plan=plan,
        candidate=candidate,
        validation=validation,
        inputs=inputs,
        competency=competency,
        profile=profile,
    )


def _validate_live_initial_output_values(
    *,
    inputs: AdvisoryCompilerInputsV1,
    boundary: CanonicalEvidenceBoundaryV1,
    competency: CompetencyContractV1,
    source_capture: CapturedJsonV1,
    candidate: NormativeSemanticCandidateSetV1,
    validation: ValidationContractV1,
    plan: CompiledRulePlanV1,
) -> None:
    candidate = _validated_live_value(
        candidate, NormativeSemanticCandidateSetV1, "candidate"
    )
    validation = _validated_live_value(validation, ValidationContractV1, "validation")
    plan = _validated_live_value(plan, CompiledRulePlanV1, "plan")
    view = _source_member_value(source_capture)
    retained_json, source_object = _compiler_source_values(inputs, view)
    profile = _validate_input_native_value(
        inputs=inputs,
        boundary=boundary,
        competency=competency,
        view=view,
        source_object=source_object,
    )
    _validate_initial_candidate_input_values(
        candidate=candidate, inputs=inputs, boundary=boundary
    )
    _validate_candidate_frames(candidate, source_object)
    _validate_initial_candidate_identity(candidate)
    expected = _validate_validation_input_values(
        validation=validation,
        inputs=inputs,
        boundary=boundary,
        profile=profile,
        human_oracle_hash="sha256:"
        + content_hash(_json_pointer(retained_json, inputs.oracle_locator)),
    )
    _validate_validation_source_members(
        validation=validation,
        source_object=source_object,
        expected_oracle_ref=expected["human_oracle_ref"],
    )
    _validate_validation_identity(validation=validation, expected=expected)
    _validate_plan_native_value(
        plan=plan,
        candidate=candidate,
        validation=validation,
        inputs=inputs,
        competency=competency,
        profile=profile,
    )


def verify_competency_contract_v1(
    *, source: CapturedSourceV1, closure: Mapping[PinKey, CapturedSourceV1]
) -> tuple[ArtifactPinV1, ...]:
    """Verify an initial source-derived competency proposal."""
    closure = _validate_closure(closure)
    source = _require_supplied_source(source, closure)
    _competency, dependencies = _competency_value(source, closure)
    return dependencies


def verify_advisory_compiler_inputs_v1(
    *, source: CapturedSourceV1, closure: Mapping[PinKey, CapturedSourceV1]
) -> tuple[ArtifactPinV1, ...]:
    """Verify exact retained source, pointers, and role-member bindings."""
    closure = _validate_closure(closure)
    source = _require_supplied_source(source, closure)
    return _input_value(source, closure).dependencies


def verify_normative_semantic_candidate_set_v1(
    *, source: CapturedSourceV1, closure: Mapping[PinKey, CapturedSourceV1]
) -> tuple[ArtifactPinV1, ...]:
    """Verify one exact initial or derived native candidate and its producer."""
    closure = _validate_closure(closure)
    source = _require_supplied_source(source, closure)
    _candidate, _input_context, dependencies = _candidate_value(source, closure)
    return dependencies


def verify_validation_contract_v1(
    *, source: CapturedSourceV1, closure: Mapping[PinKey, CapturedSourceV1]
) -> tuple[ArtifactPinV1, ...]:
    """Verify one exact initial external validation compilation."""
    closure = _validate_closure(closure)
    source = _require_supplied_source(source, closure)
    _validation, _input_context, dependencies = _validation_value(source, closure)
    return dependencies


def verify_compiled_rule_plan_v1(
    *, source: CapturedSourceV1, closure: Mapping[PinKey, CapturedSourceV1]
) -> tuple[ArtifactPinV1, ...]:
    """Verify one initial or derived deterministic rule-plan projection."""
    closure = _validate_closure(closure)
    source = _require_supplied_source(source, closure)
    _plan, dependencies = _plan_value(source, closure)
    return dependencies


__all__ = [
    "verify_advisory_compiler_inputs_v1",
    "verify_compiled_rule_plan_v1",
    "verify_competency_contract_v1",
    "verify_normative_semantic_candidate_set_v1",
    "verify_validation_contract_v1",
]
