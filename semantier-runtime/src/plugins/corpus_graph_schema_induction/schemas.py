from __future__ import annotations

import re
from typing import Any
from urllib.parse import unquote, urlsplit

from contracts.corpus_graph_schema import CorpusGraphSchemaCandidate


TOOLSET_NAME = "corpus_graph_schema_induction"
TOOL_NAMES = [
    "corpus_graph_schema_profiles",
    "corpus_graph_schema_induce",
    "corpus_graph_schema_list",
    "corpus_graph_schema_get",
    "corpus_graph_schema_intent_get",
    "corpus_graph_schema_intent_review",
    "corpus_graph_schema_review",
    "corpus_graph_schema_sample",
    "corpus_graph_schema_feedback_retest",
]

_ID = {
    "type": "string", "minLength": 1, "maxLength": 256,
    "pattern": r"^[A-Za-z0-9][A-Za-z0-9:._-]{0,255}$",
}
_PROFILE_ID = {
    "type": "string",
    "pattern": r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$",
}
_COMPETENCY_QUESTION = {
    "type": "object",
    "properties": {
        "question_id": _ID,
        "text": {"type": "string", "minLength": 1, "maxLength": 2000},
    },
    "required": ["question_id", "text"],
    "additionalProperties": False,
}
_STRING = {"type": "string"}
def _closed(properties: dict[str, Any], required: list[str] | None = None, **extra: Any) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
        **extra,
    }


_GRAPH_INTENT_REF = {
    "type": "string",
    "pattern": r"^graph-intent-[0-9a-f]{24}$",
}
_SHA256_REF = {"type": "string", "pattern": r"^sha256:[0-9a-f]{64}$"}
_ID_ARRAY = {"type": "array", "items": _ID, "maxItems": 100}
_PRIMARY_RELATIONSHIP = _closed({
    "type_id": _ID,
    "source_types": _ID_ARRAY,
    "target_types": _ID_ARRAY,
    "direction": _ID,
    "semantic_description": {"type": "string", "minLength": 1, "maxLength": 2000},
}, ["type_id", "source_types", "target_types", "direction", "semantic_description"])
_SOURCE_ROLE_INTERPRETATION = _closed({
    "canonical_objects": _ID_ARRAY,
    "lexical_evidence": _ID_ARRAY,
    "reference_examples": _ID_ARRAY,
    "classifications": _ID_ARRAY,
    "source_assessments": _ID_ARRAY,
    "remediation_fields": _ID_ARRAY,
    "contextual_fields": _ID_ARRAY,
}, [
    "canonical_objects", "lexical_evidence", "reference_examples",
    "classifications", "source_assessments", "remediation_fields",
    "contextual_fields",
])
_GRAPH_INTENT_RELATION_EDIT = _closed({
    "primaryRelationship": _PRIMARY_RELATIONSHIP,
    "sourceRoleInterpretation": _SOURCE_ROLE_INTERPRETATION,
}, ["primaryRelationship", "sourceRoleInterpretation"])


_COLUMN = _closed({
    "role_id": _STRING, "table_index": {"type": "integer"},
    "column_index": {"type": "integer"}, "header_text": _STRING,
    "evidence_ref": _STRING,
}, ["role_id", "table_index", "column_index", "header_text", "evidence_ref"])
_PARAGRAPH = _closed({
    "role_id": _STRING, "value": _STRING, "pattern": _STRING, "anchor_id": _STRING,
}, ["role_id", "value", "pattern", "anchor_id"])
_SOURCE_SELECTOR = _closed({
    "kind": {"type": "string", "enum": ["paragraph_pattern", "table_column", "combined"]},
    "matches": {"type": "array", "items": _PARAGRAPH},
    "columns": {"type": "array", "items": _COLUMN},
    "table_columns": {"type": "array", "items": _COLUMN},
    "paragraph_patterns": {"type": "array", "items": _PARAGRAPH},
}, ["kind"])
_SEMANTIC_ROLE = _closed({
    "role_id": _STRING, "source_selector": _SOURCE_SELECTOR,
    "evidence_refs": {"type": "array", "items": _STRING},
}, ["role_id", "source_selector", "evidence_refs"])
_NODE_TYPE = _closed({"type_id": _STRING, "derived_from_role": _STRING}, ["type_id", "derived_from_role"])
_RELATION_TYPE = _closed({
    "type_id": _STRING, "source_role": _STRING, "source_type": _STRING,
    "target_role": _STRING,
}, ["type_id", "target_role"])
_FRAME_TYPE = _closed({
    "type_id": _STRING, "slots": {"type": "array", "items": _STRING},
    "selected_because": {"type": "array", "items": _STRING},
}, ["type_id", "slots"])
_EVENT_TYPE = _closed({
    "type_id": _STRING, "required_roles": {"type": "array", "items": _STRING},
}, ["type_id", "required_roles"])
_REPRESENTATION = _closed({
    "initial_representation": _STRING, "binary_triple_sufficient": {"type": "boolean"},
    "selected_representation": _STRING, "reason": _STRING,
    "material_dimensions": {"type": "array", "items": _STRING},
    "evidence_refs": {"type": "array", "items": _STRING},
}, ["initial_representation", "binary_triple_sufficient", "selected_representation", "evidence_refs"])
_EVENT_RULE = _closed({
    "boundary_role": _STRING, "required_roles": {"type": "array", "items": _STRING},
    "inherit_context_roles": {"type": "array", "items": _STRING},
    "attach_following_roles": {"type": "array", "items": _STRING},
    "output_key_by_role": {
        "type": "object", "properties": {}, "patternProperties": {r"^[A-Za-z_][A-Za-z0-9_]*$": _STRING},
        "additionalProperties": False,
    },
    "sample_limit": {"type": "integer", "minimum": 1},
}, ["boundary_role", "required_roles", "inherit_context_roles", "attach_following_roles", "output_key_by_role", "sample_limit"])
_SAMPLE_RELATION = _closed({
    "type_id": _STRING, "source_ref": _STRING, "target_value": _STRING,
    "target_evidence_ref": _STRING,
}, ["type_id", "source_ref", "target_value", "target_evidence_ref"])
_EVENT_SAMPLE = {
    "type": "object",
    "properties": {
        "event_id": _STRING,
        "evidence_refs": {"type": "array", "items": _STRING},
        "relations": {"type": "array", "items": _SAMPLE_RELATION},
    },
    "required": ["event_id", "evidence_refs", "relations"],
    "patternProperties": {r"^(?!event_id$|evidence_refs$|relations$)[A-Za-z_][A-Za-z0-9_]*$": {}},
    "additionalProperties": False,
}
_AMBIGUITY = _closed({
    "ambiguity_id": _STRING, "status": _STRING, "description": _STRING,
    "missing_roles": {"type": "array", "items": _STRING},
    "evidence_refs": {"type": "array", "items": _STRING},
}, ["ambiguity_id"])
_SOURCE_PROFILE = _closed({
    "document_kind": _STRING, "element_count": {"type": "integer", "minimum": 0},
    "table_count": {"type": "integer", "minimum": 0}, "profile_id": _PROFILE_ID,
    "source_ref": {
        "type": "string", "maxLength": 1024,
        "pattern": r"^(repo|workspace)://[^\s?#\\]+$",
    },
    "document_hash": {"type": "string", "pattern": r"^sha256:[0-9a-f]{64}$"},
    "observed_column_roles": {"type": "array", "items": _STRING},
    "observed_paragraph_roles": {"type": "array", "items": _STRING},
}, ["document_kind", "element_count", "table_count", "profile_id", "source_ref", "document_hash", "observed_column_roles", "observed_paragraph_roles"])
_HINTS = _closed({
    "column_selectors": {"type": "array", "items": _COLUMN},
    "paragraph_selectors": {"type": "array", "items": _PARAGRAPH},
    "table_selector_rules": {
        "type": "object", "properties": {}, "patternProperties": {r"^.+$": _STRING},
        "additionalProperties": False,
    },
    "paragraph_selector_rules": {"type": "array", "items": _closed({"role_id": _STRING, "pattern": _STRING}, ["role_id", "pattern"])},
    "event_extraction_rule": _EVENT_RULE,
    "relation_templates": {"type": "array", "items": _RELATION_TYPE},
    "representation_type": _STRING,
    "frame_slots": {"type": "array", "items": _STRING},
    "sample_event_instances": {"type": "array", "items": _EVENT_SAMPLE},
}, ["column_selectors", "paragraph_selectors", "table_selector_rules", "paragraph_selector_rules", "relation_templates", "representation_type", "frame_slots"])
_CANDIDATE_FIELDS = {
    "schema_version",
    "candidate_id",
    "candidate_hash",
    "source_structure_profile",
    "intended_use",
    "semantic_role_proposals",
    "proposed_node_types",
    "proposed_relation_types",
    "proposed_frame_types",
    "proposed_event_types",
    "competency_questions",
    "representation_decisions",
    "extraction_contract_hints",
    "unresolved_ambiguities",
    "source_evidence_refs",
}
_CANDIDATE_REQUIRED_FIELDS = {
    "schema_version", "candidate_id", "candidate_hash",
    "source_structure_profile", "intended_use", "semantic_role_proposals",
    "proposed_node_types", "representation_decisions",
    "extraction_contract_hints", "source_evidence_refs",
}
_EDITED_CANDIDATE = {
    "type": "object",
    "properties": {
        "schema_version": _ID,
        "candidate_id": _ID,
        "candidate_hash": _ID,
        "source_structure_profile": _SOURCE_PROFILE,
        "intended_use": {"type": "string", "minLength": 1},
        "semantic_role_proposals": {"type": "array", "items": _SEMANTIC_ROLE},
        "proposed_node_types": {"type": "array", "items": _NODE_TYPE},
        "proposed_relation_types": {"type": "array", "items": _RELATION_TYPE},
        "proposed_frame_types": {"type": "array", "items": _FRAME_TYPE},
        "proposed_event_types": {"type": "array", "items": _EVENT_TYPE},
        "competency_questions": {"type": "array", "items": _COMPETENCY_QUESTION},
        "representation_decisions": {"type": "array", "items": _REPRESENTATION},
        "extraction_contract_hints": _HINTS,
        "unresolved_ambiguities": {"type": "array", "items": _AMBIGUITY},
        "source_evidence_refs": {"type": "array", "items": {"type": "string"}},
    },
    "required": sorted(_CANDIDATE_REQUIRED_FIELDS),
    "additionalProperties": False,
}


def _schema(description: str, properties: dict, required: list[str] | None = None) -> dict:
    return {
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required or [],
            "additionalProperties": False,
        },
    }


SCHEMAS = {
    "corpus_graph_schema_profiles": _schema(
        "List governed corpus graph schema induction profiles.", {}
    ),
    "corpus_graph_schema_induce": _schema(
        "Propose a candidate schema from a governed session DOCX upload.",
        {
            "lineage_id": _ID,
            "source": {
                "type": "string",
                "maxLength": 512,
                "pattern": (
                    r"^sessions/[A-Za-z0-9][A-Za-z0-9._-]{0,127}/uploads/"
                    r"(?!\.{1,2}\.docx$)[^\x00-\x1f/\\]{1,240}\.docx$"
                ),
            },
            "profile_id": _PROFILE_ID,
            "intended_use": {"type": "string", "minLength": 1, "maxLength": 4000},
            "competency_questions": {
                "type": "array",
                "items": _COMPETENCY_QUESTION,
                "maxItems": 100,
            },
        },
        ["lineage_id", "source", "profile_id", "intended_use", "competency_questions"],
    ),
    "corpus_graph_schema_list": _schema(
        "List candidate schema decisions in the authenticated workspace.",
        {"limit": {"type": "integer", "minimum": 1, "maximum": 100}},
    ),
    "corpus_graph_schema_get": _schema(
        "Get one candidate schema decision from the authenticated workspace.",
        {"decision_id": _ID},
        ["decision_id"],
    ),
    "corpus_graph_schema_intent_get": _schema(
        "Load one governed GraphIntent by its exact immutable ref and hash.",
        {"graph_intent_ref": _GRAPH_INTENT_REF, "graph_intent_hash": _SHA256_REF},
        ["graph_intent_ref", "graph_intent_hash"],
    ),
    "corpus_graph_schema_intent_review": _schema(
        "Accept, edit, or reject one pinned GraphIntent through the authenticated Chat channel.",
        {
            "graph_intent_ref": _GRAPH_INTENT_REF,
            "graph_intent_hash": _SHA256_REF,
            "state": {"type": "string", "enum": ["accepted", "edited", "rejected"]},
            "selected_question_ids": _ID_ARRAY,
            "relation_edit": _GRAPH_INTENT_RELATION_EDIT,
        },
        ["graph_intent_ref", "graph_intent_hash", "state", "selected_question_ids"],
    ),
    "corpus_graph_schema_review": _schema(
        "Record a candidate-only human review; this never approves or admits authority.",
        {
            "decision_id": _ID,
            "state": {"type": "string", "enum": ["accepted", "edited", "rejected"]},
            "note": {"type": "string", "maxLength": 4000},
            "edited_candidate": _EDITED_CANDIDATE,
        },
        ["decision_id", "state"],
    ),
    "corpus_graph_schema_sample": _schema(
        "Execute a deterministic reviewed-candidate sample or replay a pinned sample.",
        {
            "action": {"type": "string", "enum": ["execute", "replay"]},
            "decision_id": _ID,
            "sample_id": _ID,
        },
        ["action"],
    ),
    "corpus_graph_schema_feedback_retest": _schema(
        "Compare an accepted baseline with a distinct edited correction and persist the deterministic retest evidence.",
        {
            "correction_decision_id": _ID,
            "baseline_decision_id": _ID,
            "requested_relation_type": _ID,
        },
        ["correction_decision_id", "baseline_decision_id", "requested_relation_type"],
    ),
}


_OPERATIONS = {
    "profiles": "corpus_graph_schema_profiles",
    "induce": "corpus_graph_schema_induce",
    "list": "corpus_graph_schema_list",
    "get": "corpus_graph_schema_get",
    "intent_get": "corpus_graph_schema_intent_get",
    "intent_review": "corpus_graph_schema_intent_review",
    "review": "corpus_graph_schema_review",
    "sample": "corpus_graph_schema_sample",
    "feedback_retest": "corpus_graph_schema_feedback_retest",
}

_FORBIDDEN_AUTHORITY_KEYS = {
    "workspace_id", "organization_id", "user_id", "actor_id",
    "workspace_root", "host_path", "file_path", "profile_path",
    "workspaceId", "organizationId", "userId", "actorId",
    "workspaceRoot", "hostPath", "filePath", "profilePath",
}


def _validate_schema(value: Any, schema: dict[str, Any], path: str = "arguments") -> None:
    expected = schema.get("type")
    valid_type = (
        (expected == "object" and isinstance(value, dict))
        or (expected == "array" and isinstance(value, list))
        or (expected == "string" and isinstance(value, str))
        or (expected == "integer" and isinstance(value, int) and not isinstance(value, bool))
        or (expected == "boolean" and isinstance(value, bool))
        or expected is None
    )
    if not valid_type:
        raise ValueError(f"{path} must be {expected}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path} has an invalid value")
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            raise ValueError(f"{path} is too short")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise ValueError(f"{path} is too long")
        if "pattern" in schema and re.fullmatch(schema["pattern"], value) is None:
            raise ValueError(f"{path} has an invalid format")
    if isinstance(value, int) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise ValueError(f"{path} is below minimum")
        if "maximum" in schema and value > schema["maximum"]:
            raise ValueError(f"{path} exceeds maximum")
    if isinstance(value, list):
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise ValueError(f"{path} has too many items")
        for index, item in enumerate(value):
            _validate_schema(item, schema.get("items", {}), f"{path}[{index}]")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        missing = [key for key in schema.get("required", []) if key not in value]
        if missing:
            raise ValueError(f"{path}.{missing[0]} is required")
        patterns = schema.get("patternProperties", {})
        for key, item in value.items():
            child = properties.get(key)
            if child is None:
                child = next(
                    (candidate for pattern, candidate in patterns.items() if re.fullmatch(pattern, key)),
                    None,
                )
            if child is None and schema.get("additionalProperties") is False:
                raise ValueError(f"unexpected field: {path}.{key}")
            if child is not None:
                _validate_schema(item, child, f"{path}.{key}")


def _reject_authority_claims(value: Any, path: str = "edited_candidate") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in _FORBIDDEN_AUTHORITY_KEYS:
                raise ValueError(f"unexpected authority field: {path}.{key}")
            _reject_authority_claims(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_authority_claims(item, f"{path}[{index}]")


def _validate_governed_source_ref(source_ref: str) -> None:
    if not isinstance(source_ref, str) or not source_ref:
        raise ValueError("edited_candidate source_ref is required")
    if "\\" in source_ref or unquote(source_ref) != source_ref:
        raise ValueError("edited_candidate source_ref is not canonical")
    parsed = urlsplit(source_ref)
    if parsed.scheme not in {"repo", "workspace"} or parsed.query or parsed.fragment:
        raise ValueError("edited_candidate source_ref must be a governed repo or workspace ref")
    if not parsed.netloc or not parsed.path.startswith("/"):
        raise ValueError("edited_candidate source_ref is malformed")
    segments = [parsed.netloc, *parsed.path[1:].split("/")]
    if any(
        not segment
        or segment in {".", ".."}
        or any(ord(char) < 32 or char.isspace() for char in segment)
        for segment in segments
    ):
        raise ValueError("edited_candidate source_ref contains an unsafe path segment")
    if parsed.scheme == "workspace":
        normalized = "/".join(segments)
        pattern = (
            r"^sessions/[A-Za-z0-9][A-Za-z0-9._-]{0,127}/uploads/"
            r"(?!\.{1,2}\.docx$)[^\x00-\x1f/\\]{1,240}\.docx$"
        )
        if re.fullmatch(pattern, normalized) is None:
            raise ValueError("edited_candidate workspace source_ref is not a governed DOCX upload")


def _validate_edited_candidate(candidate: dict[str, Any]) -> None:
    _reject_authority_claims(candidate)
    # The canonical contract owns semantic validation and compatibility
    # defaults for additive fields. Plugin validation only closes the transport
    # envelope and rejects identity/path claims.
    normalized = CorpusGraphSchemaCandidate.from_dict(candidate).as_dict()
    _validate_governed_source_ref(
        normalized["source_structure_profile"]["source_ref"]
    )
    hints = normalized["extraction_contract_hints"]
    selector_fields = {
        "paragraph_pattern": {"kind", "matches"},
        "table_column": {"kind", "columns"},
        "combined": {"kind", "table_columns", "paragraph_patterns"},
    }
    for index, proposal in enumerate(normalized["semantic_role_proposals"]):
        selector = proposal["source_selector"]
        if set(selector) != selector_fields.get(selector.get("kind"), set()):
            raise ValueError(
                f"edited_candidate.semantic_role_proposals[{index}].source_selector "
                "does not match its selector kind"
            )
    for index, relation in enumerate(normalized["proposed_relation_types"]):
        if ("source_role" in relation) == ("source_type" in relation):
            raise ValueError(
                f"edited_candidate.proposed_relation_types[{index}] requires one source binding"
            )
    rule = hints.get("event_extraction_rule") or {}
    allowed_sample_keys = {
        "event_id", "evidence_refs", "relations",
        *map(str, rule.get("required_roles") or []),
        *map(str, (rule.get("output_key_by_role") or {}).values()),
    }
    for index, sample in enumerate(hints.get("sample_event_instances") or []):
        extra = set(sample) - allowed_sample_keys
        if extra:
            raise ValueError(
                f"unexpected field: edited_candidate.extraction_contract_hints."
                f"sample_event_instances[{index}].{sorted(extra)[0]}"
            )
        if any(not isinstance(sample.get(role), str) for role in rule.get("required_roles") or []):
            raise ValueError(f"edited_candidate sample {index} has invalid required role values")
        for output_key in (rule.get("output_key_by_role") or {}).values():
            values = sample.get(output_key)
            if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
                raise ValueError(f"edited_candidate sample {index} has invalid role collection")


def validate(operation: str, payload: dict[str, Any]) -> None:
    """Enforce the small security-sensitive subset before facade dispatch."""
    if not isinstance(payload, dict):
        raise ValueError("tool arguments must be an object")
    tool_name = _OPERATIONS.get(operation)
    if tool_name is None:
        raise ValueError(f"unsupported operation: {operation}")
    params = SCHEMAS[tool_name]["parameters"]
    _validate_schema(payload, params)
    if operation == "induce":
        if not payload["intended_use"].strip():
            raise ValueError("intended_use must not be blank")
        if any(not item["question_id"].strip() or not item["text"].strip()
               for item in payload["competency_questions"]):
            raise ValueError("competency_questions values must not be blank")
    elif operation == "review":
        state = payload.get("state")
        edited = payload.get("edited_candidate")
        if state not in {"accepted", "edited", "rejected"}:
            raise ValueError("invalid review state")
        if state == "edited" and not isinstance(edited, dict):
            raise ValueError("edited_candidate is required for edited review")
        if state == "edited" and set(edited) - _CANDIDATE_FIELDS:
            raise ValueError("edited_candidate must be a closed candidate object")
        if state == "edited":
            _validate_edited_candidate(edited)
        if state != "edited" and edited is not None:
            raise ValueError("edited_candidate is only valid for edited review")
    elif operation == "sample":
        action = payload.get("action")
        if action == "execute":
            if not isinstance(payload.get("decision_id"), str) or not payload["decision_id"].strip():
                raise ValueError("decision_id is required for sample execution")
            if "sample_id" in payload:
                raise ValueError("sample_id is only valid for replay")
        elif action == "replay":
            if not isinstance(payload.get("sample_id"), str) or not payload["sample_id"].strip():
                raise ValueError("sample_id is required for sample replay")
            if "decision_id" in payload:
                raise ValueError("decision_id is only valid for execution")
        else:
            raise ValueError("invalid sample action")
    elif operation == "feedback_retest":
        if payload["baseline_decision_id"] == payload["correction_decision_id"]:
            raise ValueError("feedback retest decisions must be distinct")
