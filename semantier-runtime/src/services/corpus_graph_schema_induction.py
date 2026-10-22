"""Deterministic, profile-driven corpus graph-schema induction kernel."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Pattern

from contracts.corpus_graph_schema import CorpusGraphSchemaCandidate, SCHEMA_VERSION
from eos.artifact_hashing import content_hash

DEFAULT_PROFILE = Path(__file__).resolve().parents[1] / "semantier/knowledge_feature_packs/profiles/tender_corpus_schema_induction.v0_1.json"
_IDENTIFIER = re.compile(r"^[\x21-\x7e]+$")
_OUTPUT_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class CompiledParagraphRolePattern:
    role_id: str
    regex: Pattern[str]


@dataclass
class _PendingEvent:
    anchor_id: str
    roles: dict[str, list[dict[str, str]]]


@dataclass(frozen=True)
class _LegacyInductionResult:
    relation_types: tuple[dict[str, Any], ...]
    frame_types: tuple[dict[str, Any], ...]
    representation_decisions: tuple[dict[str, Any], ...]
    extraction_hints: dict[str, Any]
    ambiguities: tuple[dict[str, Any], ...]
    signal_evidence: dict[str, list[str]]


def _require_identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{field} must be a nonempty ASCII identifier")
    return value


def _require_role_list(
    value: Any, field: str, *, nonempty: bool = False
) -> tuple[str, ...]:
    if not isinstance(value, list) or (nonempty and not value):
        qualifier = "nonempty " if nonempty else ""
        raise ValueError(f"{field} must be a {qualifier}list")
    return tuple(
        _require_identifier(item, f"{field}[{index}]")
        for index, item in enumerate(value)
    )


def compile_paragraph_role_patterns(
    profile: dict[str, Any],
) -> tuple[CompiledParagraphRolePattern, ...]:
    result = []
    for index, item in enumerate(profile.get("paragraph_role_patterns") or []):
        if not isinstance(item, dict):
            raise ValueError(f"paragraph_role_patterns[{index}] must be an object")
        role_id = _require_identifier(
            item.get("role_id"), f"paragraph_role_patterns[{index}].role_id"
        )
        try:
            regex = re.compile(str(item.get("pattern") or ""))
        except re.error as exc:
            raise ValueError(f"paragraph_role_patterns[{index}].pattern is invalid") from exc
        if "value" not in regex.groupindex:
            raise ValueError(f"paragraph_role_patterns[{index}].pattern is invalid")
        result.append(CompiledParagraphRolePattern(role_id, regex))
    return tuple(result)


def match_paragraph_roles(
    *, text: str, compiled_patterns: tuple[CompiledParagraphRolePattern, ...]
) -> tuple[dict[str, str], ...]:
    result = []
    for item in compiled_patterns:
        match = item.regex.search(text)
        if match is not None and (value := match.group("value").strip()):
            result.append(
                {
                    "role_id": item.role_id,
                    "value": value,
                    "pattern": item.regex.pattern,
                }
            )
    return tuple(result)


def _unique_append(values: list[str], value: str) -> None:
    if value and value not in values:
        values.append(value)


def _anchor_key(anchor: str) -> tuple[int, str]:
    try:
        return int(anchor.rsplit(":", 1)[1]), anchor
    except (IndexError, ValueError):
        return 2**31, anchor


def _validate_event_profile(profile: dict[str, Any]) -> None:
    if "event_extraction" not in profile:
        return
    rules = profile["event_extraction"]
    if not isinstance(rules, dict):
        raise ValueError("event_extraction must be an object")
    known_roles = {
        str(item["role_id"])
        for item in profile.get("paragraph_role_patterns") or []
        if isinstance(item, dict) and item.get("role_id")
    }
    node_mapping = profile.get("node_type_by_role") or {}
    if not isinstance(node_mapping, dict):
        raise ValueError("node_type_by_role must be an object")
    known_roles.update(map(str, node_mapping))
    header_mapping = profile.get("semantic_role_by_header") or {}
    if isinstance(header_mapping, dict):
        known_roles.update(map(str, header_mapping.values()))

    boundary = _require_identifier(
        rules.get("boundary_role"), "event_extraction.boundary_role"
    )
    required = _require_role_list(
        rules.get("required_roles"),
        "event_extraction.required_roles",
        nonempty=True,
    )
    inherited = _require_role_list(
        rules.get("inherit_context_roles"),
        "event_extraction.inherit_context_roles",
    )
    attached = _require_role_list(
        rules.get("attach_following_roles"),
        "event_extraction.attach_following_roles",
    )
    referenced_roles = {boundary, *required, *inherited, *attached}
    unknown = sorted(referenced_roles - known_roles)
    if unknown:
        raise ValueError(f"event_extraction references unknown roles: {unknown}")
    sample_limit = rules.get("sample_limit")
    if isinstance(sample_limit, bool) or not isinstance(sample_limit, int) or sample_limit <= 0:
        raise ValueError("event_extraction.sample_limit must be a positive integer")

    output_keys = rules.get("output_key_by_role")
    if not isinstance(output_keys, dict):
        raise ValueError("event_extraction.output_key_by_role must be an object")
    output_roles = set(inherited) | set(attached)
    if set(output_keys) != output_roles:
        raise ValueError("event_extraction.output_key_by_role must map inherited and following roles")
    values = []
    for role in sorted(output_keys):
        value = output_keys[role]
        if not isinstance(value, str) or not _OUTPUT_KEY.fullmatch(value):
            raise ValueError(f"event_extraction.output_key_by_role.{role} is invalid")
        values.append(value)
    reserved_sample_keys = {
        "event_id",
        "event_type_id",
        "evidence_refs",
        "relations",
        boundary,
        *required,
    }
    if len(values) != len(set(values)) or set(values) & reserved_sample_keys:
        raise ValueError("event_extraction.output_key_by_role values must be unique and safe")

    event_types = profile.get("event_types")
    if not isinstance(event_types, list) or not event_types:
        raise ValueError("event_types must be a nonempty list")
    event_ids: list[str] = []
    for index, item in enumerate(event_types):
        if not isinstance(item, dict):
            raise ValueError(f"event_types[{index}] must be an object")
        event_ids.append(_require_identifier(item.get("type_id"), f"event_types[{index}].type_id"))
        event_roles = _require_role_list(item.get("required_roles"), f"event_types[{index}].required_roles", nonempty=True)
        if unknown_event_roles := sorted(set(event_roles) - known_roles):
            raise ValueError(f"event_types[{index}] references unknown roles: {unknown_event_roles}")
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("event_types.type_id values must be unique")

    relations = profile.get("role_relations")
    if not isinstance(relations, list):
        raise ValueError("role_relations must be a list")
    relation_ids: list[str] = []
    for index, item in enumerate(relations):
        if not isinstance(item, dict):
            raise ValueError(f"role_relations[{index}] must be an object")
        relation_ids.append(_require_identifier(item.get("type_id"), f"role_relations[{index}].type_id"))
        source_type = _require_identifier(item.get("source_type"), f"role_relations[{index}].source_type")
        target_role = _require_identifier(item.get("target_role"), f"role_relations[{index}].target_role")
        if source_type not in event_ids:
            raise ValueError(f"role_relations[{index}].source_type is unknown")
        if target_role not in known_roles:
            raise ValueError(f"role_relations[{index}].target_role is unknown")
    if len(relation_ids) != len(set(relation_ids)):
        raise ValueError("role_relations.type_id values must be unique")

    representation = profile.get("representation")
    if not isinstance(representation, dict) or representation.get("type") != "event":
        raise ValueError("representation.type must be event")
    if not isinstance(representation.get("binary_triple_sufficient"), bool):
        raise ValueError("representation.binary_triple_sufficient must be boolean")
    if representation["binary_triple_sufficient"] is not True:
        raise ValueError(
            "representation.binary_triple_sufficient must be true for event representation"
        )
    if not str(representation.get("reason") or "").strip():
        raise ValueError("representation.reason is required")


class CorpusGraphSchemaInductionKernel:
    """Infer a reviewable schema from structural roles and profile-owned examples."""

    def __init__(self, profile_path: str | Path = DEFAULT_PROFILE) -> None:
        self.profile_path = Path(profile_path).resolve()
        self.profile = json.loads(self.profile_path.read_text(encoding="utf-8"))
        self.paragraph_patterns = compile_paragraph_role_patterns(self.profile)
        _validate_event_profile(self.profile)

    def induce(
        self,
        *,
        parsed_document: dict[str, Any],
        source_ref: str,
        intended_use: str,
        competency_questions: tuple[dict[str, str], ...] = (),
    ) -> CorpusGraphSchemaCandidate:
        elements = list(parsed_document.get("elements") or [])
        if not elements:
            raise ValueError("parsed_document.elements must not be empty")
        roles: dict[str, list[str]] = {}
        columns: list[dict[str, Any]] = []
        paragraphs: list[dict[str, str]] = []
        observations: list[dict[str, str]] = []
        tables: set[int] = set()
        header_roles = dict(self.profile.get("semantic_role_by_header") or {})
        for element in elements:
            anchor = dict(element.get("anchor") or {})
            anchor_id = str(anchor.get("anchor_id") or "")
            text = str(element.get("text") or "").strip()
            if anchor.get("table_index") is not None:
                tables.add(int(anchor["table_index"]))
            role = header_roles.get(text)
            if role is not None and anchor.get("row_index") == 0 and anchor_id:
                _unique_append(roles.setdefault(role, []), anchor_id)
                columns.append(
                    {
                        "role_id": role,
                        "table_index": anchor.get("table_index"),
                        "column_index": anchor.get("cell_index"),
                        "header_text": text,
                        "evidence_ref": anchor_id,
                    }
                )
            if anchor.get("document_part") != "paragraph" or not anchor_id:
                continue
            for match in match_paragraph_roles(text=text, compiled_patterns=self.paragraph_patterns):
                _unique_append(roles.setdefault(match["role_id"], []), anchor_id)
                observation = {**match, "anchor_id": anchor_id}
                observations.append(observation)
                paragraphs.append(observation)
        if not roles:
            raise ValueError("no semantic source roles were induced from the corpus")

        semantic_roles = tuple(
            self._semantic_role(role, refs, columns, paragraphs)
            for role, refs in sorted(roles.items())
        )
        node_types = tuple(
            {"type_id": type_id, "derived_from_role": role}
            for role, type_id in sorted(
                dict(self.profile.get("node_type_by_role") or {}).items()
            )
            if role in roles
        )
        event_rules = dict(self.profile.get("event_extraction") or {})
        if event_rules:
            samples, ambiguities, supported_event_ids = self._event_samples(
                observations, event_rules
            )
            supported_relation_ids = {
                relation["type_id"]
                for sample in samples
                for relation in sample["relations"]
            }
            relation_types = tuple(
                dict(item)
                for item in self.profile.get("role_relations") or []
                if item.get("type_id") in supported_relation_ids
            )
            event_types = tuple(
                dict(item)
                for item in self.profile.get("event_types") or []
                if item.get("type_id") in supported_event_ids
            )
            representation = dict(self.profile.get("representation") or {})
            sample_evidence = {
                ref for sample in samples for ref in sample["evidence_refs"]
            }
            role_evidence = {ref for refs in roles.values() for ref in refs}
            decision_evidence = sorted(
                sample_evidence or role_evidence, key=_anchor_key
            )
            decisions = (
                {
                    "initial_representation": "binary_triple",
                    "binary_triple_sufficient": representation[
                        "binary_triple_sufficient"
                    ],
                    "selected_representation": "event",
                    "reason": str(representation["reason"]),
                    "evidence_refs": decision_evidence,
                },
            )
            frames: tuple[dict[str, Any], ...] = ()
            hints = {
                "column_selectors": columns,
                "paragraph_selectors": paragraphs,
                "table_selector_rules": dict(header_roles),
                "paragraph_selector_rules": [
                    dict(item) for item in self.profile.get("paragraph_role_patterns") or []
                ],
                "event_extraction_rule": dict(event_rules),
                "relation_templates": [dict(item) for item in relation_types],
                "representation_type": "event",
                "frame_slots": [],
                "sample_event_instances": samples,
            }
            signal_evidence: dict[str, list[str]] = {}
        else:
            legacy = self._legacy(elements, roles, columns, paragraphs)
            relation_types = legacy.relation_types
            event_types = ()
            frames = legacy.frame_types
            decisions = legacy.representation_decisions
            hints = legacy.extraction_hints
            ambiguities = legacy.ambiguities
            signal_evidence = legacy.signal_evidence

        evidence = sorted(
            {
                *(ref for refs in roles.values() for ref in refs),
                *(ref for refs in signal_evidence.values() for ref in refs),
                *(
                    ref
                    for ambiguity in ambiguities
                    for ref in ambiguity.get("evidence_refs", [])
                ),
            },
            key=_anchor_key,
        )
        source_profile = {
            "document_kind": self.profile.get("source_kind", "docx"),
            "element_count": len(elements),
            "table_count": len(tables),
            "profile_id": str(self.profile["profile_id"]),
            "source_ref": source_ref,
            "document_hash": parsed_document.get("document_hash"),
            "observed_column_roles": sorted(
                {item["role_id"] for item in columns}
            ),
            "observed_paragraph_roles": sorted(
                {item["role_id"] for item in paragraphs}
            ),
        }
        payload = {
            "schema_version": SCHEMA_VERSION,
            "source_structure_profile": source_profile,
            "intended_use": intended_use,
            "competency_questions": [dict(item) for item in competency_questions],
            "semantic_role_proposals": list(semantic_roles),
            "proposed_node_types": list(node_types),
            "proposed_relation_types": list(relation_types),
            "proposed_event_types": list(event_types),
            "proposed_frame_types": list(frames),
            "representation_decisions": list(decisions),
            "extraction_contract_hints": hints,
            "unresolved_ambiguities": list(ambiguities),
            "source_evidence_refs": evidence,
        }
        digest = content_hash(payload)
        return CorpusGraphSchemaCandidate(
            candidate_id=f"cgsc_{digest[:24]}",
            candidate_hash=f"sha256:{digest}",
            source_structure_profile=source_profile,
            intended_use=intended_use,
            semantic_role_proposals=semantic_roles,
            proposed_node_types=node_types,
            proposed_relation_types=relation_types,
            proposed_frame_types=frames,
            representation_decisions=decisions,
            extraction_contract_hints=hints,
            unresolved_ambiguities=tuple(ambiguities),
            source_evidence_refs=tuple(evidence),
            proposed_event_types=event_types,
            competency_questions=tuple(dict(item) for item in competency_questions),
        )

    @staticmethod
    def _semantic_role(
        role: str,
        refs: list[str],
        columns: list[dict[str, Any]],
        paragraphs: list[dict[str, str]],
    ) -> dict[str, Any]:
        role_columns = [item for item in columns if item["role_id"] == role]
        role_paragraphs = [item for item in paragraphs if item["role_id"] == role]
        if role_paragraphs and not role_columns:
            selector: dict[str, Any] = {"kind": "paragraph_pattern", "matches": role_paragraphs}
        elif role_columns and not role_paragraphs:
            selector = {"kind": "table_column", "columns": role_columns}
        else:
            selector = {"kind": "combined", "table_columns": role_columns, "paragraph_patterns": role_paragraphs}
        return {
            "role_id": role,
            "source_selector": selector,
            "evidence_refs": sorted(set(refs), key=_anchor_key),
        }

    def _event_samples(
        self, observations: list[dict[str, str]], rules: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
        boundary = str(rules["boundary_role"])
        required = tuple(map(str, rules.get("required_roles") or []))
        inherited = tuple(map(str, rules.get("inherit_context_roles") or []))
        attached = tuple(map(str, rules.get("attach_following_roles") or []))
        output_keys = dict(rules["output_key_by_role"])
        limit = int(rules["sample_limit"])
        event_specs = tuple(dict(item) for item in self.profile.get("event_types") or [])
        relation_specs = tuple(dict(item) for item in self.profile.get("role_relations") or [])
        grouped: dict[str, list[dict[str, str]]] = {}
        for item in observations:
            grouped.setdefault(item["anchor_id"], []).append(item)
        context: dict[str, list[dict[str, str]]] = {}
        current: _PendingEvent | None = None
        samples: list[dict[str, Any]] = []
        ambiguities: list[dict[str, Any]] = []
        supported_event_ids: set[str] = set()

        def finalize() -> None:
            nonlocal current
            if current is None:
                return
            missing = [role for role in required if not current.roles.get(role)]
            if missing:
                ambiguities.append({"ambiguity_id": "event_required_roles_missing", "missing_roles": missing, "evidence_refs": [current.anchor_id]})
            elif len(samples) < limit:
                event_spec = next(
                    (
                        item
                        for item in event_specs
                        if set(item["required_roles"]) <= set(current.roles)
                    ),
                    None,
                )
                if event_spec is None:
                    event_missing = sorted(
                        {
                            role
                            for item in event_specs
                            for role in item["required_roles"]
                            if not current.roles.get(role)
                        }
                    )
                    ambiguities.append(
                        {
                            "ambiguity_id": "event_required_roles_missing",
                            "missing_roles": event_missing,
                            "evidence_refs": [current.anchor_id],
                        }
                    )
                    current = None
                    return
                event_id = f"event:{event_spec['type_id']}:{current.anchor_id}"
                supported_event_ids.add(str(event_spec["type_id"]))
                sample: dict[str, Any] = {"event_id": event_id}
                evidence: list[str] = []
                for role in required:
                    sample[role] = current.roles[role][0]["value"]
                for role in inherited:
                    values = current.roles.get(role, [])
                    sample[output_keys[role]] = [item["value"] for item in values]
                    for item in values: _unique_append(evidence, item["anchor_id"])
                _unique_append(evidence, current.anchor_id)
                for role in attached:
                    values = current.roles.get(role, [])
                    sample[output_keys[role]] = [item["value"] for item in values]
                    for item in values: _unique_append(evidence, item["anchor_id"])
                sample["evidence_refs"] = evidence
                relations: list[dict[str, str]] = []
                for relation in relation_specs:
                    if relation["source_type"] != event_spec["type_id"]:
                        continue
                    target_role = str(relation["target_role"])
                    for value in current.roles.get(target_role, []):
                        relations.append(
                            {
                                "type_id": str(relation["type_id"]),
                                "source_ref": event_id,
                                "target_value": value["value"],
                                "target_evidence_ref": value["anchor_id"],
                            }
                        )
                sample["relations"] = relations
                samples.append(sample)
            current = None

        for anchor, matches in grouped.items():
            if any(item["role_id"] == boundary for item in matches):
                finalize()
                for role in inherited:
                    values = [item for item in matches if item["role_id"] == role]
                    if values:
                        context[role] = values
                event_roles = {role: list(values) for role, values in context.items()}
                for item in matches:
                    role = item["role_id"]
                    if role not in inherited:
                        event_roles.setdefault(role, []).append(item)
                current = _PendingEvent(anchor_id=anchor, roles=event_roles)
                continue
            for role in inherited:
                values = [item for item in matches if item["role_id"] == role]
                if values:
                    context[role] = values
            if current is not None:
                for role in attached:
                    following = [item for item in matches if item["role_id"] == role]
                    current.roles.setdefault(role, []).extend(following)
        finalize()
        return samples, ambiguities, supported_event_ids

    def _legacy(
        self,
        elements: list[dict[str, Any]],
        roles: dict[str, list[str]],
        columns: list[dict[str, Any]],
        paragraphs: list[dict[str, str]],
    ) -> _LegacyInductionResult:
        relations = tuple(
            dict(item)
            for item in self.profile.get("role_relations") or []
            if item.get("source_role") in roles
            and item.get("target_role") in roles
        )
        signal_evidence: dict[str, list[str]] = {}
        dimensions: set[str] = set()
        for signal in self.profile.get("nary_signals") or []:
            found = []
            for element in elements:
                text = str(element.get("text") or "")
                patterns = signal.get("patterns") or []
                if any(str(pattern) in text for pattern in patterns):
                    anchor = element.get("anchor") or {}
                    anchor_id = str(anchor.get("anchor_id") or "")
                    _unique_append(found, anchor_id)
                    if len(found) == 5:
                        break
            if found:
                signal_evidence[str(signal["signal_id"])] = found
                dimensions.add(str(signal["dimension"]))
        binary = len(dimensions) < 2
        spec = dict(self.profile.get("nary_representation") or {})
        representation_type = (
            "binary_triple" if binary else str(spec["type_id"])
        )
        frame_slots = list(spec.get("slots") or []) if not binary else []
        if binary:
            frames: tuple[dict[str, Any], ...] = ()
        else:
            frames = (
                {
                    "type_id": str(spec["type_id"]),
                    "slots": list(spec.get("slots") or []),
                    "selected_because": sorted(dimensions),
                },
            )
        decision_evidence = sorted(
            {ref for refs in signal_evidence.values() for ref in refs}
        )
        decisions = (
            {
                "initial_representation": "binary_triple",
                "binary_triple_sufficient": binary,
                "selected_representation": representation_type,
                "material_dimensions": sorted(dimensions),
                "evidence_refs": decision_evidence,
            },
        )
        hints = {
            "column_selectors": columns,
            "paragraph_selectors": paragraphs,
            "table_selector_rules": dict(self.profile.get("semantic_role_by_header") or {}),
            "paragraph_selector_rules": [
                dict(item) for item in self.profile.get("paragraph_role_patterns") or []
            ],
            "relation_templates": [dict(item) for item in relations],
            "representation_type": representation_type,
            "frame_slots": frame_slots,
        }
        ambiguities = (
            {
                "ambiguity_id": "legal_disposition_not_inferred",
                "status": "requires_human_review",
                "description": (
                    "Source examples indicate detection intent but do not establish "
                    "a binding legal disposition."
                ),
            },
        )
        return _LegacyInductionResult(
            relation_types=relations,
            frame_types=frames,
            representation_decisions=decisions,
            extraction_hints=hints,
            ambiguities=ambiguities,
            signal_evidence=signal_evidence,
        )
