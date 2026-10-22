"""Deterministic execution of an authoritative corpus graph-schema candidate."""

from __future__ import annotations

import re
from typing import Any

from eos.artifact_hashing import content_hash


class CorpusGraphSchemaSampleExecutor:
    """Compile candidate-owned rules and execute them over canonical elements."""

    engine_ref = "corpus-graph-schema-sample-executor@1.0.0"

    def execution_contract(self, candidate: dict[str, Any]) -> dict[str, Any]:
        hints = candidate.get("extraction_contract_hints")
        if not isinstance(hints, dict):
            raise ValueError("authoritative candidate has no extraction contract")
        paragraph_rules = hints.get("paragraph_selector_rules")
        event_rule = hints.get("event_extraction_rule")
        event_types = candidate.get("proposed_event_types")
        relations = candidate.get("proposed_relation_types")
        if not isinstance(paragraph_rules, list) or not paragraph_rules:
            raise ValueError("authoritative candidate has no executable paragraph selectors")
        if not isinstance(event_rule, dict) or not event_rule:
            raise ValueError("authoritative candidate has no executable event rule")
        if not isinstance(event_types, list) or not event_types:
            raise ValueError("authoritative candidate has no executable event types")
        if not isinstance(relations, list):
            raise ValueError("authoritative candidate relation types are incomplete")
        return {
            "paragraph_selector_rules": paragraph_rules,
            "event_extraction_rule": event_rule,
            "event_types": event_types,
            "relation_types": relations,
        }

    def execute(
        self, *, parsed_document: dict[str, Any], candidate: dict[str, Any]
    ) -> dict[str, Any]:
        contract = self.execution_contract(candidate)
        document_hash = parsed_document.get("document_hash")
        profile = candidate.get("source_structure_profile") or {}
        if document_hash != profile.get("document_hash"):
            raise ValueError("canonical source hash does not match authoritative candidate")
        compiled: list[tuple[str, re.Pattern[str], int]] = []
        for index, rule in enumerate(contract["paragraph_selector_rules"]):
            if not isinstance(rule, dict) or not isinstance(rule.get("role_id"), str):
                raise ValueError("paragraph selector rule is incomplete")
            pattern = rule.get("pattern")
            if not isinstance(pattern, str) or not pattern:
                raise ValueError("paragraph selector pattern is incomplete")
            try:
                compiled.append((rule["role_id"], re.compile(pattern), index))
            except re.error as exc:
                raise ValueError("paragraph selector pattern is invalid") from exc

        observations: list[dict[str, str]] = []
        for element in parsed_document.get("elements") or []:
            anchor = element.get("anchor") or {}
            anchor_id = anchor.get("anchor_id")
            if anchor.get("document_part") != "paragraph" or not anchor_id:
                continue
            text = str(element.get("text") or "").strip()
            for role_id, pattern, _ in compiled:
                match = pattern.search(text)
                if match:
                    value = match.groupdict().get("value") or match.group(0)
                    observations.append(
                        {"role_id": role_id, "value": value.strip(), "anchor_id": str(anchor_id)}
                    )
        samples, ambiguities = self._events(observations, contract)
        output = {
            "schema_version": "corpus_graph_schema_sample_result.v1",
            "document_hash": document_hash,
            "events": samples,
            "unresolved_ambiguities": ambiguities,
        }
        return {**output, "result_hash": f"sha256:{content_hash(output)}"}

    @staticmethod
    def _events(
        observations: list[dict[str, str]], contract: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        rule = contract["event_extraction_rule"]
        boundary = rule.get("boundary_role")
        required = tuple(rule.get("required_roles") or [])
        inherited = tuple(rule.get("inherit_context_roles") or [])
        attached = tuple(rule.get("attach_following_roles") or [])
        keys = rule.get("output_key_by_role")
        limit = rule.get("sample_limit")
        if not isinstance(boundary, str) or not required or not isinstance(keys, dict):
            raise ValueError("event extraction rule is incomplete")
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise ValueError("event extraction sample limit is invalid")
        event_specs = contract["event_types"]
        relation_specs = contract["relation_types"]
        event_ids = {item.get("type_id") for item in event_specs if isinstance(item, dict)}
        if len(event_ids) != len(event_specs) or None in event_ids:
            raise ValueError("event type contract is inconsistent")
        for relation in relation_specs:
            if not isinstance(relation, dict) or relation.get("source_type") not in event_ids:
                raise ValueError("relation type contract is inconsistent with event types")
            if relation.get("target_role") not in set(required + inherited + attached):
                raise ValueError("relation target role is inconsistent with event rule")

        grouped: dict[str, list[dict[str, str]]] = {}
        for item in observations:
            grouped.setdefault(item["anchor_id"], []).append(item)
        # Observations follow canonical source-element order. Content-addressed
        # anchors carry identity, not a sortable paragraph position.
        anchor_order = {anchor: index for index, anchor in enumerate(grouped)}
        context: dict[str, list[dict[str, str]]] = {}
        current: dict[str, Any] | None = None
        samples: list[dict[str, Any]] = []
        ambiguities: list[dict[str, Any]] = []

        def finalize() -> None:
            nonlocal current
            if current is None:
                return
            roles = current["roles"]
            missing = [role for role in required if not roles.get(role)]
            matches = [s for s in event_specs if set(s.get("required_roles") or []) <= set(roles)]
            if missing or len(matches) != 1:
                ambiguities.append({
                    "ambiguity_id": "event_required_roles_missing" if missing else "event_type_ambiguous",
                    "missing_roles": missing,
                    "evidence_refs": [current["anchor_id"]],
                })
                current = None
                return
            spec = matches[0]
            event_id = f"event:{spec['type_id']}:{current['anchor_id']}"
            sample: dict[str, Any] = {"event_id": event_id, "event_type": spec["type_id"]}
            evidence: list[str] = []
            for role in required:
                sample[role] = roles[role][0]["value"]
            for role in inherited + attached:
                if role not in keys:
                    raise ValueError("event output mapping is incomplete")
                values = roles.get(role, [])
                sample[keys[role]] = [v["value"] for v in values]
                for value in values:
                    if value["anchor_id"] not in evidence:
                        evidence.append(value["anchor_id"])
            if current["anchor_id"] not in evidence:
                evidence.append(current["anchor_id"])
            sample["evidence_refs"] = sorted(evidence, key=anchor_order.__getitem__)
            sample["relations"] = []
            for relation in relation_specs:
                if relation["source_type"] != spec["type_id"]:
                    continue
                for value in roles.get(relation["target_role"], []):
                    sample["relations"].append({
                        "type_id": relation["type_id"], "source_ref": event_id,
                        "target_value": value["value"],
                        "target_evidence_ref": value["anchor_id"],
                    })
            samples.append(sample)
            current = None

        for anchor_id in grouped:
            matches = grouped[anchor_id]
            if any(item["role_id"] == boundary for item in matches):
                finalize()
                for role in inherited:
                    values = [item for item in matches if item["role_id"] == role]
                    if values:
                        context[role] = values
                roles = {role: list(values) for role, values in context.items()}
                for item in matches:
                    if item["role_id"] not in inherited:
                        roles.setdefault(item["role_id"], []).append(item)
                current = {"anchor_id": anchor_id, "roles": roles}
                continue
            for role in inherited:
                values = [item for item in matches if item["role_id"] == role]
                if values:
                    context[role] = values
            if current is not None:
                for role in attached:
                    current["roles"].setdefault(role, []).extend(
                        item for item in matches if item["role_id"] == role
                    )
        finalize()
        return samples[:limit], ambiguities
