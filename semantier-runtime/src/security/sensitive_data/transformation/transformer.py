from __future__ import annotations

from dataclasses import dataclass

from security.sensitive_data.classification.ontology import BoundFinding
from security.sensitive_data.policy.evaluator import PolicyDecisionResult
from semantier.sensitive_governance_contracts.enums import DisclosureAction
from semantier.sensitive_governance_contracts.hashing import canonical_hash


@dataclass(frozen=True)
class TransformationResult:
    transformed_text: str
    transformed_input_hash: str
    transformation_refs: list[dict]


def _replacement(action: DisclosureAction, finding: BoundFinding, index: int) -> str:
    if action == DisclosureAction.ALLOW:
        return ""
    if action == DisclosureAction.MASK:
        return f"[MASKED_{finding.finding_class.value}_{index}]"
    if action == DisclosureAction.TOKENIZE:
        return f"[{finding.finding_class.value}_TOKEN_{index}]"
    if action == DisclosureAction.REDACT:
        return f"[REDACTED_{finding.finding_class.value}_{index}]"
    if action == DisclosureAction.BLOCK:
        return f"[BLOCKED_{finding.finding_class.value}_{index}]"
    raise ValueError(f"unsupported disclosure action: {action}")


def transform_input(
    text: str,
    findings: list[BoundFinding],
    decisions: list[PolicyDecisionResult],
) -> TransformationResult:
    decisions_by_finding = {decision.finding_id: decision for decision in decisions}
    selected: list[tuple[BoundFinding, PolicyDecisionResult]] = []
    occupied_until = -1
    for finding in sorted(findings, key=lambda item: (item.start, -(item.end - item.start))):
        if finding.start < occupied_until:
            continue
        decision = decisions_by_finding[finding.finding_id]
        selected.append((finding, decision))
        occupied_until = finding.end

    chunks: list[str] = []
    refs: list[dict] = []
    cursor = 0
    for index, (finding, decision) in enumerate(selected, start=1):
        chunks.append(text[cursor:finding.start])
        replacement = _replacement(decision.enforcement_action, finding, index)
        if decision.enforcement_action == DisclosureAction.ALLOW:
            chunks.append(text[finding.start:finding.end])
        else:
            chunks.append(replacement)
        refs.append(
            {
                "finding_id": finding.finding_id,
                "policy_decision_ref": decision.policy_decision_ref,
                "action": decision.enforcement_action.value,
                "replacement": replacement or "original",
                "original_value_ref": finding.original_value_ref,
            }
        )
        cursor = finding.end
    chunks.append(text[cursor:])
    transformed = "".join(chunks)
    return TransformationResult(
        transformed_text=transformed,
        transformed_input_hash=canonical_hash({"transformed_input": transformed}),
        transformation_refs=refs,
    )
