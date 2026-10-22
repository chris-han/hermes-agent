"""Deterministic evaluation for proposal-only LLM regex improvements."""

from __future__ import annotations

import re
from typing import Any

from eos.artifact_hashing import content_hash


def _require_bounded_regex(pattern: str) -> None:
    """Reject constructs whose Python backtracking cost is not bounded here."""
    escaped = False
    in_class = False
    for index, character in enumerate(pattern):
        if escaped:
            if character.isdigit() and character != "0":
                raise ValueError("unsafe regex: backreferences are not supported")
            escaped = False
            continue
        if character == "\\":
            escaped = True
            continue
        if character == "[":
            in_class = True
            continue
        if character == "]":
            in_class = False
            continue
        if in_class:
            continue
        if character in {"*", "+", "{"}:
            raise ValueError("unsafe regex: unbounded or counted repetition is not supported")
        if character == "?" and (index == 0 or pattern[index - 1] != "("):
            raise ValueError("unsafe regex: optional or lazy repetition is not supported")


def evaluate_regex_proposal(
    *,
    concept_label: str,
    regex: str,
    rationale: str,
    negative_examples: list[str],
    positive_examples: list[str],
    feedback_refs: list[str],
) -> dict[str, Any]:
    if not concept_label or not regex or len(regex) > 500:
        raise ValueError("concept_label and a bounded regex are required")
    if len(positive_examples) > 50 or len(negative_examples) > 50:
        raise ValueError("regex evaluation accepts at most 50 examples per class")
    if any(len(value) > 2_000 for value in [*positive_examples, *negative_examples]):
        raise ValueError("regex evaluation examples must not exceed 2000 characters")
    _require_bounded_regex(regex)
    try:
        compiled = re.compile(regex)
    except re.error as exc:
        raise ValueError(f"invalid regex: {exc}") from exc
    metrics = {
        "positive_total": len(positive_examples),
        "positive_matched": sum(bool(compiled.search(text)) for text in positive_examples),
        "negative_total": len(negative_examples),
        "negative_matched": sum(bool(compiled.search(text)) for text in negative_examples),
    }
    payload = {
        "schema_version": "regex_improvement_proposal.v1",
        "concept_label": concept_label,
        "regex": regex,
        "rationale": rationale,
        "feedback_refs": list(feedback_refs),
        "metrics": metrics,
        "activation_state": "proposal_only",
        "eligible_for_review": (
            metrics["positive_total"] > 0
            and metrics["positive_matched"] == metrics["positive_total"]
            and metrics["negative_matched"] == 0
        ),
    }
    payload["proposal_hash"] = f"sha256:{content_hash(payload)}"
    return payload
