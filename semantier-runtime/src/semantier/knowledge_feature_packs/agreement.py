"""Deterministic agreement metrics for sampled Phase 1 double labels."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Sequence


def raw_agreement(left: Sequence[str], right: Sequence[str]) -> float:
    _require_paired(left, right)
    return sum(a == b for a, b in zip(left, right, strict=True)) / len(left)


def cohens_kappa(left: Sequence[str], right: Sequence[str]) -> float:
    _require_paired(left, right)
    observed = raw_agreement(left, right)
    left_counts = Counter(left)
    right_counts = Counter(right)
    total = len(left)
    labels = set(left_counts) | set(right_counts)
    expected = sum(
        (left_counts[label] / total) * (right_counts[label] / total) for label in labels
    )
    return 1.0 if expected == 1.0 and observed == 1.0 else (observed - expected) / (1 - expected)


def weighted_kappa(
    left: Sequence[str],
    right: Sequence[str],
    *,
    ordinal_labels: Sequence[str],
) -> float:
    _require_paired(left, right)
    if len(ordinal_labels) < 2:
        raise ValueError("at least two ordinal labels are required")
    positions = {label: index for index, label in enumerate(ordinal_labels)}
    if set(left) - positions.keys() or set(right) - positions.keys():
        raise ValueError("labels are not present in ordinal_labels")
    denominator = (len(ordinal_labels) - 1) ** 2
    observed_disagreement = sum(
        ((positions[a] - positions[b]) ** 2) / denominator
        for a, b in zip(left, right, strict=True)
    ) / len(left)
    left_counts = Counter(left)
    right_counts = Counter(right)
    total = len(left)
    expected_disagreement = sum(
        (left_counts[a] / total)
        * (right_counts[b] / total)
        * (((positions[a] - positions[b]) ** 2) / denominator)
        for a in ordinal_labels
        for b in ordinal_labels
    )
    if expected_disagreement == 0:
        return 1.0 if observed_disagreement == 0 else 0.0
    return 1 - (observed_disagreement / expected_disagreement)


def agreement_report(
    left: Sequence[str],
    right: Sequence[str],
    *,
    certainty_left: Sequence[str],
    certainty_right: Sequence[str],
    certainty_order: Sequence[str],
) -> dict[str, float]:
    return {
        "raw_agreement": raw_agreement(left, right),
        "cohens_kappa": cohens_kappa(left, right),
        "weighted_kappa": weighted_kappa(
            certainty_left,
            certainty_right,
            ordinal_labels=certainty_order,
        ),
    }


def confidence_calibration(events: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare proposal confidence with later human accept/edit/reject outcomes."""
    proposals: dict[str, float] = {}
    buckets: dict[int, Counter[str]] = defaultdict(Counter)
    for event in events:
        proposal = event.get("machine_proposal")
        if proposal and proposal.get("confidence") is not None:
            proposals[str(event["case_id"])] = float(proposal["confidence"])
        label = event.get("human_label")
        if label and str(event["case_id"]) in proposals:
            bucket = min(9, int(proposals[str(event["case_id"])] * 10))
            buckets[bucket][str(label["decision"])] += 1
    return [
        {
            "confidence_lower": bucket / 10,
            "confidence_upper": (bucket + 1) / 10,
            "accept": counts["accept"],
            "edit": counts["edit"],
            "reject": counts["reject"],
            "total": sum(counts.values()),
        }
        for bucket, counts in sorted(buckets.items())
    ]


def _require_paired(left: Sequence[str], right: Sequence[str]) -> None:
    if not left or len(left) != len(right):
        raise ValueError("paired non-empty labels are required")
