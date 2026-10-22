"""Phase1-001 manifest and deterministic finding-evaluation primitives."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping

DatasetRole = Literal["grounding", "heldout"]
LearningDecision = Literal["GO", "STOP_REVISE", "SPLIT_FIX"]


def _require_utc_timestamp(value: str) -> None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("review timestamps must be ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("review timestamps must be timezone-aware UTC")


def _canonical(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def manifest_hash(payload: Mapping[str, Any]) -> str:
    body = dict(payload)
    body.pop("manifest_hash", None)
    return f"sha256:{hashlib.sha256(_canonical(body)).hexdigest()}"


def build_review_manifest(
    *,
    manifest_id: str,
    manifest_version: str,
    dataset_role: DatasetRole,
    reviewer_user_id: str,
    reviewed_at: str,
    documents: Iterable[Mapping[str, str]],
    labels: Iterable[Mapping[str, Any]],
    frozen_before_model_output: bool = False,
) -> "EvaluationManifest":
    """Turn an externally completed review record into a frozen manifest.

    The function does not infer labels from model output.  Callers must supply
    reviewer identity, exact document hashes, and every human label.
    """
    if not reviewer_user_id.strip():
        raise ValueError("reviewer_user_id is required")
    if not reviewed_at.strip():
        raise ValueError("reviewed_at is required")
    _require_utc_timestamp(reviewed_at)
    document_rows = [dict(row) for row in documents]
    if not document_rows:
        raise ValueError("at least one reviewed document is required")
    refs = [str(row.get("document_ref") or "") for row in document_rows]
    hashes = [str(row.get("document_hash") or "") for row in document_rows]
    if any(not ref or not digest.startswith("sha256:") for ref, digest in zip(refs, hashes)):
        raise ValueError("reviewed documents require exact refs and sha256 hashes")
    payload = {
        "manifest_id": manifest_id,
        "manifest_version": manifest_version,
        "dataset_role": dataset_role,
        "created_at": reviewed_at,
        "reviewer_user_id": reviewer_user_id,
        "review_completed_at": reviewed_at,
        "source_document_refs": refs,
        "source_document_hashes": hashes,
        "labels": [dict(label) for label in labels],
        "frozen_before_model_output": frozen_before_model_output,
    }
    payload["manifest_hash"] = manifest_hash(payload)
    return EvaluationManifest.from_dict(payload)


@dataclass(frozen=True)
class EvaluationManifest:
    manifest_id: str
    manifest_version: str
    dataset_role: DatasetRole
    created_at: str
    reviewer_user_id: str
    review_completed_at: str
    source_document_refs: tuple[str, ...]
    source_document_hashes: tuple[str, ...]
    labels: tuple[dict[str, Any], ...]
    manifest_hash: str
    frozen_before_model_output: bool = False

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvaluationManifest":
        required = {
            "manifest_id", "manifest_version", "dataset_role", "created_at",
            "reviewer_user_id", "review_completed_at", "source_document_refs",
            "source_document_hashes", "labels", "manifest_hash",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"evaluation manifest missing fields: {missing}")
        role = payload["dataset_role"]
        if role not in {"grounding", "heldout"}:
            raise ValueError("dataset_role must be grounding or heldout")
        refs = tuple(str(value) for value in payload["source_document_refs"])
        hashes = tuple(str(value) for value in payload["source_document_hashes"])
        if not refs or len(refs) != len(hashes):
            raise ValueError("source refs and hashes must be non-empty and aligned")
        labels = tuple(dict(label) for label in payload["labels"])
        document_hash_by_ref = dict(zip(refs, hashes))
        seen: set[tuple[str, str]] = set()
        for label in labels:
            for field in ("case_id", "document_ref", "finding_id", "finding_type", "expected_outcome", "expected_span_anchor"):
                if field not in label:
                    raise ValueError(f"manifest label missing {field}")
            if label["expected_outcome"] not in {"finding", "no_finding"}:
                raise ValueError("expected_outcome must be finding or no_finding")
            key = (str(label["document_ref"]), str(label["finding_id"]))
            if key in seen:
                raise ValueError(f"duplicate manifest label: {key}")
            seen.add(key)
            anchor = label["expected_span_anchor"]
            if not isinstance(anchor, Mapping):
                raise ValueError("expected_span_anchor must be an object")
            for field in ("sourceRef", "sourceHash", "locator"):
                if not str(anchor.get(field) or ""):
                    raise ValueError(f"expected span anchor missing {field}")
            document_ref = str(label["document_ref"])
            if document_ref not in document_hash_by_ref:
                raise ValueError("manifest label references an unlisted document")
            if str(anchor["sourceRef"]) != document_ref or str(anchor["sourceHash"]) != document_hash_by_ref[document_ref]:
                raise ValueError("manifest label anchor does not match document pin")
        _require_utc_timestamp(str(payload["created_at"]))
        _require_utc_timestamp(str(payload["review_completed_at"]))
        expected_hash = manifest_hash(payload)
        if str(payload["manifest_hash"]) != expected_hash:
            raise ValueError("evaluation manifest hash mismatch")
        frozen = bool(payload.get("frozen_before_model_output", False))
        if role == "heldout" and not frozen:
            raise ValueError("heldout manifest must declare frozen_before_model_output=true")
        return cls(
            manifest_id=str(payload["manifest_id"]),
            manifest_version=str(payload["manifest_version"]),
            dataset_role=role,
            created_at=str(payload["created_at"]),
            reviewer_user_id=str(payload["reviewer_user_id"]),
            review_completed_at=str(payload["review_completed_at"]),
            source_document_refs=refs,
            source_document_hashes=hashes,
            labels=labels,
            manifest_hash=expected_hash,
            frozen_before_model_output=frozen,
        )

    @classmethod
    def load(cls, path: str | Path) -> "EvaluationManifest":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def as_dict(self) -> dict[str, Any]:
        return {
            "manifest_id": self.manifest_id,
            "manifest_version": self.manifest_version,
            "dataset_role": self.dataset_role,
            "created_at": self.created_at,
            "reviewer_user_id": self.reviewer_user_id,
            "review_completed_at": self.review_completed_at,
            "source_document_refs": list(self.source_document_refs),
            "source_document_hashes": list(self.source_document_hashes),
            "labels": list(self.labels),
            "frozen_before_model_output": self.frozen_before_model_output,
            "manifest_hash": self.manifest_hash,
        }

    def freeze(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            existing = self.load(destination)
            if existing.manifest_hash != self.manifest_hash:
                raise ValueError("frozen manifest cannot be overwritten with different content")
            return
        payload = _canonical(self.as_dict()) + b"\n"
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.replace(temporary, destination)
            try:
                directory_fd = os.open(destination.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except OSError:
                pass
        finally:
            temporary.unlink(missing_ok=True)


def exact_anchor_key(anchor: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(anchor.get("sourceRef") or ""),
        str(anchor.get("sourceHash") or ""),
        str(anchor.get("locator") or ""),
    )


@dataclass(frozen=True)
class FindingMetrics:
    true_positive: int
    false_positive: int
    false_negative: int
    precision: float
    recall: float
    f1: float
    evidence_anchor_accuracy: float
    graph_link_correctness: float

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class LearningGateResult:
    decision: LearningDecision
    f1_delta: float
    precision_delta: float
    recall_delta: float
    reviewer_minutes_delta_ratio: float | None
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def evaluate_learning_gate(
    v0: FindingMetrics,
    v1: FindingMetrics,
    *,
    reviewer_minutes_v0: float | None = None,
    reviewer_minutes_v1: float | None = None,
) -> LearningGateResult:
    """Apply the Phase1-001 product-learning gate exactly once.

    This is deliberately not a governance threshold.  Missing or invalid
    comparison evidence fails closed as SPLIT_FIX rather than guessing.
    """
    values = (*v0.as_dict().values(), *v1.as_dict().values())
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        return LearningGateResult("SPLIT_FIX", 0.0, 0.0, 0.0, None, "invalid metric evidence")
    f1_delta = v1.f1 - v0.f1
    precision_delta = v1.precision - v0.precision
    recall_delta = v1.recall - v0.recall
    minutes_ratio: float | None = None
    if reviewer_minutes_v0 is not None or reviewer_minutes_v1 is not None:
        if reviewer_minutes_v0 is None or reviewer_minutes_v1 is None or reviewer_minutes_v0 <= 0 or reviewer_minutes_v1 < 0:
            return LearningGateResult("SPLIT_FIX", f1_delta, precision_delta, recall_delta, None, "invalid UX metric evidence")
        minutes_ratio = (reviewer_minutes_v1 - reviewer_minutes_v0) / reviewer_minutes_v0
    quality_gate = f1_delta >= 0.05 and precision_delta >= -0.02 and recall_delta >= -0.02
    ux_gate = minutes_ratio is not None and minutes_ratio <= -0.20 and f1_delta >= -0.02
    if quality_gate:
        decision: LearningDecision = "GO"
        reason = "held-out F1 improvement gate met"
    elif ux_gate:
        decision = "GO"
        reason = "reviewer-time improvement gate met without material F1 regression"
    else:
        decision = "STOP_REVISE"
        reason = "comparison valid but improvement gate not met"
    return LearningGateResult(decision, f1_delta, precision_delta, recall_delta, minutes_ratio, reason)


def evaluate_findings(
    expected: EvaluationManifest,
    predictions: Iterable[Mapping[str, Any]],
    *,
    graph_ref_resolver: Callable[[str], bool] | None = None,
) -> FindingMetrics:
    """Score one run with exact sourceRef/sourceHash/locator/type matching."""

    expected_labels = [label for label in expected.labels if label["expected_outcome"] == "finding"]
    expected_by_key = {
        (str(label["document_ref"]), str(label["finding_type"]), exact_anchor_key(label["expected_span_anchor"])): label
        for label in expected_labels
    }
    used: set[tuple[str, str, tuple[str, str, str]]] = set()
    tp = fp = anchor_ok = graph_declared = graph_ok = 0
    for prediction in predictions:
        document_ref = str(prediction.get("document_ref") or "")
        finding_type = str(prediction.get("finding_type") or "")
        anchor = prediction.get("evidence_anchor") or {}
        key = (document_ref, finding_type, exact_anchor_key(anchor))
        if key in expected_by_key and key not in used:
            used.add(key)
            tp += 1
            if exact_anchor_key(anchor) == exact_anchor_key(expected_by_key[key]["expected_span_anchor"]):
                anchor_ok += 1
        else:
            fp += 1
        refs = tuple(str(value) for value in prediction.get("graph_refs", ()) if value)
        if refs:
            graph_declared += 1
            if graph_ref_resolver is not None and all(graph_ref_resolver(ref) for ref in refs):
                graph_ok += 1
    fn = len(expected_by_key) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return FindingMetrics(
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        evidence_anchor_accuracy=anchor_ok / tp if tp else 0.0,
        graph_link_correctness=graph_ok / graph_declared if graph_declared else 0.0,
    )
