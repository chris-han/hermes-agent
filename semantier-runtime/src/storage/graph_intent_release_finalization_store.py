"""Atomic append-only storage for GraphIntent release finalization evidence."""

from __future__ import annotations

from pathlib import Path
import re

from contracts.graph_intent_release_finalization import (
    GraphIntentReleaseFinalizationV1,
    GraphIntentReleaseReservationV1,
)
from storage.atomic_json_artifact import publish_immutable_json


_FINALIZATION_PUBLICATION_AUTHORITY = object()
_RERUN_REF = re.compile(r"^material-correction-rerun:[0-9a-f]{24}$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


def _safe_root(workspace_root: str | Path, workspace_id: str) -> Path:
    if (not isinstance(workspace_id, str) or not workspace_id or workspace_id in {".", ".."}
            or "/" in workspace_id or "\\" in workspace_id):
        raise ValueError("workspace_id must be one safe path segment")
    workspaces = Path(workspace_root).resolve() / "workspaces"
    root = (workspaces / workspace_id / "artifacts").resolve()
    if not root.is_relative_to(workspaces.resolve()):
        raise ValueError("workspace_id escapes governed workspace root")
    return root


def _require_rerun_ref(value: str) -> str:
    if _RERUN_REF.fullmatch(value) is None:
        raise ValueError("rerun_ref must be an exact material correction rerun ref")
    return value


class GraphIntentReleaseFinalizationStore:
    def __init__(self, workspace_root: str | Path, *, workspace_id: str) -> None:
        self.workspace_id = workspace_id
        self.root = (_safe_root(workspace_root, workspace_id) / "schema-discovery"
                     / "graph-intent-release-finalizations")

    def _path(self, rerun_ref: str, suffix: str) -> Path:
        return self.root / f"{_require_rerun_ref(rerun_ref)}.{suffix}.json"

    def reserve(self, reservation: GraphIntentReleaseReservationV1, *,
                _authority: object) -> GraphIntentReleaseReservationV1:
        if _authority is not _FINALIZATION_PUBLICATION_AUTHORITY:
            raise PermissionError("GraphIntent release reservation publication is sealed")
        if reservation.workspace_id != self.workspace_id:
            raise ValueError("GraphIntent release reservation workspace mismatch")
        path = self._path(reservation.rerun_ref, "reservation")
        if path.exists():
            existing = GraphIntentReleaseReservationV1.model_validate_json(path.read_text(encoding="utf-8"))
            if existing != reservation:
                raise ValueError("GraphIntent rerun already has different release reservation")
            return existing
        return GraphIntentReleaseReservationV1.model_validate(publish_immutable_json(
            path, reservation.model_dump(mode="json"),
            conflict_error="GRAPH_INTENT_RELEASE_RESERVATION_CONFLICT"))

    def get_reservation(self, rerun_ref: str) -> GraphIntentReleaseReservationV1:
        path = self._path(rerun_ref, "reservation")
        if not path.is_file():
            raise ValueError("GraphIntent release reservation is missing")
        reservation = GraphIntentReleaseReservationV1.model_validate_json(path.read_text(encoding="utf-8"))
        if reservation.workspace_id != self.workspace_id or reservation.rerun_ref != rerun_ref:
            raise ValueError("GraphIntent release reservation path binding mismatch")
        return reservation

    def complete(self, rerun_ref: str, finalization: GraphIntentReleaseFinalizationV1,
                 *, _authority: object) -> GraphIntentReleaseFinalizationV1:
        if _authority is not _FINALIZATION_PUBLICATION_AUTHORITY:
            raise PermissionError("GraphIntent release finalization publication is sealed")
        reservation = self.get_reservation(rerun_ref)
        if (finalization.workspace_id != self.workspace_id
                or finalization.organization_id != reservation.organization_id
                or finalization.rerun_ref != reservation.rerun_ref
                or finalization.rerun_hash != reservation.rerun_hash
                or finalization.reservation_ref != reservation.reservation_ref
                or finalization.reservation_hash != reservation.reservation_hash):
            raise ValueError("GraphIntent release finalization reservation mismatch")
        return GraphIntentReleaseFinalizationV1.model_validate(publish_immutable_json(
            self._path(rerun_ref, "finalization"), finalization.model_dump(mode="json"),
            conflict_error="GRAPH_INTENT_RELEASE_FINALIZATION_CONFLICT"))

    def get_verified(self, rerun_ref: str, *, rerun_hash: str, material_reruns,
                     materializer, attribution_service, adjudications,
                     _authority: object) -> GraphIntentReleaseFinalizationV1:
        if _authority is not _FINALIZATION_PUBLICATION_AUTHORITY:
            raise PermissionError("GraphIntent release finalization replay is capability-sealed")
        if _SHA256.fullmatch(rerun_hash) is None:
            raise ValueError("rerun_hash must be an exact sha256 ref")
        reservation = self.get_reservation(rerun_ref)
        if reservation.rerun_hash != rerun_hash:
            raise ValueError("GraphIntent release reservation rerun hash mismatch")
        pending = material_reruns.get(rerun_ref, rerun_hash=rerun_hash)
        if pending.get("outcome") != "pending_adjudication":
            raise ValueError("finalization reservation requires the persisted pending rerun")
        adjudications.get_verified(
            reservation.adjudication_ref,
            adjudication_hash=reservation.adjudication_hash,
            rerun_ref=rerun_ref,
            rerun_hash=rerun_hash,
        )
        path = self._path(rerun_ref, "finalization")
        if not path.is_file():
            raise ValueError("GraphIntent release finalization is missing")
        finalization = GraphIntentReleaseFinalizationV1.model_validate_json(path.read_text(encoding="utf-8"))
        if (finalization.workspace_id != self.workspace_id
                or finalization.organization_id != reservation.organization_id
                or finalization.rerun_ref != rerun_ref or finalization.rerun_hash != rerun_hash
                or finalization.reservation_ref != reservation.reservation_ref
                or finalization.reservation_hash != reservation.reservation_hash):
            raise ValueError("GraphIntent release finalization path binding mismatch")
        accepted = material_reruns.get(finalization.accepted_rerun_ref,
                                        rerun_hash=finalization.accepted_rerun_hash)
        if (accepted.get("comparison_ref") != finalization.schema_feedback_retest_ref
                or accepted.get("comparison_hash") != finalization.schema_feedback_retest_hash
                or accepted.get("feedback_ref") != pending.get("feedback_ref")
                or accepted.get("feedback_hash") != pending.get("feedback_hash")
                or accepted.get("before") != pending.get("before")):
            raise ValueError("finalization schema retest binding mismatch")
        after = dict(accepted.get("after") or {})
        if any(after.get(field) != expected for field, expected in {
            "graph_intent_ref": reservation.graph_intent_ref,
            "graph_intent_hash": reservation.graph_intent_hash,
            "candidate_ref": reservation.schema_candidate_ref,
            "candidate_hash": reservation.schema_candidate_hash,
            "schema_decision_ref": reservation.schema_decision_ref,
            "schema_decision_hash": reservation.schema_decision_hash,
        }.items()):
            raise ValueError("finalization accepted rerun successor binding mismatch")
        release_bundle = materializer.resolve_accepted_graph_release_v2(
            finalization.successor_release_ref, release_hash=finalization.successor_release_hash)
        release = release_bundle["release"]
        lineage = release_bundle["lineage"]
        if (release.get("parent_graph_version") != reservation.predecessor_release_ref
                or release.get("candidate_graph_id") != reservation.candidate_graph_id
                or tuple(release.get("human_label_event_ids") or ())
                != reservation.human_label_event_ids
                or any(lineage.get(field) != expected for field, expected in {
                    "graph_intent_ref": reservation.graph_intent_ref,
                    "graph_intent_hash": reservation.graph_intent_hash,
                    "schema_candidate_ref": reservation.schema_candidate_ref,
                    "schema_candidate_hash": reservation.schema_candidate_hash,
                    "schema_decision_ref": reservation.schema_decision_ref,
                    "schema_decision_hash": reservation.schema_decision_hash,
                    "schema_validation_extract_run_ref": reservation.extract_result_ref,
                    "schema_validation_extract_result_hash": reservation.extract_result_hash,
                    "candidate_graph_materialization_extract_run_ref": reservation.extract_result_ref,
                }.items())):
            raise ValueError("finalization child release lineage mismatch")
        predecessor_bundle = materializer.resolve_accepted_graph_release_v2(
            reservation.predecessor_release_ref,
            release_hash=reservation.predecessor_release_hash,
        )
        from contracts.semantic_release_retest import SemanticReleaseRetestV1
        semantic_path = (self.root.parent / "release-retests"
                         / f"{finalization.semantic_release_retest_ref.replace(':', '-')}.json")
        semantic = SemanticReleaseRetestV1.model_validate_json(semantic_path.read_text(encoding="utf-8"))
        if (semantic.evaluation_id != finalization.semantic_release_retest_ref
                or semantic.evaluation_hash != finalization.semantic_release_retest_hash
                or semantic.baseline_release_ref != reservation.predecessor_release_ref
                or semantic.baseline_release_hash != reservation.predecessor_release_hash
                or semantic.schema_feedback_retest_ref != finalization.schema_feedback_retest_ref
                or semantic.schema_feedback_retest_hash != finalization.schema_feedback_retest_hash
                or semantic.successor_release_ref != finalization.successor_release_ref
                or semantic.successor_release_hash != finalization.successor_release_hash
                or semantic.pattern_membership.model_dump(mode="json") != {
                    "before": predecessor_bundle["release"]["pattern_membership"],
                    "after": release["pattern_membership"],
                }
                or semantic.relation_membership.model_dump(mode="json") != {
                    "before": predecessor_bundle["release"]["relation_membership"],
                    "after": release["relation_membership"],
                }):
            raise ValueError("finalization semantic release retest binding mismatch")
        attribution = attribution_service.get(
            finalization.attribution_ref, attribution_hash=finalization.attribution_hash)
        expected_predecessors = (
            str(pending["before"]["graph_intent_ref"]),
            str(pending["before"]["candidate_ref"]),
            str(pending["before"]["schema_decision_ref"]),
            *(item["pattern_ref"] for item in predecessor_bundle["release"]["pattern_membership"]),
            reservation.predecessor_release_ref,
        )
        expected_successors = (
            reservation.graph_intent_ref,
            reservation.schema_candidate_ref,
            reservation.schema_decision_ref,
            *(item["pattern_ref"] for item in release["pattern_membership"]),
            finalization.successor_release_ref,
            finalization.semantic_release_retest_ref,
        )
        if (attribution.feedback_ref != accepted["feedback_ref"]
                or attribution.feedback_hash != accepted["feedback_hash"]
                or attribution.semantic_impact_layer != "graph_intent"
                or attribution.actor_ref != accepted["created_by"]
                or attribution.target_artifact_ref != reservation.graph_intent_ref
                or attribution.target_artifact_hash != reservation.graph_intent_hash
                or attribution.predecessor_refs != expected_predecessors
                or attribution.successor_refs != expected_successors):
            raise ValueError("finalization attribution lineage mismatch")
        return finalization


__all__ = ["GraphIntentReleaseFinalizationStore", "_FINALIZATION_PUBLICATION_AUTHORITY"]
