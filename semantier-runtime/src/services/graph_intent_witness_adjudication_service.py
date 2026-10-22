"""Create and verify Human Grounding authority for reconstructed witnesses."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from contracts.graph_intent_witness_adjudication import (
    GraphIntentWitnessAdjudicationV1,
)
from contracts.knowledge_feature_pack import (
    HumanDecision,
    HumanLabel,
    LearningEventEnvelope,
    LearningEventType,
    ProcessLabel,
    content_hash,
)
from contracts.source_semantic_witness import (
    SourceSemanticWitnessSetManifestV1,
    SourceSemanticWitnessV1,
    WitnessReviewRecord,
)
from semantier.knowledge_feature_packs.falkordb_backend import InMemoryGraphBackend
from storage.atomic_json_artifact import publish_immutable_json
from storage.human_grounding_store import HumanGroundingStore
from storage.material_correction_rerun_store import MaterialCorrectionRerunStore


_WITNESS_ADJUDICATION_AUTHORITY = object()


class GraphIntentWitnessAdjudicationService:
    """Turn one authenticated review into replay-verifiable witness authority."""

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        organization_id: str,
        workspace_id: str,
        material_reruns: MaterialCorrectionRerunStore,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.organization_id = organization_id
        self.workspace_id = workspace_id
        self.material_reruns = material_reruns
        self.grounding = HumanGroundingStore(
            self.workspace_root / "eos.db", graph_backend=InMemoryGraphBackend()
        )

    def _adjudicate_accept_all(
        self,
        rerun_ref: str,
        *,
        rerun_hash: str,
        justification: str,
        actor_ref: str,
        _authority: object | None = None,
    ) -> dict:
        if _authority is not _WITNESS_ADJUDICATION_AUTHORITY:
            raise PermissionError("GraphIntent witness adjudication is sealed")
        if not justification.strip():
            raise ValueError("witness adjudication justification is required")
        adjudication_ref = self._ref(rerun_ref, rerun_hash, actor_ref)
        path = self._path(adjudication_ref)
        if path.is_file():
            stored = GraphIntentWitnessAdjudicationV1.model_validate_json(
                path.read_text(encoding="utf-8")
            )
            return self.get_verified(
                stored.adjudication_ref,
                adjudication_hash=stored.adjudication_hash,
                rerun_ref=rerun_ref,
                rerun_hash=rerun_hash,
            ).as_dict()

        pending = self.material_reruns.get_pending_evidence(
            rerun_ref, rerun_hash=rerun_hash
        )
        decided_at = datetime.now(timezone.utc)
        bindings: dict[str, dict[str, str]] = {}
        proposed_bindings: dict[str, dict[str, str]] = {}
        for prefix in ("positive", "hard_negative"):
            proposed = tuple(
                SourceSemanticWitnessV1.model_validate(item)
                for item in pending[f"proposed_{prefix}_witnesses"]
            )
            proposed_manifest = SourceSemanticWitnessSetManifestV1.model_validate(
                pending[f"proposed_{prefix}_manifest"]
            )
            proposed_artifact_ref = self._witness_artifact_ref(proposed_manifest)
            self.grounding.persist_source_semantic_witness_package(
                manifest=proposed_manifest,
                witnesses=proposed,
                artifact_ref=proposed_artifact_ref,
            )
            accepted = tuple(
                witness.create_successor(
                    semantic_expectation=witness.semantic_expectation,
                    review=WitnessReviewRecord.create(
                        state="accepted",
                        actor_ref=actor_ref,
                        review_ref_seed=(
                            f"review:{adjudication_ref}:{prefix}:{index}"
                        ),
                        decision_evidence_refs=witness.evidence_refs,
                        predecessor_witness_ref=witness.witness_id,
                    ),
                )
                for index, witness in enumerate(proposed)
            )
            accepted_manifest = SourceSemanticWitnessSetManifestV1.create(
                witnesses=accepted,
                review_set_ref=f"review-set:{adjudication_ref}:{prefix}",
                predecessor_manifest_ref=proposed_manifest.manifest_ref,
            )
            accepted_artifact_ref = self._witness_artifact_ref(accepted_manifest)
            self.grounding.persist_source_semantic_witness_package(
                manifest=accepted_manifest,
                witnesses=accepted,
                artifact_ref=accepted_artifact_ref,
            )
            proposed_bindings[prefix] = self._binding(
                proposed_manifest, proposed_artifact_ref
            )
            bindings[prefix] = self._binding(
                accepted_manifest, accepted_artifact_ref
            )

        outcome = {
            "schema_version": "graph_intent_witness_adjudication_event.v1",
            "rerun_ref": rerun_ref,
            "rerun_hash": rerun_hash,
            "proposed_positive_manifest": proposed_bindings["positive"],
            "proposed_hard_negative_manifest": proposed_bindings["hard_negative"],
            "accepted_positive_manifest": bindings["positive"],
            "accepted_hard_negative_manifest": bindings["hard_negative"],
        }
        event_seed = content_hash({
            "adjudication_ref": adjudication_ref,
            "actor_ref": actor_ref,
            "outcome": outcome,
        })
        event = LearningEventEnvelope(
            event_id=f"event:graph-intent-witness-adjudication:{event_seed[7:31]}",
            case_id=rerun_ref,
            event_type=LearningEventType.HUMAN_ACCEPT,
            occurred_at=decided_at,
            source_document_hash=(
                SourceSemanticWitnessSetManifestV1.model_validate(
                    pending["proposed_positive_manifest"]
                ).source_binding.source_hash
            ),
            source_anchor_refs=tuple(
                sorted({
                    evidence.evidence_ref
                    for witness in (
                        SourceSemanticWitnessV1.model_validate(item)
                        for item in pending["proposed_positive_witnesses"]
                    )
                    for evidence in witness.evidence_refs
                })
            ),
            graph_version=str(pending["rerun"]["before"]["schema_decision_ref"]),
            rule_version="graph-intent-witness-adjudication.v1",
            evaluation_corpus_version="source-semantic-witness-set.v1",
            actor_ref=actor_ref,
            human_label=HumanLabel(
                decision=HumanDecision.ACCEPT,
                certainty="high",
                reason_code="reconstructed_witnesses_source_verified",
                justification=justification,
                process_labels={"relation": ProcessLabel.CORRECT},
            ),
            downstream_outcome=outcome,
        )
        stored_event = self.grounding.append_event(event)
        adjudication = GraphIntentWitnessAdjudicationV1.create({
            "adjudication_ref": adjudication_ref,
            "organization_id": self.organization_id,
            "workspace_id": self.workspace_id,
            "rerun_ref": rerun_ref,
            "rerun_hash": rerun_hash,
            "proposed_positive_manifest": proposed_bindings["positive"],
            "proposed_hard_negative_manifest": proposed_bindings["hard_negative"],
            "accepted_positive_manifest": bindings["positive"],
            "accepted_hard_negative_manifest": bindings["hard_negative"],
            "grounding_event_ref": event.event_id,
            "grounding_event_hash": stored_event["event_hash"],
            "actor_ref": actor_ref,
            "decided_at": decided_at,
        })
        publish_immutable_json(
            path,
            adjudication.as_dict(),
            conflict_error="GRAPH_INTENT_WITNESS_ADJUDICATION_CONFLICT",
        )
        return adjudication.as_dict()

    def get_verified(
        self,
        adjudication_ref: str,
        *,
        adjudication_hash: str,
        rerun_ref: str,
        rerun_hash: str,
    ) -> GraphIntentWitnessAdjudicationV1:
        path = self._path(adjudication_ref)
        if not path.is_file():
            raise ValueError("witness adjudication not found")
        adjudication = GraphIntentWitnessAdjudicationV1.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        if adjudication.adjudication_hash != adjudication_hash:
            raise ValueError("adjudication replay hash mismatch")
        if (
            adjudication.rerun_ref != rerun_ref
            or adjudication.rerun_hash != rerun_hash
            or adjudication.organization_id != self.organization_id
            or adjudication.workspace_id != self.workspace_id
        ):
            raise ValueError("witness adjudication rerun or tenant binding mismatch")
        pending = self.material_reruns.get_pending_evidence(
            rerun_ref, rerun_hash=rerun_hash
        )
        for prefix in ("positive", "hard_negative"):
            proposed = SourceSemanticWitnessSetManifestV1.model_validate(
                pending[f"proposed_{prefix}_manifest"]
            )
            binding = getattr(adjudication, f"proposed_{prefix}_manifest")
            if (
                binding.manifest_ref != proposed.manifest_ref
                or binding.manifest_hash != proposed.content_hash
                or binding.artifact_ref != self._witness_artifact_ref(proposed)
            ):
                raise ValueError("witness adjudication proposed manifest mismatch")
        stored_event = self.grounding.get_learning_event(
            adjudication.grounding_event_ref
        )
        if stored_event is None:
            raise ValueError("committed Human Grounding event not found")
        if stored_event.pop("event_hash") != adjudication.grounding_event_hash:
            raise ValueError("committed Human Grounding event hash mismatch")
        event = LearningEventEnvelope.model_validate(stored_event)
        if (
            event.event_type is not LearningEventType.HUMAN_ACCEPT
            or event.actor_ref != adjudication.actor_ref
            or event.case_id != rerun_ref
            or event.downstream_outcome is None
            or event.downstream_outcome.get("rerun_hash") != rerun_hash
        ):
            raise ValueError("Human Grounding event does not authorize adjudication")
        for name in (
            "proposed_positive_manifest",
            "proposed_hard_negative_manifest",
            "accepted_positive_manifest",
            "accepted_hard_negative_manifest",
        ):
            if event.downstream_outcome.get(name) != getattr(
                adjudication, name
            ).model_dump(mode="json"):
                raise ValueError("Human Grounding event manifest binding mismatch")
        return adjudication

    def replay_accepted_package(self, binding) -> tuple:
        manifest = self.grounding.replay_source_semantic_witness_manifest(
            organization_id=self.organization_id,
            workspace_id=self.workspace_id,
            manifest_ref=binding.manifest_ref,
            content_hash=binding.manifest_hash,
            artifact_ref=binding.artifact_ref,
        )
        witnesses = tuple(
            self.grounding.replay_source_semantic_witness(
                organization_id=self.organization_id,
                workspace_id=self.workspace_id,
                witness_ref=entry.witness_ref,
                witness_hash=entry.witness_hash,
            )
            for entry in manifest.witnesses
        )
        return manifest, witnesses

    def _ref(self, rerun_ref: str, rerun_hash: str, actor_ref: str) -> str:
        digest = content_hash({
            "rerun_ref": rerun_ref,
            "rerun_hash": rerun_hash,
            "actor_ref": actor_ref,
        })
        return f"graph-intent-witness-adjudication:{digest[7:31]}"

    def _path(self, adjudication_ref: str) -> Path:
        if not adjudication_ref.startswith("graph-intent-witness-adjudication:"):
            raise ValueError("invalid witness adjudication ref")
        return (
            self.material_reruns.workspace_path
            / "artifacts"
            / "graph-intent-witness-adjudications"
            / f"{adjudication_ref.split(':', 1)[1]}.json"
        )

    @staticmethod
    def _witness_artifact_ref(manifest) -> str:
        return (
            f"artifacts/source-semantic-witness-set/{manifest.content_hash[7:]}/"
            "source_semantic_witness_set_manifest.v1.json"
        )

    @staticmethod
    def _binding(manifest, artifact_ref: str) -> dict[str, str]:
        return {
            "manifest_ref": manifest.manifest_ref,
            "manifest_hash": manifest.content_hash,
            "artifact_ref": artifact_ref,
        }


__all__ = ["GraphIntentWitnessAdjudicationService"]
