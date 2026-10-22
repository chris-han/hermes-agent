"""Project accepted tender feedback into observed-expression candidates."""

from __future__ import annotations

from pathlib import Path

from contracts.contextgraph_reference import ObservedExpressionObservation
from contracts.knowledge_feature_pack import content_hash
from storage.knowledge_extraction_store import KnowledgeExtractionStore
from storage.tender_detection_run_store import TenderDetectionRunStore


class ContextGraphObservedExpressionProjection:
    def __init__(self, *, db_path: str | Path | None = None) -> None:
        self._tender_store = TenderDetectionRunStore(db_path=db_path)
        self._knowledge_store = KnowledgeExtractionStore(db_path or self._tender_store._db_path)

    def project_accepted_finding(
        self,
        *,
        run_id: str,
        finding_id: str,
        human_learning_event_id: str,
        reference_concept_version_ref: str,
        accepted_graph_release_id: str,
        semantic_relation_decision_hash: str,
        semantic_relation: str | None = None,
    ) -> dict:
        run = self._tender_store.get_run(run_id)
        if run is None:
            raise ValueError("tender detection run not found")
        finding = next((item for item in run["findings"] if item["finding_id"] == finding_id), None)
        if finding is None:
            raise ValueError("finding not found")
        accepted = any(
            item["finding_id"] == finding_id and item["disposition"] == "accepted"
            for item in run.get("dispositions", [])
        )
        if not accepted:
            raise ValueError("observed expression projection requires accepted disposition")
        relation = semantic_relation or finding.get("semantic_relation")
        if relation == "exact" or relation is None:
            raise ValueError("observed expression projection requires non-exact semantic relation")
        concept_id = str(finding.get("matched_sensitive_concept_id") or "")
        if not concept_id:
            raise ValueError("finding lacks matched sensitive concept")
        observation_basis = {
            "run_id": run_id,
            "finding_id": finding_id,
            "human_learning_event_id": human_learning_event_id,
        }
        observation = ObservedExpressionObservation(
            schema_version="contextgraph_observed_expression_observation.v1",
            observation_id=f"observed_obs_{content_hash(observation_basis)[7:31]}",
            concept_id=concept_id,
            text=str(finding.get("observed_expression") or finding.get("matched_text")),
            semantic_relation=relation,  # type: ignore[arg-type]
            source_anchor_refs=tuple(finding.get("source_anchor_refs") or []),
            source_document_hash=str(run["source_document_hash"]),
            finding_run_id=run_id,
            finding_id=finding_id,
            reference_concept_version_ref=reference_concept_version_ref,
            accepted_graph_release_id=accepted_graph_release_id,
            human_learning_event_id=human_learning_event_id,
            semantic_relation_decision_hash=semantic_relation_decision_hash,
            grounding_state="candidate",
        )
        aggregate = self._knowledge_store.append_observed_expression_observation(observation)
        return {"observation": observation.model_dump(mode="json"), "observedExpression": aggregate}
