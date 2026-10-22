"""Application service for source-visible, human-authoritative graph grounding."""

from __future__ import annotations

from typing import Any

from contracts.knowledge_feature_pack import LearningEventEnvelope, LearningEventType
from contracts.knowledge_graph_phase1 import GraphDelta
from storage.human_grounding_store import HumanGroundingStore


class HumanGroundingWorkbench:
    def __init__(self, store: HumanGroundingStore) -> None:
        self._store = store

    def review_view(self, candidate_graph_id: str) -> dict[str, Any]:
        graph = self._store.get_candidate_graph(candidate_graph_id)
        nodes = {node.node_id: node.model_dump(mode="json") for node in graph.nodes}
        return {
            "candidate_graph_id": candidate_graph_id,
            "grounding_state": "proposal_only",
            "machine_proposal": graph.machine_proposal.model_dump(mode="json"),
            "nodes": list(nodes.values()),
            "edges": [
                {
                    **edge.model_dump(mode="json"),
                    "source_node": nodes[edge.source_node_id],
                    "target_node": nodes[edge.target_node_id],
                }
                for edge in graph.edges
            ],
            "rules": [
                {
                    **rule.model_dump(mode="json"),
                    "condition_nodes": [nodes[node_id] for node_id in rule.condition_node_ids],
                    "exception_nodes": [nodes[node_id] for node_id in rule.exception_node_ids],
                }
                for rule in graph.rules
            ],
            "allowed_actions": ["accept", "edit", "reject", "uncertain"],
        }

    def submit_review(
        self,
        *,
        human_event: LearningEventEnvelope,
        delta: GraphDelta | None = None,
    ) -> dict[str, Any]:
        if human_event.event_type is LearningEventType.HUMAN_ACCEPT:
            if delta is not None or human_event.graph_delta_ref is not None:
                raise ValueError("accept without edit cannot persist a GraphDelta")
            return self._store.record_review_decision(human_event=human_event)
        if human_event.event_type is LearningEventType.HUMAN_EDIT:
            if delta is None:
                raise ValueError("edit review requires a GraphDelta")
            return self._store.record_review_decision(human_event=human_event, delta=delta)
        if human_event.event_type in {
            LearningEventType.HUMAN_REJECT,
            LearningEventType.HUMAN_UNCERTAIN,
        }:
            if delta is not None or human_event.graph_delta_ref is not None:
                raise ValueError("reject/uncertain review cannot materialize a GraphDelta")
            return self._store.record_review_decision(human_event=human_event)
        raise ValueError("workbench accepts only human review events")
