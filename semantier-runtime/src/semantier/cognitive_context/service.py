from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable

from .models import CognitiveItem, DecisionContext, EvidenceRecord, ItemType, LearningCandidate, Scope, Status
from .store import CognitiveContextStore


@dataclass(frozen=True)
class LearningConfig:
    auto_activate_memory: bool = True
    auto_activate_knowledge: bool = True
    auto_activate_rule: bool = False
    min_knowledge_confidence: float = 0.7
    require_human_approval: bool = False


class CognitiveContextService:
    """Phase-1 memory/knowledge boundary, context builder, and learner."""

    def __init__(self, store: CognitiveContextStore, *, config: LearningConfig | None = None) -> None:
        self.store = store
        self.config = config or LearningConfig()

    def capture_evidence(self, **kwargs: Any) -> EvidenceRecord:
        return self.store.save_evidence(EvidenceRecord.new(**kwargs))

    def index_memory(self, *, evidence: EvidenceRecord, scope: Scope = Scope.WORKSPACE,
                     user_id: str | None = None, content: dict[str, Any] | None = None) -> CognitiveItem:
        """Create observational memory without an LLM or authority metadata."""
        item = CognitiveItem.new(type=ItemType.MEMORY, scope=scope, workspace_id=evidence.workspace_id,
                                 organization_id=evidence.organization_id, user_id=user_id,
                                 content=content or evidence.content, source_refs=[evidence.id], status=Status.ACTIVE)
        return self.store.save_item(item)

    def memory_query(self, *, workspace_id: str, organization_id: str, request: str,
                     user_id: str | None = None, limit: int = 10) -> list[CognitiveItem]:
        items = self.store.list_items(workspace_id=workspace_id, organization_id=organization_id,
                                      type=ItemType.MEMORY, user_id=user_id)
        terms = set(re.findall(r"\w+", request.casefold()))
        def score(item: CognitiveItem) -> tuple[int, str]:
            text = str(item.content).casefold()
            return (sum(1 for term in terms if term in text), item.id)
        matched = [item for item in items if score(item)[0] > 0]
        # An empty request is the deterministic recent-memory window.
        return sorted(matched, key=score, reverse=True)[:limit] if terms else items[-limit:]

    def knowledge_query(self, *, workspace_id: str, organization_id: str,
                        user_id: str | None = None) -> list[CognitiveItem]:
        """Return only active admitted knowledge/rules; memory is structurally excluded."""
        return [item for item_type in (ItemType.KNOWLEDGE, ItemType.RULE)
                for item in self.store.list_items(workspace_id=workspace_id, organization_id=organization_id,
                                                  type=item_type, user_id=user_id, active_only=True)]

    def _active_authority(self, *, workspace_id: str, organization_id: str, user_id: str | None) -> list[CognitiveItem]:
        return [item for item in self.knowledge_query(workspace_id=workspace_id, organization_id=organization_id, user_id=user_id)
                if item.type in (ItemType.KNOWLEDGE, ItemType.RULE)]

    def build_context(self, *, request_ref: str, request: str, workspace_id: str,
                      organization_id: str, user_id: str | None = None, limit: int = 10,
                      resolver: Callable[[list[CognitiveItem]], tuple[list[CognitiveItem], list[dict[str, Any]]]] | None = None) -> DecisionContext:
        memories = self.memory_query(workspace_id=workspace_id, organization_id=organization_id, request=request,
                                     user_id=user_id, limit=limit)
        authorities = self._active_authority(workspace_id=workspace_id, organization_id=organization_id, user_id=user_id)
        conflicts: list[dict[str, Any]] = []
        if resolver:
            authorities, conflicts = resolver(authorities)
        else:
            for left_index, left in enumerate(authorities):
                for right in authorities[left_index + 1:]:
                    if left.type == right.type and left.scope == right.scope and left.content != right.content:
                        conflicts.append({"item_refs": [left.id, right.id], "type": left.type.value,
                                          "scope": left.scope.value, "reason": "competing_active_items"})
        context = DecisionContext.build(request_ref=request_ref, evidence_refs=[ref for item in memories for ref in item.source_refs],
                                        knowledge_refs=[item.id for item in authorities if item.type == ItemType.KNOWLEDGE],
                                        rule_refs=[item.id for item in authorities if item.type == ItemType.RULE],
                                        conflicts=conflicts, workspace_id=workspace_id, organization_id=organization_id)
        return self.store.save_context(context)

    def propose(self, *, proposed_type: ItemType, evidence_refs: list[str], hypothesis: str,
                confidence: float, proposed_change: dict[str, Any], workspace_id: str,
                organization_id: str) -> LearningCandidate:
        return self.store.save_candidate(LearningCandidate.new(proposed_type=proposed_type, evidence_refs=evidence_refs,
                                                               hypothesis=hypothesis, confidence=confidence,
                                                               proposed_change=proposed_change, workspace_id=workspace_id,
                                                               organization_id=organization_id))

    def learn_from_outcome(self, *, event_kind: str, proposed_type: ItemType, evidence_refs: list[str],
                           hypothesis: str, confidence: float, proposed_change: dict[str, Any],
                           workspace_id: str, organization_id: str) -> LearningCandidate | None:
        """Trigger learning only for a meaningful outcome signal."""
        if event_kind not in {"correction", "failure", "explicit_outcome", "completed_task", "strong_feedback", "sampled_session"}:
            return None
        return self.propose(proposed_type=proposed_type, evidence_refs=evidence_refs, hypothesis=hypothesis,
                            confidence=confidence, proposed_change=proposed_change, workspace_id=workspace_id,
                            organization_id=organization_id)

    def close(self, candidate: LearningCandidate, *, item_id: str | None = None) -> CognitiveItem | None:
        """Validate and close a candidate without inventing authority.

        Rules route to pending by default; callers can hand that candidate to
        the existing governed policy review workflow.  No second queue is
        created here.
        """
        proposed = candidate.proposed_change
        scope = Scope(proposed.get("scope", Scope.WORKSPACE.value))
        status = Status.ACTIVE
        if candidate.proposed_type == ItemType.RULE and not self.config.auto_activate_rule:
            status = Status.PENDING
        elif candidate.proposed_type == ItemType.KNOWLEDGE and candidate.confidence < self.config.min_knowledge_confidence:
            status = Status.PENDING
        elif self.config.require_human_approval:
            status = Status.PENDING
        if (not proposed.get("content") or not proposed.get("source_refs")
                or not self.store.evidence_refs_exist(candidate.evidence_refs, workspace_id=candidate.workspace_id,
                                                      organization_id=candidate.organization_id)):
            self.store.update_candidate_status(candidate.id, Status.REJECTED, workspace_id=candidate.workspace_id,
                                               organization_id=candidate.organization_id)
            return None
        active = self.store.list_items(workspace_id=candidate.workspace_id, organization_id=candidate.organization_id,
                                       type=candidate.proposed_type, user_id=proposed.get("user_id"))
        if any(item.scope == scope and item.content != proposed["content"] for item in active):
            self.store.update_candidate_status(candidate.id, Status.REJECTED, workspace_id=candidate.workspace_id,
                                               organization_id=candidate.organization_id)
            return None
        if status == Status.PENDING:
            self.store.update_candidate_status(candidate.id, Status.PENDING, workspace_id=candidate.workspace_id,
                                               organization_id=candidate.organization_id)
            return None
        item = CognitiveItem(item_id or f"item-{candidate.id}", candidate.proposed_type, scope, status,
                             list(proposed["source_refs"]), dict(proposed["content"]), candidate.workspace_id,
                             candidate.organization_id, proposed.get("user_id"), candidate.confidence)
        item = self.store.save_or_get_item(item)
        self.store.update_candidate_status(candidate.id, status, workspace_id=candidate.workspace_id,
                                           organization_id=candidate.organization_id)
        return item
