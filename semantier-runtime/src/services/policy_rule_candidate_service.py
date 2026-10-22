from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from contracts.tender_sensitive_term_detection_modeling_core import RuntimeContextDTO


class PolicyRuleCandidateService(Protocol):
    def create_candidate_idempotent(
        self,
        *,
        idempotency_key: str,
        source_anchor_refs: list[str],
        applicability_scope: dict[str, Any],
        context: RuntimeContextDTO,
    ) -> dict[str, Any]: ...

    def record_reviewer_edit(
        self,
        candidate_id: str,
        edit_payload: dict[str, Any],
        context: RuntimeContextDTO,
    ) -> dict[str, Any]: ...

    def request_review(self, candidate_id: str, context: RuntimeContextDTO) -> dict[str, Any]: ...

    def approve(self, candidate_id: str, approval_payload: dict[str, Any], context: RuntimeContextDTO) -> dict[str, Any]: ...

    def get_candidate_state(self, candidate_id: str, context: RuntimeContextDTO) -> dict[str, Any]: ...


class StoreBackedPolicyRuleCandidateService:
    def __init__(self, db_path: str | Path | None = None) -> None:
        from storage.policy_rule_candidate_store import PolicyRuleCandidateStore

        self._store = PolicyRuleCandidateStore(db_path=db_path)

    def create_candidate_idempotent(
        self,
        *,
        idempotency_key: str,
        source_anchor_refs: list[str],
        applicability_scope: dict[str, Any],
        context: RuntimeContextDTO,
        candidate_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        existing = self._store.get_candidate_by_idempotency_key(idempotency_key)
        if existing is not None:
            return existing
        payload = dict(candidate_payload or {})
        return self._store.create_candidate(
            organization_id=context.organization_id,
            proposed_workspace_scope=payload.pop("proposed_workspace_scope"),
            proposed_workspace_id=payload.pop("proposed_workspace_id"),
            candidate_idempotency_key=idempotency_key,
            source_anchor_refs=source_anchor_refs,
            applicability_scope=applicability_scope,
            created_by=context.actor_id,
            created_by_actor_type="plugin",
            **payload,
        )

    def record_reviewer_edit(
        self,
        candidate_id: str,
        edit_payload: dict[str, Any],
        context: RuntimeContextDTO,
    ) -> dict[str, Any]:
        return self._store.record_reviewer_edit(
            candidate_id,
            actor=context.actor_id,
            actor_authority=str(edit_payload.get("actor_authority") or ""),
            human_edits=list(edit_payload.get("human_edits") or []),
            review_comments=edit_payload.get("review_comments"),
        )

    def request_review(self, candidate_id: str, context: RuntimeContextDTO) -> dict[str, Any]:
        return self._store.request_review(candidate_id, actor=context.actor_id)

    def approve(
        self,
        candidate_id: str,
        approval_payload: dict[str, Any],
        context: RuntimeContextDTO,
    ) -> dict[str, Any]:
        return self._store.approve(
            candidate_id,
            actor=context.actor_id,
            actor_authority=str(approval_payload.get("actor_authority") or ""),
            approval_evidence=dict(approval_payload.get("approval_evidence") or {}),
            test_evidence=dict(approval_payload.get("test_evidence") or {}),
        )

    def get_candidate_state(self, candidate_id: str, context: RuntimeContextDTO) -> dict[str, Any]:
        candidate = self._store.get_candidate(candidate_id)
        if candidate is None:
            raise RuntimeError("RULE_CANDIDATE_NOT_FOUND")
        return candidate

    def list_candidates(
        self,
        *,
        rule_family: str | None = None,
        candidate_state: str | None = None,
    ) -> list[dict[str, Any]]:
        return self._store.list_candidates(
            rule_family=rule_family,
            candidate_state=candidate_state,  # type: ignore[arg-type]
        )
