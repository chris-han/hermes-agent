"""Responsibility-anchored commitment MVL over existing DecisionRecord authority."""

from __future__ import annotations

from datetime import datetime

from contracts.commitment_transition import (
    CommitmentProposalV1,
    CommitmentTransitionReceiptV1,
    CommitmentTransitionV1,
)
from contracts.semantic_kernel import AuthorizationBasis, DecisionRecord, HumanResponsibilityContext
from eos.artifact_hashing import content_hash


def _authority_hash(basis: AuthorizationBasis) -> str:
    return "sha256:" + content_hash(basis.model_dump(mode="json"))


class CommitmentTransitionService:
    """Derived workflow only; owns no semantic store or independent authority."""

    @staticmethod
    def _validate_basis(
        *,
        decision: DecisionRecord,
        responsibility: HumanResponsibilityContext,
        scope_ref: str,
    ) -> AuthorizationBasis:
        if decision.purpose != "CONSEQUENTIAL_DECISION":
            raise ValueError("NO_VALID_COMMIT_AUTHORITY")
        basis = decision.authorization
        if basis is None:
            raise ValueError("NO_VALID_COMMIT_AUTHORITY")
        if not responsibility.responsibility_bearer_id:
            raise ValueError("NO_RESPONSIBILITY_ANCHOR")
        if responsibility.responsibility_bearer_id == responsibility.delegated_actor_id:
            raise ValueError("AI_SELF_ORIGINATED_OR_EXPANDED_AUTHORITY")
        if (
            responsibility.responsibility_bearer_id != decision.actor_id
            or basis.actor_id != decision.actor_id
        ):
            raise ValueError("NO_RESPONSIBILITY_ANCHOR")
        if basis.scope_ref != responsibility.scope_ref or scope_ref != responsibility.scope_ref:
            raise ValueError("SCOPE_EXCEEDS_DELEGATION")
        if (decision.organization_id, decision.workspace_id) != (
            responsibility.organization_id,
            responsibility.workspace_id,
        ):
            raise ValueError("SCOPE_EXCEEDS_DELEGATION")
        if basis.operation != "COMMIT":
            raise ValueError("NO_VALID_COMMIT_AUTHORITY")
        return basis

    def propose(
        self,
        *,
        decision: DecisionRecord,
        responsibility: HumanResponsibilityContext,
        commitment_subject,
        scope_ref: str,
        effective_from: datetime,
        effective_until: datetime | None,
        proposed_at: datetime,
    ) -> CommitmentProposalV1:
        basis = self._validate_basis(
            decision=decision,
            responsibility=responsibility,
            scope_ref=scope_ref,
        )
        return CommitmentProposalV1.create(
            organization_id=decision.organization_id,
            workspace_id=decision.workspace_id,
            decision=decision.pin(),
            commitment_subject=commitment_subject,
            responsibility_anchor=responsibility.pin(),
            authority_basis_hash=_authority_hash(basis),
            scope_ref=scope_ref,
            effective_from=effective_from,
            effective_until=effective_until,
            proposed_at=proposed_at,
        )

    def admit(
        self,
        *,
        proposal: CommitmentProposalV1,
        decision: DecisionRecord,
        responsibility: HumanResponsibilityContext,
        transitioned_at: datetime,
    ) -> tuple[CommitmentTransitionV1, CommitmentTransitionReceiptV1]:
        basis = self._validate_basis(
            decision=decision,
            responsibility=responsibility,
            scope_ref=proposal.scope_ref,
        )
        if proposal.decision != decision.pin():
            raise ValueError("COMMITMENT_DECISION_BINDING_MISMATCH")
        if proposal.responsibility_anchor != responsibility.pin():
            raise ValueError("NO_RESPONSIBILITY_ANCHOR")
        if proposal.authority_basis_hash != _authority_hash(basis):
            raise ValueError("NO_VALID_COMMIT_AUTHORITY")
        transition = CommitmentTransitionV1.create(
            proposal=proposal.pin(),
            predecessor=None,
            state="ADMITTED",
            decision=decision.pin(),
            responsibility_anchor=responsibility.pin(),
            authority_basis_hash=_authority_hash(basis),
            scope_ref=proposal.scope_ref,
            effective_from=proposal.effective_from,
            effective_until=proposal.effective_until,
            replacement_ref=None,
            cause=None,
            transitioned_at=transitioned_at,
        )
        receipt = CommitmentTransitionReceiptV1.create(
            transition=transition.pin(),
            decision=decision.pin(),
            responsibility_anchor=responsibility.pin(),
            authority_basis_hash=_authority_hash(basis),
            scope_ref=proposal.scope_ref,
            recorded_at=transitioned_at,
        )
        return transition, receipt

    def activate(
        self,
        *,
        proposal: CommitmentProposalV1,
        admitted: CommitmentTransitionV1,
        decision: DecisionRecord,
        responsibility: HumanResponsibilityContext,
        transitioned_at: datetime,
    ) -> CommitmentTransitionV1:
        basis = self._validate_basis(
            decision=decision,
            responsibility=responsibility,
            scope_ref=proposal.scope_ref,
        )
        if admitted.state != "ADMITTED":
            raise ValueError("DIRECT_PROPOSED_TO_ACTIVE")
        if (
            admitted.proposal != proposal.pin()
            or admitted.decision != decision.pin()
            or admitted.responsibility_anchor != responsibility.pin()
            or admitted.authority_basis_hash != _authority_hash(basis)
        ):
            raise ValueError("COMMITMENT_ADMISSION_BINDING_MISMATCH")
        return CommitmentTransitionV1.create(
            proposal=proposal.pin(),
            predecessor=admitted.pin(),
            state="ACTIVE",
            decision=decision.pin(),
            responsibility_anchor=responsibility.pin(),
            authority_basis_hash=_authority_hash(basis),
            scope_ref=proposal.scope_ref,
            effective_from=proposal.effective_from,
            effective_until=proposal.effective_until,
            replacement_ref=None,
            cause=None,
            transitioned_at=transitioned_at,
        )
