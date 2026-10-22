from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from agents.effect_outbox import (
    AtomicEffectOutbox,
    COMPLETED,
    EFFECT_UNKNOWN,
    READY,
    EffectRecord,
    InvalidEffectTransition,
)


CONFIRMED_COMPLETED = "CONFIRMED_COMPLETED"
CONFIRMED_NOT_APPLIED = "CONFIRMED_NOT_APPLIED"
STILL_UNKNOWN = "STILL_UNKNOWN"


@dataclass(frozen=True)
class ProviderEvidence:
    outcome: str
    correlation_ref: str | None = None
    receipt_ref: str | None = None
    detail: str | None = None


@dataclass(frozen=True)
class ReconciliationResult:
    effect_id: str
    outcome: str
    final_state: str
    correlation_ref: str | None
    receipt_ref: str | None
    detail: str | None = None


class EffectEvidenceProvider(Protocol):
    def inspect(self, effect: EffectRecord) -> ProviderEvidence:
        ...


class EffectReconciliationService:
    """G3 authority for moving EFFECT_UNKNOWN using provider evidence only."""

    def __init__(self, outbox: AtomicEffectOutbox) -> None:
        self.outbox = outbox

    def reconcile(
        self, *, effect_id: str, provider: EffectEvidenceProvider
    ) -> ReconciliationResult:
        record = self.outbox.get(effect_id)
        if record is None:
            raise KeyError(effect_id)

        # Reconciliation is idempotent after a prior successful outcome.
        if record.state == COMPLETED:
            return ReconciliationResult(
                effect_id=record.effect_id,
                outcome=CONFIRMED_COMPLETED,
                final_state=COMPLETED,
                correlation_ref=record.provider_correlation_ref,
                receipt_ref=record.receipt_ref,
                detail="already_reconciled_completed",
            )
        if record.state == READY:
            if record.failure_detail == "g3_reconciled_not_applied":
                return ReconciliationResult(
                    effect_id=record.effect_id,
                    outcome=CONFIRMED_NOT_APPLIED,
                    final_state=READY,
                    correlation_ref=record.provider_correlation_ref,
                    receipt_ref=record.receipt_ref,
                    detail="already_reconciled_not_applied",
                )
            raise InvalidEffectTransition(
                "READY is not evidence of prior not-applied reconciliation"
            )
        if record.state != EFFECT_UNKNOWN:
            raise InvalidEffectTransition(
                "provider reconciliation requires EFFECT_UNKNOWN"
            )

        evidence = provider.inspect(record)
        if evidence.outcome not in {
            CONFIRMED_COMPLETED,
            CONFIRMED_NOT_APPLIED,
            STILL_UNKNOWN,
        }:
            raise ValueError(f"invalid reconciliation outcome: {evidence.outcome}")

        correlation_ref = evidence.correlation_ref or record.provider_correlation_ref
        if evidence.correlation_ref:
            self.outbox.pin_provider_correlation(
                effect_id=effect_id, correlation_ref=evidence.correlation_ref
            )

        if evidence.outcome == CONFIRMED_COMPLETED:
            receipt_ref = evidence.receipt_ref or correlation_ref
            if not receipt_ref:
                raise ValueError(
                    "confirmed completion requires provider receipt/correlation"
                )
            self.outbox.reconcile_unknown_completed(
                effect_id=effect_id, receipt_ref=receipt_ref
            )
            final = self.outbox.get(effect_id)
            assert final is not None
            return ReconciliationResult(
                effect_id=effect_id,
                outcome=CONFIRMED_COMPLETED,
                final_state=final.state,
                correlation_ref=final.provider_correlation_ref,
                receipt_ref=final.receipt_ref,
                detail=evidence.detail,
            )

        if evidence.outcome == CONFIRMED_NOT_APPLIED:
            self.outbox.reconcile_unknown_not_applied(effect_id=effect_id)
            final = self.outbox.get(effect_id)
            assert final is not None
            return ReconciliationResult(
                effect_id=effect_id,
                outcome=CONFIRMED_NOT_APPLIED,
                final_state=final.state,
                correlation_ref=final.provider_correlation_ref,
                receipt_ref=final.receipt_ref,
                detail=evidence.detail,
            )

        final = self.outbox.get(effect_id)
        assert final is not None
        return ReconciliationResult(
            effect_id=effect_id,
            outcome=STILL_UNKNOWN,
            final_state=final.state,
            correlation_ref=final.provider_correlation_ref,
            receipt_ref=final.receipt_ref,
            detail=evidence.detail,
        )


class FeishuMessageEvidenceAdapter:
    """Adapter over the read-only provider lookup seam; owns no Semantier state."""

    def __init__(self, *, get_message) -> None:
        self._get_message = get_message

    def inspect(self, effect: EffectRecord) -> ProviderEvidence:
        correlation = effect.provider_correlation_ref

        if not correlation:
            return ProviderEvidence(
                outcome=STILL_UNKNOWN,
                detail="missing provider correlation; read-only reconciliation unavailable",
            )

        try:
            lookup = self._get_message(message_id=correlation)
        except Exception as exc:
            return ProviderEvidence(
                outcome=STILL_UNKNOWN,
                correlation_ref=correlation,
                detail=f"provider lookup failed: {type(exc).__name__}",
            )
        if lookup.get("found") is True:
            message = lookup.get("message") or {}
            if bool(message.get("deleted")):
                # A deleted message still proves the original send occurred.
                return ProviderEvidence(
                    outcome=CONFIRMED_COMPLETED,
                    correlation_ref=correlation,
                    receipt_ref=correlation,
                    detail="provider message exists but is now deleted",
                )
            return ProviderEvidence(
                outcome=CONFIRMED_COMPLETED,
                correlation_ref=correlation,
                receipt_ref=correlation,
                detail="provider message lookup confirmed send",
            )

        if lookup.get("confirmed_not_applied") is True:
            return ProviderEvidence(
                outcome=CONFIRMED_NOT_APPLIED,
                correlation_ref=correlation,
                detail="provider explicitly confirmed message was not applied",
            )

        return ProviderEvidence(
            outcome=STILL_UNKNOWN,
            correlation_ref=correlation,
            detail="provider lookup insufficient to prove applied or not-applied",
        )
