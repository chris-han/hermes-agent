from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .eval import (
    EffectiveContextEvalCase,
    EffectiveContextEvalMode,
    EffectiveContextEvalResult,
    EvaluationFailureLabel,
)
from .models import ActiveSemanticContract, sha256_hex
from .service import EffectiveContextCheckpointService


@dataclass(frozen=True)
class HarborRoundOutput:
    round_id: str
    message_id: str
    response: str
    knowledge_state: dict[str, Any]
    active_contract: ActiveSemanticContract
    authority_pins: dict[str, Any]
    transcript_snapshot: dict[str, Any]
    delta: dict[str, Any] | None = None
    passed: bool = True
    failure_labels: tuple[EvaluationFailureLabel, ...] = ()


@dataclass(frozen=True)
class HarborCheckpointBoundary:
    round_id: str
    message_id: str
    checkpoint_id: str
    boundary_event_sequence: int


def compile_harbor_eval_case(
    *,
    case_id: str,
    benchmark_ref: str,
    organization_id: str,
    workspace_id: str,
    session_id: str,
    mode: EffectiveContextEvalMode,
    released_task: dict[str, Any],
    reference_checkpoint_id: str | None = None,
) -> EffectiveContextEvalCase:
    """Compile a released task envelope without rewriting its semantics."""
    semantics_hash = hashlib.sha256(
        json.dumps(released_task, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    rounds = _released_rounds(released_task)
    round_boundaries = [
        str(item.get("round_id", item.get("round")))
        for item in rounds
        if item.get("round_id", item.get("round")) is not None
    ]
    labels = sorted({
        str(label)
        for item in rounds or []
        for label in (
            item.get("requirement_labels")
            or item.get("change_types")
            or []
        )
    })
    return EffectiveContextEvalCase(
        case_id=case_id,
        benchmark_ref=benchmark_ref,
        mode=mode,
        organization_id=organization_id,
        workspace_id=workspace_id,
        session_id=session_id,
        reference_checkpoint_id=reference_checkpoint_id,
        round_boundaries=round_boundaries,
        active_requirement_labels=labels,
        released_task_semantics_hash=semantics_hash,
        fixtures={"released_task": released_task},
    )


def _released_rounds(released_task: dict[str, Any]) -> list[dict[str, Any]]:
    """Read both released EvoCodeBench task envelope generations."""
    if not isinstance(released_task, dict):
        return []
    rounds = released_task.get("rounds")
    if isinstance(rounds, list):
        return [item for item in rounds if isinstance(item, dict)]
    legacy = released_task.get("metadata", {}).get("multiround", {})
    rounds = legacy.get("rounds") if isinstance(legacy, dict) else None
    return [item for item in rounds or [] if isinstance(item, dict)]


class HarborPersistentSessionAdapter:
    """Run Harbor-shaped rounds against one persistent ECC session.

    The runner is injected so the adapter does not import Harbor or mutate a
    released benchmark checkout. A real Harbor integration can provide the
    runner while retaining identical round and checkpoint accounting.
    """

    def __init__(
        self,
        service: EffectiveContextCheckpointService,
        *,
        organization_id: str,
        workspace_id: str,
        session_id: str,
        actor_ref: str,
        workspace_root: str | Path,
    ) -> None:
        self.service = service
        self.organization_id = organization_id
        self.workspace_id = workspace_id
        self.session_id = session_id
        self.actor_ref = actor_ref
        self.workspace_root = Path(workspace_root)
        self.boundaries: list[HarborCheckpointBoundary] = []

    def run(
        self,
        case: EffectiveContextEvalCase,
        rounds: Iterable[Any],
        runner: Callable[[Any, Path, str | None], HarborRoundOutput],
    ) -> EffectiveContextEvalResult:
        parent_checkpoint_id: str | None = None
        completed = 0
        first_failed_round: int | None = None
        failure_labels: list[EvaluationFailureLabel] = []
        try:
            for index, round_spec in enumerate(rounds, start=1):
                output = runner(round_spec, self.workspace_root, parent_checkpoint_id)
                event = self.service.record_message_committed(
                    organization_id=self.organization_id,
                    workspace_id=self.workspace_id,
                    session_id=self.session_id,
                    message_id=output.message_id,
                    message_content_hash=sha256_hex({"role": "assistant", "content": output.response}),
                    actor_ref=self.actor_ref,
                    authority_context_ref=output.authority_pins["historical_admission_proof_ref"],
                )
                checkpoint = self.service.materialize_checkpoint(
                    organization_id=self.organization_id,
                    workspace_id=self.workspace_id,
                    session_id=self.session_id,
                    boundary_message_id=output.message_id,
                    actor_ref=self.actor_ref,
                    knowledge_state=output.knowledge_state,
                    active_contract=output.active_contract,
                    authority_pins=output.authority_pins,
                    transcript_snapshot=output.transcript_snapshot,
                    parent_checkpoint_id=parent_checkpoint_id,
                    delta=output.delta,
                )
                self.boundaries.append(HarborCheckpointBoundary(
                    round_id=output.round_id,
                    message_id=output.message_id,
                    checkpoint_id=checkpoint.checkpoint_id,
                    boundary_event_sequence=checkpoint.boundary_event_sequence,
                ))
                parent_checkpoint_id = checkpoint.checkpoint_id
                if not output.passed:
                    first_failed_round = index
                    failure_labels.extend(output.failure_labels or (EvaluationFailureLabel.UNKNOWN_REVIEW_REQUIRED,))
                    break
                completed += 1
        except Exception:
            first_failed_round = first_failed_round or completed + 1
            failure_labels.append(EvaluationFailureLabel.ENVIRONMENT_OR_TOOLING_FAILURE)

        total = len(case.round_boundaries) or completed
        result = EffectiveContextEvalResult(
            result_id=f"{case.case_id}:result",
            case_id=case.case_id,
            mode=case.mode,
            checkpoint_id=self.boundaries[-1].checkpoint_id if self.boundaries else None,
            trajectory_ref=f"trajectory:{self.session_id}",
            checkpoint_lineage_ref=f"lineage:{self.session_id}",
            first_failed_round=first_failed_round,
            failure_labels=failure_labels,
            metrics={},
            fail_stop=first_failed_round is not None,
            harbor_round_accounting={
                "rounds_total": total,
                "rounds_completed": completed,
                "rounds_failed": 1 if first_failed_round is not None else 0,
            },
        )
        self.service.store.put_eval_result(
            self.organization_id,
            self.workspace_id,
            self.session_id,
            result.model_dump(mode="json"),
        )
        return result
        self.service.store.put_eval_result(
            self.organization_id,
            self.workspace_id,
            self.session_id,
            result.model_dump(mode="json"),
        )
        return result

    def restore_reference(self, checkpoint_id: str) -> dict[str, Any]:
        """Fast-forward a reference checkpoint as a read-only evaluation input."""
        return self.service.restore_read_only(self.organization_id, self.workspace_id, checkpoint_id)

    def export_lineage(self) -> list[dict[str, Any]]:
        return [
            {
                "round_id": boundary.round_id,
                "message_id": boundary.message_id,
                "checkpoint_id": boundary.checkpoint_id,
                "boundary_event_sequence": boundary.boundary_event_sequence,
            }
            for boundary in self.boundaries
        ]
